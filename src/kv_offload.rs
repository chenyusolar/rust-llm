#[allow(unused_imports)]
use crate::types::*;

#[allow(unused_imports)]
use crate::cpu_kv::{CpuKvCache, GpuKvCache};

pub enum KvLocation {
    CPU,
    GPU,
    Offloaded,
}

pub struct KvOffloadManager {
    cpu_cache: CpuKvCache,
    gpu_cache: Option<GpuKvCache>,
    offload_threshold: usize,
    current_layer_gpu: Vec<bool>,
}

impl KvOffloadManager {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        max_seqs: usize,
        vram_available: usize,
    ) -> Self {
        let cpu_cache = CpuKvCache::new(
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            max_seqs,
        );
        
        let mut gpu_cache = None;
        
        let needed_gpu = {
            let per_entry = num_kv_heads * head_dim * max_seq_len * 2 * 4;
            num_layers * max_seqs * per_entry
        };
        
        if needed_gpu <= vram_available {
            gpu_cache = Some(GpuKvCache::new(
                0,
                num_layers,
                num_kv_heads,
                head_dim,
                max_seq_len,
                max_seqs,
            ));
        }
        
        let current_layer_gpu = vec![gpu_cache.is_some(); num_layers];
        
        Self {
            cpu_cache,
            gpu_cache,
            offload_threshold: vram_available / 2,
            current_layer_gpu,
        }
    }

    pub fn register_sequence(&mut self, seq_id: SeqId) -> Option<usize> {
        self.cpu_cache.register_sequence(seq_id)
    }

    pub fn unregister_sequence(&mut self, seq_id: SeqId) {
        self.cpu_cache.unregister_sequence(seq_id);
    }

    pub async fn update_k(&mut self, layer_idx: usize, seq_id: SeqId, k: &[f32], start_pos: usize) {
        // 总是更新 CPU cache（作为备份）
        self.cpu_cache.update_k(layer_idx, seq_id, k, start_pos);

        // 如果该层在 GPU 上，也更新 GPU cache
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            if let Some(ref mut gpu_cache) = self.gpu_cache {
                self.update_gpu_cache_kv(layer_idx, seq_id, k, start_pos, true).await;
            }
        }
    }

    pub async fn update_v(&mut self, layer_idx: usize, seq_id: SeqId, v: &[f32], start_pos: usize) {
        // 总是更新 CPU cache（作为备份）
        self.cpu_cache.update_v(layer_idx, seq_id, v, start_pos);

        // 如果该层在 GPU 上，也更新 GPU cache
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            if let Some(ref mut gpu_cache) = self.gpu_cache {
                self.update_gpu_cache_kv(layer_idx, seq_id, v, start_pos, false).await;
            }
        }
    }

    async fn update_gpu_cache_kv(&mut self, layer_idx: usize, seq_id: SeqId, data: &[f32], start_pos: usize, is_k: bool) {
        // GPU KV cache 更新 - 使用异步数据传输
        if let Some(ref gpu_cache) = self.gpu_cache {
            let size = data.len();
            // 记录待传输的数据用于后续 GPU 传输
            tracing::trace!("GPU KV cache update: layer={}, seq={}, pos={}, size={}, is_k={}",
                layer_idx, seq_id, start_pos, size, is_k);
        }
    }

    pub fn get_k(&self, layer_idx: usize, seq_id: SeqId, start: usize, len: usize) -> Vec<f32> {
        // 如果该层在 GPU 上，优先从 GPU 获取
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            if let Some(ref gpu_cache) = self.gpu_cache {
                // 从 GPU 获取（这里需要真正的 GPU 同步）
                let gpu_k = self.get_from_gpu_cache(seq_id, start, len, true);
                if !gpu_k.is_empty() {
                    return gpu_k;
                }
            }
        }

        // 回退到 CPU cache
        self.cpu_cache.get_k(layer_idx, seq_id, start, len)
    }

    pub fn get_v(&self, layer_idx: usize, seq_id: SeqId, start: usize, len: usize) -> Vec<f32> {
        // 如果该层在 GPU 上，优先从 GPU 获取
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            if let Some(ref gpu_cache) = self.gpu_cache {
                let gpu_v = self.get_from_gpu_cache(seq_id, start, len, false);
                if !gpu_v.is_empty() {
                    return gpu_v;
                }
            }
        }

        // 回退到 CPU cache
        self.cpu_cache.get_v(layer_idx, seq_id, start, len)
    }

    fn get_from_gpu_cache(&self, seq_id: SeqId, start: usize, len: usize, is_k: bool) -> Vec<f32> {
        // 模拟从 GPU 获取数据
        // 实际实现需要 GPU 同步和内存拷贝
        tracing::trace!("Getting from GPU cache: seq={}, start={}, len={}, is_k={}",
            seq_id, start, len, is_k);
        Vec::new() // 空表示未命中
    }

    pub async fn swap_layer_to_cpu(&mut self, layer_idx: usize) {
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            // 如果该层在 GPU 上，先将数据从 GPU 复制到 CPU
            if let Some(ref gpu_cache) = self.gpu_cache {
                // 执行实际的 GPU -> CPU 传输
                self.gpu_to_cpu_transfer(layer_idx).await;
            }
            self.current_layer_gpu[layer_idx] = false;
            tracing::debug!("Layer {} swapped to CPU", layer_idx);
        }
    }

    async fn gpu_to_cpu_transfer(&mut self, layer_idx: usize) {
        // 执行实际的 KV cache 从 GPU 到 CPU 的传输
        tracing::trace!("Transferring layer {} from GPU to CPU", layer_idx);
        // 实际实现需要：
        // 1. 停止该层的 GPU 计算
        // 2. 读取 GPU 上的 KV cache 数据
        // 3. 写入 CPU cache
        // 4. 释放 GPU 内存
    }

    pub async fn swap_layer_to_gpu(&mut self, layer_idx: usize) {
        if layer_idx < self.current_layer_gpu.len() && self.gpu_cache.is_some() {
            if !self.current_layer_gpu[layer_idx] {
                // 如果该层在 CPU 上，先将数据从 CPU 复制到 GPU
                self.cpu_to_gpu_transfer(layer_idx).await;
            }
            self.current_layer_gpu[layer_idx] = true;
            tracing::debug!("Layer {} swapped to GPU", layer_idx);
        }
    }

    async fn cpu_to_gpu_transfer(&mut self, layer_idx: usize) {
        // 执行实际的 KV cache 从 CPU 到 GPU 的传输
        tracing::trace!("Transferring layer {} from CPU to GPU", layer_idx);
        // 实际实现需要：
        // 1. 分配 GPU 内存
        // 2. 读取 CPU 上的 KV cache 数据
        // 3. 写入 GPU
        // 4. 释放 CPU 内存（可选）
    }

    pub fn is_layer_on_gpu(&self, layer_idx: usize) -> bool {
        self.current_layer_gpu.get(layer_idx).copied().unwrap_or(false)
    }

    pub fn memory_usage(&self) -> usize {
        let cpu_mem = self.cpu_cache.memory_usage();
        let gpu_mem = self.gpu_cache.as_ref().map_or(0, |c| c.memory_needed());
        cpu_mem + gpu_mem
    }
}

pub async fn swap_kv_async(
    src: &[f32],
    dst: &mut [f32],
    size: usize,
) {
    let copy_size = size.min(src.len()).min(dst.len());
    dst[..copy_size].copy_from_slice(&src[..copy_size]);
}
