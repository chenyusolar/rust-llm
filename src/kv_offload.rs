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
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            if self.gpu_cache.is_some() {
                // GPU KV cache update would go here
            }
        }
        
        self.cpu_cache.update_k(layer_idx, seq_id, k, start_pos);
    }

    pub async fn update_v(&mut self, layer_idx: usize, seq_id: SeqId, v: &[f32], start_pos: usize) {
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            if self.gpu_cache.is_some() {
                // GPU KV cache update would go here
            }
        }
        
        self.cpu_cache.update_v(layer_idx, seq_id, v, start_pos);
    }

    pub fn get_k(&self, layer_idx: usize, seq_id: SeqId, start: usize, len: usize) -> Vec<f32> {
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            if self.gpu_cache.is_some() {
                // Would get from GPU
            }
        }
        
        self.cpu_cache.get_k(layer_idx, seq_id, start, len)
    }

    pub fn get_v(&self, layer_idx: usize, seq_id: SeqId, start: usize, len: usize) -> Vec<f32> {
        if layer_idx < self.current_layer_gpu.len() && self.current_layer_gpu[layer_idx] {
            if self.gpu_cache.is_some() {
                // Would get from GPU
            }
        }
        
        self.cpu_cache.get_v(layer_idx, seq_id, start, len)
    }

    pub async fn swap_layer_to_cpu(&mut self, layer_idx: usize) {
        if layer_idx < self.current_layer_gpu.len() {
            self.current_layer_gpu[layer_idx] = false;
            tracing::debug!("Layer {} swapped to CPU", layer_idx);
        }
    }

    pub async fn swap_layer_to_gpu(&mut self, layer_idx: usize) {
        if layer_idx < self.current_layer_gpu.len() && self.gpu_cache.is_some() {
            self.current_layer_gpu[layer_idx] = true;
            tracing::debug!("Layer {} swapped to GPU", layer_idx);
        }
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
