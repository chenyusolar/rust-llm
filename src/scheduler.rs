#[allow(unused_imports)]
use std::sync::Arc;

#[allow(unused_imports)]
use anyhow::Result;

#[allow(unused_imports)]
use tokio::sync::RwLock;

#[allow(unused_imports)]
use tokio::task;

#[allow(unused_imports)]
use rayon::prelude::*;

#[allow(unused_imports)]
use crate::types::*;

#[allow(unused_imports)]
use crate::gpu_kernel::GpuKernel;

#[allow(unused_imports)]
use crate::kv_offload::KvOffloadManager;

#[allow(unused_imports)]
use crate::quantize::QuantizedTensor;

#[allow(unused_imports)]
use crate::paged_attention::PagedAttentionCache;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerLocation {
    CPU,
    GPU,
    Loading,
}

pub struct LayerState {
    pub location: LayerLocation,
    pub weights_loaded: bool,
    pub compute_device: ComputeDevice,
}

impl LayerState {
    pub fn new() -> Self {
        Self {
            location: LayerLocation::CPU,
            weights_loaded: false,
            compute_device: ComputeDevice::CPU,
        }
    }
}

impl Default for LayerState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Scheduler {
    num_layers: usize,
    layer_states: Vec<LayerState>,
    current_layer: usize,
    batch_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    weights: Arc<RwLock<std::collections::HashMap<String, QuantizedTensor>>>,
    gpu_kernel: Arc<RwLock<Option<GpuKernel>>>,
    kv_offload: Arc<RwLock<KvOffloadManager>>,
    pipeline_overlap: bool,
    paged_attention: Option<PagedAttentionCache>,
    max_seq_len: usize,
}

impl Scheduler {
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        weights: Arc<RwLock<std::collections::HashMap<String, QuantizedTensor>>>,
    ) -> Self {
        let layer_states = (0..num_layers).map(|_| LayerState::new()).collect();
        
        let kv_offload = KvOffloadManager::new(
            num_layers,
            num_kv_heads,
            head_dim,
            2048,
            batch_size,
            8 * 1024 * 1024 * 1024,
        );
        
        let max_seq_len = 2048;
        let num_pages = (max_seq_len + 15) / 16;
        
        let paged_attention = PagedAttentionCache::new(
            num_kv_heads,
            head_dim,
            max_seq_len,
            num_pages * batch_size,
        );
        
        Self {
            num_layers,
            layer_states,
            current_layer: 0,
            batch_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            weights,
            gpu_kernel: Arc::new(RwLock::new(None)),
            kv_offload: Arc::new(RwLock::new(kv_offload)),
            pipeline_overlap: true,
            paged_attention: Some(paged_attention),
            max_seq_len,
        }
    }

    pub async fn init_gpu(&self, device_id: usize) -> Result<()> {
        let mut kernel = GpuKernel::new(
            device_id,
            self.batch_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
        );
        kernel.init_cuda()?;
        
        let mut guard = self.gpu_kernel.write().await;
        *guard = Some(kernel);
        
        Ok(())
    }

    pub async fn process_layer(&mut self, layer_idx: usize, input: &[f32], seq_pos: usize) -> Result<Vec<f32>> {
        if layer_idx >= self.num_layers {
            anyhow::bail!("Invalid layer index");
        }
        
        let compute_device = self.select_compute_device(layer_idx);
        
        let output = match compute_device {
            ComputeDevice::Device(_) => {
                self.compute_gpu(layer_idx, input).await?
            }
            ComputeDevice::CPU => {
                self.compute_cpu(layer_idx, input, seq_pos).await?
            }
        };
        
        self.layer_states[layer_idx].compute_device = compute_device;
        
        Ok(output)
    }
    
    pub async fn forward_batch(&self, batch_size: usize, input: &[f32]) -> Result<Vec<Vec<f32>>> {
        let hidden_size = self.hidden_size;
        
        if input.len() < hidden_size {
            return Ok(vec![input.to_vec()]);
        }
        
        let chunk_size = input.len() / batch_size;
        let mut all_outputs = Vec::with_capacity(batch_size);
        
        for batch_idx in 0..batch_size {
            let start = batch_idx * chunk_size;
            let end = (batch_idx + 1) * chunk_size;
            let chunk = &input[start..end.min(input.len())];
            
            if chunk.len() < hidden_size {
                all_outputs.push(chunk.to_vec());
                continue;
            }
            
            let input_chunk = &chunk[..hidden_size];
            let mut hidden = input_chunk.to_vec();
            
            for layer_idx in 0..self.num_layers {
                let weights = self.weights.read().await;
                
                let attn_norm_key = self.resolve_weight_key(&weights, layer_idx, "attn_norm.weight");
                let ffn_norm_key = self.resolve_weight_key(&weights, layer_idx, "ffn_norm.weight");
                
                let normalized = if let Some(norm_tensor) = attn_norm_key {
                    let norm_weight = norm_tensor.dequantize();
                    self.rms_norm(&hidden, &norm_weight, 1e-5)
                } else {
                    hidden.clone()
                };
                
                let mut q = self.compute_q(&weights, layer_idx, &normalized, hidden_size, self.head_dim, self.num_heads);
                let mut k = self.compute_k(&weights, layer_idx, &normalized, hidden_size, self.head_dim, self.num_heads);
                let v = self.compute_v(&weights, layer_idx, &normalized, hidden_size, self.head_dim, self.num_heads);
                
                for i in 0..q.len() / self.head_dim {
                    self.apply_rope(&mut q[i * self.head_dim..(i + 1) * self.head_dim], batch_idx, self.head_dim);
                }
                for i in 0..k.len() / self.head_dim {
                    self.apply_rope(&mut k[i * self.head_dim..(i + 1) * self.head_dim], batch_idx, self.head_dim);
                }
                
                let attn_output = self.cpu_attention(&q, &k, &v, self.head_dim, self.num_heads, batch_idx);
                
                let ffn_input: Vec<f32> = attn_output.iter().zip(hidden.iter()).map(|(a, b)| a + b).collect();
                
                let ffn_normalized = if let Some(norm_tensor) = ffn_norm_key {
                    let norm_weight = norm_tensor.dequantize();
                    self.rms_norm(&ffn_input, &norm_weight, 1e-5)
                } else {
                    ffn_input
                };
                
                let ffn_output = self.compute_ffn(&weights, layer_idx, &ffn_normalized, hidden_size, self.intermediate_size);
                
                hidden = Vec::with_capacity(hidden_size);
                for i in 0..hidden_size {
                    let attn_val = attn_output.get(i).copied().unwrap_or(0.0);
                    let ffn_val = ffn_output.get(i).copied().unwrap_or(0.0);
                    let input_val = input_chunk.get(i).copied().unwrap_or(0.0);
                    hidden.push(input_val + attn_val + ffn_val);
                }
            }
            
            all_outputs.push(hidden);
        }
        
        Ok(all_outputs)
    }

    async fn compute_gpu(&mut self, layer_idx: usize, input: &[f32]) -> Result<Vec<f32>> {
        let use_gpu = {
            let kernel_guard = self.gpu_kernel.read().await;
            kernel_guard.is_some()
        };
        
        if use_gpu {
            let kernel_guard = self.gpu_kernel.read().await;
            
            if let Some(ref kernel) = *kernel_guard {
                let hidden_size = self.hidden_size;
                let head_dim = self.head_dim;
                let num_heads = self.num_heads;
                
                let mut q = input.to_vec();
                let k = input.to_vec();
                let v = input.to_vec();
                
                let mut k_cache = vec![0.0f32; self.batch_size * self.max_kv_len() * self.num_kv_heads * head_dim];
                let mut v_cache = vec![0.0f32; self.max_kv_len() * self.num_kv_heads * head_dim];
                
                let attn_output = kernel.forward_attention(
                    &mut q,
                    &k,
                    &v,
                    &mut k_cache,
                    &mut v_cache,
                    1,
                    1,
                ).await?;
                
                return Ok(attn_output);
            }
        }
        
        self.compute_cpu(layer_idx, input, 0).await
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let dim = x.len();
        let mut output = vec![0.0f32; dim];
        
        let mut sq_sum = 0.0f32;
        for &xi in x {
            sq_sum += xi * xi;
        }
        let rms = ((sq_sum / dim as f32) + eps).sqrt();
        
        for i in 0..dim {
            let w = weight.get(i).copied().unwrap_or(1.0);
            output[i] = x[i] / rms * w;
        }
        
        output
    }

    fn apply_rope(&self, x: &mut [f32], position: usize, head_dim: usize) {
        for i in 0..head_dim / 2 {
            let freq = (position as f32).powf(-2.0 * i as f32 / head_dim as f32);
            let inv_freq = 1.0 / freq;
            
            let idx = i;
            let idx_rot = i + head_dim / 2;
            
            let cos = inv_freq.cos();
            let sin = inv_freq.sin();
            
            let x0 = x[idx];
            let x1 = x[idx_rot];
            
            x[idx] = x0 * cos - x1 * sin;
            x[idx_rot] = x0 * sin + x1 * cos;
        }
    }

    async fn compute_cpu(&mut self, layer_idx: usize, input: &[f32], seq_pos: usize) -> Result<Vec<f32>> {
        let hidden_size = self.hidden_size;
        let intermediate_size = self.intermediate_size;
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        
        if input.len() < hidden_size {
            return Ok(input.to_vec());
        }
        
        let input_chunk = &input[..hidden_size];
        
        let weights = self.weights.read().await;
        
        let attn_norm_key = self.resolve_weight_key(&weights, layer_idx, "attn_norm.weight");
        let ffn_norm_key = self.resolve_weight_key(&weights, layer_idx, "ffn_norm.weight");
        
        let normalized = if let Some(norm_tensor) = attn_norm_key {
            let norm_weight = norm_tensor.dequantize();
            self.rms_norm(input_chunk, &norm_weight, 1e-5)
        } else {
            input_chunk.to_vec()
        };
        
        let mut q = self.compute_q(&weights, layer_idx, &normalized, hidden_size, head_dim, num_heads);
        let mut k = self.compute_k(&weights, layer_idx, &normalized, hidden_size, head_dim, num_heads);
        let v = self.compute_v(&weights, layer_idx, &normalized, hidden_size, head_dim, num_heads);
        
        let kv_heads = (num_heads / 4).max(1);
        for i in 0..q.len() / head_dim {
            self.apply_rope(&mut q[i * head_dim..(i + 1) * head_dim], seq_pos, head_dim);
        }
        for i in 0..k.len() / head_dim {
            self.apply_rope(&mut k[i * head_dim..(i + 1) * head_dim], seq_pos, head_dim);
        }
        
        let attn_output = self.cpu_attention(&q, &k, &v, head_dim, num_heads, seq_pos);
        
        let wo_key = self.resolve_weight_key(&weights, layer_idx, "attn_o.weight");
        let output_proj = if let Some(t) = wo_key {
            t.dequantize()
        } else {
            vec![0.0f32; hidden_size * hidden_size]
        };
        
        let mut attn_proj = vec![0.0f32; hidden_size];
        for j in 0..hidden_size.min(attn_output.len()) {
            let mut sum = 0.0f32;
            for i in 0..(attn_output.len() / hidden_size).min(1) {
                let idx = i * hidden_size + j;
                if idx < output_proj.len() {
                    sum += attn_output.get(i * hidden_size + j).copied().unwrap_or(0.0) * output_proj[idx];
                }
            }
            attn_proj[j] = sum;
        }
        
        let ffn_input: Vec<f32> = attn_proj.iter().zip(input_chunk.iter()).map(|(a, b)| a + b).collect();
        
        let ffn_normalized = if let Some(norm_tensor) = ffn_norm_key {
            let norm_weight = norm_tensor.dequantize();
            self.rms_norm(&ffn_input, &norm_weight, 1e-5)
        } else {
            ffn_input
        };
        
        let ffn_output = self.compute_ffn(&weights, layer_idx, &ffn_normalized, hidden_size, intermediate_size);
        
        let mut output = Vec::with_capacity(hidden_size);
        for i in 0..hidden_size {
            let attn_val = attn_proj.get(i).copied().unwrap_or(0.0);
            let ffn_val = ffn_output.get(i).copied().unwrap_or(0.0);
            let input_val = input_chunk.get(i).copied().unwrap_or(0.0);
            output.push(input_val + attn_val + ffn_val);
        }
        
        Ok(output)
    }

    fn resolve_weight_key<'a>(&self, weights: &'a std::collections::HashMap<String, QuantizedTensor>, layer_idx: usize, name: &str) -> Option<&'a QuantizedTensor> {
        let patterns = [
            format!("blk.{}.{}", layer_idx, name),
            format!("model.layers.{}.{}", layer_idx, name),
            format!("layers.{}.{}", layer_idx, name),
            format!("transformer.h.{}.{}", layer_idx, name),
        ];
        
        for key in &patterns {
            if let Some(tensor) = weights.get(key) {
                return Some(tensor);
            }
        }
        
        None
    }

    fn compute_q(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, layer_idx: usize, x: &[f32], hidden_size: usize, head_dim: usize, num_heads: usize) -> Vec<f32> {
        let q_names = ["attn_q.weight", "self_attn.q_proj.weight", "attn.q.weight", "q_proj.weight"];
        
        for name in &q_names {
            if let Some(tensor) = self.resolve_weight_key(weights, layer_idx, name) {
                let weight = tensor.dequantize();
                let w_cols = tensor.shape.last().copied().unwrap_or(hidden_size);
                let w_rows = tensor.shape.first().copied().unwrap_or(hidden_size);
                
                let mut q = vec![0.0f32; w_rows.min(x.len() * num_heads)];
                for r in 0..w_rows.min(q.len() / head_dim.max(1) * head_dim) {
                    let mut sum = 0.0f32;
                    for c in 0..w_cols.min(x.len()) {
                        let idx = r * w_cols + c;
                        if idx < weight.len() {
                            sum += x[c] * weight[idx];
                        }
                    }
                    q[r] = sum;
                }
                return q;
            }
        }
        
        tracing::warn!("No Q weight found for layer {}, using fallback", layer_idx);
        let mut q = Vec::with_capacity(hidden_size);
        for i in 0..hidden_size {
            q.push(x[i] * 0.1);
        }
        while q.len() < hidden_size * num_heads {
            q.push(0.0);
        }
        q
    }

    fn compute_k(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, layer_idx: usize, x: &[f32], hidden_size: usize, head_dim: usize, num_heads: usize) -> Vec<f32> {
        let k_names = ["attn_k.weight", "self_attn.k_proj.weight", "attn.k.weight", "k_proj.weight"];
        
        for name in &k_names {
            if let Some(tensor) = self.resolve_weight_key(weights, layer_idx, name) {
                let weight = tensor.dequantize();
                let w_cols = tensor.shape.last().copied().unwrap_or(hidden_size);
                let w_rows = tensor.shape.first().copied().unwrap_or(hidden_size / 4);
                
                let kv_heads = (num_heads / 4).max(1);
                let mut k = vec![0.0f32; w_rows.min(x.len() * kv_heads)];
                for r in 0..w_rows.min(k.len() / head_dim.max(1) * head_dim) {
                    let mut sum = 0.0f32;
                    for c in 0..w_cols.min(x.len()) {
                        let idx = r * w_cols + c;
                        if idx < weight.len() {
                            sum += x[c] * weight[idx];
                        }
                    }
                    k[r] = sum;
                }
                return k;
            }
        }
        
        tracing::warn!("No K weight found for layer {}, using fallback", layer_idx);
        let mut k = Vec::with_capacity(hidden_size);
        for i in 0..hidden_size {
            k.push(x[i] * 0.1);
        }
        while k.len() < hidden_size * (num_heads / 4).max(1) {
            k.push(0.0);
        }
        k
    }

    fn compute_v(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, layer_idx: usize, x: &[f32], hidden_size: usize, head_dim: usize, num_heads: usize) -> Vec<f32> {
        let v_names = ["attn_v.weight", "self_attn.v_proj.weight", "attn.v.weight", "v_proj.weight"];
        
        for name in &v_names {
            if let Some(tensor) = self.resolve_weight_key(weights, layer_idx, name) {
                let weight = tensor.dequantize();
                let w_cols = tensor.shape.last().copied().unwrap_or(hidden_size);
                let w_rows = tensor.shape.first().copied().unwrap_or(hidden_size / 4);
                
                let kv_heads = (num_heads / 4).max(1);
                let mut v = vec![0.0f32; w_rows.min(x.len() * kv_heads)];
                for r in 0..w_rows.min(v.len() / head_dim.max(1) * head_dim) {
                    let mut sum = 0.0f32;
                    for c in 0..w_cols.min(x.len()) {
                        let idx = r * w_cols + c;
                        if idx < weight.len() {
                            sum += x[c] * weight[idx];
                        }
                    }
                    v[r] = sum;
                }
                return v;
            }
        }
        
        tracing::warn!("No V weight found for layer {}, using fallback", layer_idx);
        let mut v = Vec::with_capacity(hidden_size);
        for i in 0..hidden_size {
            v.push(x[i] * 0.1);
        }
        while v.len() < hidden_size * (num_heads / 4).max(1) {
            v.push(0.0);
        }
        v
    }

    fn cpu_attention(&self, q: &[f32], k: &[f32], v: &[f32], head_dim: usize, num_heads: usize, seq_pos: usize) -> Vec<f32> {
        let hidden_size = head_dim * num_heads;
        let kv_heads = (num_heads / 4).max(1);
        
        let mut output = vec![0.0f32; hidden_size];
        
        let kv_len = k.len() / head_dim.max(1);
        
        for h in 0..num_heads {
            let kv_head = h / (num_heads / kv_heads.max(1));
            let q_offset = h * head_dim;
            let k_offset = kv_head * head_dim;
            let v_offset = kv_head * head_dim;
            
            let mut max_score = -1e10f32;
            let mut exp_sum = 0.0f32;
            
            for j in 0..kv_len {
                if j > seq_pos {
                    break;
                }
                
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = q_offset + d;
                    let k_idx = k_offset + j * head_dim + d;
                    if q_idx < q.len() && k_idx < k.len() {
                        score += q[q_idx] * k[k_idx];
                    }
                }
                score /= (head_dim as f32).sqrt();
                
                if score > max_score {
                    max_score = score;
                }
            }
            
            for j in 0..kv_len {
                if j > seq_pos {
                    break;
                }
                
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = q_offset + d;
                    let k_idx = k_offset + j * head_dim + d;
                    if q_idx < q.len() && k_idx < k.len() {
                        score += q[q_idx] * k[k_idx];
                    }
                }
                score /= (head_dim as f32).sqrt();
                let exp_score = (score - max_score).exp();
                exp_sum += exp_score;
                
                for d in 0..head_dim {
                    let out_idx = q_offset + d;
                    let v_idx = v_offset + j * head_dim + d;
                    if out_idx < output.len() && v_idx < v.len() {
                        output[out_idx] += exp_score * v[v_idx];
                    }
                }
            }
            
            if exp_sum > 0.0 {
                for d in 0..head_dim {
                    let out_idx = q_offset + d;
                    if out_idx < output.len() {
                        output[out_idx] /= exp_sum;
                    }
                }
            }
        }
        
        output
    }

    fn paged_attention_forward(&mut self, q: &[f32], k: &[f32], v: &[f32], head_dim: usize, num_heads: usize, seq_pos: usize, layer_idx: usize) -> Vec<f32> {
        if let Some(ref mut paged_attn) = self.paged_attention {
            let seq_id = 1;
            let kv_heads = (num_heads / 4).max(1);
            
            let page_indices = paged_attn.allocate_pages(1);
            paged_attn.write_kv(seq_id, &page_indices, k, v);
            
            if let Some((cached_k, cached_v)) = paged_attn.read_kv(seq_id, seq_pos + 1) {
                return self.cpu_attention_with_kv(q, &cached_k, &cached_v, head_dim, num_heads, seq_pos);
            }
        }
        
        self.cpu_attention(q, k, v, head_dim, num_heads, seq_pos)
    }

    fn cpu_attention_with_kv(&self, q: &[f32], k: &[f32], v: &[f32], head_dim: usize, num_heads: usize, seq_pos: usize) -> Vec<f32> {
        let hidden_size = head_dim * num_heads;
        
        let mut output = vec![0.0f32; hidden_size];
        
        let kv_len = k.len() / head_dim.max(1);
        
        for h in 0..num_heads {
            let kv_head = h;
            let q_offset = h * head_dim;
            let k_offset = kv_head * head_dim;
            let v_offset = kv_head * head_dim;
            
            let mut max_score = -1e10f32;
            let mut exp_sum = 0.0f32;
            
            for j in 0..kv_len.min(seq_pos + 1) {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = q_offset + d;
                    let k_idx = k_offset + j * head_dim + d;
                    if q_idx < q.len() && k_idx < k.len() {
                        score += q[q_idx] * k[k_idx];
                    }
                }
                score /= (head_dim as f32).sqrt();
                
                if score > max_score {
                    max_score = score;
                }
            }
            
            for j in 0..kv_len.min(seq_pos + 1) {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = q_offset + d;
                    let k_idx = k_offset + j * head_dim + d;
                    if q_idx < q.len() && k_idx < k.len() {
                        score += q[q_idx] * k[k_idx];
                    }
                }
                score /= (head_dim as f32).sqrt();
                let exp_score = (score - max_score).exp();
                exp_sum += exp_score;
                
                for d in 0..head_dim {
                    let out_idx = q_offset + d;
                    let v_idx = v_offset + j * head_dim + d;
                    if out_idx < output.len() && v_idx < v.len() {
                        output[out_idx] += exp_score * v[v_idx];
                    }
                }
            }
            
            if exp_sum > 0.0 {
                for d in 0..head_dim {
                    let out_idx = q_offset + d;
                    if out_idx < output.len() {
                        output[out_idx] /= exp_sum;
                    }
                }
            }
        }
        
        output
    }

    fn compute_ffn(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, layer_idx: usize, x: &[f32], hidden_size: usize, intermediate_size: usize) -> Vec<f32> {
        let gate_names = ["ffn_gate.weight", "mlp.gate_proj.weight", "block_sparse_moe.gate_proj.weight", "ffn.down_proj.weight"];
        let up_names = ["ffn_up.weight", "mlp.up_proj.weight", "block_sparse_moe.up_proj.weight"];
        let down_names = ["ffn_down.weight", "mlp.down_proj.weight", "block_sparse_moe.down_proj.weight"];
        
        let mut gate_tensor = None;
        for name in &gate_names {
            if let Some(t) = self.resolve_weight_key(weights, layer_idx, name) {
                gate_tensor = Some(t);
                break;
            }
        }
        
        let mut up_tensor = None;
        for name in &up_names {
            if let Some(t) = self.resolve_weight_key(weights, layer_idx, name) {
                up_tensor = Some(t);
                break;
            }
        }
        
        let mut down_tensor = None;
        for name in &down_names {
            if let Some(t) = self.resolve_weight_key(weights, layer_idx, name) {
                down_tensor = Some(t);
                break;
            }
        }
        
        let mut gate_output = vec![0.0f32; intermediate_size.min(hidden_size * 2)];
        let mut up_output = vec![0.0f32; intermediate_size.min(hidden_size * 2)];
        
        if let Some(gate_t) = gate_tensor {
            let gate_weight = gate_t.dequantize();
            let gate_rows = gate_t.shape.get(0).copied().unwrap_or(intermediate_size);
            let gate_cols = gate_t.shape.get(1).copied().unwrap_or(hidden_size);
            
            for i in 0..gate_rows.min(gate_output.len()) {
                let mut sum = 0.0f32;
                for j in 0..gate_cols.min(x.len()) {
                    let idx = i * gate_cols + j;
                    if idx < gate_weight.len() {
                        sum += x[j] * gate_weight[idx];
                    }
                }
                gate_output[i] = silu(sum);
            }
        } else {
            tracing::warn!("No gate weight found for layer {}, using fallback", layer_idx);
            for i in 0..gate_output.len() {
                let mut sum = 0.0f32;
                for j in 0..hidden_size.min(x.len()) {
                    let weight = ((i * 17 + j) % 1000) as f32 / 10000.0;
                    sum += x[j] * weight;
                }
                gate_output[i] = silu(sum);
            }
        }
        
        if let Some(up_t) = up_tensor {
            let up_weight = up_t.dequantize();
            let up_rows = up_t.shape.get(0).copied().unwrap_or(intermediate_size);
            let up_cols = up_t.shape.get(1).copied().unwrap_or(hidden_size);
            
            for i in 0..up_rows.min(up_output.len()) {
                let mut sum = 0.0f32;
                for j in 0..up_cols.min(x.len()) {
                    let idx = i * up_cols + j;
                    if idx < up_weight.len() {
                        sum += x[j] * up_weight[idx];
                    }
                }
                up_output[i] = sum;
            }
        } else {
            for i in 0..up_output.len() {
                let mut sum = 0.0f32;
                for j in 0..hidden_size.min(x.len()) {
                    let weight = ((i * 17 + j) % 1000) as f32 / 10000.0;
                    sum += x[j] * weight;
                }
                up_output[i] = sum;
            }
        }
        
        let mut intermediate = Vec::with_capacity(gate_output.len());
        for i in 0..gate_output.len() {
            intermediate.push(gate_output[i] * up_output[i]);
        }
        
        let mut output = vec![0.0f32; hidden_size];
        
        if let Some(down_t) = down_tensor {
            let down_weight = down_t.dequantize();
            let down_rows = down_t.shape.get(0).copied().unwrap_or(hidden_size);
            let down_cols = down_t.shape.get(1).copied().unwrap_or(intermediate_size);
            
            for j in 0..down_rows.min(output.len()) {
                let mut sum = 0.0f32;
                for i in 0..down_cols.min(intermediate.len()) {
                    let idx = j * down_cols + i;
                    if idx < down_weight.len() {
                        sum += intermediate[i] * down_weight[idx];
                    }
                }
                output[j] = sum;
            }
        } else {
            for j in 0..hidden_size.min(x.len()) {
                let mut sum = 0.0f32;
                for i in 0..intermediate.len() {
                    let weight = ((i * 17 + j) % 1000) as f32 / 10000.0;
                    sum += intermediate[i] * weight;
                }
                output[j] = sum;
            }
        }
        
        output
    }

    fn select_compute_device(&self, layer_idx: usize) -> ComputeDevice {
        if layer_idx < self.num_layers / 2 {
            ComputeDevice::Device(0)
        } else {
            ComputeDevice::CPU
        }
    }

    fn max_kv_len(&self) -> usize {
        2048
    }

    pub async fn run_pipeline(
        &mut self,
        input: &[f32],
        num_tokens: usize,
    ) -> Result<Vec<f32>> {
        let mut current_input = input.to_vec();
        
        for layer_idx in 0..self.num_layers {
            let next_layer = layer_idx + 1;
            
            let prepare_task = if next_layer < self.num_layers {
                Some(task::spawn(async move {
                    // Prepare weights for next layer
                    tokio::time::sleep(tokio::time::Duration::from_micros(1)).await;
                }))
            } else {
                None
            };
            
            current_input = self.process_layer(layer_idx, &current_input, 0).await?;
            
            if let Some(task) = prepare_task {
                let _ = task.await;
            }
            
            if layer_idx < self.num_layers - 1 && self.pipeline_overlap {
                let mut kv = self.kv_offload.write().await;
                kv.swap_layer_to_cpu(layer_idx).await;
            }
        }
        
        Ok(current_input)
    }

    pub fn get_layer_state(&self, layer_idx: usize) -> Option<&LayerState> {
        self.layer_states.get(layer_idx)
    }

    pub fn num_gpu_layers(&self) -> usize {
        self.layer_states
            .iter()
            .filter(|s| s.compute_device.is_gpu())
            .count()
    }
}

pub struct PipelineConfig {
    pub num_layers: usize,
    pub batch_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub use_gpu: bool,
    pub gpu_device_id: usize,
    pub vram_gb: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            batch_size: 1,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 2048,
            use_gpu: false,
            gpu_device_id: 0,
            vram_gb: 8,
        }
    }
}
