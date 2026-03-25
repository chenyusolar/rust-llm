use std::time::{Duration, Instant};
use std::collections::HashMap;

#[allow(unused_imports)]
use anyhow::Result;

pub struct PerfTimer {
    start: Instant,
    name: String,
}

impl PerfTimer {
    pub fn new(name: &str) -> Self {
        tracing::debug!("Starting: {}", name);
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        tracing::debug!("Finished {} in {:?}", self.name, self.elapsed());
    }
}

pub struct AsyncTimer {
    start: Instant,
    name: String,
}

impl AsyncTimer {
    pub async fn new(name: &str) -> Self {
        tracing::debug!("Starting async: {}", name);
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for AsyncTimer {
    fn drop(&mut self) {
        tracing::debug!("Finished async {} in {:?}", self.name, self.elapsed());
    }
}

pub fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

pub fn align_down(size: usize, align: usize) -> usize {
    size & !(align - 1)
}

pub fn div_up(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

pub fn round_up(x: usize, y: usize) -> usize {
    ((x + y - 1) / y) * y
}

pub fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

pub fn estimate_memory_for_kv_cache(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    num_seqs: usize,
) -> usize {
    let per_token = num_kv_heads * head_dim * 2 * 4;
    let per_seq = per_token * max_seq_len;
    per_seq * num_layers * num_seqs
}

pub fn estimate_model_memory(
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    use_quantization: bool,
) -> usize {
    let mut size = 0;
    
    size += vocab_size * hidden_size * 4;
    size += vocab_size * hidden_size * 4;
    
    let attention_weights = num_layers * hidden_size * (hidden_size + 2 * hidden_size);
    let ffn_weights = num_layers * hidden_size * (2 * intermediate_size + intermediate_size);
    size += attention_weights + ffn_weights;
    
    if use_quantization {
        size /= 2;
    }
    
    size
}

pub fn quantize_tensor_f32_to_f16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&x| f32_to_f16(x)).collect()
}

pub fn quantize_tensor_f32_to_q8(data: &[f32]) -> Vec<u8> {
    let scale = data.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b)) / 127.0;
    if scale == 0.0 {
        return vec![0u8; data.len()];
    }
    
    data.iter().map(|&x| {
        let quantized = (x / scale).round() as i8;
        ((quantized as i16 + 128) as u8).min(255)
    }).collect()
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) as u16;
    let exp = ((bits >> 23) & 0xff) as u16;
    let mant = (bits & 0x7fffff) as u16;
    
    if exp == 0 {
        sign
    } else if exp >= 143 {
        sign | 0x7c00
    } else {
        let shift = 13 - (exp as i32 - 127);
        let mantissa = if shift > 0 {
            mant | 0x400
        } else {
            mant >> (-shift)
        };
        
        let rounded = if (mant >> (13 - shift - 1)) & 1 == 1 {
            (mant >> (13 - shift)) + 1
        } else {
            mant >> (13 - shift)
        };
        
        let exp = ((exp as i32) - 127 + 13) as u16;
        
        sign | (exp << 10) | (rounded & 0x3ff)
    }
}

pub fn dequantize_tensor_q8_to_f32(data: &[u8], scale: f32) -> Vec<f32> {
    data.iter().map(|&x| {
        let quantized = (x as i16 - 128) as f32;
        quantized * scale
    }).collect()
}

pub struct MemoryPool {
    total_size: usize,
    used: usize,
    chunks: Vec<MemoryChunk>,
}

impl MemoryPool {
    pub fn new(total_size: usize) -> Self {
        Self {
            total_size,
            used: 0,
            chunks: Vec::new(),
        }
    }

    pub fn allocate(&mut self, size: usize, align: usize) -> Option<usize> {
        let aligned_size = align_up(size, align);
        
        if self.used + aligned_size > self.total_size {
            return None;
        }
        
        let offset = self.used;
        self.used += aligned_size;
        
        self.chunks.push(MemoryChunk {
            offset,
            size: aligned_size,
        });
        
        Some(offset)
    }

    pub fn reset(&mut self) {
        self.used = 0;
        self.chunks.clear();
    }

    pub fn available(&self) -> usize {
        self.total_size - self.used
    }
}

struct MemoryChunk {
    offset: usize,
    size: usize,
}

#[allow(dead_code)]
pub fn run_in_thread_pool<F, T, E>(f: F) -> std::result::Result<T, E>
where
    F: FnOnce() -> std::result::Result<T, E> + Send + 'static,
    T: Send + 'static,
    E: Send + 'static,
{
    todo!("Thread pool execution not implemented")
}

#[allow(dead_code)]
pub fn parallel_for<T, F>(_data: &[T], _f: F)
where
    F: Fn(&T) + Sync + Send + Clone,
{
    todo!("Parallel for not implemented")
}

#[allow(dead_code)]
pub fn parallel_for_mut<T, F>(_data: &mut [T], _f: F)
where
    F: Fn(&mut T) + Sync + Send + Clone,
{
    todo!("Parallel for mut not implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align() {
        assert_eq!(align_up(100, 64), 128);
        assert_eq!(align_up(128, 64), 128);
        assert_eq!(align_up(129, 64), 192);
    }

    #[test]
    fn test_quantize() {
        let data = vec![0.0f32, 1.0, -1.0, 0.5, -0.5];
        let quantized = quantize_tensor_f32_to_q8(&data);
        assert_eq!(quantized.len(), data.len());
    }
}
