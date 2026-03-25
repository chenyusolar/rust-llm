use crate::quantize::QuantizedTensor;
use crate::types::*;

pub struct KvCacheQuantizer {
    quantization_bits: u8,
    block_size: usize,
}

impl KvCacheQuantizer {
    pub fn new(quantization_bits: u8) -> Self {
        Self {
            quantization_bits: quantization_bits.max(2).min(8),
            block_size: 32,
        }
    }

    pub fn quantize(&self, data: &[f32]) -> Vec<u8> {
        let n = data.len();
        let block_size = self.block_size;
        let n_blocks = (n + block_size - 1) / block_size;

        let mut quantized =
            Vec::with_capacity(n_blocks * (block_size / 8 * self.quantization_bits as usize + 2));

        for block in 0..n_blocks {
            let start = block * block_size;
            let end = (start + block_size).min(n);
            let actual_size = end - start;

            let block_data = &data[start..end];

            let max_val = block_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let min_val = block_data.iter().fold(0.0f32, |a, &b| a.min(b));

            let scale = if max_val > 0.0 {
                (1 << self.quantization_bits as usize) as f32 / (max_val - min_val)
            } else {
                1.0
            };

            let scale_bytes = (scale as u16).to_le_bytes();
            quantized.extend_from_slice(&scale_bytes);

            let zero_bytes = ((-min_val * scale) as u8).to_le_bytes();
            quantized.extend_from_slice(&zero_bytes);

            let bits_per_value = self.quantization_bits;
            let mut current_byte = 0u8;
            let mut bits_used = 0;

            for i in 0..actual_size {
                let value = ((block_data[i] - min_val) * scale) as u8;
                let mask = (1 << bits_per_value) - 1;
                let quantized_val = value & mask;

                current_byte |= quantized_val << bits_used;
                bits_used += bits_per_value;

                if bits_used >= 8 {
                    quantized.push(current_byte);
                    current_byte = 0;
                    bits_used = 0;
                }
            }

            if bits_used > 0 {
                quantized.push(current_byte);
            }
        }

        quantized
    }

    pub fn dequantize(&self, quantized: &[f32], original_size: usize) -> Vec<f32> {
        let block_size = self.block_size;
        let n_blocks = (original_size + block_size - 1) / block_size;

        let mut result = Vec::with_capacity(original_size);

        let mut offset = 0;

        for _ in 0..n_blocks {
            if offset + 2 > quantized.len() {
                break;
            }

            let scale = quantized[offset];
            offset += 1;

            let zero = quantized.get(offset).copied().unwrap_or(0.0);
            offset += 1;

            for _ in 0..block_size {
                if result.len() >= original_size {
                    break;
                }

                result.push(scale * result.len() as f32 + zero);
            }
        }

        result
    }

    pub fn quantize_fp8(&self, data: &[f32]) -> Vec<u8> {
        data.iter()
            .map(|&v| {
                if v > 240.0 {
                    255
                } else if v < -240.0 {
                    0
                } else {
                    (v + 128.0).round() as u8
                }
            })
            .collect()
    }

    pub fn dequantize_fp8(&self, quantized: &[u8]) -> Vec<f32> {
        quantized.iter().map(|&v| v as f32 - 128.0).collect()
    }
}

pub struct QuantizedKvCache {
    pub k_cache: Vec<u8>,
    pub v_cache: Vec<u8>,
    pub k_scales: Vec<f32>,
    pub v_scales: Vec<f32>,
    pub shape: (usize, usize),
    pub dtype: DataType,
}

impl QuantizedKvCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let cache_size = num_layers * max_seq_len * num_kv_heads * head_dim;

        Self {
            k_cache: vec![0u8; cache_size / 8],
            v_cache: vec![0u8; cache_size / 8],
            k_scales: vec![1.0f32; num_layers * max_seq_len],
            v_scales: vec![1.0f32; num_layers * max_seq_len],
            shape: (cache_size, cache_size),
            dtype: DataType::Q8_0,
        }
    }

    pub fn write(
        &mut self,
        layer_idx: usize,
        pos: usize,
        num_heads: usize,
        head_dim: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) {
        let quantizer = KvCacheQuantizer::new(8);

        let offset = layer_idx * num_heads * head_dim * 1024 + pos * num_heads * head_dim;

        if offset + k_data.len() <= self.k_cache.len() {
            let k_quantized = quantizer.quantize_fp8(k_data);
            for (i, &v) in k_quantized.iter().enumerate() {
                if offset + i < self.k_cache.len() {
                    self.k_cache[offset + i] = v;
                }
            }
        }

        if offset + v_data.len() <= self.v_cache.len() {
            let v_quantized = quantizer.quantize_fp8(v_data);
            for (i, &v) in v_quantized.iter().enumerate() {
                if offset + i < self.v_cache.len() {
                    self.v_cache[offset + i] = v;
                }
            }
        }
    }

    pub fn read(
        &self,
        layer_idx: usize,
        pos: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let quantizer = KvCacheQuantizer::new(8);

        let offset = layer_idx * num_heads * head_dim * 1024 + pos * num_heads * head_dim;

        let k_size = num_heads * head_dim;
        let v_size = num_heads * head_dim;

        let k_slice = self.k_cache.get(offset..offset + k_size)?.to_vec();
        let v_slice = self.v_cache.get(offset..offset + v_size)?.to_vec();

        let k_data: Vec<f32> = quantizer.dequantize_fp8(&k_slice);
        let v_data: Vec<f32> = quantizer.dequantize_fp8(&v_slice);

        Some((k_data, v_data))
    }

    pub fn memory_size(&self) -> usize {
        self.k_cache.len() + self.v_cache.len() + self.k_scales.len() * 4 + self.v_scales.len() * 4
    }
}

pub struct AdaptiveKvCache {
    quantized: QuantizedKvCache,
    full_precision: Option<(Vec<f32>, Vec<f32>)>,
    use_full_precision: bool,
    layer_idx: usize,
}

impl AdaptiveKvCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            quantized: QuantizedKvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len),
            full_precision: None,
            use_full_precision: false,
            layer_idx: 0,
        }
    }

    pub fn set_layer(&mut self, layer_idx: usize) {
        self.layer_idx = layer_idx;
    }

    pub fn write(
        &mut self,
        pos: usize,
        num_heads: usize,
        head_dim: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) {
        if self.use_full_precision {
            self.full_precision = Some((k_data.to_vec(), v_data.to_vec()));
        } else {
            self.quantized
                .write(self.layer_idx, pos, num_heads, head_dim, k_data, v_data);
        }
    }

    pub fn read(
        &self,
        pos: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        if self.use_full_precision {
            self.full_precision.clone()
        } else {
            self.quantized
                .read(self.layer_idx, pos, num_heads, head_dim)
        }
    }

    pub fn switch_to_full_precision(&mut self) {
        self.use_full_precision = true;
    }

    pub fn switch_to_quantized(&mut self) {
        self.use_full_precision = false;
    }

    pub fn memory_size(&self) -> usize {
        if self.use_full_precision {
            self.full_precision
                .as_ref()
                .map_or(0, |(k, v)| k.len() * 4 + v.len() * 4)
        } else {
            self.quantized.memory_size()
        }
    }
}
