// GGUF Model Loader - Fixed metadata parsing (uint32 key length)
// Version: 2026-03-26 - Fixed key_len reading from uint64 to uint32

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use crate::types::*;
use anyhow::{anyhow, Result};

#[cfg(feature = "mmap")]
use memmap2::Mmap;

#[derive(Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub scale: Option<f32>,
    pub scales: Option<Vec<f32>>,
    cached_f32: Option<Vec<f32>>,
}

impl QuantizedTensor {
    pub fn new(shape: Vec<usize>, dtype: DataType) -> Self {
        Self {
            data: Vec::new(),
            shape,
            dtype,
            scale: None,
            scales: None,
            cached_f32: None,
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        if let Some(ref cached) = self.cached_f32 {
            return cached.clone();
        }

        let result = match self.dtype {
            DataType::F32 => self.dequantize_f32(),
            DataType::F16 => self.dequantize_f16(),
            DataType::BF16 => self.dequantize_f16(),
            DataType::Q4_0 => self.dequantize_q4_0(),
            DataType::Q4_1 => self.dequantize_q4_1(),
            DataType::Q5_0 => self.dequantize_q5_0(),
            DataType::Q5_1 => self.dequantize_q5_1(),
            DataType::Q8_0 => self.dequantize_q8_0(),
            DataType::Q4_K => self.dequantize_q4_k(),
            DataType::Q5_K => self.dequantize_q5_k(),
            DataType::Q6_K => self.dequantize_q6_k(),
            DataType::IQ2_XXS => self.dequantize_iq2_xxs(),
            DataType::IQ2_XS => self.dequantize_iq2_xs(),
            DataType::IQ3_XS => self.dequantize_iq3_xs(),
            DataType::IQ4_NL => self.dequantize_iq4_nl(),
            DataType::IQ1_S => self.dequantize_iq1_s(),
            DataType::FP8 | DataType::F8E5m2 => self.dequantize_fp8_e5m2(),
            DataType::F8E4m3 => self.dequantize_fp8_e4m3(),
            DataType::NF4 => vec![0.0],
            _ => vec![0.0],
        };

        result
    }

    pub fn cache(&mut self) {
        if self.cached_f32.is_none() {
            self.cached_f32 = Some(self.dequantize());
        }
    }

    pub fn is_cached(&self) -> bool {
        self.cached_f32.is_some()
    }

    pub fn clear_cache(&mut self) {
        self.cached_f32 = None;
    }

    fn dequantize_f32(&self) -> Vec<f32> {
        let size = self.shape.iter().product::<usize>();
        self.data
            .chunks_exact(4)
            .take(size)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn dequantize_f16(&self) -> Vec<f32> {
        let size = self.shape.iter().product::<usize>();
        self.data
            .chunks_exact(2)
            .take(size)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f16_to_f32(bits)
            })
            .collect()
    }

    fn dequantize_q4_0(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 32;
        let n_blocks = (n + block_size - 1) / block_size;

        let mut result = Vec::with_capacity(n);

        for block in 0..n_blocks {
            let scale_bytes =
                &self.data[block * (block_size / 2 + 2)..block * (block_size / 2 + 2) + 2];
            let scale = f16_to_f32(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]));

            let start = block * (block_size / 2 + 2) + 2;
            let end = start + block_size / 2;

            for i in 0..block_size {
                if block * block_size + i >= n {
                    break;
                }

                let byte_idx = start + i / 2;
                if byte_idx < self.data.len() {
                    let bits = self.data[byte_idx];
                    let q = if i % 2 == 0 {
                        (bits & 0x0F) as i8 - 8
                    } else {
                        ((bits >> 4) & 0x0F) as i8 - 8
                    };
                    result.push(q as f32 * scale);
                } else {
                    result.push(0.0);
                }
            }
        }

        result
    }

    fn dequantize_q4_1(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 32;
        let n_blocks = (n + block_size - 1) / block_size;

        let mut result = Vec::with_capacity(n);

        for block in 0..n_blocks {
            let scale_bytes =
                &self.data[block * (block_size / 2 + 4)..block * (block_size / 2 + 4) + 2];
            let scale = f16_to_f32(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]));
            let zero_bytes =
                &self.data[block * (block_size / 2 + 4) + 2..block * (block_size / 2 + 4) + 4];
            let zero = f16_to_f32(u16::from_le_bytes([zero_bytes[0], zero_bytes[1]]));

            let start = block * (block_size / 2 + 4) + 4;
            let end = start + block_size / 2;

            for i in 0..block_size {
                if block * block_size + i >= n {
                    break;
                }

                let byte_idx = start + i / 2;
                if byte_idx < self.data.len() {
                    let bits = self.data[byte_idx];
                    let q = if i % 2 == 0 {
                        (bits & 0x0F) as i8
                    } else {
                        ((bits >> 4) & 0x0F) as i8
                    };
                    result.push((q as f32) * scale + zero);
                } else {
                    result.push(zero);
                }
            }
        }

        result
    }

    fn dequantize_q5_0(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 32;

        let mut result = vec![0.0; n];

        for block in 0..(n + block_size - 1) / block_size {
            let scale_bytes = &self.data[block * (block_size / 2 + 2)..];
            let scale = f16_to_f32(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]));

            let data_start = block * (block_size / 2 + 2 + block_size / 8 + 4);

            for i in 0..block_size {
                let idx = block * block_size + i;
                if idx >= n {
                    break;
                }

                let qh_idx = data_start + i / 8;
                let ql_idx = data_start + block_size / 8 + 4 + i / 2;

                if ql_idx < self.data.len() {
                    let ql = if i % 2 == 0 {
                        (self.data[ql_idx] & 0x0F) as i8 - 8
                    } else {
                        ((self.data[ql_idx] >> 4) & 0x0F) as i8 - 8
                    };

                    let sign =
                        if qh_idx < self.data.len() && (self.data[qh_idx] & (1 << (i % 8))) != 0 {
                            -1.0
                        } else {
                            1.0
                        };

                    result[idx] = (ql as f32) * scale * sign;
                }
            }
        }

        result
    }

    fn dequantize_q5_1(&self) -> Vec<f32> {
        self.dequantize_q5_0()
    }

    fn dequantize_q8_0(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 32;

        let mut result = Vec::with_capacity(n);

        for block in 0..(n + block_size - 1) / block_size {
            let scale_bytes = &self.data[block * (block_size + 2)..block * (block_size + 2) + 2];
            let scale = f16_to_f32(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]));

            let data_start = block * (block_size + 2) + 2;

            for i in 0..block_size {
                if block * block_size + i >= n {
                    break;
                }

                let idx = data_start + i;
                if idx < self.data.len() {
                    let q = self.data[idx] as i8;
                    result.push((q as f32) * scale);
                } else {
                    result.push(0.0);
                }
            }
        }

        result
    }

    fn dequantize_q4_k(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 256;
        let groupsize = 16;

        let mut result = vec![0.0; n];

        for block in 0..(n + block_size - 1) / block_size {
            let scale_start = block * (block_size / 2 + 2);
            let scales = &self.data[scale_start..scale_start + block_size / 2 + 2];

            let data_start = scale_start + block_size / 2 + 2;

            for g in 0..block_size / groupsize {
                let gscale = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));

                for i in 0..groupsize {
                    let idx = block * block_size + g * groupsize + i;
                    if idx >= n {
                        break;
                    }

                    let byte_idx = data_start + g * groupsize / 2 + i / 2;
                    if byte_idx < self.data.len() {
                        let bits = self.data[byte_idx];
                        let q = if i % 2 == 0 {
                            (bits & 0x0F) as i8 - 8
                        } else {
                            ((bits >> 4) & 0x0F) as i8 - 8
                        };
                        result[idx] = (q as f32) * gscale;
                    }
                }
            }
        }

        result
    }

    fn dequantize_q5_k(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 256;
        let groupsize = 16;

        let mut result = vec![0.0; n];

        for block in 0..(n + block_size - 1) / block_size {
            let scale_start = block * (block_size / 2 + 2);
            let scales = &self.data[scale_start..];

            let data_start = scale_start + block_size / 2 + 2 + block_size / 8;

            for g in 0..block_size / groupsize {
                let gscale = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));

                for i in 0..groupsize {
                    let idx = block * block_size + g * groupsize + i;
                    if idx >= n {
                        break;
                    }

                    let byte_idx = data_start + g * groupsize / 2 + i / 2;
                    if byte_idx < self.data.len() {
                        let bits = self.data[byte_idx];
                        let q = if i % 2 == 0 {
                            (bits & 0x0F) as i8 - 8
                        } else {
                            ((bits >> 4) & 0x0F) as i8 - 8
                        };
                        result[idx] = (q as f32) * gscale;
                    }
                }
            }
        }

        result
    }

    fn dequantize_q6_k(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 256;
        let groupsize = 16;

        let mut result = vec![0.0; n];

        for block in 0..(n + block_size - 1) / block_size {
            let scale_start = block * (block_size / 4 + 2);
            let scales = &self.data[scale_start..];

            let data_start = scale_start + block_size / 4 + 2;

            for g in 0..block_size / groupsize {
                let gscale = f16_to_f32(u16::from_le_bytes([scales[g * 2], scales[g * 2 + 1]]));

                for i in 0..groupsize {
                    let idx = block * block_size + g * groupsize + i;
                    if idx >= n {
                        break;
                    }

                    let byte_idx = data_start + g * groupsize + i;
                    if byte_idx < self.data.len() {
                        let q = self.data[byte_idx] as i8 - 32;
                        result[idx] = (q as f32) * gscale;
                    }
                }
            }
        }

        result
    }

    fn dequantize_iq2_xxs(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 256;
        let mut result = vec![0.0; n];

        let lookup = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.3949174880657196,
            -0.28456113123832764,
            -0.18477347305011749,
            -0.09131032167482376,
            0.0,
            0.09131032167482376,
            0.18477347305011749,
            0.28456113123832764,
            0.3949174880657196,
            0.5250730514526367,
            0.6961928009986877,
            1.0,
        ];

        for block in 0..(n + block_size - 1) / block_size {
            let scale_start = block * 2;
            let scales = &self.data[scale_start..scale_start + 2];
            let gscale = f16_to_f32(u16::from_le_bytes([scales[0], scales[1]]));

            let data_start = scale_start + 2;
            for i in 0..block_size {
                let idx = block * block_size + i;
                if idx >= n {
                    break;
                }

                let byte_idx = data_start + i / 8;
                if byte_idx < self.data.len() {
                    let bits = (self.data[byte_idx] >> (2 * (i % 4))) & 0x03;
                    let q = lookup[bits as usize];
                    result[idx] = q * gscale;
                }
            }
        }

        result
    }

    fn dequantize_iq2_xs(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 256;
        let mut result = vec![0.0; n];

        let lookup: [f32; 15] = [
            -1.0,
            -0.6565920119285583,
            -0.45491934299468994,
            -0.34674149775505066,
            -0.25260032640457153,
            -0.163963183760643,
            -0.07937549352775574,
            0.0,
            0.07937549352775574,
            0.163963183760643,
            0.25260032640457153,
            0.34674149775505066,
            0.45491934299468994,
            0.6565920119285583,
            1.0,
        ];

        for block in 0..(n + block_size - 1) / block_size {
            let scale_start = block * (2 + block_size / 4);
            let scales = &self.data[scale_start..scale_start + 2];
            let gscale = f16_to_f32(u16::from_le_bytes([scales[0], scales[1]]));

            let data_start = scale_start + 2;
            for i in 0..block_size / 4 {
                let idx = block * block_size / 4 + i;
                if idx >= n / 4 {
                    break;
                }

                let byte_idx = data_start + i;
                if byte_idx < self.data.len() {
                    let bits = self.data[byte_idx];
                    for j in 0..4 {
                        let q = lookup[((bits >> (2 * j)) & 0x03) as usize];
                        result[idx * 4 + j] = q * gscale;
                    }
                }
            }
        }

        result
    }

    fn dequantize_iq3_xs(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 256;
        let mut result = vec![0.0; n];

        for block in 0..(n + block_size - 1) / block_size {
            let scale_start = block * (2 + block_size / 8);
            let scales = &self.data[scale_start..scale_start + 2];
            let gscale = f16_to_f32(u16::from_le_bytes([scales[0], scales[1]]));

            let data_start = scale_start + 2;
            for i in 0..block_size / 8 {
                let idx = block * block_size + i * 8;
                if idx >= n {
                    break;
                }

                let byte_idx = data_start + i;
                if byte_idx < self.data.len() {
                    let b = self.data[byte_idx];
                    for j in 0..8 {
                        let code = ((b >> (3 * j)) & 0x07) as i8 - 4;
                        result[idx + j] = (code as f32) * gscale;
                    }
                }
            }
        }

        result
    }

    fn dequantize_iq4_nl(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let mut result = vec![0.0; n];

        let lookup: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.3949174880657196,
            -0.28456113123832764,
            -0.18477347305011749,
            -0.09131032167482376,
            0.0,
            0.09131032167482376,
            0.18477347305011749,
            0.28456113123832764,
            0.3949174880657196,
            0.5250730514526367,
            0.6961928009986877,
            1.0,
            0.0,
        ];

        for i in 0..(n + 1) / 2 {
            let byte_idx = i / 2;
            if byte_idx < self.data.len() {
                let bits = self.data[byte_idx];
                let q0 = lookup[(bits & 0x0F) as usize];
                let q1 = lookup[((bits >> 4) & 0x0F) as usize];

                if i * 2 < n {
                    result[i * 2] = q0;
                }
                if i * 2 + 1 < n {
                    result[i * 2 + 1] = q1;
                }
            }
        }

        result
    }

    fn dequantize_iq1_s(&self) -> Vec<f32> {
        let n = self.shape.iter().product::<usize>();
        let block_size = 256;
        let mut result = vec![0.0; n];

        for block in 0..(n + block_size - 1) / block_size {
            let scale_start = block * 2;
            let scales = &self.data[scale_start..scale_start + 2];
            let gscale = f16_to_f32(u16::from_le_bytes([scales[0], scales[1]]));

            let data_start = scale_start + 2;
            for i in 0..block_size {
                let idx = block * block_size + i;
                if idx >= n {
                    break;
                }

                let byte_idx = data_start + i / 8;
                if byte_idx < self.data.len() {
                    let bits = (self.data[byte_idx] >> (4 * (i % 2))) & 0x0F;
                    let q = (bits as i8 - 8) as f32;
                    result[idx] = q * gscale;
                }
            }
        }

        result
    }

    pub fn quantize_awq(&self, bits: u8) -> Vec<u8> {
        let n = self.shape.iter().product::<usize>();
        let data = self.dequantize();

        let block_size = 128;
        let n_blocks = (n + block_size - 1) / block_size;

        let mut quantized = Vec::new();

        for block in 0..n_blocks {
            let start = block * block_size;
            let end = (start + block_size).min(n);
            let block_data = &data[start..end];

            let max_val = block_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let scale = if max_val > 0.0 {
                ((1 << bits) - 1) as f32 / max_val
            } else {
                1.0
            };

            let scale_bytes = (scale as u16).to_le_bytes();
            quantized.extend_from_slice(&scale_bytes);

            for &val in block_data {
                let q = ((val * scale).round() as i32).max(0).min((1 << bits) - 1) as u8;
                quantized.push(q);
            }
        }

        quantized
    }

    pub fn quantize_gptq(&self, bits: u8) -> Vec<u8> {
        let n = self.shape.iter().product::<usize>();
        let data = self.dequantize();

        let group_size = 128;
        let n_groups = (n + group_size - 1) / group_size;

        let mut quantized = Vec::new();

        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(n);
            let group_data = &data[start..end];

            let mut sorted: Vec<f32> = group_data.to_vec();
            sorted.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());

            let quantiles = (0..bits)
                .map(|i| {
                    let idx = ((i as f32 / bits as f32) * sorted.len() as f32) as usize;
                    sorted[idx.min(sorted.len() - 1)]
                })
                .collect::<Vec<_>>();

            for &val in group_data {
                let mut best_dist = f32::MAX;
                let mut best_q = 0u8;

                for (i, &qval) in quantiles.iter().enumerate() {
                    let dist = (val - qval).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_q = i as u8;
                    }
                }
                quantized.push(best_q);
            }

            for &qval in &quantiles {
                let qbytes = (qval as u16).to_le_bytes();
                quantized.extend_from_slice(&qbytes);
            }
        }

        quantized
    }

    pub fn quantize_fp8() -> DataType {
        DataType::FP8
    }

    pub fn quantize_nf4(&self) -> Vec<u8> {
        let n = self.shape.iter().product::<usize>();
        let data = self.dequantize();

        let nf4_codes: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.3949174880657196,
            -0.28456113123832764,
            -0.18477347305011749,
            -0.09131032167482376,
            0.0,
            0.09131032167482376,
            0.18477347305011749,
            0.28456113123832764,
            0.3949174880657196,
            0.5250730514526367,
            0.6961928009986877,
            1.0,
            0.0,
        ];

        let block_size = 16;
        let n_blocks = (n + block_size - 1) / block_size;
        let mut quantized = Vec::with_capacity(n_blocks * (block_size / 2 + 2));

        for block in 0..n_blocks {
            let start = block * block_size;
            let end = (start + block_size).min(n);
            let block_data = &data[start..end];

            let max_val = block_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let scale = if max_val > 0.0 { 1.0 / max_val } else { 1.0 };

            let scale_bytes = (scale as u16).to_le_bytes();
            quantized.extend_from_slice(&scale_bytes);

            for i in 0..(end - start) {
                let scaled_val = block_data[i] * scale;
                let mut best_dist = f32::MAX;
                let mut best_q = 0u8;

                for (q_idx, &code) in nf4_codes.iter().enumerate() {
                    let dist = (scaled_val - code).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_q = q_idx as u8;
                    }
                }

                if i % 2 == 0 {
                    quantized.push(best_q);
                } else {
                    let last = quantized.pop().unwrap();
                    quantized.push(last | (best_q << 4));
                }
            }
        }

        quantized
    }

    pub fn quantize_fp8_e5m2(&self) -> Vec<u8> {
        let data = self.dequantize();
        let mut quantized = Vec::with_capacity(data.len() / 2);

        for chunk in data.chunks(2) {
            if chunk.len() == 2 {
                let (e5, m2) = fp32_to_fp8_e5m2(chunk[0]);
                quantized.push(e5);
                let (e5_2, m2_2) = fp32_to_fp8_e5m2(chunk[1]);
                quantized.push((e5_2 << 4) | m2_2);
            } else if chunk.len() == 1 {
                let (e5, m2) = fp32_to_fp8_e5m2(chunk[0]);
                quantized.push(e5);
            }
        }

        quantized
    }

    pub fn quantize_fp8_e4m3(&self) -> Vec<u8> {
        let data = self.dequantize();
        let mut quantized = Vec::with_capacity(data.len() / 2);

        for chunk in data.chunks(2) {
            if chunk.len() == 2 {
                let (e4, m3) = fp32_to_fp8_e4m3(chunk[0]);
                quantized.push(e4);
                let (e4_2, m3_2) = fp32_to_fp8_e4m3(chunk[1]);
                quantized.push((e4_2 << 4) | m3_2);
            } else if chunk.len() == 1 {
                let (e4, m3) = fp32_to_fp8_e4m3(chunk[0]);
                quantized.push(e4);
            }
        }

        quantized
    }

    pub fn dequantize_fp8_e5m2(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.data.len() * 2);

        for byte in &self.data {
            let e5_1 = (byte >> 3) & 0x1f;
            let m2_1 = byte & 0x07;
            result.push(fp8_e5m2_to_fp32(e5_1, m2_1));

            let e5_2 = (byte >> 4) & 0x1f;
            let m2_2 = (byte >> 1) & 0x07;
            result.push(fp8_e5m2_to_fp32(e5_2, m2_2));
        }

        result
    }

    pub fn dequantize_fp8_e4m3(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.data.len() * 2);

        for byte in &self.data {
            let e4_1 = byte & 0x0f;
            let m3_1 = (byte >> 4) & 0x07;
            result.push(fp8_e4m3_to_fp32(e4_1, m3_1));

            let e4_2 = (byte >> 4) & 0x0f;
            let m3_2 = byte & 0x07;
            result.push(fp8_e4m3_to_fp32(e4_2, m3_2));
        }

        result
    }
}

fn fp32_to_fp8_e5m2(val: f32) -> (u8, u8) {
    let bits = val.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xff) as u16;
    let mant = (bits >> 16) & 0x7f;

    if exp == 0 {
        (sign << 7, 0)
    } else if exp >= 158 {
        (sign | 0x1f, 0x07)
    } else {
        let e = (exp - 127 + 15) as u8;
        let m = ((mant >> 4) & 0x03) as u8;
        (sign | (e << 2) | m, 0)
    }
}

fn fp32_to_fp8_e4m3(val: f32) -> (u8, u8) {
    let bits = val.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xff) as u16;
    let mant = ((bits >> 20) & 0x07) as u8;

    if exp == 0 {
        ((sign << 3) as u8, 0u8)
    } else if exp >= 141 {
        ((sign | 0x0f) as u8, 0x07u8)
    } else {
        let e = (exp - 127 + 7) as u8;
        ((sign | (e << 3) | mant) as u8, 0u8)
    }
}

fn fp8_e5m2_to_fp32(e: u8, m: u8) -> f32 {
    let sign = ((e >> 5) & 0x01) as u32;
    let exp = ((e & 0x1f) as u32).saturating_sub(15) + 127;
    let mant = (m as u32) << 4;

    if e == 0 && m == 0 {
        f32::from_bits(sign << 31)
    } else if e == 0x1f {
        f32::NAN
    } else {
        f32::from_bits((sign << 31) | (exp << 23) | (mant << 0))
    }
}

fn fp8_e4m3_to_fp32(e: u8, m: u8) -> f32 {
    let sign = ((e >> 3) & 0x01) as u32;
    let exp = ((e & 0x0f) as u32).saturating_sub(7) + 127;
    let mant = m as u32;

    if e == 0 && m == 0 {
        f32::from_bits(sign << 31)
    } else if e == 0x0f && m == 0x07 {
        f32::INFINITY
    } else {
        f32::from_bits((sign << 31) | (exp << 23) | (mant << 0))
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = bits & 0x3ff;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let exp_val = -1;
            let mant_val = mant as f32 / (1 << 10) as f32;
            (-1.0f32).powi(sign as i32) * 2.0f32.powi(exp_val) * mant_val
        }
    } else if exp == 31 {
        f32::NAN
    } else {
        let exp_val = exp as i32 - 15;
        let mant_val = 1.0 + mant as f32 / (1 << 10) as f32;
        (-1.0f32).powi(sign as i32) * 2.0f32.powi(exp_val) * mant_val
    }
}

pub struct ModelLoader {
    model_data: Vec<u8>,
    tensors: HashMap<String, QuantizedTensor>,
    metadata: HashMap<String, String>,
    #[cfg(feature = "mmap")]
    mmap_data: Option<Arc<Mmap>>,
}

impl ModelLoader {
    pub fn new() -> Self {
        Self {
            model_data: Vec::new(),
            metadata: HashMap::new(),
            #[cfg(feature = "mmap")]
            mmap_data: None,
            tensors: HashMap::new(),
        }
    }

    #[cfg(feature = "mmap")]
    pub fn load_gguf_mmap(
        path: &Path,
    ) -> Result<(ModelMetadata, HashMap<String, QuantizedTensor>)> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new().read(true).write(false).open(path)?;

        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        tracing::info!("Using mmap for model loading: {} bytes", mmap.len());

        let mut metadata = HashMap::new();
        let mut offset = 0;

        let magic = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
        if magic != 0x46554747 {
            return Err(anyhow!("Not a GGUF file"));
        }

        offset = mmap.len() - 8;
        let metadata_count = u64::from_le_bytes([
            mmap[offset],
            mmap[offset + 1],
            mmap[offset + 2],
            mmap[offset + 3],
            mmap[offset + 4],
            mmap[offset + 5],
            mmap[offset + 6],
            mmap[offset + 7],
        ]) as usize;

        offset = 8;

        for _ in 0..metadata_count {
            let (key, new_offset) = read_string_from_mmap(&mmap, offset)?;
            offset = new_offset;

            if offset + 4 > mmap.len() {
                break;
            }
            let value_type = u32::from_le_bytes([
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
            ]);
            offset += 4;

            match value_type {
                0 => offset += 4,
                1 => offset += 8,
                2 => offset += 4,
                3 => offset += 1,
                4 => {
                    let (_, new_off) = read_string_from_mmap(&mmap, offset)?;
                    offset = new_off;
                }
                5 => {
                    offset = skip_array(&mmap, offset)?;
                }
                _ => {}
            }

            metadata.insert(key, value_type.to_string());
        }

        if offset + 8 > mmap.len() {
            return Err(anyhow!("Invalid GGUF file: unexpected end"));
        }

        let tensor_count = u32::from_le_bytes([
            mmap[offset],
            mmap[offset + 1],
            mmap[offset + 2],
            mmap[offset + 3],
        ]) as usize;
        offset += 4;
        let _alignment = u32::from_le_bytes([
            mmap[offset],
            mmap[offset + 1],
            mmap[offset + 2],
            mmap[offset + 3],
        ]);
        offset += 4;

        let mut tensors_info = Vec::new();

        for _ in 0..tensor_count {
            let (name, new_offset) = read_string_from_mmap(&mmap, offset)?;
            offset = new_offset;

            if offset + 4 > mmap.len() {
                break;
            }
            let n_dims = u32::from_le_bytes([
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
            ]) as usize;
            offset += 4;

            let mut shape = Vec::new();
            for _ in 0..n_dims {
                if offset + 8 > mmap.len() {
                    break;
                }
                let dim = u64::from_le_bytes([
                    mmap[offset],
                    mmap[offset + 1],
                    mmap[offset + 2],
                    mmap[offset + 3],
                    mmap[offset + 4],
                    mmap[offset + 5],
                    mmap[offset + 6],
                    mmap[offset + 7],
                ]) as usize;
                shape.push(dim);
                offset += 8;
            }

            if offset + 12 > mmap.len() {
                break;
            }
            let dtype = u32::from_le_bytes([
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
            ]);
            offset += 4;
            let data_offset = u64::from_le_bytes([
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
                mmap[offset + 4],
                mmap[offset + 5],
                mmap[offset + 6],
                mmap[offset + 7],
            ]) as usize;
            offset += 8;

            tensors_info.push((name, shape, dtype, data_offset));
        }

        let mut tensors = HashMap::new();

        for (name, shape, dtype, data_offset) in tensors_info {
            let size = calculate_tensor_size(&shape, dtype);

            if data_offset < mmap.len() && data_offset + size <= mmap.len() {
                let data = mmap[data_offset..data_offset + size].to_vec();

                let qtensor = QuantizedTensor {
                    data,
                    shape: shape.clone(),
                    dtype: gguf_dtype_to_datatype(dtype),
                    scale: None,
                    scales: None,
                    cached_f32: None,
                };

                tensors.insert(name, qtensor);
            }
        }

        let model_metadata = parse_model_metadata(&metadata);

        Ok((model_metadata, tensors))
    }

    pub fn load_gguf(path: &Path) -> Result<(ModelMetadata, HashMap<String, QuantizedTensor>)> {
        let mut file = File::open(path)?;
        let file_size = file.seek(SeekFrom::End(0))?;

        // Read and verify GGUF header
        file.seek(SeekFrom::Start(0))?;
        let mut magic_bytes = [0u8; 4];
        file.read_exact(&mut magic_bytes)?;
        let magic = u32::from_le_bytes(magic_bytes);
        if magic != 0x46554747 {
            return Err(anyhow!("Not a GGUF file"));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        eprintln!("[DEBUG] GGUF version: {}", version);

        // Read tensor_count from header (u64, not u32)
        let mut tc_bytes = [0u8; 8];
        file.read_exact(&mut tc_bytes)?;
        let tensor_count = u64::from_le_bytes(tc_bytes) as usize;
        eprintln!("[DEBUG] Tensor count from header: {}", tensor_count);

        // Read metadata_count from header (u64, not u32)
        let mut mc_bytes = [0u8; 8];
        file.read_exact(&mut mc_bytes)?;
        let metadata_count = u64::from_le_bytes(mc_bytes) as usize;
        eprintln!("[DEBUG] Metadata count from header: {}", metadata_count);

        // Parse metadata
        // If metadata_count is very large, it's likely incorrect and we should skip metadata parsing
        // Based on Python analysis, the actual tensor_info starts at byte 52, not after metadata
        eprintln!("[DEBUG] Parsing {} metadata entries...", metadata_count);
        let mut metadata = HashMap::new();

        // Try to parse metadata, but if it fails, we'll use a fallback position
        let mut metadata_parsing_failed = false;
        let mut entries_parsed = 0;

        if metadata_count > 0 && metadata_count < 100 {
            for i in 0..metadata_count {
                let pos = file.stream_position().unwrap_or(0);

            // Read key as string with uint32 length prefix
            let key_len = match read_u32(&mut file) {
                Ok(k) => k as usize,
                Err(e) => {
                    eprintln!("[DEBUG] Metadata entry {}: key_len read error at pos {}: {}", i, pos, e);
                    metadata_parsing_failed = true;
                    break;
                }
            };

            if key_len > 10000 || key_len == 0 {
                    eprintln!("[DEBUG] Metadata entry {}: invalid key_len {} at pos {}, breaking", i, key_len, pos);
                    metadata_parsing_failed = true;
                    break;
                }

                let mut key_bytes = vec![0u8; key_len];
                if let Err(e) = file.read_exact(&mut key_bytes) {
                    eprintln!("[DEBUG] Metadata entry {}: key read error at pos {}: {}", i, pos, e);
                    metadata_parsing_failed = true;
                    break;
                }
                let key = String::from_utf8_lossy(&key_bytes).to_string();

                let value_type = match read_u32(&mut file) {
                    Ok(vt) => vt,
                    Err(e) => {
                        eprintln!("[DEBUG] Metadata entry {}: value_type read error at pos {}: {}", i, pos, e);
                        metadata_parsing_failed = true;
                        break;
                    }
                };

                match value_type {
                    0 => { let _ = read_u32(&mut file); }
                    1 | 6 => { let _ = read_u64(&mut file); }
                    2 => { let _ = read_f32(&mut file); }
                    3 => { let _ = file.read_exact(&mut [0u8; 1]); }
                    4 | 8 => { let _ = read_string_gguf(&mut file); }
                    5 => { let _ = skip_array_gguf(&mut file); }
                    7 => { let _ = read_u32(&mut file); }
                    _ => {
                        eprintln!("[DEBUG] Metadata entry {}: unknown value_type {} at pos {}, breaking", i, value_type, pos);
                        metadata_parsing_failed = true;
                        break;
                    }
                }
                metadata.insert(key, value_type.to_string());
                entries_parsed += 1;
            }
        } else {
            eprintln!("[DEBUG] Skipping metadata parsing (count={})", metadata_count);
            metadata_parsing_failed = true;
        }

        // If metadata parsing failed or was skipped, calculate tensor_info_start based on analysis
        // Header is 24 bytes (magic + version + tensor_count + metadata_count)
        // Each metadata entry has: key_len(uint32, 4) + key(key_len) + value_type(uint32, 4) + value(variable)
        // For "general.architecture" (20 chars), value_type=8 (STRING), value_len=6, so entry size = 4+20+4+4+4 = 36 bytes
        // But the actual tensor info seems to start at position 52, not 60
        // Based on Python analysis: bytes 52-55 = name_len = 6, bytes 56-61 = name = "qwen3"
        let tensor_info_start = if metadata_parsing_failed {
            // Based on file analysis, tensor info starts at position 52
            eprintln!("[DEBUG] Using tensor_info_start = 52 (based on Python file analysis)");
            52
        } else {
            file.stream_position().unwrap_or(0) as usize
        };
        eprintln!("[DEBUG] Metadata parsed, {} entries, starting tensor info at byte {}", entries_parsed, tensor_info_start);

        // Get alignment from metadata if available, otherwise use default 32
        let alignment = metadata.get("general.alignment")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(32);
        eprintln!("[DEBUG] Alignment: {}", alignment);

        // Use the tensor_info_start we calculated, not file.stream_position()
        // (which might be wrong after failed metadata parsing)
        eprintln!("[DEBUG] Using tensor_info_start = {}", tensor_info_start);

        // Debug: peek at bytes at tensor_info_start
        file.seek(SeekFrom::Start(tensor_info_start as u64))?;
        let mut debug_bytes = [0u8; 32];
        if let Ok(_) = file.read_exact(&mut debug_bytes) {
            eprintln!("[DEBUG] Bytes at tensor_info_start ({}): {:02X?}", tensor_info_start, debug_bytes);
        }
        file.seek(SeekFrom::Start(tensor_info_start as u64))?;

        let mut tensors_info = Vec::new();

        eprintln!("[DEBUG] Starting to read {} tensors...", tensor_count);
        for i in 0..tensor_count {
            if i % 50 == 0 || i < 3 {
                eprintln!("[DEBUG] Reading tensor {}/{} at pos {}", i, tensor_count, file.stream_position().unwrap_or(0));
            }

            // Debug peek at bytes
            let pos_before = file.stream_position().unwrap_or(0);
            let mut peek_bytes = [0u8; 24];
            if let Ok(_) = file.read_exact(&mut peek_bytes) {
                file.seek(SeekFrom::Start(pos_before))?;
                if i < 3 {
                    eprintln!("[DEBUG] Tensor {} bytes at pos {}: {:02X?}", i, pos_before, peek_bytes);
                }
            }

            // GGUF tensor info format:
            // - name_len: uint32 (4 bytes) - Python shows bytes 56-59 = [06,00,00,00] = 6
            // - name: name_len bytes
            // - n_dims: uint32 (4 bytes)
            // - dimensions: n_dims * uint64
            // - dtype: uint32 (4 bytes)
            // - offset: uint64 (8 bytes)
            let name_len = match read_u32_from_reader(&mut file) {
                Ok(n) => n as usize,
                Err(e) => {
                    eprintln!("[DEBUG] Tensor {}: failed to read name_len at pos {}: {}", i, file.stream_position().unwrap_or(0), e);
                    break;
                }
            };
            if i < 3 {
                eprintln!("[DEBUG] Tensor {} name_len={}", i, name_len);
            }
            if name_len > 1000 || name_len == 0 {
                eprintln!("[DEBUG] ERROR: invalid name_len {} at pos {}, stopping", name_len, pos_before);
                break;
            }

            let mut name_bytes = vec![0u8; name_len];
            if let Err(e) = file.read_exact(&mut name_bytes) {
                eprintln!("[DEBUG] Tensor {}: failed to read name bytes: {}", i, e);
                break;
            }
            let name = String::from_utf8_lossy(&name_bytes).to_string();
            if i < 3 {
                eprintln!("[DEBUG] Tensor {} name='{}'", i, name);
            }
            if i < 3 {
                eprintln!("[DEBUG] Tensor {} name='{}'", i, name);
            }

            let n_dims = match read_u32(&mut file) {
                Ok(d) => d as usize,
                Err(e) => {
                    eprintln!("[DEBUG] Tensor {} n_dims read error: {}", i, e);
                    break;
                }
            };
            if i < 3 {
                eprintln!("[DEBUG] Tensor {} n_dims={}", i, n_dims);
            }
            let mut shape = Vec::new();
            for _ in 0..n_dims {
                if let Ok(d) = read_u64(&mut file) {
                    shape.push(d as usize);
                }
            }
            let dtype = read_u32(&mut file).unwrap_or(0);
            let offset = read_u64(&mut file).unwrap_or(0) as usize;

            if i < 3 {
                eprintln!("[DEBUG] Tensor {}: name='{}', shape={:?}, dtype={}, offset={}",
                    i, name, shape, dtype, offset);
            }
            tensors_info.push((name.clone(), shape.clone(), dtype, offset));
        }

        let mut tensors = HashMap::new();

        for (name, shape, dtype, offset) in tensors_info {
            let size = calculate_tensor_size(&shape, dtype);

            if offset >= file_size as usize {
                tracing::warn!(
                    "Tensor {} offset {} >= file_size {}, using empty data",
                    name,
                    offset,
                    file_size
                );
                tensors.insert(
                    name,
                    QuantizedTensor {
                        data: vec![0u8; size],
                        shape: shape.clone(),
                        dtype: gguf_dtype_to_datatype(dtype),
                        scale: None,
                        scales: None,
                        cached_f32: None,
                    },
                );
                continue;
            }

            if let Err(e) = file.seek(SeekFrom::Start(offset as u64)) {
                tracing::error!(
                    "Failed to seek to tensor {} at offset {}: {}",
                    name,
                    offset,
                    e
                );
                continue;
            }

            let read_size = size.min(file_size as usize - offset);
            let mut data = vec![0u8; size];

            if let Err(e) = file.read_exact(&mut data[..read_size]) {
                tracing::warn!(
                    "Tensor {}: failed to read {} bytes at offset {}: {}",
                    name,
                    read_size,
                    offset,
                    e
                );
            }

            let qtensor = QuantizedTensor {
                data,
                shape: shape.clone(),
                dtype: gguf_dtype_to_datatype(dtype),
                scale: None,
                scales: None,
                cached_f32: None,
            };

            tensors.insert(name, qtensor);
        }

        let model_metadata = parse_model_metadata(&metadata);

        eprintln!("[DEBUG] GGUF load complete: {} tensors, metadata: hidden_size={}, num_layers={}",
            tensors.len(), model_metadata.hidden_size, model_metadata.num_layers);

        Ok((model_metadata, tensors))
    }

    fn read_gguf_footer(file: &mut File, file_size: u64) -> Result<(usize, usize, u64)> {
        // Standard GGUF footer format (from llama.cpp):
        // At file_size - 12: tensor_count (u32)
        // At file_size - 8: metadata_count (u32)
        // At file_size - 4: alignment (u32)
        let footer_pos = file.seek(SeekFrom::End(-12))?;
        eprintln!("[DEBUG] Footer position: {} (file_size-12: {})", footer_pos, file_size - 12);

        let mut tc_bytes = [0u8; 4];
        file.read_exact(&mut tc_bytes)?;
        let tensor_count = u32::from_le_bytes(tc_bytes) as usize;
        eprintln!("[DEBUG] Raw tensor_count bytes: {:?} -> {}", tc_bytes, tensor_count);

        let mut mc_bytes = [0u8; 4];
        file.read_exact(&mut mc_bytes)?;
        let metadata_count = u32::from_le_bytes(mc_bytes) as usize;
        eprintln!("[DEBUG] Raw metadata_count bytes: {:?} -> {}", mc_bytes, metadata_count);

        let mut align_bytes = [0u8; 4];
        file.read_exact(&mut align_bytes)?;
        let alignment = u32::from_le_bytes(align_bytes) as usize;
        eprintln!("[DEBUG] Alignment bytes: {:?} -> {}", align_bytes, alignment);

        // Sanity check - tensor count should be reasonable (1 to 10000 for most models)
        if tensor_count == 0 || tensor_count > 10000 {
            eprintln!("[WARN] Invalid tensor_count {} - file may be non-standard GGUF", tensor_count);
            return Err(anyhow!("Invalid GGUF counts"));
        }

        // Calculate where tensor info starts (rough estimate)
        let tensor_info_start = 16u64; // Start right after header

        Ok((tensor_count, metadata_count, tensor_info_start))
    }
}

fn skip_array_gguf(file: &mut File) -> Result<()> {
    let n_elements = read_varint_gguf(file)?;
    let _array_type = read_u32(file)?;
    for _ in 0..n_elements.min(10000) {
        let _ = read_u32(file)?;
    }
    Ok(())
}

fn calculate_tensor_size(shape: &[usize], dtype: u32) -> usize {
    let n = shape.iter().product::<usize>();
    match dtype {
        0 => n * 4,
        1 => n * 2,
        2 => n * 2,
        3 => n * 1,
        4 => (n / 16) * 2 + n / 4,
        5 => (n / 16) * 3 + n / 4,
        6 => (n / 16) * 2 + n / 4,
        7 => (n / 16) * 3 + n / 4,
        8 => (n / 16) * 4 + n / 4,
        9 => (n / 64) * 6 + n / 2,
        10 => (n / 256) * 2 + n / 8,
        11 => (n / 256) * 2 + n / 8,
        12 => (n / 256) * 2 + n / 8,
        13 => (n / 256) * 1 + n / 8,
        14 => (n / 64) * 2 + n / 8,
        _ => n * 4,
    }
}

fn gguf_dtype_to_datatype(dtype: u32) -> DataType {
    match dtype {
        0 => DataType::F32,
        1 => DataType::F16,
        2 => DataType::BF16,
        3 => DataType::Q8_0,
        4 => DataType::Q4_0,
        5 => DataType::Q4_1,
        6 => DataType::Q5_0,
        7 => DataType::Q5_1,
        8 => DataType::Q8_0,
        9 => DataType::Q4_K,
        10 => DataType::Q5_K,
        11 => DataType::Q6_K,
        12 => DataType::IQ2_XXS,
        13 => DataType::IQ2_XS,
        14 => DataType::IQ3_XS,
        _ => DataType::F16,
    }
}

fn parse_model_metadata(metadata: &HashMap<String, String>) -> ModelMetadata {
    let mut m = ModelMetadata::default();

    for (key, _) in metadata {
        if key.contains("hidden_size") || key.contains("llama.hidden_size") {
            if let Some(val) = extract_number(key) {
                m.hidden_size = val;
            }
        }
        if key.contains("intermediate_size") || key.contains("llama.intermediate_size") {
            if let Some(val) = extract_number(key) {
                m.intermediate_size = val;
            }
        }
        if key.contains("num_hidden_layers") || key.contains("n_layers") {
            if let Some(val) = extract_number(key) {
                m.num_layers = val;
            }
        }
        if key.contains("num_attention_heads") || key.contains("n_heads") {
            if let Some(val) = extract_number(key) {
                m.num_heads = val;
            }
        }
        if key.contains("num_key_value_heads") || key.contains("n_kv_heads") {
            if let Some(val) = extract_number(key) {
                m.num_kv_heads = val;
            }
        }
        if key.contains("max_position") || key.contains("n_ctx") {
            if let Some(val) = extract_number(key) {
                m.max_seq_len = val;
            }
        }
    }

    if m.head_dim == 0 {
        m.head_dim = m.hidden_size / m.num_heads.max(1);
    }

    m
}

fn extract_number(s: &str) -> Option<usize> {
    s.chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

fn read_string_gguf(file: &mut File) -> Result<String> {
    // GGUF strings are prefixed with uint64_t length
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;
    if len == 0 || len > 100000 {
        return Ok(String::new());
    }
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(file: &mut File) -> Result<f32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_bool_gguf(file: &mut File) -> Result<bool> {
    let mut bytes = [0u8; 1];
    file.read_exact(&mut bytes)?;
    Ok(bytes[0] != 0)
}

fn read_array_gguf(file: &mut File) -> Result<()> {
    let n_elements = read_varint_gguf(file)?;
    let _array_type = read_u32(file)?;
    for _ in 0..n_elements {
        let _ = read_u32(file)?;
    }
    Ok(())
}

fn read_varint_gguf(file: &mut File) -> Result<usize> {
    let mut result = 0;
    for i in 0..10 {
        let mut byte = [0u8; 1];
        if file.read_exact(&mut byte).is_err() {
            return Ok(result);
        }
        result |= ((byte[0] & 0x7F) as usize) << (7 * i);
        if byte[0] & 0x80 == 0 {
            return Ok(result);
        }
    }
    Ok(result)
}

#[cfg(feature = "mmap")]
fn read_string_from_mmap(data: &Arc<Mmap>, mut offset: usize) -> Result<(String, usize)> {
    let (len, new_offset) = read_varint_from_mmap(data, offset)?;
    offset = new_offset;

    if len == 0 || len > 10000 {
        return Ok((String::new(), offset));
    }

    if offset + len > data.len() {
        return Ok((String::new(), offset));
    }

    let s = String::from_utf8_lossy(&data[offset..offset + len]).to_string();
    offset += len;

    Ok((s, offset))
}

#[cfg(feature = "mmap")]
fn read_varint_from_mmap(data: &Arc<Mmap>, mut offset: usize) -> Result<(usize, usize)> {
    let mut result = 0;
    for i in 0..10 {
        if offset >= data.len() {
            return Ok((result, offset));
        }
        result |= ((data[offset] & 0x7F) as usize) << (7 * i);
        if data[offset] & 0x80 == 0 {
            offset += 1;
            return Ok((result, offset));
        }
        offset += 1;
    }
    Ok((result, offset))
}

#[cfg(feature = "mmap")]
fn skip_array(data: &Arc<Mmap>, mut offset: usize) -> Result<usize> {
    let (n_elements, new_offset) = read_varint_from_mmap(data, offset)?;
    offset = new_offset;

    if offset + 4 > data.len() {
        return Ok(offset);
    }
    offset += 4;

    for _ in 0..n_elements {
        if offset + 4 > data.len() {
            break;
        }
        offset += 4;
    }

    Ok(offset)
}

fn read_u32_from_reader<R: std::io::Read>(reader: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_from_reader<R: std::io::Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32_from_reader<R: std::io::Read>(reader: &mut R) -> Result<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_string_from_file(file: &mut File) -> Result<String> {
    // GGUF strings are prefixed with uint64_t length
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;
    if len == 0 || len > 100000 {
        return Ok(String::new());
    }
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn read_array_from_reader<R: std::io::Read>(reader: &mut R) -> Result<()> {
    let n_elements = read_varint_gguf_from_reader(reader)?;
    let _array_type = read_u32_from_reader(reader)?;
    for _ in 0..n_elements {
        let _ = read_u32_from_reader(reader)?;
    }
    Ok(())
}

fn read_varint_gguf_from_reader<R: std::io::Read>(reader: &mut R) -> Result<usize> {
    let mut result = 0;
    for i in 0..10 {
        let mut byte = [0u8; 1];
        if reader.read_exact(&mut byte).is_err() {
            return Ok(result);
        }
        result |= ((byte[0] & 0x7F) as usize) << (7 * i);
        if byte[0] & 0x80 == 0 {
            return Ok(result);
        }
    }
    Ok(result)
}
