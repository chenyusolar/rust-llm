use crate::quantize::QuantizedTensor;
use std::collections::HashMap;

pub struct MoELayer {
    pub num_experts: usize,
    pub expert_used: usize,
    pub gate_weight: QuantizedTensor,
    pub ffn_weights: Vec<FFNExpert>,
}

pub struct FFNExpert {
    pub gate_proj: QuantizedTensor,
    pub up_proj: QuantizedTensor,
    pub down_proj: QuantizedTensor,
}

impl MoELayer {
    pub fn new(num_experts: usize, expert_used: usize) -> Self {
        Self {
            num_experts,
            expert_used,
            gate_weight: QuantizedTensor::new(vec![], crate::types::DataType::F16),
            ffn_weights: Vec::new(),
        }
    }

    pub fn load_expert(
        &mut self,
        expert_idx: usize,
        gate: QuantizedTensor,
        up: QuantizedTensor,
        down: QuantizedTensor,
    ) {
        if expert_idx < self.num_experts {
            self.ffn_weights.push(FFNExpert {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
            });
        }
    }

    pub fn forward(&self, x: &[f32], hidden_size: usize, top_k: usize) -> Vec<f32> {
        let num_experts = self.ffn_weights.len();
        if num_experts == 0 {
            return x.to_vec();
        }

        let mut expert_outputs: Vec<Vec<f32>> = Vec::with_capacity(top_k);
        let mut expert_indices: Vec<usize> = Vec::with_capacity(top_k);

        for _ in 0..top_k {
            expert_outputs.push(vec![0.0f32; hidden_size]);
        }

        for (i, expert) in self.ffn_weights.iter().enumerate() {
            let gate_out = self.matmul(
                x,
                &expert.gate_proj.dequantize(),
                hidden_size,
                self.num_experts,
            );
            let top_idx = self.argmax(&gate_out);

            if !expert_indices.contains(&top_idx) {
                if expert_indices.len() < top_k {
                    expert_indices.push(top_idx);

                    let up_out =
                        self.matmul(x, &expert.up_proj.dequantize(), hidden_size, hidden_size);
                    let intermediate: Vec<f32> = up_out.iter().map(|v| silu(*v)).collect();
                    let down_out = self.matmul(
                        &intermediate,
                        &expert.down_proj.dequantize(),
                        hidden_size,
                        hidden_size,
                    );

                    let idx = expert_indices.len() - 1;
                    expert_outputs[idx] = down_out;
                }
            }
        }

        let mut output = vec![0.0f32; hidden_size];
        for expert_out in &expert_outputs {
            for (i, val) in expert_out.iter().enumerate() {
                output[i] += val / top_k as f32;
            }
        }

        output
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize) -> Vec<f32> {
        let k = a.len() / m;
        let mut c = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        c
    }

    fn argmax(&self, v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub struct MultiGPUManager {
    num_devices: usize,
    current_device: usize,
    layer_distribution: HashMap<usize, usize>,
}

impl MultiGPUManager {
    pub fn new(num_devices: usize) -> Self {
        Self {
            num_devices,
            current_device: 0,
            layer_distribution: HashMap::new(),
        }
    }

    pub fn distribute_layers(&mut self, num_layers: usize) {
        for layer_idx in 0..num_layers {
            let device = layer_idx % self.num_devices;
            self.layer_distribution.insert(layer_idx, device);
        }
    }

    pub fn get_device(&self, layer_idx: usize) -> usize {
        *self.layer_distribution.get(&layer_idx).unwrap_or(&0)
    }

    pub fn next_device(&mut self) -> usize {
        self.current_device = (self.current_device + 1) % self.num_devices;
        self.current_device
    }

    pub fn sync_all(&self) {
        tracing::debug!("Syncing {} devices", self.num_devices);
    }
}

pub struct TensorParallelism {
    num_devices: usize,
    rank: usize,
}

impl TensorParallelism {
    pub fn new(num_devices: usize, rank: usize) -> Self {
        Self { num_devices, rank }
    }

    pub fn split_tensor<T: Clone>(&self, tensor: &[T], dim: usize) -> Vec<Vec<T>> {
        let chunk_size = tensor.len() / self.num_devices;
        let mut chunks = Vec::with_capacity(self.num_devices);

        for i in 0..self.num_devices {
            let start = i * chunk_size;
            let end = if i == self.num_devices - 1 {
                tensor.len()
            } else {
                (i + 1) * chunk_size
            };
            chunks.push(tensor[start..end].to_vec());
        }

        chunks
    }

    pub fn all_reduce(&self, data: &mut [f32]) {
        let sum: f32 = data.iter().sum();
        let avg = sum / self.num_devices as f32;

        for val in data.iter_mut() {
            *val = avg;
        }
    }

    pub fn reduce_scatter(&self, data: &[f32]) -> Vec<f32> {
        let chunk_size = data.len() / self.num_devices;
        let start = self.rank * chunk_size;
        let end = if self.rank == self.num_devices - 1 {
            data.len()
        } else {
            (self.rank + 1) * chunk_size
        };

        let mut result = vec![0.0f32; end - start];

        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            let sum: f32 = chunk.iter().sum();
            if i == self.rank {
                for (j, &val) in chunk.iter().enumerate() {
                    result[j] = val;
                }
            }
        }

        result
    }
}

pub struct PipelineParallelism {
    num_stages: usize,
    stage_id: usize,
}

impl PipelineParallelism {
    pub fn new(num_stages: usize, stage_id: usize) -> Self {
        Self {
            num_stages,
            stage_id,
        }
    }

    pub fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.stage_id == self.num_stages - 1
    }

    pub fn next_stage(&self) -> usize {
        (self.stage_id + 1) % self.num_stages
    }
}
