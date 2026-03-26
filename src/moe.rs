use crate::quantize::QuantizedTensor;
use std::collections::HashMap;

pub struct MoELayer {
    pub num_experts: usize,
    pub expert_used: usize,
    pub gate_weight: QuantizedTensor,
    pub ffn_weights: Vec<FFNExpert>,
    pub router: MoERouter,
}

pub struct MoERouter {
    pub top_k: usize,
    pub score_softmax_temp: f32,
}

impl MoERouter {
    pub fn new(top_k: usize, temperature: f32) -> Self {
        Self {
            top_k,
            score_softmax_temp: temperature,
        }
    }

    /// 路由输入到 top-k experts
    pub fn route(&self, x: &[f32], gate_weights: &[f32], num_experts: usize) -> Vec<(usize, f32)> {
        // 计算每个 expert 的分数
        let scores = self.matmul_vec(x, gate_weights, x.len(), num_experts);

        // 应用 softmax
        let softmax_scores = self.softmax(&scores);

        // 选择 top-k
        let mut expert_scores: Vec<(usize, f32)> = softmax_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        expert_scores.truncate(self.top_k);
        expert_scores
    }

    fn matmul_vec(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; out_dim];

        for i in 0..out_dim {
            let mut sum = 0.0f32;
            for j in 0..in_dim {
                let idx = i * in_dim + j;
                if idx < weight.len() {
                    sum += input[j] * weight[idx];
                }
            }
            output[i] = sum;
        }

        output
    }

    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().fold(f32::MIN, |a, &b| a.max(b));
        let exp_sum: f32 = x.iter().map(|&v| (v - max_val).exp()).sum();

        x.iter().map(|&v| (v - max_val).exp() / exp_sum.max(1e-8)).collect()
    }
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
            router: MoERouter::new(expert_used, 1.0),
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

    /// MoE 前向传播
    /// # Arguments
    /// * `x` - 输入 hidden states [hidden_size]
    /// * `gate_weights` - 路由器的门权重 [num_experts, hidden_size]
    /// * `top_k` - 使用的 expert 数量
    pub fn forward(&self, x: &[f32], gate_weights: &[f32], top_k: usize) -> Vec<f32> {
        let hidden_size = x.len();
        let num_experts = self.ffn_weights.len();

        if num_experts == 0 {
            return x.to_vec();
        }

        // 1. 路由到 top-k experts
        let expert_weights = self.router.route(x, gate_weights, num_experts);

        if expert_weights.is_empty() {
            return x.to_vec();
        }

        // 2. 对每个选中的 expert 执行 FFN
        let mut output = vec![0.0f32; hidden_size];

        for &(expert_idx, weight) in &expert_weights {
            if expert_idx < self.ffn_weights.len() {
                let expert = &self.ffn_weights[expert_idx];

                // Gate projection
                let gate_out = self.matmul_fc(
                    x,
                    &expert.gate_proj.dequantize(),
                    hidden_size,
                    expert.gate_proj.shape.first().copied().unwrap_or(0),
                );

                // Up projection
                let up_out = self.matmul_fc(
                    x,
                    &expert.up_proj.dequantize(),
                    hidden_size,
                    expert.up_proj.shape.first().copied().unwrap_or(0),
                );

                // SwiGLU activation
                let silu_gate: Vec<f32> = gate_out
                    .iter()
                    .zip(up_out.iter())
                    .map(|(&g, &u)| silu(g) * u)
                    .collect();

                // Down projection
                let down_out = self.matmul_fc(
                    &silu_gate,
                    &expert.down_proj.dequantize(),
                    silu_gate.len(),
                    hidden_size,
                );

                // 加权累加
                for (i, val) in down_out.iter().enumerate() {
                    output[i] += val * weight;
                }
            }
        }

        output
    }

    /// 全连接层: input @ weight
    fn matmul_fc(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; out_dim];

        for i in 0..out_dim {
            let mut sum = 0.0f32;
            for j in 0..in_dim.min(weight.len() / out_dim.max(1)) {
                let weight_idx = i * in_dim + j;
                if weight_idx < weight.len() {
                    sum += input[j] * weight[weight_idx];
                }
            }
            output[i] = sum;
        }

        output
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
        tracing::info!("Distributed {} layers across {} devices", num_layers, self.num_devices);
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
        // 实际的多 GPU 同步需要使用 CUDA/NCCL 或其他 GPU 通信库
    }

    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    pub fn current_device(&self) -> usize {
        self.current_device
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

    pub fn split_tensor<T: Clone + Send + Sync>(&self, tensor: &[T], dim: usize) -> Vec<Vec<T>> {
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

    /// AllReduce 操作 - 所有设备求和后平均
    pub fn all_reduce(&self, data: &mut [f32]) {
        let sum: f32 = data.iter().sum();
        let avg = sum / self.num_devices as f32;

        for val in data.iter_mut() {
            *val = avg;
        }
        tracing::trace!("TensorParallelism all_reduce completed on rank {}", self.rank);
    }

    /// AllGather 操作 - 收集所有设备的数据
    pub fn all_gather(&self, data: &[f32]) -> Vec<f32> {
        let chunk_size = data.len();
        let mut result = vec![0.0f32; data.len() * self.num_devices];

        // 简化实现：假设每个 rank 的数据相同
        for i in 0..self.num_devices {
            let offset = i * chunk_size;
            result[offset..offset + chunk_size].copy_from_slice(data);
        }

        result
    }

    /// ReduceScatter - 先求和再分发到各设备
    pub fn reduce_scatter(&self, data: &[f32]) -> Vec<f32> {
        let chunk_size = data.len() / self.num_devices;
        let start = self.rank * chunk_size;
        let end = if self.rank == self.num_devices - 1 {
            data.len()
        } else {
            (self.rank + 1) * chunk_size
        };

        let mut result = vec![0.0f32; end - start];

        // 简化实现：每个 rank 获取对应分片
        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            if i == self.rank {
                for (j, &val) in chunk.iter().enumerate() {
                    result[j] = val;
                }
            }
        }

        result
    }

    pub fn rank(&self) -> usize {
        self.rank
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

    pub fn prev_stage(&self) -> usize {
        if self.stage_id == 0 {
            self.num_stages - 1
        } else {
            self.stage_id - 1
        }
    }

    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    pub fn stage_id(&self) -> usize {
        self.stage_id
    }
}
