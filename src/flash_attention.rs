use std::cmp::min;

#[derive(Clone)]
pub struct FlashAttentionConfig {
    pub head_dim: usize,
    pub num_heads: usize,
    pub softmax_scale: f32,
    pub is_causal: bool,
    pub num_kv_heads: usize,
}

impl FlashAttentionConfig {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        Self {
            head_dim,
            num_heads,
            softmax_scale,
            is_causal: true,
            num_kv_heads: num_heads,
        }
    }

    pub fn with_kv_heads(mut self, num_kv_heads: usize) -> Self {
        self.num_kv_heads = num_kv_heads;
        self
    }

    pub fn with_causal(mut self, is_causal: bool) -> Self {
        self.is_causal = is_causal;
        self
    }
}

pub struct FlashAttention {
    config: FlashAttentionConfig,
    block_size: usize,
}

impl FlashAttention {
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self {
            config,
            block_size: 64,
        }
    }

    /// Flash Attention 前向传播 (online softmax 算法)
    /// 使用 KV cache 优化：每次只处理一个新 token
    pub fn forward_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
    ) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let softmax_scale = self.config.softmax_scale;
        let is_causal = self.config.is_causal;

        // q: [num_heads, head_dim], k_cache/v_cache: [num_kv_heads, seq_len, head_dim]
        let num_heads = self.config.num_heads;
        let kv_per_head = num_heads / num_kv_heads.max(1);

        let mut output = vec![0.0f32; num_heads * head_dim];

        for h in 0..num_heads {
            let kv_head = h / kv_per_head;
            let q_offset = h * head_dim;
            let k_offset = kv_head * seq_len * head_dim;
            let v_offset = kv_head * seq_len * head_dim;

            let q_vec = &q[q_offset..q_offset + head_dim];

            // 计算 attention scores
            let mut scores = vec![0.0f32; seq_len];
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..seq_len {
                if is_causal && j >= seq_len - 1 {
                    break; // 只有最后一个位置是新 token，不需要之前的
                }
                let k_j = &k_cache[k_offset + j * head_dim..k_offset + (j + 1) * head_dim];

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_vec[d] * k_j[d];
                }
                dot *= softmax_scale;
                scores[j] = dot;
                max_score = max_score.max(dot);
            }

            // 当前 token 自己的 attention (如果是自回归)
            let self_score = scores.iter().fold(f32::MIN, |a, &b| a.max(b));

            // Online softmax
            let mut exp_sum = 0.0f32;
            for s in scores.iter() {
                exp_sum += (s - max_score).exp();
            }
            exp_sum += (self_score - max_score).exp();

            // 计算输出
            let mut o_vec = vec![0.0f32; head_dim];

            for j in 0..seq_len {
                if is_causal && j >= seq_len - 1 {
                    break;
                }
                let k_j = &k_cache[k_offset + j * head_dim..k_offset + (j + 1) * head_dim];
                let v_j = &v_cache[v_offset + j * head_dim..v_offset + (j + 1) * head_dim];

                let weight = ((scores[j] - max_score).exp()) / exp_sum.max(1e-8);
                for d in 0..head_dim {
                    o_vec[d] += weight * v_j[d];
                }
            }

            // 加上自己位置的 attention
            let self_weight = ((self_score - max_score).exp()) / exp_sum.max(1e-8);
            let v_self = &v_cache[k_offset + (seq_len - 1) * head_dim..k_offset + seq_len * head_dim];
            for d in 0..head_dim {
                o_vec[d] += self_weight * v_self[d];
            }

            // 复制到输出
            let out_offset = h * head_dim;
            output[out_offset..out_offset + head_dim].copy_from_slice(&o_vec);
        }

        output
    }

    /// 标准 Flash Attention 前向传播 (完整序列)
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        self.forward_tiled_impl(q, k, v, seq_len)
    }

    /// Tiled Flash Attention 实现
    pub fn forward_tiled(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        self.forward_tiled_impl(q, k, v, seq_len)
    }

    fn forward_tiled_impl(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_heads;
        let softmax_scale = self.config.softmax_scale;
        let is_causal = self.config.is_causal;
        let num_kv_heads = self.config.num_kv_heads;
        let kv_per_head = (num_heads / num_kv_heads.max(1)).max(1);
        let block_size = self.block_size;

        let mut output = vec![0.0f32; seq_len * head_dim * num_heads];

        for h in 0..num_heads {
            let kv_head = h / kv_per_head;
            let q_offset = h * seq_len * head_dim;
            let k_offset = kv_head * seq_len * head_dim;
            let v_offset = kv_head * seq_len * head_dim;
            let out_offset = h * seq_len * head_dim;

            // Flash Attention online softmax
            let mut m = vec![f32::NEG_INFINITY; seq_len];  // row max
            let mut l = vec![0.0f32; seq_len];            // row sum of exp
            let mut o = vec![0.0f32; seq_len * head_dim]; // output

            // 第一趟：计算 row max 和 exp sum
            for j in 0..seq_len {
                let q_j = &q[q_offset + j * head_dim..q_offset + (j + 1) * head_dim];

                let k_range = if is_causal { j + 1 } else { seq_len };
                let k_blocks = (k_range + block_size - 1) / block_size;

                let mut row_max = f32::NEG_INFINITY;
                for kb in 0..k_blocks {
                    let i_start = kb * block_size;
                    let i_end = min(i_start + block_size, k_range);

                    for i in i_start..i_end {
                        let k_i = &k[k_offset + i * head_dim..k_offset + (i + 1) * head_dim];
                        let mut s_ij = 0.0f32;
                        for d in 0..head_dim {
                            s_ij += q_j[d] * k_i[d];
                        }
                        s_ij *= softmax_scale;
                        row_max = row_max.max(s_ij);
                    }
                }
                m[j] = row_max;
            }

            // 第二趟：计算 attention
            for j in 0..seq_len {
                let q_j = &q[q_offset + j * head_dim..q_offset + (j + 1) * head_dim];

                let k_range = if is_causal { j + 1 } else { seq_len };
                let k_blocks = (k_range + block_size - 1) / block_size;

                let prev_m = m.get(j.wrapping_sub(1)).copied().unwrap_or(0.0);
                let scale = (prev_m - m[j]).exp();

                // rescale previous output
                if j > 0 {
                    for d in 0..head_dim {
                        o[j * head_dim + d] = o[(j - 1) * head_dim + d] * scale;
                    }
                    l[j] = l[j - 1] * scale;
                }

                let mut row_sum = 0.0f32;
                for kb in 0..k_blocks {
                    let i_start = kb * block_size;
                    let i_end = min(i_start + block_size, k_range);

                    for i in i_start..i_end {
                        let k_i = &k[k_offset + i * head_dim..k_offset + (i + 1) * head_dim];
                        let v_i = &v[v_offset + i * head_dim..v_offset + (i + 1) * head_dim];

                        let mut s_ij = 0.0f32;
                        for d in 0..head_dim {
                            s_ij += q_j[d] * k_i[d];
                        }
                        s_ij *= softmax_scale;

                        let exp_s = (s_ij - m[j]).exp();
                        row_sum += exp_s;

                        for d in 0..head_dim {
                            o[j * head_dim + d] += exp_s * v_i[d];
                        }
                    }
                }
                l[j] += row_sum;

                // normalize
                let inv_l = 1.0 / l[j].max(1e-8);
                for d in 0..head_dim {
                    o[j * head_dim + d] *= inv_l;
                }
            }

            // 复制到输出
            for j in 0..seq_len {
                for d in 0..head_dim {
                    output[out_offset + j * head_dim + d] = o[j * head_dim + d];
                }
            }
        }

        output
    }
}

pub struct AttentionKernel {
    use_flash_attention: bool,
    flash: Option<FlashAttention>,
    head_dim: usize,
    num_heads: usize,
}

impl AttentionKernel {
    pub fn new(num_heads: usize, head_dim: usize, use_flash: bool) -> Self {
        let flash = if use_flash {
            Some(FlashAttention::new(FlashAttentionConfig::new(
                num_heads, head_dim,
            )))
        } else {
            None
        };

        Self {
            use_flash_attention: use_flash,
            flash,
            head_dim,
            num_heads,
        }
    }

    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        if self.use_flash_attention {
            if let Some(ref flash) = self.flash {
                return flash.forward_tiled(q, k, v, seq_len);
            }
        }

        self.sdpa_forward(q, k, v, seq_len)
    }

    /// 标准 SDPA (可能使用 Flash Attention 后端)
    pub fn sdpa_forward(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = vec![0.0f32; seq_len * head_dim * num_heads];

        for h in 0..num_heads {
            let q_off = h * seq_len * head_dim;
            let k_off = h * seq_len * head_dim;
            let v_off = h * seq_len * head_dim;
            let o_off = h * seq_len * head_dim;

            // 计算 query 和 key 的点积
            for i in 0..seq_len {
                let q_i = &q[q_off + i * head_dim..q_off + (i + 1) * head_dim];

                // 找 max 用于 softmax 稳定性
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq_len];

                for j in 0..=i {
                    let k_j = &k[k_off + j * head_dim..k_off + (j + 1) * head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_i[d] * k_j[d];
                    }
                    dot *= softmax_scale;
                    scores[j] = dot;
                    max_score = max_score.max(dot);
                }

                // softmax
                let mut exp_sum = 0.0f32;
                for j in 0..=i {
                    scores[j] = (scores[j] - max_score).exp();
                    exp_sum += scores[j];
                }

                let inv_sum = 1.0 / exp_sum.max(1e-8);

                // 加权求和
                for j in 0..=i {
                    let v_j = &v[v_off + j * head_dim..v_off + (j + 1) * head_dim];
                    let w = scores[j] * inv_sum;
                    for d in 0..head_dim {
                        output[o_off + i * head_dim + d] += w * v_j[d];
                    }
                }
            }
        }

        output
    }
}
