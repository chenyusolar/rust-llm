use std::cmp::min;

#[derive(Clone)]
pub struct FlashAttentionConfig {
    pub head_dim: usize,
    pub num_heads: usize,
    pub softmax_scale: f32,
    pub is_causal: bool,
}

impl FlashAttentionConfig {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        Self {
            head_dim,
            num_heads,
            softmax_scale,
            is_causal: true,
        }
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

    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_heads;
        let softmax_scale = self.config.softmax_scale;

        let mut output = vec![0.0f32; seq_len * head_dim * num_heads];

        for h in 0..num_heads {
            let q_offset = h * seq_len * head_dim;
            let k_offset = h * seq_len * head_dim;
            let v_offset = h * seq_len * head_dim;
            let out_offset = h * seq_len * head_dim;

            let mut m = vec![f32::NEG_INFINITY; seq_len];
            let mut l = vec![0.0f32; seq_len];
            let mut o = vec![0.0f32; seq_len * head_dim];

            for j in 0..seq_len {
                let q_j = &q[q_offset + j * head_dim..q_offset + (j + 1) * head_dim];

                let mut row_max = f32::NEG_INFINITY;
                let mut row_sum = 0.0f32;
                let mut new_o = vec![0.0f32; head_dim];

                for i in 0..j + 1 {
                    if self.config.is_causal && i > j {
                        break;
                    }

                    let k_i = &k[k_offset + i * head_dim..k_offset + (i + 1) * head_dim];

                    let mut s_ij = 0.0f32;
                    for d in 0..head_dim {
                        s_ij += q_j[d] * k_i[d];
                    }
                    s_ij *= softmax_scale;

                    if s_ij > row_max {
                        row_max = s_ij;
                    }
                }

                let mut exp_sum = 0.0f32;
                for i in 0..j + 1 {
                    if self.config.is_causal && i > j {
                        break;
                    }

                    let k_i = &k[k_offset + i * head_dim..k_offset + (i + 1) * head_dim];

                    let mut s_ij = 0.0f32;
                    for d in 0..head_dim {
                        s_ij += q_j[d] * k_i[d];
                    }
                    s_ij *= softmax_scale;

                    let exp_s = (s_ij - row_max).exp();
                    exp_sum += exp_s;

                    let v_i = &v[v_offset + i * head_dim..v_offset + (i + 1) * head_dim];
                    for d in 0..head_dim {
                        new_o[d] += exp_s * v_i[d];
                    }
                }

                m[j] = row_max;
                l[j] = l.get(j - 1).unwrap_or(&0.0) * (m[j - 1] - row_max).exp() + exp_sum;

                let scale = (m[j - 1] - row_max).exp();
                for d in 0..head_dim {
                    o[j * head_dim + d] =
                        o.get((j - 1) * head_dim + d).copied().unwrap_or(0.0) * scale;
                }

                for d in 0..head_dim {
                    o[j * head_dim + d] += new_o[d];
                }

                let inv_l = 1.0 / l[j];
                for d in 0..head_dim {
                    o[j * head_dim + d] *= inv_l;
                }
            }

            for j in 0..seq_len {
                for d in 0..head_dim {
                    output[out_offset + j * head_dim + d] = o[j * head_dim + d];
                }
            }
        }

        output
    }

    pub fn forward_tiled(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_heads;
        let softmax_scale = self.config.softmax_scale;
        let block_size = self.block_size;

        let mut output = vec![0.0f32; seq_len * head_dim * num_heads];

        for h in 0..num_heads {
            let q_offset = h * seq_len * head_dim;
            let k_offset = h * seq_len * head_dim;
            let v_offset = h * seq_len * head_dim;
            let out_offset = h * seq_len * head_dim;

            let mut o = vec![0.0f32; seq_len * head_dim];
            let mut l = vec![0.0f32; seq_len];
            let mut m = vec![f32::NEG_INFINITY; seq_len];

            let num_blocks = (seq_len + block_size - 1) / block_size;

            for j_block in 0..num_blocks {
                let j_start = j_block * block_size;
                let j_end = min(j_start + block_size, seq_len);

                for j in j_start..j_end {
                    let q_j = &q[q_offset + j * head_dim..q_offset + (j + 1) * head_dim];

                    let mut row_max = f32::NEG_INFINITY;
                    let mut row_sum = 0.0f32;
                    let mut new_o = vec![0.0f32; head_dim];

                    let k_range = if self.config.is_causal {
                        j + 1
                    } else {
                        seq_len
                    };
                    let k_blocks = (k_range + block_size - 1) / block_size;

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

                            if s_ij > row_max {
                                row_max = s_ij;
                            }
                        }
                    }

                    let mut exp_sum = 0.0f32;

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

                            let exp_s = (s_ij - row_max).exp();
                            exp_sum += exp_s;

                            for d in 0..head_dim {
                                new_o[d] += exp_s * v_i[d];
                            }
                        }
                    }

                    if j > 0 {
                        let scale = (m[j - 1] - row_max).exp();
                        for d in 0..head_dim {
                            o[j * head_dim + d] = o[(j - 1) * head_dim + d] * scale;
                        }
                        l[j] = l[j - 1] * scale;
                    }

                    for d in 0..head_dim {
                        o[j * head_dim + d] += new_o[d];
                    }

                    m[j] = row_max;
                    l[j] += exp_sum;

                    let inv_l = 1.0 / l[j];
                    for d in 0..head_dim {
                        o[j * head_dim + d] *= inv_l;
                    }
                }
            }

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

    fn sdpa_forward(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let head_dim = 128;
        let num_heads = 32;
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = vec![0.0f32; seq_len * head_dim * num_heads];

        for h in 0..num_heads {
            let q_off = h * seq_len * head_dim;
            let k_off = h * seq_len * head_dim;
            let v_off = h * seq_len * head_dim;
            let o_off = h * seq_len * head_dim;

            for i in 0..seq_len {
                let q_i = &q[q_off + i * head_dim..q_off + (i + 1) * head_dim];

                let mut scores = vec![0.0f32; seq_len];
                let mut max_score = f32::NEG_INFINITY;

                for j in 0..=i {
                    let k_j = &k[k_off + j * head_dim..k_off + (j + 1) * head_dim];

                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_i[d] * k_j[d];
                    }
                    dot *= softmax_scale;

                    scores[j] = dot;
                    if dot > max_score {
                        max_score = dot;
                    }
                }

                let mut exp_sum = 0.0f32;
                for j in 0..=i {
                    scores[j] = (scores[j] - max_score).exp();
                    exp_sum += scores[j];
                }

                let inv_sum = 1.0 / exp_sum;

                for j in 0..=i {
                    let v_j = &v[v_off + j * head_dim..v_off + (j + 1) * head_dim];

                    for d in 0..head_dim {
                        output[o_off + i * head_dim + d] += scores[j] * inv_sum * v_j[d];
                    }
                }
            }
        }

        output
    }
}
