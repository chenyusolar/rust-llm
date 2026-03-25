use crate::types::*;

pub struct FIMConfig {
    pub enabled: bool,
    pub prefix_token_id: Token,
    pub suffix_token_id: Token,
    pub middle_token_id: Token,
}

impl Default for FIMConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prefix_token_id: 0,
            suffix_token_id: 0,
            middle_token_id: 0,
        }
    }
}

impl FIMConfig {
    pub fn new(prefix: Token, suffix: Token, middle: Token) -> Self {
        Self {
            enabled: true,
            prefix_token_id: prefix,
            suffix_token_id: suffix,
            middle_token_id: middle,
        }
    }
}

pub struct FIMEncoder {
    config: FIMConfig,
}

impl FIMEncoder {
    pub fn new(config: FIMConfig) -> Self {
        Self { config }
    }

    pub fn encode(&self, prefix: &str, suffix: &str, middle: Option<&str>) -> Vec<Token> {
        if !self.config.enabled {
            return vec![];
        }

        let mut tokens = vec![self.config.prefix_token_id];

        tokens.extend(self.tokenize(prefix));
        tokens.push(self.config.suffix_token_id);

        if let Some(m) = middle {
            tokens.push(self.config.middle_token_id);
            tokens.extend(self.tokenize(m));
        }

        tokens
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        text.chars().map(|c| c as Token).collect()
    }

    pub fn create_mask(&self, seq_len: usize, prefix_len: usize, suffix_len: usize) -> Vec<bool> {
        let mut mask = vec![true; seq_len];

        let middle_start = prefix_len + 1;
        let middle_end = middle_start + suffix_len;

        for i in 0..seq_len {
            if i < prefix_len {
                mask[i] = false;
            } else if i >= middle_start && i < middle_end {
                mask[i] = false;
            }
        }

        mask
    }

    pub fn apply_fim_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        prefix_len: usize,
        suffix_len: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; q.len()];
        let head_dim = q.len() / seq_len;

        let mask = self.create_mask(seq_len, prefix_len, suffix_len);

        for i in 0..seq_len {
            for j in 0..seq_len {
                if !mask[j] && j > i {
                    continue;
                }

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                dot /= (head_dim as f32).sqrt();

                let exp_dot = dot.exp();

                for d in 0..head_dim {
                    output[i * head_dim + d] += exp_dot * v[j * head_dim + d];
                }
            }
        }

        output
    }
}

pub struct FIMDecoder {
    encoder: FIMEncoder,
}

impl FIMDecoder {
    pub fn new(config: FIMConfig) -> Self {
        Self {
            encoder: FIMEncoder::new(config),
        }
    }

    pub fn generate(&self, prefix: &str, suffix: &str, max_tokens: usize) -> Vec<Token> {
        let mut tokens = self.encoder.encode(prefix, suffix, None);

        tokens.truncate(prefix.len() + suffix.len() + 2);

        tokens
    }
}

pub struct SlidingWindowAttention {
    window_size: usize,
    look_forward: usize,
}

impl SlidingWindowAttention {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            look_forward: 0,
        }
    }

    pub fn with_look_forward(mut self, look_forward: usize) -> Self {
        self.look_forward = look_forward;
        self
    }

    pub fn apply(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * head_dim];

        for i in 0..seq_len {
            let start = if i > self.window_size {
                i - self.window_size
            } else {
                0
            };
            let end = (i + self.look_forward + 1).min(seq_len);

            let mut max_score = f32::MIN;
            for j in start..end {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                dot /= (head_dim as f32).sqrt();
                max_score = max_score.max(dot);
            }

            let mut exp_sum = 0.0f32;
            for j in start..end {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                dot /= (head_dim as f32).sqrt();
                let exp_dot = (dot - max_score).exp();
                exp_sum += exp_dot;

                for d in 0..head_dim {
                    output[i * head_dim + d] += exp_dot * v[j * head_dim + d];
                }
            }

            if exp_sum > 0.0 {
                for d in 0..head_dim {
                    output[i * head_dim + d] /= exp_sum;
                }
            }
        }

        output
    }
}

pub struct AttentionSink {
    sink_tokens: usize,
}

impl AttentionSink {
    pub fn new(sink_tokens: usize) -> Self {
        Self { sink_tokens }
    }

    pub fn apply_with_sink(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * head_dim];

        for i in 0..seq_len {
            let mut max_score = f32::MIN;

            for j in 0..=i.min(self.sink_tokens) {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                max_score = max_score.max(dot);
            }

            for j in (i.saturating_sub(self.sink_tokens))..=i {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }

                let exp_dot = (dot - max_score).exp();

                for d in 0..head_dim {
                    output[i * head_dim + d] += exp_dot * v[j * head_dim + d];
                }
            }
        }

        output
    }
}
