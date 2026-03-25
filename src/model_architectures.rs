use crate::types::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    LLaMA,
    Mistral,
    Mixtral,
    Phi3,
    Qwen,
    Gemma,
    Falcon,
}

impl ModelArchitecture {
    pub fn from_string(s: &str) -> Self {
        let s_lower = s.to_lowercase();
        if s_lower.contains("mistral") {
            ModelArchitecture::Mistral
        } else if s_lower.contains("mixtral") {
            ModelArchitecture::Mixtral
        } else if s_lower.contains("phi") {
            ModelArchitecture::Phi3
        } else if s_lower.contains("qwen") {
            ModelArchitecture::Qwen
        } else if s_lower.contains("gemma") {
            ModelArchitecture::Gemma
        } else if s_lower.contains("falcon") {
            ModelArchitecture::Falcon
        } else {
            ModelArchitecture::LLaMA
        }
    }

    pub fn supports_sliding_window(&self) -> bool {
        matches!(
            self,
            ModelArchitecture::Mistral | ModelArchitecture::Mixtral
        )
    }

    pub fn supports_rope_scaling(&self) -> bool {
        matches!(
            self,
            ModelArchitecture::LLaMA | ModelArchitecture::Qwen | ModelArchitecture::Gemma
        )
    }

    pub fn supports_gated_act(&self) -> bool {
        matches!(self, ModelArchitecture::Gemma)
    }
}

pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub sliding_window: Option<usize>,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub use_gated_act: bool,
    pub expert_count: Option<usize>,
    pub expert_used_count: Option<usize>,
}

pub struct RopeScaling {
    pub factor: f32,
    pub original_max_seq_len: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::LLaMA,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            vocab_size: 32000,
            max_seq_len: 2048,
            sliding_window: None,
            rope_theta: 10000.0,
            rope_scaling: None,
            use_gated_act: false,
            expert_count: None,
            expert_used_count: None,
        }
    }
}

impl ModelConfig {
    pub fn mistral_7b() -> Self {
        Self {
            architecture: ModelArchitecture::Mistral,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            max_seq_len: 8192,
            sliding_window: Some(4096),
            rope_theta: 10000.0,
            rope_scaling: None,
            use_gated_act: false,
            expert_count: None,
            expert_used_count: None,
        }
    }

    pub fn mixtral_8x7b() -> Self {
        Self {
            architecture: ModelArchitecture::Mixtral,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            max_seq_len: 32768,
            sliding_window: Some(4096),
            rope_theta: 1000000.0,
            rope_scaling: None,
            use_gated_act: false,
            expert_count: Some(8),
            expert_used_count: Some(2),
        }
    }

    pub fn phi3_4b() -> Self {
        Self {
            architecture: ModelArchitecture::Phi3,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 2,
            head_dim: 96,
            vocab_size: 32064,
            max_seq_len: 4096,
            sliding_window: None,
            rope_theta: 10000.0,
            rope_scaling: Some(RopeScaling {
                factor: 1.2,
                original_max_seq_len: 4096,
            }),
            use_gated_act: false,
            expert_count: None,
            expert_used_count: None,
        }
    }

    pub fn qwen2_4b() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_layers: 28,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 151936,
            max_seq_len: 32768,
            sliding_window: None,
            rope_theta: 1000000.0,
            rope_scaling: Some(RopeScaling {
                factor: 1.0,
                original_max_seq_len: 32768,
            }),
            use_gated_act: false,
            expert_count: None,
            expert_used_count: None,
        }
    }

    pub fn gemma_2b() -> Self {
        Self {
            architecture: ModelArchitecture::Gemma,
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 128,
            intermediate_size: 16384,
            num_layers: 18,
            vocab_size: 256000,
            max_seq_len: 8192,
            sliding_window: None,
            rope_theta: 10000.0,
            rope_scaling: None,
            use_gated_act: true,
            expert_count: None,
            expert_used_count: None,
        }
    }

    pub fn from_metadata(metadata: &HashMap<String, String>) -> Self {
        let mut config = Self::default();

        for (key, _) in metadata {
            let key_lower = key.to_lowercase();

            if key_lower.contains("hidden_size") || key_lower.contains("hsize") {
                if let Some(val) = extract_num(key) {
                    config.hidden_size = val;
                }
            } else if key_lower.contains("intermediate_size")
                || key_lower.contains("ffn_hidden_size")
            {
                if let Some(val) = extract_num(key) {
                    config.intermediate_size = val;
                }
            } else if key_lower.contains("num_layers")
                || key_lower.contains("n_layers")
                || key_lower.contains("num_hidden_layers")
            {
                if let Some(val) = extract_num(key) {
                    config.num_layers = val;
                }
            } else if key_lower.contains("num_attention_heads") || key_lower.contains("n_heads") {
                if let Some(val) = extract_num(key) {
                    config.num_heads = val;
                }
            } else if key_lower.contains("num_key_value_heads") || key_lower.contains("n_kv_heads")
            {
                if let Some(val) = extract_num(key) {
                    config.num_kv_heads = val;
                }
            } else if key_lower.contains("vocab_size") {
                if let Some(val) = extract_num(key) {
                    config.vocab_size = val;
                }
            } else if key_lower.contains("max_position") || key_lower.contains("n_ctx") {
                if let Some(val) = extract_num(key) {
                    config.max_seq_len = val;
                }
            } else if key_lower.contains("sliding_window") {
                if let Some(val) = extract_num(key) {
                    config.sliding_window = Some(val);
                }
            } else if key_lower.contains("rope_theta") {
                if let Some(val) = extract_num_f(key) {
                    config.rope_theta = val;
                }
            }
        }

        config.head_dim = config.hidden_size / config.num_heads.max(1);
        config.architecture = ModelArchitecture::from_string(
            metadata
                .get("model_type")
                .cloned()
                .unwrap_or_default()
                .as_str(),
        );

        config
    }
}

fn extract_num(s: &str) -> Option<usize> {
    s.chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

fn extract_num_f(s: &str) -> Option<f32> {
    s.chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>()
        .parse()
        .ok()
}

pub struct MistralAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
}

impl MistralAttention {
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
    ) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
        }
    }

    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; q.len()];

        for h in 0..self.num_heads {
            let kv_head = h / (self.num_heads / self.num_kv_heads);
            let q_offset = h * self.head_dim;
            let k_offset = kv_head * self.head_dim;
            let v_offset = kv_head * self.head_dim;

            let window = self.sliding_window.min(seq_len);
            let start = seq_len.saturating_sub(window);

            for i in start..seq_len {
                let mut max_score = f32::MIN;

                for j in start..=i {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q[q_offset + i * self.num_heads * self.head_dim + d]
                            * k[k_offset + j * self.num_kv_heads * self.head_dim + d];
                    }
                    dot /= (self.head_dim as f32).sqrt();
                    max_score = max_score.max(dot);
                }

                let mut exp_sum = 0.0f32;
                for j in start..=i {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q[q_offset + i * self.num_heads * self.head_dim + d]
                            * k[k_offset + j * self.num_kv_heads * self.head_dim + d];
                    }
                    dot /= (self.head_dim as f32).sqrt();
                    let exp_dot = (dot - max_score).exp();
                    exp_sum += exp_dot;

                    for d in 0..self.head_dim {
                        output[q_offset + i * self.num_heads * self.head_dim + d] +=
                            exp_dot * v[v_offset + j * self.num_kv_heads * self.head_dim + d];
                    }
                }

                if exp_sum > 0.0 {
                    for d in 0..self.head_dim {
                        output[q_offset + i * self.num_heads * self.head_dim + d] /= exp_sum;
                    }
                }
            }
        }

        output
    }
}
