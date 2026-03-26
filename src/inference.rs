use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use futures::Stream;

use crate::types::*;
use crate::paged_attention::{DynamicBatcher, InferenceRequest, PagedAttentionCache};
use crate::decoder::Decoder;

/// 流式生成的 token
#[derive(Debug, Clone)]
pub struct StreamToken {
    pub token: Token,
    pub text: String,
    pub is_eos: bool,
    pub full_output: String,
}

pub struct ContinuousBatchingEngine {
    model: Arc<RwLock<crate::Model>>,
    batcher: DynamicBatcher,
    vocab_size: usize,
    max_batch_size: usize,
    max_waiting_ms: u64,
    running_sequences: HashMap<SeqId, SequenceState>,
}

#[derive(Debug, Clone)]
pub struct SequenceState {
    pub seq_id: SeqId,
    pub tokens: Vec<Token>,
    pub position: usize,
    pub finished: bool,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl ContinuousBatchingEngine {
    pub fn new(
        model: Arc<RwLock<crate::Model>>,
        vocab_size: usize,
        max_batch_size: usize,
        max_waiting_ms: u64,
    ) -> Self {
        let batcher = DynamicBatcher::new(max_batch_size, max_waiting_ms);
        
        Self {
            model,
            batcher,
            vocab_size,
            max_batch_size,
            max_waiting_ms,
            running_sequences: HashMap::new(),
        }
    }

    pub async fn add_request(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> SeqId {
        let seq_id = self.running_sequences.len() as SeqId;
        
        let tokens = {
            let model = self.model.read().await;
            model.tokenizer.encode(prompt, None, None)
        };
        
        let request = InferenceRequest {
            seq_id,
            prompt_tokens: tokens.clone(),
            max_tokens,
            temperature,
            top_p,
        };
        
        self.batcher.add_request(request);
        
        let pos = tokens.len();
        self.running_sequences.insert(
            seq_id,
            SequenceState {
                seq_id,
                tokens: tokens,
                position: pos,
                finished: false,
                generated_tokens: 0,
                max_tokens,
                temperature,
                top_p,
            },
        );
        
        seq_id
    }

    pub async fn step(&mut self) -> HashMap<SeqId, Token> {
        let batch = self.batcher.get_batch();
        
        if batch.is_empty() {
            return HashMap::new();
        }
        
        let mut results = HashMap::new();
        
        for request in &batch {
            let state = match self.running_sequences.get_mut(&request.seq_id) {
                Some(s) => s,
                None => continue,
            };
            
            if state.finished || state.generated_tokens >= state.max_tokens {
                continue;
            }
            
            let mut model = self.model.write().await;
            let logits = model.forward(&state.tokens).await.unwrap_or_else(|_| vec![0.0f32; self.vocab_size]);
            
            let token = model.sample_with_params(&logits, state.temperature, state.top_p);
            
            state.tokens.push(token);
            state.generated_tokens += 1;
            state.position += 1;
            
            results.insert(request.seq_id, token);
            
            if token == 0 || state.generated_tokens >= state.max_tokens {
                state.finished = true;
            }
            
            self.batcher.mark_completed(request.seq_id, token);
        }
        
        results
    }

    pub async fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<String, String> {
        let seq_id = self.add_request(prompt, max_tokens, temperature, top_p).await;

        loop {
            let results = self.step().await;

            if let Some(state) = self.running_sequences.get(&seq_id) {
                if state.finished {
                    let model = self.model.read().await;
                    let output = model.decode(&state.tokens);
                    return Ok(output);
                }
            }

            if results.is_empty() && !self.batcher.has_pending() {
                break;
            }

            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        Err("Generation incomplete".to_string())
    }

    /// 流式生成 - 返回每次生成的 token
    pub async fn generate_stream<'a>(
        &'a mut self,
        prompt: &'a str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> impl futures::Stream<Item = Result<StreamToken, String>> + 'a {
        async_stream::stream! {
            let seq_id = self.add_request(prompt, max_tokens, temperature, top_p).await;
            let mut generated_tokens = Vec::new();

            loop {
                let results = self.step().await;

                if let Some(state) = self.running_sequences.get(&seq_id) {
                    if state.finished {
                        let model = self.model.read().await;
                        let output = model.decode(&generated_tokens);
                        yield Ok(StreamToken {
                            token: 0, // EOS
                            text: String::new(),
                            is_eos: true,
                            full_output: output,
                        });
                        break;
                    }

                    // 检查是否有新生成的 token
                    let tokenizer = {
                        let model = self.model.read().await;
                        model.tokenizer.clone()
                    };
                    for &token in &state.tokens[generated_tokens.len()..] {
                        generated_tokens.push(token);
                        let text = tokenizer.decode_token(token)
                            .unwrap_or_default()
                            .to_string();
                        yield Ok(StreamToken {
                            token,
                            text,
                            is_eos: false,
                            full_output: String::new(),
                        });
                    }
                }

                if results.is_empty() && !self.batcher.has_pending() {
                    break;
                }

                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    }

    pub fn num_running(&self) -> usize {
        self.batcher.num_running()
    }

    pub fn has_pending(&self) -> bool {
        self.batcher.has_pending()
    }
}

pub struct SpeculativeDecoder {
    draft_model: Arc<RwLock<crate::Model>>,
    target_model: Arc<RwLock<crate::Model>>,
    max_draft_tokens: usize,
    acceptance_threshold: f32,
}

impl SpeculativeDecoder {
    pub fn new(
        draft_model: Arc<RwLock<crate::Model>>,
        target_model: Arc<RwLock<crate::Model>>,
        max_draft_tokens: usize,
        acceptance_threshold: f32,
    ) -> Self {
        Self {
            draft_model,
            target_model,
            max_draft_tokens,
            acceptance_threshold,
        }
    }

    pub async fn generate(
        &mut self,
        tokens: &[Token],
        max_tokens: usize,
    ) -> Result<Vec<Token>, String> {
        let mut all_tokens = tokens.to_vec();
        let mut generated = Vec::new();

        while generated.len() < max_tokens {
            // 1. Draft model 生成多个候选 token
            let draft_tokens = self.generate_draft_tokens(&all_tokens).await?;

            if draft_tokens.is_empty() {
                break;
            }

            // 2. Target model 验证所有 token
            let mut draft_and_target_tokens = all_tokens.clone();
            draft_and_target_tokens.extend(&draft_tokens);

            let target_logits = {
                let model = self.target_model.read().await;
                model.forward(&draft_and_target_tokens).await.map_err(|e| e.to_string())?
            };

            // 3. 验证并接受 token
            let (accepted_count, bonus_token) = self.verify_and_accept(
                &draft_tokens,
                &target_logits,
                &mut generated,
                max_tokens,
            ).await?;

            // 4. 更新 all_tokens
            all_tokens.extend(&draft_tokens[..accepted_count]);
            if let Some(bonus) = bonus_token {
                generated.push(bonus);
                all_tokens.push(bonus);
            }

            if accepted_count == 0 {
                // 如果没有接受任何 token，从 target 采样一个
                let sample_token = self.sample(&target_logits, 1.0, draft_tokens.len());
                if sample_token > 0 {
                    generated.push(sample_token);
                    all_tokens.push(sample_token);
                }
            }
        }

        Ok(generated)
    }

    async fn generate_draft_tokens(&self, prefix: &[Token]) -> Result<Vec<Token>, String> {
        let mut draft_tokens = Vec::new();
        let mut current_tokens = prefix.to_vec();

        for _ in 0..self.max_draft_tokens {
            let draft_logits = {
                let model = self.draft_model.read().await;
                model.forward(&current_tokens).await.map_err(|e| e.to_string())?
            };

            let token = self.sample(&draft_logits, 0.8, 1);
            if token <= 0 {
                break;
            }

            draft_tokens.push(token);
            current_tokens.push(token);
        }

        Ok(draft_tokens)
    }

    async fn verify_and_accept(
        &self,
        draft_tokens: &[Token],
        target_logits: &[f32],
        generated: &mut Vec<Token>,
        max_tokens: usize,
    ) -> Result<(usize, Option<Token>), String> {
        let mut accepted_count = 0;
        let mut bonus_token = None;

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let draft_prob = self.softmax_at(target_logits, draft_token as usize);
            let target_prob = self.softmax_at(target_logits, draft_token as usize);

            // 标准的 speculative decoding acceptance
            let acceptance_prob = if draft_prob > 0.0 {
                (target_prob / draft_prob).min(1.0)
            } else {
                0.0
            };

            let r = rand_simple();
            if r < acceptance_prob {
                accepted_count = i + 1;
            } else {
                // 拒绝，接受到目前为止的
                break;
            }
        }

        // 如果所有都被接受，从 target 采样一个 bonus token
        if accepted_count == draft_tokens.len() && generated.len() < max_tokens {
            let bonus = self.sample(target_logits, 1.0, draft_tokens.len());
            if bonus > 0 && bonus != draft_tokens.last().copied().unwrap_or(0) {
                bonus_token = Some(bonus);
            }
        }

        Ok((accepted_count, bonus_token))
    }

    fn sample(&self, logits: &[f32], temperature: f32, offset: usize) -> Token {
        let vocab_size = logits.len();

        if temperature <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .skip(offset)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as Token)
                .unwrap_or(0);
        }

        let mut adjusted: Vec<f32> = logits
            .iter()
            .enumerate()
            .skip(offset)
            .map(|(_, &v)| v / temperature)
            .collect();

        let max_val = adjusted.iter().fold(f32::MIN, |a, &b| a.max(b));
        adjusted.iter_mut().for_each(|v| *v = (*v - max_val).exp());

        let sum: f32 = adjusted.iter().sum();
        if sum > 0.0 {
            adjusted.iter_mut().for_each(|v| *v /= sum);
        }

        let r = rand_simple();
        let mut cumsum = 0.0f32;
        for (i, &p) in adjusted.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return (i + offset) as Token;
            }
        }

        (vocab_size - 1) as Token
    }

    fn softmax_at(&self, logits: &[f32], idx: usize) -> f32 {
        if idx >= logits.len() {
            return 0.0;
        }

        let max_logit = logits.iter().fold(f32::MIN, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&v| (v - max_logit).exp()).sum();

        if exp_sum > 0.0 {
            (logits[idx] - max_logit).exp() / exp_sum
        } else {
            0.0
        }
    }
}

fn rand_simple() -> f32 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f32) / (u32::MAX as f32)
}
