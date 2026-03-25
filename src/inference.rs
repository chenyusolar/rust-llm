use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::types::*;
use crate::paged_attention::{DynamicBatcher, InferenceRequest, PagedAttentionCache};
use crate::decoder::Decoder;

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
            let draft_logits = {
                let mut model = self.draft_model.write().await;
                model.forward(&all_tokens).await.map_err(|e| e.to_string())?
            };
            
            let mut draft_tokens = Vec::new();
            for _ in 0..self.max_draft_tokens {
                let draft_token = self.sample(&draft_logits, 0.8);
                if draft_token <= 0 {
                    break;
                }
                draft_tokens.push(draft_token);
                
                all_tokens.push(draft_token);
            }
            
            if draft_tokens.is_empty() {
                break;
            }
            
            let target_logits = {
                let mut model = self.target_model.write().await;
                model.forward(&all_tokens).await.map_err(|e| e.to_string())?
            };
            
            let mut accepted = Vec::new();
            let mut i = 0;
            while i < draft_tokens.len() {
                let draft_token = draft_tokens[i];
                let target_prob = target_logits.get(draft_token as usize).copied().unwrap_or(0.0);
                let draft_prob = draft_logits.get(draft_token as usize).copied().unwrap_or(0.0);
                
                let acceptance_ratio = if draft_prob > 0.0 {
                    (target_prob / draft_prob).min(1.0)
                } else {
                    0.0
                };
                
                if acceptance_ratio > self.acceptance_threshold || i == draft_tokens.len() - 1 {
                    accepted.push(draft_token);
                    
                    let final_token = if i == draft_tokens.len() - 1 {
                        draft_token
                    } else {
                        self.sample(&target_logits, 1.0)
                    };
                    
                    if final_token > 0 {
                        generated.push(final_token);
                    }
                    
                    break;
                } else {
                    accepted.push(draft_token);
                }
                
                i += 1;
            }
            
            if accepted.is_empty() || generated.len() >= max_tokens {
                break;
            }
            
            all_tokens = tokens.to_vec();
            all_tokens.extend(&generated);
        }
        
        Ok(generated)
    }

    fn sample(&self, logits: &[f32], temperature: f32) -> Token {
        if temperature <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as Token)
                .unwrap_or(0);
        }
        
        let mut adjusted: Vec<f32> = logits
            .iter()
            .map(|v| v / temperature)
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
                return i as Token;
            }
        }
        
        (adjusted.len() - 1) as Token
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
