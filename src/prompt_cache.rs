use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::types::*;

pub struct PromptCache {
    cache: Arc<RwLock<HashMap<u64, CachedPrompt>>>,
    max_cache_size: usize,
}

#[derive(Clone)]
pub struct CachedPrompt {
    pub prompt_hash: u64,
    pub tokens: Vec<Token>,
    pub kv_cache: Vec<CachedLayer>,
    pub created_at: u64,
    pub last_used: u64,
    pub use_count: usize,
}

#[derive(Clone)]
pub struct CachedLayer {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

impl PromptCache {
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size,
        }
    }

    pub async fn get(&self, prompt_hash: u64) -> Option<CachedPrompt> {
        let cache = self.cache.read().await;
        if let Some(mut cached) = cache.get(&prompt_hash).cloned() {
            cached.last_used = current_time();
            cached.use_count += 1;
            return Some(cached);
        }
        None
    }

    pub async fn insert(&self, prompt_hash: u64, tokens: Vec<Token>, kv_cache: Vec<CachedLayer>) {
        let mut cache = self.cache.write().await;
        
        if cache.len() >= self.max_cache_size {
            self.evict_oldest(&mut cache).await;
        }
        
        let cached = CachedPrompt {
            prompt_hash,
            tokens,
            kv_cache,
            created_at: current_time(),
            last_used: current_time(),
            use_count: 1,
        };
        
        cache.insert(prompt_hash, cached);
    }

    pub async fn invalidate(&self, prompt_hash: u64) {
        let mut cache = self.cache.write().await;
        cache.remove(&prompt_hash);
    }

    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let total_uses: usize = cache.values().map(|c| c.use_count).sum();
        
        CacheStats {
            num_cached: cache.len(),
            total_uses,
            max_size: self.max_cache_size,
        }
    }

    async fn evict_oldest(&self, cache: &mut HashMap<u64, CachedPrompt>) {
        if let Some((hash_to_remove, _)) = cache
            .iter()
            .min_by_key(|(_, v)| v.last_used)
            .map(|(k, v)| (*k, v.last_used))
        {
            cache.remove(&hash_to_remove);
        }
    }

    pub fn hash_prompt(prompt: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub num_cached: usize,
    pub total_uses: usize,
    pub max_size: usize,
}

fn current_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub struct ChunkedPrefill {
    max_chunk_size: usize,
    prompt_cache: PromptCache,
}

impl ChunkedPrefill {
    pub fn new(max_chunk_size: usize, max_cache_size: usize) -> Self {
        Self {
            max_chunk_size,
            prompt_cache: PromptCache::new(max_cache_size),
        }
    }

    pub async fn process_prompt(
        &self,
        tokens: &[Token],
        process_fn: impl Fn(&[Token]) -> Vec<f32>,
    ) -> Vec<f32> {
        let num_chunks = (tokens.len() + self.max_chunk_size - 1) / self.max_chunk_size;
        
        if num_chunks == 1 {
            return process_fn(tokens);
        }
        
        let mut all_logits = Vec::new();
        
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * self.max_chunk_size;
            let end = (start + self.max_chunk_size).min(tokens.len());
            let chunk = &tokens[start..end];
            
            let chunk_logits = process_fn(chunk);
            
            if chunk_idx == num_chunks - 1 {
                all_logits = chunk_logits;
            }
        }
        
        all_logits
    }

    pub fn get_cache(&self) -> &PromptCache {
        &self.prompt_cache
    }
}

pub struct PrefillBatch {
    pub tokens: Vec<Token>,
    pub start_pos: usize,
    pub chunk_size: usize,
}

impl PrefillBatch {
    pub fn new(tokens: Vec<Token>, chunk_size: usize) -> Self {
        Self {
            start_pos: 0,
            tokens,
            chunk_size,
        }
    }

    pub fn next_chunk(&mut self) -> Option<&[Token]> {
        if self.start_pos >= self.tokens.len() {
            return None;
        }
        
        let end = (self.start_pos + self.chunk_size).min(self.tokens.len());
        let chunk = &self.tokens[self.start_pos..end];
        self.start_pos = end;
        
        Some(chunk)
    }

    pub fn is_complete(&self) -> bool {
        self.start_pos >= self.tokens.len()
    }

    pub fn remaining(&self) -> usize {
        self.tokens.len() - self.start_pos
    }
}

pub struct DynamicBatch {
    prefill_queue: Vec<PrefillBatch>,
    decode_queue: Vec<Vec<Token>>,
    max_batch_size: usize,
    max_wait_ms: u64,
}

impl DynamicBatch {
    pub fn new(max_batch_size: usize, max_wait_ms: u64) -> Self {
        Self {
            prefill_queue: Vec::new(),
            decode_queue: Vec::new(),
            max_batch_size,
            max_wait_ms,
        }
    }

    pub fn add_prefill(&mut self, tokens: Vec<Token>, chunk_size: usize) {
        let batch = PrefillBatch::new(tokens, chunk_size);
        self.prefill_queue.push(batch);
    }

    pub fn add_decode(&mut self, tokens: Vec<Token>) {
        self.decode_queue.push(tokens);
    }

    pub fn get_next_batch(&mut self) -> Option<BatchType> {
        let total_pending = self.prefill_queue.len() + self.decode_queue.len();
        
        if total_pending == 0 {
            return None;
        }
        
        let batch_size = total_pending.min(self.max_batch_size);
        
        let prefill_count = batch_size.min(self.prefill_queue.len());
        
        if prefill_count > 0 {
            let mut prefill_tokens = Vec::new();
            for _ in 0..prefill_count {
                if let Some(mut batch) = self.prefill_queue.pop() {
                    if let Some(chunk) = batch.next_chunk() {
                        prefill_tokens.extend_from_slice(chunk);
                        
                        if !batch.is_complete() {
                            self.prefill_queue.push(batch);
                        }
                        
                        return Some(BatchType::Prefill(prefill_tokens.clone()));
                    }
                }
            }
        }
        
        if let Some(tokens) = self.decode_queue.pop() {
            return Some(BatchType::Decode(tokens));
        }
        
        None
    }
}

pub enum BatchType {
    Prefill(Vec<Token>),
    Decode(Vec<Token>),
}

pub struct SplitFuse {
    max_concurrent_slots: usize,
    active_slots: usize,
}

impl SplitFuse {
    pub fn new(max_concurrent_slots: usize) -> Self {
        Self {
            max_concurrent_slots,
            active_slots: 0,
        }
    }

    pub fn can_accept(&self) -> bool {
        self.active_slots < self.max_concurrent_slots
    }

    pub fn acquire_slot(&mut self) -> bool {
        if self.can_accept() {
            self.active_slots += 1;
            true
        } else {
            false
        }
    }

    pub fn release_slot(&mut self) {
        if self.active_slots > 0 {
            self.active_slots -= 1;
        }
    }

    pub fn process_prompt(&mut self, tokens: Vec<Token>, process_fn: impl Fn(&[Token]) -> Vec<f32>) -> Option<Vec<f32>> {
        if self.acquire_slot() {
            let result = process_fn(&tokens);
            self.release_slot();
            Some(result)
        } else {
            None
        }
    }
}
