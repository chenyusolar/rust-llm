use crate::types::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    Prefill,
    Decode,
}

pub struct InferenceRequest2 {
    pub id: String,
    pub tokens: Vec<Token>,
    pub max_length: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub phase: RequestPhase,
    pub prompt_len: usize,
    pub generated_len: usize,
    pub is_complete: bool,
    pub priority: i32,
    pub arrival_time: u64,
}

impl InferenceRequest2 {
    pub fn new(id: String, tokens: Vec<Token>, max_length: usize) -> Self {
        Self {
            id,
            tokens,
            max_length,
            temperature: 1.0,
            top_p: 0.9,
            phase: RequestPhase::Prefill,
            prompt_len: 0,
            generated_len: 0,
            is_complete: false,
            priority: 0,
            arrival_time: 0,
        }
    }

    pub fn with_sampling(mut self, temperature: f32, top_p: f32) -> Self {
        self.temperature = temperature;
        self.top_p = top_p;
        self
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

pub struct BatchScheduler {
    max_batch_size: usize,
    max_prefill_batch_size: usize,
    max_decode_batch_size: usize,
    prefill_queue: VecDeque<Arc<RwLock<InferenceRequest2>>>,
    decode_queue: VecDeque<Arc<RwLock<InferenceRequest2>>>,
    running_requests: HashMap<String, Arc<RwLock<InferenceRequest2>>>,
    prefill_chunk_size: usize,
    enable_mingle: bool,
    kv_cache_quantizer: Option<Box<dyn KvCacheQuantizer>>,
}

impl BatchScheduler {
    pub fn new(
        max_batch_size: usize,
        max_prefill_batch_size: usize,
        max_decode_batch_size: usize,
    ) -> Self {
        Self {
            max_batch_size,
            max_prefill_batch_size,
            max_decode_batch_size,
            prefill_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            running_requests: HashMap::new(),
            prefill_chunk_size: 512,
            enable_mingle: true,
            kv_cache_quantizer: None,
        }
    }

    pub fn with_mingle(mut self, enable: bool) -> Self {
        self.enable_mingle = enable;
        self
    }

    pub fn with_kv_quantizer(mut self, quantizer: Box<dyn KvCacheQuantizer>) -> Self {
        self.kv_cache_quantizer = Some(quantizer);
        self
    }

    pub fn add_request(&mut self, request: InferenceRequest2) {
        let request = Arc::new(RwLock::new(request));
        
        let phase = request.blocking_read().phase;
        if phase == RequestPhase::Prefill {
            self.prefill_queue.push_back(request.clone());
        } else {
            self.decode_queue.push_back(request.clone());
        }
        
        let id = request.blocking_read().id.clone();
        self.running_requests.insert(id, request);
    }

    pub async fn schedule_batch(&mut self) -> Vec<Arc<RwLock<InferenceRequest2>>> {
        let mut batch = Vec::new();
        
        let prefill_size = self.prefill_queue.len().min(self.max_prefill_batch_size);
        
        for _ in 0..prefill_size {
            if let Some(req) = self.prefill_queue.pop_front() {
                batch.push(req);
            }
        }
        
        if self.enable_mingle && batch.len() < self.max_batch_size {
            let decode_size = (self.max_batch_size - batch.len()).min(self.decode_queue.len());
            
            for _ in 0..decode_size {
                if let Some(req) = self.decode_queue.pop_front() {
                    batch.push(req);
                }
            }
        }
        
        batch
    }

    pub async fn process_batch(&mut self, batch: Vec<Arc<RwLock<InferenceRequest2>>>) {
        let mut completed = Vec::new();
        
        for request in batch {
            let mut req = request.write().await;
            
            if req.phase == RequestPhase::Prefill {
                req.prompt_len = req.tokens.len();
                req.phase = RequestPhase::Decode;
                self.decode_queue.push_back(request.clone());
            } else {
                req.generated_len += 1;
                
                if req.generated_len >= req.max_length || req.tokens.last() == Some(&0) {
                    req.is_complete = true;
                    completed.push(req.id.clone());
                } else {
                    self.decode_queue.push_back(request.clone());
                }
            }
        }
        
        for id in completed {
            self.running_requests.remove(&id);
        }
    }

    pub async fn get_queue_sizes(&self) -> (usize, usize, usize) {
        (
            self.prefill_queue.len(),
            self.decode_queue.len(),
            self.running_requests.len(),
        )
    }

    pub fn pending_count(&self) -> usize {
        self.prefill_queue.len() + self.decode_queue.len()
    }
}

pub trait KvCacheQuantizer: Send + Sync {
    fn quantize(&self, data: &[f32]) -> Vec<u8>;
    fn dequantize(&self, data: &[u8]) -> Vec<f32>;
    fn compression_ratio(&self) -> f32;
}

pub struct AdaptiveBatchScheduler {
    scheduler: BatchScheduler,
    target_latency_ms: u64,
    last_batch_size: usize,
    history: Vec<u64>,
}

impl AdaptiveBatchScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            scheduler: BatchScheduler::new(max_batch_size, max_batch_size, max_batch_size),
            target_latency_ms: 100,
            last_batch_size: 1,
            history: Vec::new(),
        }
    }

    pub fn with_target_latency(mut self, latency_ms: u64) -> Self {
        self.target_latency_ms = latency_ms;
        self
    }

    pub async fn schedule_adaptive(&mut self) -> Vec<Arc<RwLock<InferenceRequest2>>> {
        let batch = self.scheduler.schedule_batch().await;
        
        let avg_latency = if !self.history.is_empty() {
            self.history.iter().sum::<u64>() / self.history.len() as u64
        } else {
            self.target_latency_ms
        };
        
        if avg_latency > self.target_latency_ms && self.last_batch_size > 1 {
            self.last_batch_size = (self.last_batch_size - 1).max(1);
        } else if avg_latency < self.target_latency_ms / 2 {
            self.last_batch_size = (self.last_batch_size + 1).min(128);
        }
        
        batch
    }

    pub async fn record_latency(&mut self, latency_ms: u64) {
        self.history.push(latency_ms);
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }
}

pub struct PrefillDecodeDisaggregation {
    prefill_device: DeviceType,
    decode_device: DeviceType,
    kv_transfer_overhead_us: u64,
}

impl PrefillDecodeDisaggregation {
    pub fn new(prefill_device: DeviceType, decode_device: DeviceType) -> Self {
        Self {
            prefill_device,
            decode_device,
            kv_transfer_overhead_us: 100,
        }
    }

    pub fn should_disaggregate(&self, prompt_len: usize, batch_size: usize) -> bool {
        prompt_len > 512 || batch_size > 8
    }

    pub fn estimate_overhead(&self, prompt_len: usize, kv_size: usize) -> u64 {
        self.kv_transfer_overhead_us + (kv_size as u64 / 1000)
    }
}
