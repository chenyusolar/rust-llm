use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct Profiler {
    counters: Mutex<HashMap<String, AtomicU64>>,
    timers: HashMap<String, Duration>,
    layer_latencies: Vec<LayerLatency>,
    memory_samples: Vec<MemorySample>,
    enabled: bool,
}

#[derive(Clone)]
pub struct LayerLatency {
    pub layer: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

#[derive(Clone)]
pub struct MemorySample {
    pub timestamp_ms: u64,
    pub allocated_bytes: u64,
    pub cached_bytes: u64,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
            timers: HashMap::new(),
            layer_latencies: Vec::new(),
            memory_samples: Vec::new(),
            enabled: true,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn increment_counter(&self, name: &str, value: u64) {
        if !self.enabled {
            return;
        }

        let mut counters = self.counters.lock().unwrap();
        let counter = counters
            .entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0));
        counter.fetch_add(value, Ordering::Relaxed);
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn record_layer_latency(&mut self, layer: usize, prefill_ms: f64, decode_ms: f64) {
        if !self.enabled {
            return;
        }

        self.layer_latencies.push(LayerLatency {
            layer,
            prefill_ms,
            decode_ms,
        });
    }

    pub fn record_memory(&mut self, allocated: u64, cached: u64) {
        if !self.enabled {
            return;
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.memory_samples.push(MemorySample {
            timestamp_ms: timestamp,
            allocated_bytes: allocated,
            cached_bytes: cached,
        });
    }

    pub fn get_counter(&self, name: &str) -> u64 {
        let counters = self.counters.lock().unwrap();
        counters
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    pub fn get_summary(&self) -> ProfilerSummary {
        let total_prefill: f64 = self.layer_latencies.iter().map(|l| l.prefill_ms).sum();
        let total_decode: f64 = self.layer_latencies.iter().map(|l| l.decode_ms).sum();

        let avg_memory = if !self.memory_samples.is_empty() {
            self.memory_samples
                .iter()
                .map(|s| s.allocated_bytes)
                .sum::<u64>()
                / self.memory_samples.len() as u64
        } else {
            0
        };

        ProfilerSummary {
            total_requests: self.get_counter("requests"),
            total_tokens: self.get_counter("tokens"),
            prefill_latency_ms: total_prefill,
            decode_latency_ms: total_decode,
            avg_memory_bytes: avg_memory,
            kv_cache_hits: self.get_counter("kv_cache_hits"),
            kv_cache_misses: self.get_counter("kv_cache_misses"),
        }
    }

    pub fn reset(&mut self) {
        self.counters.lock().unwrap().clear();
        self.layer_latencies.clear();
        self.memory_samples.clear();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ProfilerSummary {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub prefill_latency_ms: f64,
    pub decode_latency_ms: f64,
    pub avg_memory_bytes: u64,
    pub kv_cache_hits: u64,
    pub kv_cache_misses: u64,
}

impl ProfilerSummary {
    pub fn tokens_per_second(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs > 0.0 {
            self.total_tokens as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    pub fn kv_cache_hit_rate(&self) -> f64 {
        let total = self.kv_cache_hits + self.kv_cache_misses;
        if total > 0 {
            self.kv_cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn print(&self) {
        println!("\n=== Performance Summary ===");
        println!("Total Requests: {}", self.total_requests);
        println!("Total Tokens: {}", self.total_tokens);
        println!("Prefill Latency: {:.2} ms", self.prefill_latency_ms);
        println!("Decode Latency: {:.2} ms", self.decode_latency_ms);
        println!(
            "Avg Memory: {:.2} MB",
            self.avg_memory_bytes as f64 / 1024.0 / 1024.0
        );
        println!(
            "KV Cache Hit Rate: {:.1}%",
            self.kv_cache_hit_rate() * 100.0
        );
    }
}

pub struct MetricsCollector {
    start_time: Instant,
    request_count: AtomicU64,
    token_count: AtomicU64,
    error_count: AtomicU64,
    latencies: Mutex<Vec<f64>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            request_count: AtomicU64::new(0),
            token_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            latencies: Mutex::new(Vec::new()),
        }
    }

    pub fn record_request(&self, tokens: usize, latency_ms: f64) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.token_count.fetch_add(tokens as u64, Ordering::Relaxed);

        let mut latencies = self.latencies.lock().unwrap();
        latencies.push(latency_ms);
    }

    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_metrics(&self) -> Metrics {
        let elapsed = self.start_time.elapsed().as_secs_f64();

        let latencies = self.latencies.lock().unwrap();
        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else {
            0.0
        };

        let p99_latency = if !latencies.is_empty() {
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
            sorted[idx]
        } else {
            0.0
        };

        Metrics {
            uptime_seconds: elapsed,
            total_requests: self.request_count.load(Ordering::Relaxed),
            total_tokens: self.token_count.load(Ordering::Relaxed),
            total_errors: self.error_count.load(Ordering::Relaxed),
            requests_per_second: if elapsed > 0.0 {
                self.request_count.load(Ordering::Relaxed) as f64 / elapsed
            } else {
                0.0
            },
            tokens_per_second: if elapsed > 0.0 {
                self.token_count.load(Ordering::Relaxed) as f64 / elapsed
            } else {
                0.0
            },
            avg_latency_ms: avg_latency,
            p99_latency_ms: p99_latency,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Metrics {
    pub uptime_seconds: f64,
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_errors: u64,
    pub requests_per_second: f64,
    pub tokens_per_second: f64,
    pub avg_latency_ms: f64,
    pub p99_latency_ms: f64,
}
