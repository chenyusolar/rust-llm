use std::time::{Duration, Instant};

pub struct Benchmark {
    name: String,
    iterations: usize,
    warmup_iterations: usize,
    results: Vec<Duration>,
}

impl Benchmark {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            iterations: 100,
            warmup_iterations: 10,
            results: Vec::new(),
        }
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_iterations = warmup;
        self
    }

    pub fn run<F, R>(&mut self, f: F) -> BenchmarkResult
    where
        F: Fn() -> R + Clone,
    {
        for _ in 0..self.warmup_iterations {
            let _ = f();
        }

        self.results.clear();

        for _ in 0..self.iterations {
            let start = Instant::now();
            let _ = f();
            let elapsed = start.elapsed();
            self.results.push(elapsed);
        }

        self.compute_stats()
    }

    pub async fn run_async<F, R, Fut>(&mut self, f: F) -> BenchmarkResult
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = R>,
    {
        for _ in 0..self.warmup_iterations {
            let _ = f().await;
        }

        self.results.clear();

        for _ in 0..self.iterations {
            let start = Instant::now();
            let _ = f().await;
            let elapsed = start.elapsed();
            self.results.push(elapsed);
        }

        self.compute_stats()
    }

    fn compute_stats(&self) -> BenchmarkResult {
        if self.results.is_empty() {
            return BenchmarkResult::default();
        }

        let total: Duration = self.results.iter().sum();
        let mean = total / self.results.len() as u32;

        let mut sorted = self.results.clone();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        let min = sorted.first().cloned().unwrap_or(Duration::ZERO);
        let max = sorted.last().cloned().unwrap_or(Duration::ZERO);

        let variance: f64 = self.results.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>() / self.results.len() as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt() / 1_000_000_000.0);

        BenchmarkResult {
            name: self.name.clone(),
            iterations: self.iterations,
            mean,
            median,
            min,
            max,
            std_dev,
            total,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub mean: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
    pub total: Duration,
}

impl BenchmarkResult {
    pub fn throughput(&self, items: usize) -> f64 {
        let secs = self.mean.as_secs_f64();
        if secs > 0.0 {
            items as f64 / secs
        } else {
            0.0
        }
    }

    pub fn print(&self) {
        println!("\n=== {} ===", self.name);
        println!("Iterations: {}", self.iterations);
        println!("Mean:       {:.3} ms", self.mean.as_secs_f64() * 1000.0);
        println!("Median:     {:.3} ms", self.median.as_secs_f64() * 1000.0);
        println!("Min:        {:.3} ms", self.min.as_secs_f64() * 1000.0);
        println!("Max:        {:.3} ms", self.max.as_secs_f64() * 1000.0);
        println!("Std Dev:    {:.3} ms", self.std_dev.as_secs_f64() * 1000.0);
        println!("Total:      {:.3} ms", self.total.as_secs_f64() * 1000.0);
    }
}

pub struct BenchmarkSuite {
    benchmarks: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
        }
    }

    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.benchmarks.push(result);
    }

    pub fn run_benchmark<F, R>(&mut self, name: &str, f: F)
    where
        F: Fn() -> R + Clone,
    {
        let mut bench = Benchmark::new(name);
        let result = bench.run(f);
        result.print();
        self.add_result(result);
    }

    pub async fn run_benchmark_async<F, R, Fut>(&mut self, name: &str, f: F)
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = R>,
    {
        let mut bench = Benchmark::new(name);
        let result = bench.run_async(f).await;
        result.print();
        self.add_result(result);
    }

    pub fn print_summary(&self) {
        println!("\n\n=== BENCHMARK SUMMARY ===");
        for result in &self.benchmarks {
            println!("{}: {:.3} ms/iter", result.name, result.mean.as_secs_f64() * 1000.0);
        }
    }
}

pub fn benchmark_matmul(m: usize, k: usize, n: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();

    let mut suite = BenchmarkSuite::new();
    suite.run_benchmark(&format!("matmul {}x{}x{}", m, k, n), || {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    });
}

pub fn benchmark_attention(seq_len: usize, num_heads: usize, head_dim: usize) {
    let hidden_size = num_heads * head_dim;
    let q: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01).collect();
    let k: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01).collect();
    let v: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01).collect();

    let mut suite = BenchmarkSuite::new();
    suite.run_benchmark(
        &format!("attention seq={} heads={} dim={}", seq_len, num_heads, head_dim),
        || {
            let mut output = vec![0.0f32; seq_len * hidden_size];

            for h in 0..num_heads {
                let q_offset = h * head_dim;
                let k_offset = h * head_dim;
                let v_offset = h * head_dim;

                let mut scores = vec![0.0f32; seq_len];
                let mut max_score = f32::MIN;

                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k[k_offset + j * head_dim + d];
                    }
                    dot /= (head_dim as f32).sqrt();
                    scores[j] = dot;
                    max_score = max_score.max(dot);
                }

                let mut exp_sum = 0.0f32;
                for j in 0..seq_len {
                    scores[j] = (scores[j] - max_score).exp();
                    exp_sum += scores[j];
                }

                for j in 0..seq_len {
                    scores[j] /= exp_sum;
                }

                for j in 0..seq_len {
                    for d in 0..head_dim {
                        output[q_offset + j * head_dim + d] +=
                            scores[j] * v[v_offset + j * head_dim + d];
                    }
                }
            }

            output
        },
    );
}

pub fn benchmark_rms_norm(dim: usize) {
    let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
    let weight: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 + 1.0).collect();

    let mut suite = BenchmarkSuite::new();
    suite.run_benchmark(&format!("rms_norm dim={}", dim), || {
        let mut sq_sum = 0.0f32;
        for &xi in &x {
            sq_sum += xi * xi;
        }
        let mean = sq_sum / dim as f32;
        let rms = (mean + 1e-5).sqrt();

        let mut output = vec![0.0f32; dim];
        for i in 0..dim {
            output[i] = x[i] / rms * weight[i];
        }

        output
    });
}

pub fn benchmark_rope(seq_len: usize, head_dim: usize, base: f32) {
    let mut suite = BenchmarkSuite::new();
    suite.run_benchmark(
        &format!("rope seq={} head_dim={}", seq_len, head_dim),
        || {
            let mut q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();
            let mut k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();

            for i in 0..seq_len {
                for j in 0..head_dim / 2 {
                    let freq = (i as f32).powf(j as f32 * -2.0 / head_dim as f32) * base;
                    let cos = freq.cos();
                    let sin = freq.sin();

                    let q_idx = i * head_dim + j;
                    let q_idx_rot = i * head_dim + j + head_dim / 2;
                    let q0 = q[q_idx];
                    let q1 = q[q_idx_rot];
                    q[q_idx] = q0 * cos - q1 * sin;
                    q[q_idx_rot] = q0 * sin + q1 * cos;

                    let k_idx = i * head_dim + j;
                    let k_idx_rot = i * head_dim + j + head_dim / 2;
                    let k0 = k[k_idx];
                    let k1 = k[k_idx_rot];
                    k[k_idx] = k0 * cos - k1 * sin;
                    k[k_idx_rot] = k0 * sin + k1 * cos;
                }
            }
            (q, k)
        },
    );
}

pub fn benchmark_mlp(hidden_size: usize, intermediate_size: usize) {
    let x: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();
    let gate_proj: Vec<f32> = (0..intermediate_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
    let up_proj: Vec<f32> = (0..intermediate_size * hidden_size).map(|i| (i as f32) * 0.01).collect();
    let down_proj: Vec<f32> = (0..hidden_size * intermediate_size).map(|i| (i as f32) * 0.01).collect();

    let mut suite = BenchmarkSuite::new();
    suite.run_benchmark(
        &format!("mlp hidden={} intermediate={}", hidden_size, intermediate_size),
        || {
            let mut gate_output = vec![0.0f32; intermediate_size];
            let mut up_output = vec![0.0f32; intermediate_size];

            for i in 0..intermediate_size {
                for j in 0..hidden_size {
                    gate_output[i] += x[j] * gate_proj[j * intermediate_size + i];
                    up_output[i] += x[j] * up_proj[j * intermediate_size + i];
                }
            }

            for i in 0..intermediate_size {
                gate_output[i] = gate_output[i] / (1.0 + (-gate_output[i]).exp());
            }

            let mut intermediate = vec![0.0f32; intermediate_size];
            for i in 0..intermediate_size {
                intermediate[i] = gate_output[i] * up_output[i];
            }

            let mut output = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                for i in 0..intermediate_size {
                    output[j] += intermediate[i] * down_proj[i * hidden_size + j];
                }
            }

            output
        },
    );
}

pub fn run_all_benchmarks() {
    println!("Running LLM Inference Benchmarks...\n");

    benchmark_matmul(4096, 4096, 4096);
    benchmark_matmul(512, 4096, 512);

    benchmark_attention(512, 32, 128);
    benchmark_attention(2048, 32, 128);
    benchmark_attention(4096, 32, 128);

    benchmark_rms_norm(4096);
    benchmark_rms_norm(8192);

    benchmark_rope(512, 128, 10000.0);
    benchmark_rope(2048, 128, 10000.0);

    benchmark_mlp(4096, 11008);
    benchmark_mlp(3584, 18944);
}
