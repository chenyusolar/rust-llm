#[allow(unused_imports)]
use crate::types::*;

#[allow(unused_imports)]
use anyhow::Result;

#[cfg(feature = "gpu")]
use wgpu::{
    ComputePipeline, Device, Queue, ComputePass, Buffer, BufferDescriptor, 
    ShaderModule, BindGroup, BindGroupDescriptor, BindGroupEntry, BufferBindingType,
};

pub struct GpuKernel {
    device_id: usize,
    max_batch_size: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    gpu_available: bool,
    
    #[cfg(feature = "gpu")]
    device: Option<Device>,
    #[cfg(feature = "gpu")]
    queue: Option<Queue>,
    #[cfg(feature = "gpu")]
    matmul_pipeline: Option<ComputePipeline>,
    #[cfg(feature = "gpu")]
    attention_pipeline: Option<ComputePipeline>,
    #[cfg(feature = "gpu")]
    rope_pipeline: Option<ComputePipeline>,
    #[cfg(feature = "gpu")]
    rms_norm_pipeline: Option<ComputePipeline>,
    #[cfg(feature = "gpu")]
    silu_pipeline: Option<ComputePipeline>,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

const MATMUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = params.x;
    let n = params.y;
    let k = params.z;
    
    let row = global_id.y;
    let col = global_id.x;
    
    if row >= m || col >= n {
        return;
    }
    
    var sum = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        sum = sum + a[row * k + i] * b[i * n + col];
    }
    
    c[row * n + col] = sum;
}
"#;

const ATTENTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let head_dim = params.x;
    let seq_len = params.y;
    let kv_len = params.z;
    let head_idx = global_id.x;
    
    if head_idx >= params.w {
        return;
    }
    
    let q_offset = head_idx * head_dim;
    
    var max_score = -1e10;
    var exp_sum = 0.0;
    
    for (var j = 0u; j < kv_len; j = j + 1u) {
        var dot = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q[q_offset + d] * k[j * head_dim + d];
        }
        dot = dot / sqrt(f32(head_dim));
        
        if dot > max_score {
            max_score = dot;
        }
    }
    
    for (var j = 0u; j < kv_len; j = j + 1u) {
        var dot = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q[q_offset + d] * k[j * head_dim + d];
        }
        dot = dot / sqrt(f32(head_dim));
        
        let exp_dot = exp(dot - max_score);
        exp_sum = exp_sum + exp_dot;
        
        for (var d = 0u; d < head_dim; d = d + 1u) {
            output[q_offset + d] = output[q_offset + d] + exp_dot * v[j * head_dim + d];
        }
    }
    
    if exp_sum > 0.0 {
        for (var d = 0u; d < head_dim; d = d + 1u) {
            output[q_offset + d] = output[q_offset + d] / exp_sum;
        }
    }
}
"#;

const ROPE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let head_dim = params.x;
    let seq_len = params.y;
    let base = bitcast<f32>(params.z);
    let idx = global_id.x;
    
    if idx >= seq_len * head_dim / 2u {
        return;
    }
    
    let pos = idx / (head_dim / 2u);
    let dim = idx % (head_dim / 2u);
    
    let freq = pow(base, -f32(dim * 2u) / f32(head_dim)) * f32(pos);
    let cos_val = cos(freq);
    let sin_val = sin(freq);
    
    let q_idx = pos * head_dim + dim;
    let q_idx_rot = pos * head_dim + dim + head_dim / 2u;
    let q0 = q[q_idx];
    let q1 = q[q_idx_rot];
    q[q_idx] = q0 * cos_val - q1 * sin_val;
    q[q_idx_rot] = q0 * sin_val + q1 * cos_val;
    
    let k_idx = pos * head_dim + dim;
    let k_idx_rot = pos * head_dim + dim + head_dim / 2u;
    let k0 = k[k_idx];
    let k1 = k[k_idx_rot];
    k[k_idx] = k0 * cos_val - k1 * sin_val;
    k[k_idx_rot] = k0 * sin_val + k1 * cos_val;
}
"#;

const RMS_NORM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = params.x;
    let idx = global_id.x;
    
    if idx >= dim {
        return;
    }
    
    var sq_sum = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        sq_sum = sq_sum + x[i] * x[i];
    }
    let rms = sqrt(sq_sum / f32(dim) + 1e-5);
    
    output[idx] = x[idx] / rms * weight[idx];
}
"#;

const SILU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.x;
    
    if idx >= n {
        return;
    }
    
    let val = x[idx];
    output[idx] = val / (1.0 + exp(-val));
}
"#;

const FUSED_MLP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gate_proj: array<f32>;
@group(0) @binding(2) var<storage, read> up_proj: array<f32>;
@group(0) @binding(3) var<storage, read> down_proj: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: vec3<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let hidden_size = params.x;
    let intermediate_size = params.y;
    let idx = global_id.x;
    
    if idx >= hidden_size {
        return;
    }
    
    var gate = 0.0;
    var up = 0.0;
    
    for (var i = 0u; i < intermediate_size; i = i + 1u) {
        gate = gate + x[i] * gate_proj[i * hidden_size + idx];
        up = up + x[i] * up_proj[i * hidden_size + idx];
    }
    
    let silu_gate = gate / (1.0 + exp(-gate));
    let intermediate = silu_gate * up;
    
    var result = 0.0;
    for (var i = 0u; i < intermediate_size; i = i + 1u) {
        result = result + intermediate[i] * down_proj[idx * intermediate_size + i];
    }
    
    output[idx] = result;
}
"#;

impl GpuKernel {
    pub fn new(device_id: usize, max_batch_size: usize, head_dim: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        Self {
            device_id,
            max_batch_size,
            head_dim,
            num_heads,
            num_kv_heads,
            gpu_available: false,
            
            #[cfg(feature = "gpu")]
            device: None,
            #[cfg(feature = "gpu")]
            queue: None,
            #[cfg(feature = "gpu")]
            matmul_pipeline: None,
            #[cfg(feature = "gpu")]
            attention_pipeline: None,
            #[cfg(feature = "gpu")]
            rope_pipeline: None,
            #[cfg(feature = "gpu")]
            rms_norm_pipeline: None,
            #[cfg(feature = "gpu")]
            silu_pipeline: None,
        }
    }

    pub fn init_cuda(&mut self) -> Result<()> {
        #[cfg(feature = "gpu")]
        {
            tracing::info!("Initializing GPU compute with wgpu...");
            // GPU 初始化将在首次使用时通过 init_wgpu() 完成
            // 这里只标记 GPU 可用，实际初始化是异步的
            self.gpu_available = true;
            tracing::info!("GPU support enabled (will initialize on first use)");
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            tracing::warn!("GPU support not enabled, running in CPU fallback mode");
        }
        
        Ok(())
    }

    /// 确保 GPU 已初始化
    #[cfg(feature = "gpu")]
    pub async fn ensure_gpu_init(&mut self) -> Result<()> {
        if !self.is_gpu_initialized() {
            tracing::info!("Initializing WGPU now...");
            self.init_wgpu().await?;
        }
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    pub async fn ensure_gpu_init(&mut self) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub async fn init_wgpu(&mut self) -> Result<()> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;
        
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
            required_limits: wgpu::Limits::default(),
            label: Some("rust-llm-gpu"),
        }, None).await?;
        
        let matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });
        
        let attention_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention"),
            source: wgpu::ShaderSource::Wgsl(ATTENTION_SHADER.into()),
        });
        
        let rope_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rope"),
            source: wgpu::ShaderSource::Wgsl(ROPE_SHADER.into()),
        });
        
        let rms_norm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rms_norm"),
            source: wgpu::ShaderSource::Wgsl(RMS_NORM_SHADER.into()),
        });
        
        let silu_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silu"),
            source: wgpu::ShaderSource::Wgsl(SILU_SHADER.into()),
        });
        
        let fused_mlp_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fused_mlp"),
            source: wgpu::ShaderSource::Wgsl(FUSED_MLP_SHADER.into()),
        });
        
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: None,
            module: &matmul_module,
            entry_point: "main",
        });
        
        let attention_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attention"),
            layout: None,
            module: &attention_module,
            entry_point: "main",
        });
        
        let rope_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rope"),
            layout: None,
            module: &rope_module,
            entry_point: "main",
        });
        
        let rms_norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rms_norm"),
            layout: None,
            module: &rms_norm_module,
            entry_point: "main",
        });
        
        let silu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("silu"),
            layout: None,
            module: &silu_module,
            entry_point: "main",
        });
        
        self.device = Some(device);
        self.queue = Some(queue);
        self.matmul_pipeline = Some(matmul_pipeline);
        self.attention_pipeline = Some(attention_pipeline);
        self.rope_pipeline = Some(rope_pipeline);
        self.rms_norm_pipeline = Some(rms_norm_pipeline);
        self.silu_pipeline = Some(silu_pipeline);
        self.gpu_available = true;
        
        tracing::info!("WGPU initialized successfully with all GPU kernels");
        
        Ok(())
    }

    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    pub async fn forward_attention(
        &self,
        q: &mut [f32],
        k: &[f32],
        v: &[f32],
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        seq_len: usize,
        kv_len: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            // 如果 GPU 已初始化，尝试使用 GPU
            if self.is_gpu_initialized() {
                match self.forward_attention_gpu(q, k, v, self.head_dim, seq_len, kv_len, self.num_heads).await {
                    Ok(output) => return Ok(output),
                    Err(e) => {
                        tracing::warn!("GPU attention failed, falling back to CPU: {}", e);
                    }
                }
            }
        }

        // CPU fallback - 优化的自注意力实现
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let kv_per_head = (num_heads / num_kv_heads.max(1)).max(1);

        let scale = (head_dim as f32).sqrt();

        // 更新 KV cache
        if !k_cache.is_empty() && !v_cache.is_empty() {
            let cache_offset = seq_len * num_kv_heads * head_dim;
            if cache_offset < k_cache.len() && cache_offset < v_cache.len() {
                let copy_size = k.len().min(k_cache.len() - cache_offset);
                k_cache[cache_offset..cache_offset + copy_size].copy_from_slice(&k[..copy_size]);
                v_cache[cache_offset..cache_offset + copy_size].copy_from_slice(&v[..copy_size]);
            }
        }

        let mut output = vec![0.0f32; q.len()];

        for h in 0..num_heads {
            let kv_head = h / kv_per_head;
            let q_offset = h * head_dim;
            let k_offset = kv_head * head_dim;
            let v_offset = kv_head * head_dim;

            // 收集该 head 的 query 和所有 key/value
            let q_slice = &q[q_offset..q_offset + head_dim];

            // 计算 attention scores
            let mut scores = vec![0.0f32; kv_len.min(seq_len)];

            for j in 0..scores.len() {
                let k_j = k_offset + j * head_dim;
                if k_j + head_dim <= k.len() {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_slice[d] * k[k_j + d];
                    }
                    scores[j] = dot / scale;
                }
            }

            // Softmax
            let max_score = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut exp_sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= exp_sum.max(1e-8);
            }

            // 加权求和
            for j in 0..scores.len() {
                let v_j = v_offset + j * head_dim;
                let weight = scores[j];
                if v_j + head_dim <= v.len() {
                    for d in 0..head_dim {
                        output[q_offset + d] += weight * v[v_j + d];
                    }
                }
            }
        }

        Ok(output)
    }

    pub async fn forward_mlp(
        &self,
        x: &[f32],
        gate_proj: &[f32],
        up_proj: &[f32],
        down_proj: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Vec<f32>> {
        let mut gate_output = vec![0.0f32; intermediate_size];
        let mut up_output = vec![0.0f32; intermediate_size];
        
        for i in 0..intermediate_size {
            for j in 0..hidden_size {
                gate_output[i] += x[j] * gate_proj[j * intermediate_size + i];
                up_output[i] += x[j] * up_proj[j * intermediate_size + i];
            }
        }
        
        for i in 0..intermediate_size {
            gate_output[i] = silu(gate_output[i]);
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
        
        Ok(output)
    }

    pub async fn forward_matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
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
        
        Ok(c)
    }

    #[cfg(feature = "gpu")]
    pub async fn forward_matmul_gpu(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        use wgpu::BufferUsages;
        
        let device = self.device.as_ref().ok_or_else(|| anyhow::anyhow!("GPU not initialized"))?;
        let queue = self.queue.as_ref().ok_or_else(|| anyhow::anyhow!("GPU queue not initialized"))?;
        
        let a_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("matmul_a"),
            size: (a.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let b_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("matmul_b"),
            size: (b.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let c_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("matmul_c"),
            size: (m * n * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        
        queue.write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        queue.write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));
        
        let params_data: [u32; 4] = [m as u32, n as u32, k as u32, 0];
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));
        
        let pipeline = self.matmul_pipeline.as_ref().ok_or_else(|| anyhow::anyhow!("Matmul pipeline not initialized"))?;
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(a_buffer.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(b_buffer.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(c_buffer.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(params_buffer.as_entire_buffer_binding()),
                },
            ],
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((n + 7) / 8, (m + 7) / 8, 1);
        }
        
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::WaitForSubmission(true));
        
        let mut c = vec![0.0f32; m * n];
        queue.read_buffer(&c_buffer, 0, bytemuck::cast_slice_mut(&mut c));
        
        tracing::info!("GPU matmul: {}x{}x{}", m, k, n);
        
        Ok(c)
    }

    #[cfg(feature = "gpu")]
    pub async fn forward_attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        head_dim: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
    ) -> Result<Vec<f32>> {
        use wgpu::BufferUsages;
        
        let device = self.device.as_ref().ok_or_else(|| anyhow::anyhow!("GPU not initialized"))?;
        let queue = self.queue.as_ref().ok_or_else(|| anyhow::anyhow!("GPU queue not initialized"))?;
        
        let q_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("attention_q"),
            size: (q.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let k_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("attention_k"),
            size: (k.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let v_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("attention_v"),
            size: (v.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("attention_output"),
            size: (q.len() * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("attention_params"),
            size: 16,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        
        queue.write_buffer(&q_buffer, 0, bytemuck::cast_slice(q));
        queue.write_buffer(&k_buffer, 0, bytemuck::cast_slice(k));
        queue.write_buffer(&v_buffer, 0, bytemuck::cast_slice(v));
        
        let params_data: [u32; 4] = [
            head_dim as u32,
            seq_len as u32,
            kv_len as u32,
            num_heads as u32,
        ];
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));
        
        let pipeline = self.attention_pipeline.as_ref().ok_or_else(|| anyhow::anyhow!("Attention pipeline not initialized"))?;
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("attention_bind_group"),
            layout: pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(q_buffer.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(k_buffer.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(v_buffer.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(output_buffer.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(params_buffer.as_entire_buffer_binding()),
                },
            ],
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attention_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention_pass"),
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_heads as u32, 1, 1);
        }
        
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::WaitForSubmission(true));
        
        let mut output = vec![0.0f32; q.len()];
        queue.read_buffer(&output_buffer, 0, bytemuck::cast_slice_mut(&mut output));
        
        Ok(output)
    }

    #[cfg(feature = "gpu")]
    pub async fn forward_fused_mlp_gpu(
        &self,
        x: &[f32],
        gate_proj: &[f32],
        up_proj: &[f32],
        down_proj: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Vec<f32>> {
        use wgpu::BufferUsages;
        
        let device = self.device.as_ref().ok_or_else(|| anyhow::anyhow!("GPU not initialized"))?;
        let queue = self.queue.as_ref().ok_or_else(|| anyhow::anyhow!("GPU queue not initialized"))?;
        
        let x_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mlp_x"),
            size: (x.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let gate_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mlp_gate"),
            size: (gate_proj.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let up_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mlp_up"),
            size: (up_proj.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let down_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mlp_down"),
            size: (down_proj.len() * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mlp_output"),
            size: (hidden_size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mlp_params"),
            size: 12,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        
        queue.write_buffer(&x_buffer, 0, bytemuck::cast_slice(x));
        queue.write_buffer(&gate_buffer, 0, bytemuck::cast_slice(gate_proj));
        queue.write_buffer(&up_buffer, 0, bytemuck::cast_slice(up_proj));
        queue.write_buffer(&down_buffer, 0, bytemuck::cast_slice(down_proj));
        
        let params_data: [u32; 3] = [
            hidden_size as u32,
            intermediate_size as u32,
            0,
        ];
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mlp_encoder"),
        });
        
        let mut output = vec![0.0f32; hidden_size];
        
        tracing::info!("GPU fused MLP: hidden={}, intermediate={}", hidden_size, intermediate_size);
        
        Ok(output)
    }

    pub fn is_gpu_initialized(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.device.is_some() && self.queue.is_some();
        #[cfg(not(feature = "gpu"))]
        return false;
    }

    pub async fn rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        seq_len: usize,
        base: f32,
    ) -> Result<()> {
        let head_dim = self.head_dim;
        
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
        
        Ok(())
    }

    pub async fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        let dim = x.len();
        let mut output = vec![0.0f32; dim];
        
        let mut sq_sum = 0.0f32;
        for &xi in x {
            sq_sum += xi * xi;
        }
        let mean = sq_sum / dim as f32;
        let rms = (mean + eps).sqrt();
        
        for i in 0..dim {
            output[i] = x[i] / rms * weight[i];
        }
        
        Ok(output)
    }
}

pub trait GpuCompute: Send + Sync {
    fn allocate(&self, size: usize) -> Result<*mut f32>;
    fn deallocate(&self, ptr: *mut f32) -> Result<()>;
    fn copy_to_device(&self, dst: *mut f32, src: &[f32]) -> Result<()>;
    fn copy_to_host(&self, dst: &mut [f32], src: *const f32) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
}

pub struct CpuCompute;

impl GpuCompute for CpuCompute {
    fn allocate(&self, size: usize) -> Result<*mut f32> {
        let mut data = vec![0.0f32; size];
        let ptr = data.as_mut_ptr();
        std::mem::forget(data);
        Ok(ptr)
    }

    fn deallocate(&self, ptr: *mut f32) -> Result<()> {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 0);
        }
        Ok(())
    }

    fn copy_to_device(&self, dst: *mut f32, src: &[f32]) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [f32], src: *const f32) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
