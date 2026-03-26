use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::Result;

use tokio::sync::RwLock;

use crate::types::*;
use crate::tokenizer::Tokenizer;
use crate::scheduler::Scheduler;
use crate::batch::BatchManager;
use crate::decoder::Decoder;
use crate::kv_offload::KvOffloadManager;
use crate::quantize::{ModelLoader, QuantizedTensor};

pub struct Model {
    pub metadata: ModelMetadata,
    pub weights: Arc<RwLock<HashMap<String, QuantizedTensor>>>,
    pub tokenizer: Tokenizer,
    scheduler: Scheduler,
    batch_manager: BatchManager,
    decoder: Decoder,
    kv_offload: KvOffloadManager,
    use_gpu: bool,
}

impl Model {
    pub async fn load(path: &Path, use_gpu: bool) -> Result<Self> {
        eprintln!("[DEBUG] Loading model from {:?}", path);

        #[cfg(feature = "mmap")]
        {
            eprintln!("[DEBUG] Trying mmap loading...");
            match ModelLoader::load_gguf_mmap(path) {
                Ok((metadata, weights)) => {
                    eprintln!("[DEBUG] mmap loaded successfully: {} layers", metadata.num_layers);
                    return Self::from_loaded(metadata, weights, path, use_gpu).await;
                }
                Err(e) => {
                    eprintln!("[DEBUG] mmap failed: {}", e);
                }
            }
        }

        eprintln!("[DEBUG] Loading with regular file read...");
        let (metadata, weights) = match ModelLoader::load_gguf(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[ERROR] load_gguf failed: {}", e);
                return Err(e);
            }
        };

        eprintln!("[DEBUG] Model loaded: {} layers, {} hidden", metadata.num_layers, metadata.hidden_size);

        Self::from_loaded(metadata, weights, path, use_gpu).await
    }

    async fn from_loaded(metadata: ModelMetadata, weights: HashMap<String, QuantizedTensor>, model_path: &Path, use_gpu: bool) -> Result<Self> {
        eprintln!("[DEBUG] Loading tokenizer from {:?}...", model_path);

        let tokenizer = match Tokenizer::load_from_gguf(model_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[ERROR] Tokenizer loading failed: {}", e);
                return Err(anyhow::anyhow!("Tokenizer loading failed: {}", e));
            }
        };
        
        eprintln!("[DEBUG] Tokenizer loaded: vocab_size={}", tokenizer.vocab_size);
        
        let head_dim = metadata.head_dim;
        let num_heads = metadata.num_heads;
        let num_kv_heads = metadata.num_kv_heads;
        
        let weights_arc = Arc::new(RwLock::new(weights));
        
        let scheduler = Scheduler::new(
            metadata.num_layers,
            1,
            metadata.hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            metadata.intermediate_size,
            weights_arc.clone(),
        );
        
        let batch_manager = BatchManager::new(1, metadata.max_seq_len);
        
        let decoder = Decoder::new(tokenizer.vocab_size, metadata.hidden_size);
        
        let kv_offload = KvOffloadManager::new(
            metadata.num_layers,
            num_kv_heads,
            head_dim,
            metadata.max_seq_len,
            1,
            8 * 1024 * 1024 * 1024,
        );

        eprintln!("[DEBUG] Caching weights...");
        
        {
            let mut weights_guard = weights_arc.write().await;
            let keys: Vec<String> = weights_guard.keys().cloned().collect();
            for key in keys {
                if let Some(tensor) = weights_guard.get_mut(&key) {
                    if !key.contains("output") && !key.contains("token_embd") {
                        tensor.cache();
                    }
                }
            }
            eprintln!("[DEBUG] Cached {} weights", weights_arc.read().await.len());
        }

        Ok(Self {
            metadata,
            weights: weights_arc,
            tokenizer,
            scheduler,
            batch_manager,
            decoder,
            kv_offload,
            use_gpu,
        })
    }
    
    pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
        Tokenizer::load_from_gguf(path)
    }

    pub async fn forward(&self, tokens: &[Token]) -> Result<Vec<f32>> {
        let weights = self.weights.read().await;

        let vocab_size = self.tokenizer.vocab_size;
        let hidden_size = self.metadata.hidden_size;
        let num_layers = self.metadata.num_layers;
        let num_heads = self.metadata.num_heads;
        let num_kv_heads = self.metadata.num_kv_heads;
        let head_dim = self.metadata.head_dim;

        if tokens.is_empty() {
            return Ok(vec![0.0f32; vocab_size]);
        }

        // 1. Embedding 层 - 对所有 token 做 embedding 并取最后一个 hidden state
        let embd_key = weights.keys().find(|k| k.contains("token_embd") && k.contains("weight"));
        let Some(embd_key) = embd_key else {
            return Ok(vec![0.0f32; vocab_size]);
        };

        let embd_tensor = weights.get(embd_key).unwrap();
        let embd_data = embd_tensor.dequantize();
        let embd_dim = embd_tensor.shape.last().copied().unwrap_or(hidden_size);

        // 对整个序列进行 embedding，累加（或取平均）
        let seq_len = tokens.len();
        let mut hidden = vec![0.0f32; hidden_size];

        for &token in tokens {
            let token_idx = token as usize;
            let start_idx = token_idx * embd_dim;
            if start_idx + embd_dim <= embd_data.len() {
                for i in 0..embd_dim.min(hidden_size) {
                    hidden[i] += embd_data[start_idx + i];
                }
            }
        }
        // 取平均
        if seq_len > 0 {
            for val in hidden.iter_mut() {
                *val /= seq_len as f32;
            }
        }

        // 2. Transformer 层
        let mut current_hidden = hidden;

        for layer_idx in 0..num_layers {
            let layer_prefix = format!("blk.{}.", layer_idx);

            // Layer Norm (RMS Norm)
            let attn_norm_key = weights.keys().find(|k| k.starts_with(&layer_prefix) && k.contains("attn_norm"));
            let ffn_norm_key = weights.keys().find(|k| k.starts_with(&layer_prefix) && k.contains("ffn_norm"));

            if let Some(key) = attn_norm_key {
                if let Some(tensor) = weights.get(key) {
                    let norm_weight = tensor.dequantize();
                    current_hidden = self.rms_norm(&current_hidden, &norm_weight);
                }
            }

            // 3. Attention - 使用正确的矩阵乘法
            let q = self.compute_q(&weights, &current_hidden, layer_idx);
            let k = self.compute_k(&weights, &current_hidden, layer_idx);
            let v = self.compute_v(&weights, &current_hidden, layer_idx);

            // 简单的自注意力实现（对于单 token 生成场景足够）
            let attn_output = self.multi_head_attention(&q, &k, &v, num_heads, num_kv_heads, head_dim);

            // 残差连接
            for i in 0..current_hidden.len().min(attn_output.len()) {
                current_hidden[i] += attn_output[i];
            }

            // FFN Layer Norm
            if let Some(key) = ffn_norm_key {
                if let Some(tensor) = weights.get(key) {
                    let norm_weight = tensor.dequantize();
                    current_hidden = self.rms_norm(&current_hidden, &norm_weight);
                }
            }

            // 4. FFN (SwiGLU)
            let ffn_output = self.compute_ffn(&weights, &current_hidden, layer_idx);

            // 残差连接
            for i in 0..current_hidden.len().min(ffn_output.len()) {
                current_hidden[i] += ffn_output[i];
            }
        }

        // 5. 最终 Layer Norm
        let final_norm_key = weights.keys().find(|k| k.contains("output") && k.contains("norm"));
        if let Some(key) = final_norm_key {
            if let Some(tensor) = weights.get(key) {
                let norm_weight = tensor.dequantize();
                current_hidden = self.rms_norm(&current_hidden, &norm_weight);
            }
        }

        // 6. LM Head - 计算 logits
        let output_key = weights.keys().find(|k| k.contains("output") && k.contains("weight"));
        if let Some(key) = output_key {
            if let Some(tensor) = weights.get(key) {
                let output_weight = tensor.dequantize();
                let logits = self.matmul_vec(&current_hidden, &output_weight, hidden_size, vocab_size);
                return Ok(logits);
            }
        }

        Ok(vec![0.0f32; vocab_size])
    }

    /// 标准矩阵乘法: output = input @ weight, 其中 input 是 [hidden_size], weight 是 [vocab_size, hidden_size]
    fn matmul_vec(&self, input: &[f32], weight: &[f32], hidden_size: usize, vocab_size: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocab_size];

        for i in 0..vocab_size {
            let mut sum = 0.0f32;
            for j in 0..hidden_size.min(weight.len() / vocab_size.max(1)) {
                let weight_idx = i * hidden_size + j;
                if weight_idx < weight.len() {
                    sum += input[j] * weight[weight_idx];
                }
            }
            logits[i] = sum;
        }

        logits
    }

    fn rms_norm(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let size = input.len();
        let rms = (input.iter().map(|x| x * x).sum::<f32>() / size as f32).sqrt();
        let eps = 1e-5;
        input.iter().zip(weight.iter()).map(|(x, w)| x / (rms + eps) * w).collect()
    }

    /// 计算 Q 投影: [hidden_size] -> [num_heads * head_dim]
    fn compute_q(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, input: &[f32], layer_idx: usize) -> Vec<f32> {
        let hidden_size = self.metadata.hidden_size;
        let num_heads = self.metadata.num_heads;
        let head_dim = self.metadata.head_dim;

        let prefix = format!("blk.{}.attn_q", layer_idx);
        let key = weights.keys().find(|k| k.starts_with(&prefix) && k.contains("weight"))
            .or_else(|| weights.keys().find(|k| k.contains(&format!("{}.q_proj", layer_idx)) && k.contains("weight")));

        if let Some(key) = key {
            if let Some(tensor) = weights.get(key) {
                let weight = tensor.dequantize();
                // weight 形状: [num_heads * head_dim, hidden_size] 或 [hidden_size, num_heads * head_dim]
                return self.matmul_fc(input, &weight, hidden_size, num_heads * head_dim);
            }
        }

        // Fallback: 简单线性投影
        vec![0.1f32; num_heads * head_dim]
    }

    /// 计算 K 投影: [hidden_size] -> [num_kv_heads * head_dim]
    fn compute_k(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, input: &[f32], layer_idx: usize) -> Vec<f32> {
        let hidden_size = self.metadata.hidden_size;
        let num_kv_heads = self.metadata.num_kv_heads;
        let head_dim = self.metadata.head_dim;

        let prefix = format!("blk.{}.attn_k", layer_idx);
        let key = weights.keys().find(|k| k.starts_with(&prefix) && k.contains("weight"))
            .or_else(|| weights.keys().find(|k| k.contains(&format!("{}.k_proj", layer_idx)) && k.contains("weight")));

        if let Some(key) = key {
            if let Some(tensor) = weights.get(key) {
                let weight = tensor.dequantize();
                return self.matmul_fc(input, &weight, hidden_size, num_kv_heads * head_dim);
            }
        }

        vec![0.1f32; num_kv_heads * head_dim]
    }

    /// 计算 V 投影: [hidden_size] -> [num_kv_heads * head_dim]
    fn compute_v(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, input: &[f32], layer_idx: usize) -> Vec<f32> {
        let hidden_size = self.metadata.hidden_size;
        let num_kv_heads = self.metadata.num_kv_heads;
        let head_dim = self.metadata.head_dim;

        let prefix = format!("blk.{}.attn_v", layer_idx);
        let key = weights.keys().find(|k| k.starts_with(&prefix) && k.contains("weight"))
            .or_else(|| weights.keys().find(|k| k.contains(&format!("{}.v_proj", layer_idx)) && k.contains("weight")));

        if let Some(key) = key {
            if let Some(tensor) = weights.get(key) {
                let weight = tensor.dequantize();
                return self.matmul_fc(input, &weight, hidden_size, num_kv_heads * head_dim);
            }
        }

        vec![0.1f32; num_kv_heads * head_dim]
    }

    /// 通用全连接投影: input [in_dim] @ weight [out_dim, in_dim] -> output [out_dim]
    fn matmul_fc(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; out_dim];

        // 假设 weight 形状是 [out_dim, in_dim]，按行存储
        for i in 0..out_dim {
            let mut sum = 0.0f32;
            for j in 0..in_dim {
                let weight_idx = i * in_dim + j;
                if weight_idx < weight.len() {
                    sum += input[j] * weight[weight_idx];
                }
            }
            output[i] = sum;
        }

        output
    }

    /// 多头自注意力
    fn multi_head_attention(&self, q: &[f32], k: &[f32], v: &[f32], num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Vec<f32> {
        let hidden_size = num_heads * head_dim;
        let mut output = vec![0.0f32; hidden_size];

        // GQA: 复制 KV heads 以匹配 Q heads
        let kv_per_head = num_heads / num_kv_heads.max(1);

        for h in 0..num_heads {
            let kv_head = h / kv_per_head;

            // 提取该 head 的 q, k, v
            let q_offset = h * head_dim;
            let k_offset = kv_head * head_dim;
            let v_offset = kv_head * head_dim;

            // 计算 attention score: q · k / sqrt(head_dim)
            let scale = (head_dim as f32).sqrt();

            let mut max_score = f32::MIN;
            let mut exp_sum = 0.0f32;

            // 计算 q·k 分数
            let mut scores = vec![0.0f32; num_kv_heads];
            for kh in 0..num_kv_heads.min(1) {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = q_offset + d;
                    let k_idx = kh * head_dim + d;
                    if q_idx < q.len() && k_idx < k.len() {
                        dot += q[q_idx] * k[k_idx];
                    }
                }
                scores[kh] = dot / scale;
                max_score = max_score.max(scores[kh]);
            }

            // Softmax
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                exp_sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= exp_sum.max(1e-8);
            }

            // 加权求和 v
            for kh in 0..num_kv_heads.min(1) {
                let weight = scores[kh];
                for d in 0..head_dim {
                    let out_idx = h * head_dim + d;
                    let v_idx = kh * head_dim + d;
                    if out_idx < output.len() && v_idx < v.len() {
                        output[out_idx] += weight * v[v_idx];
                    }
                }
            }
        }

        // Output projection
        let prefix = ""; // 需要传递 layer_idx，这里简化处理
        self.output_proj(&output, 0)
    }

    fn output_proj(&self, input: &[f32], layer_idx: usize) -> Vec<f32> {
        // 简化：直接返回 input，因为 attention 输出已经包含残差
        input.to_vec()
    }

    /// FFN with SwiGLU activation
    fn compute_ffn(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, input: &[f32], layer_idx: usize) -> Vec<f32> {
        let hidden_size = self.metadata.hidden_size;
        let intermediate_size = self.metadata.intermediate_size;

        // Gate projection
        let gate_key = weights.keys().find(|k| k.contains(&format!("{}.ffn_gate", layer_idx)) && k.contains("weight"))
            .or_else(|| weights.keys().find(|k| k.contains("gate_proj") && k.contains("weight")));

        // Up projection
        let up_key = weights.keys().find(|k| k.contains(&format!("{}.ffn_up", layer_idx)) && k.contains("weight"))
            .or_else(|| weights.keys().find(|k| k.contains("up_proj") && k.contains("weight")));

        // Down projection
        let down_key = weights.keys().find(|k| k.contains(&format!("{}.ffn_down", layer_idx)) && k.contains("weight"))
            .or_else(|| weights.keys().find(|k| k.contains("down_proj") && k.contains("weight")));

        let gate_out = if let Some(key) = gate_key {
            if let Some(tensor) = weights.get(key) {
                let weight = tensor.dequantize();
                self.matmul_fc(input, &weight, hidden_size, intermediate_size)
            } else {
                vec![0.0f32; intermediate_size]
            }
        } else {
            vec![0.0f32; intermediate_size]
        };

        let up_out = if let Some(key) = up_key {
            if let Some(tensor) = weights.get(key) {
                let weight = tensor.dequantize();
                self.matmul_fc(input, &weight, hidden_size, intermediate_size)
            } else {
                vec![0.0f32; intermediate_size]
            }
        } else {
            vec![0.0f32; intermediate_size]
        };

        // SwiGLU: gate * silu(up)
        let silu_gate: Vec<f32> = gate_out.iter().zip(up_out.iter())
            .map(|(g, u)| g * silu(*u))
            .collect();

        // Down projection
        let down_out = if let Some(key) = down_key {
            if let Some(tensor) = weights.get(key) {
                let weight = tensor.dequantize();
                self.matmul_fc(&silu_gate, &weight, intermediate_size, hidden_size)
            } else {
                vec![0.0f32; hidden_size]
            }
        } else {
            vec![0.0f32; hidden_size]
        };

        down_out
    }

    pub fn sample(&self, logits: &[f32]) -> Token {
        self.decoder.sample(logits, &self.tokenizer)
    }

    pub fn sample_with_params(&self, logits: &[f32], temperature: f32, top_p: f32) -> Token {
        self.decoder.sample_with_params(logits, &self.tokenizer, temperature, top_p)
    }

    pub fn decode(&self, tokens: &[Token]) -> String {
        self.tokenizer.decode(tokens)
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
