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
                    return Self::from_loaded(metadata, weights, use_gpu).await;
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
        
        Self::from_loaded(metadata, weights, use_gpu).await
    }

    async fn from_loaded(metadata: ModelMetadata, weights: HashMap<String, QuantizedTensor>, use_gpu: bool) -> Result<Self> {
        eprintln!("[DEBUG] Loading tokenizer...");
        
        let tokenizer = match Tokenizer::load_from_gguf(Path::new("")) {
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
        
        if tokens.is_empty() {
            return Ok(vec![0.0f32; vocab_size]);
        }
        
        let embd_key = weights.keys().find(|k| k.contains("token_embd") && k.contains("weight"));
        let Some(embd_key) = embd_key else {
            return Ok(vec![0.0f32; vocab_size]);
        };
        
        let embd_tensor = weights.get(embd_key).unwrap();
        let embd_data = embd_tensor.dequantize();
        let embd_dim = embd_tensor.shape.last().copied().unwrap_or(hidden_size);
        
        let last_token = *tokens.last().unwrap_or(&0) as usize;
        let start_idx = last_token * embd_dim;
        
        if start_idx >= embd_data.len() {
            return Ok(vec![0.0f32; vocab_size]);
        }
        
        let mut hidden = embd_data[start_idx..start_idx + embd_dim].to_vec();
        
        for layer_idx in 0..self.metadata.num_layers {
            let layer_prefix = format!("blk.{}.", layer_idx);
            
            let attn_norm_key = weights.keys().find(|k| k.starts_with(&layer_prefix) && k.contains("attn_norm"));
            let ffn_norm_key = weights.keys().find(|k| k.starts_with(&layer_prefix) && k.contains("ffn_norm"));
            
            if let Some(key) = attn_norm_key {
                if let Some(tensor) = weights.get(key) {
                    let norm_weight = tensor.dequantize();
                    hidden = self.rms_norm(&hidden, &norm_weight);
                }
            }
            
            let q = self.compute_attn(&weights, &hidden, layer_idx, "q");
            let k = self.compute_attn(&weights, &hidden, layer_idx, "k");
            let v = self.compute_attn(&weights, &hidden, layer_idx, "v");
            
            let attn_output = self.simple_attention(&q, &k, &v);
            hidden = hidden.iter().zip(attn_output.iter()).map(|(a, b)| a + b).collect();
            
            if let Some(key) = ffn_norm_key {
                if let Some(tensor) = weights.get(key) {
                    let norm_weight = tensor.dequantize();
                    hidden = self.rms_norm(&hidden, &norm_weight);
                }
            }
            
            let ffn_output = self.compute_ffn(&weights, &hidden, layer_idx);
            hidden = hidden.iter().zip(ffn_output.iter()).map(|(a, b)| a + b).collect();
        }
        
        let output_key = weights.keys().find(|k| k.contains("output") && k.contains("weight"));
        if let Some(key) = output_key {
            if let Some(tensor) = weights.get(key) {
                let output_weight = tensor.dequantize();
                let mut logits = vec![0.0f32; vocab_size];
                
                for i in 0..vocab_size.min(output_weight.len() / hidden_size) {
                    let mut sum = 0.0f32;
                    for j in 0..hidden_size.min(output_weight.len()) {
                        sum += hidden[j] * output_weight[i * hidden_size + j];
                    }
                    logits[i] = sum;
                }
                
                return Ok(logits);
            }
        }
        
        Ok(vec![0.0f32; vocab_size])
    }
    
    fn rms_norm(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let size = input.len();
        let rms = (input.iter().map(|x| x * x).sum::<f32>() / size as f32).sqrt();
        let eps = 1e-5;
        input.iter().zip(weight.iter()).map(|(x, w)| x / (rms + eps) * w).collect()
    }
    
    fn compute_attn(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, input: &[f32], layer_idx: usize, attn_type: &str) -> Vec<f32> {
        let prefix = format!("blk.{}.attn.{}", layer_idx, attn_type);
        let key = weights.keys().find(|k| k.starts_with(&prefix) && k.contains("weight"));
        
        if let Some(key) = key {
            if let Some(tensor) = weights.get(key) {
                let weight = tensor.dequantize();
                let hidden_size = self.metadata.hidden_size;
                let head_dim = self.metadata.head_dim;
                let num_heads = self.metadata.num_heads;
                
                let mut output = vec![0.0f32; input.len()];
                for i in 0..num_heads {
                    for j in 0..head_dim {
                        let mut sum = 0.0f32;
                        for k in 0..input.len().min(weight.len() / (num_heads * head_dim)) {
                            sum += input[k] * weight[i * head_dim * input.len() + j * input.len() + k];
                        }
                        output[i * head_dim + j] = sum;
                    }
                }
                return output;
            }
        }
        input.to_vec()
    }
    
    fn simple_attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let head_dim = self.metadata.head_dim;
        let num_heads = self.metadata.num_heads;
        
        if q.is_empty() || k.is_empty() || v.is_empty() {
            return vec![0.0f32; head_dim * num_heads];
        }
        
        let mut output = vec![0.0f32; q.len()];
        
        for i in 0..num_heads {
            let q_start = i * head_dim;
            let k_start = i * head_dim;
            let v_start = i * head_dim;
            
            let q_slice = &q[q_start..q_start + head_dim.min(q.len() - q_start)];
            let k_slice = &k[k_start..k_start + head_dim.min(k.len() - k_start)];
            let v_slice = &v[v_start..v_start + head_dim.min(v.len() - v_start)];
            
            let mut scores = vec![0.0f32; head_dim];
            for j in 0..head_dim {
                scores[j] = q_slice.get(j).unwrap_or(&0.0) * k_slice.get(j).unwrap_or(&0.0);
            }
            
            let max_score = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
            scores.iter_mut().for_each(|s| *s = (*s - max_score).exp());
            
            let sum: f32 = scores.iter().sum();
            if sum > 0.0 {
                scores.iter_mut().for_each(|s| *s /= sum);
            }
            
            for j in 0..head_dim {
                let mut val = 0.0f32;
                for k in 0..head_dim {
                    val += scores[k] * v_slice.get(k).unwrap_or(&0.0);
                }
                output[i * head_dim + j] = val;
            }
        }
        
        output
    }
    
    fn compute_ffn(&self, weights: &std::collections::HashMap<String, QuantizedTensor>, input: &[f32], layer_idx: usize) -> Vec<f32> {
        let prefix = format!("blk.{}.ffn", layer_idx);
        
        let gate_key = weights.keys().find(|k| k.starts_with(&prefix) && k.contains("gate"));
        let up_key = weights.keys().find(|k| k.starts_with(&prefix) && k.contains("up"));
        let down_key = weights.keys().find(|k| k.starts_with(&prefix) && k.contains("down"));
        
        let intermediate = self.metadata.intermediate_size;
        let hidden_size = self.metadata.hidden_size;
        
        let mut gate = vec![0.0f32; intermediate];
        let mut up = vec![0.0f32; intermediate];
        
        if let Some(key) = gate_key {
            if let Some(tensor) = weights.get(key) {
                let w = tensor.dequantize();
                for i in 0..intermediate.min(w.len() / hidden_size) {
                    let mut sum = 0.0f32;
                    for j in 0..input.len().min(w.len() / intermediate) {
                        sum += input[j] * w[i * input.len() + j];
                    }
                    gate[i] = sum;
                }
            }
        }
        
        if let Some(key) = up_key {
            if let Some(tensor) = weights.get(key) {
                let w = tensor.dequantize();
                for i in 0..intermediate.min(w.len() / hidden_size) {
                    let mut sum = 0.0f32;
                    for j in 0..input.len().min(w.len() / intermediate) {
                        sum += input[j] * w[i * input.len() + j];
                    }
                    up[i] = sum;
                }
            }
        }
        
        let silu_gate: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| g * (1.0 / (1.0 + (-*g).exp())) * u).collect();
        
        let mut output = vec![0.0f32; hidden_size];
        
        if let Some(key) = down_key {
            if let Some(tensor) = weights.get(key) {
                let w = tensor.dequantize();
                for i in 0..hidden_size.min(w.len() / intermediate) {
                    let mut sum = 0.0f32;
                    for j in 0..silu_gate.len().min(w.len() / hidden_size) {
                        sum += silu_gate[j] * w[i * silu_gate.len() + j];
                    }
                    output[i] = sum;
                }
            }
        }
        
        output
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
