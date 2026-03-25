use crate::types::*;
use std::collections::HashMap;

pub struct VisionEncoder {
    num_patches: usize,
    patch_size: usize,
    hidden_size: usize,
    num_heads: usize,
    patch_embedding: Vec<f32>,
}

impl VisionEncoder {
    pub fn new(patch_size: usize, hidden_size: usize, num_heads: usize) -> Self {
        let num_patches = (224 / patch_size) * (224 / patch_size);

        let patch_dim = patch_size * patch_size * 3;
        let mut patch_embedding = vec![0.0f32; hidden_size * patch_dim];
        for i in 0..hidden_size.min(patch_dim) {
            patch_embedding[i * patch_dim / hidden_size] = ((i * 31) as f32) / 10000.0;
        }

        Self {
            num_patches,
            patch_size,
            hidden_size,
            num_heads,
            patch_embedding,
        }
    }

    pub fn encode(&self, image: &[f32], width: usize, height: usize) -> Vec<f32> {
        let num_patches_h = width / self.patch_size;
        let num_patches_w = height / self.patch_size;
        let total_patches = num_patches_h * num_patches_w;

        let mut embeddings = Vec::with_capacity((total_patches + 1) * self.hidden_size);

        let cls_token = vec![0.1f32; self.hidden_size];
        embeddings.extend(cls_token);

        for py in 0..num_patches_h {
            for px in 0..num_patches_w {
                let patch_start = (py * self.patch_size * width + px * self.patch_size) * 3;

                let mut patch_embedding = vec![0.0f32; self.hidden_size];

                for i in 0..self.patch_size {
                    for j in 0..self.patch_size {
                        let pixel_idx = patch_start + (i * width + j) * 3;

                        if pixel_idx + 2 < image.len() {
                            let r = image[pixel_idx];
                            let g = image[pixel_idx + 1];
                            let b = image[pixel_idx + 2];

                            for d in 0..self.hidden_size.min(3) {
                                let idx = (i * self.patch_size + j) * 3 + d;
                                if idx < patch_embedding.len() {
                                    patch_embedding[idx] = if d == 0 {
                                        r
                                    } else if d == 1 {
                                        g
                                    } else {
                                        b
                                    };
                                }
                            }
                        }
                    }
                }

                embeddings.extend(patch_embedding);
            }
        }

        embeddings
    }

    pub fn encode_with_projection(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        projection: &[f32],
    ) -> Vec<f32> {
        let patch_embeddings = self.encode(image, width, height);

        let num_patches = patch_embeddings.len() / self.hidden_size;
        let proj_dim = projection.len() / self.hidden_size;

        let mut output = vec![0.0f32; num_patches * proj_dim];

        for i in 0..num_patches {
            let src_start = i * self.hidden_size;
            let patch = &patch_embeddings
                [src_start..src_start + self.hidden_size.min(patch_embeddings.len() - src_start)];

            let dst_start = i * proj_dim;

            for j in 0..proj_dim.min(patch.len()) {
                output[dst_start + j] = patch[j % patch.len()];
            }
        }

        output
    }

    pub fn get_num_patches(&self) -> usize {
        self.num_patches + 1
    }
}

pub struct VisionTransformer {
    config: VisionConfig,
    encoder: VisionEncoder,
    cls_token: Vec<f32>,
    position_embeddings: Vec<f32>,
    layers: Vec<VisionLayer>,
    norm: Vec<f32>,
}

impl VisionTransformer {
    pub fn new(config: VisionConfig) -> Self {
        let encoder = VisionEncoder::new(config.patch_size, config.hidden_size, config.num_heads);

        let num_patches = encoder.get_num_patches();
        let position_embeddings = vec![0.0f32; num_patches * config.hidden_size];

        let head_dim = config.hidden_size / config.num_heads;
        let layers = (0..config.num_layers)
            .map(|_| {
                VisionLayer::new(
                    config.num_heads,
                    head_dim,
                    config.hidden_size,
                    config.hidden_size * 4,
                )
            })
            .collect();

        let norm = vec![1.0f32; config.hidden_size];

        Self {
            config,
            encoder,
            cls_token: vec![0.0f32; 0],
            position_embeddings,
            layers,
            norm,
        }
    }

    pub fn forward(&self, image: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut embeddings = self.encoder.encode(image, width, height);

        let seq_len = embeddings.len() / self.config.hidden_size;

        for i in 0..embeddings.len().min(self.position_embeddings.len()) {
            embeddings[i] += self.position_embeddings[i];
        }

        for layer in &self.layers {
            embeddings = layer.forward(&embeddings);
        }

        let mut output = vec![0.0f32; self.config.hidden_size];
        let patch_count = embeddings.len() / self.config.hidden_size;

        for i in 0..self.config.hidden_size.min(embeddings.len()) {
            let patch_idx = i / self.config.hidden_size;
            let dim_idx = i % self.config.hidden_size;
            if patch_idx < patch_count {
                output[dim_idx] +=
                    embeddings[patch_idx * self.config.hidden_size + dim_idx] * self.norm[dim_idx];
            }
        }

        output
    }
}

pub struct CLIPModel {
    vision: VisionTransformer,
    text_projection: Vec<f32>,
    logit_scale: f32,
    embed_dim: usize,
}

impl CLIPModel {
    pub fn new(vision_config: VisionConfig, embed_dim: usize) -> Self {
        let hidden_size = vision_config.hidden_size;
        let vision = VisionTransformer::new(vision_config);
        let text_projection = vec![0.1f32; hidden_size * embed_dim];

        Self {
            vision,
            text_projection,
            logit_scale: 2.65906,
            embed_dim,
        }
    }

    pub fn encode_image(&self, image: &[f32], width: usize, height: usize) -> Vec<f32> {
        let image_features = self.vision.forward(image, width, height);

        let mut normalized = vec![0.0f32; image_features.len()];
        let mut norm_sq = 0.0f32;
        for &x in &image_features {
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt().max(1e-8);

        for (i, &x) in image_features.iter().enumerate() {
            normalized[i] = x / norm;
        }

        let mut projected = vec![0.0f32; self.embed_dim];
        for i in 0..self.embed_dim.min(normalized.len()) {
            projected[i] = normalized[i]
                * self
                    .text_projection
                    .get(i * normalized.len())
                    .copied()
                    .unwrap_or(0.1);
        }

        projected
    }

    pub fn encode_text(&self, text_embeddings: &[f32]) -> Vec<f32> {
        let input_dim = text_embeddings.len();
        let mut normalized = vec![0.0f32; input_dim];

        let mut norm_sq = 0.0f32;
        for &x in text_embeddings {
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt().max(1e-8);

        for (i, &x) in text_embeddings.iter().enumerate() {
            normalized[i] = x / norm;
        }

        let mut projected = vec![0.0f32; self.embed_dim];
        for i in 0..self.embed_dim.min(normalized.len()) {
            let weight = self
                .text_projection
                .get(i * normalized.len())
                .copied()
                .unwrap_or(0.1);
            projected[i] = normalized[i] * weight;
        }

        projected
    }

    pub fn compute_similarity(&self, image_features: &[f32], text_features: &[f32]) -> f32 {
        let mut similarity = 0.0f32;

        for i in 0..image_features.len().min(text_features.len()) {
            similarity += image_features[i] * text_features[i];
        }

        similarity * self.logit_scale
    }
}

pub struct MultimodalModel {
    vision_config: VisionConfig,
    language_hidden_size: usize,
    vision_projection: Vec<f32>,
    clip: Option<CLIPModel>,
}

impl MultimodalModel {
    pub fn new(vision_config: VisionConfig, language_hidden_size: usize) -> Self {
        let vision_projection = vec![0.1f32; vision_config.projection_dim * language_hidden_size];

        Self {
            vision_config,
            language_hidden_size,
            vision_projection,
            clip: None,
        }
    }

    pub fn init_clip(&mut self) {
        self.clip = Some(CLIPModel::new(
            self.vision_config.clone(),
            self.language_hidden_size,
        ));
    }

    pub fn project_vision_features(&self, vision_features: &[f32]) -> Vec<f32> {
        let num_features = vision_features.len() / self.vision_config.projection_dim;
        let mut projected = vec![0.0f32; num_features * self.language_hidden_size];

        for i in 0..num_features {
            let src_start = i * self.vision_config.projection_dim;
            let src = &vision_features[src_start
                ..src_start
                    + self
                        .vision_config
                        .projection_dim
                        .min(vision_features.len() - src_start)];

            let dst_start = i * self.language_hidden_size;

            for j in 0..self.language_hidden_size.min(src.len() * 2) {
                let src_idx = j % src.len();
                let weight = self.vision_projection.get(j).copied().unwrap_or(0.1);
                projected[dst_start + j] = src[src_idx] * weight;
            }
        }

        projected
    }

    pub fn forward(
        &self,
        image: &[f32],
        image_width: usize,
        image_height: usize,
        text_embeddings: &[f32],
    ) -> Vec<f32> {
        if let Some(ref clip) = self.clip {
            let image_features = clip.encode_image(image, image_width, image_height);
            let projected_vision = self.project_vision_features(&image_features);

            let mut combined = projected_vision;
            combined.extend_from_slice(text_embeddings);

            combined
        } else {
            let mut combined = vec![0.0f32; self.language_hidden_size];
            combined.extend_from_slice(text_embeddings);
            combined
        }
    }
}

pub struct VisionModel {
    encoder: VisionEncoder,
    projection: Vec<f32>,
    config: VisionConfig,
    layers: Vec<VisionLayer>,
}

impl VisionModel {
    pub fn new(config: VisionConfig) -> Self {
        let encoder = VisionEncoder::new(config.patch_size, config.hidden_size, config.num_heads);

        let projection = vec![0.1f32; config.hidden_size * config.projection_dim];

        let head_dim = config.hidden_size / config.num_heads;
        let layers = (0..config.num_layers)
            .map(|_| {
                VisionLayer::new(
                    config.num_heads,
                    head_dim,
                    config.hidden_size,
                    config.hidden_size * 4,
                )
            })
            .collect();

        Self {
            encoder,
            projection,
            config,
            layers,
        }
    }

    pub fn encode_image(&self, image: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut patch_embeddings = self.encoder.encode(image, width, height);

        for layer in &self.layers {
            patch_embeddings = layer.forward(&patch_embeddings);
        }

        let num_patches = patch_embeddings.len() / self.config.hidden_size;
        let mut output = vec![0.0f32; num_patches * self.config.projection_dim];

        for i in 0..num_patches.min(patch_embeddings.len() / self.config.hidden_size) {
            let src_start = i * self.config.hidden_size;
            let patch = &patch_embeddings[src_start
                ..src_start
                    + self
                        .config
                        .hidden_size
                        .min(patch_embeddings.len() - src_start)];

            let dst_start = i * self.config.projection_dim;

            for j in 0..self.config.projection_dim.min(patch.len()) {
                output[dst_start + j] = patch[j]
                    * self
                        .projection
                        .get(i * self.config.projection_dim + j)
                        .copied()
                        .unwrap_or(0.1);
            }
        }

        output
    }

    pub fn multimodal_forward(
        &self,
        image_embeddings: &[f32],
        text_tokens: &[Token],
        text_embeddings: &[f32],
    ) -> Vec<f32> {
        let image_features = image_embeddings.len() / self.config.projection_dim;
        let text_features = text_embeddings.len() / self.config.hidden_size;

        let total_features = 1 + image_features + text_features;
        let hidden_size = self.config.hidden_size;

        let mut combined = Vec::with_capacity(total_features * hidden_size);

        combined.extend(vec![0.1f32; hidden_size]);

        for i in 0..image_features {
            let src_start = i * self.config.projection_dim;
            let src_end = src_start + self.config.projection_dim;
            let img_emb = &image_embeddings[src_start..src_end];

            let mut projected = vec![0.0f32; hidden_size];
            for j in 0..hidden_size.min(img_emb.len()) {
                projected[j] = img_emb[j % img_emb.len()];
            }
            combined.extend(projected);
        }

        combined.extend_from_slice(text_embeddings);

        combined
    }
}

pub struct VisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub projection_dim: usize,
    pub attention_dropout: f32,
    pub hidden_act: String,
}

impl Clone for VisionConfig {
    fn clone(&self) -> Self {
        Self {
            image_size: self.image_size,
            patch_size: self.patch_size,
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            projection_dim: self.projection_dim,
            attention_dropout: self.attention_dropout,
            hidden_act: self.hidden_act.clone(),
        }
    }
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            hidden_size: 768,
            num_heads: 12,
            num_layers: 12,
            projection_dim: 512,
            attention_dropout: 0.0,
            hidden_act: "gelu".to_string(),
        }
    }
}

impl VisionConfig {
    pub fn from_gguf(metadata: &HashMap<String, String>) -> Self {
        let mut config = Self::default();

        for (key, _) in metadata {
            if key.contains("vision") || key.contains("image") {
                if let Some(val) = extract_number(key) {
                    if key.contains("image_size") {
                        config.image_size = val;
                    } else if key.contains("patch_size") {
                        config.patch_size = val;
                    } else if key.contains("hidden_size") {
                        config.hidden_size = val;
                    } else if key.contains("num_layers") || key.contains("n_layers") {
                        config.num_layers = val;
                    } else if key.contains("projection_dim") {
                        config.projection_dim = val;
                    }
                }
            }
        }

        config
    }
}

fn extract_number(s: &str) -> Option<usize> {
    s.chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

pub struct MultiModalProjector {
    vision_hidden_size: usize,
    language_hidden_size: usize,
    projector_weights: Vec<f32>,
}

impl MultiModalProjector {
    pub fn new(vision_hidden_size: usize, language_hidden_size: usize) -> Self {
        Self {
            vision_hidden_size,
            language_hidden_size,
            projector_weights: Vec::new(),
        }
    }

    pub fn project(&self, vision_features: &[f32]) -> Vec<f32> {
        let num_features = vision_features.len() / self.vision_hidden_size;
        let mut projected = Vec::with_capacity(num_features * self.language_hidden_size);

        for i in 0..num_features {
            let src_start = i * self.vision_hidden_size;
            let src_end = src_start + self.vision_hidden_size;
            let src = &vision_features[src_start..src_end];

            let mut dst = vec![0.0f32; self.language_hidden_size];

            for j in 0..self.language_hidden_size.min(src.len()) {
                dst[j] = src[j] * 0.1;
            }

            projected.extend(dst);
        }

        projected
    }
}

pub struct VisionAttention {
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let seq_len = q.len() / self.head_dim;
        let hidden_size = seq_len * self.head_dim;

        let mut output = vec![0.0f32; hidden_size];

        for h in 0..self.num_heads {
            let q_offset = h * self.head_dim;
            let k_offset = h * self.head_dim;
            let v_offset = h * self.head_dim;

            let mut scores = vec![0.0f32; seq_len];
            let mut max_score = f32::MIN;

            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..self.head_dim {
                    dot += q[q_offset + d] * k[k_offset + j * self.head_dim + d];
                }
                dot /= (self.head_dim as f32).sqrt();
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
                for d in 0..self.head_dim {
                    output[q_offset + j * self.head_dim + d] +=
                        scores[j] * v[v_offset + j * self.head_dim + d];
                }
            }
        }

        output
    }
}

pub struct VisionMLP {
    hidden_size: usize,
    intermediate_size: usize,
}

impl VisionMLP {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut intermediate = vec![0.0f32; self.intermediate_size];

        for i in 0..self.intermediate_size.min(x.len()) {
            let mut sum = 0.0f32;
            for j in 0..self.hidden_size.min(x.len()) {
                let weight = ((i * 17 + j) % 1000) as f32 / 10000.0;
                sum += x[j] * weight;
            }
            intermediate[i] = sum;
        }

        let mut output = vec![0.0f32; self.hidden_size];

        for i in 0..self.hidden_size {
            let mut sum = 0.0f32;
            for j in 0..self.intermediate_size.min(intermediate.len()) {
                let weight = ((i * 31 + j) % 1000) as f32 / 10000.0;
                sum += intermediate[j] * weight;
            }
            output[i] = gelu(sum);
        }

        output
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x / (2.0_f32).sqrt()).tanh())
}

pub struct VisionLayer {
    attention: VisionAttention,
    mlp: VisionMLP,
    layernorm1: Vec<f32>,
    layernorm2: Vec<f32>,
}

impl VisionLayer {
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Self {
        Self {
            attention: VisionAttention::new(num_heads, head_dim),
            mlp: VisionMLP::new(hidden_size, intermediate_size),
            layernorm1: vec![1.0f32; hidden_size],
            layernorm2: vec![1.0f32; hidden_size],
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let hidden_size = self.layernorm1.len();

        let mut normed = vec![0.0f32; x.len()];
        for i in 0..x.len().min(hidden_size) {
            normed[i] = x[i] * self.layernorm1[i];
        }

        let attn_output = self.attention.forward(&normed, &normed, &normed);

        let mut residual = vec![0.0f32; x.len()];
        for i in 0..x.len().min(attn_output.len()) {
            residual[i] = x[i] + attn_output[i];
        }

        let mut normed2 = vec![0.0f32; residual.len()];
        for i in 0..residual.len().min(hidden_size) {
            normed2[i] = residual[i] * self.layernorm2[i];
        }

        let mlp_output = self.mlp.forward(&normed2);

        let mut output = vec![0.0f32; residual.len()];
        for i in 0..residual.len().min(mlp_output.len()) {
            output[i] = residual[i] + mlp_output[i];
        }

        output
    }
}
