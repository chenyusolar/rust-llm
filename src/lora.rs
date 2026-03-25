use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;

pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub scaling: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 8.0,
            scaling: 1.0,
        }
    }
}

impl LoRAConfig {
    pub fn new(rank: usize, alpha: f32) -> Self {
        let scaling = alpha / rank as f32;
        Self {
            rank,
            alpha,
            scaling,
        }
    }
}

pub struct LoRAAdapter {
    config: LoRAConfig,
    modules: HashMap<String, LoRAModule>,
    is_active: bool,
}

impl LoRAAdapter {
    pub fn new(config: LoRAConfig) -> Self {
        Self {
            config,
            modules: HashMap::new(),
            is_active: false,
        }
    }

    pub fn load_from_path(&mut self, path: &str) -> Result<(), String> {
        tracing::info!("Loading LoRA adapter from: {}", path);
        self.is_active = true;
        Ok(())
    }

    pub fn add_module(&mut self, name: String, module: LoRAModule) {
        self.modules.insert(name, module);
    }

    pub fn apply(&self, name: &str, base_weights: &[f32], input: &[f32], output: &mut [f32]) {
        if let Some(module) = self.modules.get(name) {
            self.apply_lora(base_weights, module, input, output);
        } else {
            output.copy_from_slice(base_weights);
        }
    }

    fn apply_lora(
        &self,
        base_weights: &[f32],
        module: &LoRAModule,
        input: &[f32],
        output: &mut [f32],
    ) {
        let rank = self.config.rank;

        let mut lora_a_out = vec![0.0f32; rank];
        for i in 0..rank {
            for j in 0..input.len() {
                lora_a_out[i] += input[j] * module.lora_a[i * input.len() + j];
            }
        }

        let mut lora_out = vec![0.0f32; output.len()];
        for i in 0..output.len() {
            for j in 0..rank {
                lora_out[i] += lora_a_out[j] * module.lora_b[j * output.len() + i];
            }
        }

        for i in 0..output.len() {
            output[i] = base_weights[i] + lora_out[i] * self.config.scaling;
        }
    }

    pub fn merge(&self, name: &str, weights: &mut [f32]) {
        if let Some(module) = self.modules.get(name) {
            let rank = self.config.rank;
            let hidden_dim = weights.len();

            let mut merged = vec![0.0f32; hidden_dim];

            for i in 0..hidden_dim {
                for j in 0..rank {
                    merged[i] += module.lora_a[j * hidden_dim + i]
                        * module.lora_b[j * hidden_dim + i]
                        * self.config.scaling;
                }
                weights[i] += merged[i];
            }
        }
    }

    pub fn is_active(&self) -> bool {
        self.is_active
    }

    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    pub fn activate(&mut self) {
        self.is_active = true;
    }
}

pub struct LoRAModule {
    pub lora_a: Vec<f32>,
    pub lora_b: Vec<f32>,
    pub rank: usize,
}

impl LoRAModule {
    pub fn new(input_dim: usize, rank: usize) -> Self {
        Self {
            lora_a: vec![0.1f32; input_dim * rank],
            lora_b: vec![0.1f32; rank * input_dim],
            rank,
        }
    }

    pub fn from_weights(lora_a: Vec<f32>, lora_b: Vec<f32>, rank: usize) -> Self {
        Self {
            lora_a,
            lora_b,
            rank,
        }
    }
}

pub struct LoRAManager {
    adapters: HashMap<String, Arc<std::sync::RwLock<LoRAAdapter>>>,
    active_adapter: Option<String>,
    max_adapters: usize,
}

impl LoRAManager {
    pub fn new(max_adapters: usize) -> Self {
        Self {
            adapters: HashMap::new(),
            active_adapter: None,
            max_adapters,
        }
    }

    pub fn add_adapter(&mut self, name: String, adapter: LoRAAdapter) -> Result<(), String> {
        if self.adapters.len() >= self.max_adapters {
            return Err("Maximum number of adapters reached".to_string());
        }

        self.adapters
            .insert(name, Arc::new(std::sync::RwLock::new(adapter)));
        Ok(())
    }

    pub fn set_active(&mut self, name: &str) -> Result<(), String> {
        if self.adapters.contains_key(name) {
            self.active_adapter = Some(name.to_string());
            Ok(())
        } else {
            Err(format!("Adapter '{}' not found", name))
        }
    }

    pub fn get_active_adapter(&self) -> Option<Arc<std::sync::RwLock<LoRAAdapter>>> {
        self.active_adapter
            .as_ref()
            .and_then(|name| self.adapters.get(name).cloned())
    }

    pub fn remove_adapter(&mut self, name: &str) -> Result<(), String> {
        if self.adapters.remove(name).is_none() {
            return Err(format!("Adapter '{}' not found", name));
        }

        if self.active_adapter.as_deref() == Some(name) {
            self.active_adapter = None;
        }

        Ok(())
    }

    pub fn apply_to_layer(&self, layer_name: &str, weights: &mut [f32]) {
        if let Some(adapter) = self.get_active_adapter() {
            if let Ok(adapter) = adapter.read() {
                adapter.merge(layer_name, weights);
            }
        }
    }

    pub fn list_adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }
}

pub struct LoRAOptimizer {
    pub rank: usize,
    pub learning_rate: f32,
    pub lora_alpha: f32,
    pub target_modules: Vec<String>,
}

impl Default for LoRAOptimizer {
    fn default() -> Self {
        Self {
            rank: 8,
            learning_rate: 0.0001,
            lora_alpha: 8.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }
    }
}

impl LoRAOptimizer {
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ..Default::default()
        }
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.lora_alpha = alpha;
        self
    }

    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    pub fn create_lora_weights(&self, input_dim: usize) -> LoRAModule {
        LoRAModule::new(input_dim, self.rank)
    }

    pub fn train_step(&self, module: &mut LoRAModule, grad_a: &[f32], grad_b: &[f32]) {
        let lr = self.learning_rate;
        let alpha = self.lora_alpha / self.rank as f32;

        for (i, g) in grad_a.iter().enumerate() {
            module.lora_a[i] -= lr * g * alpha;
        }

        for (i, g) in grad_b.iter().enumerate() {
            module.lora_b[i] -= lr * g * alpha;
        }
    }
}
