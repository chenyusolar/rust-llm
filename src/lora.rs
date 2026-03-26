use crate::types::*;
use byteorder::ReadBytesExt;
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

        let path = std::path::Path::new(path);
        if !path.exists() {
            return Err(format!("LoRA adapter path does not exist: {:?}", path));
        }

        // 尝试加载 safetensors 或 bin 格式
        if let Ok(loaded_modules) = self.load_safetensors(path) {
            for (name, module) in loaded_modules {
                self.modules.insert(name, module);
            }
            self.is_active = true;
            tracing::info!("Loaded {} LoRA modules", self.modules.len());
            return Ok(());
        }

        if let Ok(loaded_modules) = self.load_bin(path) {
            for (name, module) in loaded_modules {
                self.modules.insert(name, module);
            }
            self.is_active = true;
            tracing::info!("Loaded {} LoRA modules from bin", self.modules.len());
            return Ok(());
        }

        Err(format!("Failed to load LoRA adapter from: {:?}", path))
    }

    /// 从 safetensors 文件加载
    fn load_safetensors(&self, path: &std::path::Path) -> Result<Vec<(String, LoRAModule)>, String> {
        use std::fs::File;
        use std::io::Read;

        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        let mut modules = Vec::new();

        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                if let Ok(data) = self.read_safetensors_file(&file_path) {
                    modules.extend(data);
                }
            }
        }

        if modules.is_empty() {
            return Err("No safetensors files found".to_string());
        }

        Ok(modules)
    }

    fn read_safetensors_file(&self, path: &std::path::Path) -> Result<Vec<(String, LoRAModule)>, String> {
        use std::fs::File;
        use std::io::{BufReader, Read, Seek, SeekFrom};

        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        let mut header_size_bytes = [0u8; 8];
        reader.read_exact(&mut header_size_bytes)
            .map_err(|e| format!("Failed to read header size: {}", e))?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;

        let mut header = vec![0u8; header_size];
        reader.read_exact(&mut header)
            .map_err(|e| format!("Failed to read header: {}", e))?;

        let header_str = String::from_utf8(header)
            .map_err(|e| format!("Invalid UTF-8 in header: {}", e))?;

        let metadata: serde_json::Value = serde_json::from_str(&header_str)
            .map_err(|e| format!("Failed to parse JSON header: {}", e))?;

        let mut modules = Vec::new();

        if let Some(obj) = metadata.as_object() {
            for (name, tensor_info) in obj {
                if name.ends_with(".weight") || name.contains("lora_") {
                    let tensor_info = tensor_info.as_object()
                        .ok_or_else(|| "Invalid tensor info".to_string())?;

                    let dtype = tensor_info.get("dtype")
                        .and_then(|d| d.as_str())
                        .unwrap_or("F32");

                    let shape = tensor_info.get("shape")
                        .and_then(|s| s.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect::<Vec<_>>())
                        .unwrap_or_default();

                    let data_offsets = tensor_info.get("data_offsets")
                        .and_then(|o| o.as_array())
                        .map(|arr| {
                            let a = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let b = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            (a, b)
                        })
                        .unwrap_or((0, 0));

                    let tensor_size = data_offsets.1 - data_offsets.0;
                    let n_elements = tensor_size / 4; // f32 = 4 bytes

                    if shape.len() == 2 {
                        let rank = shape[0];
                        let in_dim = shape[1];

                        // 读取数据
                        let mut data = vec![0f32; n_elements];
                        let mut file = File::open(path).map_err(|e| format!("Failed to reopen: {}", e))?;
                        use std::io::Read;
                        let mut buf = vec![0u8; tensor_size];
                        file.seek(SeekFrom::Start((data_offsets.0 + header_size + 8) as u64))
                            .map_err(|e| format!("Seek error: {}", e))?;
                        file.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;

                        for (i, chunk) in buf.chunks(4).enumerate().take(n_elements) {
                            if chunk.len() == 4 {
                                data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            }
                        }

                        // LoRA A: [rank, in_dim] -> transpose -> [in_dim, rank]
                        // LoRA B: [out_dim, rank] -> transpose -> [rank, out_dim]
                        let module_name = name.replace(".weight", "");
                        modules.push((module_name.clone(), LoRAModule::from_weights(data.clone(), vec![0.0f32; rank * in_dim], rank)));
                    }
                }
            }
        }

        Ok(modules)
    }

    /// 从 bin 文件加载（简单实现）
    fn load_bin(&self, path: &std::path::Path) -> Result<Vec<(String, LoRAModule)>, String> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        let mut modules = Vec::new();

        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.extension().and_then(|s| s.to_str()) == Some("bin") {
                let file_name = file_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");

                let file = File::open(&file_path).map_err(|e| format!("Failed to open: {}", e))?;
                let mut reader = BufReader::new(file);

                let metadata_len = reader.by_ref().take(4).read_u32::<byteorder::LittleEndian>()
                    .map_err(|e| format!("Failed to read metadata len: {}", e))? as usize;

                let mut metadata_buf = vec![0u8; metadata_len];
                reader.read_exact(&mut metadata_buf).map_err(|e| format!("Failed to read metadata: {}", e))?;

                // 读取剩余数据作为权重
                let mut data = Vec::new();
                reader.read_to_end(&mut data).map_err(|e| format!("Failed to read data: {}", e))?;

                let n_elements = data.len() / 4;
                let mut f32_data = Vec::with_capacity(n_elements);
                for chunk in data.chunks(4).take(n_elements) {
                    if chunk.len() == 4 {
                        f32_data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                }

                let rank = self.config.rank;
                let module_name = file_name.to_string();
                modules.push((module_name, LoRAModule::from_weights(f32_data, vec![0.0f32; rank * 128], rank)));
            }
        }

        if modules.is_empty() {
            return Err("No bin files found".to_string());
        }

        Ok(modules)
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
