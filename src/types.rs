pub type Token = i32;
pub type SeqId = i32;
pub type Pos = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    #[allow(non_camel_case_types)]
    Q2_K,
    #[allow(non_camel_case_types)]
    Q3_K_S,
    #[allow(non_camel_case_types)]
    Q3_K_M,
    #[allow(non_camel_case_types)]
    Q3_K_L,
    #[allow(non_camel_case_types)]
    Q4_K_S,
    #[allow(non_camel_case_types)]
    Q4_K_M,
    #[allow(non_camel_case_types)]
    Q4_K,
    #[allow(non_camel_case_types)]
    Q5_K_S,
    #[allow(non_camel_case_types)]
    Q5_K_M,
    #[allow(non_camel_case_types)]
    Q5_K,
    #[allow(non_camel_case_types)]
    Q6_K,
    #[allow(non_camel_case_types)]
    IQ2_XXS,
    #[allow(non_camel_case_types)]
    IQ2_XS,
    #[allow(non_camel_case_types)]
    IQ3_XS,
    #[allow(non_camel_case_types)]
    IQ1_S,
    #[allow(non_camel_case_types)]
    IQ4_NL,
    FP8,
    NF4,
    F8E5m2,
    F8E4m3,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::Q4_0 => 2,
            DataType::Q4_1 => 3,
            DataType::Q5_0 => 3,
            DataType::Q5_1 => 4,
            DataType::Q8_0 => 2,
            DataType::Q2_K => 1,
            DataType::Q3_K_S => 2,
            DataType::Q3_K_M => 3,
            DataType::Q3_K_L => 4,
            DataType::Q4_K_S => 2,
            DataType::Q4_K_M => 3,
            DataType::Q4_K => 3,
            DataType::Q5_K_S => 3,
            DataType::Q5_K_M => 4,
            DataType::Q5_K => 4,
            DataType::Q6_K => 5,
            DataType::IQ2_XXS => 1,
            DataType::IQ2_XS => 2,
            DataType::IQ3_XS => 2,
            DataType::IQ1_S => 1,
            DataType::IQ4_NL => 2,
            DataType::FP8 => 1,
            DataType::NF4 => 1,
            DataType::F8E5m2 => 1,
            DataType::F8E4m3 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabType {
    None,
    SPM,
    BPE,
    WPM,
    UGM,
    RWKV,
    PLAMO2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    None,
    Norm,
    Neox,
    MRoPE,
    IMRoPE,
    Vision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    Unspecified,
    Causal,
    NonCausal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    None,
    Layer,
    Row,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingType {
    Unspecified,
    None,
    Linear,
    Yarn,
    LongRoPE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    CPU,
    GPU,
    GPUIntegrated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    Device(usize),
    CPU,
}

impl ComputeDevice {
    pub fn is_gpu(&self) -> bool {
        matches!(self, ComputeDevice::Device(_))
    }
}

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_type: VocabType,
    pub rope_type: RopeType,
    pub rope_scale: f32,
    pub max_seq_len: usize,
    pub use_parallel_residual: bool,
    pub expert_count: usize,
    pub expert_used_count: usize,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            vocab_size: 0,
            hidden_size: 0,
            intermediate_size: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            vocab_type: VocabType::SPM,
            rope_type: RopeType::Neox,
            rope_scale: 1.0,
            max_seq_len: 2048,
            use_parallel_residual: true,
            expert_count: 0,
            expert_used_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attention_norm: Option<TensorRef>,
    pub attention_norm_scale: Option<TensorRef>,
    pub wq: TensorRef,
    pub wk: TensorRef,
    pub wv: TensorRef,
    pub wo: TensorRef,
    pub bq: Option<TensorRef>,
    pub bk: Option<TensorRef>,
    pub bv: Option<TensorRef>,
    pub bo: Option<TensorRef>,
    pub ffn_norm: Option<TensorRef>,
    pub ffn_gate: TensorRef,
    pub ffn_up: TensorRef,
    pub ffn_down: TensorRef,
    pub ffn_gate_b: Option<TensorRef>,
    pub ffn_up_b: Option<TensorRef>,
    pub ffn_down_b: Option<TensorRef>,
    pub ffn_gate_scale: Option<TensorRef>,
    pub ffn_up_scale: Option<TensorRef>,
    pub ffn_down_scale: Option<TensorRef>,
    pub ffn_gate_inp: Option<TensorRef>,
    pub ffn_experts: Option<Vec<TensorRef>>,
    pub ffn_expert_scale: Option<Vec<TensorRef>>,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorRef {
    pub offset: usize,
    pub shape: (usize, usize),
    pub dtype: DataType,
}

impl TensorRef {
    pub fn element_count(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    pub fn size_bytes(&self) -> usize {
        self.element_count() * self.dtype.size()
    }
}

pub struct EmbeddingTable {
    pub data: Vec<u8>,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub dtype: DataType,
}

impl EmbeddingTable {
    pub fn get_embedding(&self, token: Token) -> Option<&[u8]> {
        if token < 0 || token as usize >= self.vocab_size {
            return None;
        }
        let offset = (token as usize) * self.embedding_dim * self.dtype.size();
        let size = self.embedding_dim * self.dtype.size();
        self.data.get(offset..offset + size)
    }
}

pub struct OutputLayer {
    pub weight: TensorRef,
    pub bias: Option<TensorRef>,
}

pub struct ModelWeights {
    pub embedding: EmbeddingTable,
    pub output: OutputLayer,
    pub layers: Vec<LayerWeights>,
    pub norm: Option<TensorRef>,
    pub rope_freqs: Option<Vec<f32>>,
}

impl ModelWeights {
    pub fn layer_size(&self, layer_idx: usize) -> usize {
        if layer_idx >= self.layers.len() {
            return 0;
        }
        let layer = &self.layers[layer_idx];
        let mut size = 0;
        size += layer.attention_norm.as_ref().map_or(0, |t| t.size_bytes());
        size += layer.wq.size_bytes();
        size += layer.wk.size_bytes();
        size += layer.wv.size_bytes();
        size += layer.wo.size_bytes();
        size += layer.ffn_norm.as_ref().map_or(0, |t| t.size_bytes());
        size += layer.ffn_gate.size_bytes();
        size += layer.ffn_up.size_bytes();
        size += layer.ffn_down.size_bytes();
        size
    }
}
