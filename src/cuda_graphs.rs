#[allow(unused_imports)]
use crate::types::*;

#[allow(unused_imports)]
use anyhow::Result;

#[cfg(feature = "gpu")]
use wgpu::*;

pub struct CudaGraph {
    pub id: usize,
    graph_nodes: Vec<GraphNode>,
}

#[derive(Clone)]
pub struct GraphNode {
    pub node_type: NodeType,
    pub inputs: Vec<TensorRef>,
    pub output: TensorRef,
    pub params: NodeParams,
}

#[derive(Clone)]
pub enum NodeType {
    Matmul,
    Attention,
    Softmax,
    RmsNorm,
    Silu,
    Add,
    Mul,
    Cast,
}

#[derive(Clone)]
pub struct NodeParams {
    pub shape: Option<(usize, usize, usize)>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
}

impl CudaGraph {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            graph_nodes: Vec::new(),
        }
    }

    pub fn add_matmul(&mut self, a: TensorRef, b: TensorRef, output: TensorRef) {
        self.graph_nodes.push(GraphNode {
            node_type: NodeType::Matmul,
            inputs: vec![a, b],
            output: output.clone(),
            params: NodeParams {
                shape: None,
                alpha: Some(1.0),
                beta: Some(0.0),
            },
        });
    }

    pub fn add_attention(&mut self, q: TensorRef, k: TensorRef, v: TensorRef, output: TensorRef) {
        self.graph_nodes.push(GraphNode {
            node_type: NodeType::Attention,
            inputs: vec![q, k, v],
            output: output.clone(),
            params: NodeParams::default(),
        });
    }

    pub fn add_rms_norm(&mut self, input: TensorRef, weight: TensorRef, output: TensorRef, eps: f32) {
        self.graph_nodes.push(GraphNode {
            node_type: NodeType::RmsNorm,
            inputs: vec![input, weight],
            output: output.clone(),
            params: NodeParams {
                shape: None,
                alpha: Some(eps),
                beta: None,
            },
        });
    }

    pub fn add_silu(&mut self, input: TensorRef, output: TensorRef) {
        self.graph_nodes.push(GraphNode {
            node_type: NodeType::Silu,
            inputs: vec![input],
            output: output.clone(),
            params: NodeParams::default(),
        });
    }

    pub fn add_add(&mut self, a: TensorRef, b: TensorRef, output: TensorRef) {
        self.graph_nodes.push(GraphNode {
            node_type: NodeType::Add,
            inputs: vec![a, b],
            output: output.clone(),
            params: NodeParams::default(),
        });
    }

    pub fn add_cast(&mut self, input: TensorRef, output: TensorRef, dtype: DataType) {
        self.graph_nodes.push(GraphNode {
            node_type: NodeType::Cast,
            inputs: vec![input],
            output: output.clone(),
            params: NodeParams {
                shape: None,
                alpha: None,
                beta: None,
            },
        });
    }

    pub fn num_nodes(&self) -> usize {
        self.graph_nodes.len()
    }

    pub fn estimate_flops(&self) -> usize {
        let mut flops = 0;
        
        for node in &self.graph_nodes {
            match node.node_type {
                NodeType::Matmul => {
                    if let Some((m, k, n)) = node.params.shape {
                        flops += 2 * m * k * n;
                    }
                }
                NodeType::Attention => {
                    flops += 4 * 4096 * 4096;
                }
                NodeType::RmsNorm => {
                    flops += 4096 * 2;
                }
                _ => {}
            }
        }
        
        flops
    }
}

impl Default for NodeParams {
    fn default() -> Self {
        Self {
            shape: None,
            alpha: None,
            beta: None,
        }
    }
}

pub struct GraphExecutor {
    graphs: Vec<CudaGraph>,
    max_graphs: usize,
    current_graph_id: usize,
}

impl GraphExecutor {
    pub fn new(max_graphs: usize) -> Self {
        Self {
            graphs: Vec::with_capacity(max_graphs),
            max_graphs,
            current_graph_id: 0,
        }
    }

    pub fn capture_graph(&mut self) -> &mut CudaGraph {
        if self.graphs.len() >= self.max_graphs {
            self.graphs.remove(0);
        }
        
        let graph = CudaGraph::new(self.current_graph_id);
        self.current_graph_id += 1;
        self.graphs.push(graph);
        self.graphs.last_mut().unwrap()
    }

    pub fn capture_inference_step(&mut self, num_layers: usize) -> &mut CudaGraph {
        let graph = self.capture_graph();
        
        let hidden_size = 4096;
        let intermediate_size = 11008;
        let head_dim = 128;
        
        for layer_idx in 0..num_layers {
            let input_ref = TensorRef {
                offset: 0,
                shape: (hidden_size, hidden_size),
                dtype: DataType::F16,
            };
            
            let q_ref = TensorRef {
                offset: layer_idx * hidden_size * head_dim,
                shape: (hidden_size, head_dim),
                dtype: DataType::F16,
            };
            
            let k_ref = TensorRef {
                offset: layer_idx * hidden_size * head_dim + 1024,
                shape: (hidden_size / 4, head_dim),
                dtype: DataType::F16,
            };
            
            let v_ref = TensorRef {
                offset: layer_idx * hidden_size * head_dim + 2048,
                shape: (hidden_size / 4, head_dim),
                dtype: DataType::F16,
            };
            
            let attn_output_ref = TensorRef {
                offset: layer_idx * hidden_size,
                shape: (hidden_size, hidden_size),
                dtype: DataType::F16,
            };
            
            graph.add_matmul(input_ref.clone(), q_ref.clone(), q_ref.clone());
            graph.add_matmul(input_ref.clone(), k_ref.clone(), k_ref.clone());
            graph.add_matmul(input_ref.clone(), v_ref.clone(), v_ref.clone());
            graph.add_attention(q_ref, k_ref, v_ref, attn_output_ref.clone());
            
            let ffn_input_ref = TensorRef {
                offset: layer_idx * hidden_size * 2,
                shape: (hidden_size, intermediate_size),
                dtype: DataType::F16,
            };
            
            let ffn_output_ref = TensorRef {
                offset: layer_idx * hidden_size * 3,
                shape: (hidden_size, intermediate_size),
                dtype: DataType::F16,
            };
            
            graph.add_matmul(attn_output_ref.clone(), ffn_input_ref.clone(), ffn_input_ref.clone());
            graph.add_silu(ffn_input_ref.clone(), ffn_output_ref.clone());
            
            let final_output_ref = TensorRef {
                offset: layer_idx * hidden_size * 4,
                shape: (hidden_size, hidden_size),
                dtype: DataType::F16,
            };
            
            graph.add_matmul(ffn_output_ref.clone(), final_output_ref.clone(), final_output_ref.clone());
            graph.add_add(attn_output_ref, final_output_ref.clone(), final_output_ref.clone());
        }
        
        graph
    }

    pub fn get_graph(&self, id: usize) -> Option<&CudaGraph> {
        self.graphs.iter().find(|g| g.id == id)
    }

    pub fn clear(&mut self) {
        self.graphs.clear();
    }
}

pub struct CudaGraphRunner {
    #[cfg(feature = "gpu")]
    device: Option<Device>,
    #[cfg(feature = "gpu")]
    queue: Option<Queue>,
    executor: GraphExecutor,
}

impl CudaGraphRunner {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "gpu")]
            device: None,
            #[cfg(feature = "gpu")]
            queue: None,
            executor: GraphExecutor::new(16),
        }
    }

    #[cfg(feature = "gpu")]
    pub async fn init(&mut self, adapter: &Adapter) -> Result<()> {
        let (device, queue) = adapter.request_device(&DeviceDescriptor {
            required_features: Features::GRAPHICS_PROTOTYPES,
            required_limits: Limits::default(),
            label: Some("cuda-graph-runner"),
        }, None).await?;
        
        self.device = Some(device);
        self.queue = Some(queue);
        
        tracing::info!("CUDA Graph runner initialized");
        Ok(())
    }

    pub fn capture_and_execute<F>(&mut self, num_layers: usize, execute_fn: F)
    where
        F: Fn(),
    {
        let _graph = self.executor.capture_inference_step(num_layers);
        
        execute_fn();
    }

    pub fn estimate_speedup(&self) -> f32 {
        let total_flops: usize = self.executor.graphs
            .iter()
            .map(|g| g.estimate_flops())
            .sum();
        
        if total_flops > 0 {
            1.2
        } else {
            1.0
        }
    }
}

impl Default for CudaGraphRunner {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GraphPool {
    available: Vec<usize>,
    in_use: Vec<usize>,
    max_size: usize,
}

impl GraphPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            available: (0..max_size).collect(),
            in_use: Vec::new(),
            max_size,
        }
    }

    pub fn acquire(&mut self) -> Option<usize> {
        self.available.pop().map(|id| {
            self.in_use.push(id);
            id
        })
    }

    pub fn release(&mut self, id: usize) {
        if let Some(pos) = self.in_use.iter().position(|&x| x == id) {
            self.in_use.remove(pos);
            self.available.push(id);
        }
    }

    pub fn available_count(&self) -> usize {
        self.available.len()
    }
}
