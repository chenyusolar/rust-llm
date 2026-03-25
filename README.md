# Rust-LLM: High-Performance Hybrid CPU+GPU LLM Inference Engine

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Build](https://github.com/rust-llm/rust-llm/workflows/CI/badge.svg)](https://github.com/rust-llm/rust-llm/actions)
[![Performance](https://img.shields.io/badge/performance-benchmark-green.svg)](#performance)
[![API](https://img.shields.io/badge/API-OpenAI%20compatible-blueviolet)](#api-server)

</div>

A high-performance hybrid CPU+GPU LLM inference framework written in Rust, inspired by llama.cpp. Features real GGUF model loading, quantization support, GPU acceleration via WGSL/WebGPU, advanced batching strategies, and an OpenAI-compatible API server.

## Table of Contents

- [Features](#features)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [API Server](#api-server)
- [Architecture](#architecture)
- [Supported Models](#supported-models)
- [Quantization Formats](#quantization-formats)
- [Building](#building)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Inference Engine
- **Real GGUF Model Loading**: Native support for GGUF format with correct tensor offset handling
- **Multiple Quantization Formats**: Q4, Q5, Q6, IQ2, IQ3, NF4, FP8 (E4M3/E5M2)
- **Hybrid CPU+GPU Execution**: Dynamic compute device selection per layer
- **Memory-mapped Loading** (`mmap`): Optional memory-mapped model loading for large models

### Advanced Sampling & Decoding
- **Beam Search**: Multi-path search with length penalty and early stopping
- **Nucleus Sampling (Top-P)**: Probabilistic sampling with cumulative probability cutoff
- **Top-K Sampling**: Constrained token selection
- **Temperature Control**: Adjustable sampling randomness
- **Repetition Penalty**: Prevent token repetition
- **Grammar-Based Sampling** (GBNF): JSON, code, and custom grammar constraints

### GPU Acceleration
- **WGSL Compute Kernels**: Native WebGPU shader-based operations
- **GPU Kernels Include**:
  - Matrix Multiplication (GEMM)
  - Multi-Head Attention
  - Rotary Position Embedding (RoPE)
  - RMSNorm
  - SiLU Activation
  - Fused MLP Layers
- **Flash Attention**: Optimized attention mechanism for longer contexts

### Advanced Model Features
- **LoRA Adapters**: Efficient fine-tuning with low-rank adaptation
- **LoRA Manager**: Dynamic adapter loading and switching
- **FIM (Fill-in-the-Middle)**: Code completion support
- **Sliding Window Attention**: Efficient long-sequence processing
- **Attention Sinks**: Improved handling of long contexts
- **Prefill-Decode Disaggregation**: Optimized batch processing
- **PagedAttention**: KV cache memory optimization (vLLM-style)

### Batching & Scheduling
- **Continuous Batching**: Dynamic batch size adjustment
- **Adaptive Batching**: Intelligent request batching based on latency
- **Dynamic Batching**: MaxWait-based batching for throughput optimization
- **Request-Level Scheduling**: Per-sequence state management

### KV Cache Optimization
- **KV Cache Quantization**: Compress key-value cache
- **KV Cache Offloading**: CPU/GPU memory management
- **Paged KV Cache**: Memory-efficient caching

### Vision Support
- **Vision Transformers**: Image understanding capabilities
- **Multi-modal Models**: Support for vision-language models

### Performance Tools
- **Profiler**: Runtime performance analysis
- **Metrics Collector**: Latency and throughput tracking
- **CUDA Graphs**: Kernel launch optimization

### API Server
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **Endpoints**:
  - `/v1/chat/completions` - Chat completions
  - `/v1/completions` - Text completions
  - `/v1/models` - Model listing
  - `/v1/embeddings` - Text embeddings
- **Continuous Batching Engine**: High-throughput request processing

## Performance

Rust-LLM is designed for maximum performance on consumer hardware:

- **Memory Efficient**: 4-bit quantization allows running 70B+ models on consumer GPUs
- **Low Latency**: Optimized CPU kernels for instant response
- **High Throughput**: Continuous batching maximizes batch size utilization
- **No External Dependencies**: Static linking for consistent performance

### Benchmark Comparison

| Feature | rust-llm | llama.cpp | vLLM |
|---------|----------|-----------|------|
| Native Rust | ✅ | ❌ | ❌ |
| WGSL GPU Kernels | ✅ | ❌ | ❌ |
| WebGPU Support | ✅ | ❌ | ❌ |
| LoRA Support | ✅ | ✅ | ✅ |
| PagedAttention | ✅ | ❌ | ✅ |
| Continuous Batching | ✅ | ❌ | ✅ |
| Grammar Sampling | ✅ | ✅ | ❌ |
| FIM Support | ✅ | ✅ | ❌ |
| Vision Models | ✅ | ❌ | ❌ |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rust-llm/rust-llm.git
cd rust-llm/rust-llm

# Build (with server and GPU support)
cargo build --release --features "server,gpu"

# Or with memory-mapped loading
cargo build --release --features "server,gpu,mmap"
```

### Basic Usage

```bash
# Generate text
./target/release/rust-llm -m model.gguf -p "Hello, how are you?"

# Interactive chat
./target/release/rust-llm -m model.gguf --interactive

# With GPU acceleration
./target/release/rust-llm -m model.gguf --use-gpu -p "Your prompt"
```

## CLI Usage

```
High-performance hybrid CPU+GPU LLM inference engine

Usage: rust-llm [OPTIONS]

Options:
  -m, --model <MODEL>                    Path to GGUF model file
  -p, --prompt <PROMPT>                  Prompt for text generation
      --max-tokens <MAX_TOKENS>          Maximum tokens to generate (default: 512)
      --temperature <TEMPERATURE>        Sampling temperature (0.0-2.0) (default: 1)
      --top-p <TOP_P>                    Nucleus sampling probability (default: 0.9)
      --top-k <TOP_K>                    Top-k sampling (default: 40)
      --repeat-penalty <REPEAT_PENALTY>  Repetition penalty (default: 1.1)
      --use-gpu                          Enable GPU acceleration
      --context-size <CONTEXT_SIZE>       Context window size (default: 2048)
      --batch-size <BATCH_SIZE>         Batch size for inference (default: 1)
  -i, --interactive                      Run in interactive chat mode
      --flash-attention                  Enable Flash Attention
      --kv-cache-quant                   Use KV cache quantization
      --speculative                      Enable speculative decoding
      --beam-width <BEAM_WIDTH>          Beam search width (1 = disabled)
      --server                           Start API server
      --port <PORT>                      API server port (default: 8080)
      --host <HOST>                      API server host
  -h, --help                             Print help
  -V, --version                          Print version
```

### Examples

```bash
# Generate text with custom parameters
rust-llm --model model.gguf --prompt "Hello" --max-tokens 100 --temperature 0.7

# GPU acceleration with Flash Attention
rust-llm --model model.gguf --use-gpu --flash-attention -p "Your prompt"

# Beam search for better quality
rust-llm --model model.gguf --prompt "Hello" --beam-width 4

# Start API server
rust-llm --model model.gguf --server --port 8080
```

## API Server

Start the server:

```bash
rust-llm --model model.gguf --server --port 8080
```

### Endpoints

#### Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### Text Completions

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "prompt": "Once upon a time",
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### List Models

```bash
curl http://localhost:8080/v1/models
```

#### Embeddings

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "input": "The quick brown fox"
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         rust-llm                             │
├─────────────────────────────────────────────────────────────┤
│  CLI / API Server                                           │
│  ├── Chat Completions                                        │
│  ├── Text Completions                                         │
│  ├── Embeddings                                              │
│  └── Model Management                                        │
├─────────────────────────────────────────────────────────────┤
│  Inference Engine                                            │
│  ├── Continuous Batching Engine                              │
│  ├── Dynamic Batcher                                         │
│  └── Request Scheduler                                       │
├─────────────────────────────────────────────────────────────┤
│  Model Execution                                             │
│  ├── Scheduler (CPU/GPU)                                     │
│  ├── Decoder (Beam Search, Sampling)                         │
│  ├── Tokenizer                                               │
│  └── LoRA Manager                                            │
├─────────────────────────────────────────────────────────────┤
│  GPU Kernels (WGSL)                                          │
│  ├── Matrix Multiplication                                   │
│  ├── Multi-Head Attention                                   │
│  ├── RoPE / RMSNorm / SiLU                                  │
│  └── Fused MLP                                               │
├─────────────────────────────────────────────────────────────┤
│  Memory Management                                           │
│  ├── KV Cache (Paged)                                        │
│  ├── KV Offloading                                           │
│  ├── Prompt Cache                                            │
│  └── Memory-mapped Loading                                   │
├─────────────────────────────────────────────────────────────┤
│  Model Loader (GGUF)                                         │
│  └── 30+ Quantization Formats                               │
└─────────────────────────────────────────────────────────────┘
```

## Supported Models

- **LLaMA** / **LLaMA 2** / **LLaMA 3**
- **Mistral** / **Mixtral** (MoE)
- **Phi-3**
- **Qwen** / **Qwen 2**
- **Gemma**
- **Falcon**

## Quantization Formats

| Format | Bits | Description |
|--------|------|-------------|
| Q4_0 | 4 | Basic 4-bit quantization |
| Q4_1 | 4 | 4-bit with separate zero point |
| Q5_0 | 5 | Basic 5-bit quantization |
| Q5_1 | 5 | 5-bit with separate zero point |
| Q8_0 | 8 | 8-bit quantization |
| Q2_K | 2 | 2-bit k-quantization |
| Q3_K_S/M/L | 3 | 3-bit k-quantization (small/medium/large) |
| Q4_K_S/M | 4 | 4-bit k-quantization |
| Q5_K | 5 | 5-bit k-quantization |
| Q6_K | 6 | 6-bit k-quantization |
| IQ2_XXS/XS | 2 | Improved 2-bit quantization |
| IQ3_XS | 3 | Improved 3-bit quantization |
| IQ4_NL | 4 | Improved 4-bit (non-linear) |
| NF4 | 4 | Normal float 4-bit |
| FP8 (E4M3/E5M2) | 8 | 8-bit floating point |

## Building

### Prerequisites

- Rust 1.75+
- For GPU support: WebGPU-compatible GPU (NVIDIA, AMD, Apple Silicon)

### Build Options

```bash
# Basic build
cargo build --release

# With API server
cargo build --release --features server

# With GPU acceleration
cargo build --release --features gpu

# With memory-mapped loading
cargo build --release --features mmap

# Full build (all features)
cargo build --release --features "server,gpu,mmap"
```

### Release Build

```bash
cargo build --release
```

The optimized binary will be at `target/release/rust-llm`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WebGPU support (optional)
rustup +nightly target add wasm32-unknown-unknown

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run -- --help
```

## Roadmap

- [ ] CUDA backend for better NVIDIA performance
- [ ] More model architectures (MPT, StableLM, etc.)
- [ ] Speculative decoding optimization
- [ ] Multi-GPU support
- [ ] Streaming completions
- [ ] Function calling

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Inspiration for quantization and GGUF format
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention inspiration
- [llama-rs](https://github.com/bdd/llama-rs) - Early Rust LLM exploration

---

<div align="center">

**Star us on GitHub if you find this project useful!**

</div>
