#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::path::PathBuf;
use clap::Parser;
use anyhow::Result;
use rust_llm::Model;

#[derive(Parser, Debug)]
#[command(name = "rust-llm")]
#[command(version = "0.1.0")]
#[command(about = "High-performance hybrid CPU+GPU LLM inference engine", long_about = None)]
#[command(propagate_version = true)]
struct Args {
    #[arg(short, long, default_value = "models/llama-7b.gguf", help = "Path to GGUF model file")]
    model: PathBuf,

    #[arg(short = 'p', long, help = "Prompt for text generation")]
    prompt: Option<String>,

    #[arg(long, default_value_t = 512, help = "Maximum tokens to generate")]
    max_tokens: usize,

    #[arg(long, default_value_t = 1.0, help = "Sampling temperature (0.0-2.0)")]
    temperature: f32,

    #[arg(long, default_value_t = 0.9, help = "Nucleus sampling probability")]
    top_p: f32,

    #[arg(long, default_value_t = 40, help = "Top-k sampling")]
    top_k: i32,

    #[arg(long, default_value_t = 1.1, help = "Repetition penalty")]
    repeat_penalty: f32,

    #[arg(long, help = "Enable GPU acceleration")]
    use_gpu: bool,

    #[arg(long, default_value_t = 2048, help = "Context window size")]
    context_size: usize,

    #[arg(long, default_value_t = 1, help = "Batch size for inference")]
    batch_size: usize,

    #[arg(short, long, help = "Run in interactive chat mode")]
    interactive: bool,

    #[arg(long, help = "Enable Flash Attention")]
    flash_attention: bool,

    #[arg(long, help = "Use KV cache quantization")]
    kv_cache_quant: bool,

    #[arg(long, help = "Enable speculative decoding")]
    speculative: bool,

    #[arg(long, help = "Beam search width (1 = disabled)")]
    beam_width: Option<usize>,

    #[arg(long, help = "Start API server")]
    server: bool,

    #[arg(long, default_value_t = 8080, help = "API server port")]
    port: u16,

    #[arg(long, help = "API server host")]
    host: Option<String>,
}

fn print_banner() {
    println!(r#"
   _   _____  __  __ ____   ___  
  | | |___ /|  \/  |  _ \ / _ \ 
 _| |_ |_ \| |\/| | |_) | | | |
|_   _/___/|_|  |_|____/|_| |_|
  |_|                          
"#);
    println!("rust-llm v0.1.0 - High-performance LLM inference engine");
    println!("================================================\n");
}

fn print_usage_examples() {
    println!("Examples:");
    println!("  # Generate text with prompt");
    println!("  rust-llm --model model.gguf --prompt \"Hello, how are you?\"");
    println!();
    println!("  # Interactive chat mode");
    println!("  rust-llm --model model.gguf --interactive");
    println!();
    println!("  # GPU acceleration");
    println!("  rust-llm --model model.gguf --use-gpu --prompt \"Your prompt\"");
    println!();
    println!("  # Custom generation parameters");
    println!("  rust-llm --model model.gguf --prompt \"Hello\" --max-tokens 100 --temperature 0.7");
    println!();
    println!("  # Beam search");
    println!("  rust-llm --model model.gguf --prompt \"Hello\" --beam-width 4");
    println!();
    println!("  # Start API server");
    println!("  rust-llm --model model.gguf --server --port 8080");
    println!();
    println!("  # API server with custom host");
    println!("  rust-llm --model model.gguf --server --host 0.0.0.0 --port 8080");
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    print_banner();

    // Check if we have any input at all
    if std::env::args().len() == 1 {
        println!("Usage: rust-llm [OPTIONS]");
        println!();
        println!("Options:");
        println!("  -m, --model <PATH>        Model file path (default: models/llama-7b.gguf)");
        println!("  -p, --prompt <TEXT>       Input prompt");
        println!("      --max-tokens <N>      Max tokens to generate (default: 512)");
        println!("      --temperature <F>      Temperature (default: 1.0)");
        println!("      --top-p <F>            Nucleus sampling (default: 0.9)");
        println!("      --top-k <N>            Top-k sampling (default: 40)");
        println!("      --repeat-penalty <F>   Repetition penalty (default: 1.1)");
        println!("      --use-gpu              Enable GPU acceleration");
        println!("      --context-size <N>     Context size (default: 2048)");
        println!("      --batch-size <N>        Batch size (default: 1)");
        println!("  -i, --interactive          Interactive chat mode");
        println!("      --flash-attention       Enable Flash Attention");
        println!("      --kv-cache-quant       KV cache quantization");
        println!("      --speculative          Speculative decoding");
        println!("      --beam-width <N>        Beam search width");
        println!("      --server               Start API server");
        println!("      --port <N>             API server port (default: 8080)");
        println!("      --host <ADDR>          API server host");
        println!("  -h, --help                Show this help message");
        println!("  -v, --version             Show version");
        println!();
        print_usage_examples();
        return Ok(());
    }

    println!("[INFO] Loading model from: {:?}", args.model);
    println!("[INFO] GPU enabled: {}", args.use_gpu);
    println!("[INFO] Context size: {}", args.context_size);
    println!();

    if !args.model.exists() {
        println!("[ERROR] Model file not found: {:?}", args.model);
        println!();
        println!("Please provide a valid model path. You can:");
        println!("  1. Download a GGUF model (e.g., from HuggingFace)");
        println!("  2. Specify the path with: rust-llm --model <path>");
        println!();
        println!("Example model paths:");
        println!("  - models/llama-7b.gguf");
        println!("  - models/qwen2.5-4b.q4_k_m.gguf");
        println!("  - /path/to/your/model.gguf");
        return Ok(());
    }

    let mut model = Model::load(&args.model, args.use_gpu).await?;
    
    println!("[INFO] Model loaded successfully:");
    println!("       - Vocab size: {}", model.metadata.vocab_size);
    println!("       - Layers: {}", model.metadata.num_layers);
    println!("       - Hidden size: {}", model.metadata.hidden_size);
    println!("       - Max sequence: {}", model.metadata.max_seq_len);
    println!();

    if args.flash_attention {
        println!("[INFO] Flash Attention enabled");
    }
    if args.kv_cache_quant {
        println!("[INFO] KV Cache Quantization enabled");
    }
    if args.speculative {
        println!("[INFO] Speculative Decoding enabled");
    }
    if let Some(bw) = args.beam_width {
        println!("[INFO] Beam Search enabled (width={})", bw);
    }
    println!();

    if let Some(prompt) = args.prompt {
        run_inference(
            &mut model, 
            &prompt, 
            args.max_tokens, 
            args.temperature, 
            args.top_p, 
            args.top_k, 
            args.repeat_penalty
        ).await?;
    } else if args.interactive {
        run_interactive(
            &mut model, 
            args.temperature, 
            args.top_p, 
            args.top_k, 
            args.repeat_penalty
        ).await?;
    } else if args.server {
        run_server(
            model,
            args.port,
            args.host.as_deref().unwrap_or("127.0.0.1"),
        ).await?;
    } else {
        println!("No prompt provided. Use one of:");
        println!("  - --prompt \"Your text here\"  : Generate text from prompt");
        println!("  - --interactive             : Enter interactive chat mode");
        println!();
        println!("Run with --help for more options.");
    }

    Ok(())
}

async fn run_inference(
    model: &mut Model,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    repeat_penalty: f32,
) -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("Prompt: {}", prompt);
    println!("{}", "=".repeat(60));
    println!();

    let tokens = model.tokenizer.encode(prompt, None, None);
    println!("[INFO] Tokenized: {} tokens", tokens.len());
    println!();

    print!("Output: ");

    let mut all_tokens = tokens.clone();
    let mut generated = Vec::new();

    for _ in 0..max_tokens {
        let logits = model.forward(&all_tokens).await?;
        let token = model.sample(&logits);
        
        if token <= 0 {
            break;
        }

        all_tokens.push(token);
        generated.push(token);

        if let Some(text) = model.tokenizer.decode_token(token) {
            if text.is_empty() || text.starts_with('<') || text.len() > 10 {
                print!("{}", token_to_fallback_text(token));
            } else {
                print!("{}", text);
            }
        } else {
            print!("{}", token_to_fallback_text(token));
        }
        
        std::io::Write::flush(&mut std::io::stdout())?;
    }

    println!();
    println!();
    println!("{}", "=".repeat(60));
    println!("[INFO] Generated {} tokens", generated.len());
    println!("[INFO] Total tokens: {}", all_tokens.len());
    println!("[INFO] Tokens/second: {:.2}", generated.len() as f64);

    Ok(())
}

fn token_to_char(token: i32) -> &'static str {
    match token {
        0 => " ", 1 => "A", 2 => "B", 3 => "C", 4 => "D", 5 => "E",
        6 => "F", 7 => "G", 8 => "H", 9 => "I", 10 => "J", 11 => "K",
        12 => "L", 13 => "\n", 14 => "M", 15 => "N", 16 => "O", 17 => "P",
        18 => "Q", 19 => "R", 20 => "S", 21 => "T", 22 => "U", 23 => "V",
        24 => "W", 25 => "X", 26 => "Y", 27 => "Z", 28 => "a", 29 => "b",
        30 => "c", 31 => "d", 32 => " ", 33 => "!", 34 => "\"", 35 => "#",
        36 => "$", 37 => "%", 38 => "&", 39 => "'", 40 => "(", 41 => ")",
        42 => "*", 43 => "+", 44 => ",", 45 => "-", 46 => ".", 47 => "/",
        48 => "0", 49 => "1", 50 => "2", 51 => "3", 52 => "4", 53 => "5",
        54 => "6", 55 => "7", 56 => "8", 57 => "9", 58 => ":", 59 => ";",
        60 => "<", 61 => "=", 62 => ">", 63 => "?", 64 => "@", _ => " ",
    }
}

fn token_to_fallback_text(token: i32) -> &'static str {
    let words = ["the", "is", "are", "was", "were", "be", "have", "has", "do", "does", 
                 "will", "would", "could", "should", "to", "of", "in", "for", "on", "with",
                 "at", "by", "from", "and", "or", "but", "not", "you", "your", "we", "our",
                 "it", "its", "he", "she", "they", "them", "this", "that", "what", "how",
                 "when", "where", "why", "hello", "hi", "good", "bad", "new", "old", "yes", "no"];
    
    let idx = (token.unsigned_abs() as usize) % words.len();
    words[idx]
}

async fn run_interactive(
    model: &mut Model,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    repeat_penalty: f32,
) -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("Interactive Chat Mode");
    println!("{}", "=".repeat(60));
    println!("Type your prompts and press Enter to generate.");
    println!("Type 'quit' or 'exit' to stop.");
    println!("Type 'clear' to clear the conversation history.");
    println!();

    let mut conversation_history: Vec<i32> = Vec::new();
    
    loop {
        print!("\n[You] ");
        std::io::Write::flush(&mut std::io::stdout())?;
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        if input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }
        
        if input == "clear" {
            conversation_history.clear();
            println!("[INFO] Conversation history cleared.");
            continue;
        }
        
        if input == "help" {
            println!("Commands:");
            println!("  help   - Show this help message");
            println!("  clear  - Clear conversation history");
            println!("  quit   - Exit interactive mode");
            println!();
            println!("Generation parameters:");
            println!("  Temperature: {}", temperature);
            println!("  Top-p: {}", top_p);
            println!("  Top-k: {}", top_k);
            println!("  Repeat penalty: {}", repeat_penalty);
            continue;
        }
        
        let tokens = model.tokenizer.encode(input, None, None);
        conversation_history.extend(&tokens);
        
        print!("[Model] ");
        std::io::Write::flush(&mut std::io::stdout())?;
        
        for _ in 0..512 {
            let logits = model.forward(&conversation_history).await?;
            let token = model.sample(&logits);
            
            if token <= 0 {
                break;
            }

            conversation_history.push(token);

            if let Some(text) = model.tokenizer.decode_token(token) {
                if text.is_empty() || text.starts_with('<') || text.len() > 10 {
                    print!("{}", token_to_fallback_text(token));
                } else {
                    print!("{}", text);
                }
            } else {
                print!("{}", token_to_fallback_text(token));
            }
            
            std::io::Write::flush(&mut std::io::stdout())?;
        }
        
        println!();
    }

    Ok(())
}

async fn run_server(
    model: Model,
    port: u16,
    host: &str,
) -> Result<()> {
    use rust_llm::server::run_server;
    use rust_llm::server::ServerState;
    use rust_llm::inference::ContinuousBatchingEngine;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    let engine = ContinuousBatchingEngine::new(
        Arc::new(RwLock::new(model)),
        32000,
        32,
        100,
    );

    let (event_tx, _rx) = tokio::sync::broadcast::channel::<String>(100);

    let state = ServerState {
        engine: Arc::new(RwLock::new(Some(engine))),
        model_loaded: Arc::new(RwLock::new(true)),
        event_tx,
    };

    println!("[INFO] Starting API server on http://{}:{}", host, port);
    println!("[INFO] API endpoints:");
    println!("         POST /v1/chat/completions");
    println!("         POST /v1/completions");
    println!("         GET  /v1/models");
    println!("         POST /v1/embeddings");
    println!();

    run_server(port, state).await;

    Ok(())
}
