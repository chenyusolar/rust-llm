#[allow(unused_imports)]
use crate::tokenizer::Tokenizer;

#[allow(unused_imports)]
use crate::types::*;

use std::net::SocketAddr;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

#[cfg(feature = "server")]
use axum::{
    extract::{Query, State},
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};

#[cfg(feature = "server")]
use crate::inference::ContinuousBatchingEngine;

#[derive(Clone)]
#[cfg(feature = "server")]
pub struct ServerState {
    pub engine: Arc<RwLock<Option<ContinuousBatchingEngine>>>,
    pub model_loaded: Arc<RwLock<bool>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct CompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct TextChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelData>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct ModelData {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct EmbeddingsRequest {
    pub input: Vec<String>,
    pub model: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg(feature = "server")]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[cfg(feature = "server")]
pub async fn chat_completion(
    State(state): State<ServerState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    let prompt = req.messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(1.0);

    let mut engine = state.engine.write().await;
    let eng = engine.as_mut().ok_or(StatusCode::NOT_FOUND)?;

    let result = eng.generate(&prompt, max_tokens, temperature, req.top_p.unwrap_or(0.9)).await;

    match result {
        Ok(output) => {
            let response = ChatCompletionResponse {
                id: rand_id(),
                object: "chat.completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as u64,
                model: req.model,
                choices: vec![CompletionChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: output,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: prompt.len() as i32 / 4,
                    completion_tokens: 0,
                    total_tokens: prompt.len() as i32 / 4,
                },
            };
            Ok(Json(response))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

#[cfg(feature = "server")]
pub async fn completion(
    State(state): State<ServerState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(1.0);

    let mut engine = state.engine.write().await;
    let eng = engine.as_mut().ok_or(StatusCode::NOT_FOUND)?;

    let result = eng.generate(&req.prompt, max_tokens, temperature, 0.9).await;

    match result {
        Ok(output) => {
            let response = CompletionResponse {
                id: rand_id(),
                object: "text_completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as u64,
                model: req.model,
                choices: vec![TextChoice {
                    index: 0,
                    text: output,
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: req.prompt.len() as i32 / 4,
                    completion_tokens: 0,
                    total_tokens: req.prompt.len() as i32 / 4,
                },
            };
            Ok(Json(response))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

#[cfg(feature = "server")]
pub async fn models(
    State(_state): State<ServerState>,
) -> Result<Json<ModelListResponse>, StatusCode> {
    let response = ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelData {
            id: "rust-llm".to_string(),
            object: "model".to_string(),
            owned_by: "rust-llm".to_string(),
        }],
    };
    Ok(Json(response))
}

#[cfg(feature = "server")]
pub async fn embeddings(
    State(_state): State<ServerState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, StatusCode> {
    let text = req.input.join(" ");
    let embedding: Vec<f32> = text.chars().map(|c| c as u32 as f32 / 128.0).collect();

    let response = EmbeddingsResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            embedding,
            index: 0,
        }],
        usage: Usage {
            prompt_tokens: text.len() as i32 / 4,
            completion_tokens: 0,
            total_tokens: text.len() as i32 / 4,
        },
    };
    Ok(Json(response))
}

#[cfg(feature = "server")]
fn rand_id() -> String {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("chatcmpl-{:x}", nanos)
}

#[cfg(feature = "server")]
pub async fn run_server(port: u16, state: ServerState) {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completion))
        .route("/v1/completions", post(completion))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .route("/health", get(health_check))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Starting API server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(feature = "server")]
async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

#[cfg(feature = "server")]
pub struct ApiServer {
    port: u16,
    state: ServerState,
}

#[cfg(feature = "server")]
impl ApiServer {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            state: ServerState {
                engine: Arc::new(RwLock::new(None)),
                model_loaded: Arc::new(RwLock::new(false)),
            },
        }
    }

    pub fn set_model_loaded(&self, loaded: bool) {
        let state = self.state.model_loaded.clone();
        tokio::spawn(async move {
            *state.write().await = loaded;
        });
    }

    pub async fn run(&self) {
        run_server(self.port, self.state.clone()).await;
    }

    pub fn get_state(&self) -> ServerState {
        self.state.clone()
    }
}

#[cfg(not(feature = "server"))]
pub struct ApiServer;

#[cfg(not(feature = "server"))]
impl ApiServer {
    pub fn new(_port: u16) -> Self {
        Self
    }

    pub async fn run(&self) {
        tracing::warn!("Server not enabled. Build with --features server");
    }
}
