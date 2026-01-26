use anyhow::{anyhow, Result};
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

mod direct_map;
mod sparse_trigram;
mod trigram_iterator;

use sparse_trigram::SparseTrigram;

/// Request payload for generation
#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
}

/// Response payload for generation
#[derive(Serialize)]
struct GenerateResponse {
    text: String,
    tokens: usize,
}

/// Application state
struct AppState {
    model: SparseTrigram,
    tokenizer: Tokenizer,
}

/// Number of tokens to generate in each request
const MAX_TOKENS_PER_REQUEST: usize = 500;

#[derive(Parser)]
#[command(name = "markov-web")]
#[command(about = "Web interface for trigram language model")]
struct Args {
    /// Path to the model file
    #[arg(short, long, default_value = "trigram_model.bin")]
    model: String,

    /// Path to tokenizer file
    #[arg(short, long, default_value = "data/tokenizer.ml.json")]
    tokenizer: String,

    /// Server host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Server port
    #[arg(short, long, default_value = "3000")]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Load model and tokenizer
    println!("Loading model from: {}", args.model);
    let model = SparseTrigram::load(&args.model)?;
    println!("Model loaded with {} trigrams", model.total_trigrams);

    println!("Loading tokenizer from: {}", args.tokenizer);
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
    println!("Tokenizer loaded");

    let state = Arc::new(AppState { model, tokenizer });

    // Build router
    let app = Router::new()
        .route("/", get(serve_index))
        .route("/api/generate", post(generate_handler))
        .nest_service("/static", ServeDir::new("static"))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(1024 * 1024)) // 1MB limit
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    println!("\n🚀 Server running at http://{}", addr);
    println!("   Visit in your browser to use the web interface\n");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Serve the main index.html
async fn serve_index() -> impl IntoResponse {
    match tokio::fs::read_to_string("static/index.html").await {
        Ok(content) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
            content,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "index.html not found").into_response(),
    }
}

/// Generate text and return as JSON
async fn generate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<serde_json::Value>)> {
    let prompt = req.prompt.trim();

    // Validate prompt
    if prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Prompt cannot be empty"})),
        ));
    }

    // Tokenize prompt
    let (mut all_tokens, mut w1, mut w2) = match tokenize_prompt(&state.tokenizer, prompt) {
        Ok(result) => result,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            ));
        }
    };

    // Create RNG
    let mut rng = StdRng::seed_from_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );

    // Generate tokens
    for _ in 0..MAX_TOKENS_PER_REQUEST {
        let next_token = match state.model.sample_next(w1, w2, &mut rng) {
            Some(token) => token,
            None => rng.random_range(0..state.model.vocabulary_size as u32),
        };

        all_tokens.push(next_token);
        w1 = w2;
        w2 = next_token;
    }

    // Decode to text
    let text = match state.tokenizer.decode(&all_tokens, true) {
        Ok(decoded) => decoded,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("Failed to decode tokens: {}", e)})),
            ));
        }
    };

    let tokens = all_tokens.len();

    Ok(Json(GenerateResponse { text, tokens }))
}

/// Tokenize the prompt and extract initial context (w1, w2)
fn tokenize_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<(Vec<u32>, u32, u32)> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow!("Failed to tokenize prompt: {}", e))?;

    let token_ids = encoding.get_ids();

    if token_ids.len() < 2 {
        return Err(anyhow!(
            "Prompt must tokenize to at least 2 tokens. Got {} token(s).",
            token_ids.len()
        ));
    }

    let w1 = token_ids[token_ids.len() - 2];
    let w2 = token_ids[token_ids.len() - 1];

    Ok((token_ids.to_vec(), w1, w2))
}
