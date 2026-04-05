mod lru;
use lru::LruCache;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
struct CacheEntry {
    query: String,
    embedding: Vec<f32>,
    response: String,
}

#[derive(Debug)]
struct AppState{
    entries: Vec<CacheEntry>, // placeholder, later the LRU as cache: LruCache<...,...>
    hit_count: usize,
    miss_count: usize,
}

type SharedState = Arc<Mutex<AppState>>;

#[derive(Deserialize)]
struct CacheRequest {
    query: String,
    embedding: Vec<f32>,
    response: String,
}

#[derive(Deserialize)]
struct LookupRequest {
    query: String,
    embedding: Vec<f32>,
    threshold: f32,
}

#[derive(Serialize)]
struct LookupResponse {
    query: String,
    response: String,
    similarity: f32,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err("embedding length mismatch".to_string());
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("zero-magnitude vector".to_string());
    }

    Ok(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

async fn health() -> &'static str {
    "OK"
}

async fn cache(
    State(state): State<SharedState>,
    Json(req): Json<CacheRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let entry = CacheEntry {
        query: req.query,
        embedding: req.embedding,
        response: req.response,
    };

    let mut s = state // state is Arc<Mutex<AppState>>
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "mutex poisoned".to_string()))?;

    s.entries.push(entry);
    Ok(StatusCode::CREATED)
}

async fn lookup(
    State(state): State<SharedState>,
    Json(req): Json<LookupRequest>,
) -> Result<Json<LookupResponse>, (StatusCode, String)> {
    let mut s = state
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "mutex poisoned".to_string()))?;

    let mut best_index: Option<usize> = None; //Some(value) or None
    let mut best_similarity = f32::NEG_INFINITY;

    for (i, entry) in s.entries.iter().enumerate() {
        let sim = cosine_similarity(&req.embedding, &entry.embedding)
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

        if sim > best_similarity {
            best_similarity = sim;
            best_index = Some(i);
        }
    }

    match best_index {
        Some(i) if best_similarity >= req.threshold => {
            s.hit_count += 1;
            let entry = &s.entries[i];
            Ok(Json(LookupResponse {
                query: entry.query.clone(),
                response: entry.response.clone(),
                similarity: best_similarity,
            }))
        }
        _ => {
            s.miss_count += 1;
            Err((StatusCode::NOT_FOUND, "no matching cached response".to_string()))
        }
    }
}

#[tokio::main]
async fn main(){
    let state: SharedState = Arc::new(Mutex::new(AppState {
        entries: vec![],
        hit_count: 0,
        miss_count: 0,
    }));

    // attach shared state to router so handlers extract same shared state
    let app = Router::new()
        .route("/health", get(health))
        .route("/cache", post(cache))
        .route("/lookup", post(lookup))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();

    axum::serve(listener,app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn zero_vector_errors() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn mismatched_lengths_error() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }
}
