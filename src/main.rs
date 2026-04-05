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
    //entries: Vec<CacheEntry>, // placeholder, later the LRU as cache: LruCache<...,...>
    cache: LruCache<String, CacheEntry>,
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

#[derive(Serialize)]
struct StatsResponse {
    hit_count: usize,
    miss_count: usize,
    cache_size: usize,
    hit_rate: f64,
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

    //s.entries.push(entry);
    s.cache.insert(entry.query.clone(), entry);
    Ok(StatusCode::CREATED)
}

async fn lookup(
    State(state): State<SharedState>,
    Json(req): Json<LookupRequest>,
) -> Result<Json<LookupResponse>, (StatusCode, String)> {
    let mut s = state
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "mutex poisoned".to_string()))?;

    let (best_key, best_response, best_similarity) = {
        let mut best_key: Option<String> = None;
        let mut best_response: Option<String> = None;
        let mut best_similarity = f32::NEG_INFINITY;

        for (key, entry) in s.cache.iter() {
            let sim = cosine_similarity(&req.embedding, &entry.embedding)
                .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

            if sim > best_similarity {
                best_similarity = sim;
                best_key = Some(key.clone());
                best_response = Some(entry.response.clone());
            }
        }

        (best_key, best_response, best_similarity)
    };

    match (best_key, best_response) {
        (Some(key), Some(response)) if best_similarity >= req.threshold => {
            s.hit_count += 1;
            s.cache.touch(&key);

            Ok(Json(LookupResponse {
                query: key,
                response,
                similarity: best_similarity,
            }))
        }
        _ => {
            s.miss_count += 1;
            Err((StatusCode::NOT_FOUND, "no matching cached response".to_string()))
        }
    }
}

async fn stats(
    State(state): State<SharedState>,
) -> Result<Json<StatsResponse>, (StatusCode, String)> {
    let s = state
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "mutex poisoned".to_string()))?;

    let total = s.hit_count + s.miss_count;
    let hit_rate = if total == 0 {
        0.0
    } else {
        s.hit_count as f64 / total as f64
    };

    Ok(Json(StatsResponse {
        hit_count: s.hit_count,
        miss_count: s.miss_count,
        cache_size: s.cache.len(),
        hit_rate,
    }))
}

#[tokio::main]
async fn main(){
    let state: SharedState = Arc::new(Mutex::new(AppState {
        //entries: vec![],
        cache: LruCache::new(2),
        hit_count: 0,
        miss_count: 0,
    }));

    // attach shared state to router so handlers extract same shared state
    let app = Router::new()
        .route("/health", get(health))
        .route("/cache", post(cache))
        .route("/lookup", post(lookup))
        .route("/stats", get(stats))
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
