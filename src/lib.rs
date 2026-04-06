pub mod lru;
pub use lru::LruCache;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use futures::future::join_all;

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub query: String,
    pub embedding: Vec<f32>,
    pub response: String,
}

#[derive(Debug)]
pub struct AppState{
    //entries: Vec<CacheEntry>, // placeholder, later the LRU as cache: LruCache<...,...>
    pub cache: LruCache<String, CacheEntry>,
    pub hit_count: usize,
    pub miss_count: usize,
}

pub type SharedState = Arc<Mutex<AppState>>;

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

#[derive(Deserialize)]
struct BatchLookupRequest {
    requests: Vec<LookupRequest>,
}

#[derive(Serialize)]
struct BatchLookupItem {
    found: bool,
    query: String,
    response: Option<String>,
    similarity: Option<f32>,
    error: Option<String>,
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

fn lookup_best_match(
    cache: &mut LruCache<String, CacheEntry>,
    req: &LookupRequest,
) -> Result<Option<(String, String, f32)>, String> {
    validate_non_empty("query", &req.query).map_err(|(_, e)| e)?;
    validate_embedding(&req.embedding).map_err(|(_, e)| e)?;
    validate_threshold(req.threshold).map_err(|(_, e)| e)?;

    let (best_key, best_response, best_similarity) = {
        let mut best_key: Option<String> = None;
        let mut best_response: Option<String> = None;
        let mut best_similarity = f32::NEG_INFINITY;

        for (key, entry) in cache.iter() {
            let sim = cosine_similarity(&req.embedding, &entry.embedding)?;

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
            cache.touch(&key);
            Ok(Some((key, response, best_similarity)))
        }
        _ => Ok(None),
    }
}

async fn health() -> &'static str {
    "OK"
}


// VALIDATION HELPERS
fn validate_embedding(embedding: &[f32]) -> Result<(), (StatusCode, String)> {
    if embedding.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "embedding must not be empty".to_string()));
    }
    Ok(())
}

fn validate_threshold(threshold: f32) -> Result<(), (StatusCode, String)> {
    if !(-1.0..=1.0).contains(&threshold) {
        return Err((StatusCode::BAD_REQUEST, "threshold must be between -1.0 and 1.0".to_string()));
    }
    Ok(())
}

fn validate_non_empty(field_name: &str, value: &str) -> Result<(), (StatusCode, String)> {
    if value.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, format!("{field_name} must not be empty")));
    }
    Ok(())
}

async fn cache(
    State(state): State<SharedState>,
    Json(req): Json<CacheRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    validate_non_empty("query", &req.query)?;
    validate_non_empty("response", &req.response)?;
    validate_embedding(&req.embedding)?;

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

    // lookup_best_match() owns matching logic and recency update
    // /lookup does HTTP level handling
    match lookup_best_match(&mut s.cache, &req){
        Ok(Some((query, response, similarity))) => {
            s.hit_count += 1;
            Ok(Json(LookupResponse {
                query,
                response,
                similarity,
            }))
        }
        Ok(None) => {
            s.miss_count += 1;
            Err((StatusCode::NOT_FOUND, "no matching cached response".to_string()))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, e))
    }
}

async fn lookup_batch(
    State(state): State<SharedState>,
    Json(req): Json<BatchLookupRequest>,
) -> Result<Json<Vec<BatchLookupItem>>, (StatusCode, String)> {
    let futures = req.requests.into_iter().map(|lookup_req| {
        let state = Arc::clone(&state);
        async move {
            let mut s = match state.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    return BatchLookupItem {
                        found: false,
                        query: lookup_req.query,
                        response: None,
                        similarity: None,
                        error: Some("mutex poisoned".to_string()),
                    }
                }
            };

            match lookup_best_match(&mut s.cache, &lookup_req) {
                Ok(Some((query, response, similarity))) => {
                    s.hit_count += 1;
                    BatchLookupItem {
                        found: true,
                        query,
                        response: Some(response),
                        similarity: Some(similarity),
                        error: None,
                    }
                }
                Ok(None) => {
                    s.miss_count += 1;
                    BatchLookupItem {
                        found: false,
                        query: lookup_req.query,
                        response: None,
                        similarity: None,
                        error: None,
                    }
                }
                Err(e) => BatchLookupItem {
                    found: false,
                    query: lookup_req.query,
                    response: None,
                    similarity: None,
                    error: Some(e),
                },
            }
        }
    });

    Ok(Json(join_all(futures).await))
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


// app-building
pub fn build_app(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/cache", post(cache))
        .route("/lookup", post(lookup))
        .route("/lookup/batch", post(lookup_batch))
        .with_state(state)
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
