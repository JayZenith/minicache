pub mod lru;
pub use lru::LruCache;
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};

use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub query: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub inserted_at: Instant,
}

#[derive(Debug)]
pub struct AppState {
    //entries: Vec<CacheEntry>, // placeholder, later the LRU as cache: LruCache<...,...>
    pub cache: LruCache<String, CacheEntry>,
    pub hit_count: usize,
    pub miss_count: usize,
    pub ttl: Duration,
}

// Concurrency choice:
// I use Arc<Mutex<AppState>> rather than Arc<RwLock<AppState>>.
// Although cache lookups may sound read-heavy, LRU get()/touch logic mutates
// recency order, and hit/miss counters also mutate shared state.
// That means the dominant access pattern is effectively write-heavy, so a
// Mutex is simpler and more appropriate here than an RwLock.
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

#[derive(Serialize)]
struct BatchLookupItem {
    found: bool,
    query: String,
    response: Option<String>,
    similarity: Option<f32>,
    error: Option<String>,
}

// ERROR HANDLING
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

type ApiError = (StatusCode, Json<ErrorResponse>);

fn bad_request(message: impl Into<String>) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
}

fn internal_error(message: impl Into<String>) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
}

fn not_found(message: impl Into<String>) -> ApiError {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
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
) -> Result<Option<(String, String, f32)>, ApiError> {
    validate_non_empty("query", &req.query)?;
    validate_embedding(&req.embedding)?;
    validate_threshold(req.threshold)?;

    let (best_key, best_response, best_similarity) = {
        let mut best_key: Option<String> = None;
        let mut best_response: Option<String> = None;
        let mut best_similarity = f32::NEG_INFINITY;

        for (key, entry) in cache.iter() {
            let sim = match cosine_similarity(&req.embedding, &entry.embedding) {
                Ok(sim) => sim,
                Err(_) => continue,
            };

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
fn validate_embedding(embedding: &[f32]) -> Result<(), ApiError> {
    if embedding.is_empty() {
        return Err(bad_request("embedding must not be empty"));
    }
    Ok(())
}

fn validate_threshold(threshold: f32) -> Result<(), ApiError> {
    if !(-1.0..=1.0).contains(&threshold) {
        return Err(bad_request("threshold must be between -1.0 and 1.0"));
    }
    Ok(())
}

fn validate_non_empty(field_name: &str, value: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(bad_request(format!("{field_name} must not be empty")));
    }
    Ok(())
}

async fn cache(
    State(state): State<SharedState>,
    Json(req): Json<CacheRequest>,
) -> Result<StatusCode, ApiError> {
    validate_non_empty("query", &req.query)?;
    validate_non_empty("response", &req.response)?;
    validate_embedding(&req.embedding)?;

    let entry = CacheEntry {
        query: req.query,
        embedding: req.embedding,
        response: req.response,
        inserted_at: Instant::now(),
    };

    let mut s = state.lock().map_err(|_| internal_error("mutex poisoned"))?;

    let ttl = s.ttl;
    prune_expired(&mut s.cache, ttl);

    s.cache.insert(entry.query.clone(), entry);
    Ok(StatusCode::CREATED)
}

async fn lookup(
    State(state): State<SharedState>,
    Json(req): Json<LookupRequest>,
) -> Result<Json<LookupResponse>, ApiError> {
    let mut s = state.lock().map_err(|_| internal_error("mutex poisoned"))?;

    let ttl = s.ttl;
    prune_expired(&mut s.cache, ttl);

    match lookup_best_match(&mut s.cache, &req) {
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
            Err(not_found("no matching cached response"))
        }
        Err(e) => Err(e),
    }
}

async fn lookup_batch(
    State(state): State<SharedState>,
    Json(requests): Json<Vec<LookupRequest>>,
) -> Result<Json<Vec<BatchLookupItem>>, ApiError> {
    let futures = requests.into_iter().map(|lookup_req| {
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
                    };
                }
            };

            let ttl = s.ttl;
            prune_expired(&mut s.cache, ttl);

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
                Err((_, json_err)) => BatchLookupItem {
                    found: false,
                    query: lookup_req.query,
                    response: None,
                    similarity: None,
                    error: Some(json_err.0.error),
                },
            }
        }
    });

    Ok(Json(join_all(futures).await))
}

fn prune_expired(cache: &mut LruCache<String, CacheEntry>, ttl: Duration) {
    let now = Instant::now();

    let expired_keys: Vec<String> = cache
        .iter()
        .filter_map(|(key, entry)| {
            if now.duration_since(entry.inserted_at) > ttl {
                Some(key.clone())
            } else {
                None
            }
        })
        .collect();

    for key in expired_keys {
        let _ = cache.remove(&key);
    }
}

async fn stats(State(state): State<SharedState>) -> Result<Json<StatsResponse>, ApiError> {
    let mut s = state.lock().map_err(|_| internal_error("mutex poisoned"))?;

    let ttl = s.ttl;
    prune_expired(&mut s.cache, ttl);

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
