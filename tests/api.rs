use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use minicache::{build_app, AppState, SharedState};
use minicache::lru::LruCache;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

fn test_app() -> axum::Router {
    let state: SharedState = Arc::new(Mutex::new(AppState {
        cache: LruCache::new(10),
        hit_count: 0,
        miss_count: 0,
        ttl: Duration::from_secs(1),
    }));
    build_app(state)
}

#[tokio::test]
async fn health_returns_ok() {
    let app = test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn cache_and_lookup_work() {
    let app = test_app();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/cache")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"a","embedding":[1.0,0.0],"response":"A"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/lookup")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"qa","embedding":[1.0,0.0],"threshold":0.8}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn stats_updates_after_hit_and_miss() {
    let app = test_app();

    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/cache")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"a","embedding":[1.0,0.0],"response":"A"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/lookup")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"qa","embedding":[1.0,0.0],"threshold":0.8}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/lookup")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"qb","embedding":[0.0,1.0],"threshold":0.99}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();

    assert!(body_str.contains(r#""hit_count":1"#));
    assert!(body_str.contains(r#""miss_count":1"#));
    assert!(body_str.contains(r#""cache_size":1"#));
}

#[tokio::test]
async fn batch_lookup_returns_mixed_results() {
    let app = test_app();

    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/cache")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"a","embedding":[1.0,0.0],"response":"A"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/lookup/batch")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"requests":[{"query":"x","embedding":[1.0,0.0],"threshold":0.8},{"query":"","embedding":[1.0,0.0],"threshold":0.8}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();

    assert!(body_str.contains(r#""found":true"#));
    assert!(body_str.contains(r#""error":"query must not be empty""#));
}


#[tokio::test]
async fn expired_entry_is_pruned_and_lookup_misses() {
    let app = test_app();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/cache")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"a","embedding":[1.0,0.0],"response":"A"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);

    sleep(Duration::from_secs(2)).await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/lookup")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"query":"a","embedding":[1.0,0.0],"threshold":0.8}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/stats")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();

    assert!(body_str.contains(r#""cache_size":0"#));
}
