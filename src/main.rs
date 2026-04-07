use minicache::lru::LruCache;
use minicache::{AppState, SharedState, build_app};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[tokio::main]
async fn main() {
    let state: SharedState = Arc::new(Mutex::new(AppState {
        cache: LruCache::new(100),
        hit_count: 0,
        miss_count: 0,
        ttl: Duration::from_secs(60),
    }));

    let app = build_app(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    axum::serve(listener, app).await.unwrap();
}
