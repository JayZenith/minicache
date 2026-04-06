use minicache::{build_app, AppState, SharedState};
use minicache::lru::LruCache;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    let state: SharedState = Arc::new(Mutex::new(AppState {
        cache: LruCache::new(100),
        hit_count: 0,
        miss_count: 0,
    }));

    let app = build_app(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();

    axum::serve(listener, app).await.unwrap();
}
