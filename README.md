# minicache

Async semantic cache server in Rust using `axum`.

Stores query/response pairs with embeddings, supports semantic lookup via cosine similarity, maintains an in-memory LRU cache, exposes metrics, supports batch lookup, and lazy TTL expiration.

## Build and Run

```bash
cargo build
cargo run
```

Server runs on:
```bash
0.0.0.0:3000
```

## API
- POST /cache — store (query, embedding, response)
- POST /lookup — semantic match via cosine similarity (thresholded)
- POST /lookup/batch — concurrent lookup (join_all)
- GET /health — liveness
- GET /stats — hit/miss and cache size

## Spec Compliance
- LRU Cache: O(1) get/insert/evict via HashMap and DLL
- Cosine Similarity: manual implementation with zero-vector and length checks
- Async API: axum and tokio, JSON via serde
- Concurrency: Arc<Mutex<AppState>> (write-heavy workload)
- Error Handling: proper HTTP status codes (400/404/500)

## Example Requests
### Cache
```bash
curl -i -X POST http://127.0.0.1:3000/cache \
-H "Content-Type: application/json" \
-d '{"query":"hello","embedding":[1.0,0.0],"response":"world"}'
```

### Lookup
```bash
curl -i -X POST http://127.0.0.1:3000/lookup \
-H "Content-Type: application/json" \
-d '{"query":"hi","embedding":[1.0,0.0],"threshold":0.8}'
```

### Batch Lookup
```bash
curl -i -X POST http://127.0.0.1:3000/lookup/batch \
-H "Content-Type: application/json" \
-d '[{"query":"q1","embedding":[1.0,0.0],"threshold":0.8},{"query":"q2","embedding":[0.0,1.0],"threshold":0.8}]'
```

### Health
```bash
curl -i http://127.0.0.1:3000/health
```

### Stats
```bash
curl -i http://127.0.0.1:3000/stats
```

## Design Decisions
* LRU Cache: HashMap and index-based doubly linked list -> O(1) get/insert, no external crates
* Semantic Lookup: Linear scan (O(n)); LRU handles eviction, not similarity search
* Concurrency: Arc<Mutex<_>> since lookups mutate recency and counters; effectively write-heavy
* TTL: Lazy expiration on access; avoids background complexity
* Batch Behavior: POST `/lookup/batch` fans out work with async futures, but shared `Mutex<AppState>` still serializes cache access. This is intentional because lookups mutate LRU recency and hit/miss counters, so simpler correctness was prioritized over maximum batch throughput.
* Error Handling:
  * 400 invalid input
  * 404 no match
  * 500 internal failure


## Improvements
* Replace linear scan with ANN/vector index
* Reduce lock contention (sharding or finer-grained locking)
* Expose TTL and capacity as environment variables or config file
* Background TTL cleanup (optional)

## Tests
```bash
cargo test
```

### Includes:
* LRU unit tests (eviction order, capacity=1, duplicate keys)
* Cosine similarity tests (identical, orthogonal, zero vectors, mismatched lengths)
* Integration tests for:
  * /health
  * /cache + /lookup
  * /stats
  * /lookup/batch
  * TTL expiration
