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


## Endpoints
* POST /cache
* POST /lookup
* POST /lookup/batch
* GET /health
* GET /stats

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

### Batch Lookup (Accepts raw JSON array)
```bash
curl -i -X POST http://127.0.0.1:3000/lookup/batch \
-H "Content-Type: application/json" \
-d '[{"query":"q1","embedding":[1.0,0.0],"threshold":0.8},{"query":"q2","embedding":[0.0,1.0],"threshold":0.8}]'
```

### Stats
```bash
curl -i http://127.0.0.1:3000/stats
```

## Design Decisions
* LRU Cache: HashMap and index-based doubly linked list -> O(1) get/insert, no external crates, no unsafe
* Semantic Lookup: Linear scan (O(n)); LRU handles eviction, not similarity search
* Concurrency: Arc<Mutex<_>> since lookups mutate recency and counters; effectively write-heavy
* TTL: Lazy execution on access; avoids background complexity
* Batch Behavior: Uses async task fan-out, but shared mutex and LRU mutation serialize cache access
* Error Handling:
  * 400 invalid input
  * 404 no match
  * 500 internal failure

## Improvements
* Replace linear scan with ANN/vector index
* Reduce lock contention (sharding or finer-grained locking)
* Configurable TTL and capacity
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
