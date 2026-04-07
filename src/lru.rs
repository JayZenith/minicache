use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug)]
struct Node<K, V> {
    key: K,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug)]
pub struct LruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    capacity: usize,
    map: HashMap<K, usize>,
    nodes: Vec<Option<Node<K, V>>>,
    free_list: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
    len: usize,
}

//attach methods to struct
impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            capacity,
            map: HashMap::new(),
            nodes: Vec::new(),
            free_list: Vec::new(),
            head: None,
            tail: None,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let idx = *self.map.get(key)?;
        self.move_to_front(idx);
        self.nodes[idx].as_ref().map(|node| &node.value)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let idx = *self.map.get(key)?;
        self.move_to_front(idx);
        self.nodes[idx].as_mut().map(|node| &mut node.value)
    }

    pub fn insert(&mut self, key: K, value: V) {
        if let Some(&idx) = self.map.get(&key) {
            if let Some(node) = self.nodes[idx].as_mut() {
                node.value = value;
            }
            self.move_to_front(idx);
            return;
        }

        if self.len == self.capacity {
            self.evict_lru();
        }

        let idx = self.alloc_node(Node {
            key: key.clone(),
            value,
            prev: None,
            next: None,
        });

        self.attach_front(idx);
        self.map.insert(key, idx);
        self.len += 1;
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.map.remove(key)?;
        self.detach(idx);
        self.len -= 1;

        let node = self.nodes[idx].take().unwrap();
        self.free_list.push(idx);
        Some(node.value)
    }

    fn evict_lru(&mut self) {
        let tail_idx = self.tail.expect("tail must exist when evicting");
        let key = self.nodes[tail_idx].as_ref().unwrap().key.clone();
        let _ = self.remove(&key);
    }

    fn alloc_node(&mut self, node: Node<K, V>) -> usize {
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx] = Some(node);
            idx
        } else {
            self.nodes.push(Some(node));
            self.nodes.len() - 1
        }
    }

    fn move_to_front(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.detach(idx);
        self.attach_front(idx);
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let node = self.nodes[idx].as_ref().unwrap();
            (node.prev, node.next)
        };

        match prev {
            Some(p) => self.nodes[p].as_mut().unwrap().next = next,
            None => self.head = next,
        }

        match next {
            Some(n) => self.nodes[n].as_mut().unwrap().prev = prev,
            None => self.tail = prev,
        }

        let node = self.nodes[idx].as_mut().unwrap();
        node.prev = None;
        node.next = None;
    }

    fn attach_front(&mut self, idx: usize) {
        {
            let node = self.nodes[idx].as_mut().unwrap();
            node.prev = None;
            node.next = self.head;
        }

        if let Some(old_head) = self.head {
            self.nodes[old_head].as_mut().unwrap().prev = Some(idx);
        }

        self.head = Some(idx);

        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter().map(|(k, &idx)| {
            let node = self.nodes[idx].as_ref().unwrap();
            (k, &node.value)
        })
    }

    pub fn touch(&mut self, key: &K) -> bool {
        if let Some(&idx) = self.map.get(key) {
            self.move_to_front(idx);
            true
        } else {
            false
        }
    }
}

// TESTS
#[cfg(test)]
mod tests {
    use super::LruCache;

    // access "a" and evict "b"
    #[test]
    fn evicts_least_recently_used() {
        let mut cache = LruCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        let _ = cache.get(&"a");
        cache.insert("c", 3);

        assert!(cache.contains_key(&"a"));
        assert!(!cache.contains_key(&"b"));
        assert!(cache.contains_key(&"c"));
    }

    // checks capacity=1 and keep newest
    #[test]
    fn capacity_one() {
        let mut cache = LruCache::new(1);
        cache.insert("a", 1);
        cache.insert("b", 2);

        assert!(!cache.contains_key(&"a"));
        assert!(cache.contains_key(&"b"));
    }

    // a should be 10 and b should get evicted
    #[test]
    fn duplicate_key_updates_value_and_recency() {
        let mut cache = LruCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("a", 10);
        cache.insert("c", 3);

        assert_eq!(cache.get(&"a"), Some(&10));
        assert!(!cache.contains_key(&"b"));
        assert!(cache.contains_key(&"c"));
    }
}
