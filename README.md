# bimodal_array

`bimodal_array` provides phase-separated, dual-mode access to a contiguous array.

It enables a scatter–gather access pattern where:

- Independent workers can update elements concurrently.
- A coordinator can later acquire exclusive access to the entire array.

---

## Access Model

Access is granted through two handle types:

- `ElementHandle<T>` — grants mutable access to a single element.
- `ArrayHandle<T>` — grants exclusive mutable access to the entire array.

Handles represent the ability to *attempt* a lock. The actual lock is held by guards:

- `ElementHandle::lock()` returns an `ElementGuard<T>`
- `ArrayHandle::lock()` returns an `ArrayGuard<T>`

### Locking Rules

- Guards for different elements may be held concurrently.
- An `ArrayGuard<T>` provides exclusive access to the entire array.
- An `ArrayGuard<T>` cannot coexist with any `ElementGuard<T>`.
- Lock acquisition is non-blocking.

Internally, lock state is coordinated via a single `AtomicUsize`.
As a result, `bimodal_array` may not scale well under heavy contention.

---

## Example

```rust
use bimodal_array::bimodal_array;
use std::thread;

const WORKER_COUNT: usize = 8;

let data = vec![0usize; WORKER_COUNT];
let (mut array_handle, element_handles) = bimodal_array(data);

// Scatter phase
let threads: Vec<_> = element_handles
    .into_iter()
    .enumerate()
    .map(|(idx, mut element)| {
        thread::spawn(move || {
            let mut guard = element.lock().unwrap();
            *guard = 10 - idx;
        })
    })
    .collect();

for t in threads {
    t.join().unwrap();
}

// Gather phase
let mut array_guard = array_handle.lock().unwrap();
array_guard.sort();
```
