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
- `ArrayHandle::lock_map()` returns an `ArrayMap<T>`

### Locking Rules

- Guards for different elements may be held concurrently.
- An `ArrayGuard<T>` provides exclusive access to the entire array.
- An `ArrayGuard<T>` cannot coexist with any `ElementGuard<T>`.
- `ArrayHandle::lock_map()` can build mapped views, including views that borrow
  from the array, while keeping the array lock held for the returned wrapper.
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

## Mapped Views

`ArrayHandle::lock_map()` is useful when you want to derive a collection of
views from the array while keeping the array lock held for as long as those
views exist.

```rust
use bimodal_array::bimodal_array;

struct View<'a> {
    value: &'a u32,
}

let (mut array_handle, _) = bimodal_array(vec![1, 2, 3]);
let views = array_handle.lock_map(|x| View { value: &*x }).unwrap();

assert_eq!(*views.as_ref()[0].value, 1);
```
