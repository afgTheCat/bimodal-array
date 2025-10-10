//! This example is used to stress test the current implementation and is mirrored in
//! `tests/correctness_test.rs`.

use bimodal_array::{BimodalArrayError, ElementHandle, bimodal_array};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, atomic::AtomicBool};
use std::thread;

const WORKER_COUNT: usize = 8;
const ITERS_PER_ELEMENT: usize = 20_000;
const ARRAY_LOCK_ROUNDS: usize = 100;

fn worker(pause: Arc<AtomicBool>, mut element: ElementHandle<usize>) {
    for _ in 0..ITERS_PER_ELEMENT {
        loop {
            while pause.load(AtomicOrdering::Acquire) {
                thread::yield_now();
            }
            match element.lock() {
                Ok(mut guard) => {
                    *guard += 1;
                    break;
                }
                Err(BimodalArrayError::CouldNotAcquireElementLock) => {
                    thread::yield_now();
                }
                Err(e) => panic!("unexpected lock error: {e:?}"),
            }
        }
    }
}

fn main() {
    let (mut array_handle, element_handles) = bimodal_array(vec![0usize; WORKER_COUNT]).unwrap();
    let pause = Arc::new(AtomicBool::new(false));
    let threads: Vec<_> = element_handles
        .into_iter()
        .map(|element| {
            let pause = pause.clone();
            thread::spawn(|| worker(pause, element))
        })
        .collect();

    for _ in 0..ARRAY_LOCK_ROUNDS {
        pause.store(true, AtomicOrdering::Release);
        let acquired = (0..50_000).any(|_| match array_handle.lock() {
            Ok(guard) => {
                drop(guard);
                true
            }
            Err(BimodalArrayError::CouldNotAcquireArrayLock) => {
                thread::yield_now();
                false
            }
            Err(e) => panic!("unexpected lock error: {e:?}"),
        });
        assert!(acquired, "failed to acquire array lock under pause");

        pause.store(false, AtomicOrdering::Release);
        thread::yield_now();
    }

    threads.into_iter().for_each(|t| t.join().unwrap());
    let guard = array_handle.lock().unwrap();
    assert!(guard.iter().all(|&v| v == ITERS_PER_ELEMENT));
}
