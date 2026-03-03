//! A basic usage example

use bimodal_array::bimodal_array;
use std::thread;

const WORKER_COUNT: usize = 8;

fn main() {
    let data = vec![0usize; WORKER_COUNT];
    let (mut array_handle, element_handles) = bimodal_array(data);
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
    threads.into_iter().for_each(|t| t.join().unwrap());
    let mut array_guard = array_handle.lock().unwrap();
    array_guard.sort();
}
