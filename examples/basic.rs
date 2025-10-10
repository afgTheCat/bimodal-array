//! The following example illustrates

use bimodal_array::{BimodalArrayError, ElementHandle, bimodal_array};

const WORKER_COUNT: usize = 8;

fn main() {
    let (mut array_handle, element_handles) = bimodal_array(vec![0usize; WORKER_COUNT]).unwrap();
}
