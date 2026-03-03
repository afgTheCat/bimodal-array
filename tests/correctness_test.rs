use bimodal_array::BimodalArrayError;
use bimodal_array::ElementHandle;
use bimodal_array::try_bimodal_array;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::mpsc;
use std::thread;

#[test]
fn array_handle_must_not_be_send_or_sync_for_rc() {
    use static_assertions::assert_not_impl_any;
    use std::rc::Rc;

    assert_not_impl_any!(bimodal_array::ArrayHandle<Rc<u8>>: Send, Sync);
}

#[test]
fn array_handle_is_send_and_sync_for_u8() {
    use static_assertions::assert_impl_all;

    assert_impl_all!(bimodal_array::ArrayHandle<u8>: Send, Sync);
}

#[test]
fn array_handle_is_send_but_not_sync_for_cell() {
    use static_assertions::assert_impl_all;
    use static_assertions::assert_not_impl_any;
    use std::cell::Cell;

    assert_impl_all!(bimodal_array::ArrayHandle<Cell<u8>>: Send);
    assert_not_impl_any!(bimodal_array::ArrayHandle<Cell<u8>>: Sync);
}

#[test]
fn array_lock_fails_while_element_active() {
    let data = vec![0u32];
    let (mut array_handle, mut elements) = try_bimodal_array(data).unwrap();
    let mut element = elements.pop().unwrap();

    let (ready_tx, ready_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();

    let t = thread::spawn(move || {
        let _guard = element.lock().unwrap();
        ready_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });

    ready_rx.recv().unwrap();
    assert!(matches!(
        array_handle.lock(),
        Err(BimodalArrayError::CouldNotAcquireArrayLock)
    ));
    release_tx.send(()).unwrap();
    t.join().unwrap();
    array_handle.lock().unwrap();
}

#[test]
fn element_lock_fails_while_array_lock_active() {
    let data = vec![0u32];
    let (mut array_handle, mut elements) = try_bimodal_array(data).unwrap();
    let mut elements = elements.pop().unwrap();

    let _array = array_handle.lock().unwrap();

    let t = thread::spawn(move || {
        assert!(matches!(
            elements.lock(),
            Err(BimodalArrayError::CouldNotAcquireElementLock)
        ));
    });
    t.join().unwrap();
}

#[derive(Clone, Debug)]
struct DropItem {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropItem {
    fn drop(&mut self) {
        self.drops.fetch_add(1, AtomicOrdering::Relaxed);
    }
}

fn make_drop_items(len: usize) -> (Arc<AtomicUsize>, Vec<DropItem>) {
    let drops = Arc::new(AtomicUsize::new(0));
    let items = (0..len)
        .map(|_| DropItem {
            drops: drops.clone(),
        })
        .collect();
    (drops, items)
}

#[test]
fn drop_array_handle_while_element_lock_held_other_handles_exist() {
    let (drops, data) = make_drop_items(4);
    let (array_handle, mut element_handles) = try_bimodal_array(data).unwrap();

    let mut element = element_handles.pop().unwrap();
    let guard = element.lock().unwrap();

    drop(array_handle);
    assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);

    drop(guard);
    drop(element_handles);
    assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);

    drop(element);
    assert_eq!(drops.load(AtomicOrdering::Relaxed), 4);
}

#[test]
fn drop_element_handles_while_array_lock_held() {
    let (drops, data) = make_drop_items(5);
    let (mut array_handle, element_handles) = try_bimodal_array(data).unwrap();

    let guard = array_handle.lock().unwrap();

    drop(element_handles);
    assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);

    drop(guard);
    drop(array_handle);
    assert_eq!(drops.load(AtomicOrdering::Relaxed), 5);
}

#[test]
fn drop_other_element_handles_while_element_lock_held() {
    let (drops, data) = make_drop_items(2);
    let (array_handle, mut element_handles) = try_bimodal_array(data).unwrap();

    let mut element = element_handles.pop().unwrap();
    let guard = element.lock().unwrap();

    drop(element_handles);
    assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);

    drop(guard);
    drop(array_handle);
    assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);

    drop(element);
    assert_eq!(drops.load(AtomicOrdering::Relaxed), 2);
}

const ELEMENTS: usize = 8;
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

#[test]
fn stress_element_locks_with_periodic_array_lock() {
    let (mut array_handle, element_handles) = try_bimodal_array(vec![0usize; ELEMENTS]).unwrap();

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

#[test]
fn contrived_unsupported_length() {
    let data = vec![(); usize::MAX];
    let unsupported_length_error = try_bimodal_array(data);
    assert!(matches!(
        unsupported_length_error,
        Err(BimodalArrayError::UnsupportedLength)
    ));
}
