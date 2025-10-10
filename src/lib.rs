//! BimodalArray is a simple synchronization primitive that allows accessing an array via a shared,
//! array level handle, and individual elements via their own handle. This crate is motivated by
//! reinforcement learning. During rollout, multiple workers might interact with their own
//! environments, during which learning is not happening. Once all workers have finishes, the
//! learning module processes the results of individual workers. Copying the reults from each
//! thread is expensive. Having all the elements inside Arc<Mutex<T>> makes working with the API
//! inconvinient.

use core::slice;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::atomic::fence;

#[derive(Debug)]
pub enum BimodalArrayError {
    VectorTooLarge,
    CouldNotAcquireArrayLock,
    CouldNotAcquireElementLock,
}

/// The LockState represents how the BinomialArray is currently used
///
///
#[repr(transparent)]
struct LockState(AtomicUsize);

/// This represents when a GLOBAL_LOCK_STATE is in effect
const ARRAY_LOCK_STATE: usize = usize::MAX;

impl LockState {
    fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    /// Releases the global lock that is currently in place. Should only be called when dropping a
    /// GloablLock.
    fn release_array_lock(&self) {
        let old_state = self.0.fetch_xor(ARRAY_LOCK_STATE, Ordering::Release);
        debug_assert!(
            old_state == ARRAY_LOCK_STATE,
            "An accessor was active when trying to release the global lock"
        )
    }

    /// Rleases an accessor lock. Should only be called when dropping the accessor lock.
    fn release_element_lock(&self) {
        let old_state = self.0.fetch_sub(1, Ordering::Release);
        debug_assert!(
            (old_state != ARRAY_LOCK_STATE),
            "Global lock was in place when trying to release accessor"
        )
    }

    /// Acquires an accessor lock. Will only work if it can successfully increase the number of
    /// accessors and there is no Global lock in place
    fn acquire_element_lock(&self) -> Result<&Self, BimodalArrayError> {
        let mut current_state = self.0.load(Ordering::Relaxed);
        loop {
            if current_state == ARRAY_LOCK_STATE {
                #[cfg(debug_assertions)]
                eprintln!(
                    "Could not acquire accessor, since there still is a global lock in place"
                );
                return Err(BimodalArrayError::CouldNotAcquireElementLock);
            }
            match self.0.compare_exchange(
                current_state,
                current_state + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(self),
                Err(new_state) => current_state = new_state,
            }
        }
    }

    /// Acquires a global lock. Will only work if there are no more accessors in place
    fn acquire_array_lock(&self) -> Result<&Self, BimodalArrayError> {
        match self
            .0
            .compare_exchange(0, ARRAY_LOCK_STATE, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Ok(self),
            Err(acceessor_count) => {
                #[cfg(debug_assertions)]
                eprintln!(
                    "Coudld not acquire global lock as there still are {acceessor_count} accesors presnent"
                );
                Err(BimodalArrayError::CouldNotAcquireArrayLock)
            }
        }
    }
}

struct BimodalArrayInner<T> {
    state: LockState,
    owner_count: AtomicUsize,
    len: usize,
    capacity: usize,
    data: NonNull<T>,
}

impl<T> BimodalArrayInner<T> {
    fn new(len: usize, capacity: usize, data: NonNull<T>) -> Self {
        let owner_count = len + 1;
        Self {
            state: LockState::new(),
            owner_count: AtomicUsize::new(owner_count),
            len,
            capacity,
            data,
        }
    }

    fn to_raw_parts(&self) -> (*mut T, usize, usize) {
        (self.data.as_ptr(), self.len, self.capacity)
    }
}

fn inner_free<T>(inner: NonNull<BimodalArrayInner<T>>) {
    let p = inner.as_ptr();
    let prev = unsafe { (*p).owner_count.fetch_sub(1, Ordering::Release) };
    if prev == 1 {
        fence(Ordering::Acquire);

        unsafe {
            let (data_ptr, len, capacity) = (*p).to_raw_parts();
            drop(Vec::from_raw_parts(data_ptr, len, capacity));
            drop(Box::from_raw(p));
        }
    }
}

pub struct ElementHandle<T> {
    inner: NonNull<BimodalArrayInner<T>>,
    data_ptr: NonNull<T>,
}

/// Should be able to Send when T is Send
unsafe impl<T: Send> Send for ElementHandle<T> {}
unsafe impl<T: Sync> Sync for ElementHandle<T> {}

pub struct ElementGuard<'a, T> {
    state: &'a LockState,
    data_ptr: NonNull<T>,
}

impl<'a, T> Deref for ElementGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.data_ptr.as_ref() }
    }
}

impl<'a, T> DerefMut for ElementGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.data_ptr.as_mut() }
    }
}

impl<'a, T> Drop for ElementGuard<'a, T> {
    fn drop(&mut self) {
        self.state.release_element_lock();
    }
}

impl<T> ElementHandle<T> {
    fn new(inner: NonNull<BimodalArrayInner<T>>, data_ptr: NonNull<T>) -> Self {
        Self { inner, data_ptr }
    }

    pub fn lock<'a>(&'a mut self) -> Result<ElementGuard<'a, T>, BimodalArrayError> {
        let inner = unsafe { &*self.inner.as_ptr() };
        let state = inner.state.acquire_element_lock()?;
        Ok(ElementGuard {
            state,
            data_ptr: self.data_ptr,
        })
    }
}

impl<T> Drop for ElementHandle<T> {
    fn drop(&mut self) {
        inner_free(self.inner);
    }
}

pub struct ArrayHandle<T> {
    inner: NonNull<BimodalArrayInner<T>>,
}

unsafe impl<T: Send> Send for ArrayHandle<T> {}
unsafe impl<T: Sync> Sync for ArrayHandle<T> {}

pub struct ArrayGuard<'a, T> {
    state: &'a LockState,
    data_ptr: NonNull<T>,
    len: usize,
}

impl<'a, T> Deref for ArrayGuard<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.data_ptr.as_ptr(), self.len) }
    }
}

impl<'a, T> DerefMut for ArrayGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.data_ptr.as_ptr(), self.len) }
    }
}

impl<'a, T> Drop for ArrayGuard<'a, T> {
    fn drop(&mut self) {
        self.state.release_array_lock();
    }
}

impl<T> ArrayHandle<T> {
    pub fn lock<'a>(&'a mut self) -> Result<ArrayGuard<'a, T>, BimodalArrayError> {
        let p = self.inner.as_ptr();
        let state = unsafe { (*p).state.acquire_array_lock()? };
        let (data_ptr, len, _) = unsafe { (*p).to_raw_parts() };
        let data_ptr = NonNull::new(data_ptr).unwrap();
        Ok(ArrayGuard {
            state,
            data_ptr,
            len,
        })
    }
}

impl<T> Drop for ArrayHandle<T> {
    fn drop(&mut self) {
        inner_free(self.inner);
    }
}

pub fn bimodal_array<T>(
    data: Vec<T>,
) -> Result<(ArrayHandle<T>, Vec<ElementHandle<T>>), BimodalArrayError> {
    if data.len() == usize::MAX {
        return Err(BimodalArrayError::VectorTooLarge);
    }
    let (data_ptr, len, capacity) = data.into_raw_parts();
    let data_ptr = NonNull::new(data_ptr).unwrap();
    let inner = Box::leak(Box::new(BimodalArrayInner::new(len, capacity, data_ptr))).into();
    let array_handle = ArrayHandle { inner };
    let element_handles = (0..len)
        .map(|idx| {
            let elem_ptr = unsafe { data_ptr.add(idx) };
            ElementHandle::new(inner, elem_ptr)
        })
        .collect();
    Ok((array_handle, element_handles))
}
