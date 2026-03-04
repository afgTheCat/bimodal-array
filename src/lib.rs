//! `bimodal_array` provides dual-mode access to a contiguous array. Access
//! is granted through two handler types:
//!
//! 1. [`ElementHandle<T>`] grants mutable access to a single element.
//! 2. [`ArrayHandle<T>`] grants exclusive mutable access to the entire array.
//!
//! Handles represent the ability to *attempt* a lock. The actual lock is held
//! by the returned guards:
//!
//! 1. when `ElementHandle::lock()` succeeds, it produces an [`ElementGuard<T>`]
//! 2. when `ArrayHandle::lock()` succeeds, it produces an [`ArrayGuard<T>`].
//!
//! Each [`ElementHandle`] is bound to a specific element. Guards for different
//! elements may may be held concurrently. An [`ArrayGuard<T>`] provides
//! exclusive access to the entire array and cannot coexist with any
//! [`ElementGuard<T>`].
//!
//! This pattern is useful for scatter–gather style workloads where independent
//! workers update elements in parallel, followed by an orchestration step that
//! processes the entire array. Internally, lock state is coordinated via a
//! single `AtomicUsize`. As a result, `bimodal_array` may not scale well under
//! high contention.
//!
//! # Examples
//!
//! ```no_run
//! # use bimodal_array::bimodal_array;
//! # use std::thread;
//! const WORKER_COUNT: usize = 8;
//! let data = vec![0usize; WORKER_COUNT];
//! let (mut array_handle, element_handles) = bimodal_array(data);
//! let threads: Vec<_> = element_handles
//!     .into_iter()
//!     .enumerate()
//!     .map(|(idx, mut element)| {
//!         thread::spawn(move || {
//!             let mut guard = element.lock().unwrap();
//!             *guard = 10 - idx;
//!         })
//!     })
//!     .collect();
//! threads.into_iter().for_each(|t| t.join().unwrap());
//! let mut array_guard = array_handle.lock().unwrap();
//! array_guard.sort();
//! ```

use core::slice;
use std::collections::TryReserveError;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::atomic::fence;

/// Errors that can occur when constructing or using a `bimodal_array`.
#[derive(Debug)]
pub enum BimodalArrayError {
    /// The provided collection length is not supported.
    ///
    /// Internally, `bimodal_array` reserves certain length values for state
    /// encoding. Currently, a length of `usize::MAX` is rejected.
    UnsupportedLength,

    /// Allocation failed.
    ///
    /// This typically occurs when reserving memory for the element handle
    /// storage. The contained [`TryReserveError`] provides additional details
    /// about whether the failure was due to capacity overflow or allocator
    /// refusal.
    AllocationFailed(TryReserveError),

    /// The array-wide lock could not be acquired.
    ///
    /// This happens when one or more [`ElementGuard<T>`] values are currently
    /// active.
    CouldNotAcquireArrayLock,

    /// The element-level lock could not be acquired.
    ///
    /// This happens when an [`ArrayGuard<T>`] is currently active.
    CouldNotAcquireElementLock,
}

/// The LockState represents how the BinomialArray is currently used
#[repr(transparent)]
struct LockState(AtomicUsize);

/// This represents when a GLOBAL_LOCK_STATE is in effect
const ARRAY_LOCK_STATE: usize = usize::MAX;

impl LockState {
    fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    /// Releases the global lock that is currently in place. Should only be
    /// called when dropping a GloablLock.
    fn release_array_lock(&self) {
        let old_state = self.0.fetch_xor(ARRAY_LOCK_STATE, Ordering::Release);
        debug_assert!(
            old_state == ARRAY_LOCK_STATE,
            "An accessor was active when trying to release the global lock"
        )
    }

    /// Rleases an accessor lock. Should only be called when dropping the
    /// accessor lock.
    fn release_element_lock(&self) {
        let old_state = self.0.fetch_sub(1, Ordering::Release);
        debug_assert!(
            (old_state != ARRAY_LOCK_STATE),
            "Global lock was in place when trying to release accessor"
        )
    }

    /// Acquires an accessor lock. Will only work if it can successfully
    /// increase the number of accessors and there is no Global lock in
    /// place
    fn acquire_element_lock(&self) -> Result<&Self, BimodalArrayError> {
        let mut current_state = self.0.load(Ordering::Relaxed);
        loop {
            if current_state == ARRAY_LOCK_STATE {
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

    /// Acquires a global lock. Will only work if there are no more accessors in
    /// place
    fn acquire_array_lock(&self) -> Result<&Self, BimodalArrayError> {
        match self
            .0
            .compare_exchange(0, ARRAY_LOCK_STATE, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => Ok(self),
            Err(_) => Err(BimodalArrayError::CouldNotAcquireArrayLock),
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

/// A handle granting access to a single element of a `bimodal_array`.
///
/// An `ElementHandle` represents the capability to acquire a lock for a
/// specific element. The actual lock is held by the returned
/// [`ElementGuard`], which provides mutable access to the element.
///
/// # Locking Semantics
///
/// - Calling [`lock`](Self::lock) attempts to acquire element-level access.
/// - If successful, it returns an [`ElementGuard`] providing `&mut T`.
/// - If an [`ArrayGuard`] is currently active, lock acquisition fails with
///   [`BimodalArrayError::CouldNotAcquireElementLock`].
///
/// Multiple element guards for *different* elements may coexist.
/// Element-level exclusivity is enforced through Rust's borrowing rules,
/// as `lock` requires `&mut self`.
pub struct ElementHandle<T> {
    inner: NonNull<BimodalArrayInner<T>>,
    data_ptr: NonNull<T>,
}

/// Should be able to Send when T is Send
unsafe impl<T: Send> Send for ElementHandle<T> {}
unsafe impl<T: Sync> Sync for ElementHandle<T> {}

impl<T> ElementHandle<T> {
    fn new(inner: NonNull<BimodalArrayInner<T>>, data_ptr: NonNull<T>) -> Self {
        Self { inner, data_ptr }
    }

    /// Attempts to acquire mutable access to this element.
    ///
    /// Returns an [`ElementGuard`] on success. If an [`ArrayGuard`] is
    /// currently active, this method returns
    /// [`BimodalArrayError::CouldNotAcquireElementLock`].
    ///
    /// Lock acquisition is non-blocking.
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

/// A guard providing mutable access to a single element.
///
/// An `ElementGuard` is returned by [`ElementHandle::lock`]. While an
/// `ElementGuard` is alive:
///
/// - No [`ArrayGuard<T>`] can be acquired.
/// - No other `ElementGuard` for the same element can exist.
///
/// Guards for *different* elements may coexist concurrently.
///
/// The guard dereferences to `&mut T`, allowing direct mutation of the
/// underlying element.
///
/// # RAII Semantics
///
/// The element-level lock is released automatically when this guard is
/// dropped. This ensures that access is tied to the lifetime of the guard.
///
/// # Examples
///
/// ```no_run
/// # use bimodal_array::bimodal_array;
/// let data = vec![0];
/// let (_, mut elements) = bimodal_array(data);
///
/// let mut guard = elements[0].lock().unwrap();
/// *guard = 42;
/// // Lock is released here when `guard` is dropped.
/// ```
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

impl<'a, T> AsRef<T> for ElementGuard<'a, T> {
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<'a, T> AsMut<T> for ElementGuard<'a, T> {
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut()
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

/// A handle granting array-wide access to a `bimodal_array`.
///
/// An `ArrayHandle` represents the capability to acquire an exclusive lock for
/// the entire collection. The actual lock is held by the returned
/// [`ArrayGuard`], which provides mutable access to the full slice `&mut [T]`.
///
/// # Locking Semantics
///
/// - Calling [`lock`](Self::lock) attempts to acquire array-wide exclusive
///   access.
/// - If successful, it returns an [`ArrayGuard`] providing `&mut [T]`.
/// - If one or more [`ElementGuard<T>`] values are currently active, lock
///   acquisition fails with [`BimodalArrayError::CouldNotAcquireArrayLock`].
///
/// Lock acquisition is non-blocking.
pub struct ArrayHandle<T> {
    inner: NonNull<BimodalArrayInner<T>>,
}

unsafe impl<T: Send> Send for ArrayHandle<T> {}
unsafe impl<T: Sync> Sync for ArrayHandle<T> {}

impl<T> ArrayHandle<T> {
    /// Attempts to acquire exclusive mutable access to the entire array.
    ///
    /// Returns an [`ArrayGuard`] on success. If any element guards are
    /// currently active, this method returns
    /// [`BimodalArrayError::CouldNotAcquireArrayLock`].
    ///
    /// Lock acquisition is non-blocking.
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

/// A guard providing exclusive access to the entire array.
///
/// An `ArrayGuard` is returned by [`ArrayHandle::lock`]. While an
/// `ArrayGuard` is alive:
///
/// - No [`ElementGuard<T>`] can be acquired.
/// - No other `ArrayGuard` can exist.
///
/// The guard dereferences to `&mut [T]`, allowing full mutable access to
/// the underlying slice.
///
/// # RAII Semantics
///
/// The array-wide lock is released automatically when this guard is dropped.
/// This ensures that exclusivity is tied to the lifetime of the guard.
///
/// # Examples
///
/// ```no_run
/// # use bimodal_array::bimodal_array;
/// let data = vec![3, 1, 2];
/// let (mut array_handle, _) = bimodal_array(data);
///
/// let mut guard = array_handle.lock().unwrap();
/// guard.sort();
/// // Lock is released here when `guard` is dropped.
/// ```
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

impl<'a, T> AsRef<[T]> for ArrayGuard<'a, T> {
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<'a, T> AsMut<[T]> for ArrayGuard<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

impl<'a, T> Drop for ArrayGuard<'a, T> {
    fn drop(&mut self) {
        self.state.release_array_lock();
    }
}

/// Constructs a bimodal array from an owned `Vec<T>`.
///
/// This is the fallible constructor. On success, it returns:
///
/// - an [`ArrayHandle<T>`], which can be used to acquire an [`ArrayGuard<T>`]
///   for exclusive access to the full slice, and
/// - a `Vec<ElementHandle<T>>`, containing one handle per element, each of
///   which can be used to acquire an [`ElementGuard<T>`] for element-level
///   access.
///
/// The returned handles enable a scatter–gather access pattern: independent
/// workers can lock and update different elements concurrently, followed by an
/// orchestration step that locks the entire array for consolidation.
///
/// # Errors
///
/// Returns [`BimodalArrayError::UnsupportedLength`] if the length of `data` is
/// not supported by the internal state encoding (currently `usize::MAX`).
///
/// Returns [`BimodalArrayError::AllocationFailed`] if allocating fails.
pub fn try_bimodal_array<T>(
    data: Vec<T>,
) -> Result<(ArrayHandle<T>, Vec<ElementHandle<T>>), BimodalArrayError> {
    let len = data.len();
    if len == usize::MAX {
        return Err(BimodalArrayError::UnsupportedLength);
    }
    let mut element_handles = Vec::new();
    element_handles
        .try_reserve_exact(len)
        .map_err(BimodalArrayError::AllocationFailed)?;
    let (data_ptr, _len, capacity) = data.into_raw_parts();
    let data_ptr = NonNull::new(data_ptr).unwrap();
    let inner = Box::leak(Box::new(BimodalArrayInner::new(len, capacity, data_ptr))).into();
    let array_handle = ArrayHandle { inner };
    element_handles.extend((0..len).map(|idx| {
        let elem_ptr = unsafe { data_ptr.add(idx) };
        ElementHandle::new(inner, elem_ptr)
    }));
    Ok((array_handle, element_handles))
}

/// Constructs a bimodal array from an owned `Vec<T>`.
///
/// This is a convenience wrapper around [`try_bimodal_array`] that panics on
/// error. Prefer [`try_bimodal_array`] if construction failure should be
/// handled gracefully.
///
/// # Panics
///
/// Panics if [`try_bimodal_array`] returns an error (e.g. unsupported length or
/// allocation failure).
pub fn bimodal_array<T>(data: Vec<T>) -> (ArrayHandle<T>, Vec<ElementHandle<T>>) {
    try_bimodal_array(data).unwrap_or_else(|e| panic!("bimodal_array construction failed: {e:?}"))
}
