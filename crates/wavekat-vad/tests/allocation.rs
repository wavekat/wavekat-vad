//! Allocation accounting for the real-time processing path.
//!
//! [`FrameAdapter::process_each`] is documented as allocation-free once the
//! carry buffer exists. This file proves it by counting real calls into the
//! global allocator, not by inspecting `Vec::capacity()` — a capacity that
//! happens not to change is not evidence that nothing was allocated.
//!
//! The counter is **per thread**, so the parallel test harness (and anything
//! it allocates on other threads) cannot pollute a measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use wavekat_vad::{FrameAdapter, VadCapabilities, VadError, VoiceActivityDetector};

// ---------------------------------------------------------------------------
// Counting global allocator
// ---------------------------------------------------------------------------

thread_local! {
    /// Allocations observed on this thread while counting is armed.
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    /// Whether this thread is currently counting.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

/// Records one allocation if this thread is counting.
///
/// Both thread-locals are const-initialised `Cell`s with no destructor, so
/// touching them from inside the allocator cannot itself allocate or recurse.
fn record_allocation() {
    let _ = COUNTING.try_with(|counting| {
        if counting.get() {
            let _ = ALLOC_COUNT.try_with(|n| n.set(n.get() + 1));
        }
    });
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Runs `f` and reports how many allocations it made on this thread.
fn count_allocs<R>(f: impl FnOnce() -> R) -> (R, usize) {
    ALLOC_COUNT.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    let result = f();
    COUNTING.with(|c| c.set(false));
    let count = ALLOC_COUNT.with(|n| n.get());
    (result, count)
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Minimal non-allocating detector.
struct MockVad {
    frame_size: usize,
}

impl VoiceActivityDetector for MockVad {
    fn capabilities(&self) -> VadCapabilities {
        VadCapabilities {
            sample_rate: 16_000,
            frame_size: self.frame_size,
            frame_duration_ms: (self.frame_size as u32 * 1000) / 16_000,
        }
    }

    fn process(&mut self, samples: &[i16], _sample_rate: u32) -> Result<f32, VadError> {
        assert_eq!(samples.len(), self.frame_size);
        Ok(0.5)
    }

    fn reset(&mut self) {}
}

fn adapter(frame_size: usize) -> FrameAdapter {
    FrameAdapter::new(Box::new(MockVad { frame_size }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Guards against a vacuous suite: if the counter cannot see a known
/// allocation, none of the assertions below mean anything.
#[test]
fn counter_actually_observes_allocations() {
    let (v, allocs) = count_allocs(|| Vec::<u8>::with_capacity(4096));
    assert_eq!(v.capacity(), 4096);
    assert!(
        allocs >= 1,
        "counting allocator saw {allocs} allocations for a 4096-byte Vec"
    );

    let (_, allocs) = count_allocs(|| {
        let mut v: Vec<u64> = Vec::new();
        for i in 0..10_000u64 {
            v.push(i);
        }
        v
    });
    assert!(allocs >= 2, "growing a Vec should realloc; saw {allocs}");

    // And it must report zero for work that genuinely does not allocate.
    let (sum, allocs) = count_allocs(|| (0u64..1000).sum::<u64>());
    assert_eq!(sum, 499_500);
    assert_eq!(allocs, 0, "arithmetic must not allocate");
}

#[test]
fn process_each_does_not_allocate_after_warm_up() {
    let mut adapter = adapter(256);
    let chunk = vec![0i16; 320]; // 20 ms transport packets

    // Warm-up: exercise every branch (carry top-up, whole frames from the
    // slice, trailing carry) so nothing is left to lazily initialise.
    for _ in 0..8 {
        adapter.process_each(&chunk, 16_000, |_| {}).unwrap();
    }

    let (_, allocs) = count_allocs(|| {
        for _ in 0..500 {
            adapter.process_each(&chunk, 16_000, |_| {}).unwrap();
        }
    });
    assert_eq!(allocs, 0, "process_each allocated {allocs} times");
}

#[test]
fn process_each_does_not_allocate_for_any_chunk_shape() {
    let mut adapter = adapter(256);
    let sizes = [0usize, 1, 7, 255, 256, 257, 320, 1024, 4096];
    let audio = vec![0i16; 4096];

    for _ in 0..4 {
        for &n in &sizes {
            adapter.process_each(&audio[..n], 16_000, |_| {}).unwrap();
        }
    }

    let (_, allocs) = count_allocs(|| {
        for _ in 0..20 {
            for &n in &sizes {
                adapter.process_each(&audio[..n], 16_000, |_| {}).unwrap();
            }
        }
    });
    assert_eq!(allocs, 0, "process_each allocated {allocs} times");
}

#[test]
fn process_latest_does_not_allocate() {
    let mut adapter = adapter(256);
    let chunk = vec![0i16; 320];
    for _ in 0..8 {
        adapter.process_latest(&chunk, 16_000).unwrap();
    }

    let (_, allocs) = count_allocs(|| {
        for _ in 0..500 {
            adapter.process_latest(&chunk, 16_000).unwrap();
        }
    });
    assert_eq!(allocs, 0, "process_latest allocated {allocs} times");
}

#[test]
fn process_does_not_allocate() {
    let mut adapter = adapter(256);
    let chunk = vec![0i16; 1024];
    for _ in 0..8 {
        adapter.process(&chunk, 16_000).unwrap();
    }

    let (_, allocs) = count_allocs(|| {
        for _ in 0..500 {
            adapter.process(&chunk, 16_000).unwrap();
        }
    });
    assert_eq!(allocs, 0, "process allocated {allocs} times");
}

/// `process_all` is the documented *allocating* wrapper: exactly one `Vec`.
#[test]
fn process_all_allocates_exactly_one_vec() {
    let mut adapter = adapter(256);
    let chunk = vec![0i16; 1024];
    for _ in 0..8 {
        adapter.process_all(&chunk, 16_000).unwrap();
    }

    let (scores, allocs) = count_allocs(|| adapter.process_all(&chunk, 16_000).unwrap());
    assert_eq!(scores.len(), 4);
    assert_eq!(
        allocs, 1,
        "process_all should allocate its result Vec exactly once, saw {allocs}"
    );
}

#[cfg(feature = "earshot")]
mod earshot {
    use super::*;
    use wavekat_vad::backends::earshot::EarshotVad;

    #[test]
    fn earshot_process_does_not_allocate_per_frame() {
        let mut vad = EarshotVad::new();
        let frame = vec![0i16; 256];
        for _ in 0..8 {
            vad.process(&frame, 16_000).unwrap();
        }

        let (_, allocs) = count_allocs(|| {
            for _ in 0..500 {
                vad.process(&frame, 16_000).unwrap();
            }
        });
        assert_eq!(allocs, 0, "EarshotVad::process allocated {allocs} times");
    }

    /// The exact downstream hot path: 20 ms / 320-sample ingress chunks fed to
    /// Earshot's 16 ms / 256-sample frames via `process_latest`, ~50x/second
    /// per concurrent call. Every chunk lands mid-frame, so the carry path
    /// runs on every single call.
    #[test]
    fn earshot_process_latest_on_320_sample_chunks_does_not_allocate() {
        let mut adapter = FrameAdapter::new(Box::new(EarshotVad::new()));
        let chunk = vec![0i16; 320];
        for _ in 0..8 {
            adapter.process_latest(&chunk, 16_000).unwrap();
        }

        let (_, allocs) = count_allocs(|| {
            // 1000 chunks == 20 seconds of call audio.
            for _ in 0..1000 {
                let score = adapter.process_latest(&chunk, 16_000).unwrap();
                assert!(score.is_finite());
            }
        });
        assert_eq!(
            allocs, 0,
            "the 20 ms ingress hot path allocated {allocs} times"
        );
    }

    #[test]
    fn earshot_behind_the_adapter_does_not_allocate() {
        let mut adapter = FrameAdapter::new(Box::new(EarshotVad::new()));
        let chunk = vec![0i16; 320];
        for _ in 0..8 {
            adapter.process_each(&chunk, 16_000, |_| {}).unwrap();
        }

        let (_, allocs) = count_allocs(|| {
            for _ in 0..500 {
                adapter.process_each(&chunk, 16_000, |_| {}).unwrap();
            }
        });
        assert_eq!(allocs, 0, "adapter + EarshotVad allocated {allocs} times");
    }
}
