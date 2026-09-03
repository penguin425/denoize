//! Shared tract runtime preparation for CPU, Metal, and CUDA execution.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use crate::AcceleratorRuntime;
use tract_onnx::prelude::{TValue, TVec, TractResult, TypedModel};
use tract_onnx::tract_core::ops::source::TypedSource;
use tract_onnx::tract_core::runtime::{runtime_for_name, Runnable, State};

pub(crate) type SharedRunnable = Arc<dyn Runnable>;

struct CachedState {
    runnable: Weak<dyn Runnable>,
    state: Box<dyn State>,
}

thread_local! {
    /// Tract runtime states are deliberately `!Send`. Keep one state for each
    /// live stateless runnable on the OS thread that executes it so repeated
    /// real-time hops do not rebuild the plan's scratch state.
    static STATE_CACHE: RefCell<HashMap<usize, CachedState>> = RefCell::new(HashMap::new());
}

pub(crate) fn prepare(
    model: TypedModel,
    runtime: AcceleratorRuntime,
    context: &str,
) -> Result<SharedRunnable, String> {
    let runtime_name = runtime.name();
    let runtime = runtime_for_name(runtime_name)
        .map_err(|error| format!("select {runtime_name} runtime for {context}: {error:#}"))?
        .ok_or_else(|| format!("{runtime_name} runtime is not registered for {context}"))?;
    runtime
        .prepare(model)
        .map(Arc::from)
        .map_err(|error| format!("prepare {context} with {runtime_name}: {error:#}"))
}

/// Whether repeated calls may safely share one tract runtime state.
///
/// A fresh [`State`] is semantically significant for stateful operators, so
/// callers must retain [`Runnable::run`] behavior unless every optimized node
/// is stateless. Tract's input `Source` is the sole exception: its state only
/// stores the immutable input-node index. Recurrent models with explicit state
/// tensors, such as the authenticated DPDFNet graph, satisfy this condition.
pub(crate) fn supports_state_reuse(runnable: &SharedRunnable) -> bool {
    runnable.typed_model().is_some_and(|model| {
        model.nodes().iter().all(|node| {
            let op = node.op();
            op.is_stateless() || op.downcast_ref::<TypedSource>().is_some()
        })
    })
}

/// Run a stateless graph using its thread-local tract state.
///
/// The weak runnable guard prevents an allocator-reused trait-object address
/// from selecting a state belonging to a model that has already been dropped.
/// Failed states are discarded because tract may have stopped mid-turn.
pub(crate) fn run_reusing_state(
    runnable: &SharedRunnable,
    inputs: TVec<TValue>,
) -> TractResult<TVec<TValue>> {
    STATE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let key = Arc::as_ptr(runnable) as *const () as usize;
        let matches = cache
            .get(&key)
            .and_then(|cached| cached.runnable.upgrade())
            .is_some_and(|cached| Arc::ptr_eq(&cached, runnable));

        if !matches {
            cache.remove(&key);
            cache.retain(|_, cached| cached.runnable.strong_count() != 0);
            cache.insert(
                key,
                CachedState {
                    runnable: Arc::downgrade(runnable),
                    state: runnable.spawn()?,
                },
            );
        }

        let result = cache
            .get_mut(&key)
            .expect("tract state was inserted above")
            .state
            .run(inputs);
        if result.is_err() {
            cache.remove(&key);
        }
        result
    })
}
