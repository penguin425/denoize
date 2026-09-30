//! Shared tract runtime preparation for CPU, Metal, and CUDA execution.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use crate::AcceleratorRuntime;
use tract_onnx::prelude::{TValue, TVec, TractResult, TypedModel};
use tract_onnx::tract_core::internal::DimLike;
use tract_onnx::tract_core::ops::scan::{OptScan, Scan};
use tract_onnx::tract_core::ops::source::TypedSource;
use tract_onnx::tract_core::runtime::{runtime_for_name, Runnable, State};

pub(crate) type SharedRunnable = Arc<dyn Runnable>;

struct CachedState {
    runnable: Weak<dyn Runnable>,
    state: Box<dyn State>,
    runs: usize,
}

// Independent stereo reaches this boundary roughly every 20 seconds, so the
// ordinary 60-second worker gate also exercises fresh scratch-state creation.
const MAX_REUSED_RUNS: usize = 4_096;
const MAX_SCAN_ITERATIONS: usize = 65_536;

thread_local! {
    /// Tract runtime states are deliberately `!Send`. Keep one state for each
    /// live reusable runnable on the OS thread that executes it so repeated
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
/// is stateless or explicitly turn-local. Input `Source` state stores only
/// its immutable node index. A Scan is turn-local only when it resets hidden
/// state from its inputs every call, never skips iterations, and has a plain
/// stateless body. Its static iteration bound and periodic cache respawn keep
/// tract's otherwise persistent position counter below overflow on 32-bit too.
pub(crate) fn supports_state_reuse(runnable: &SharedRunnable) -> bool {
    runnable.typed_model().is_some_and(|model| {
        model.nodes().iter().enumerate().all(|(node_id, node)| {
            let op = node.op();
            if op.is_stateless() || op.downcast_ref::<TypedSource>().is_some() {
                return true;
            }
            let Some(scan) = op.downcast_ref::<OptScan>() else {
                return false;
            };
            scan.reset_every_turn
                && scan.skip == 0
                && plain_model_is_stateless(scan.plan.model())
                && model
                    .node_input_facts(node_id)
                    .ok()
                    .and_then(|facts| scan.iteration_count(&facts).and_then(|n| n.to_usize().ok()))
                    .is_some_and(|n| n <= MAX_SCAN_ITERATIONS)
        })
    })
}

/// Mark only top-level, plain-body scans as turn-local.
///
/// Nested scans are deliberately not inspected or changed: a stateful scan in
/// their body must retain tract's normal `Runnable::run` semantics.
pub(crate) fn reset_plain_scans_for_reuse(model: &mut TypedModel) {
    for node in model.nodes_mut() {
        let Some(scan) = node.op_as_mut::<Scan>() else {
            continue;
        };
        if scan.skip == 0 && plain_model_is_stateless(&scan.body) {
            scan.reset_every_turn = true;
        }
    }
}

fn plain_model_is_stateless(model: &TypedModel) -> bool {
    model.nodes().iter().all(|node| {
        let op = node.op();
        op.is_stateless() || op.downcast_ref::<TypedSource>().is_some()
    })
}

/// Run a graph accepted by [`supports_state_reuse`] with thread-local state.
///
/// The weak runnable guard prevents an allocator-reused trait-object address
/// from selecting a state belonging to a model that has already been dropped.
/// Failed states are discarded because tract may have stopped mid-turn.
pub(crate) fn run_reusing_state(
    runnable: &SharedRunnable,
    inputs: TVec<TValue>,
) -> TractResult<TVec<TValue>> {
    run_reusing_state_with_limit(runnable, inputs, MAX_REUSED_RUNS)
}

// Kept separate so focused tests can exercise the respawn boundary without
// performing thousands of model evaluations.
fn run_reusing_state_with_limit(
    runnable: &SharedRunnable,
    inputs: TVec<TValue>,
    max_runs: usize,
) -> TractResult<TVec<TValue>> {
    STATE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let key = Arc::as_ptr(runnable) as *const () as usize;
        let matches = cache.get(&key).is_some_and(|cached| {
            cached.runs < max_runs
                && cached
                    .runnable
                    .upgrade()
                    .is_some_and(|live| Arc::ptr_eq(&live, runnable))
        });

        if !matches {
            cache.remove(&key);
            cache.retain(|_, cached| cached.runnable.strong_count() != 0);
            cache.insert(
                key,
                CachedState {
                    runnable: Arc::downgrade(runnable),
                    state: runnable.spawn()?,
                    runs: 0,
                },
            );
        }

        let cached = cache.get_mut(&key).expect("tract state was inserted above");
        cached.runs += 1;
        let result = cached.state.run(inputs);
        if result.is_err() {
            cache.remove(&key);
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_onnx::prelude::*;
    use tract_onnx::tract_core::ops::{
        math,
        scan::{InputMapping, OutputMapping, ScanInfo},
    };

    fn additive_body() -> TypedModel {
        let mut body = TypedModel::default();
        let x = body.add_source("x", f32::fact([1])).unwrap();
        let h = body.add_source("h", f32::fact([1])).unwrap();
        let sum = body.wire_node("sum", math::add(), &[x, h]).unwrap()[0];
        body.set_input_outlets(&[x, h]).unwrap();
        body.select_output_outlets(&[sum]).unwrap();
        body
    }

    fn scan_for_body(body: TypedModel, skip: usize) -> Scan {
        Scan::new(
            body,
            vec![
                InputMapping::Scan(ScanInfo { axis: 0, chunk: 1 }),
                InputMapping::State,
            ],
            vec![OutputMapping {
                scan: Some((0, ScanInfo { axis: 0, chunk: 1 })),
                full_dim_hint: None,
                last_value_slot: Some(1),
                state: true,
            }],
            skip,
        )
        .unwrap()
    }

    fn model_with_scan(body: TypedModel, sequence_dim: Option<usize>, skip: usize) -> TypedModel {
        let mut model = TypedModel::default();
        let sequence_dim: TDim = sequence_dim
            .map(Into::into)
            .unwrap_or_else(|| model.symbols.sym("iterations").into());
        let x = model.add_source("x", f32::fact([sequence_dim])).unwrap();
        let h = model.add_source("h", f32::fact([1])).unwrap();
        let outputs = model
            .wire_node("scan", scan_for_body(body, skip), &[x, h])
            .unwrap();
        model.set_input_outlets(&[x, h]).unwrap();
        model.select_output_outlets(&outputs).unwrap();
        model
    }

    fn cpu_runnable(model: TypedModel) -> SharedRunnable {
        prepare(model, AcceleratorRuntime::Cpu, "scan reuse unit test").unwrap()
    }

    fn output_bits(outputs: &[TValue]) -> Vec<Vec<u32>> {
        outputs
            .iter()
            .map(|output| {
                output
                    .try_as_plain()
                    .unwrap()
                    .as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .map(|sample| sample.to_bits())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn reset_scan_reuse_matches_fresh_runs_across_channels_and_respawn() {
        let original = model_with_scan(additive_body(), Some(3), 0);
        let fresh = cpu_runnable(original.clone());
        assert!(!supports_state_reuse(&fresh));
        let mut transformed = original;
        reset_plain_scans_for_reuse(&mut transformed);
        let reused = cpu_runnable(transformed);
        assert!(supports_state_reuse(&reused));
        assert!(reused
            .typed_model()
            .unwrap()
            .nodes()
            .iter()
            .any(|node| node.op().downcast_ref::<OptScan>().is_some()));
        for turn in 0..8 {
            let (x, h) = if turn % 2 == 0 {
                ([1.0f32, 2.0, 3.0], 10.0f32)
            } else {
                ([-4.0f32, 1.5, 0.125], -0.25f32)
            };
            let inputs = tvec!(
                Tensor::from_shape(&[3], &x).unwrap().into_tvalue(),
                Tensor::from_shape(&[1], &[h]).unwrap().into_tvalue(),
            );
            let expected = fresh.run(inputs.clone()).unwrap();
            let actual = run_reusing_state_with_limit(&reused, inputs, 2).unwrap();
            assert_eq!(output_bits(&actual), output_bits(&expected), "turn={turn}");
            if turn % 2 == 0 {
                assert_eq!(
                    output_bits(&actual),
                    vec![
                        [11.0f32, 13.0, 16.0].map(f32::to_bits).to_vec(),
                        vec![16.0f32.to_bits()],
                    ]
                );
            }
            STATE_CACHE.with(|cache| {
                let key = Arc::as_ptr(&reused) as *const () as usize;
                assert_eq!(cache.borrow().get(&key).unwrap().runs, turn % 2 + 1);
            });
        }
    }

    #[test]
    fn failed_run_evicts_cached_scan_state_before_retry() {
        let original = model_with_scan(additive_body(), Some(3), 0);
        let fresh = cpu_runnable(original.clone());
        let mut transformed = original;
        reset_plain_scans_for_reuse(&mut transformed);
        let reused = cpu_runnable(transformed);
        assert!(supports_state_reuse(&reused));
        let inputs = tvec!(
            Tensor::from_shape(&[3], &[1.0f32, 2.0, 3.0])
                .unwrap()
                .into_tvalue(),
            Tensor::from_shape(&[1], &[10.0f32]).unwrap().into_tvalue(),
        );
        run_reusing_state(&reused, inputs.clone()).unwrap();
        assert!(run_reusing_state(&reused, tvec!(inputs[0].clone())).is_err());
        STATE_CACHE.with(|cache| {
            let key = Arc::as_ptr(&reused) as *const () as usize;
            assert!(!cache.borrow().contains_key(&key));
        });
        assert_eq!(
            output_bits(&run_reusing_state(&reused, inputs.clone()).unwrap()),
            output_bits(&fresh.run(inputs).unwrap()),
        );
    }

    #[test]
    fn reuse_rejects_nonreset_and_skipping_scans() {
        let original = cpu_runnable(model_with_scan(additive_body(), Some(3), 0));
        assert!(original.typed_model().unwrap().nodes().iter().any(|node| {
            node.op()
                .downcast_ref::<OptScan>()
                .is_some_and(|scan| !scan.reset_every_turn)
        }));
        assert!(!supports_state_reuse(&original));
        for force_reset in [false, true] {
            let mut skipping = model_with_scan(additive_body(), Some(3), 1);
            reset_plain_scans_for_reuse(&mut skipping);
            let scan = skipping
                .nodes_mut()
                .iter_mut()
                .find_map(|node| node.op_as_mut::<Scan>())
                .unwrap();
            assert!(!scan.reset_every_turn);
            scan.reset_every_turn = force_reset;
            let skipping = cpu_runnable(skipping);
            assert!(skipping.typed_model().unwrap().nodes().iter().any(|node| {
                node.op()
                    .downcast_ref::<OptScan>()
                    .is_some_and(|scan| scan.skip == 1)
            }));
            assert!(!supports_state_reuse(&skipping));
        }
    }

    #[test]
    fn reuse_does_not_allow_nested_stateful_body() {
        let mut body = TypedModel::default();
        let x = body.add_source("x", f32::fact([1])).unwrap();
        let h = body.add_source("h", f32::fact([1])).unwrap();
        let sequence = body
            .add_const(
                "nested-input",
                Tensor::from_shape(&[2], &[1.0f32, 2.0]).unwrap(),
            )
            .unwrap();
        let nested = body
            .wire_node(
                "nested-scan",
                scan_for_body(additive_body(), 0),
                &[sequence, h],
            )
            .unwrap();
        let sum = body.wire_node("sum", math::add(), &[x, nested[1]]).unwrap()[0];
        body.set_input_outlets(&[x, h]).unwrap();
        body.select_output_outlets(&[sum]).unwrap();
        assert!(!plain_model_is_stateless(&body));
        let mut model = model_with_scan(body, Some(3), 0);
        reset_plain_scans_for_reuse(&mut model);
        let scan = model
            .nodes_mut()
            .iter_mut()
            .find_map(|node| node.op_as_mut::<Scan>())
            .unwrap();
        assert!(!scan.reset_every_turn);
        scan.reset_every_turn = true;
        let model = cpu_runnable(model);
        assert!(model.typed_model().unwrap().nodes().iter().any(|node| {
            node.op().downcast_ref::<OptScan>().is_some_and(|scan| {
                scan.reset_every_turn && !plain_model_is_stateless(scan.plan.model())
            })
        }));
        assert!(!supports_state_reuse(&model));
    }

    #[test]
    fn reuse_rejects_unbounded_and_too_many_scan_iterations() {
        for dim in [None, Some(MAX_SCAN_ITERATIONS + 1)] {
            let mut model = model_with_scan(additive_body(), dim, 0);
            reset_plain_scans_for_reuse(&mut model);
            let model = cpu_runnable(model);
            let typed = model.typed_model().unwrap();
            let node = typed
                .nodes()
                .iter()
                .find(|node| node.op().downcast_ref::<OptScan>().is_some())
                .expect("the negative case must retain its optimized Scan");
            let scan = node.op().downcast_ref::<OptScan>().unwrap();
            assert!(scan.reset_every_turn);
            assert_eq!(scan.skip, 0);
            assert!(plain_model_is_stateless(scan.plan.model()));
            let count = scan
                .iteration_count(&typed.node_input_facts(node.id).unwrap())
                .unwrap()
                .to_usize();
            match dim {
                None => assert!(count.is_err()),
                Some(expected) => assert_eq!(count.unwrap(), expected),
            }
            assert!(!supports_state_reuse(&model));
        }
    }

    #[test]
    fn reuse_limits_are_bounded_for_32_bit_state_counters() {
        assert_eq!(MAX_REUSED_RUNS * MAX_SCAN_ITERATIONS, 1 << 28);
        assert!(MAX_SCAN_ITERATIONS > 0);
    }

    #[test]
    fn empty_model_is_a_plain_body_and_reset_helper_is_noop() {
        let mut model = TypedModel::default();
        assert!(plain_model_is_stateless(&model));
        reset_plain_scans_for_reuse(&mut model);
        assert!(model.nodes().is_empty());
    }
}
