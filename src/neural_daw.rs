//! Portable contracts shared by the off-callback neural DAW plug-in and its
//! CLI/Desktop control surfaces.

use crate::daw::{
    read_bounded_regular_file, serialize_bounded, write_document, MAX_DAW_DOCUMENT_BYTES,
};
use crate::CommitMode;
use serde::{Deserialize, Serialize};
use std::path::Path;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;

pub const NEURAL_DAW_PLUGIN_ID: &str = "org.penguin425.denoize.neural";
pub const NEURAL_DAW_SESSION_SCHEMA: &str = "denoize-neural-daw-session-v1";
pub const NEURAL_DAW_SESSION_SCHEMA_VERSION: u32 = 1;
pub const NEURAL_DAW_MODEL_ID: &str = "gtcrn-dns3";
pub const NEURAL_DAW_MODEL_SHA256: &str =
    "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87";
pub const NEURAL_HQ_DAW_PLUGIN_ID: &str = "org.penguin425.denoize.neural-hq";
pub const NEURAL_HQ_DAW_MODEL_ID: &str = "dpdfnet2-48khz-hr";
pub const NEURAL_HQ_DAW_MODEL_SHA256: &str =
    "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b";
pub const NEURAL_DAW_CHUNK_MILLIS: u32 = 10;
pub const NEURAL_DAW_LATENCY_CHUNKS: u32 = 24;
pub const NEURAL_DAW_LATENCY_POLICY: &str = "fixed-24x10ms-worker-v1";
pub const NEURAL_DAW_MAX_SAMPLE_RATE: u32 = crate::daw::DAW_MAX_SAMPLE_RATE;

#[cfg(target_os = "macos")]
const MACOS_NEURAL_PERIOD_NANOS: u64 = NEURAL_DAW_CHUNK_MILLIS as u64 * 1_000_000;
#[cfg(target_os = "macos")]
const MACOS_NEURAL_COMPUTATION_NANOS: u64 = 8_000_000;

#[cfg(target_os = "macos")]
enum MacOsPreviousMachPolicy {
    TimeConstraint(mach2::thread_policy::thread_time_constraint_policy_data_t),
    Extended(mach2::thread_policy::thread_extended_policy_data_t),
}

/// Keeps the neural inference worker in the platform's interactive audio class.
///
/// The CLAP audio callback only exchanges bounded queue entries with the
/// worker, but Windows DAWs can run that callback in the `Pro Audio` class.
/// Keeping the dependent inference worker at ordinary priority can therefore
/// starve it even when the model's measured compute time is below its buffered
/// deadline. Windows uses the `Pro Audio` MMCSS task at its critical relative
/// priority. macOS combines a Mach time constraint with an Audio Work Interval
/// that describes each 10 ms inference cycle; other platforms deliberately
/// keep their existing scheduling.
#[doc(hidden)]
#[must_use = "dropping the guard releases the neural worker scheduling class"]
pub struct NeuralDawWorkerPriorityGuard {
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
    #[cfg(target_os = "macos")]
    mach_thread: mach2::port::mach_port_t,
    #[cfg(target_os = "macos")]
    previous_mach_policy: MacOsPreviousMachPolicy,
    #[cfg(target_os = "macos")]
    previous_qos_class: libc::qos_class_t,
    #[cfg(target_os = "macos")]
    previous_relative_priority: libc::c_int,
    #[cfg(target_os = "macos")]
    audio_work_interval: Option<MacOsAudioWorkInterval>,
    // Platform scheduling state must be released on the thread that acquired
    // it. Make that invariant a type property instead of relying on callers.
    #[cfg(any(windows, target_os = "macos"))]
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl NeuralDawWorkerPriorityGuard {
    /// Enters the scheduling class appropriate for a neural audio worker.
    ///
    /// On Windows and macOS, activation fails closed if the current thread
    /// cannot enter the platform scheduling class. On other platforms this is
    /// a no-op guard.
    pub fn acquire() -> Result<Self, String> {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Threading::{
                AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW,
                AvSetMmThreadPriority, AVRT_PRIORITY_CRITICAL,
            };

            let mut task_index = 0u32;
            // SAFETY: `w!` supplies a process-lifetime, NUL-terminated UTF-16
            // string and `task_index` is a valid writable DWORD. The returned
            // handle is owned by this guard and reverted on this same thread.
            let handle = unsafe {
                AvSetMmThreadCharacteristicsW(windows_sys::w!("Pro Audio"), &mut task_index)
            };
            if handle.is_null() {
                return Err(format!(
                    "register neural inference worker with Windows Pro Audio MMCSS: {}",
                    std::io::Error::last_os_error()
                ));
            }
            // Registration chooses the task category, while this call selects
            // the worker's relative priority inside that task. The inference
            // result is a dependency of the audio callback, so it must not be
            // left at MMCSS's default relative priority.
            if unsafe { AvSetMmThreadPriority(handle, AVRT_PRIORITY_CRITICAL) } == 0 {
                let priority_error = std::io::Error::last_os_error();
                // SAFETY: `handle` was returned to this calling thread above.
                // Revert it before failing activation so registration cannot
                // leak when setting the relative priority fails.
                let reverted = unsafe { AvRevertMmThreadCharacteristics(handle) } != 0;
                let mut message = format!(
                    "set neural inference worker Windows Pro Audio MMCSS priority: {priority_error}"
                );
                if !reverted {
                    message.push_str(&format!(
                        "; leave Windows Pro Audio MMCSS after priority failure: {}",
                        std::io::Error::last_os_error()
                    ));
                }
                return Err(message);
            }
            Ok(Self {
                handle,
                _not_send: std::marker::PhantomData,
            })
        }

        #[cfg(target_os = "macos")]
        {
            let mut previous_qos_class = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
            let mut previous_relative_priority = 0;
            // A Mach time constraint removes any explicit pthread QoS. Capture
            // it first for restoration, but do not install a second active
            // priority policy alongside the real-time one.
            let result = unsafe {
                libc::pthread_get_qos_class_np(
                    libc::pthread_self(),
                    &mut previous_qos_class,
                    &mut previous_relative_priority,
                )
            };
            if result != 0 {
                return Err(format!(
                    "read neural inference worker macOS QoS: {}",
                    std::io::Error::from_raw_os_error(result)
                ));
            }
            // XNU removes an explicit pthread QoS when a Mach scheduling
            // policy is applied. Establish one coherent real-time policy
            // before joining the Audio Work Interval instead of mixing the
            // two mutually exclusive priority mechanisms.
            let (mach_thread, previous_mach_policy) = acquire_macos_time_constraint()?;
            let audio_work_interval = match MacOsAudioWorkInterval::acquire() {
                Ok(interval) => interval,
                Err(error) => {
                    // Audio Work Intervals are unavailable before macOS 11
                    // and may be unavailable in restricted hosts. Preserve the
                    // established Mach time constraint as the fallback.
                    eprintln!(
                        "denoize Neural worker could not enter a macOS Audio Work Interval: {error}"
                    );
                    None
                }
            };
            if audio_work_interval.is_none() && macos_scheduling_evidence_enabled() {
                eprintln!("DENOIZE_MACOS_AUDIO_WORK_INTERVAL unavailable=true");
            }
            Ok(Self {
                mach_thread,
                previous_mach_policy,
                previous_qos_class,
                previous_relative_priority,
                audio_work_interval,
                _not_send: std::marker::PhantomData,
            })
        }

        #[cfg(not(any(windows, target_os = "macos")))]
        {
            Ok(Self {})
        }
    }

    /// Runs one bounded inference cycle under the platform's periodic audio
    /// scheduling contract.
    ///
    /// Audio Work Interval bookkeeping is real-time safe. A scheduling API
    /// failure disables the optional interval for the rest of this worker but
    /// never converts otherwise valid audio into a processing failure.
    #[doc(hidden)]
    pub fn run_inference_cycle<T>(&mut self, process: impl FnOnce() -> T) -> T {
        #[cfg(target_os = "macos")]
        {
            if let Some(interval) = self.audio_work_interval.as_mut() {
                return interval.run_cycle(process);
            }
        }
        process()
    }

    /// Starts a new measured-cycle window after an unmeasured scheduler
    /// pre-roll.
    ///
    /// This only resets diagnostic counters. It does not alter the active
    /// scheduling class or Audio Work Interval membership.
    #[doc(hidden)]
    pub fn begin_inference_cycle_measurement(&mut self) {
        #[cfg(target_os = "macos")]
        if let Some(interval) = self.audio_work_interval.as_mut() {
            interval.started_cycles = 0;
            interval.finished_cycles = 0;
            interval.start_failures = 0;
            interval.finish_failures = 0;
        }
    }
}

impl Drop for NeuralDawWorkerPriorityGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Threading::AvRevertMmThreadCharacteristics;

            // SAFETY: the guard is !Send, so it is dropped on the acquiring
            // thread, and `handle` is the live handle returned by the matching
            // AvSetMmThreadCharacteristicsW call above.
            if unsafe { AvRevertMmThreadCharacteristics(self.handle) } == 0 {
                eprintln!(
                    "denoize Neural worker could not leave Windows Pro Audio MMCSS: {}",
                    std::io::Error::last_os_error()
                );
            }
        }

        #[cfg(target_os = "macos")]
        {
            use libc::thread_policy_t;
            use mach2::kern_return::KERN_SUCCESS;
            use mach2::thread_policy::{
                thread_policy_set, THREAD_EXTENDED_POLICY, THREAD_EXTENDED_POLICY_COUNT,
                THREAD_TIME_CONSTRAINT_POLICY, THREAD_TIME_CONSTRAINT_POLICY_COUNT,
            };

            // Leave and release the workgroup on the same pthread that joined
            // it, before restoring that thread's prior Mach scheduling mode.
            drop(self.audio_work_interval.take());

            let (flavor, policy, count) = match &mut self.previous_mach_policy {
                MacOsPreviousMachPolicy::TimeConstraint(policy) => (
                    THREAD_TIME_CONSTRAINT_POLICY,
                    (policy as *mut _) as thread_policy_t,
                    THREAD_TIME_CONSTRAINT_POLICY_COUNT,
                ),
                MacOsPreviousMachPolicy::Extended(policy) => (
                    THREAD_EXTENDED_POLICY,
                    (policy as *mut _) as thread_policy_t,
                    THREAD_EXTENDED_POLICY_COUNT,
                ),
            };
            // SAFETY: the guard is !Send, so it is dropped on the acquiring
            // pthread. The policy value was captured from this Mach thread
            // before the guard changed it.
            let result = unsafe { thread_policy_set(self.mach_thread, flavor, policy, count) };
            if result != KERN_SUCCESS {
                eprintln!(
                    "denoize Neural worker could not restore macOS scheduling policy: Mach error {result}"
                );
            }
            if self.previous_qos_class as u32 != libc::qos_class_t::QOS_CLASS_UNSPECIFIED as u32 {
                // Restore an explicit prior QoS only after leaving the active
                // Mach real-time policy, keeping the two mechanisms separate.
                let result = unsafe {
                    libc::pthread_set_qos_class_self_np(
                        self.previous_qos_class,
                        self.previous_relative_priority,
                    )
                };
                if result != 0 {
                    eprintln!(
                        "denoize Neural worker could not restore macOS QoS: {}",
                        std::io::Error::from_raw_os_error(result)
                    );
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
type AudioWorkIntervalCreateFn =
    unsafe extern "C" fn(*const libc::c_char, u32, *mut libc::c_void) -> *mut libc::c_void;
#[cfg(target_os = "macos")]
type OsWorkgroupJoinFn =
    unsafe extern "C" fn(*mut libc::c_void, *mut MacOsWorkgroupJoinToken) -> libc::c_int;
#[cfg(target_os = "macos")]
type OsWorkgroupLeaveFn = unsafe extern "C" fn(*mut libc::c_void, *mut MacOsWorkgroupJoinToken);
#[cfg(target_os = "macos")]
type OsWorkgroupIntervalStartFn =
    unsafe extern "C" fn(*mut libc::c_void, u64, u64, *mut libc::c_void) -> libc::c_int;
#[cfg(target_os = "macos")]
type OsWorkgroupIntervalFinishFn =
    unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> libc::c_int;
#[cfg(target_os = "macos")]
type OsReleaseFn = unsafe extern "C" fn(*mut libc::c_void);

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct MacOsAudioWorkIntervalApi {
    create: AudioWorkIntervalCreateFn,
    join: OsWorkgroupJoinFn,
    leave: OsWorkgroupLeaveFn,
    start: OsWorkgroupIntervalStartFn,
    finish: OsWorkgroupIntervalFinishFn,
    release: OsReleaseFn,
}

#[cfg(target_os = "macos")]
impl MacOsAudioWorkIntervalApi {
    fn get() -> Option<&'static Self> {
        static API: OnceLock<Option<MacOsAudioWorkIntervalApi>> = OnceLock::new();
        API.get_or_init(|| {
            // Load the framework at worker startup instead of hard-linking a
            // macOS 11 symbol into binaries that retain an older deployment
            // target. The successful handle intentionally remains open for
            // the process lifetime because the cached function pointers do.
            let framework = unsafe {
                libc::dlopen(
                    c"/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox".as_ptr(),
                    libc::RTLD_LAZY | libc::RTLD_LOCAL,
                )
            };
            if framework.is_null() {
                return None;
            }
            let create = unsafe { libc::dlsym(framework, c"AudioWorkIntervalCreate".as_ptr()) };
            let join = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"os_workgroup_join".as_ptr()) };
            let leave = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"os_workgroup_leave".as_ptr()) };
            let start =
                unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"os_workgroup_interval_start".as_ptr()) };
            let finish = unsafe {
                libc::dlsym(libc::RTLD_DEFAULT, c"os_workgroup_interval_finish".as_ptr())
            };
            let release = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"os_release".as_ptr()) };
            if [create, join, leave, start, finish, release]
                .iter()
                .any(|symbol| symbol.is_null())
            {
                unsafe { libc::dlclose(framework) };
                return None;
            }
            Some(Self {
                // SAFETY: each non-null address above was resolved by its
                // exported C symbol name from the owning Apple framework.
                create: unsafe {
                    std::mem::transmute::<*mut libc::c_void, AudioWorkIntervalCreateFn>(create)
                },
                join: unsafe { std::mem::transmute::<*mut libc::c_void, OsWorkgroupJoinFn>(join) },
                leave: unsafe {
                    std::mem::transmute::<*mut libc::c_void, OsWorkgroupLeaveFn>(leave)
                },
                start: unsafe {
                    std::mem::transmute::<*mut libc::c_void, OsWorkgroupIntervalStartFn>(start)
                },
                finish: unsafe {
                    std::mem::transmute::<*mut libc::c_void, OsWorkgroupIntervalFinishFn>(finish)
                },
                release: unsafe { std::mem::transmute::<*mut libc::c_void, OsReleaseFn>(release) },
            })
        })
        .as_ref()
    }
}

#[cfg(target_os = "macos")]
#[repr(C, align(8))]
struct MacOsWorkgroupJoinToken {
    sig: u32,
    opaque: [u8; 36],
}

#[cfg(target_os = "macos")]
const _: [(); 40] = [(); std::mem::size_of::<MacOsWorkgroupJoinToken>()];
#[cfg(target_os = "macos")]
const _: [(); 8] = [(); std::mem::align_of::<MacOsWorkgroupJoinToken>()];

#[cfg(target_os = "macos")]
struct MacOsAudioWorkInterval {
    api: &'static MacOsAudioWorkIntervalApi,
    workgroup: *mut libc::c_void,
    join_token: MacOsWorkgroupJoinToken,
    period_ticks: u64,
    started_cycles: u64,
    finished_cycles: u64,
    start_failures: u64,
    finish_failures: u64,
    disabled: bool,
    joined: bool,
    poisoned: bool,
}

#[cfg(target_os = "macos")]
impl MacOsAudioWorkInterval {
    fn acquire() -> Result<Option<Self>, String> {
        const OS_CLOCK_MACH_ABSOLUTE_TIME: u32 = 32;

        let Some(api) = MacOsAudioWorkIntervalApi::get() else {
            return Ok(None);
        };
        let period_ticks = macos_absolute_ticks(MACOS_NEURAL_PERIOD_NANOS)?;
        let workgroup = unsafe {
            (api.create)(
                c"denoize Neural".as_ptr(),
                OS_CLOCK_MACH_ABSOLUTE_TIME,
                std::ptr::null_mut(),
            )
        };
        if workgroup.is_null() {
            return Err("create macOS audio work interval".to_owned());
        }
        let mut join_token = MacOsWorkgroupJoinToken {
            sig: 0,
            opaque: [0; 36],
        };
        let result = unsafe { (api.join)(workgroup, &mut join_token) };
        if result != 0 {
            unsafe { (api.release)(workgroup) };
            return Err(format!(
                "join macOS audio work interval: {}",
                std::io::Error::from_raw_os_error(result)
            ));
        }
        Ok(Some(Self {
            api,
            workgroup,
            join_token,
            period_ticks,
            started_cycles: 0,
            finished_cycles: 0,
            start_failures: 0,
            finish_failures: 0,
            disabled: false,
            joined: true,
            poisoned: false,
        }))
    }

    fn run_cycle<T>(&mut self, process: impl FnOnce() -> T) -> T {
        // An unwind may have prevented `run_cycle` from observing a failed
        // finish. Detach before doing any more work if the cycle guard marked
        // this interval as poisoned while unwinding.
        if self.poisoned {
            self.disable_poisoned();
        }
        if self.disabled {
            return process();
        }
        let start = unsafe { mach2::mach_time::mach_absolute_time() };
        let deadline = start.saturating_add(self.period_ticks);
        let result =
            unsafe { (self.api.start)(self.workgroup, start, deadline, std::ptr::null_mut()) };
        if result != 0 {
            self.start_failures = self.start_failures.saturating_add(1);
            // libdispatch clears its STARTED state on every start failure, so
            // this object remains safe to leave and release before falling
            // back to the existing Mach time-constraint policy.
            self.disable_cleanly();
            return process();
        }
        self.started_cycles = self.started_cycles.saturating_add(1);
        let cycle = MacOsAudioWorkIntervalCycle {
            api: self.api,
            workgroup: self.workgroup,
            active: true,
            poisoned: &mut self.poisoned,
            finished_cycles: &mut self.finished_cycles,
            finish_failures: &mut self.finish_failures,
        };
        let processed = process();
        if !cycle.finish() {
            // A failed finish leaves libdispatch's STARTED bit set. Leaving
            // clears the thread membership, but releasing the final reference
            // would deliberately abort the host. Detach immediately and leak
            // only this unusable reference in that exceptional state.
            self.disable_poisoned();
        }
        processed
    }

    fn leave(&mut self) {
        if self.joined {
            unsafe { (self.api.leave)(self.workgroup, &mut self.join_token) };
            self.joined = false;
        }
    }

    fn disable_cleanly(&mut self) {
        self.leave();
        unsafe { (self.api.release)(self.workgroup) };
        self.workgroup = std::ptr::null_mut();
        self.disabled = true;
    }

    fn disable_poisoned(&mut self) {
        self.poisoned = true;
        self.leave();
        // Releasing a workgroup whose interval is still STARTED is a
        // documented client-programming crash in libdispatch. Forget this
        // single reference instead; process teardown reclaims it.
        self.workgroup = std::ptr::null_mut();
        self.disabled = true;
    }
}

#[cfg(target_os = "macos")]
impl Drop for MacOsAudioWorkInterval {
    fn drop(&mut self) {
        if macos_scheduling_evidence_enabled() {
            eprintln!(
                "DENOIZE_MACOS_AUDIO_WORK_INTERVAL started_cycles={} finished_cycles={} start_failures={} finish_failures={}",
                self.started_cycles,
                self.finished_cycles,
                self.start_failures,
                self.finish_failures,
            );
        }
        if self.workgroup.is_null() {
            return;
        }
        self.leave();
        if !self.poisoned {
            unsafe { (self.api.release)(self.workgroup) };
        }
    }
}

#[cfg(target_os = "macos")]
struct MacOsAudioWorkIntervalCycle<'a> {
    api: &'static MacOsAudioWorkIntervalApi,
    workgroup: *mut libc::c_void,
    active: bool,
    poisoned: &'a mut bool,
    finished_cycles: &'a mut u64,
    finish_failures: &'a mut u64,
}

#[cfg(target_os = "macos")]
impl MacOsAudioWorkIntervalCycle<'_> {
    fn finish(mut self) -> bool {
        let result = unsafe { (self.api.finish)(self.workgroup, std::ptr::null_mut()) };
        self.active = false;
        if result == 0 {
            *self.finished_cycles = (*self.finished_cycles).saturating_add(1);
            true
        } else {
            // libdispatch only clears STARTED after a successful finish.
            *self.finish_failures = (*self.finish_failures).saturating_add(1);
            *self.poisoned = true;
            false
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for MacOsAudioWorkIntervalCycle<'_> {
    fn drop(&mut self) {
        if self.active {
            let result = unsafe { (self.api.finish)(self.workgroup, std::ptr::null_mut()) };
            if result == 0 {
                *self.finished_cycles = (*self.finished_cycles).saturating_add(1);
            } else {
                // Mark poison before unwinding continues. Logging here could
                // itself panic while another panic is already in flight.
                *self.finish_failures = (*self.finish_failures).saturating_add(1);
                *self.poisoned = true;
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn macos_scheduling_evidence_enabled() -> bool {
    std::env::var_os("DENOIZE_MACOS_SCHEDULING_EVIDENCE").is_some()
}

#[cfg(target_os = "macos")]
fn macos_absolute_ticks(nanoseconds: u64) -> Result<u64, String> {
    use mach2::kern_return::KERN_SUCCESS;
    use mach2::mach_time::{mach_timebase_info, mach_timebase_info_data_t};

    let mut timebase = mach_timebase_info_data_t { numer: 0, denom: 0 };
    let result = unsafe { mach_timebase_info(&mut timebase) };
    if result != KERN_SUCCESS || timebase.numer == 0 || timebase.denom == 0 {
        return Err(format!(
            "read macOS Mach timebase for neural worker: Mach error {result}"
        ));
    }
    nanoseconds
        .checked_mul(u64::from(timebase.denom))
        .and_then(|value| value.checked_div(u64::from(timebase.numer)))
        .ok_or_else(|| "convert neural worker period to Mach absolute time".to_owned())
}

#[cfg(target_os = "macos")]
fn acquire_macos_time_constraint(
) -> Result<(mach2::port::mach_port_t, MacOsPreviousMachPolicy), String> {
    use libc::thread_policy_t;
    use mach2::boolean::boolean_t;
    use mach2::kern_return::KERN_SUCCESS;
    use mach2::message::mach_msg_type_number_t;
    use mach2::thread_policy::{
        thread_extended_policy_data_t, thread_policy_get, thread_policy_set,
        thread_time_constraint_policy_data_t, THREAD_EXTENDED_POLICY, THREAD_EXTENDED_POLICY_COUNT,
        THREAD_TIME_CONSTRAINT_POLICY, THREAD_TIME_CONSTRAINT_POLICY_COUNT,
    };

    // pthread_mach_thread_np returns the non-owning Mach port for this exact
    // pthread, so the guard does not allocate a send right that needs separate
    // deallocation.
    let thread = unsafe { libc::pthread_mach_thread_np(libc::pthread_self()) };
    if thread == 0 {
        return Err("resolve neural inference worker Mach thread".into());
    }

    let mut previous_time_constraint = thread_time_constraint_policy_data_t {
        period: 0,
        computation: 0,
        constraint: 0,
        preemptible: 0,
    };
    let mut count: mach_msg_type_number_t = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
    let mut get_default: boolean_t = 0;
    // SAFETY: the port represents the calling pthread and both output buffers
    // are valid for the declared Mach policy size.
    let result = unsafe {
        thread_policy_get(
            thread,
            THREAD_TIME_CONSTRAINT_POLICY,
            (&mut previous_time_constraint as *mut _) as thread_policy_t,
            &mut count,
            &mut get_default,
        )
    };
    if result != KERN_SUCCESS || count != THREAD_TIME_CONSTRAINT_POLICY_COUNT {
        return Err(format!(
            "read neural inference worker macOS time constraint: Mach error {result}"
        ));
    }

    let previous = if get_default == 0 {
        MacOsPreviousMachPolicy::TimeConstraint(previous_time_constraint)
    } else {
        // A default time-constraint response means this thread was not in the
        // real-time band. Capture its actual time-sharing/fixed mode so Drop
        // can leave the real-time band instead of installing default-looking
        // values as a new time constraint.
        let mut extended = thread_extended_policy_data_t { timeshare: 1 };
        let mut count: mach_msg_type_number_t = THREAD_EXTENDED_POLICY_COUNT;
        let mut get_default: boolean_t = 0;
        let result = unsafe {
            thread_policy_get(
                thread,
                THREAD_EXTENDED_POLICY,
                (&mut extended as *mut _) as thread_policy_t,
                &mut count,
                &mut get_default,
            )
        };
        if result != KERN_SUCCESS || count != THREAD_EXTENDED_POLICY_COUNT {
            return Err(format!(
                "read neural inference worker macOS extended policy: Mach error {result}"
            ));
        }
        MacOsPreviousMachPolicy::Extended(extended)
    };

    let absolute_ticks = |nanoseconds: u64| -> Result<u32, String> {
        u32::try_from(macos_absolute_ticks(nanoseconds)?)
            .map_err(|_| "neural worker Mach time constraint exceeds u32".to_owned())
    };
    // Direct production-path measurements put ordinary inference below 7 ms
    // at p99. Advertise 8 ms of nominal computation in each 10 ms arrival
    // period, leaving a real deadline margin rather than claiming the complete
    // period as uninterrupted CPU demand.
    let mut active = thread_time_constraint_policy_data_t {
        period: absolute_ticks(MACOS_NEURAL_PERIOD_NANOS)?,
        computation: absolute_ticks(MACOS_NEURAL_COMPUTATION_NANOS)?,
        constraint: absolute_ticks(MACOS_NEURAL_PERIOD_NANOS)?,
        preemptible: 1,
    };
    // SAFETY: this changes only the calling pthread. The !Send guard retains
    // its prior policy and restores it on this same thread.
    let result = unsafe {
        thread_policy_set(
            thread,
            THREAD_TIME_CONSTRAINT_POLICY,
            (&mut active as *mut _) as thread_policy_t,
            THREAD_TIME_CONSTRAINT_POLICY_COUNT,
        )
    };
    if result != KERN_SUCCESS {
        return Err(format!(
            "register neural inference worker with macOS time constraint: Mach error {result}"
        ));
    }
    if macos_scheduling_evidence_enabled() {
        eprintln!(
            "DENOIZE_MACOS_TIME_CONSTRAINT period_ns={MACOS_NEURAL_PERIOD_NANOS} computation_ns={MACOS_NEURAL_COMPUTATION_NANOS} constraint_ns={MACOS_NEURAL_PERIOD_NANOS}"
        );
    }
    Ok((thread, previous))
}

/// Closed model identities exposed by the off-callback neural plug-ins.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NeuralDawModel {
    #[default]
    Gtcrn,
    Dpdfnet2,
}

impl NeuralDawModel {
    pub const fn plugin_id(self) -> &'static str {
        match self {
            Self::Gtcrn => NEURAL_DAW_PLUGIN_ID,
            Self::Dpdfnet2 => NEURAL_HQ_DAW_PLUGIN_ID,
        }
    }

    pub const fn model_id(self) -> &'static str {
        match self {
            Self::Gtcrn => NEURAL_DAW_MODEL_ID,
            Self::Dpdfnet2 => NEURAL_HQ_DAW_MODEL_ID,
        }
    }

    pub const fn model_sha256(self) -> &'static str {
        match self {
            Self::Gtcrn => NEURAL_DAW_MODEL_SHA256,
            Self::Dpdfnet2 => NEURAL_HQ_DAW_MODEL_SHA256,
        }
    }

    pub const fn backend(self) -> &'static str {
        match self {
            Self::Gtcrn => "gtcrn",
            Self::Dpdfnet2 => "dpdfnet",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Gtcrn => "denoize Neural",
            Self::Dpdfnet2 => "denoize Neural HQ",
        }
    }

    fn from_identity(plugin_id: &str, model_id: &str, model_sha256: &str) -> Option<Self> {
        [Self::Gtcrn, Self::Dpdfnet2].into_iter().find(|model| {
            plugin_id == model.plugin_id()
                && model_id == model.model_id()
                && model_sha256 == model.model_sha256()
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NeuralDawOverloadFallback {
    #[default]
    DelayedDry,
    LastSafeGain,
    Silence,
}

impl NeuralDawOverloadFallback {
    pub const fn index(self) -> u32 {
        match self {
            Self::DelayedDry => 0,
            Self::LastSafeGain => 1,
            Self::Silence => 2,
        }
    }

    pub const fn from_index(index: u32) -> Self {
        match index {
            1 => Self::LastSafeGain,
            2 => Self::Silence,
            _ => Self::DelayedDry,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::DelayedDry => "Delayed Dry",
            Self::LastSafeGain => "Last Safe Gain",
            Self::Silence => "Silence",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().replace([' ', '_'], "-").as_str() {
            "delayed-dry" | "dry" => Some(Self::DelayedDry),
            "last-safe-gain" | "gain" => Some(Self::LastSafeGain),
            "silence" => Some(Self::Silence),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeuralDawParameters {
    pub bypass: bool,
    pub mix: f32,
    pub output_gain_db: f32,
    pub overload_fallback: NeuralDawOverloadFallback,
}

impl Default for NeuralDawParameters {
    fn default() -> Self {
        Self {
            bypass: false,
            mix: 1.0,
            output_gain_db: 0.0,
            overload_fallback: NeuralDawOverloadFallback::DelayedDry,
        }
    }
}

impl NeuralDawParameters {
    pub fn validate(&self) -> Result<(), String> {
        if !self.mix.is_finite() || !(0.0..=1.0).contains(&self.mix) {
            return Err("neural plug-in mix must be finite and within [0, 1]".into());
        }
        if !self.output_gain_db.is_finite() || !(-24.0..=24.0).contains(&self.output_gain_db) {
            return Err("neural plug-in output gain must be finite and within [-24, 24] dB".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NeuralDawPortConfiguration {
    Mono,
    #[default]
    Stereo,
}

impl NeuralDawPortConfiguration {
    pub const fn channels(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeuralDawSessionState {
    pub schema: String,
    pub schema_version: u32,
    pub plugin_id: String,
    pub model_id: String,
    pub model_sha256: String,
    pub latency_policy: String,
    pub port_configuration: NeuralDawPortConfiguration,
    pub parameters: NeuralDawParameters,
}

impl NeuralDawSessionState {
    pub fn new(
        port_configuration: NeuralDawPortConfiguration,
        parameters: NeuralDawParameters,
    ) -> Result<Self, String> {
        Self::new_for_model(NeuralDawModel::Gtcrn, port_configuration, parameters)
    }

    pub fn new_for_model(
        model: NeuralDawModel,
        port_configuration: NeuralDawPortConfiguration,
        parameters: NeuralDawParameters,
    ) -> Result<Self, String> {
        let state = Self {
            schema: NEURAL_DAW_SESSION_SCHEMA.to_owned(),
            schema_version: NEURAL_DAW_SESSION_SCHEMA_VERSION,
            plugin_id: model.plugin_id().to_owned(),
            model_id: model.model_id().to_owned(),
            model_sha256: model.model_sha256().to_owned(),
            latency_policy: NEURAL_DAW_LATENCY_POLICY.to_owned(),
            port_configuration,
            parameters,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != NEURAL_DAW_SESSION_SCHEMA
            || self.schema_version != NEURAL_DAW_SESSION_SCHEMA_VERSION
        {
            return Err(format!(
                "unsupported neural DAW session {} version {}; expected {} version {}",
                self.schema,
                self.schema_version,
                NEURAL_DAW_SESSION_SCHEMA,
                NEURAL_DAW_SESSION_SCHEMA_VERSION
            ));
        }
        NeuralDawModel::from_identity(&self.plugin_id, &self.model_id, &self.model_sha256)
            .ok_or_else(|| "neural DAW session identity does not match this build".to_string())?;
        if self.latency_policy != NEURAL_DAW_LATENCY_POLICY {
            return Err(format!(
                "unsupported neural DAW latency policy {}; expected {}",
                self.latency_policy, NEURAL_DAW_LATENCY_POLICY
            ));
        }
        self.parameters.validate()
    }

    pub fn validate_for_model(&self, model: NeuralDawModel) -> Result<(), String> {
        self.validate()?;
        if self.plugin_id != model.plugin_id()
            || self.model_id != model.model_id()
            || self.model_sha256 != model.model_sha256()
        {
            return Err(format!(
                "neural DAW session targets {}/{}, expected {}/{}",
                self.plugin_id,
                self.model_id,
                model.plugin_id(),
                model.model_id()
            ));
        }
        Ok(())
    }

    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        serialize_bounded(self, "neural DAW session")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() as u64 > MAX_DAW_DOCUMENT_BYTES {
            return Err(format!(
                "neural DAW session is {} bytes, exceeding the {MAX_DAW_DOCUMENT_BYTES}-byte limit",
                bytes.len()
            ));
        }
        let state: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse neural DAW session JSON: {error}"))?;
        state.validate()?;
        Ok(state)
    }
}

pub fn read_neural_daw_session(path: impl AsRef<Path>) -> Result<NeuralDawSessionState, String> {
    NeuralDawSessionState::from_bytes(&read_bounded_regular_file(
        path.as_ref(),
        "neural DAW session",
    )?)
}

pub fn write_neural_daw_session(
    path: impl AsRef<Path>,
    state: &NeuralDawSessionState,
    mode: CommitMode,
) -> Result<(), String> {
    write_document(
        path.as_ref(),
        &state.to_canonical_bytes()?,
        mode,
        "neural DAW session",
    )
}

pub fn neural_daw_chunk_frames(sample_rate: f64) -> Result<u32, String> {
    validate_sample_rate(sample_rate)?;
    let frames = (sample_rate * f64::from(NEURAL_DAW_CHUNK_MILLIS) / 1_000.0).ceil();
    if !frames.is_finite() || frames < 1.0 || frames > f64::from(u32::MAX) {
        return Err("neural DAW chunk geometry exceeds the public contract".to_owned());
    }
    Ok(frames as u32)
}

pub fn neural_daw_latency_frames(sample_rate: f64) -> Result<u32, String> {
    neural_daw_chunk_frames(sample_rate)?
        .checked_mul(NEURAL_DAW_LATENCY_CHUNKS)
        .ok_or_else(|| "neural DAW latency geometry overflow".to_owned())
}

pub fn neural_daw_latency_millis(sample_rate: f64) -> Result<f64, String> {
    Ok(f64::from(neural_daw_latency_frames(sample_rate)?) * 1_000.0 / sample_rate)
}

fn validate_sample_rate(sample_rate: f64) -> Result<(), String> {
    if !sample_rate.is_finite()
        || sample_rate < 1.0
        || sample_rate > f64::from(NEURAL_DAW_MAX_SAMPLE_RATE)
    {
        return Err(format!(
            "neural DAW processing requires a finite sample rate within [1, {NEURAL_DAW_MAX_SAMPLE_RATE}], got {sample_rate}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trips_and_rejects_unknown_or_mismatched_identity() {
        let state = NeuralDawSessionState::new(
            NeuralDawPortConfiguration::Stereo,
            NeuralDawParameters::default(),
        )
        .unwrap();
        let bytes = state.to_canonical_bytes().unwrap();
        assert_eq!(NeuralDawSessionState::from_bytes(&bytes).unwrap(), state);

        let mut object = serde_json::to_value(&state).unwrap();
        object["future"] = serde_json::json!(true);
        assert!(serde_json::from_value::<NeuralDawSessionState>(object).is_err());

        let mut mismatch = state;
        mismatch.model_sha256.replace_range(..1, "0");
        assert!(mismatch.validate().is_err());

        let hq = NeuralDawSessionState::new_for_model(
            NeuralDawModel::Dpdfnet2,
            NeuralDawPortConfiguration::Mono,
            NeuralDawParameters::default(),
        )
        .unwrap();
        hq.validate_for_model(NeuralDawModel::Dpdfnet2).unwrap();
        assert!(hq.validate_for_model(NeuralDawModel::Gtcrn).is_err());
        assert_eq!(
            NeuralDawSessionState::from_bytes(&hq.to_canonical_bytes().unwrap()).unwrap(),
            hq
        );
    }

    #[test]
    fn latency_is_a_closed_finite_rate_contract() {
        assert_eq!(neural_daw_chunk_frames(44_100.0).unwrap(), 441);
        assert_eq!(neural_daw_latency_frames(44_100.0).unwrap(), 10_584);
        assert_eq!(neural_daw_latency_frames(48_000.0).unwrap(), 11_520);
        assert_eq!(neural_daw_latency_frames(96_000.0).unwrap(), 23_040);
        assert_eq!(neural_daw_chunk_frames(44_100.5).unwrap(), 442);
        assert_eq!(neural_daw_latency_frames(44_100.5).unwrap(), 10_608);
        assert!(neural_daw_latency_frames(f64::NAN).is_err());
        assert!(neural_daw_latency_frames(0.0).is_err());
        assert_eq!(neural_daw_chunk_frames(1_234_567.8).unwrap(), 12_346);
        assert_eq!(neural_daw_latency_frames(1_234_567.8).unwrap(), 296_304);
        assert!(neural_daw_latency_frames(f64::from(NEURAL_DAW_MAX_SAMPLE_RATE) + 0.1).is_err());
    }

    #[test]
    fn session_publication_is_no_clobber_and_handles_symlinks_safely() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("neural.json");
        let state = NeuralDawSessionState::new(
            NeuralDawPortConfiguration::Mono,
            NeuralDawParameters::default(),
        )
        .unwrap();
        write_neural_daw_session(&path, &state, CommitMode::NoClobber).unwrap();
        assert_eq!(read_neural_daw_session(&path).unwrap(), state);
        assert!(write_neural_daw_session(&path, &state, CommitMode::NoClobber).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let link = directory.path().join("neural-link.json");
            symlink(&path, &link).unwrap();
            assert!(read_neural_daw_session(&link).is_err());
            assert!(write_neural_daw_session(&link, &state, CommitMode::NoClobber).is_err());
            write_neural_daw_session(&link, &state, CommitMode::Replace).unwrap();
            assert!(!std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink());
            assert_eq!(read_neural_daw_session(&link).unwrap(), state);
            assert_eq!(read_neural_daw_session(&path).unwrap(), state);
        }
    }
}
