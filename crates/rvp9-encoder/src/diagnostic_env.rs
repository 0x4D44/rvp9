//! Production environment reads with isolated unit-test diagnostic overrides.
//!
//! Unit tests never mutate the process environment. An absent map entry reads
//! the launch environment; an explicit `None` masks it. Worker jobs capture the
//! submitting thread's overrides and restore their previous state on completion.

#[cfg(not(test))]
pub(crate) use std::env::{var, var_os};

#[cfg(not(test))]
#[inline]
pub(crate) fn inherit<F>(job: F) -> F {
    job
}

#[cfg(test)]
pub(crate) use local::*;

#[cfg(test)]
mod local {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::ffi::{OsStr, OsString};
    use std::marker::PhantomData;
    use std::rc::Rc;

    type Overrides = BTreeMap<OsString, Option<OsString>>;

    thread_local! {
        static OVERRIDES: RefCell<Overrides> = const { RefCell::new(BTreeMap::new()) };
    }

    /// Restores exactly the previous overrides, including during unwinding.
    /// The guard must be dropped on the thread where it was installed.
    pub(crate) struct Scope {
        previous: Overrides,
        _thread: PhantomData<Rc<()>>,
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            OVERRIDES.with(|values| *values.borrow_mut() = std::mem::take(&mut self.previous));
        }
    }

    pub(crate) fn scope() -> Scope {
        Scope {
            previous: OVERRIDES.with(|values| values.borrow().clone()),
            _thread: PhantomData,
        }
    }

    pub(crate) fn set_var(key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
        OVERRIDES.with(|values| {
            values
                .borrow_mut()
                .insert(key.as_ref().into(), Some(value.as_ref().into()));
        });
    }

    pub(crate) fn remove_var(key: impl AsRef<OsStr>) {
        OVERRIDES.with(|values| {
            values.borrow_mut().insert(key.as_ref().into(), None);
        });
    }

    pub(crate) fn var_os(key: impl AsRef<OsStr>) -> Option<OsString> {
        OVERRIDES
            .with(|values| values.borrow().get(key.as_ref()).cloned())
            .unwrap_or_else(|| std::env::var_os(key))
    }

    pub(crate) fn var(key: impl AsRef<OsStr>) -> Result<String, std::env::VarError> {
        var_os(key)
            .ok_or(std::env::VarError::NotPresent)?
            .into_string()
            .map_err(std::env::VarError::NotUnicode)
    }

    pub(crate) fn inherit<F: FnOnce() -> R, R>(job: F) -> impl FnOnce() -> R {
        let snapshot = OVERRIDES.with(|values| values.borrow().clone());
        move || {
            let _restore = scope();
            OVERRIDES.with(|values| *values.borrow_mut() = snapshot);
            job()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_override_scope_restores_nested_and_unwound_values() {
        let key = "RVP9_FIXED_Q";
        let original = var_os(key);
        {
            let _outer = scope();
            remove_var(key);
            assert_eq!(var_os(key), None);
            {
                let _inner = scope();
                set_var(key, "100");
                assert_eq!(var(key).as_deref(), Ok("100"));
            }
            assert_eq!(var_os(key), None);
            let result = std::panic::catch_unwind(|| {
                let _inner = scope();
                set_var(key, "200");
                panic!("exercise diagnostic scope restoration");
            });
            assert!(result.is_err());
            assert_eq!(var_os(key), None);
        }
        assert_eq!(var_os(key), original);
    }

    #[test]
    fn diagnostic_override_workers_inherit_snapshot_and_restore_between_jobs() {
        use rvp9_core::tile_threading::WorkerPool;
        use std::num::NonZeroUsize;

        let key = "RVP9_HIGHMV";
        // A single persistent worker guarantees both jobs run on the same OS
        // thread, rather than accidentally passing because each gets a fresh TLS.
        let mut pool = WorkerPool::new(NonZeroUsize::MIN);
        let mut observed = None;
        let worker_id;
        {
            let _scope = scope();
            set_var(key, "1");
            let job = inherit(|| {
                assert!(crate::frame_allow_hp(100));
                observed = Some(std::thread::current().id());
                // A worker-local mutation must also disappear after its job.
                set_var(key, "worker-only");
            });
            // Submission uses the captured snapshot, not the current override.
            set_var(key, "0");
            pool.scope(|tasks| tasks.submit(job));
            worker_id = observed.expect("worker executed");
            assert_eq!(var(key).as_deref(), Ok("0"));
        }
        let baseline = std::env::var_os(key);
        pool.scope(|tasks| {
            tasks.submit(|| {
                assert_eq!(std::thread::current().id(), worker_id);
                assert_eq!(var_os(key), baseline);
            })
        });
        {
            let _scope = scope();
            remove_var(key);
            pool.scope(|tasks| tasks.submit(inherit(|| assert_eq!(var_os(key), None))));
        }
    }
}
