//! Entry-thread broker for TRUEOS child workers and parent stdio streams.
//!
//! The child Hull ABI requires the original Hull stack. Native std threads
//! therefore submit bounded requests to the future driving that entry thread.
//! The stream has no exit code or captured stderr: callers must validate their
//! complete success protocol after the child stops.

use anyhow::{anyhow, Context, Result};
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::future::Future;
use std::io;
use std::marker::PhantomData;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock, Weak};
use std::task::{Context as TaskContext, Poll, Waker};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

const REQUEST_CAPACITY: usize = 32;
const READ_CHUNK: usize = 3000;
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const EXITED: i32 = 4;
pub(super) const START_FAILURE: &str = "native extraction child Hull could not start";

static ACTIVE: OnceLock<Mutex<Option<Weak<State>>>> = OnceLock::new();

#[derive(Debug)]
enum Operation {
    Spawn(Vec<u8>),
    Read(u64, usize),
    Send(u64, Vec<u8>),
    Status(u64),
    Terminate(u64),
}

#[derive(Debug)]
enum Reply {
    Handle(u64),
    Bytes(Vec<u8>),
    Written(usize),
    Status(i32),
    Unit,
}

trait Backend: Send + Sync {
    fn execute(&self, operation: &Operation) -> io::Result<Reply>;
}

fn child_api_error(code: i32) -> io::Error {
    let kind = match code {
        -3 => io::ErrorKind::WouldBlock,
        -2 => io::ErrorKind::BrokenPipe,
        -1 => io::ErrorKind::InvalidInput,
        -4 => io::ErrorKind::Unsupported,
        _ => io::ErrorKind::Other,
    };
    io::Error::new(kind, format!("child Hull API returned {code}"))
}

#[cfg(target_os = "trueos")]
struct Kernel;

#[cfg(target_os = "trueos")]
impl Backend for Kernel {
    fn execute(&self, operation: &Operation) -> io::Result<Reply> {
        let error = child_api_error;
        match operation {
            Operation::Spawn(request) => trueos::child_hull::spawn_program(request)
                .map(Reply::Handle)
                .map_err(error),
            Operation::Read(handle, cap) => {
                let mut bytes = vec![0; (*cap).min(READ_CHUNK)];
                if bytes.is_empty() {
                    return Ok(Reply::Bytes(bytes));
                }
                let read = trueos::child_hull::read(*handle, &mut bytes).map_err(error)?;
                if read > bytes.len() {
                    return Err(io::Error::other("child Hull returned an invalid read size"));
                }
                bytes.truncate(read);
                Ok(Reply::Bytes(bytes))
            }
            Operation::Send(handle, bytes) => {
                let bytes = &bytes[..bytes.len().min(READ_CHUNK)];
                let written = trueos::child_hull::send(*handle, bytes).map_err(error)?;
                if written > bytes.len() {
                    return Err(io::Error::other(
                        "child Hull returned an invalid write size",
                    ));
                }
                Ok(Reply::Written(written))
            }
            Operation::Status(handle) => {
                let status = trueos::child_hull::status(*handle);
                if status < 0 {
                    Err(error(status))
                } else {
                    Ok(Reply::Status(status))
                }
            }
            Operation::Terminate(handle) => trueos::child_hull::terminate(*handle)
                .map(|()| Reply::Unit)
                .map_err(error),
        }
    }
}

struct Request {
    operation: Operation,
    reply: mpsc::SyncSender<io::Result<Reply>>,
}

struct State {
    entry: ThreadId,
    requests: mpsc::SyncSender<Request>,
    waker: Mutex<Option<Waker>>,
    running: AtomicBool,
    children: Mutex<BTreeSet<u64>>,
    backend: Arc<dyn Backend>,
}

fn disconnected() -> io::Error {
    io::Error::new(
        io::ErrorKind::BrokenPipe,
        "native worker broker has stopped",
    )
}

impl State {
    fn wake(&self) {
        let waker = self.waker.lock().unwrap().clone();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn execute(&self, operation: Operation) -> io::Result<Reply> {
        debug_assert_eq!(self.entry, std::thread::current().id());
        // Never hold a registry or channel lock across a native ABI call.
        let result = self.backend.execute(&operation);
        match (&operation, &result) {
            (Operation::Spawn(_), Ok(Reply::Handle(0))) => {
                return Err(io::Error::other(
                    "native child spawn returned the parent handle",
                ));
            }
            (Operation::Spawn(_), Ok(Reply::Handle(handle))) => {
                self.children.lock().unwrap().insert(*handle);
            }
            (Operation::Terminate(handle), Ok(Reply::Unit)) => {
                self.children.lock().unwrap().remove(handle);
            }
            _ => {}
        }
        result
    }

    fn call(&self, operation: Operation) -> io::Result<Reply> {
        if !self.running.load(Ordering::Acquire) {
            return Err(disconnected());
        }
        if self.entry == std::thread::current().id() {
            return self.execute(operation);
        }
        let (reply, receive) = mpsc::sync_channel(1);
        // Wake before a potentially blocking bounded send, and afterwards so
        // a poll racing the enqueue cannot miss the request.
        self.wake();
        self.requests
            .send(Request { operation, reply })
            .map_err(|_| disconnected())?;
        self.wake();
        receive.recv().map_err(|_| disconnected())?
    }
}

struct Broker<F> {
    future: Pin<Box<F>>,
    state: Arc<State>,
    requests: Option<mpsc::Receiver<Request>>,
    // Construction, polling and cleanup must remain on the Hull entry thread.
    _entry_only: PhantomData<Rc<()>>,
}

fn broker_with_backend<F: Future>(future: F, backend: Arc<dyn Backend>) -> Broker<F> {
    let (send, receive) = mpsc::sync_channel(REQUEST_CAPACITY);
    let state = Arc::new(State {
        entry: std::thread::current().id(),
        requests: send,
        waker: Mutex::new(None),
        running: AtomicBool::new(true),
        children: Mutex::new(BTreeSet::new()),
        backend,
    });
    let mut active = ACTIVE.get_or_init(|| Mutex::new(None)).lock().unwrap();
    assert!(
        active.as_ref().and_then(Weak::upgrade).is_none(),
        "only one entry-thread worker broker may run at once"
    );
    *active = Some(Arc::downgrade(&state));
    Broker {
        future: Box::pin(future),
        state,
        requests: Some(receive),
        _entry_only: PhantomData,
    }
}

/// Drive native extraction requests on the original Hull entry thread while
/// polling `future`. Wrap the future passed to `Runtime::block_on`; the wrapper
/// is deliberately not Send. Dropping it disconnects waiters and stops every
/// child still owned by this broker.
#[cfg(target_os = "trueos")]
pub fn with_broker<F: Future>(future: F) -> impl Future<Output = F::Output> {
    broker_with_backend(future, Arc::new(Kernel))
}

impl<F: Future> Future for Broker<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert_eq!(this.state.entry, std::thread::current().id());
        *this.state.waker.lock().unwrap() = Some(cx.waker().clone());
        let receiver = this.requests.as_ref().unwrap();
        for _ in 0..REQUEST_CAPACITY {
            match receiver.try_recv() {
                Ok(request) => {
                    let result = this.state.execute(request.operation);
                    let _ = request.reply.send(result);
                }
                Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => {
                    return this.future.as_mut().poll(cx);
                }
            }
        }
        cx.waker().wake_by_ref();
        this.future.as_mut().poll(cx)
    }
}

impl<F> Drop for Broker<F> {
    fn drop(&mut self) {
        self.state.running.store(false, Ordering::Release);
        // Release queued replies and blocked senders before native cleanup.
        drop(self.requests.take());
        let mut active = ACTIVE.get().unwrap().lock().unwrap();
        if active
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some_and(|state| Arc::ptr_eq(&state, &self.state))
        {
            *active = None;
        }
        drop(active);
        let handles = std::mem::take(&mut *self.state.children.lock().unwrap());
        for handle in handles {
            if let Err(error) = self.state.backend.execute(&Operation::Terminate(handle)) {
                eprintln!("could not stop extraction child Hull {handle}: {error}");
            }
        }
    }
}

fn active_state() -> io::Result<Arc<State>> {
    ACTIVE
        .get()
        .and_then(|active| active.lock().unwrap().as_ref().and_then(Weak::upgrade))
        .ok_or_else(|| io::Error::other("native child I/O requires the entry-thread worker broker"))
}

/// Read a bounded portion of the parent stream without consuming bytes beyond
/// the caller's buffer. Empty reads mean pending until `parent_status` reports
/// exit; the async adapter owns that distinction. Handle zero is never owned.
pub fn stdio_read(out: &mut [u8]) -> io::Result<usize> {
    if out.is_empty() {
        return Ok(0);
    }
    let cap = out.len().min(READ_CHUNK);
    let Reply::Bytes(bytes) = active_state()?.call(Operation::Read(0, cap))? else {
        return Err(io::Error::other(
            "native stdio broker returned an invalid read reply",
        ));
    };
    if bytes.len() > cap {
        return Err(io::Error::other(
            "native stdio broker exceeded the requested read capacity",
        ));
    }
    out[..bytes.len()].copy_from_slice(&bytes);
    Ok(bytes.len())
}

/// Submit one bounded parent-stream frame. A full native queue is WouldBlock;
/// the async adapter retries after a timer instead of losing output bytes.
pub fn stdio_write(bytes: &[u8]) -> io::Result<usize> {
    if bytes.is_empty() {
        return Ok(0);
    }
    let bytes = &bytes[..bytes.len().min(READ_CHUNK)];
    let Reply::Written(written) = active_state()?.call(Operation::Send(0, bytes.to_vec()))? else {
        return Err(io::Error::other(
            "native stdio broker returned an invalid write reply",
        ));
    };
    if written > bytes.len() {
        return Err(io::Error::other(
            "native stdio broker exceeded the submitted write size",
        ));
    }
    Ok(written)
}

/// Observe the parent lifetime on the entry thread, including from Tokio
/// worker callers. This never registers or terminates handle zero as a child.
pub fn parent_status() -> io::Result<i32> {
    let Reply::Status(status) = active_state()?.call(Operation::Status(0))? else {
        return Err(io::Error::other(
            "native stdio broker returned an invalid status reply",
        ));
    };
    Ok(status)
}

struct Child {
    state: Arc<State>,
    handle: u64,
}

impl Drop for Child {
    fn drop(&mut self) {
        if self.state.running.load(Ordering::Acquire) {
            if let Err(error) = self.state.call(Operation::Terminate(self.handle)) {
                eprintln!(
                    "could not stop extraction child Hull {}: {error}",
                    self.handle
                );
            }
        }
    }
}

/// One live parser Hull at a time: native descriptor usage cannot be probed by
/// rlimit/procfs. This is an explicit admission policy, not a measured FD cap.
static PARSER: Mutex<()> = Mutex::new(());

pub(super) fn run(
    worker_env: &str,
    args: &[OsString],
    output_cap: usize,
    timeout: Duration,
) -> Result<Vec<u8>> {
    let _parser = PARSER.lock().unwrap_or_else(|error| error.into_inner());
    let child = spawn(worker_env, args).context(START_FAILURE)?;
    #[cfg(target_os = "trueos")]
    super::rlimit::native_worker_started();
    collect(&child.state, child.handle, output_cap, timeout)
}

fn spawn(worker_env: &str, args: &[OsString]) -> Result<Child> {
    let state = active_state()?;
    let program = std::env::var_os(worker_env)
        .unwrap_or_else(|| OsString::from("xerj"))
        .into_string()
        .map_err(|_| anyhow!("{worker_env} must be UTF-8 for native child launch"))?;
    let args: Vec<&str> = args
        .iter()
        .map(|arg| {
            arg.to_str()
                .ok_or_else(|| anyhow!("native worker paths must be UTF-8"))
        })
        .collect::<Result<_>>()?;
    let env: std::collections::BTreeMap<String, String> = std::env::vars_os()
        .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
        .filter(|(key, _)| !key.starts_with("TRUEOS_"))
        .collect();
    let cwd = std::env::current_dir().context("locate native extraction working directory")?;
    let cwd = cwd
        .to_str()
        .ok_or_else(|| anyhow!("native extraction working directory must be UTF-8"))?;
    let request = serde_json::to_vec(&serde_json::json!({
        "program": program, "args": args, "env": env, "cwd": cwd,
    }))?;
    let Reply::Handle(handle) = state.call(Operation::Spawn(request))? else {
        return Err(anyhow!(
            "native worker broker returned an invalid spawn reply"
        ));
    };
    Ok(Child { state, handle })
}

fn collect(state: &State, handle: u64, output_cap: usize, timeout: Duration) -> Result<Vec<u8>> {
    let started = Instant::now();
    let mut output = Vec::new();
    let mut exited = false;
    loop {
        if started.elapsed() >= timeout {
            return Err(anyhow!(
                "native extraction worker exceeded its {}s wall-clock limit",
                timeout.as_secs()
            ));
        }
        let Reply::Bytes(bytes) = state.call(Operation::Read(handle, READ_CHUNK))? else {
            return Err(anyhow!(
                "native worker broker returned an invalid read reply"
            ));
        };
        if !bytes.is_empty() {
            if bytes.len() > output_cap.saturating_sub(output.len()) {
                return Err(anyhow!(
                    "native extraction worker exceeded its {output_cap}-byte output limit"
                ));
            }
            output.extend_from_slice(&bytes);
            continue;
        }
        // An empty read is pending while alive. After EXITED, do another read
        // to drain bytes queued between the first read and status observation.
        if exited {
            return Ok(output);
        }
        let Reply::Status(status) = state.call(Operation::Status(handle))? else {
            return Err(anyhow!(
                "native worker broker returned an invalid status reply"
            ));
        };
        match status {
            EXITED => exited = true,
            1..=3 => std::thread::sleep(POLL_INTERVAL),
            _ => {
                return Err(anyhow!(
                    "native extraction worker returned invalid status {status}"
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests;
