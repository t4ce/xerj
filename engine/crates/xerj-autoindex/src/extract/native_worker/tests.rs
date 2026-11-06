use super::*;
use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::task::Wake;

// ACTIVE is process-wide, just as the native entry-thread broker is.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|error| error.into_inner())
}

struct WakeThread(std::thread::Thread);

impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn drive<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = TaskContext::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    let started = Instant::now();
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            return result;
        }
        assert!(started.elapsed() < Duration::from_secs(5), "broker stalled");
        std::thread::park_timeout(Duration::from_millis(2));
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Expected {
    Spawn,
    Read(u64, usize),
    Send(u64, Vec<u8>),
    Status(u64),
    Terminate(u64),
}

impl Expected {
    fn of(operation: &Operation) -> Self {
        match operation {
            Operation::Spawn(_) => Self::Spawn,
            Operation::Read(handle, cap) => Self::Read(*handle, *cap),
            Operation::Status(handle) => Self::Status(*handle),
            Operation::Send(handle, bytes) => Self::Send(*handle, bytes.clone()),
            Operation::Terminate(handle) => Self::Terminate(*handle),
        }
    }
}

struct Scripted {
    steps: Mutex<VecDeque<(Expected, Reply)>>,
    threads: Mutex<Vec<ThreadId>>,
}

impl Scripted {
    fn new(steps: impl IntoIterator<Item = (Expected, Reply)>) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(steps.into_iter().collect()),
            threads: Mutex::new(Vec::new()),
        })
    }

    fn assert_finished(&self) {
        assert!(self.steps.lock().unwrap().is_empty());
        assert!(self
            .threads
            .lock()
            .unwrap()
            .iter()
            .all(|thread| *thread == std::thread::current().id()));
    }
}

impl Backend for Scripted {
    fn execute(&self, operation: &Operation) -> io::Result<Reply> {
        self.threads
            .lock()
            .unwrap()
            .push(std::thread::current().id());
        let (expected, reply) = self
            .steps
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected backend operation");
        assert_eq!(Expected::of(operation), expected);
        Ok(reply)
    }
}

#[test]
fn entry_thread_calls_execute_directly_and_track_child_ownership() {
    let _serial = serial();
    let backend = Scripted::new([
        (Expected::Spawn, Reply::Handle(17)),
        (
            Expected::Read(17, READ_CHUNK),
            Reply::Bytes(b"native".to_vec()),
        ),
        (Expected::Terminate(17), Reply::Unit),
    ]);
    let broker = broker_with_backend(std::future::pending::<()>(), backend.clone());
    let state = active_state().unwrap();
    assert!(Arc::ptr_eq(&state, &broker.state));

    assert!(matches!(
        state.call(Operation::Spawn(vec![])),
        Ok(Reply::Handle(17))
    ));
    assert!(state.children.lock().unwrap().contains(&17));
    let Reply::Bytes(bytes) = state.call(Operation::Read(17, READ_CHUNK)).unwrap() else {
        panic!("unexpected read reply");
    };
    assert_eq!(bytes, b"native");
    state.call(Operation::Terminate(17)).unwrap();
    assert!(state.children.lock().unwrap().is_empty());
    drop(broker);
    assert!(active_state().is_err());
    backend.assert_finished();
}

#[test]
fn request_queue_is_bounded_and_a_poll_services_its_capacity() {
    let _serial = serial();
    let backend = Scripted::new(
        (0..REQUEST_CAPACITY).map(|handle| (Expected::Status(handle as u64), Reply::Status(2))),
    );
    let mut broker = broker_with_backend(std::future::pending::<()>(), backend.clone());
    let mut replies = Vec::new();
    for handle in 0..REQUEST_CAPACITY {
        let (reply, receive) = mpsc::sync_channel(1);
        broker
            .state
            .requests
            .try_send(Request {
                operation: Operation::Status(handle as u64),
                reply,
            })
            .unwrap();
        replies.push(receive);
    }
    let (reply, _receive) = mpsc::sync_channel(1);
    assert!(matches!(
        broker.state.requests.try_send(Request {
            operation: Operation::Status(1000),
            reply,
        }),
        Err(mpsc::TrySendError::Full(_))
    ));
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = TaskContext::from_waker(&waker);
    assert!(Pin::new(&mut broker).poll(&mut context).is_pending());
    for reply in replies {
        assert!(matches!(reply.try_recv(), Ok(Ok(Reply::Status(2)))));
    }
    backend.assert_finished();
}

struct ThreadRecording {
    calls: Mutex<Vec<(ThreadId, Expected)>>,
}

impl Backend for ThreadRecording {
    fn execute(&self, operation: &Operation) -> io::Result<Reply> {
        self.calls
            .lock()
            .unwrap()
            .push((std::thread::current().id(), Expected::of(operation)));
        match operation {
            Operation::Status(handle) => Ok(Reply::Status(*handle as i32)),
            Operation::Terminate(_) => Ok(Reply::Unit),
            Operation::Spawn(_) => Ok(Reply::Handle(71)),
            Operation::Read(_, _) | Operation::Send(_, _) => unreachable!(),
        }
    }
}

#[test]
fn concurrent_std_callers_run_every_backend_operation_on_the_entry_thread() {
    let _serial = serial();
    let entry = std::thread::current().id();
    let backend = Arc::new(ThreadRecording {
        calls: Mutex::new(Vec::new()),
    });
    let completed = Arc::new(AtomicUsize::new(0));
    let total = REQUEST_CAPACITY * 2 + 3;
    let finished = completed.clone();
    let broker = broker_with_backend(
        std::future::poll_fn(move |_| {
            if finished.load(Ordering::Acquire) == total {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }),
        backend.clone(),
    );
    let state = active_state().unwrap();
    let mut callers = Vec::new();
    for handle in 0..total {
        let state = state.clone();
        let completed = completed.clone();
        callers.push(std::thread::spawn(move || {
            assert_ne!(std::thread::current().id(), entry);
            let reply = state.call(Operation::Status(handle as u64)).unwrap();
            assert!(matches!(reply, Reply::Status(value) if value == handle as i32));
            completed.fetch_add(1, Ordering::Release);
        }));
    }
    drive(broker);
    for caller in callers {
        caller.join().unwrap();
    }
    let calls = backend.calls.lock().unwrap();
    assert_eq!(calls.len(), total);
    assert!(calls.iter().all(|(thread, _)| *thread == entry));
}

#[test]
fn collection_waits_through_pending_and_drains_bytes_after_exit() {
    let _serial = serial();
    let backend = Scripted::new([
        (Expected::Read(19, READ_CHUNK), Reply::Bytes(vec![])),
        (Expected::Status(19), Reply::Status(2)),
        (
            Expected::Read(19, READ_CHUNK),
            Reply::Bytes(b"first".to_vec()),
        ),
        (Expected::Read(19, READ_CHUNK), Reply::Bytes(vec![])),
        (Expected::Status(19), Reply::Status(EXITED)),
        (
            Expected::Read(19, READ_CHUNK),
            Reply::Bytes(b"-tail".to_vec()),
        ),
        (Expected::Read(19, READ_CHUNK), Reply::Bytes(vec![])),
    ]);
    let broker = broker_with_backend(std::future::pending::<()>(), backend.clone());
    assert_eq!(
        collect(&broker.state, 19, 10, Duration::from_secs(1)).unwrap(),
        b"first-tail"
    );
    backend.assert_finished();
}

#[test]
fn collection_rejects_output_over_the_cap_before_appending_it() {
    let _serial = serial();
    let backend = Scripted::new([
        (Expected::Read(23, READ_CHUNK), Reply::Bytes(b"ab".to_vec())),
        (
            Expected::Read(23, READ_CHUNK),
            Reply::Bytes(b"cde".to_vec()),
        ),
    ]);
    let broker = broker_with_backend(std::future::pending::<()>(), backend.clone());
    let error = collect(&broker.state, 23, 4, Duration::from_secs(1)).unwrap_err();
    assert!(error.to_string().contains("4-byte output limit"));
    backend.assert_finished();
}

struct Pending;

impl Backend for Pending {
    fn execute(&self, operation: &Operation) -> io::Result<Reply> {
        match operation {
            Operation::Read(_, _) => Ok(Reply::Bytes(vec![])),
            Operation::Status(_) => Ok(Reply::Status(2)),
            _ => unreachable!(),
        }
    }
}

#[test]
fn collection_deadline_bounds_a_child_that_never_finishes() {
    let _serial = serial();
    let broker = broker_with_backend(std::future::pending::<()>(), Arc::new(Pending));
    let started = Instant::now();
    let error = collect(&broker.state, 29, 100, Duration::from_millis(25)).unwrap_err();
    assert!(error.to_string().contains("wall-clock limit"));
    assert!(started.elapsed() >= Duration::from_millis(25));
    assert!(started.elapsed() < Duration::from_secs(1));
}

struct CountWake(AtomicUsize);

impl Wake for CountWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Release);
    }
}

fn wait_for_wakes(wake: &CountWake, count: usize) {
    let started = Instant::now();
    while wake.0.load(Ordering::Acquire) < count {
        assert!(started.elapsed() < Duration::from_secs(2), "caller stalled");
        std::thread::yield_now();
    }
}

#[test]
fn dropping_broker_disconnects_a_queued_caller_and_cleans_up_owned_children() {
    let _serial = serial();
    let backend = Scripted::new([
        (Expected::Spawn, Reply::Handle(31)),
        (Expected::Terminate(31), Reply::Unit),
    ]);
    let broker = broker_with_backend(std::future::pending::<()>(), backend.clone());
    let state = active_state().unwrap();
    state.call(Operation::Spawn(vec![])).unwrap();
    let wake = Arc::new(CountWake(AtomicUsize::new(0)));
    *state.waker.lock().unwrap() = Some(Waker::from(wake.clone()));
    let caller_state = state.clone();
    let caller = std::thread::spawn(move || caller_state.call(Operation::Status(31)));
    // The second wake is after enqueue, so this caller is waiting for a reply.
    wait_for_wakes(&wake, 2);
    drop(broker);
    assert_eq!(
        caller.join().unwrap().unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    assert_eq!(
        state.call(Operation::Status(31)).unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    assert!(active_state().is_err());
    backend.assert_finished();
}

#[test]
fn dropping_broker_disconnects_a_sender_blocked_by_the_bounded_queue() {
    let _serial = serial();
    let backend = Scripted::new([]);
    let broker = broker_with_backend(std::future::pending::<()>(), backend.clone());
    let state = active_state().unwrap();
    for _ in 0..REQUEST_CAPACITY {
        let (reply, _receive) = mpsc::sync_channel(1);
        state
            .requests
            .try_send(Request {
                operation: Operation::Status(37),
                reply,
            })
            .unwrap();
    }
    let wake = Arc::new(CountWake(AtomicUsize::new(0)));
    *state.waker.lock().unwrap() = Some(Waker::from(wake.clone()));
    let caller = std::thread::spawn(move || state.call(Operation::Status(41)));
    wait_for_wakes(&wake, 1);
    drop(broker);
    assert_eq!(
        caller.join().unwrap().unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    backend.assert_finished();
}

#[test]
fn parent_stream_calls_from_a_worker_preserve_capacities_and_never_own_handle_zero() {
    let _serial = serial();
    let first = vec![b'x'; READ_CHUNK];
    let tail = b"tail".to_vec();
    let mut output = first.clone();
    output.extend_from_slice(&tail);
    let backend = Scripted::new([
        (Expected::Read(0, 2), Reply::Bytes(b"ab".to_vec())),
        (Expected::Read(0, 3), Reply::Bytes(b"cde".to_vec())),
        (Expected::Send(0, first), Reply::Written(READ_CHUNK)),
        (Expected::Send(0, tail), Reply::Written(4)),
        (Expected::Status(0), Reply::Status(2)),
    ]);
    let completed = Arc::new(AtomicBool::new(false));
    let finished = completed.clone();
    let broker = broker_with_backend(
        std::future::poll_fn(move |_| {
            if finished.load(Ordering::Acquire) {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }),
        backend.clone(),
    );
    let state = broker.state.clone();
    let caller = std::thread::spawn(move || {
        assert_eq!(stdio_read(&mut []).unwrap(), 0);
        assert_eq!(stdio_write(&[]).unwrap(), 0);
        let mut small = [0; 2];
        assert_eq!(stdio_read(&mut small).unwrap(), 2);
        assert_eq!(&small, b"ab");
        let mut rest = [0; 3];
        assert_eq!(stdio_read(&mut rest).unwrap(), 3);
        assert_eq!(&rest, b"cde");
        let first = stdio_write(&output).unwrap();
        assert_eq!(first, READ_CHUNK);
        assert_eq!(stdio_write(&output[first..]).unwrap(), 4);
        assert_eq!(parent_status().unwrap(), 2);
        completed.store(true, Ordering::Release);
    });
    drive(broker);
    caller.join().unwrap();
    assert!(state.children.lock().unwrap().is_empty());
    backend.assert_finished();
}

struct FullParentQueue;

impl Backend for FullParentQueue {
    fn execute(&self, operation: &Operation) -> io::Result<Reply> {
        assert!(matches!(operation, Operation::Send(0, bytes) if bytes == b"pending"));
        Err(child_api_error(-3))
    }
}

#[test]
fn full_parent_send_queue_is_would_block_and_never_a_successful_write() {
    let _serial = serial();
    let broker = broker_with_backend(std::future::pending::<()>(), Arc::new(FullParentQueue));
    let error = stdio_write(b"pending").unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert!(broker.state.children.lock().unwrap().is_empty());
}

#[test]
fn invalid_broker_read_or_write_lengths_are_rejected_before_copying() {
    let _serial = serial();
    let backend = Scripted::new([
        (Expected::Read(0, 2), Reply::Bytes(b"too long".to_vec())),
        (Expected::Send(0, b"small".to_vec()), Reply::Written(6)),
    ]);
    let broker = broker_with_backend(std::future::pending::<()>(), backend.clone());
    let mut out = [b'!'; 2];
    assert!(stdio_read(&mut out).is_err());
    assert_eq!(out, [b'!'; 2]);
    assert!(stdio_write(b"small").is_err());
    drop(broker);
    backend.assert_finished();
}

#[test]
fn a_spawn_cannot_register_parent_handle_for_cleanup() {
    let _serial = serial();
    let backend = Scripted::new([(Expected::Spawn, Reply::Handle(0))]);
    let broker = broker_with_backend(std::future::pending::<()>(), backend.clone());
    assert!(broker.state.call(Operation::Spawn(vec![])).is_err());
    assert!(broker.state.children.lock().unwrap().is_empty());
    drop(broker);
    backend.assert_finished();
}
