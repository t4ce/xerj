//! MCP byte streams carried by the native parent/child Hull transport.
//!
//! The broker performs every ABI operation on the Hull entry thread. These
//! adapters may be polled by Tokio workers: an empty live stream or full queue
//! waits on a Tokio timer rather than blocking an executor with a spin loop.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::time::Sleep;

const STDIO_CHUNK: usize = 3000;
const RETRY_INTERVAL: Duration = Duration::from_millis(10);

trait Backend: Send + Sync {
    fn read(&self, out: &mut [u8]) -> io::Result<usize>;
    fn write(&self, bytes: &[u8]) -> io::Result<usize>;
    fn parent_status(&self) -> io::Result<i32>;
}

#[cfg(target_os = "trueos")]
struct Native;

#[cfg(target_os = "trueos")]
impl Backend for Native {
    fn read(&self, out: &mut [u8]) -> io::Result<usize> {
        xerj_autoindex::extract::native_worker::stdio_read(out)
    }

    fn write(&self, bytes: &[u8]) -> io::Result<usize> {
        xerj_autoindex::extract::native_worker::stdio_write(bytes)
    }

    fn parent_status(&self) -> io::Result<i32> {
        xerj_autoindex::extract::native_worker::parent_status()
    }
}

#[derive(Default)]
struct Retry {
    sleep: Option<Pin<Box<Sleep>>>,
}

impl Retry {
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        if let Some(sleep) = self.sleep.as_mut() {
            if sleep.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            self.sleep = None;
        }
        Poll::Ready(())
    }

    fn schedule(&mut self, cx: &mut Context<'_>) {
        let mut sleep = Box::pin(tokio::time::sleep(RETRY_INTERVAL));
        // Register this task's waker immediately. If the thread was descheduled
        // for the whole interval, request the next poll ourselves instead.
        if sleep.as_mut().poll(cx).is_ready() {
            cx.waker().wake_by_ref();
        }
        self.sleep = Some(sleep);
    }
}

fn invalid_status(status: i32) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("native MCP parent returned invalid status {status}"),
    )
}

fn disconnected() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "native MCP parent has exited")
}

pub(crate) struct Stdin {
    backend: Arc<dyn Backend>,
    retry: Retry,
    parent_exited: bool,
    eof: bool,
}

impl Stdin {
    #[cfg(target_os = "trueos")]
    pub(crate) fn new() -> Self {
        Self::with_backend(Arc::new(Native))
    }

    fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            retry: Retry::default(),
            parent_exited: false,
            eof: false,
        }
    }

    fn read_once(&self, out: &mut ReadBuf<'_>) -> io::Result<usize> {
        let mut bytes = [0; STDIO_CHUNK];
        let capacity = out.remaining().min(bytes.len());
        let read = self.backend.read(&mut bytes[..capacity])?;
        if read > capacity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "native MCP read exceeded its requested capacity",
            ));
        }
        out.put_slice(&bytes[..read]);
        Ok(read)
    }
}

impl AsyncRead for Stdin {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if out.remaining() == 0 || this.eof {
            return Poll::Ready(Ok(()));
        }
        if this.retry.poll_ready(cx).is_pending() {
            return Poll::Pending;
        }
        match this.read_once(out) {
            Ok(read) if read != 0 => return Poll::Ready(Ok(())),
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                this.retry.schedule(cx);
                return Poll::Pending;
            }
            Err(error) => return Poll::Ready(Err(error)),
        }
        if this.parent_exited {
            this.eof = true;
            return Poll::Ready(Ok(()));
        }
        match this.backend.parent_status() {
            Ok(1..=3) => {
                this.retry.schedule(cx);
                Poll::Pending
            }
            Ok(status) if status >= 4 => {
                this.parent_exited = true;
                // Bytes may have arrived between the empty read and observing
                // the parent's exit. Drain once more before declaring EOF.
                match this.read_once(out) {
                    Ok(read) => {
                        this.eof = read == 0;
                        Poll::Ready(Ok(()))
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        this.retry.schedule(cx);
                        Poll::Pending
                    }
                    Err(error) => Poll::Ready(Err(error)),
                }
            }
            Ok(status) => Poll::Ready(Err(invalid_status(status))),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                this.retry.schedule(cx);
                Poll::Pending
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}

pub(crate) struct Stdout {
    backend: Arc<dyn Backend>,
    retry: Retry,
}

impl Stdout {
    #[cfg(target_os = "trueos")]
    pub(crate) fn new() -> Self {
        Self::with_backend(Arc::new(Native))
    }

    fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            retry: Retry::default(),
        }
    }
}

impl AsyncWrite for Stdout {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if bytes.is_empty() {
            return Poll::Ready(Ok(0));
        }
        if this.retry.poll_ready(cx).is_pending() {
            return Poll::Pending;
        }
        match this.backend.parent_status() {
            Ok(1..=3) => {}
            Ok(status) if status >= 4 => return Poll::Ready(Err(disconnected())),
            Ok(status) => return Poll::Ready(Err(invalid_status(status))),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                this.retry.schedule(cx);
                return Poll::Pending;
            }
            Err(error) => return Poll::Ready(Err(error)),
        }
        let bytes = &bytes[..bytes.len().min(STDIO_CHUNK)];
        match this.backend.write(bytes) {
            Ok(0) => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "native MCP write accepted no bytes",
            ))),
            Ok(written) if written <= bytes.len() => Poll::Ready(Ok(written)),
            Ok(_) => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "native MCP write exceeded its offered byte count",
            ))),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                this.retry.schedule(cx);
                Poll::Pending
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // Every successful write has already entered the kernel's parent queue.
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // Closing this adapter never terminates handle zero (the parent Hull).
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::task::{Wake, Waker};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[derive(Default)]
    struct Mock {
        state: Mutex<MockState>,
    }

    #[derive(Default)]
    struct MockState {
        reads: VecDeque<Result<Vec<u8>, io::ErrorKind>>,
        statuses: VecDeque<Result<i32, io::ErrorKind>>,
        writes: VecDeque<Result<usize, io::ErrorKind>>,
        read_capacities: Vec<usize>,
        write_attempts: Vec<Vec<u8>>,
        accepted: Vec<u8>,
        status_calls: usize,
    }

    impl Backend for Mock {
        fn read(&self, out: &mut [u8]) -> io::Result<usize> {
            let mut state = self.state.lock().unwrap();
            state.read_capacities.push(out.len());
            let Some(frame) = state.reads.pop_front() else {
                return Ok(0);
            };
            let mut frame = frame.map_err(io::Error::from)?;
            let read = frame.len().min(out.len());
            out[..read].copy_from_slice(&frame[..read]);
            if read != frame.len() {
                frame.drain(..read);
                state.reads.push_front(Ok(frame));
            }
            Ok(read)
        }

        fn write(&self, bytes: &[u8]) -> io::Result<usize> {
            let mut state = self.state.lock().unwrap();
            state.write_attempts.push(bytes.to_vec());
            let written = state
                .writes
                .pop_front()
                .unwrap_or(Ok(bytes.len()))
                .map_err(io::Error::from)?;
            if written <= bytes.len() {
                state.accepted.extend_from_slice(&bytes[..written]);
            }
            Ok(written)
        }

        fn parent_status(&self) -> io::Result<i32> {
            let mut state = self.state.lock().unwrap();
            state.status_calls += 1;
            state
                .statuses
                .pop_front()
                .unwrap_or(Ok(2))
                .map_err(io::Error::from)
        }
    }

    #[derive(Default)]
    struct WakeCounter(AtomicUsize);

    impl Wake for WakeCounter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn stream_types_are_send_and_unpin() {
        fn require<T: Send + Unpin>() {}
        require::<Stdin>();
        require::<Stdout>();
    }

    #[tokio::test]
    async fn zero_capacity_and_empty_writes_do_not_consume_or_probe() {
        let backend = Arc::new(Mock::default());
        backend
            .state
            .lock()
            .unwrap()
            .reads
            .push_back(Ok(b"keep".to_vec()));
        let mut input = Stdin::with_backend(backend.clone());
        let mut output = Stdout::with_backend(backend.clone());
        assert_eq!(input.read(&mut []).await.unwrap(), 0);
        assert_eq!(output.write(&[]).await.unwrap(), 0);
        output.flush().await.unwrap();
        output.shutdown().await.unwrap();
        let state = backend.state.lock().unwrap();
        assert!(state.read_capacities.is_empty());
        assert!(state.write_attempts.is_empty());
        assert_eq!(state.status_calls, 0);
        assert_eq!(state.reads.front().unwrap().as_ref().unwrap(), b"keep");
    }

    #[tokio::test]
    async fn partial_reads_preserve_every_byte_across_frames() {
        let backend = Arc::new(Mock::default());
        backend.state.lock().unwrap().reads.extend([
            Ok(b"first\n".to_vec()),
            Ok(vec![b'x'; STDIO_CHUNK + 17]),
            Ok(b"last\n".to_vec()),
        ]);
        let mut input = Stdin::with_backend(backend.clone());
        let mut first = [0; 2];
        assert_eq!(input.read(&mut first).await.unwrap(), 2);
        assert_eq!(&first, b"fi");
        {
            let state = backend.state.lock().unwrap();
            assert_eq!(state.read_capacities, [2]);
            assert_eq!(state.reads.front().unwrap().as_ref().unwrap(), b"rst\n");
        }
        backend.state.lock().unwrap().statuses.push_back(Ok(4));
        let mut rest = Vec::new();
        input.read_to_end(&mut rest).await.unwrap();
        let mut expected = b"rst\n".to_vec();
        expected.extend(vec![b'x'; STDIO_CHUNK + 17]);
        expected.extend_from_slice(b"last\n");
        assert_eq!(rest, expected);
        assert!(backend
            .state
            .lock()
            .unwrap()
            .read_capacities
            .iter()
            .all(|capacity| (1..=STDIO_CHUNK).contains(capacity)));
    }

    #[tokio::test]
    async fn empty_live_stream_waits_for_a_real_timer_wake() {
        let backend = Arc::new(Mock::default());
        let mut input = Stdin::with_backend(backend.clone());
        let counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(counter.clone());
        let mut cx = Context::from_waker(&waker);
        let mut bytes = [0; 4];
        let mut read = ReadBuf::new(&mut bytes);
        assert!(Pin::new(&mut input)
            .poll_read(&mut cx, &mut read)
            .is_pending());
        assert!(Pin::new(&mut input)
            .poll_read(&mut cx, &mut read)
            .is_pending());
        assert_eq!(backend.state.lock().unwrap().read_capacities.len(), 1);
        assert_eq!(counter.0.load(Ordering::Relaxed), 0);
        tokio::time::timeout(Duration::from_secs(1), async {
            while counter.0.load(Ordering::Relaxed) == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        backend
            .state
            .lock()
            .unwrap()
            .reads
            .push_back(Ok(b"ok".to_vec()));
        assert!(matches!(
            Pin::new(&mut input).poll_read(&mut cx, &mut read),
            Poll::Ready(Ok(()))
        ));
        assert_eq!(read.filled(), b"ok");
    }

    #[tokio::test]
    async fn would_block_reads_retry_without_becoming_eof() {
        let backend = Arc::new(Mock::default());
        backend
            .state
            .lock()
            .unwrap()
            .reads
            .extend([Err(io::ErrorKind::WouldBlock), Ok(b"message\n".to_vec())]);
        let mut input = Stdin::with_backend(backend.clone());
        let mut bytes = [0; 8];
        tokio::time::timeout(Duration::from_secs(1), input.read_exact(&mut bytes))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(&bytes, b"message\n");
        assert_eq!(backend.state.lock().unwrap().read_capacities.len(), 2);
    }

    #[tokio::test]
    async fn all_live_parent_states_remain_pending_until_a_terminal_state() {
        let backend = Arc::new(Mock::default());
        backend
            .state
            .lock()
            .unwrap()
            .statuses
            .extend([Ok(1), Ok(2), Ok(3), Ok(5)]);
        let mut input = Stdin::with_backend(backend.clone());
        let mut bytes = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), input.read_to_end(&mut bytes))
            .await
            .unwrap()
            .unwrap();
        assert!(bytes.is_empty());
        let state = backend.state.lock().unwrap();
        assert_eq!(state.status_calls, 4);
        assert_eq!(state.read_capacities.len(), 5);
    }

    #[tokio::test]
    async fn parent_exit_drains_racing_bytes_then_latches_eof() {
        let backend = Arc::new(Mock::default());
        {
            let mut state = backend.state.lock().unwrap();
            state
                .reads
                .extend([Ok(Vec::new()), Ok(b"late\n".to_vec()), Ok(Vec::new())]);
            state.statuses.push_back(Ok(4));
        }
        let mut input = Stdin::with_backend(backend.clone());
        let mut bytes = [0; 16];
        assert_eq!(input.read(&mut bytes).await.unwrap(), 5);
        assert_eq!(&bytes[..5], b"late\n");
        assert_eq!(input.read(&mut bytes).await.unwrap(), 0);
        assert_eq!(input.read(&mut bytes).await.unwrap(), 0);
        let state = backend.state.lock().unwrap();
        assert_eq!(state.read_capacities.len(), 3);
        assert_eq!(state.status_calls, 1);
    }

    #[tokio::test]
    async fn read_and_parent_status_failures_propagate() {
        let backend = Arc::new(Mock::default());
        backend
            .state
            .lock()
            .unwrap()
            .reads
            .push_back(Err(io::ErrorKind::PermissionDenied));
        let mut input = Stdin::with_backend(backend.clone());
        let mut bytes = [0; 8];
        assert_eq!(
            input.read(&mut bytes).await.unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        backend
            .state
            .lock()
            .unwrap()
            .statuses
            .push_back(Err(io::ErrorKind::BrokenPipe));
        assert_eq!(
            input.read(&mut bytes).await.unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        backend.state.lock().unwrap().statuses.push_back(Ok(0));
        assert_eq!(
            input.read(&mut bytes).await.unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[tokio::test]
    async fn write_backpressure_and_partial_acceptance_preserve_bytes() {
        let backend = Arc::new(Mock::default());
        backend
            .state
            .lock()
            .unwrap()
            .writes
            .extend([Err(io::ErrorKind::WouldBlock), Ok(2)]);
        let mut output = Stdout::with_backend(backend.clone());
        let bytes: Vec<u8> = (0..STDIO_CHUNK + 505)
            .map(|index| (index % 251) as u8)
            .collect();
        tokio::time::timeout(Duration::from_secs(1), output.write_all(&bytes))
            .await
            .unwrap()
            .unwrap();
        let state = backend.state.lock().unwrap();
        assert_eq!(state.accepted, bytes);
        assert_eq!(state.write_attempts.len(), 4);
        assert_eq!(state.write_attempts[0].len(), STDIO_CHUNK);
        assert_eq!(state.write_attempts[1].len(), STDIO_CHUNK);
        assert_eq!(state.write_attempts[2].len(), STDIO_CHUNK);
        assert_eq!(state.write_attempts[3].len(), 503);
    }

    #[tokio::test]
    async fn writes_reject_disconnected_zero_or_invalid_progress() {
        let backend = Arc::new(Mock::default());
        let mut output = Stdout::with_backend(backend.clone());
        backend.state.lock().unwrap().statuses.push_back(Ok(4));
        assert_eq!(
            output.write(b"x").await.unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        assert!(backend.state.lock().unwrap().write_attempts.is_empty());
        backend.state.lock().unwrap().statuses.push_back(Ok(0));
        assert_eq!(
            output.write(b"x").await.unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        backend.state.lock().unwrap().writes.push_back(Ok(0));
        assert_eq!(
            output.write(b"x").await.unwrap_err().kind(),
            io::ErrorKind::WriteZero
        );
        backend.state.lock().unwrap().writes.push_back(Ok(2));
        assert_eq!(
            output.write(b"x").await.unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        backend
            .state
            .lock()
            .unwrap()
            .writes
            .push_back(Err(io::ErrorKind::PermissionDenied));
        assert_eq!(
            output.write(b"x").await.unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
    }
}
