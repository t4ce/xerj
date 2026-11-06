//! Standalone `xerj-mcp` binary — a thin wrapper over [`xerj_mcp::run`].
//!
//! The same entry point is reachable as `xerj mcp` on the main binary, which
//! is the path a user who ran the published installer has. This binary stays
//! because CI builds and smoke-tests it, and because embedding the MCP proxy
//! without the engine is occasionally useful.

#[tokio::main]
#[cfg(not(target_os = "trueos"))]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    xerj_mcp::run(&args).await
}

#[cfg(target_os = "trueos")]
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    // The broker and child ABI must remain on the original Hull entry stack.
    // Runtime::block_on polls this !Send wrapper on its calling thread.
    runtime.block_on(xerj_autoindex::extract::native_worker::with_broker(
        xerj_mcp::run(&args),
    ))
}
