# XERJ on TRUEOS

The native package contains the XERJ engine, its CLI and `xerj mcp`. The
`neural` and `decide-local` Cargo defaults remain enabled. The package metadata
allows independent instances, preserves argv, and grants shared TRUEOSFS paths
to indexing, model loading and persistent storage.

## Build

Use the existing sibling TRUEOS Blueprint SDK and its patched Rust std runtime:

```sh
cd ../TRUEOS-Blueprints
cargo bp xerj-server
```

The package is written to the SDK's `dist/xerj.bp`.
Import that artifact into the target machine's AppDB using the normal local
Blueprint deployment workflow. A separate MCP package is optional:

```sh
cargo bp ../xerj/engine/crates/xerj-mcp
```

That produces the SDK's `dist/xerj-mcp.bp`. Packaging uses
the normal Cargo defaults; there is no reduced `trueos-blueprint` feature set.
Experimental ONNX and diagnostic profiling are outside those upstream defaults.

Console/passkey support builds the pinned OpenSSL source through the SDK's
native overlay. This requires Perl, make and the SDK's x86_64 C/assembler
toolchain. Keep the upstream x86 assembly enabled: the native configuration
uses OpenSSL's `rdcpu` entropy source, which requires working, CPUID-advertised
RDSEED or RDRAND instructions on the target CPU. Missing or failed hardware
entropy makes RNG-dependent operations return errors; there is no fallback to
time, random-device files or the void `sys_rand` interface.

## Start the native engine

The companion launcher supplies fixed argv and owns the engine's lifetime:

```sh
cd ../TRUEOS-Blueprints
cargo bp probes/xerj_service
```

Install `dist/xerj-service.bp` alongside `xerj.bp` and launch `xerj-service`
from Shell3. It starts the authenticated engine on loopback ports 9200–9202,
stores data in `/apps/xerj/data`, and disables feedback for this trial. Stop
the service Hull to stop its engine child. It retains normal embedding and
decision defaults; compiled neural backends can be selected in another launch.

Launching the bare `xerj` name in Shell3 starts the full engine with normal
authentication and data-directory defaults. TRUEOS defaults to one to four
Tokio workers according to available cores, with four to eight additional
blocking threads admitted on demand. Worker stacks retain the upstream 4 MiB
size. `TOKIO_WORKER_THREADS` can override the native worker default.

Shell3 recognizes exact app names and does not provide general argv. Use
a process-shaped launcher when supplying flags or environment.
For example, an existing SDK parent Blueprint can call
`trueos::child_hull::spawn_program` from its Hull entry thread with this JSON:

```json
{
  "program": "xerj",
  "args": [
    "--bind", "127.0.0.1", "--port", "9200", "--no-auto-port",
    "--data-dir", "/apps/xerj/data", "--disable-feedback"
  ],
  "env": {
    "XERJ_NUM_CPUS": "4",
    "TOKIO_WORKER_THREADS": "4",
    "XERJ_DISABLE_FEEDBACK": "true"
  },
  "cwd": null
}
```

Keep the parent alive while the server runs, drain the child's output and
observe `child_hull::status`. The logical program `xerj` resolves `xerj.bp` in
AppDB. The data directory must remain outside every source tree being indexed.
The pool sizes above are optional runtime tuning; they do not remove features.

This launch uses cleartext loopback with authentication enabled. The
explicit base port selects ES-compatible HTTP on 9200, native REST on 9201 and
gRPC on 9202. Check `GET http://127.0.0.1:9200/` from the client Hull before
using MCP; listening successfully in the server Hull alone does not establish
connectivity between Hulls.

For an auth/TLS deployment use a selected config file (`--config` or
`XERJ_CONFIG`), keep authentication enabled, enable TLS explicitly, and configure
clients with the endpoint and authorization appropriate to that node. A bare Shell3
`xerj` launch uses upstream defaults, including `./data`; it does not read a
launch-script command as argv and does not discover an arbitrary config file.

## Connect Codex over MCP

Install the complete Codex appbundle, including `xerj-service.bp` and its
`xerj.bp` dependency. The native Codex package supplies this ordinary MCP
configuration by default; user configuration can override it:

```toml
[mcp_servers.xerj]
command = "xerj"
args = ["mcp", "--ensure-local"]
startup_timeout_sec = 120
required = false

[mcp_servers.xerj.env]
XERJ_URL = "http://127.0.0.1:9200"
XERJ_DISABLE_FEEDBACK = "true"
```

Codex resolves the command through AppDB and starts a fresh instance with stdin,
stdout, argv and environment. The `mcp` path dispatches before startup banners:
stdout carries only newline-delimited JSON-RPC, while diagnostics use stderr.
`--ensure-local` starts the installed `xerj-service` archive headlessly through
the generic launch API, independently of the MCP proxy. The launcher holds an
exclusive lock before spawning its engine, so simultaneous Codex sessions
share one engine. Closing a proxy leaves the engine available for other
clients; stopping the service Hull stops its child. Reopening an MCP session
can start the service again, preserving `/apps/xerj/data`.

Activation has a 90-second deadline. It requires the local endpoint to reject
an unauthenticated request, then accept the configured key and return a XERJ
identity. It refuses an open node, a bad credential or another endpoint; it
never passes `--insecure`. The returned credential is also used for subsequent
MCP tool requests. Diagnostics go to stderr and contain no credential.

Default `xerj mcp` remains a proxy to an existing node on every platform. For
a remote node or another local port omit `--ensure-local` and set `XERJ_URL`
and `XERJ_AUTH` normally. The opt-in flag rejects those endpoints before launch;
outside TRUEOS it returns an unsupported-platform error.
For the optional standalone package use `command = "xerj-mcp"` and remove
`"mcp"` from `args`.

Native MCP streams use the same bounded entry-thread broker as parser Hulls.
Tokio tasks forward reads and writes to the original Hull entry thread, where
the child transport ABI is callable; an empty live stream waits for more data
instead of being treated as EOF. Both MCP entrypoints install this broker.

The native activation path reads `/apps/xerj/data/admin.key` directly, so first
run does not require a pre-existing working directory. For a node with a
different data directory, pass `XERJ_AUTH = "ApiKey <key>"` in the MCP env block.
The node's key is stored in `<data-dir>/admin.key`; use the startup banner's
resolved data path when configuring another instance.

Native Git corpus cloning and Claude Code Unix hook setup return an explicit
unsupported error before changing corpus content or hook files. The native
subprocess and filesystem APIs do not yet supply the inheritable lock leases
and private Unix directory guarantees these operations require. Direct MCP
agent memory and local-file indexing use their existing HTTP and native file
paths.

## Exercise the package

Build the finite native integration probe from the SDK with
`cargo bp probes/xerj_smoke`, then install and launch `xerj-smoke`. It creates
fresh test data and checks HTTP, gRPC Health, local Rust/Markdown/SQLite autoindex,
MCP source-line and exact SQLite-row retrieval through actual native Codex, single-Codex admission,
packaged-default MCP service activation, memory and persistence after restart. Success ends
with `xerj-smoke v=1 stage=end ok=true`; a compiled probe alone establishes
none of those runtime results. The probe stops its fixture engine/MCP/Codex
children; the independent normal XERJ service remains running.

The separate opt-in `openssl-check` feature checks RNG, P256/SHA256 signing and tamper rejection, a fixed
SHA256 digest and UTC formatting on two native threads and the entry thread.
It is off by default and requires a native OpenSSL provider update after the
Unix ABI removal; native execution has not been verified. Historical host tests
have passed against the native-configured OpenSSL archives, including RNG
failure with CPU entropy masked, plus 155,949 utility regression checks.
Those host results do not establish native passkey or engine behavior.

Use the same process-shaped launcher to run these argv lists against the node:

```json
["autoindex", "/path/to/sources", "--prefix", "trueos", "--url", "http://127.0.0.1:9200", "--dry-run"]
["autoindex", "/path/to/sources", "--prefix", "trueos", "--url", "http://127.0.0.1:9200"]
["autoindex", "map", "--prefix", "trueos", "--url", "http://127.0.0.1:9200"]
["search", "--prefix", "trueos", "--url", "http://127.0.0.1:9200", "your query"]
```

Retain `XERJ_DISABLE_FEEDBACK=true` in these trial launches. Inspect the dry-run
estimate and completion status before selecting a larger corpus. Verify MCP
initialization, `tools/list`, `xerj_map` and a search with source locations;
then store and recall one memory and restart the node to check persistence.

Default embedding is offline lexical feature hashing. The compiled neural
backend is activated with `--embed-mode neural`; it needs MiniLM model assets
downloaded from Hugging Face or supplied locally through the embedding config.
`--decide-mode local --decide-model-dir <path>` activates the compiled local
decision backend. Model loading and neural results require separate native
verification; a successful lexical search does not establish either.

Native PDF and legacy XLS extraction use isolated child Hulls through an
entry-thread broker. The parent admits one parser at a time, bounds its output
and enforces a wall-clock deadline. It requires complete worker JSON or the XLS
end record before accepting results. The native child API exposes lifecycle and
stdout, without numeric exit codes, captured stderr or per-worker POSIX resource
limits; parser diagnostics use the platform log.

Some CLI integrations still depend on host programs and process lifecycle APIs:
Git corpus management, automatic detached Hub/brain nodes and background pulls,
Hub sync subprocesses, browser opening and the optional cloudflared tunnel.
These require further native adapters or ports of the external programs. Hub
queries can connect to the already running engine through `--url` or `XERJ_URL`.
Keeping Cargo features enabled does not establish parity for those integrations.

Native publication synchronizes regular-file writes and waits for completed
TRUEOSFS rename operations. TRUEOS does not support POSIX directory fsync, so
the publication helper validates that the parent still exists as a directory
without requesting another flush. Hardware power-loss barriers and durability
across sudden power loss are not established for this native path; successful
publication or a normal restart must not be presented as proof of them.
