//! Opt-in native service activation: eager `--ensure-local` or first-tool
//! `--lazy-local`. Without either flag MCP remains a pure proxy.
//!
//! Uses the same independent headless Blueprint launch boundary as the native
//! SQLite client (`sqlx-sqlite/src/remote/connection.rs::ensure_service`).
//! The retained launcher lock admits one engine owner across MCP clients.

use std::future::Future;
use std::time::Duration;

pub(super) const ENDPOINT: &str = "http://127.0.0.1:9200";
const KEY_PATH: &str = "/apps/xerj/data/admin.key";
const READY_TIMEOUT: Duration = Duration::from_secs(90);
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const RETRY_DELAY: Duration = Duration::from_millis(100);

pub(super) fn validate_endpoint(endpoint: &str) -> anyhow::Result<()> {
    let url = reqwest::Url::parse(endpoint)?;
    anyhow::ensure!(
        url.scheme() == "http"
            && matches!(url.host_str(), Some("127.0.0.1" | "localhost"))
            && url.port_or_known_default() == Some(9200)
            && url.username().is_empty()
            && url.password().is_none()
            && url.path() == "/"
            && url.query().is_none()
            && url.fragment().is_none(),
        "--ensure-local/--lazy-local require http://127.0.0.1:9200; omit the flag to use another node"
    );
    Ok(())
}

#[cfg(target_os = "trueos")]
pub(super) async fn ensure(endpoint: &str, explicit_auth: Option<&str>) -> anyhow::Result<String> {
    validate_endpoint(endpoint)?;
    let client = super::node_http_builder(ENDPOINT)
        .redirect(reqwest::redirect::Policy::none())
        .timeout(PROBE_TIMEOUT)
        .build()?;
    let auth = tokio::time::timeout(
        READY_TIMEOUT,
        ensure_ready(
            || probe(&client, ENDPOINT, explicit_auth, read_local_auth),
            || {
                // Launch the installed archive directly: appbundle installation
                // already supplied it, so startup does not fetch a catalog.
                trueos::vshell::launch_with_destination(
                    "xerj-service",
                    "",
                    trueos::vshell::LaunchDestination::Headless,
                )
                .map_err(|code| anyhow::anyhow!(
                    "cannot launch installed xerj-service Blueprint ({code}); install the complete Codex appbundle"
                ))
            },
        ),
    )
    .await
    .map_err(|_| anyhow::anyhow!(
        "xerj-service was not authenticated and ready within 90 seconds; inspect its Hull output and retry"
    ))??;
    eprintln!("xerj-mcp: local xerj-service is authenticated and ready at {ENDPOINT}");
    Ok(auth)
}

fn read_local_auth() -> anyhow::Result<String> {
    let metadata = std::fs::metadata(KEY_PATH).map_err(|_| {
        anyhow::anyhow!(
            "local XERJ requires its admin key at {KEY_PATH}; inspect xerj-service startup"
        )
    })?;
    anyhow::ensure!(metadata.len() <= 128, "invalid local XERJ admin key file");
    #[cfg(target_os = "trueos")]
    let key = xerj_common::native_secret::read_to_string(std::path::Path::new(KEY_PATH))?;
    #[cfg(not(target_os = "trueos"))]
    let key = std::fs::read_to_string(KEY_PATH)?;
    let key = key.trim();
    anyhow::ensure!(
        key.len() == 64 && key.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid local XERJ admin key file"
    );
    Ok(format!("ApiKey {key}"))
}

async fn ensure_ready<P, F, L>(mut probe: P, launch: L) -> anyhow::Result<String>
where
    P: FnMut() -> F,
    F: Future<Output = anyhow::Result<Option<String>>>,
    L: FnOnce() -> anyhow::Result<()>,
{
    if let Some(auth) = probe().await? {
        return Ok(auth);
    }
    launch()?;
    loop {
        tokio::time::sleep(RETRY_DELAY).await;
        if let Some(auth) = probe().await? {
            return Ok(auth);
        }
    }
}

async fn probe(
    client: &reqwest::Client,
    endpoint: &str,
    explicit_auth: Option<&str>,
    read_auth: impl FnOnce() -> anyhow::Result<String>,
) -> anyhow::Result<Option<String>> {
    let url = format!("{endpoint}/_security/_authenticate");
    let response = match client.get(&url).send().await {
        Ok(response) => response,
        Err(error) if error.is_connect() || error.is_timeout() => return Ok(None),
        Err(_) => anyhow::bail!("local XERJ readiness request failed; inspect xerj-service output"),
    };
    if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE {
        return starting_response(response).await;
    }
    anyhow::ensure!(
        response.status() == reqwest::StatusCode::UNAUTHORIZED,
        "local XERJ endpoint does not require authentication; --ensure-local refuses it"
    );
    drop(response);
    let auth = match explicit_auth {
        Some(auth) => auth.to_string(),
        None => read_auth()?,
    };
    let response = match client
        .get(url)
        .header(reqwest::header::AUTHORIZATION, &auth)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) if error.is_connect() || error.is_timeout() => return Ok(None),
        Err(_) => anyhow::bail!("local XERJ authenticated readiness request failed"),
    };
    if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE {
        return starting_response(response).await;
    }
    anyhow::ensure!(
        response.status().is_success(),
        "local XERJ did not accept its configured credential; inspect xerj-service and the admin key"
    );
    let body = read_response(response).await?;
    let identity: serde_json::Value = serde_json::from_slice(&body)?;
    anyhow::ensure!(
        identity["username"] == "xerj" && identity["enabled"] == true,
        "local endpoint did not return an authenticated XERJ identity"
    );
    Ok(Some(auth))
}

async fn read_response(mut response: reqwest::Response) -> anyhow::Result<Vec<u8>> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        anyhow::ensure!(
            chunk.len() <= (16 * 1024_usize).saturating_sub(body.len()),
            "local XERJ authentication response exceeded its limit"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn starting_response(response: reqwest::Response) -> anyhow::Result<Option<String>> {
    let body = read_response(response).await?;
    let response: serde_json::Value = serde_json::from_slice(&body)?;
    anyhow::ensure!(
        response
            .pointer("/error/type")
            .and_then(|value| value.as_str())
            == Some("node_starting_exception"),
        "local endpoint returned an unexpected unavailable response"
    );
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(not(target_os = "trueos"))]
    #[tokio::test]
    async fn native_activation_flag_is_explicitly_unsupported_on_hosts() {
        for flag in ["--ensure-local", "--lazy-local"] {
            let error = super::super::run(&[flag.into()]).await.unwrap_err();
            assert!(error
                .to_string()
                .contains(&format!("{flag} requires TRUEOS")));
        }
    }

    #[tokio::test]
    async fn local_activation_modes_cannot_be_combined() {
        let error = super::super::run(&["--ensure-local".into(), "--lazy-local".into()])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn ensure_local_rejects_remote_or_redirected_endpoints() {
        for endpoint in [
            "https://127.0.0.1:9200",
            "http://example.com:9200",
            "http://localhost:9200@evil.example/",
            "http://user@localhost:9200",
            "http://127.0.0.1:9201",
            "http://127.0.0.1:9200/prefix",
            "http://127.0.0.1:9200/?target=other",
        ] {
            assert!(validate_endpoint(endpoint).is_err(), "{endpoint}");
        }
        assert!(validate_endpoint(ENDPOINT).is_ok());
        assert!(validate_endpoint("http://localhost:9200/").is_ok());
    }

    #[tokio::test]
    async fn already_authenticated_service_does_not_launch_another_owner() {
        let auth = ensure_ready(
            || std::future::ready(Ok(Some("ApiKey fixture".into()))),
            || panic!("ready service must not launch another owner"),
        )
        .await
        .unwrap();
        assert_eq!(auth, "ApiKey fixture");
    }

    #[tokio::test(start_paused = true)]
    async fn cold_service_launches_once_and_waits_for_readiness() {
        let probes = AtomicUsize::new(0);
        let launches = AtomicUsize::new(0);
        let auth = ensure_ready(
            || {
                std::future::ready(Ok(
                    (probes.fetch_add(1, Ordering::SeqCst) == 2).then(|| "ApiKey fixture".into())
                ))
            },
            || {
                launches.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .await
        .unwrap();
        assert_eq!(auth, "ApiKey fixture");
        assert_eq!(launches.load(Ordering::SeqCst), 1);
        assert_eq!(probes.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn unsafe_existing_service_fails_before_launch() {
        let result = ensure_ready(
            || {
                std::future::ready(Err(anyhow::anyhow!(
                    "endpoint accepts unauthenticated requests"
                )))
            },
            || panic!("unsafe existing endpoint must not launch an engine"),
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn failed_launch_returns_without_waiting() {
        let result = ensure_ready(
            || std::future::ready(Ok(None)),
            || anyhow::bail!("missing installed service"),
        )
        .await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("missing installed service"));
    }

    #[tokio::test(start_paused = true)]
    async fn unavailable_service_has_a_bounded_readiness_wait() {
        let launches = AtomicUsize::new(0);
        let result = tokio::time::timeout(
            READY_TIMEOUT,
            ensure_ready(
                || std::future::ready(Ok(None)),
                || {
                    launches.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            ),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(launches.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn readiness_requires_a_credential_and_denies_open_nodes() {
        use axum::{extract::State, http::HeaderMap, routing::get, Router};
        async fn authenticate(
            State(open): State<bool>,
            headers: HeaderMap,
        ) -> (reqwest::StatusCode, axum::Json<serde_json::Value>) {
            let status = if open
                || headers
                    .get("authorization")
                    .is_some_and(|value| value == "ApiKey fixture")
            {
                reqwest::StatusCode::OK
            } else {
                reqwest::StatusCode::UNAUTHORIZED
            };
            (
                status,
                axum::Json(serde_json::json!({"username":"xerj", "enabled":true})),
            )
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(PROBE_TIMEOUT)
            .build()
            .unwrap();
        for open in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let router = Router::new()
                .route("/_security/_authenticate", get(authenticate))
                .with_state(open);
            let server = tokio::spawn(async move {
                axum::serve(listener, router).await.unwrap();
            });
            let result = probe(&client, &endpoint, None, || Ok("ApiKey fixture".into())).await;
            if open {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap().as_deref(), Some("ApiKey fixture"));
            }
            server.abort();
        }
    }

    #[tokio::test]
    async fn starting_node_retries_without_reading_a_key_then_requires_authentication() {
        use axum::{extract::State, http::HeaderMap, routing::get, Router};
        use std::sync::Arc;
        async fn authenticate(
            State(calls): State<Arc<AtomicUsize>>,
            headers: HeaderMap,
        ) -> (reqwest::StatusCode, axum::Json<serde_json::Value>) {
            match calls.fetch_add(1, Ordering::SeqCst) {
                0 => (
                    reqwest::StatusCode::SERVICE_UNAVAILABLE,
                    axum::Json(serde_json::json!({"error":{"type":"node_starting_exception"}})),
                ),
                1 => (
                    reqwest::StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::Value::Null),
                ),
                2 if headers
                    .get("authorization")
                    .is_some_and(|v| v == "ApiKey fixture") =>
                {
                    (
                        reqwest::StatusCode::OK,
                        axum::Json(serde_json::json!({"username":"xerj","enabled":true})),
                    )
                }
                _ => (
                    reqwest::StatusCode::SERVICE_UNAVAILABLE,
                    axum::Json(serde_json::json!({"error":{"type":"another_service"}})),
                ),
            }
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new()
            .route("/_security/_authenticate", get(authenticate))
            .with_state(Arc::new(AtomicUsize::new(0)));
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(PROBE_TIMEOUT)
            .build()
            .unwrap();
        assert!(probe(&client, &endpoint, None, || panic!(
            "starting node needs no key"
        ))
        .await
        .unwrap()
        .is_none());
        assert_eq!(
            probe(&client, &endpoint, None, || Ok("ApiKey fixture".into()))
                .await
                .unwrap()
                .as_deref(),
            Some("ApiKey fixture")
        );
        assert!(probe(&client, &endpoint, None, || panic!(
            "foreign node needs no key"
        ))
        .await
        .is_err());
        server.abort();
    }
}
