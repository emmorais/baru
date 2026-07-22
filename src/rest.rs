//! REST control plane — the per-node HTTP API.
//!
//! The operator/UX surface, separate from the machine-to-machine p2p protocol.
//! It exposes the node's agreed state and accepts submissions:
//!
//! | Method & path      | Purpose                                            |
//! |--------------------|----------------------------------------------------|
//! | `GET  /health`     | liveness + party id + current epoch                |
//! | `GET  /state`      | committed roots (state / reputation / forum)       |
//! | `GET  /reputation` | the reputation ledger (per-identity + treasury)    |
//! | `GET  /forum`      | forum posts + root                                 |
//! | `POST /forum`      | publish a signed post (`{"body": "..."}`)          |
//! | `GET  /programs`   | submitted program queue                            |
//! | `POST /programs`   | submit a program for the network (`{"source":...}`)|
//!
//! Built on tokio + axum (pure Rust).  The shared [`NodeState`] is what the
//! node updates as it runs (epoch, roots, reputation, forum); the API reads and
//! appends to it under a mutex.  Handlers are self-contained and unit-tested
//! in-process; [`serve`] / [`serve_blocking`] bind a real socket for the node.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::crypto::identity::Identity;
use crate::messaging::{Forum, ForumPost};
use crate::reputation::ReputationLedger;

/// 32-byte commitment.
pub type Hash = [u8; 32];

/// The node's mutable, queryable state behind the REST API.
pub struct NodeState {
    pub epoch: u64,
    pub state_root: Hash,
    pub reputation: ReputationLedger,
    pub forum: Forum,
    pub programs: Vec<String>,
    post_seq: u64,
}

impl NodeState {
    pub fn new() -> Self {
        NodeState {
            epoch: 0,
            state_root: [0u8; 32],
            reputation: ReputationLedger::new(),
            forum: Forum::new(),
            programs: Vec::new(),
            post_seq: 0,
        }
    }
}

impl Default for NodeState {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared application state handed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub party_id: usize,
    /// Secret bytes of the node's identity, used to sign forum posts made via
    /// the API.
    identity_secret: [u8; 32],
    inner: Arc<Mutex<NodeState>>,
}

impl AppState {
    pub fn new(party_id: usize, identity: &Identity, state: Arc<Mutex<NodeState>>) -> Self {
        AppState {
            party_id,
            identity_secret: identity.to_secret_bytes(),
            inner: state,
        }
    }

    /// The shared state handle (so the node can keep updating it).
    pub fn state(&self) -> Arc<Mutex<NodeState>> {
        self.inner.clone()
    }
}

/// The control-plane router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/state", get(get_state))
        .route("/reputation", get(get_reputation))
        .route("/forum", get(get_forum).post(post_forum))
        .route("/programs", get(get_programs).post(post_program))
        .with_state(state)
}

/// Bind `addr` and serve the API (async).
pub async fn serve(addr: SocketAddr, state: AppState) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(state)).await
}

/// Run the API on a fresh runtime (blocking) — spawn this in a thread from a
/// synchronous node.
pub fn serve_blocking(addr: SocketAddr, state: AppState) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let _ = rt.block_on(serve(addr, state));
}

// ── handlers ───────────────────────────────────────────────────────────────────

async fn health(State(s): State<AppState>) -> Json<Value> {
    let st = s.inner.lock().expect("state lock");
    Json(json!({ "status": "ok", "party_id": s.party_id, "epoch": st.epoch }))
}

async fn get_state(State(s): State<AppState>) -> Json<Value> {
    let st = s.inner.lock().expect("state lock");
    Json(json!({
        "party_id": s.party_id,
        "epoch": st.epoch,
        "state_root": hex::encode(st.state_root),
        "reputation_root": hex::encode(st.reputation.root()),
        "forum_root": hex::encode(st.forum.root()),
    }))
}

async fn get_reputation(State(s): State<AppState>) -> Json<Value> {
    let st = s.inner.lock().expect("state lock");
    let entries: Vec<Value> = st
        .reputation
        .entries()
        .iter()
        .map(|(id, rep)| json!({ "id": hex::encode(id), "reputation": rep.to_string() }))
        .collect();
    Json(json!({
        "treasury": st.reputation.treasury().to_string(),
        "root": hex::encode(st.reputation.root()),
        "entries": entries,
    }))
}

async fn get_forum(State(s): State<AppState>) -> Json<Value> {
    let st = s.inner.lock().expect("state lock");
    let posts: Vec<Value> = st
        .forum
        .posts()
        .iter()
        .map(|p| json!({ "author": p.author.id().to_hex(), "seq": p.seq, "body": p.body }))
        .collect();
    Json(json!({ "root": hex::encode(st.forum.root()), "posts": posts }))
}

#[derive(Deserialize)]
struct PostForum {
    body: String,
}

async fn post_forum(State(s): State<AppState>, Json(p): Json<PostForum>) -> Json<Value> {
    let id = Identity::from_secret_bytes(s.identity_secret);
    let mut st = s.inner.lock().expect("state lock");
    st.post_seq += 1;
    let seq = st.post_seq;
    let post = ForumPost::create(&id, seq, &p.body);
    let ok = st.forum.append(post).is_ok();
    Json(json!({ "accepted": ok, "seq": seq, "root": hex::encode(st.forum.root()) }))
}

async fn get_programs(State(s): State<AppState>) -> Json<Value> {
    let st = s.inner.lock().expect("state lock");
    Json(json!({ "count": st.programs.len(), "programs": st.programs }))
}

#[derive(Deserialize)]
struct SubmitProgram {
    source: String,
}

async fn post_program(State(s): State<AppState>, Json(p): Json<SubmitProgram>) -> Json<Value> {
    let mut st = s.inner.lock().expect("state lock");
    st.programs.push(p.source);
    Json(json!({ "accepted": true, "count": st.programs.len() }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for `oneshot`

    fn app() -> (Router, Arc<Mutex<NodeState>>) {
        let state = Arc::new(Mutex::new(NodeState::new()));
        let id = Identity::generate();
        (router(AppState::new(0, &id, state.clone())), state)
    }

    async fn get_json(app: &Router, uri: &str) -> (StatusCode, Value) {
        let resp = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    async fn post_json(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn health_and_state() {
        let (app, _s) = app();
        let (status, body) = get_json(&app, "/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["party_id"], 0);

        let (status, body) = get_json(&app, "/state").await;
        assert_eq!(status, StatusCode::OK);
        // Roots are present as hex strings.
        assert_eq!(body["state_root"].as_str().unwrap().len(), 64);
        assert!(body["forum_root"].is_string());
    }

    #[tokio::test]
    async fn submit_and_list_programs() {
        let (app, _s) = app();
        let (status, body) = post_json(&app, "/programs", json!({"source": "(+ 1 2)"})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["accepted"], true);
        assert_eq!(body["count"], 1);

        let (_s, body) = get_json(&app, "/programs").await;
        assert_eq!(body["count"], 1);
        assert_eq!(body["programs"][0], "(+ 1 2)");
    }

    #[tokio::test]
    async fn post_to_forum_appends_and_changes_root() {
        let (app, _s) = app();
        let (_s, before) = get_json(&app, "/forum").await;
        let root_before = before["root"].as_str().unwrap().to_string();

        let (status, body) = post_json(&app, "/forum", json!({"body": "hello network"})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["accepted"], true);
        assert_eq!(body["seq"], 1);

        let (_s, after) = get_json(&app, "/forum").await;
        assert_ne!(after["root"].as_str().unwrap(), root_before);
        assert_eq!(after["posts"][0]["body"], "hello network");
        assert_eq!(after["posts"][0]["seq"], 1);
    }

    #[tokio::test]
    async fn reputation_reflects_ledger() {
        use crate::fhe::lifecycle::Effect;
        let (app, state) = app();
        // The node folds a settlement into its ledger…
        {
            let mut st = state.lock().unwrap();
            st.reputation.apply(&Effect::Reputation {
                id: [7u8; 32],
                delta: 250,
            });
            st.epoch = 3;
        }
        // …and the API reflects it.
        let (status, body) = get_json(&app, "/reputation").await;
        assert_eq!(status, StatusCode::OK);
        let entry = &body["entries"][0];
        assert_eq!(entry["id"], hex::encode([7u8; 32]));
        assert_eq!(entry["reputation"], "250");

        let (_s, st) = get_json(&app, "/state").await;
        assert_eq!(st["epoch"], 3);
    }

    #[tokio::test]
    async fn unknown_route_is_404() {
        let (app, _s) = app();
        let (status, _b) = {
            let resp = app
                .clone()
                .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
                .await
                .unwrap();
            (resp.status(), ())
        };
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
