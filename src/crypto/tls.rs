//! Post-quantum secure channels for the P2P mesh (feature `pq-tls`).
//!
//! Transport encryption is TLS 1.3 with a **post-quantum hybrid** key exchange
//! (`X25519MLKEM768`, via the `rustls-post-quantum` / aws-lc-rs provider), but
//! authentication does **not** use X.509/CA PKI — there is no certificate
//! authority in a permissionless network.  Instead:
//!
//! * each peer presents an *ephemeral, self-signed* certificate (accepted by a
//!   custom verifier that checks the handshake signature but not any chain), so
//!   the TLS session is confidential and integrity-protected; and
//! * each peer proves its long-term [`Identity`] by signing the RFC 5705 TLS
//!   **exporter** value for this exact session.  Because an active MITM would
//!   see a *different* exporter on each side and cannot forge the peer's
//!   Ed25519 signature, this binds the identity to the channel and defeats
//!   man-in-the-middle attacks — the standard "channel binding" construction.
//!
//! The result is a [`TlsTransport`] with the same [`Transport`] interface and
//! persistent-connection / readiness semantics as [`super::net::TcpTransport`],
//! so the MPC backend runs over it unchanged.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::{Deserialize, Serialize};

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, WebPkiSupportedAlgorithms};
use rustls::pki_types::{
    CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime,
};
use rustls::{
    ClientConfig, ClientConnection, DigitallySignedStruct, ServerConfig, ServerConnection,
    SignatureScheme, StreamOwned,
};

use super::identity::{Identity, IdentityId, IdentitySig, PublicIdentity};
use super::net::{read_frames, write_frame, Frame, Message, PartyId, Transport};

/// Label for the RFC 5705 keying-material exporter used for identity binding.
const EKM_LABEL: &[u8] = b"baru/tls-identity-binding/v1";
const EKM_LEN: usize = 32;

/// Identity proof sent once, right after the TLS handshake, over the encrypted
/// channel: a signature over this session's exporter value.
#[derive(Serialize, Deserialize)]
struct Auth {
    from: PartyId,
    identity: PublicIdentity,
    /// Signature by `identity` over the session exporter (`EKM`).
    sig: IdentitySig,
}

/// Server-certificate verifier that accepts any (self-signed) certificate but
/// still verifies the handshake signature against the presented key.  Peer
/// authentication is done out-of-band via the identity exporter binding, not
/// the certificate chain.
#[derive(Debug)]
struct AcceptAnySelfSigned {
    algs: WebPkiSupportedAlgorithms,
}

impl ServerCertVerifier for AcceptAnySelfSigned {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, cert, dss, &self.algs)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

/// Shared rustls client/server configs, built once per participant.
#[derive(Clone)]
struct TlsConfigs {
    client: Arc<ClientConfig>,
    server: Arc<ServerConfig>,
}

impl TlsConfigs {
    /// Build client + server configs using the PQ-hybrid provider and a fresh
    /// ephemeral self-signed certificate.
    fn build() -> Result<TlsConfigs, String> {
        let provider = Arc::new(rustls_post_quantum::provider());
        let algs = provider.signature_verification_algorithms;

        // Ephemeral self-signed certificate for the TLS handshake.
        let cert = rcgen::generate_simple_self_signed(vec!["baru".to_string()])
            .map_err(|e| format!("rcgen: {e}"))?;
        let cert_der = cert.cert.der().clone();
        let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
            cert.key_pair.serialize_der(),
        ));

        let server = ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("server versions: {e}"))?
            .with_no_client_auth()
            .with_single_cert(vec![cert_der], key_der)
            .map_err(|e| format!("server cert: {e}"))?;
        let mut server = server;
        // Avoid post-handshake writes from the server (we use each connection
        // one-directionally, client -> server).
        server.send_tls13_tickets = 0;

        let client = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("client versions: {e}"))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnySelfSigned { algs }))
            .with_no_client_auth();

        Ok(TlsConfigs {
            client: Arc::new(client),
            server: Arc::new(server),
        })
    }
}

type ClientTls = StreamOwned<ClientConnection, TcpStream>;

/// A `Transport` whose per-peer connections are post-quantum TLS 1.3 channels,
/// each bound to the peer's long-term [`Identity`].
pub struct TlsTransport {
    id: PartyId,
    addrs: Vec<String>,
    identity: Arc<Identity>,
    configs: TlsConfigs,
    inbox: Arc<Mutex<VecDeque<(PartyId, Message)>>>,
    outbound: Vec<Mutex<Option<ClientTls>>>,
    peers_seen: Arc<Mutex<HashSet<PartyId>>>,
    /// Authenticated identity of each peer (recorded after exporter binding).
    authed: Arc<Mutex<HashMap<PartyId, IdentityId>>>,
}

use std::collections::VecDeque;

/// Write a length-prefixed blob (same 4-byte BE framing as `write_frame`).
fn write_blob<S: Write>(s: &mut S, bytes: &[u8]) -> std::io::Result<()> {
    let len = (bytes.len() as u32).to_be_bytes();
    s.write_all(&len)
        .and_then(|_| s.write_all(bytes))
        .and_then(|_| s.flush())
}

/// Read a length-prefixed blob.
fn read_blob<S: Read>(s: &mut S) -> std::io::Result<Vec<u8>> {
    let mut lenbuf = [0u8; 4];
    s.read_exact(&mut lenbuf)?;
    let len = u32::from_be_bytes(lenbuf) as usize;
    let mut buf = vec![0u8; len];
    s.read_exact(&mut buf)?;
    Ok(buf)
}

impl TlsTransport {
    /// Create a TLS transport for participant `id` with a per-party address
    /// list and this participant's long-term identity.  Starts the accepting
    /// listener thread.
    pub fn new(id: PartyId, addrs: Vec<String>, identity: Identity) -> Result<Self, String> {
        let configs = TlsConfigs::build()?;
        let inbox = Arc::new(Mutex::new(VecDeque::new()));
        let peers_seen = Arc::new(Mutex::new(HashSet::new()));
        let authed = Arc::new(Mutex::new(HashMap::new()));

        let listener_addr = addrs[id].clone();
        let server_cfg = configs.server.clone();
        let inbox_l = inbox.clone();
        let peers_l = peers_seen.clone();
        let authed_l = authed.clone();

        thread::spawn(move || {
            let listener = match TcpListener::bind(&listener_addr) {
                Ok(l) => l,
                Err(_) => return,
            };
            for sock in listener.incoming().flatten() {
                let server_cfg = server_cfg.clone();
                let inbox = inbox_l.clone();
                let peers = peers_l.clone();
                let authed = authed_l.clone();
                thread::spawn(move || {
                    accept_peer(sock, server_cfg, inbox, peers, authed);
                });
            }
        });

        let outbound = (0..addrs.len()).map(|_| Mutex::new(None)).collect();
        Ok(TlsTransport {
            id,
            addrs,
            identity: Arc::new(identity),
            configs,
            inbox,
            outbound,
            peers_seen,
            authed,
        })
    }

    /// The authenticated identity of `peer`, if the binding handshake completed.
    pub fn peer_identity(&self, peer: PartyId) -> Option<IdentityId> {
        self.authed.lock().ok().and_then(|m| m.get(&peer).copied())
    }

    /// Establish (once) a persistent authenticated TLS connection to `to`.
    fn ensure_connected(&self, to: PartyId) -> bool {
        let mut slot = match self.outbound[to].lock() {
            Ok(s) => s,
            Err(_) => return false,
        };
        if slot.is_some() {
            return true;
        }
        match self.dial(to) {
            Ok(stream) => {
                *slot = Some(stream);
                true
            }
            Err(_) => false,
        }
    }

    /// Connect + TLS handshake + send our identity binding.
    fn dial(&self, to: PartyId) -> Result<ClientTls, String> {
        let server_name = ServerName::try_from("baru").map_err(|_| "bad server name")?;
        let mut last_err = String::from("no attempt");
        for _ in 0..8 {
            let sock = match TcpStream::connect(&self.addrs[to]) {
                Ok(s) => s,
                Err(e) => {
                    last_err = e.to_string();
                    thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            };
            let mut conn = ClientConnection::new(self.configs.client.clone(), server_name.clone())
                .map_err(|e| format!("client conn: {e}"))?;
            let mut sock = sock;
            if let Err(e) = conn.complete_io(&mut sock) {
                last_err = format!("handshake: {e}");
                thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            let mut ekm = [0u8; EKM_LEN];
            if conn
                .export_keying_material(&mut ekm, EKM_LABEL, None)
                .is_err()
            {
                last_err = "export_keying_material failed".to_string();
                continue;
            }
            let auth = Auth {
                from: self.id,
                identity: self.identity.public(),
                sig: self.identity.sign(&ekm),
            };
            let mut stream = StreamOwned::new(conn, sock);
            let bytes = bincode::serialize(&auth).map_err(|e| e.to_string())?;
            if let Err(e) = write_blob(&mut stream, &bytes) {
                last_err = format!("send auth: {e}");
                continue;
            }
            return Ok(stream);
        }
        Err(last_err)
    }
}

/// Server side of a new inbound connection: TLS handshake, verify the peer's
/// identity binding, then stream frames into the inbox.
fn accept_peer(
    sock: TcpStream,
    server_cfg: Arc<ServerConfig>,
    inbox: Arc<Mutex<VecDeque<(PartyId, Message)>>>,
    peers_seen: Arc<Mutex<HashSet<PartyId>>>,
    authed: Arc<Mutex<HashMap<PartyId, IdentityId>>>,
) {
    let mut conn = match ServerConnection::new(server_cfg) {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut sock = sock;
    if conn.complete_io(&mut sock).is_err() {
        return;
    }
    let mut ekm = [0u8; EKM_LEN];
    if conn.export_keying_material(&mut ekm, EKM_LABEL, None).is_err() {
        return;
    }
    let mut stream = StreamOwned::new(conn, sock);

    // First blob must be the identity binding.
    let auth_bytes = match read_blob(&mut stream) {
        Ok(b) => b,
        Err(_) => return,
    };
    let auth: Auth = match bincode::deserialize(&auth_bytes) {
        Ok(a) => a,
        Err(_) => return,
    };
    // The signature must be over *this* session's exporter — binds identity to
    // the channel and defeats MITM.
    if !auth.identity.verify(&ekm, &auth.sig) {
        return;
    }
    if let Ok(mut m) = authed.lock() {
        m.insert(auth.from, auth.identity.id());
    }
    if let Ok(mut p) = peers_seen.lock() {
        p.insert(auth.from);
    }

    // Remaining blobs are `Frame`s.
    read_frames(stream, inbox, peers_seen);
}

impl Transport for TlsTransport {
    fn id(&self) -> PartyId {
        self.id
    }

    fn send(&self, to: PartyId, msg: Message) {
        if to >= self.addrs.len() || to == self.id {
            return;
        }
        if !self.ensure_connected(to) {
            return;
        }
        if let Ok(mut slot) = self.outbound[to].lock() {
            if let Some(s) = slot.as_mut() {
                if write_frame(s, &Frame::Msg(self.id, msg)).is_err() {
                    *slot = None;
                }
            }
        }
    }

    fn broadcast(&self, msg: Message) {
        for p in 0..self.addrs.len() {
            if p == self.id {
                continue;
            }
            self.send(p, msg.clone());
        }
    }

    fn wait_ready(&self, timeout_ms: u64) -> bool {
        let npeers = self.addrs.len().saturating_sub(1);
        for p in 0..self.addrs.len() {
            if p != self.id {
                self.ensure_connected(p);
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        loop {
            let seen = self.peers_seen.lock().map(|p| p.len()).unwrap_or(0);
            if seen >= npeers {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn recv(&self) -> Option<(PartyId, Message)> {
        self.inbox.lock().ok().and_then(|mut q| q.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_free_ports() -> Vec<String> {
        let ls: Vec<_> = (0..2)
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect();
        let addrs = ls
            .iter()
            .map(|l| l.local_addr().unwrap().to_string())
            .collect();
        drop(ls);
        addrs
    }

    fn recv_timeout(t: &TlsTransport, ms: u64) -> Option<(PartyId, Message)> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(ms);
        loop {
            if let Some(m) = t.recv() {
                return Some(m);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn tls_roundtrip_and_identity_binding() {
        let addrs = two_free_ports();
        let id0 = Identity::generate();
        let id1 = Identity::generate();
        let id0_pub = id0.id();
        let id1_pub = id1.id();

        let t0 = Arc::new(TlsTransport::new(0, addrs.clone(), id0).unwrap());
        let t1 = Arc::new(TlsTransport::new(1, addrs, id1).unwrap());

        // Readiness must run concurrently: each blocks until it sees the peer's
        // inbound identity binding.
        let (a, b) = (t0.clone(), t1.clone());
        let h0 = thread::spawn(move || a.wait_ready(10_000));
        let h1 = thread::spawn(move || b.wait_ready(10_000));
        assert!(h0.join().unwrap(), "t0 not ready");
        assert!(h1.join().unwrap(), "t1 not ready");

        // 0 -> 1 over the PQ-TLS channel.
        t0.send(1, b"hello over pq-tls".to_vec());
        assert_eq!(
            recv_timeout(&t1, 5000),
            Some((0usize, b"hello over pq-tls".to_vec()))
        );
        // t1 authenticated t0's long-term identity via the exporter binding.
        assert_eq!(t1.peer_identity(0), Some(id0_pub));

        // 1 -> 0 the other way.
        t1.send(0, b"ack".to_vec());
        assert_eq!(recv_timeout(&t0, 5000), Some((1usize, b"ack".to_vec())));
        assert_eq!(t0.peer_identity(1), Some(id1_pub));
    }
}
