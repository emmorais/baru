use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde::{Serialize, Deserialize};

pub type Message = Vec<u8>;
pub type PartyId = usize;

/// Minimal transport trait used by MPC protocols in this crate.
/// Implementations may be in-process (testing) or real networked.
pub trait Transport: Send + Sync {
    /// id of this participant (0..n-1)
    fn id(&self) -> PartyId;
    /// send a message to a specific party (non-blocking)
    fn send(&self, to: PartyId, msg: Message);
    /// broadcast a message to all other parties (non-blocking)
    fn broadcast(&self, msg: Message);
    /// receive the next message addressed to this participant, if any.
    /// Non-blocking: returns None if no message is available.
    fn recv(&self) -> Option<(PartyId, Message)>;
}

/// Simple MPC message types for simulator usage.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MpcMsg {
    D(u128),
    E(u128),
    PublicD(u128),
    PublicE(u128),
}

/// Serialize a message using bincode.
pub fn serialize_msg<T: Serialize>(msg: &T) -> Message {
    bincode::serialize(msg).expect("serialize_msg failed")
}

/// Deserialize a message using bincode.
pub fn deserialize_msg<T: for<'de> serde::Deserialize<'de>>(bytes: &Message) -> T {
    bincode::deserialize(bytes).expect("deserialize_msg failed")
}

/// In-process network: simple shared queues per party. Not secure; for tests.
struct InprocNetwork {
    queues: Vec<Mutex<VecDeque<(PartyId, Message)>>>,
}

impl InprocNetwork {
    fn new(nparties: usize) -> Self {
        let mut queues = Vec::with_capacity(nparties);
        for _ in 0..nparties {
            queues.push(Mutex::new(VecDeque::new()));
        }
        InprocNetwork { queues }
    }
}

/// A per-party handle to an `InprocNetwork` that implements `Transport`.
#[derive(Clone)]
pub struct InprocHandle {
    id: PartyId,
    net: Arc<InprocNetwork>,
    nparties: usize,
}

impl InprocHandle {
    /// Create `nparties` handles connected to the same in-process network.
    pub fn create(nparties: usize) -> Vec<InprocHandle> {
        let net = Arc::new(InprocNetwork::new(nparties));
        (0..nparties)
            .map(|id| InprocHandle { id, net: net.clone(), nparties })
            .collect()
    }
}

impl Transport for InprocHandle {
    fn id(&self) -> PartyId {
        self.id
    }

    fn send(&self, to: PartyId, msg: Message) {
        if to >= self.nparties { return; }
        if let Ok(mut q) = self.net.queues[to].lock() {
            q.push_back((self.id, msg));
        }
    }

    fn broadcast(&self, msg: Message) {
        for p in 0..self.nparties {
            if p == self.id { continue; }
            if let Ok(mut q) = self.net.queues[p].lock() {
                q.push_back((self.id, msg.clone()));
            }
        }
    }

    fn recv(&self) -> Option<(PartyId, Message)> {
        if let Ok(mut q) = self.net.queues[self.id].lock() {
            q.pop_front()
        } else {
            None
        }
    }
}

/// TCP wire format: include sender id and raw payload bytes.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Wire {
    from: PartyId,
    payload: Message,
}

use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use std::thread;

/// TCP-based Transport implementation. Creates a listener (background thread)
/// to accept incoming framed messages and pushes them into a local inbox queue.
/// Sending performs a connect-and-send to the peer address and writes a
/// 4-byte big-endian length followed by the bincode-serialized Wire payload.
pub struct TcpTransport {
    id: PartyId,
    addrs: Vec<String>,
    inbox: Arc<Mutex<VecDeque<(PartyId, Message)>>>,
}

impl TcpTransport {
    /// Create a new TcpTransport for participant `id` with a list of addresses
    /// (one address per party, length == nparties). This starts a listener
    /// thread bound to addrs[id].
    pub fn new(id: PartyId, addrs: Vec<String>) -> Self {
        let inbox = Arc::new(Mutex::new(VecDeque::new()));
        let listener_addr = addrs[id].clone();
        let inbox_clone = inbox.clone();

        // spawn listener thread
        thread::spawn(move || {
            if let Ok(listener) = TcpListener::bind(&listener_addr) {
                for stream in listener.incoming() {
                    match stream {
                        Ok(mut s) => {
                            // read length prefix
                            let mut lenbuf = [0u8; 4];
                            if s.read_exact(&mut lenbuf).is_err() { continue; }
                            let len = u32::from_be_bytes(lenbuf) as usize;
                            let mut buf = vec![0u8; len];
                            if s.read_exact(&mut buf).is_err() { continue; }
                            if let Ok(wire) = bincode::deserialize::<Wire>(&buf) {
                                if let Ok(mut q) = inbox_clone.lock() {
                                    q.push_back((wire.from, wire.payload));
                                }
                            }
                        }
                        Err(_) => { /* ignore accept errors */ }
                    }
                }
            }
        });

        TcpTransport { id, addrs, inbox }
    }

    /// Helper: connect and send a Wire to an address (best-effort).
    fn send_to_addr(addr: &str, wire: &Wire) -> std::io::Result<()> {
        // serialize once outside the retry loop
        let bytes = bincode::serialize(wire).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let len = (bytes.len() as u32).to_be_bytes();

        // retry connecting/sending a few times to tolerate races where the
        // peer hasn't yet bound its listener. This improves reliability in
        // CI and on slow machines.
        let mut last_err: Option<std::io::Error> = None;
        let attempts = 8;
        for _ in 0..attempts {
            match TcpStream::connect(addr) {
                Ok(mut s) => {
                    if let Err(e) = s.write_all(&len).and_then(|_| s.write_all(&bytes)).and_then(|_| s.flush()) {
                        // writing failed even though we connected; return the error
                        return Err(e);
                    }
                    return Ok(());
                }
                Err(e) => {
                    last_err = Some(e);
                    // short backoff before retrying
                    thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
        // all attempts failed — return last error if any
        if let Some(e) = last_err {
            Err(e)
        } else {
            Ok(())
        }
    }
}

impl Transport for TcpTransport {
    fn id(&self) -> PartyId { self.id }

    fn send(&self, to: PartyId, msg: Message) {
        if to >= self.addrs.len() { return; }
        let wire = Wire { from: self.id, payload: msg };
        // best-effort send; ignore result
        let _ = Self::send_to_addr(&self.addrs[to], &wire);
    }

    fn broadcast(&self, msg: Message) {
        for p in 0..self.addrs.len() {
            if p == self.id { continue; }
            self.send(p, msg.clone());
        }
    }

    fn recv(&self) -> Option<(PartyId, Message)> {
        if let Ok(mut q) = self.inbox.lock() {
            q.pop_front()
        } else { None }
    }
}
