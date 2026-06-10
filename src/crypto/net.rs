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

/// Placeholder for a TCP-based transport. To be implemented later.
pub struct TcpTransport;
impl TcpTransport {
    pub fn new(_addr: &str) -> Self { TcpTransport }
}

impl Transport for TcpTransport {
    fn id(&self) -> PartyId { unimplemented!() }
    fn send(&self, _to: PartyId, _msg: Message) { unimplemented!() }
    fn broadcast(&self, _msg: Message) { unimplemented!() }
    fn recv(&self) -> Option<(PartyId, Message)> { unimplemented!() }
}
