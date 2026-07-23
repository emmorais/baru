//! Communication instrumentation: `Transport` decorators that count messages
//! and bytes, and optionally inject simulated network latency/bandwidth
//! limits — without touching any protocol code.
//!
//! These wrap **any** `Transport` (in-process, TCP, or TLS), so the exact
//! same MPC protocol code (`TransportBackend`, `aby3_protocol`) runs
//! unmodified while this layer observes what crosses the wire.  That is the
//! basis for the "how many messages, how large, at what simulated RTT"
//! measurements the NN benchmarks report.
//!
//! Two independent concerns, composable:
//!   - [`CountingTransport`] — counts every `send`/`broadcast` and its bytes.
//!     `broadcast` is charged as reaching `nparties - 1` peers, matching what
//!     actually goes out on the wire (the trait's `broadcast` is a single
//!     logical call but a real fan-out).
//!   - [`LatencyTransport`] — delays message *arrival* by a configured
//!     one-way latency plus a bytes/bandwidth term, without blocking the
//!     sender (matches real async network semantics: the sender proceeds
//!     immediately, the message is simply not visible to `recv` until later).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::crypto::net::{Message, PartyId, Transport};

/// A snapshot of message-count and byte-count totals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct CommStats {
    /// Total messages that crossed the wire (a `broadcast` counts once per
    /// recipient, matching real wire traffic).
    pub messages: u64,
    /// Total payload bytes across those messages (post-serialization, i.e.
    /// the actual bytes handed to the transport).
    pub bytes: u64,
}

impl CommStats {
    pub fn saturating_sub(&self, other: &CommStats) -> CommStats {
        CommStats {
            messages: self.messages.saturating_sub(other.messages),
            bytes: self.bytes.saturating_sub(other.bytes),
        }
    }
}

/// Wraps a `Transport`, counting every message sent (and its size) without
/// altering delivery.  `nparties` is needed to correctly charge `broadcast`
/// as the multi-recipient fan-out it really is on the wire.
pub struct CountingTransport<T: Transport> {
    inner: T,
    nparties: usize,
    messages: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
}

impl<T: Transport> CountingTransport<T> {
    pub fn new(inner: T, nparties: usize) -> Self {
        CountingTransport {
            inner,
            nparties,
            messages: Arc::new(AtomicU64::new(0)),
            bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Read the current counters.
    pub fn stats(&self) -> CommStats {
        CommStats {
            messages: self.messages.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
        }
    }

    /// A cheap, `Send + Clone` handle onto the same counters, extractable
    /// *before* the transport itself is moved (e.g. into a backend that's
    /// then moved into a worker thread) so the caller can still read the
    /// final counts after that thread joins.
    pub fn handle(&self) -> CommHandle {
        CommHandle {
            messages: self.messages.clone(),
            bytes: self.bytes.clone(),
        }
    }
}

/// A cheap, cloneable handle onto a [`CountingTransport`]'s counters,
/// independent of the transport's own lifetime.
#[derive(Clone)]
pub struct CommHandle {
    messages: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
}

impl CommHandle {
    pub fn stats(&self) -> CommStats {
        CommStats {
            messages: self.messages.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
        }
    }
}

impl<T: Transport> Transport for CountingTransport<T> {
    fn id(&self) -> PartyId {
        self.inner.id()
    }

    fn send(&self, to: PartyId, msg: Message) {
        self.messages.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(msg.len() as u64, Ordering::Relaxed);
        self.inner.send(to, msg);
    }

    fn broadcast(&self, msg: Message) {
        let fanout = self.nparties.saturating_sub(1) as u64;
        self.messages.fetch_add(fanout, Ordering::Relaxed);
        self.bytes
            .fetch_add(msg.len() as u64 * fanout, Ordering::Relaxed);
        self.inner.broadcast(msg);
    }

    fn recv(&self) -> Option<(PartyId, Message)> {
        self.inner.recv()
    }

    fn wait_ready(&self, timeout_ms: u64) -> bool {
        self.inner.wait_ready(timeout_ms)
    }
}

/// A simple latency + bandwidth model: fixed one-way delay plus a
/// bytes-proportional term (`len / bandwidth_bytes_per_sec`).
#[derive(Debug, Clone, Copy)]
pub struct NetworkProfile {
    pub one_way_latency: Duration,
    /// `None` = unlimited (no bandwidth term).
    pub bandwidth_bytes_per_sec: Option<f64>,
}

impl NetworkProfile {
    pub const INSTANT: NetworkProfile = NetworkProfile {
        one_way_latency: Duration::ZERO,
        bandwidth_bytes_per_sec: None,
    };

    pub fn fixed_latency(one_way: Duration) -> Self {
        NetworkProfile {
            one_way_latency: one_way,
            bandwidth_bytes_per_sec: None,
        }
    }

    pub fn with_bandwidth(mut self, bytes_per_sec: f64) -> Self {
        self.bandwidth_bytes_per_sec = Some(bytes_per_sec);
        self
    }

}

/// Wraps a `Transport`, delaying message *arrival* by [`NetworkProfile`]
/// without blocking the sender.
///
/// Delivery goes through ONE background worker (per wrapped transport)
/// that processes messages strictly in send order, sleeping until each
/// message's scheduled arrival time — so outgoing messages keep their
/// send order, exactly like a real TCP connection. (An earlier version
/// spawned an independent timer thread per message; two back-to-back
/// sends could then race on wakeup and arrive REORDERED, which silently
/// corrupts protocol steps that rely on in-order delivery — e.g.
/// `share_input`'s two same-id `InputShare` messages are distinguished
/// only by arrival order. Real networks don't reorder a connection's
/// stream; neither does this.)
pub struct LatencyTransport<T: Transport + 'static> {
    inner: Arc<T>,
    profile: NetworkProfile,
    /// FIFO delivery queue; `None` when the profile adds no delay (direct
    /// passthrough). Mutex only for `Sync` — one party uses one transport.
    queue: Option<Mutex<std::sync::mpsc::Sender<Delayed>>>,
    /// When this party's outgoing link finishes serializing the previous
    /// message — the state that makes the bandwidth term *cumulative*:
    /// message k cannot start transmitting before message k−1 has left the
    /// NIC, so N back-to-back messages of `len` bytes cost `N·len/bw`, not
    /// `len/bw`. (Without this, bulk phases — exactly what distinguishes a
    /// byte-heavy MPC linear phase from a compact one — were undercharged
    /// by a factor of the message count.)
    last_departure: Mutex<std::time::Instant>,
}

enum Delayed {
    Send(std::time::Instant, PartyId, Message),
    Broadcast(std::time::Instant, Message),
}

impl<T: Transport + 'static> LatencyTransport<T> {
    pub fn new(inner: T, profile: NetworkProfile) -> Self {
        let inner = Arc::new(inner);
        let queue = if profile.one_way_latency.is_zero()
            && profile.bandwidth_bytes_per_sec.is_none()
        {
            None
        } else {
            let (tx, rx) = std::sync::mpsc::channel::<Delayed>();
            let worker_inner = inner.clone();
            thread::spawn(move || {
                // Exits when every sender handle is dropped.
                for item in rx {
                    match item {
                        Delayed::Send(at, to, msg) => {
                            let now = std::time::Instant::now();
                            if at > now {
                                thread::sleep(at - now);
                            }
                            worker_inner.send(to, msg);
                        }
                        Delayed::Broadcast(at, msg) => {
                            let now = std::time::Instant::now();
                            if at > now {
                                thread::sleep(at - now);
                            }
                            worker_inner.broadcast(msg);
                        }
                    }
                }
            });
            Some(Mutex::new(tx))
        };
        LatencyTransport {
            inner,
            profile,
            queue,
            last_departure: Mutex::new(std::time::Instant::now()),
        }
    }

    /// Arrival time under the pipe model: the message starts serializing
    /// when the link is free (`max(now, last_departure)`), takes `len/bw`
    /// to get onto the wire, then flies for one propagation delay.
    fn deliver_at(&self, len: usize) -> std::time::Instant {
        let now = std::time::Instant::now();
        let serialize = self
            .profile
            .bandwidth_bytes_per_sec
            .map(|bw| Duration::from_secs_f64(len as f64 / bw))
            .unwrap_or(Duration::ZERO);
        let mut ld = self
            .last_departure
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let departure = ld.max(now) + serialize;
        *ld = departure;
        departure + self.profile.one_way_latency
    }
}

impl<T: Transport + 'static> Transport for LatencyTransport<T> {
    fn id(&self) -> PartyId {
        self.inner.id()
    }

    fn send(&self, to: PartyId, msg: Message) {
        match &self.queue {
            None => self.inner.send(to, msg),
            Some(q) => {
                let at = self.deliver_at(msg.len());
                q.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .send(Delayed::Send(at, to, msg))
                    .expect("latency delivery worker gone");
            }
        }
    }

    fn broadcast(&self, msg: Message) {
        match &self.queue {
            None => self.inner.broadcast(msg),
            Some(q) => {
                let at = self.deliver_at(msg.len());
                q.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .send(Delayed::Broadcast(at, msg))
                    .expect("latency delivery worker gone");
            }
        }
    }

    fn recv(&self) -> Option<(PartyId, Message)> {
        self.inner.recv()
    }

    fn wait_ready(&self, timeout_ms: u64) -> bool {
        self.inner.wait_ready(timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::net::InprocHandle;

    #[test]
    fn counting_tracks_unicast_and_broadcast_fanout() {
        let handles = InprocHandle::create(3);
        let counted = CountingTransport::new(handles[0].clone(), 3);

        counted.send(1, vec![0u8; 10]);
        assert_eq!(
            counted.stats(),
            CommStats {
                messages: 1,
                bytes: 10
            }
        );

        counted.broadcast(vec![0u8; 5]);
        // broadcast to 2 other parties -> 2 messages, 10 bytes, on top of the unicast.
        assert_eq!(
            counted.stats(),
            CommStats {
                messages: 3,
                bytes: 20
            }
        );
    }

    #[test]
    fn counting_does_not_alter_delivery() {
        let handles = InprocHandle::create(2);
        let a = CountingTransport::new(handles[0].clone(), 2);
        let b = handles[1].clone();
        a.send(1, b"hello".to_vec());
        assert_eq!(b.recv(), Some((0, b"hello".to_vec())));
    }

    #[test]
    fn bandwidth_serializes_bulk_transfers_cumulatively() {
        // 20 × 10 KB at ~1 MB/s must take ≈ 200 ms of serialization, not
        // the ≈ 10 ms a per-message (non-cumulative) model would charge.
        let handles = InprocHandle::create(2);
        let profile = NetworkProfile::fixed_latency(Duration::ZERO).with_bandwidth(1_000_000.0);
        let a = LatencyTransport::new(handles[0].clone(), profile);
        let b = handles[1].clone();
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            a.send(1, vec![0u8; 10_000]);
        }
        let mut got = 0;
        while got < 20 {
            if b.recv().is_some() {
                got += 1;
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let elapsed = t0.elapsed();
        assert!(
            elapsed >= Duration::from_millis(150),
            "bulk transfer finished in {elapsed:?} — bandwidth is not being \
             charged cumulatively"
        );
    }

    #[test]
    fn latency_preserves_send_order() {
        // Regression: per-message timer threads can wake out of order and
        // reorder back-to-back sends, corrupting protocols that rely on
        // in-order delivery (share_input's paired InputShare messages).
        let handles = InprocHandle::create(2);
        let profile = NetworkProfile::fixed_latency(Duration::from_millis(1));
        let a = LatencyTransport::new(handles[0].clone(), profile);
        let b = handles[1].clone();
        for i in 0..50u8 {
            a.send(1, vec![i]);
        }
        std::thread::sleep(Duration::from_millis(200));
        for i in 0..50u8 {
            assert_eq!(b.recv(), Some((0, vec![i])), "message {i} out of order");
        }
    }

    #[test]
    fn latency_delays_arrival_without_blocking_sender() {
        let handles = InprocHandle::create(2);
        let profile = NetworkProfile::fixed_latency(Duration::from_millis(50));
        let a = LatencyTransport::new(handles[0].clone(), profile);
        let b = handles[1].clone();

        let t0 = std::time::Instant::now();
        a.send(1, b"hi".to_vec());
        // The call itself returns near-instantly (sender is not blocked).
        assert!(t0.elapsed() < Duration::from_millis(20));

        // But the message is not there yet...
        assert_eq!(b.recv(), None);
        // ...and shows up only after roughly the configured delay.
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(b.recv(), Some((0, b"hi".to_vec())));
    }

    #[test]
    fn handle_reads_counters_after_transport_is_moved() {
        let handles = InprocHandle::create(2);
        let counted = CountingTransport::new(handles[0].clone(), 2);
        let h = counted.handle(); // extracted before `counted` is moved away
        let moved: Box<dyn Transport> = Box::new(counted);
        moved.send(1, vec![0u8; 7]);
        assert_eq!(
            h.stats(),
            CommStats {
                messages: 1,
                bytes: 7
            }
        );
    }

    #[test]
    fn zero_latency_profile_delivers_synchronously() {
        let handles = InprocHandle::create(2);
        let a = LatencyTransport::new(handles[0].clone(), NetworkProfile::INSTANT);
        let b = handles[1].clone();
        a.send(1, b"sync".to_vec());
        // No injected delay -> delivered before send() even returns.
        assert_eq!(b.recv(), Some((0, b"sync".to_vec())));
    }
}
