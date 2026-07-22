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
use std::sync::Arc;
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

    fn delay_for(&self, len: usize) -> Duration {
        let bw_delay = self
            .bandwidth_bytes_per_sec
            .map(|bw| Duration::from_secs_f64(len as f64 / bw))
            .unwrap_or(Duration::ZERO);
        self.one_way_latency + bw_delay
    }
}

/// Wraps a `Transport`, delaying message *arrival* by [`NetworkProfile`]
/// without blocking the sender — a short-lived thread sleeps for the delay
/// then performs the real send, matching real async network semantics
/// (the sender is free to keep computing while the message is "in flight").
pub struct LatencyTransport<T: Transport + 'static> {
    inner: Arc<T>,
    profile: NetworkProfile,
}

impl<T: Transport + 'static> LatencyTransport<T> {
    pub fn new(inner: T, profile: NetworkProfile) -> Self {
        LatencyTransport {
            inner: Arc::new(inner),
            profile,
        }
    }
}

impl<T: Transport + 'static> Transport for LatencyTransport<T> {
    fn id(&self) -> PartyId {
        self.inner.id()
    }

    fn send(&self, to: PartyId, msg: Message) {
        if self.profile.one_way_latency.is_zero() && self.profile.bandwidth_bytes_per_sec.is_none()
        {
            self.inner.send(to, msg);
            return;
        }
        let delay = self.profile.delay_for(msg.len());
        let inner = self.inner.clone();
        thread::spawn(move || {
            thread::sleep(delay);
            inner.send(to, msg);
        });
    }

    fn broadcast(&self, msg: Message) {
        if self.profile.one_way_latency.is_zero() && self.profile.bandwidth_bytes_per_sec.is_none()
        {
            self.inner.broadcast(msg);
            return;
        }
        let delay = self.profile.delay_for(msg.len());
        let inner = self.inner.clone();
        thread::spawn(move || {
            thread::sleep(delay);
            inner.broadcast(msg);
        });
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
