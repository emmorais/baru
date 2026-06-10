use crate::crypto::types::{Share3, Fr};
use crate::crypto::arith::{gen_beaver_triple, share_secret, scalar_mul, add_shares};

/// Simple broadcast message used by the scheduler.
#[derive(Debug, Clone)]
struct Message {
    from: usize,
    tag: &'static str,
    value: Fr,
}

/// Scheduler that simulates broadcast rounds between three parties.
/// It can simulate a single corrupted party that tampers with its outgoing
/// broadcasts (to exercise detection / failure cases).
pub struct Scheduler {
    messages: Vec<Message>,
    corrupted: Option<usize>,
    rounds: usize,
}

impl Scheduler {
    pub fn new(corrupted: Option<usize>) -> Self {
        Scheduler { messages: Vec::new(), corrupted, rounds: 0 }
    }

    /// Simulate a broadcast from `from` with tag and value. If `from` matches
    /// the corrupted party, the value will be tampered by adding 1 (simple
    /// adversarial behavior used for testing).
    pub fn broadcast(&mut self, from: usize, tag: &'static str, value: Fr) {
        let mut v = value;
        if let Some(c) = self.corrupted {
            if c == from {
                // tamper with the value to simulate malicious behavior
                v = v.wrapping_add(1);
            }
        }
        for _to in 0..3 {
            // In a real network each recipient would receive a copy; we store
            // the (from,tag,value) triple once per recipient for clarity.
            self.messages.push(Message { from, tag, value: v });
        }
        self.rounds += 1;
    }

    /// Collect all messages with given tag and return their values grouped by sender.
    /// We consider the first message from each sender as that sender's broadcast.
    pub fn collect_sum(&self, tag: &'static str) -> Fr {
        let mut sums: [Option<Fr>; 3] = [None, None, None];
        for m in &self.messages {
            if m.tag != tag { continue; }
            let idx = m.from as usize;
            if sums[idx].is_none() {
                sums[idx] = Some(m.value);
            }
        }
        let mut total: Fr = 0;
        for i in 0..3 {
            if let Some(v) = sums[i] {
                total = total.wrapping_add(v);
            } else {
                panic!("missing broadcast from party {} for tag {}", i, tag);
            }
        }
        total
    }

    pub fn rounds(&self) -> usize { self.rounds }
}

/// Run the Beaver triple online phase using the scheduler to simulate broadcasts.
/// Returns (output_share, rounds_used).
pub fn mul_with_scheduler(a: &Share3, b: &Share3, corrupted: Option<usize>) -> (Share3, usize) {
    let mut sched = Scheduler::new(corrupted);

    // Preprocessing: generate triple
    let (x_sh, y_sh, z_sh) = gen_beaver_triple();

    // split into per-party "local" components (party i holds s_i, s_{i+1})
    let a_local = [(a.0, a.1), (a.1, a.2), (a.2, a.0)];
    let b_local = [(b.0, b.1), (b.1, b.2), (b.2, b.0)];
    let x_local = [(x_sh.0, x_sh.1), (x_sh.1, x_sh.2), (x_sh.2, x_sh.0)];
    let y_local = [(y_sh.0, y_sh.1), (y_sh.1, y_sh.2), (y_sh.2, y_sh.0)];

    // Each party broadcasts its local d_i and e_i in separate rounds
    for i in 0..3 {
        let d_i = a_local[i].0.wrapping_sub(x_local[i].0);
        sched.broadcast(i, "d", d_i);
    }
    for i in 0..3 {
        let e_i = b_local[i].0.wrapping_sub(y_local[i].0);
        sched.broadcast(i, "e", e_i);
    }

    // collect public d and e
    let d = sched.collect_sum("d");
    let e = sched.collect_sum("e");

    // reshare cross term and compute final
    let cross = share_secret(d.wrapping_mul(e));
    let term_dy = scalar_mul(&y_sh, d);
    let term_ex = scalar_mul(&x_sh, e);
    let s1 = add_shares(&z_sh, &term_dy);
    let s2 = add_shares(&term_ex, &cross);
    let out = add_shares(&s1, &s2);
    (out, sched.rounds())
}
