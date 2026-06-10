use rand::Rng;
use std::ops::{Add, Sub, Mul};

/// Field modulus used for arithmetic sharing. For simplicity use a 128-bit
/// unsigned integer for local simulation.
pub type Fr = u128;

/// A Share in the 3-party ABY3 additive sharing: (x0, x1, x2) such that
/// x = x0 + x1 + x2 mod M. The struct holds three local share values.
#[derive(Debug, Clone)]
pub struct Share3(pub Fr, pub Fr, pub Fr);

impl Share3 {
    pub fn zero() -> Self { Share3(0,0,0) }
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let a: u128 = rng.gen();
        let b: u128 = rng.gen();
        let c: u128 = rng.gen();
        Share3(a,b,c)
    }

    pub fn to_plain(&self) -> Fr {
        // naive reconstruction
        self.0.wrapping_add(self.1).wrapping_add(self.2)
    }
}

impl Add for Share3 {
    type Output = Share3;
    fn add(self, rhs: Share3) -> Share3 {
        Share3(self.0.wrapping_add(rhs.0), self.1.wrapping_add(rhs.1), self.2.wrapping_add(rhs.2))
    }
}

impl Sub for Share3 {
    type Output = Share3;
    fn sub(self, rhs: Share3) -> Share3 {
        Share3(self.0.wrapping_sub(rhs.0), self.1.wrapping_sub(rhs.1), self.2.wrapping_sub(rhs.2))
    }
}

// placeholder mul: in ABY3 multiplication requires interaction; here we
// provide a local naive multiplication that reconstructs and reshares (insecure)
impl Mul for Share3 {
    type Output = Share3;
    fn mul(self, rhs: Share3) -> Share3 {
        let a = self.to_plain();
        let b = rhs.to_plain();
        let prod = a.wrapping_mul(b);
        // reshare prod into three random shares
        let mut rng = rand::thread_rng();
        let s0: u128 = rng.gen();
        let s1: u128 = rng.gen();
        let s2 = prod.wrapping_sub(s0).wrapping_sub(s1);
        Share3(s0,s1,s2)
    }
}
