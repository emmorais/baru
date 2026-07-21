//! Binary Merkle tree over content hashes, with inclusion proofs.
//!
//! The hash-consed [`super::hashcons::Store`] gives every datum a [`ZHash`];
//! this builds a succinct commitment over an *ordered set* of such leaves (a
//! participant's private or public data tree, the group's program tree) and
//! lets the holder prove a leaf is included without revealing the others —
//! the basis for ZK statements about private data and for storage challenges.
//!
//! Domain separation: leaves and internal nodes are hashed with distinct
//! prefixes so a leaf can never be reinterpreted as an internal node.  An odd
//! node at a level is promoted unchanged to the next level.

use sha2::{Digest, Sha256};

use super::hashcons::ZHash;

/// Root of the empty tree.
pub const EMPTY_ROOT: ZHash = [0u8; 32];

/// One step of an inclusion proof: a sibling hash and which side it is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofStep {
    pub sibling: ZHash,
    /// True if the sibling sits to the *right* of the running hash.
    pub sibling_on_right: bool,
}

fn hash_leaf(leaf: &ZHash) -> ZHash {
    let mut h = Sha256::new();
    h.update([0x00]);
    h.update(leaf);
    h.finalize().into()
}

fn hash_internal(left: &ZHash, right: &ZHash) -> ZHash {
    let mut h = Sha256::new();
    h.update([0x01]);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// Compute the Merkle root over `leaves` (in order).
pub fn merkle_root(leaves: &[ZHash]) -> ZHash {
    if leaves.is_empty() {
        return EMPTY_ROOT;
    }
    let mut level: Vec<ZHash> = leaves.iter().map(hash_leaf).collect();
    while level.len() > 1 {
        level = fold_level(&level);
    }
    level[0]
}

/// Fold one level into the next, promoting a lone final node unchanged.
fn fold_level(level: &[ZHash]) -> Vec<ZHash> {
    let mut next = Vec::with_capacity(level.len().div_ceil(2));
    let mut i = 0;
    while i < level.len() {
        if i + 1 < level.len() {
            next.push(hash_internal(&level[i], &level[i + 1]));
        } else {
            next.push(level[i]);
        }
        i += 2;
    }
    next
}

/// Produce an inclusion proof for the leaf at `index`.
pub fn merkle_proof(leaves: &[ZHash], index: usize) -> Option<Vec<ProofStep>> {
    if index >= leaves.len() {
        return None;
    }
    let mut proof = Vec::new();
    let mut level: Vec<ZHash> = leaves.iter().map(hash_leaf).collect();
    let mut idx = index;
    while level.len() > 1 {
        if idx ^ 1 < level.len() {
            // Sibling exists; record it and its side.
            let sibling_on_right = idx.is_multiple_of(2);
            let sib = if sibling_on_right { idx + 1 } else { idx - 1 };
            proof.push(ProofStep {
                sibling: level[sib],
                sibling_on_right,
            });
        }
        // else: lone promoted node, no sibling at this level.
        idx /= 2;
        level = fold_level(&level);
    }
    Some(proof)
}

/// Verify that `leaf` at the position described by `proof` is included in `root`.
pub fn verify_proof(leaf: &ZHash, proof: &[ProofStep], root: &ZHash) -> bool {
    let mut acc = hash_leaf(leaf);
    for step in proof {
        acc = if step.sibling_on_right {
            hash_internal(&acc, &step.sibling)
        } else {
            hash_internal(&step.sibling, &acc)
        };
    }
    &acc == root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(n: u8) -> ZHash {
        let mut z = [0u8; 32];
        z[0] = n;
        z
    }

    #[test]
    fn empty_and_singleton() {
        assert_eq!(merkle_root(&[]), EMPTY_ROOT);
        let one = merkle_root(&[leaf(1)]);
        assert_eq!(one, hash_leaf(&leaf(1)));
    }

    #[test]
    fn proofs_verify_for_all_positions_power_of_two() {
        let leaves: Vec<ZHash> = (0..8).map(leaf).collect();
        let root = merkle_root(&leaves);
        for i in 0..leaves.len() {
            let proof = merkle_proof(&leaves, i).unwrap();
            assert!(verify_proof(&leaves[i], &proof, &root), "index {i}");
        }
    }

    #[test]
    fn proofs_verify_for_odd_counts() {
        // Odd counts exercise the promote-lone-node path.
        for n in [1usize, 3, 5, 7, 9] {
            let leaves: Vec<ZHash> = (0..n as u8).map(leaf).collect();
            let root = merkle_root(&leaves);
            for i in 0..n {
                let proof = merkle_proof(&leaves, i).unwrap();
                assert!(verify_proof(&leaves[i], &proof, &root), "n={n} i={i}");
            }
        }
    }

    #[test]
    fn wrong_leaf_or_root_rejected() {
        let leaves: Vec<ZHash> = (0..4).map(leaf).collect();
        let root = merkle_root(&leaves);
        let proof = merkle_proof(&leaves, 2).unwrap();
        assert!(verify_proof(&leaves[2], &proof, &root));
        // Wrong leaf.
        assert!(!verify_proof(&leaf(99), &proof, &root));
        // Wrong root.
        assert!(!verify_proof(&leaves[2], &proof, &leaf(0)));
        // Proof for a different index.
        let other = merkle_proof(&leaves, 1).unwrap();
        assert!(!verify_proof(&leaves[2], &other, &root));
    }

    #[test]
    fn out_of_range_proof_is_none() {
        let leaves: Vec<ZHash> = (0..4).map(leaf).collect();
        assert!(merkle_proof(&leaves, 4).is_none());
    }

    #[test]
    fn order_matters() {
        let a = merkle_root(&[leaf(1), leaf(2), leaf(3)]);
        let b = merkle_root(&[leaf(3), leaf(2), leaf(1)]);
        assert_ne!(a, b);
    }
}
