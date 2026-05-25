//! Tree-cost computation — walk a flat `vpx_tree_index` tree and
//! produce one bit-cost per leaf.
//!
//! Mirrors libvpx `vp9_cost_tokens` and `vp9_cost_tokens_skip` from
//! `vp9/encoder/vp9_cost.c:39-65`.
//!
//! ## Tree layout (libvpx convention)
//!
//! The tree is a flat `[i8]` of length `2 * (num_leaves - 1)`. Each
//! pair `(tree[i], tree[i+1])` is one internal node's two children:
//! a non-positive entry is `-leaf_index` (the negation of a leaf
//! index), and a positive entry is the index in the same array of
//! another internal-node pair. Root is at `i = 0`. The probability
//! `probs[i >> 1]` is consulted at internal node `i`.

use super::prob_cost::{vp9_cost_bit, vp9_cost_zero};

/// Walk the sub-tree rooted at `tree[i]` accumulating bit costs into
/// `c`. At each leaf the running cost `c` is stored in
/// `costs[-leaf_index]`.
fn cost_walk(costs: &mut [i32], tree: &[i8], probs: &[u8], i: usize, c: i32) {
    let prob = probs[i >> 1];
    for b in 0..=1u8 {
        let bit_cost = vp9_cost_bit(prob, b != 0) as i32;
        let cc = c + bit_cost;
        let ii = tree[i + usize::from(b)];
        if ii <= 0 {
            let leaf = -(ii as i32) as usize;
            costs[leaf] = cc;
        } else {
            cost_walk(costs, tree, probs, ii as usize, cc);
        }
    }
}

/// Build the per-leaf cost table for a `vpx_tree_index` tree. Mirrors
/// libvpx `vp9_cost_tokens` (`vp9_cost.c:56-58`).
///
/// `costs.len()` must be at least the maximum leaf index plus one.
/// `probs.len()` must be at least the number of internal nodes
/// (`tree.len() / 2`).
pub fn vp9_cost_tokens(costs: &mut [i32], tree: &[i8], probs: &[u8]) {
    cost_walk(costs, tree, probs, 0, 0);
}

/// Variant of [`vp9_cost_tokens`] used when the root's first branch is
/// the EOB leaf — the caller already accounted for the EOB token cost
/// elsewhere, so this fills the EOB slot with just `cost_zero(probs[0])`
/// and recurses into the right sub-tree with a zero accumulator.
/// Mirrors libvpx `vp9_cost_tokens_skip` (`vp9_cost.c:60-65`).
pub fn vp9_cost_tokens_skip(costs: &mut [i32], tree: &[i8], probs: &[u8]) {
    debug_assert!(tree[0] <= 0 && tree[1] > 0);
    let leaf0 = -(tree[0] as i32) as usize;
    costs[leaf0] = vp9_cost_zero(probs[0]) as i32;
    cost_walk(costs, tree, probs, 2, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial 2-leaf tree: one internal node, two leaves indexed 0 and
    /// 1. Verify the two costs match `cost_zero` / `cost_one`.
    #[test]
    fn two_leaf_tree() {
        // Tree: leaf 0 on left, leaf 1 on right.
        let tree: [i8; 2] = [-0, -1];
        let probs: [u8; 1] = [128];
        let mut costs = [0i32; 2];
        vp9_cost_tokens(&mut costs, &tree, &probs);
        // At prob=128 each branch costs 512 (1 bit).
        assert_eq!(costs[0], 512);
        assert_eq!(costs[1], 512);
    }

    /// 3-leaf tree: one internal node on the right.
    ///
    /// Tree shape (indices into `tree[]`):
    /// ```text
    ///         root (i=0, probs[0])
    ///         /                 \
    ///    leaf 0           internal (i=2, probs[1])
    ///                       /              \
    ///                  leaf 1            leaf 2
    /// ```
    #[test]
    fn three_leaf_tree() {
        let tree: [i8; 4] = [-0, 2, -1, -2];
        let probs: [u8; 2] = [128, 128];
        let mut costs = [0i32; 3];
        vp9_cost_tokens(&mut costs, &tree, &probs);
        // leaf 0 = one branch at p=128: 512.
        assert_eq!(costs[0], 512);
        // leaves 1 and 2 each take two branches at p=128: 1024 each.
        assert_eq!(costs[1], 1024);
        assert_eq!(costs[2], 1024);
    }

    /// Asymmetric probability: at `prob=1` (almost certain bit-1), the
    /// "0" branch is expensive and the "1" branch is cheap. Hand-traced
    /// expected: `vp9_cost_bit(1, false)` ≫ `vp9_cost_bit(1, true)`.
    #[test]
    fn asymmetric_prob_costs_differ() {
        let tree: [i8; 2] = [-0, -1];
        let probs: [u8; 1] = [1]; // P(bit=0) = 1/256 → 1 is "near-certain"
        let mut costs = [0i32; 2];
        vp9_cost_tokens(&mut costs, &tree, &probs);
        // Leaf 0 path takes the "0" branch: expensive.
        // Leaf 1 path takes the "1" branch: cheap.
        assert!(
            costs[0] > costs[1],
            "asymmetric prob: cost[0]={} should be > cost[1]={}",
            costs[0],
            costs[1]
        );
    }

    /// Deep unbalanced tree: 5 leaves. Verifies the recursion handles
    /// nested internal nodes and that each leaf's accumulated cost is
    /// the sum of all decisions on its path.
    /// ```text
    ///        root (i=0, probs[0])
    ///        /                \
    ///   internal 2          leaf 0
    ///   /         \
    /// leaf 1    internal 4
    ///           /         \
    ///        leaf 2     internal 6
    ///                   /         \
    ///                leaf 3     leaf 4
    /// ```
    #[test]
    fn deep_unbalanced_tree() {
        let tree: [i8; 8] = [2, -0, -1, 4, -2, 6, -3, -4];
        let probs: [u8; 4] = [128, 128, 128, 128];
        let mut costs = [0i32; 5];
        vp9_cost_tokens(&mut costs, &tree, &probs);
        // Leaf 0: one branch at root. cost = 512.
        assert_eq!(costs[0], 512);
        // Leaf 1: 2 branches. cost = 1024.
        assert_eq!(costs[1], 1024);
        // Leaf 2: 3 branches. cost = 1536.
        assert_eq!(costs[2], 1536);
        // Leaf 3: 4 branches. cost = 2048.
        assert_eq!(costs[3], 2048);
        // Leaf 4: 4 branches. cost = 2048.
        assert_eq!(costs[4], 2048);
    }

    /// `vp9_cost_tokens_skip` zeros the EOB-leaf cost to just the
    /// EOB-prob = 0 branch and skips its sub-tree.
    #[test]
    fn skip_variant() {
        let tree: [i8; 4] = [-0, 2, -1, -2];
        let probs: [u8; 2] = [128, 128];
        let mut costs = [0i32; 3];
        vp9_cost_tokens_skip(&mut costs, &tree, &probs);
        // Leaf 0 cost = cost_zero(probs[0]) = 512.
        assert_eq!(costs[0], 512);
        // Sub-tree below cost_walk(probs[1]=128): leaves 1 & 2 each
        // get one branch at 128, so 512 — note we reset to 0, not
        // continue from leaf-0's cost.
        assert_eq!(costs[1], 512);
        assert_eq!(costs[2], 512);
    }
}
