//! Classification metrics for the trained-model benchmarks.
//!
//! # Why this exists
//!
//! `bench-medmnist` used to report accuracy and a majority-class baseline and
//! nothing else. That is not enough to judge a *medical* classifier, and on
//! this repo's own weights it actively misleads: PneumoniaMNIST's test set is
//! ~60% positive, and the embedded model scores **98.3% sensitivity against
//! 65.3% specificity** — it earns its 85% accuracy largely by over-predicting
//! disease, flagging 42 of 121 healthy scans. Accuracy alone cannot see that;
//! balanced accuracy and the confusion matrix can.
//!
//! The second gap was comparability. MedMNIST's published tables lead with
//! **AUC** (ResNet-18 at 28×28: 0.944 on pneumonia, 0.997 on organa), and a
//! benchmark that reports only accuracy cannot be placed against them on their
//! headline metric. AUC needs a *score*, not a decision, which is why
//! [`CnnModel::forward_logits`](crate::bench::cnn::CnnModel::forward_logits)
//! exists.
//!
//! The third was that no interval was ever attached to a number. On 624 test
//! images, 84.78% carries a 95% interval of roughly [81.7, 87.4] — wide enough
//! to contain the published ResNet-18 figure, so "matches ResNet-18" is a claim
//! about indistinguishability, not about parity. Reporting the interval makes
//! the difference between the two visible instead of rhetorical.
//!
//! Everything here is computed on **plaintext** weights. The encrypted pipeline
//! reveals only the argmax by construction, so these are properties of the
//! model, measured on the same integer weights the hybrid runs.

use serde::Serialize;

/// Wilson score interval for a binomial proportion at 95%.
///
/// Wilson rather than the textbook normal approximation because accuracies
/// here sit near 1 and the samples are small (624 images, or the 300-image
/// embedded subset), exactly where the normal interval misbehaves and can
/// even run past 1.
pub fn wilson_interval(successes: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    const Z: f64 = 1.96;
    let (k, n_f) = (successes as f64, n as f64);
    let p = k / n_f;
    let denom = 1.0 + Z * Z / n_f;
    let centre = (p + Z * Z / (2.0 * n_f)) / denom;
    let half = Z * (p * (1.0 - p) / n_f + Z * Z / (4.0 * n_f * n_f)).sqrt() / denom;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Area under the ROC curve for one binary problem, via the Mann–Whitney U
/// statistic over average ranks.
///
/// Rank-based rather than trapezoidal so that **ties are handled correctly**:
/// these logits are integers and genuinely tie, and counting a tie as a win
/// would inflate the number. A tie contributes exactly 0.5, which is what
/// averaging the ranks within each tied group achieves.
///
/// Returns `None` when one class is absent — AUC is undefined there, and a
/// benchmark must say "undefined" rather than print 0.5 or 1.0.
pub fn auc(scores: &[i64], positive: &[bool]) -> Option<f64> {
    assert_eq!(scores.len(), positive.len(), "scores/labels length mismatch");
    let n_pos = positive.iter().filter(|&&p| p).count();
    let n_neg = positive.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by_key(|&i| scores[i]);
    // Average rank within each tied group (1-based).
    let mut ranks = vec![0.0f64; scores.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &idx in &order[i..=j] {
            ranks[idx] = avg;
        }
        i = j + 1;
    }
    let rank_sum: f64 = (0..scores.len()).filter(|&i| positive[i]).map(|i| ranks[i]).sum();
    let u = rank_sum - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0;
    Some(u / (n_pos as f64 * n_neg as f64))
}

/// Everything the benchmark reports about a set of predictions.
#[derive(Debug, Serialize)]
pub struct ClassificationMetrics {
    pub n: usize,
    pub correct: usize,
    pub accuracy: f64,
    /// 95% Wilson interval on `accuracy`.
    pub accuracy_ci: (f64, f64),
    /// Mean of the per-class recalls — the number to read instead of accuracy
    /// when the test set is imbalanced, as PneumoniaMNIST's is.
    pub balanced_accuracy: f64,
    /// Recall per class, in class order.
    pub per_class_recall: Vec<f64>,
    /// Support (true count) per class.
    pub per_class_n: Vec<usize>,
    /// `confusion[true][predicted]`.
    pub confusion: Vec<Vec<usize>>,
    /// Binary AUC, or the macro one-vs-rest average over classes. `None` when
    /// undefined (a class with no positives or no negatives).
    pub auc: Option<f64>,
    /// Binary only: recall on the positive (disease) class.
    pub sensitivity: Option<f64>,
    /// Binary only: recall on the negative (healthy) class.
    pub specificity: Option<f64>,
}

impl ClassificationMetrics {
    /// `logits[i]` are the output-layer scores for image `i`; `labels[i]` its
    /// true class. `nclasses` is taken explicitly so a class absent from the
    /// evaluated subset still gets a row.
    pub fn compute(logits: &[Vec<i64>], labels: &[u8], nclasses: usize) -> Self {
        assert_eq!(logits.len(), labels.len(), "logits/labels length mismatch");
        let n = labels.len();
        let preds: Vec<usize> = logits.iter().map(|l| crate::gadgets::argmax_ref(l)).collect();

        let mut confusion = vec![vec![0usize; nclasses]; nclasses];
        for (&t, &p) in labels.iter().zip(&preds) {
            confusion[t as usize][p] += 1;
        }
        let per_class_n: Vec<usize> = confusion.iter().map(|row| row.iter().sum()).collect();
        let per_class_recall: Vec<f64> = (0..nclasses)
            .map(|c| {
                if per_class_n[c] == 0 {
                    f64::NAN
                } else {
                    confusion[c][c] as f64 / per_class_n[c] as f64
                }
            })
            .collect();
        // Classes with no support are skipped rather than counted as 0, which
        // would silently drag the mean down on a subset that misses a class.
        let present: Vec<f64> = per_class_recall.iter().copied().filter(|r| !r.is_nan()).collect();
        let balanced_accuracy = if present.is_empty() {
            f64::NAN
        } else {
            present.iter().sum::<f64>() / present.len() as f64
        };

        let correct: usize = (0..nclasses).map(|c| confusion[c][c]).sum();
        let accuracy = if n == 0 { f64::NAN } else { correct as f64 / n as f64 };

        // AUC: the binary case uses the decision margin, which is the score the
        // model actually ranks by; the multiclass case averages one-vs-rest.
        let auc = if nclasses == 2 {
            let scores: Vec<i64> = logits.iter().map(|l| l[1] - l[0]).collect();
            let pos: Vec<bool> = labels.iter().map(|&l| l == 1).collect();
            auc(&scores, &pos)
        } else {
            let per: Vec<f64> = (0..nclasses)
                .filter_map(|c| {
                    let scores: Vec<i64> = logits.iter().map(|l| l[c]).collect();
                    let pos: Vec<bool> = labels.iter().map(|&l| l as usize == c).collect();
                    auc(&scores, &pos)
                })
                .collect();
            if per.is_empty() {
                None
            } else {
                Some(per.iter().sum::<f64>() / per.len() as f64)
            }
        };

        let (sensitivity, specificity) = if nclasses == 2 {
            (Some(per_class_recall[1]), Some(per_class_recall[0]))
        } else {
            (None, None)
        };

        Self {
            n,
            correct,
            accuracy,
            accuracy_ci: wilson_interval(correct, n),
            balanced_accuracy,
            per_class_recall,
            per_class_n,
            confusion,
            auc,
            sensitivity,
            specificity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A perfect ranking is 1.0, a reversed one 0.0, and an all-ties one
    /// exactly 0.5 — the tie case is the one a trapezoidal implementation
    /// silently gets wrong on integer logits.
    #[test]
    fn auc_handles_perfect_reversed_and_tied_rankings() {
        let pos = [false, false, true, true];
        assert_eq!(auc(&[1, 2, 3, 4], &pos), Some(1.0));
        assert_eq!(auc(&[4, 3, 2, 1], &pos), Some(0.0));
        assert_eq!(auc(&[7, 7, 7, 7], &pos), Some(0.5));
        // One tied pair straddling the boundary: 3 clean wins + 1 half.
        assert_eq!(auc(&[1, 5, 5, 9], &pos), Some(0.875));
    }

    #[test]
    fn auc_is_undefined_when_a_class_is_missing() {
        assert_eq!(auc(&[1, 2, 3], &[true, true, true]), None);
        assert_eq!(auc(&[1, 2, 3], &[false, false, false]), None);
    }

    /// The Wilson interval must bracket the estimate and stay inside [0,1]
    /// even at the extremes, where the normal approximation escapes it.
    #[test]
    fn wilson_stays_inside_the_unit_interval() {
        let (lo, hi) = wilson_interval(624, 624);
        assert!(lo > 0.99 && hi <= 1.0, "got [{lo}, {hi}]");
        let (lo, hi) = wilson_interval(0, 100);
        assert!(lo >= 0.0 && hi < 0.05, "got [{lo}, {hi}]");
        let (lo, hi) = wilson_interval(529, 624); // the pneumonia figure
        assert!(lo < 0.8478 && 0.8478 < hi, "got [{lo}, {hi}]");
    }

    /// Balanced accuracy must expose the imbalance that plain accuracy hides:
    /// a model that calls everything "positive" on a 90/10 split scores 0.9
    /// accuracy but only 0.5 balanced.
    #[test]
    fn balanced_accuracy_exposes_a_majority_class_collapse() {
        let mut logits = vec![vec![0i64, 1i64]; 100]; // always predicts class 1
        let mut labels = vec![1u8; 90];
        labels.extend(std::iter::repeat(0u8).take(10));
        logits.truncate(100);
        let m = ClassificationMetrics::compute(&logits, &labels, 2);
        assert!((m.accuracy - 0.90).abs() < 1e-12, "accuracy {}", m.accuracy);
        assert!(
            (m.balanced_accuracy - 0.5).abs() < 1e-12,
            "balanced {}",
            m.balanced_accuracy
        );
        assert_eq!(m.sensitivity, Some(1.0));
        assert_eq!(m.specificity, Some(0.0));
    }

    /// The confusion matrix must be indexed `[true][predicted]`, not the
    /// transpose — getting this backwards swaps sensitivity and specificity,
    /// which reads as a plausible model rather than a bug.
    #[test]
    fn confusion_is_indexed_true_then_predicted() {
        // One true-0 predicted as 1, nothing else.
        let logits = vec![vec![0i64, 5i64]];
        let labels = vec![0u8];
        let m = ClassificationMetrics::compute(&logits, &labels, 2);
        assert_eq!(m.confusion[0][1], 1, "true 0 predicted 1 belongs at [0][1]");
        assert_eq!(m.confusion[1][0], 0);
    }
}
