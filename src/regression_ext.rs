//! Extended regression & classification: Elastic Net, Softmax, Naive Bayes,
//! k-NN, Decision Tree, metrics (precision/recall/F1/AUC), cross-val.

use crate::errors::{SciError, SciResult};
use crate::regression::LinearModel;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

// ══════════════════════════════════════════════════════════════════════════════
// Elastic Net (L1 + L2 via coordinate descent)
// ══════════════════════════════════════════════════════════════════════════════

/// **Elastic Net** regression: minimise ‖y − Xβ‖² + λ(α‖β‖₁ + (1−α)‖β‖²).
/// `alpha` ∈ [0,1]: 0 = Ridge, 1 = Lasso.
pub fn elastic_net(
    x: &[Vec<f64>],
    y: &[f64],
    lambda: f64,
    alpha: f64,
    tolerance: f64,
    max_iter: usize,
) -> SciResult<LinearModel> {
    let n = x.len();
    let p = x[0].len();
    if y.len() != n {
        return Err(SciError::InvalidParameter("x/y length mismatch"));
    }
    let mut beta = vec![0.0f64; p + 1];
    let l1 = lambda * alpha;
    let l2 = lambda * (1.0 - alpha);
    for _ in 0..max_iter {
        let old = beta.clone();
        // intercept
        let res_int: f64 = y
            .iter()
            .zip(x)
            .map(|(&yi, xi)| yi - dot(xi, &beta[1..]))
            .sum();
        beta[0] = res_int / n as f64;
        for j in 0..p {
            let resj: f64 = y
                .iter()
                .zip(x)
                .map(|(&yi, xi)| {
                    let mut pred = beta[0];
                    for k in 0..p {
                        if k != j {
                            pred += beta[k + 1] * xi[k];
                        }
                    }
                    (yi - pred) * xi[j]
                })
                .sum();
            let zj = resj / n as f64;
            beta[j + 1] = soft_threshold(zj, l1) / (1.0 + l2);
        }
        let diff: f64 = beta.iter().zip(&old).map(|(a, b)| (a - b).abs()).sum();
        if diff < tolerance {
            break;
        }
    }
    let intercept = beta[0];
    let coefficients = beta[1..].to_vec();
    let y_pred: Vec<f64> = x
        .iter()
        .map(|xi| intercept + dot(xi, &coefficients))
        .collect();
    let ss_res: f64 = y.iter().zip(&y_pred).map(|(a, b)| (a - b) * (a - b)).sum();
    let y_mean = mean(y);
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean) * (yi - y_mean)).sum();
    let r_squared = 1.0 - ss_res / ss_tot;
    Ok(LinearModel {
        coefficients,
        intercept,
        r_squared,
        rse: ss_res.sqrt(),
    })
}

fn soft_threshold(z: f64, lam: f64) -> f64 {
    if z > lam {
        z - lam
    } else if z < -lam {
        z + lam
    } else {
        0.0
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Softmax / Multinomial Logistic Regression (one-vs-rest via SGD)
// ══════════════════════════════════════════════════════════════════════════════

/// Softmax multinomial logistic regression result.
pub struct SoftmaxModel {
    pub weights: Vec<Vec<f64>>, // [n_classes][n_features+1]
    pub n_classes: usize,
}

impl SoftmaxModel {
    /// Predict class probabilities for each sample.
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> SciResult<Vec<Vec<f64>>> {
        x.iter()
            .map(|xi| {
                let logits: Vec<f64> = self
                    .weights
                    .iter()
                    .map(|w| w[0] + dot(&w[1..], xi))
                    .collect();
                Ok(softmax_vec(&logits))
            })
            .collect()
    }
    /// Predict class index (argmax).
    pub fn predict(&self, x: &[Vec<f64>]) -> SciResult<Vec<usize>> {
        self.predict_proba(x)?
            .iter()
            .map(|probs| {
                Ok(probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0))
            })
            .collect()
    }
}

/// Train softmax multinomial logistic regression via mini-batch SGD.
pub fn softmax_regression(
    x: &[Vec<f64>],
    y_labels: &[usize],
    n_classes: usize,
    lr: f64,
    lambda: f64,
    max_iter: usize,
) -> SciResult<SoftmaxModel> {
    let n = x.len();
    let p = x[0].len();
    if y_labels.len() != n {
        return Err(SciError::InvalidParameter("x/y length mismatch"));
    }
    if y_labels.iter().any(|&c| c >= n_classes) {
        return Err(SciError::InvalidParameter("label >= n_classes"));
    }
    let mut w = vec![vec![0.0f64; p + 1]; n_classes];
    for iter in 0..max_iter {
        let lr_t = lr / (1.0 + 0.001 * iter as f64);
        for i in 0..n {
            let logits: Vec<f64> = w.iter().map(|wk| wk[0] + dot(&wk[1..], &x[i])).collect();
            let probs = softmax_vec(&logits);
            for k in 0..n_classes {
                let err = probs[k] - if y_labels[i] == k { 1.0 } else { 0.0 };
                w[k][0] -= lr_t * err;
                for j in 0..p {
                    w[k][j + 1] -= lr_t * (err * x[i][j] + lambda * w[k][j + 1]);
                }
            }
        }
    }
    Ok(SoftmaxModel {
        weights: w,
        n_classes,
    })
}

fn softmax_vec(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = v.iter().map(|&vi| (vi - max).exp()).collect();
    let s: f64 = exp.iter().sum();
    exp.iter().map(|e| e / s).collect()
}

// ══════════════════════════════════════════════════════════════════════════════
// Gaussian Naive Bayes
// ══════════════════════════════════════════════════════════════════════════════

pub struct NaiveBayes {
    pub class_log_prior: Vec<f64>,
    pub means: Vec<Vec<f64>>,
    pub vars: Vec<Vec<f64>>,
    pub n_classes: usize,
}

impl NaiveBayes {
    pub fn fit(x: &[Vec<f64>], y: &[usize], n_classes: usize) -> SciResult<Self> {
        let p = x[0].len();
        let mut counts = vec![0usize; n_classes];
        let mut sums = vec![vec![0.0f64; p]; n_classes];
        let mut sum_sq = vec![vec![0.0f64; p]; n_classes];
        for (xi, &yi) in x.iter().zip(y) {
            if yi >= n_classes {
                return Err(SciError::InvalidParameter("label out of range"));
            }
            counts[yi] += 1;
            for j in 0..p {
                sums[yi][j] += xi[j];
                sum_sq[yi][j] += xi[j] * xi[j];
            }
        }
        let n = x.len() as f64;
        let means: Vec<Vec<f64>> = (0..n_classes)
            .map(|k| {
                sums[k]
                    .iter()
                    .map(|&s| s / counts[k].max(1) as f64)
                    .collect()
            })
            .collect();
        let vars: Vec<Vec<f64>> = (0..n_classes)
            .map(|k| {
                let nk = counts[k].max(1) as f64;
                (0..p)
                    .map(|j| (sum_sq[k][j] / nk - means[k][j] * means[k][j]).max(1e-9))
                    .collect()
            })
            .collect();
        let class_log_prior: Vec<f64> = counts.iter().map(|&c| (c as f64 / n).ln()).collect();
        Ok(Self {
            class_log_prior,
            means,
            vars,
            n_classes,
        })
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter()
            .map(|xi| {
                let log_posts: Vec<f64> = (0..self.n_classes)
                    .map(|k| {
                        let pi = core::f64::consts::PI;
                        let ll: f64 = xi
                            .iter()
                            .enumerate()
                            .map(|(j, &xij)| {
                                let v = self.vars[k][j];
                                -0.5 * ((xij - self.means[k][j]).powi(2) / v
                                    + v.ln()
                                    + (2.0 * pi).ln())
                            })
                            .sum();
                        self.class_log_prior[k] + ll
                    })
                    .collect();
                log_posts
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// k-Nearest Neighbours
// ══════════════════════════════════════════════════════════════════════════════

/// k-NN classifier: predicts majority class among k nearest (Euclidean) neighbours.
pub fn knn_classify(
    train_x: &[Vec<f64>],
    train_y: &[usize],
    test_x: &[Vec<f64>],
    k: usize,
) -> SciResult<Vec<usize>> {
    if k == 0 {
        return Err(SciError::InvalidParameter("k >= 1"));
    }
    test_x
        .iter()
        .map(|xi| {
            let mut dists: Vec<(f64, usize)> = train_x
                .iter()
                .zip(train_y)
                .map(|(ti, &yi)| {
                    let d: f64 = xi
                        .iter()
                        .zip(ti)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    (d, yi)
                })
                .collect();
            dists.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut votes = std::collections::HashMap::new();
            for &(_, c) in &dists[..k.min(dists.len())] {
                *votes.entry(c).or_insert(0) += 1;
            }
            Ok(votes
                .into_iter()
                .max_by_key(|e| e.1)
                .map(|(c, _)| c)
                .unwrap_or(0))
        })
        .collect()
}

/// k-NN regressor: predicts mean of k nearest neighbours.
pub fn knn_regress(
    train_x: &[Vec<f64>],
    train_y: &[f64],
    test_x: &[Vec<f64>],
    k: usize,
) -> SciResult<Vec<f64>> {
    if k == 0 {
        return Err(SciError::InvalidParameter("k >= 1"));
    }
    test_x
        .iter()
        .map(|xi| {
            let mut dists: Vec<(f64, f64)> = train_x
                .iter()
                .zip(train_y)
                .map(|(ti, &yi)| {
                    let d: f64 = xi
                        .iter()
                        .zip(ti)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    (d, yi)
                })
                .collect();
            dists.sort_by(|a, b| a.0.total_cmp(&b.0));
            let s: f64 = dists[..k.min(dists.len())].iter().map(|e| e.1).sum();
            Ok(s / k.min(dists.len()) as f64)
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════════════
// Decision Tree (CART, binary splits, Gini impurity, classification)
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum TreeNode {
    Leaf {
        class: usize,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn predict_one(&self, x: &[f64]) -> usize {
        match self {
            TreeNode::Leaf { class } => *class,
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if x[*feature] <= *threshold {
                    left.predict_one(x)
                } else {
                    right.predict_one(x)
                }
            }
        }
    }
}

/// Build a CART decision tree (Gini, binary splits).
pub fn decision_tree_fit(
    x: &[Vec<f64>],
    y: &[usize],
    n_classes: usize,
    max_depth: usize,
    min_samples: usize,
) -> SciResult<TreeNode> {
    if x.is_empty() {
        return Err(SciError::InvalidParameter("empty data"));
    }
    Ok(build_tree(x, y, n_classes, max_depth, min_samples, 0))
}

pub fn decision_tree_predict(tree: &TreeNode, x: &[Vec<f64>]) -> Vec<usize> {
    x.iter().map(|xi| tree.predict_one(xi)).collect()
}

fn build_tree(
    x: &[Vec<f64>],
    y: &[usize],
    nc: usize,
    max_d: usize,
    min_s: usize,
    depth: usize,
) -> TreeNode {
    let majority = majority_class(y, nc);
    if depth >= max_d || y.len() <= min_s || gini(y, nc) < 1e-10 {
        return TreeNode::Leaf { class: majority };
    }
    let p = x[0].len();
    let mut best = (f64::INFINITY, 0usize, 0.0f64);
    for j in 0..p {
        let mut vals: Vec<f64> = x.iter().map(|xi| xi[j]).collect();
        vals.sort_by(|a, b| a.total_cmp(b));
        vals.dedup();
        for i in 0..vals.len().saturating_sub(1) {
            let thresh = (vals[i] + vals[i + 1]) / 2.0;
            let (yl, yr): (Vec<_>, Vec<_>) = y.iter().zip(x).partition(|(_, xi)| xi[j] <= thresh);
            let yl: Vec<usize> = yl.iter().map(|p| *p.0).collect();
            let yr: Vec<usize> = yr.iter().map(|p| *p.0).collect();
            if yl.is_empty() || yr.is_empty() {
                continue;
            }
            let score = (yl.len() as f64 * gini(&yl, nc) + yr.len() as f64 * gini(&yr, nc))
                / y.len() as f64;
            if score < best.0 {
                best = (score, j, thresh);
            }
        }
    }
    if best.0 == f64::INFINITY {
        return TreeNode::Leaf { class: majority };
    }
    let (_, feat, thresh) = best;
    let (left_data, right_data): (Vec<_>, Vec<_>) =
        x.iter().zip(y).partition(|(xi, _)| xi[feat] <= thresh);
    let (xl, yl): (Vec<_>, Vec<_>) = left_data
        .iter()
        .map(|(xi, yi)| ((*xi).clone(), **yi))
        .unzip();
    let (xr, yr): (Vec<_>, Vec<_>) = right_data
        .iter()
        .map(|(xi, yi)| ((*xi).clone(), **yi))
        .unzip();
    TreeNode::Split {
        feature: feat,
        threshold: thresh,
        left: Box::new(build_tree(&xl, &yl, nc, max_d, min_s, depth + 1)),
        right: Box::new(build_tree(&xr, &yr, nc, max_d, min_s, depth + 1)),
    }
}

fn gini(y: &[usize], nc: usize) -> f64 {
    let n = y.len() as f64;
    let mut counts = vec![0usize; nc];
    for &c in y {
        if c < nc {
            counts[c] += 1;
        }
    }
    1.0 - counts.iter().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
}

fn majority_class(y: &[usize], nc: usize) -> usize {
    let mut counts = vec![0usize; nc.max(1)];
    for &c in y {
        if c < nc {
            counts[c] += 1;
        }
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|e| e.1)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ══════════════════════════════════════════════════════════════════════════════
// Classification metrics
// ══════════════════════════════════════════════════════════════════════════════

/// Confusion matrix (rows = true, cols = predicted), shape n_classes × n_classes.
pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize], n_classes: usize) -> Vec<Vec<usize>> {
    let mut cm = vec![vec![0usize; n_classes]; n_classes];
    for (&t, &p) in y_true.iter().zip(y_pred) {
        if t < n_classes && p < n_classes {
            cm[t][p] += 1;
        }
    }
    cm
}

/// Per-class precision, recall, F1.  Returns (precision, recall, f1) per class.
pub fn classification_report(
    y_true: &[usize],
    y_pred: &[usize],
    n_classes: usize,
) -> Vec<(f64, f64, f64)> {
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    (0..n_classes)
        .map(|k| {
            let tp = cm[k][k] as f64;
            let fp: f64 = (0..n_classes)
                .filter(|&r| r != k)
                .map(|r| cm[r][k] as f64)
                .sum();
            let fn_: f64 = (0..n_classes)
                .filter(|&c| c != k)
                .map(|c| cm[k][c] as f64)
                .sum();
            let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let rec = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f1 = if prec + rec > 0.0 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };
            (prec, rec, f1)
        })
        .collect()
}

/// Binary AUC-ROC via trapezoidal rule over threshold-sorted predictions.
pub fn auc_roc(y_true: &[usize], y_score: &[f64]) -> SciResult<f64> {
    if y_true.len() != y_score.len() {
        return Err(SciError::InvalidParameter("length mismatch"));
    }
    let mut pairs: Vec<(f64, usize)> = y_score
        .iter()
        .cloned()
        .zip(y_true.iter().cloned())
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let n_pos = y_true.iter().filter(|&&c| c == 1).count() as f64;
    let n_neg = y_true.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return Err(SciError::InvalidParameter("need both classes"));
    }
    let (mut tp, mut fp, mut auc) = (0.0f64, 0.0f64, 0.0f64);
    let (mut prev_tp, mut prev_fp) = (0.0f64, 0.0f64);
    for (_, c) in &pairs {
        if *c == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        if fp != prev_fp {
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
            prev_fp = fp;
            prev_tp = tp;
        }
    }
    auc += (n_neg - prev_fp) * (n_pos + prev_tp) / 2.0;
    Ok(auc / (n_pos * n_neg))
}

/// Macro-averaged F1 across all classes.
pub fn macro_f1(y_true: &[usize], y_pred: &[usize], n_classes: usize) -> f64 {
    let report = classification_report(y_true, y_pred, n_classes);
    report.iter().map(|(_, _, f1)| f1).sum::<f64>() / n_classes as f64
}

// ══════════════════════════════════════════════════════════════════════════════
// Cross-validation for classification (stratified k-fold accuracy)
// ══════════════════════════════════════════════════════════════════════════════

/// k-fold cross-validation accuracy using k-NN classifier.
pub fn kfold_cv_knn(
    x: &[Vec<f64>],
    y: &[usize],
    _n_classes: usize,
    k_fold: usize,
    k_nn: usize,
) -> SciResult<f64> {
    let n = x.len();
    if k_fold < 2 {
        return Err(SciError::InvalidParameter("k_fold >= 2"));
    }
    let fold_size = n / k_fold;
    let mut total_acc = 0.0f64;
    for fold in 0..k_fold {
        let lo = fold * fold_size;
        let hi = if fold == k_fold - 1 {
            n
        } else {
            lo + fold_size
        };
        let (tx, ty): (Vec<_>, Vec<_>) = (0..n)
            .filter(|&i| i < lo || i >= hi)
            .map(|i| (x[i].clone(), y[i]))
            .unzip();
        let vx: Vec<Vec<f64>> = (lo..hi).map(|i| x[i].clone()).collect();
        let vy: Vec<usize> = (lo..hi).map(|i| y[i]).collect();
        let preds = knn_classify(&tx, &ty, &vx, k_nn)?;
        let acc = preds.iter().zip(&vy).filter(|(a, b)| a == b).count() as f64 / vy.len() as f64;
        total_acc += acc;
    }
    Ok(total_acc / k_fold as f64)
}

/// Poisson regression via IRLS (log link, Poisson likelihood).
pub fn poisson_regression(
    x: &[Vec<f64>],
    y: &[f64],
    max_iter: usize,
    tol: f64,
) -> SciResult<LinearModel> {
    let p = x[0].len();
    if y.iter().any(|&v| v < 0.0) {
        return Err(SciError::InvalidParameter("y must be non-negative"));
    }
    let mut beta = vec![0.0f64; p + 1];
    for _ in 0..max_iter {
        // μᵢ = exp(Xβ)
        let mu: Vec<f64> = x
            .iter()
            .map(|xi| (beta[0] + dot(xi, &beta[1..])).exp().max(1e-300))
            .collect();
        // Score: Xᵀ(y − μ), Hessian: Xᵀ·diag(μ)·X  — gradient step
        let mut grad = vec![0.0f64; p + 1];
        grad[0] = y.iter().zip(&mu).map(|(&yi, &mi)| yi - mi).sum();
        for j in 0..p {
            grad[j + 1] = x
                .iter()
                .zip(y.iter())
                .zip(&mu)
                .map(|((xi, &yi), &mi)| (yi - mi) * xi[j])
                .sum();
        }
        let fisher_diag0: f64 = mu.iter().sum();
        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm < tol {
            break;
        }
        beta[0] += 0.01 * grad[0] / fisher_diag0.max(1.0);
        for j in 0..p {
            let fjj: f64 = x.iter().zip(&mu).map(|(xi, &mi)| mi * xi[j] * xi[j]).sum();
            beta[j + 1] += 0.01 * grad[j + 1] / fjj.max(1e-10);
        }
    }
    let intercept = beta[0];
    let coefficients = beta[1..].to_vec();
    let y_pred: Vec<f64> = x
        .iter()
        .map(|xi| (intercept + dot(xi, &coefficients)).exp())
        .collect();
    let ss_res: f64 = y.iter().zip(&y_pred).map(|(a, b)| (a - b) * (a - b)).sum();
    let ym = mean(y);
    let ss_tot: f64 = y.iter().map(|yi| (yi - ym) * (yi - ym)).sum();
    let r_squared = 1.0 - ss_res / ss_tot.max(f64::EPSILON);
    Ok(LinearModel {
        coefficients,
        intercept,
        r_squared,
        rse: ss_res.sqrt(),
    })
}
