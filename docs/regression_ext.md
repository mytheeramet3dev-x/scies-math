# Extended Regression and Machine Learning (`regression_ext`)

The `regression_ext` module provides regularized linear regression, multi-class classification algorithms, decision trees, cross-validation utilities, and classification metrics.

---

## 1. Elastic Net Regularization (`elastic_net`)

Combines $L_1$ (Lasso) and $L_2$ (Ridge) penalties to balance sparsity and group selection:

$$\min_{\beta} \frac{1}{2n} \|y - X \beta\|_2^2 + \lambda \left(\alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2\right)$$

where:
- $\lambda \ge 0$ controls the overall regularization strength.
- $\alpha \in [0, 1]$ balances between pure Ridge ($\alpha = 0$) and pure Lasso ($\alpha = 1$).

### Coordinate Descent Algorithm
Updates each coefficient $\beta_j$ sequentially via cyclical soft-thresholding:

$$\beta_j \leftarrow \frac{S\left(\frac{1}{n} \sum_{i=1}^n x_{ij} r_i^{(j)}, \alpha \lambda\right)}{1 + (1 - \alpha) \lambda}$$

where $r_i^{(j)} = y_i - \sum_{k \ne j} x_{ik} \beta_k$ is the partial residual, and $S(z, \tau) = \text{sign}(z) \max(0, |z| - \tau)$ is the soft-thresholding operator.

```rust
use scies_math_th::regression_ext::elastic_net;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 4.0],
        vec![4.0, 3.0],
    ];
    let y = vec![5.0, 4.0, 11.0, 10.0]; // y = 2*x0 + x1 + 1

    let model = elastic_net(
        &x, &y,
        0.01, // lambda
        0.5,  // alpha (50% Lasso, 50% Ridge)
        1e-6, // tolerance
        1000, // max_iter
    )?;

    println!("Intercept: {:.4}", model.intercept);
    println!("Coefficients: {:?}", model.coefficients);
    Ok(())
}
```

---

## 2. Multi-Class Softmax Regression (`softmax_train`)

Generalizes logistic regression to $K$ classes using normalized exponential probabilities:

$$P(Y = k \mid x) = \frac{\exp(w_k^T x + b_k)}{\sum_{j=1}^K \exp(w_j^T x + b_j)}$$

Optimized via gradient descent with cross-entropy loss:

$$\mathcal{L}(W) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K \mathbb{I}(y_i = k) \ln P(Y = k \mid x_i)$$

---

## 3. Gaussian Naive Bayes (`naive_bayes_fit`)

Assumes class-conditional feature independence:

$$P(x \mid Y = k) = \prod_{j=1}^p \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2 \sigma_{kj}^2}\right)$$

Fast $O(n \cdot p)$ training and $O(K \cdot p)$ prediction, well suited for high-dimensional baseline classification.

---

## 4. $k$-Nearest Neighbors Classifier (`knn_predict`)

Non-parametric lazy classification using Euclidean distance metric:

$$d(x, x_i) = \sqrt{\sum_{j=1}^p (x_j - x_{ij})^2}$$

Assigns the majority class vote among the $k$ nearest training samples.

```rust
use scies_math_th::regression_ext::knn_predict;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_x = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.2],
        vec![5.0, 5.0],
        vec![5.2, 4.8],
    ];
    let train_y = vec![0, 0, 1, 1];

    let query = vec![0.05, 0.1];
    let pred_class = knn_predict(&train_x, &train_y, &query, 3)?;
    assert_eq!(pred_class, 0);

    Ok(())
}
```

---

## 5. Classification and Validation Metrics

The module provides full evaluation metrics:
- `accuracy(y_true, y_pred)`: Proportion of correct predictions.
- `confusion_matrix(y_true, y_pred, n_classes)`: Returns full contingency table.
- `precision_recall_f1(y_true, y_pred)`: Returns precision, recall, and harmonic F1 score.
- `roc_auc(y_true, y_scores)`: Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC) using trapezoidal integration.
- `k_fold_split(n, k, seed)`: Splits $n$ samples into $k$ deterministic train/test index folds.
