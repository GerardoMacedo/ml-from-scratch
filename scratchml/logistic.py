from __future__ import annotations
import numpy as np

class LogisticRegressionScratch:
    """
    Multiclass via one-vs-rest (OVR).
    Implements batch gradient descent with L2 regularization.
    """
    def __init__(self, lr: float = 0.1, n_iters: int = 2000, reg: float = 0.0, fit_intercept: bool = True, random_state: int = 42):
        self.lr = lr
        self.n_iters = n_iters
        self.reg = reg
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None   # shape: (n_classes, n_features [+1 if intercept])
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # stable sigmoid
        z_clip = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit binary logistic regression (labels 0/1) and return weights w.
        """
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        w = rng.normal(0, 0.01, size=(d,))
        for _ in range(self.n_iters):
            y_hat = self._sigmoid(X @ w)
            # gradient of negative log-likelihood + L2
            grad = (X.T @ (y_hat - y)) / n + self.reg * np.r_[0.0 if self.fit_intercept else [], w[1:]] if self.fit_intercept else self.reg * w
            # The above needs careful intercept handling:
            if self.reg > 0:
                if self.fit_intercept:
                    reg_vec = np.concatenate(([0.0], w[1:]))  # don't regularize intercept
                else:
                    reg_vec = w
            else:
                reg_vec = np.zeros_like(w)
            grad = (X.T @ (y_hat - y)) / n + self.reg * reg_vec
            w -= self.lr * grad
        return w

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Xb = self._add_intercept(X)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        d = Xb.shape[1]

        if n_classes == 2:
            # binary: train one classifier on {0,1}
            y_bin = (y == self.classes_[1]).astype(float)
            self.coef_ = self._fit_binary(Xb, y_bin)[None, :]  # shape (1,d)
        else:
            # multiclass OVR: fit one binary per class (class vs rest)
            W = []
            for c in self.classes_:
                y_bin = (y == c).astype(float)
                w = self._fit_binary(Xb, y_bin)
                W.append(w)
            self.coef_ = np.vstack(W)  # shape (n_classes,d)
        return self

    def _scores(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_intercept(X)
        # for binary we keep coef_ shape (1,d); make consistent
        W = self.coef_
        logits = Xb @ W.T  # (n_samples, n_classes or 1)
        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._scores(X)
        if logits.shape[1] == 1:
            # binary: return 2-column probabilities
            p1 = self._sigmoid(logits[:, 0])
            p0 = 1 - p1
            return np.column_stack([p0, p1])
        # multiclass OVR: apply sigmoid per class, then normalize
        P = self._sigmoid(logits)
        P_sum = P.sum(axis=1, keepdims=True) + 1e-12
        return P / P_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]
