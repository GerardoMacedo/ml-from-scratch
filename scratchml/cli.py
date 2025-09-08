import argparse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLogReg

from scratchml.logistic import LogisticRegressionScratch

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def load_iris_data(multiclass=True, test_size=0.25, random_state=42):
    data = load_iris()
    X, y = data.data, data.target
    if not multiclass:
        # binary: class 0 vs the rest
        y = (y == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # standardize (fit on train, apply to both)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test

def main():
    ap = argparse.ArgumentParser(description="Logistic Regression from scratch (compare vs scikit-learn)")
    ap.add_argument("--lr", type=float, default=0.1, help="learning rate")
    ap.add_argument("--iters", type=int, default=3000, help="training iterations")
    ap.add_argument("--reg", type=float, default=0.0, help="L2 regularization strength")
    ap.add_argument("--binary", action="store_true", help="use binary task (setosa vs rest)")
    args = ap.parse_args()

    X_train, X_test, y_train, y_test = load_iris_data(multiclass=not args.binary)

    # our model
    ours = LogisticRegressionScratch(lr=args.lr, n_iters=args.iters, reg=args.reg, fit_intercept=True)
    ours.fit(X_train, y_train)
    y_pred_ours = ours.predict(X_test)
    acc_ours = accuracy(y_test, y_pred_ours)

    # sklearn model (as reference)
    solver = "lbfgs" if not args.binary else "liblinear"
    sk = SkLogReg(max_iter=1000, multi_class="ovr" if args.binary else "auto", solver=solver)
    sk.fit(X_train, y_train)
    y_pred_sk = sk.predict(X_test)
    acc_sk = accuracy(y_test, y_pred_sk)

    task = "binary (setosa vs rest)" if args.binary else "multiclass (3 classes)"
    print(f"Task: {task}")
    print(f"Ours accuracy:      {acc_ours:.3f}")
    print(f"scikit-learn acc.:  {acc_sk:.3f}")

if __name__ == "__main__":
    main()
