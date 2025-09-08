import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLogReg
from scratchml.logistic import LogisticRegressionScratch

def prepare_data(binary=False):
    data = load_iris()
    X, y = data.data, data.target
    if binary:
        y = (y == 0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te

class TestLogRegScratch(unittest.TestCase):
    def test_binary_close_to_sklearn(self):
        X_tr, X_te, y_tr, y_te = prepare_data(binary=True)
        ours = LogisticRegressionScratch(lr=0.1, n_iters=3000, reg=0.0).fit(X_tr, y_tr)
        acc_ours = (ours.predict(X_te) == y_te).mean()

        sk = SkLogReg(max_iter=1000, solver="liblinear")  # good for binary
        sk.fit(X_tr, y_tr)
        acc_sk = (sk.predict(X_te) == y_te).mean()

        self.assertGreaterEqual(acc_ours, 0.95)
        self.assertGreaterEqual(acc_sk, 0.95)

    def test_multiclass_reasonable_accuracy(self):
        X_tr, X_te, y_tr, y_te = prepare_data(binary=False)
        ours = LogisticRegressionScratch(lr=0.1, n_iters=3000, reg=0.0).fit(X_tr, y_tr)
        acc_ours = (ours.predict(X_te) == y_te).mean()
        self.assertGreaterEqual(acc_ours, 0.80)  # reasonable bar for scratch OVR

if __name__ == "__main__":
    unittest.main()
