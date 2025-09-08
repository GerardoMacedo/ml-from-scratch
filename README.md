# Logistic Regression — From Scratch (NumPy) vs scikit-learn

Hands-on ML project implementing logistic regression **from first principles** (NumPy only), with comparisons to scikit-learn on the Iris dataset. Includes tests and a CLI.

## Highlights
- Batch gradient descent with L2 regularization
- One-vs-rest for multiclass
- Standardization pipeline and train/test split
- Unit tests to validate behavior and accuracy
- CLI to compare our model vs scikit-learn

## Quick Start
```bash
# create/enter repo, then:
pip install -U pip
pip install numpy scikit-learn
python -m scratchml.cli
python -m scratchml.cli --binary

