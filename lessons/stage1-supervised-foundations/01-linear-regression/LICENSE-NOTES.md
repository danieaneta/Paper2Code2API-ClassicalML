# License notes — Linear Regression

| Layer | Status |
|---|---|
| **Algorithm** | Ordinary least-squares linear regression is public-domain mathematics (Legendre 1805, Gauss). No single seminal paper, no patent, no license to inherit. |
| **Implementation** | Original pure-NumPy code written for classical-ML-paper2code2api. |
| **Parameters** | The shipped `linear_regression.npz` (weights, bias, scaler mean/scale) was trained from scratch here on California Housing → produced by this repo, MIT-shippable, no upstream weight license. |
| **Dataset (California Housing)** | Derived from the **1990 US Census** and distributed with scikit-learn. Public-domain census-derived data, widely used for research/education. We fetch it at train time via `sklearn.datasets.fetch_california_housing` rather than vendoring it. |

**Bottom line:** Safe for open-source / commercial-educational release. The algorithm is public math, the code is original, the parameters are self-trained, and the dataset is public and downloaded at runtime, not redistributed.

> As with every entry in this repo: code license ≠ parameters license ≠ dataset license. This note reflects the state at authoring and should be re-checked before publishing.
