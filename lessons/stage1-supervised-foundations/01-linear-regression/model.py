"""Linear Regression from scratch, in pure NumPy.

A textbook algorithm: least-squares linear regression. There is no single
seminal paper — the method of least squares traces to Legendre (1805) and Gauss.
We fit the closed-form **normal equation**, the exact one-shot solution:

    w = (Xb^T Xb)^-1 Xb^T y

where Xb is the feature matrix X with a column of 1s prepended so the model can
learn a bias (intercept) alongside the per-feature weights. No iteration, no
learning rate — just linear algebra. (Lesson 2 re-fits this same model with
gradient descent, so this file stays deliberately closed-form.)

The model is tiny: for d input features it is just d + 1 numbers — one weight per
feature plus the bias.
"""

import numpy as np


class LinearRegression:
    """Ordinary least-squares linear regression via the normal equation.

    Attributes (populated by ``fit``):
        weights: shape (n_features,) — one coefficient per input feature.
        bias:    scalar intercept.
    """

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Solve for the weights that minimize mean squared error.

        Args:
            X: shape (n_samples, n_features) — the feature matrix.
            y: shape (n_samples,)            — the continuous target.

        Returns self, so you can chain ``LinearRegression().fit(X, y)``.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Prepend a column of 1s so the bias is learned as just another weight.
        # Xb has shape (n_samples, n_features + 1).
        ones = np.ones((X.shape[0], 1))
        Xb = np.hstack([ones, X])

        # The normal equation: theta = (Xb^T Xb)^-1 Xb^T y.
        # np.linalg.lstsq solves this least-squares system directly and is more
        # numerically stable than forming the inverse by hand (it uses SVD and
        # copes with rank-deficient X). It returns the same theta the boxed
        # formula does — see README §4 for the by-hand inverse version.
        theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)

        # theta[0] is the bias (weight on the all-ones column); the rest are the
        # per-feature weights.
        self.bias = float(theta[0])
        self.weights = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for X. Shape (n_samples,).

        Each prediction is the dot product of the features with the learned
        weights, plus the bias:  y_hat = X @ w + b.
        """
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias


if __name__ == "__main__":
    # Sanity check: fit on a tiny slice of a known linear relationship and show
    # the model recovers it. y = 3*x0 - 2*x1 + 5, so weights ~= [3, -2], bias ~= 5.
    rng = np.random.default_rng(0)
    X_demo = rng.normal(size=(50, 2))
    y_demo = 3.0 * X_demo[:, 0] - 2.0 * X_demo[:, 1] + 5.0

    model = LinearRegression().fit(X_demo, y_demo)

    # R^2 on the training slice: 1.0 means the line fits perfectly (this data is
    # exactly linear, so it should).
    y_pred = model.predict(X_demo)
    ss_res = np.sum((y_demo - y_pred) ** 2)
    ss_tot = np.sum((y_demo - y_demo.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    n_params = model.weights.size + 1
    print(
        f"fitted; train R^2: {r2:.4f}  |  params: {n_params}  "
        f"(weights={np.round(model.weights, 3)}, bias={model.bias:.3f})"
    )
