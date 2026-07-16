"""Train the from-scratch Linear Regression on California Housing.

Usage:
    python train.py                    # seeded 80/20 split, saves linear_regression.npz
    python train.py --test-size 0.25   # custom held-out fraction
    python train.py --seed 7           # custom split seed

What it does, in order (this is the honest supervised-learning loop):
    1. Load California Housing (sklearn; derived from the 1990 US Census).
    2. Split into train/test with a fixed seed (reproducible).
    3. Standardize features — fit the scaler on TRAIN ONLY, then apply to both.
       Fitting the scaler on all the data would leak test statistics into
       training; we never do that (see README §9).
    4. Fit model.py on the scaled training features.
    5. Report R^2 on the HELD-OUT test set — data the model never saw.
    6. Save weights + bias + the scaler's mean/scale to linear_regression.npz,
       so inference callers can send RAW, unscaled feature values.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from model import LinearRegression

HERE = Path(__file__).parent
ARTIFACT_PATH = HERE / "linear_regression.npz"


def train(test_size: float = 0.2, seed: int = 0) -> None:
    # --- 1. Load data ---------------------------------------------------------
    data = fetch_california_housing()
    X, y = data.data, data.target  # X: (20640, 8), y: median house value ($100k)
    print(f"dataset: California Housing  |  X={X.shape}  features={list(data.feature_names)}")

    # --- 2. Seeded train/test split ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # --- 3. Standardize features (fit on TRAIN only — no leakage) ------------
    # z = (x - mean) / scale, using the training set's per-column statistics.
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0.0] = 1.0  # guard against a constant column (division by 0)
    X_train_scaled = (X_train - mean) / scale
    X_test_scaled = (X_test - mean) / scale

    # --- 4. Fit our from-scratch model ---------------------------------------
    model = LinearRegression().fit(X_train_scaled, y_train)

    # --- 5. Honest evaluation on the held-out test set -----------------------
    # R^2 = fraction of the target's variance the model explains. 0.0 = no better
    # than always predicting the mean; 1.0 = perfect.
    r2_test = r2_score(y_test, model.predict(X_test_scaled))

    # Baselines to keep us honest (ML_STANDARDS: always establish a baseline).
    #  * mean predictor: R^2 is ~= 0.0 (predicting the train mean on the test
    #    set lands just under 0) — the floor to beat.
    #  * sklearn LinearRegression: fits the SAME normal equation, so matching it
    #    proves our from-scratch math is correct.
    baseline_mean = r2_score(y_test, np.full_like(y_test, y_train.mean()))
    sk = SklearnLinearRegression().fit(X_train_scaled, y_train)
    r2_sklearn = r2_score(y_test, sk.predict(X_test_scaled))

    print(f"test R^2 (ours):      {r2_test:.4f}")
    print(f"test R^2 (mean base): {baseline_mean:.4f}   <- the floor to beat")
    print(f"test R^2 (sklearn):   {r2_sklearn:.4f}   <- should match ours")

    # --- 6. Save the artifact (arrays only; it OWNS the scaler) --------------
    np.savez(
        ARTIFACT_PATH,
        weights=model.weights,
        bias=np.array(model.bias),
        mean=mean,
        scale=scale,
        feature_names=np.array(data.feature_names),
    )
    print(f"saved -> {ARTIFACT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Linear Regression on California Housing")
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out fraction")
    parser.add_argument("--seed", type=int, default=0, help="train/test split seed")
    args = parser.parse_args()
    train(test_size=args.test_size, seed=args.seed)
