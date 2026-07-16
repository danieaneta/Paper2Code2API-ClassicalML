"""Inference for Linear Regression.

Shared by api.py and usable standalone:
    python infer.py '[8.3, 41, 6.98, 1.02, 322, 2.55, 37.88, -122.23]'

Callers pass **RAW, unscaled** feature values in the dataset's column order:
    [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
The saved artifact OWNS the scaler, so we standardize here before predicting —
the caller never has to know the training-set statistics.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from model import LinearRegression

HERE = Path(__file__).parent
ARTIFACT_PATH = HERE / "linear_regression.npz"

# California Housing column order — what a /predict feature vector must contain.
FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


@lru_cache(maxsize=1)
def load_model() -> tuple[LinearRegression, np.ndarray, np.ndarray]:
    """Load params once and cache. Returns (model, scaler_mean, scaler_scale).

    Raises FileNotFoundError (with a fix-it message) if the artifact is missing.
    """
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"artifact not found at {ARTIFACT_PATH}. Run `python train.py` first."
        )
    data = np.load(ARTIFACT_PATH, allow_pickle=False)
    model = LinearRegression()
    model.weights = data["weights"]
    model.bias = float(data["bias"])
    return model, data["mean"], data["scale"]


def predict(features: list[float]) -> dict:
    """Turn one RAW feature vector into the contract's regress shape.

    Returns {"value": <predicted median house value, in $100,000s>}.
    """
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"expected {len(FEATURE_NAMES)} features {FEATURE_NAMES}, "
            f"got {len(features)}."
        )

    model, mean, scale = load_model()

    # Apply the SAME scaling the model was trained with, then predict.
    x_raw = np.asarray(features, dtype=np.float64)
    x_scaled = (x_raw - mean) / scale
    value = float(model.predict(x_scaled.reshape(1, -1))[0])

    return {"value": value}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python infer.py '[8.3, 41, 6.98, 1.02, 322, 2.55, 37.88, -122.23]'")
        raise SystemExit(1)
    vec = json.loads(sys.argv[1])
    print(predict(vec))
