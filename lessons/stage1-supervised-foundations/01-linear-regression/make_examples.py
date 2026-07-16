"""Generate the input/output example figures for the README.

Produces (into ./assets):
  * examples_grid.png      - the HERO figure: predicted-vs-actual on the test set,
                             each point a house, colored by absolute error
                             (bright/yellow = accurate, dark/purple = far off). The classical-ML
                             analogue of the CV course's image grid.
  * prediction_detail.png  - one test house: its raw features on the left, the
                             predicted vs. actual value on the right.

Usage:
    python make_examples.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from infer import FEATURE_NAMES, load_model

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

# Must match train.py's defaults so the "test set" here is the same held-out data.
SEED = 0
TEST_SIZE = 0.2


def _test_set_predictions():
    """Recreate the seeded split and predict on the held-out test set."""
    data = fetch_california_housing()
    _, X_test, _, y_test = train_test_split(
        data.data, data.target, test_size=TEST_SIZE, random_state=SEED
    )
    model, mean, scale = load_model()
    y_pred = model.predict((X_test - mean) / scale)
    return X_test, y_test, y_pred


def main() -> None:
    X_test, y_test, y_pred = _test_set_predictions()
    abs_err = np.abs(y_pred - y_test)

    # ---- 1) HERO: predicted vs actual, colored by absolute error ----
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    sc = ax.scatter(
        y_test, y_pred, c=abs_err, cmap="viridis_r",
        s=10, alpha=0.5, edgecolors="none",
    )
    lims = [0, 5.2]  # target is capped at 5.0 ($500k) in this dataset
    ax.plot(lims, lims, "--", color="#c62828", linewidth=1.5, label="perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("actual median house value ($100,000s)")
    ax.set_ylabel("predicted value ($100,000s)")
    ax.set_title("Linear Regression on California Housing\npredicted vs. actual (test set)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("absolute error ($100,000s)")
    fig.tight_layout()
    fig.savefig(ASSETS / "examples_grid.png", dpi=130)
    plt.close(fig)

    # ---- 2) single input -> output detail ----
    i = 0  # first test house
    x_raw, actual, pred = X_test[i], float(y_test[i]), float(y_pred[i])

    fig, (ax_feat, ax_val) = plt.subplots(
        1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    # Left: the raw feature vector as a labeled table-ish bar.
    ax_feat.axis("off")
    ax_feat.set_title("INPUT — one house (raw features)", fontsize=10, fontweight="bold")
    lines = [f"{name:>11} = {val:.2f}" for name, val in zip(FEATURE_NAMES, x_raw)]
    ax_feat.text(0.02, 0.95, "\n".join(lines), family="monospace", fontsize=10,
                 va="top", ha="left", transform=ax_feat.transAxes)

    # Right: predicted vs actual value as two bars.
    bars = ax_val.bar(["predicted", "actual"], [pred, actual],
                      color=["#1565c0", "#2e7d32"])
    ax_val.set_ylabel("median house value ($100,000s)")
    ax_val.set_title(f"OUTPUT — value\nerror = {abs(pred - actual):.2f} ($100k)",
                     fontsize=10, fontweight="bold")
    for bar, v in zip(bars, [pred, actual]):
        ax_val.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10)
    ax_val.set_ylim(0, max(pred, actual) * 1.25)

    fig.tight_layout()
    fig.savefig(ASSETS / "prediction_detail.png", dpi=130)
    plt.close(fig)

    print(f"wrote {ASSETS / 'examples_grid.png'}")
    print(f"wrote {ASSETS / 'prediction_detail.png'}")


if __name__ == "__main__":
    main()
