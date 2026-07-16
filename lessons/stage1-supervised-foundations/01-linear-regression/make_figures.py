"""Generate the teaching diagrams for the README.

Produces (into ./assets):
  * fit_line.png        - the single strongest feature (MedInc, median income)
                          vs. house value, with the best-fit line drawn through
                          the cloud. The "what regression does" picture.
  * pred_vs_actual.png  - a clean predicted-vs-actual diagnostic on the test set
                          with the y = x reference line and the test R^2 annotated.

Usage:
    python make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from model import LinearRegression
from infer import load_model

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

SEED = 0
TEST_SIZE = 0.2
MEDINC_IDX = 0  # MedInc is the first California Housing column


def main() -> None:
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=TEST_SIZE, random_state=SEED
    )

    # ---- 1) Best-fit line over the single strongest feature (MedInc) ----
    # Fit a one-feature regression purely for this teaching plot, so the drawn
    # line is exactly the least-squares fit of value ~ MedInc.
    medinc = X_train[:, MEDINC_IDX].reshape(-1, 1)
    line_model = LinearRegression().fit(medinc, y_train)

    xs = np.linspace(medinc.min(), medinc.max(), 100).reshape(-1, 1)
    ys = line_model.predict(xs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(medinc, y_train, s=6, alpha=0.15, color="#1565c0", edgecolors="none",
               label="houses (train)")
    ax.plot(xs, ys, color="#c62828", linewidth=2.5, label="best-fit line")
    ax.set_xlabel("median income in block (MedInc, $10,000s)")
    ax.set_ylabel("median house value ($100,000s)")
    ax.set_title("What regression does: fit a line through the cloud\n"
                 "median income vs. house value",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "fit_line.png", dpi=130)
    plt.close(fig)

    # ---- 2) Predicted-vs-actual diagnostic (full 8-feature model) ----
    model, mean, scale = load_model()
    y_pred = model.predict((X_test - mean) / scale)
    r2 = r2_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y_test, y_pred, s=8, alpha=0.3, color="#1565c0", edgecolors="none")
    lims = [0, 5.2]
    ax.plot(lims, lims, "--", color="#c62828", linewidth=1.5, label="perfect (y = x)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("actual value ($100,000s)")
    ax.set_ylabel("predicted value ($100,000s)")
    ax.set_title("Diagnostic: predicted vs. actual (test set)",
                 fontsize=12, fontweight="bold")
    ax.text(0.05, 0.92, f"test R$^2$ = {r2:.3f}", transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999"))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "pred_vs_actual.png", dpi=130)
    plt.close(fig)

    print(f"wrote {ASSETS / 'fit_line.png'}")
    print(f"wrote {ASSETS / 'pred_vs_actual.png'}")


if __name__ == "__main__":
    main()
