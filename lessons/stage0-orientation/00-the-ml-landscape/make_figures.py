"""Generate the teaching diagrams for Lesson 0 (The Machine Learning Landscape).

This lesson is a no-code concept primer, so there is no model to train or serve
here -- just two figures that make the two big ideas of the course visual:

Produces (into ./assets):
  * supervised_vs_unsupervised.png
        A two-panel figure on the SAME point cloud. Left: points colored by their
        known class with a legend (supervised = we already have the answers).
        Right: the identical points in one neutral color (unsupervised = labels
        hidden, the algorithm must find the groups itself).
  * ml_taxonomy.png
        A taxonomy tree of machine learning drawn from matplotlib boxes and
        connector lines. The two branches this course teaches -- Supervised
        (Regression, Classification) and Unsupervised (Clustering, Dimensionality
        Reduction) -- are drawn solid; Reinforcement Learning and Semi-supervised
        are greyed/dashed as "mentioned, not built in this course".

Usage:
    python make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import make_blobs

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

SEED = 0

# A small, consistent palette reused across both figures.
CLASS_COLORS = ["#1565c0", "#c62828", "#2e7d32"]  # blue / red / green
CLASS_MARKERS = ["o", "s", "^"]
CLASS_NAMES = ["class A", "class B", "class C"]
NEUTRAL = "#6b6b6b"  # the "no label" grey


def make_supervised_vs_unsupervised() -> Path:
    """Same three-cluster cloud, shown labeled (left) then unlabeled (right)."""
    # Explicit, well-separated centers + a tighter spread so the three groups
    # are visually obvious even in the unlabeled (grey) panel — the whole
    # teaching point is that the eye can find the clumps without labels.
    X, y = make_blobs(
        n_samples=300,
        centers=[[-4.5, -2.0], [4.5, -3.0], [0.0, 5.0]],
        cluster_std=0.85,
        random_state=SEED,
    )

    fig, (ax_sup, ax_unsup) = plt.subplots(1, 2, figsize=(11, 5.2))

    # ---- Left: supervised -- points colored by their KNOWN class ----
    for k in range(3):
        mask = y == k
        ax_sup.scatter(
            X[mask, 0], X[mask, 1],
            s=28, alpha=0.85, color=CLASS_COLORS[k], marker=CLASS_MARKERS[k],
            edgecolors="white", linewidths=0.4, label=CLASS_NAMES[k],
        )
    ax_sup.set_title(
        "Supervised: labeled examples\n"
        "every point comes with its answer -- learn to reproduce it",
        fontsize=12, fontweight="bold",
    )
    ax_sup.legend(loc="upper left", framealpha=0.9, title="known label")

    # ---- Right: unsupervised -- identical points, ALL one neutral color ----
    ax_unsup.scatter(
        X[:, 0], X[:, 1],
        s=28, alpha=0.85, color=NEUTRAL, marker="o",
        edgecolors="white", linewidths=0.4,
    )
    ax_unsup.set_title(
        "Unsupervised: no labels\n"
        "same data, labels hidden -- find the groups yourself",
        fontsize=12, fontweight="bold",
    )

    # Shared, deliberately bare axes: this figure is about the coloring, not units.
    for ax in (ax_sup, ax_unsup):
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    out = ASSETS / "supervised_vs_unsupervised.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _box(ax, xy, text, *, width, height, facecolor, edgecolor,
         textcolor="black", fontsize=11, fontweight="bold",
         linestyle="solid", linewidth=1.6):
    """Draw a centered, rounded text box; return its center (for connectors)."""
    cx, cy = xy
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, linestyle=linestyle, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", zorder=3,
            fontsize=fontsize, fontweight=fontweight, color=textcolor)
    return cx, cy


def _connect(ax, top, bottom, *, color="#555", linestyle="solid", linewidth=1.4,
             mid_y=None):
    """Elbow connector from the bottom of a parent box to the top of a child.

    `mid_y` pins the height of the horizontal jog; by default it sits halfway
    between the two boxes. Pinning it lets the greyed secondary branches route
    their horizontal run clear of the leaf-row connectors.
    """
    x0, y0 = top      # parent center
    x1, y1 = bottom   # child center
    mid = (y0 + y1) / 2 + 0.02 if mid_y is None else mid_y
    ax.plot([x0, x0, x1, x1], [y0, mid, mid, y1],
            color=color, linestyle=linestyle, linewidth=linewidth,
            zorder=1, solid_capstyle="round")


def make_taxonomy() -> Path:
    """A hand-laid taxonomy tree of ML using matplotlib boxes + connectors."""
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Palette for the tree.
    ROOT_FC, ROOT_EC = "#263238", "#263238"      # dark slate, filled
    SUP_FC, SUP_EC = "#bbdefb", "#1565c0"        # blue family (supervised)
    UNSUP_FC, UNSUP_EC = "#c8e6c9", "#2e7d32"    # green family (unsupervised)
    GREY_FC, GREY_EC = "#eeeeee", "#9e9e9e"      # greyed secondary branches
    LEAF_W, LEAF_H = 2.5, 0.72

    # ---- Root ----
    root = _box(ax, (6.0, 6.6), "Machine Learning",
                width=3.2, height=0.82,
                facecolor=ROOT_FC, edgecolor=ROOT_EC,
                textcolor="white", fontsize=14)

    # ---- Primary branches (taught) ----
    sup = _box(ax, (3.0, 4.9), "Supervised\n(labeled data)",
               width=3.0, height=0.9,
               facecolor=SUP_FC, edgecolor=SUP_EC, fontsize=12)
    unsup = _box(ax, (9.0, 4.9), "Unsupervised\n(no labels)",
                 width=3.0, height=0.9,
                 facecolor=UNSUP_FC, edgecolor=UNSUP_EC, fontsize=12)

    _connect(ax, root, sup, color=SUP_EC)
    _connect(ax, root, unsup, color=UNSUP_EC)

    # ---- Supervised leaves ----
    reg = _box(ax, (1.65, 2.9), "Regression\npredict a number",
               width=LEAF_W, height=LEAF_H + 0.14,
               facecolor="white", edgecolor=SUP_EC, fontsize=10.5)
    clf = _box(ax, (4.35, 2.9), "Classification\npredict a category",
               width=LEAF_W, height=LEAF_H + 0.14,
               facecolor="white", edgecolor=SUP_EC, fontsize=10.5)
    _connect(ax, sup, reg, color=SUP_EC)
    _connect(ax, sup, clf, color=SUP_EC)

    # ---- Unsupervised leaves ----
    clust = _box(ax, (7.65, 2.9), "Clustering\ngroup points",
                 width=LEAF_W, height=LEAF_H + 0.14,
                 facecolor="white", edgecolor=UNSUP_EC, fontsize=10.5)
    dimred = _box(ax, (10.35, 2.9), "Dimensionality\nReduction\ncompress features",
                  width=LEAF_W, height=LEAF_H + 0.42,
                  facecolor="white", edgecolor=UNSUP_EC, fontsize=10.5)
    _connect(ax, unsup, clust, color=UNSUP_EC)
    _connect(ax, unsup, dimred, color=UNSUP_EC)

    # ---- Secondary branches (greyed / dashed, not built) ----
    rl = _box(ax, (3.0, 0.95), "Reinforcement\nLearning",
              width=2.8, height=0.9,
              facecolor=GREY_FC, edgecolor=GREY_EC, textcolor="#616161",
              fontsize=10.5, linestyle="dashed", linewidth=1.4)
    semi = _box(ax, (9.0, 0.95), "Semi-supervised",
                width=2.8, height=0.9,
                facecolor=GREY_FC, edgecolor=GREY_EC, textcolor="#616161",
                fontsize=10.5, linestyle="dashed", linewidth=1.4)
    # Route the horizontal jog low (below the leaf row) so the dashed secondary
    # branches stay visually clear of the solid leaf connectors.
    _connect(ax, root, rl, color=GREY_EC, linestyle="dashed", mid_y=2.05)
    _connect(ax, root, semi, color=GREY_EC, linestyle="dashed", mid_y=2.05)

    ax.text(6.0, 0.28,
            "(mentioned, not built in this course)",
            ha="center", va="center", fontsize=10, style="italic", color="#616161")

    ax.set_title("The machine learning landscape",
                 fontsize=15, fontweight="bold", pad=12)

    fig.tight_layout()
    out = ASSETS / "ml_taxonomy.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    for out in (make_supervised_vs_unsupervised(), make_taxonomy()):
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
