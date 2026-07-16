# Paper2Code2API · Classical ML

> **A hands-on course in classical machine learning — learn by rebuilding the algorithms ML is built from.**
> For each foundational algorithm you'll understand the idea in plain English, build it yourself from scratch in NumPy, train it, and wrap it in a working API.

Most people learn machine learning backwards: they call `model.fit()` on a black box and never see what's inside. This course does the opposite. You start at the very first ideas — a line through some points — and work forward, **re-implementing each algorithm from scratch** with heavily-commented NumPy and a beginner-friendly lesson — then turning each model into a callable API so you can actually *use* what you built.

Every lesson does the thing that usually stops people cold: it **maps the algorithm's actual math to the exact lines of code that implement it.** That math-to-code leap is the real skill — and the whole point of this course.

Think of it as a textbook where every chapter ends with a running program.

> **This is the free foundations course.** When you're ready to go deeper — neural networks and the landmark papers behind modern computer vision — continue with the companion course: **[Paper2Code2API · Computer Vision](https://github.com/danieaneta/paper2code2api)** (LeNet-5, AlexNet, ResNet, and beyond). Classical ML here is the on-ramp; the perceptron in Lesson 5 is your first taste of the neural nets that course is built on.

---

## Why this course

- **You'll actually understand ML, not just call it.** Every model is built from scratch in NumPy — no `sklearn.fit()` hiding the machinery. scikit-learn shows up only to load datasets and to *check your from-scratch version is correct*.
- **Math becomes code.** Each lesson takes the defining equations (MSE, the normal equation, the gradient step, Gini impurity...) and shows the handful of NumPy lines that *are* that math.
- **You build things that run.** Every algorithm ends as a real `POST /predict` API you can send data to — the same contract across all lessons, so the models are interchangeable.
- **It's genuinely free and $0 to follow.** Open datasets, CPU-only, no paid compute, no accounts.

---

## How to use this as a course

1. **Start at Lesson 0** and work down in order — each lesson builds on ideas from the last.
2. **Open the lesson README** in each folder. It teaches the concept from scratch — no prior ML knowledge assumed.
3. **Build it yourself.** Each lesson walks through the code (`model.py`, `train.py`, `infer.py`, `api.py`) and ends with exercises.
4. **Run it.** Train the model, then start the API and send it your own feature vectors.

**Prerequisites:** basic Python (functions, classes, lists). The math is explained in plain language as it comes up — you do *not* need a math degree to start.

```bash
# run any built lesson
cd lessons/stage1-supervised-foundations/01-linear-regression
pip install -r requirements.txt
python train.py            # trains the model from scratch, ships the parameters
uvicorn api:app --reload   # then open http://127.0.0.1:8000/docs
```

---

## Table of Contents — the curriculum

Lessons marked **Live** are written and runnable; **Planned** ones are added over time.

### Orientation

| # | Lesson | What it covers | Status |
|---|---|---|---|
| 0 | [**The Machine Learning Landscape**](lessons/stage0-orientation/00-the-ml-landscape) | The no-code map: supervised vs. unsupervised, the core vocabulary, and where every lesson fits | Live |

### Stage 1 · Supervised Learning & the Optimization Engine

| # | Lesson | What you'll build | Status |
|---|---|---|---|
| 1 | [**Linear Regression**](lessons/stage1-supervised-foundations/01-linear-regression) | A house-price predictor (California Housing) + a `POST /predict` API | Live |
| 2 | **Gradient Descent** | Re-fit linear regression iteratively — the optimizer everything reuses | Planned |
| 3 | **Logistic Regression** | From regression to classification (sigmoid, cross-entropy) | Planned |
| 4 | **Regularization (Ridge & Lasso) + Cross-Validation** | Overfitting, bias-variance, and honest model selection | Planned |

### Stage 2 · Classical Classifiers

| # | Lesson | What you'll build | Status |
|---|---|---|---|
| 5 | **Perceptron** | The ancestor of neural nets — an online linear classifier | Planned |
| 6 | **k-Nearest Neighbors** | Instance-based classification; the idea behind vector search | Planned |
| 7 | **Naive Bayes** | Probabilistic classification (the classic spam filter) | Planned |
| 8 | **Support Vector Machine** | Max-margin classification & the kernel idea | Planned |

### Stage 3 · Trees & Ensembles

| # | Lesson | What you'll build | Status |
|---|---|---|---|
| 9 | **Decision Tree (CART)** | Recursive partitioning — an interpretable classifier | Planned |
| 10 | **Random Forest** | Bagging: many trees beat one | Planned |
| 11 | **AdaBoost** | Boosting: turn weak learners into a strong one | Planned |

### Stage 4 · Unsupervised Learning

| # | Lesson | What you'll build | Status |
|---|---|---|---|
| 12 | **k-Means** | Clustering without labels (customer segmentation) | Planned |
| 13 | **PCA** | Dimensionality reduction & visualization | Planned |

---

## Quick taste — Lesson 1 output

The first build lesson trains linear regression from scratch to predict California house values, reaching **R² ≈ 0.59** — and proves the from-scratch math is right by matching scikit-learn to four decimals:

![Linear Regression predicted-vs-actual on California Housing](lessons/stage1-supervised-foundations/01-linear-regression/assets/examples_grid.png)

**Start here: [Lesson 0 — The Machine Learning Landscape](lessons/stage0-orientation/00-the-ml-landscape)**

---

## What every lesson folder contains

| File | What it's for |
|---|---|
| `README.md` | **The lesson** — explains the algorithm for beginners and walks you through building it |
| `model.py` | The algorithm, implemented from scratch in pure NumPy and commented |
| `train.py` | Trains the model on its dataset and saves the learned parameters |
| `infer.py` | Runs a prediction on a single feature vector |
| `api.py` | A FastAPI server exposing the shared `POST /predict` contract |
| `make_figures.py` / `make_examples.py` | Generate the teaching diagrams and input/output examples |
| `<model>.npz` / `.pkl` | Pretrained parameters, shipped so `infer.py`/`api.py` work the moment you clone — retrain anytime with `train.py` |
| `requirements.txt` | What to `pip install` |
| `LICENSE-NOTES.md` | License status for that lesson's algorithm, parameters, and data |

> Lesson 0 is the one exception — a no-code concept primer with just the reading and two diagrams.

## The shared API contract

Every model speaks the same language, so they're interchangeable:

```
POST /predict   (JSON: { "features": [f1, f2, ... ] })  ->  JSON result
GET  /health                                            ->  { status, model_loaded }
```

The return shape matches the task: `classify` returns `{ label, prob }`, `regress` returns `{ value }`, `cluster-assign` returns `{ cluster_id, distance }`, `transform` returns `{ components }`. You send **raw** feature values — each model ships with its own scaler, so you never have to standardize by hand.

---

## Stay updated

New lessons drop over time. To follow along:

- **Star** this repo and set **Watch → Releases** to get notified when new lessons land.
- Work through the lessons in order — the curriculum above is the roadmap.

<!--
  NEWSLETTER SLOT — uncomment and drop in the signup link once the platform is chosen:
  **Prefer email?** [Get each new lesson in your inbox](NEWSLETTER_SIGNUP_URL)
-->

---

## Licensing

Course code is **MIT** (see [`LICENSE`](LICENSE)). The datasets are the standard open ones bundled with scikit-learn (e.g. California Housing, Iris, Wine, Breast Cancer, Digits) — public and free to use for learning. Each lesson's `LICENSE-NOTES.md` states the status for that lesson's algorithm, parameters, and data.

## Status

Early and active. Lesson 0 (the ML Landscape) and Lesson 1 (Linear Regression) are live and runnable; the rest follow the same template and are added over time. Contributions welcome — pick a lesson, follow the lesson pattern, open a PR.
