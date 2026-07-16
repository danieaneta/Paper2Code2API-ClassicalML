<!--
============================================================================
 LESSON_TEMPLATE.md — the reusable skeleton every lesson in
 classical-ML-paper2code2api is built from.

 HOW TO USE THIS FILE
 - Copy the "README skeleton" (Part B, below) into a new lesson folder as
   README.md, then fill every <!-- fill: ... --> and replace every
   > TEMPLATE NOTE line with real content. Delete the TEMPLATE NOTE lines
   before shipping — they are guidance, not lesson text.
 - Follow the file-structure list (Part A) and the API contract (Part C)
   when writing the code files.
 - This template was extracted faithfully from the computer-vision course's
   canonical lesson (LeNet-5) and confirmed invariant against Lesson 2
   (AlexNet). Adaptations for classical ML are called out in
   > TEMPLATE NOTE lines. Do not invent new sections; do fill these in.

 HOUSE STANDARDS THIS ENCODES (from EDUCATION_STANDARDS.md):
 - Learning objectives up front (3–6 concrete "by the end you can…").
 - One concept at a time, motivate WHY before HOW, worked example then generalize.
 - Code-along, not code-dump: introduce code in digestible steps with prose
   between blocks; end with the full working version.
 - Every code block must run exactly as written (pin versions, set seeds).
 - Connect idea/math to implementation; cite the source paper (or note textbook).
 - Anticipate misconceptions inline; recap + bridge to next lesson.
 - Match the house voice — a learner should not feel a different person wrote
   each lesson.
============================================================================
-->

# Classical-ML paper2code2api — Lesson Template

This is the fill-in-the-blank skeleton for **every** lesson in the free
`classical-ML-paper2code2api` course. It has three parts:

- **Part A — Lesson folder structure**: the exact files a lesson folder contains.
- **Part B — README section-by-section template**: the lesson prose skeleton.
- **Part C — The shared API contract**: what `POST /predict` + `GET /health` return.

> The course is the free feeder for the paid `computer_vision_paper2code2api`
> course and must feel like the same brand: same voice, same section flow, same
> "understand it → build it from scratch → train it → serve it as an API" arc.

---

## Part A · Lesson folder structure

Every lesson folder contains the same set of files. This mirrors the CV course's
8-file pattern, adapted for classical ML (pure NumPy, tabular data, a pickled/JSON
params artifact instead of a `.pt` checkpoint).

| File | What it's for |
|---|---|
| `README.md` | **The lesson** — the Part B skeleton, filled in. Explains the algorithm for beginners and walks through building it. |
| `model.py` | The algorithm, **implemented from scratch in pure NumPy** (no scikit-learn `fit`/`predict` internals). A class with `fit`, `predict`, and (where relevant) `predict_proba` / `transform`. Includes an `if __name__ == "__main__"` sanity check that fits on a tiny slice and prints a shape/score so the learner can run the file directly. |
| `train.py` | Loads the lesson's sklearn dataset, does an honest train/test split (set the seed), fits `model.py`, prints test metric(s), and **saves the learned parameters** to the model artifact. `python train.py` runs with sane defaults; expose knobs (`--epochs`/`--lr`/`--k`/…) via `argparse` as the algorithm needs. |
| `<model>.pkl` **or** `<model>.npz` | **Shipped pretrained parameters** so `infer.py`/`api.py` work the instant the repo is cloned. `train.py` overwrites it. Use `.npz` for pure-array params (weights, centroids, components); use `.pkl` when the artifact is a small Python structure (e.g. a decision tree). **Not a `.pt`** — there is no PyTorch here. |
| `infer.py` | Loads the saved parameters (cache with `@lru_cache`), turns **one feature vector into a prediction**, and is usable both standalone (`python infer.py '[5.1, 3.5, 1.4, 0.2]'`) and as a library imported by `api.py`. Holds the single `predict()` function the API wraps. |
| `api.py` | FastAPI server exposing the shared `POST /predict` + `GET /health` contract (Part C). A thin wrapper over `infer.predict`; add per-lesson routes (e.g. `GET /classes`, `GET /features`) only where the algorithm warrants. |
| `make_examples.py` | Generates the **input→output example figure(s)** for the README — e.g. a scatter of test points colored by prediction (green = correct, red = wrong), or a single-sample prediction detail (features in, probability bar / predicted value out). Classical-ML analogue of the CV course's image grid. |
| `make_figures.py` | Generates the **teaching diagrams** — e.g. the decision boundary over a 2-D feature slice, the loss-surface / gradient-descent path, a scatter with fitted line, the tree diagram, the PCA variance/eigenvector plot. Classical-ML analogue of the CV course's architecture/feature-map figures. |
| `requirements.txt` | Pinned deps. Baseline for this course: `numpy`, `scikit-learn` (datasets + metrics + baseline only), `matplotlib` (figures), `fastapi`, `uvicorn[standard]`. **No `torch`.** Add `python-multipart` only if a lesson deviates to file uploads. |
| `LICENSE-NOTES.md` | License status for the algorithm (paper vs textbook), the self-trained parameters, and the dataset — the same 4-row table + "bottom line" the CV course uses. |

> TEMPLATE NOTE (assets): figures render to an `assets/` folder created by the
> two generator scripts, exactly as in the CV course. Reference them from the
> README with alt-text (accessibility is a house standard).

> TEMPLATE NOTE (NumPy-first rule): the whole point of this course is
> *from-scratch in NumPy*. scikit-learn is allowed for loading datasets, the
> train/test split, metrics, and as the **baseline to check your result
> against** — never as the implementation of the algorithm being taught. If a
> faithful from-scratch build is too heavy for one sitting (e.g. full kernel
> SVM via SMO), teach a simplified variant and say so in Part B §5.

---

## Part B · README section-by-section template

> Copy everything in this Part into the lesson's `README.md` and fill it in.
> Keep the numbered section order — it is the invariant that makes every lesson
> feel like one course. The only structural difference from the CV template is
> the **new, required §8 "In production"** section.

<!-- ======================= BEGIN README SKELETON ======================= -->

# Lesson &lt;N&gt; · &lt;Algorithm Name&gt; — &lt;Plain-English tagline&gt;

<!-- fill: e.g. "Lesson 3 · Logistic Regression — From Regression to Classification" -->

> **Stage &lt;S&gt; · &lt;Stage Name&gt;** · Difficulty &lt;Beginner|Moderate|Hard&gt; · Dataset: &lt;sklearn dataset&gt; · License: public (see `LICENSE-NOTES.md`)
> **Type:** &lt;[paper] | [textbook]&gt;
>
> Part of [**classical-ML-paper2code2api**](../../README.md) — learn machine learning by rebuilding the algorithms it's built from.

> TEMPLATE NOTE (metadata line): difficulty legend is easy · moderate ·
> hard *to reimplement from scratch in one sitting* (from the curriculum
> draft). The **[paper] vs [textbook]** tag is required and drives §1 and §4:
> [paper] cites a seminal paper; [textbook] is a canonical algorithm with no
> single paper (note its historical roots instead).

<!-- fill: opening hook (2–4 sentences). Promise the concrete artifact the
     learner will have by the end — "you will have built, trained, and served
     <algorithm> and understand every line." Name the payoff and the vibe:
     no heavy math prerequisites, just Python + curiosity. Match the CV hook's
     warmth. If this lesson reuses earlier code (e.g. gradient descent, the
     decision tree), say so here. -->

![&lt;Algorithm&gt; input/output example](assets/examples_grid.png)

<!-- fill: one sentence describing the hero figure and what "good" looks like
     in it (e.g. "each point is a test example, colored by the model's
     prediction; green rings are correct, red are wrong"). -->

### What you'll learn

<!-- fill: 3–6 concrete "by the end you can…" bullets — the learning objectives,
     UP FRONT (house standard). Cover: the problem/why it matters, the ONE core
     idea, how the algorithm works, the paper-math→code mapping skill, how to
     train it and the metric it reaches, and how to serve it behind the API. -->

- **Why** &lt;the problem&gt; is hard / what older approaches did.
- **The one big idea**: &lt;core concept in a phrase&gt;.
- **How &lt;algorithm&gt; works**, step by step, and *why* each step is there.
- **How to turn the &lt;paper's equations | algorithm's math&gt; into code** — the skill that lets you implement *any* method, not just this one.
- **How to train it** on &lt;dataset&gt; and reach &lt;expected metric, e.g. ~96% accuracy / R² ≈ 0.6&gt;.
- **How to wrap it in an API** so you can send it feature vectors and get predictions back.

**Prerequisites:** &lt;basic Python; and any earlier lesson this one builds on, e.g. "Lesson 2 (Gradient Descent)"&gt;. Everything else is explained as it comes up.

---

## 1. The problem: &lt;why this task is hard&gt;

<!-- fill: motivate WHY before HOW (house standard). Set up the concrete task in
     plain language, show why the naive/older approach struggles, and end on the
     question this algorithm answers. Use a small worked example or a tiny
     printed data snippet, like the CV lesson's pixel grid. -->

> TEMPLATE NOTE — pick ONE citation block depending on the lesson's type tag:

> **Paper:** &lt;Author(s) (Year), *Title*, Venue.&gt; ([PDF/DOI](url)) — &lt;one line on what it did / why it mattered&gt;.
>
> *(use for **[paper]** lessons — cite the seminal source, house standard)*

> **A textbook algorithm.** &lt;Algorithm&gt; has no single seminal paper; it traces to &lt;historical roots, e.g. "least squares — Legendre 1805 & Gauss"&gt;. We build the standard modern formulation.
>
> *(use for **[textbook]** lessons instead of a paper citation)*

---

## 2. The big idea: &lt;the core concept&gt;

<!-- fill: the heart of the lesson — the single concept, explained with a plain
     analogy before any math (the CV course's "stamp that slides" for
     convolution is the bar). ONE concept at a time, built incrementally. Use
     sub-headings (###) for the 2–3 moving parts. End with a one-paragraph
     "that's it — everything below is just this idea arranged a certain way." -->

> TEMPLATE NOTE: if a teaching figure helps here (a decision boundary forming,
> the sigmoid curve, a loss surface, centroids moving), reference it —
> generated by `make_figures.py`:
> `![<what it shows>](assets/<name>.png)`

---

## 3. &lt;Algorithm&gt;, step by step

<!-- fill: the algorithm's structure, walked through concretely, then mapped
     ONE-TO-ONE onto model.py — the CV course does this layer-by-layer; here it's
     step-by-step (fit loop, the update rule, the split search, the assign/update
     iteration, the eigen-decomposition, etc.). Show a schematic first, then the
     code. -->

Here's the shape of the algorithm:

```
<!-- fill: a short ASCII/pseudocode sketch of the algorithm's flow, e.g.
     initialize params
     repeat until converged:
        compute predictions
        compute loss / gradient  (or: pick best split / assign points / update centroids)
        update params
     return learned params
-->
```

> TEMPLATE NOTE: include a structural/architecture figure where it helps
> (`assets/architecture.png` or an algorithm-flow diagram from `make_figures.py`).

Here it is in `model.py`, which maps onto the sketch almost line for line:

```python
# fill: the from-scratch NumPy class. Introduce it in digestible pieces with
# prose between blocks (code-along, not code-dump), then show the full version.
# Comment each step against the sketch above.
class <Algorithm>:
    def fit(self, X, y):        # (unsupervised: fit(self, X))
        ...
    def predict(self, X):
        ...
    # predict_proba / transform where the algorithm provides them
```

<!-- fill: read the code back against the sketch in prose. Then report the
     "size" of the model the way the CV lesson reports the parameter count —
     for classical ML that might be "this model is just <k> numbers: a weight
     per feature plus a bias", or "the tree has <n> nodes". Give the learner a
     one-liner to verify it: -->

```bash
python model.py
# fill: expected sanity-check output (e.g. "fitted; train score: 0.97  |  params: 5")
```

> TEMPLATE NOTE (the "see it work" payoff): the CV lesson ends §3 by running a
> real input through the trained model and *showing* what it learned (feature
> maps). Do the classical-ML equivalent — show the fitted decision boundary, the
> regression line over the data, the learned centroids, or the top principal
> components on real data:
> `![<real fitted result>](assets/<name>.png)`

---

## 4. From the &lt;paper's equations | algorithm's math&gt; to code

<!-- fill: THE signature section of this course — the paper-math-to-code leap.
     Take each core equation and show the exact NumPy that IS that formula.
     This section applies to BOTH [paper] and [textbook] lessons: textbook
     algorithms still have defining equations (MSE, the gradient step, Gini
     impurity, the covariance eigen-problem). Use 4.1, 4.2, … sub-sections, one
     per equation. For each: show the math, translate every symbol into plain
     words, then show the NumPy. Recurring punchline: "a page of math becomes a
     handful of well-named NumPy lines." -->

> **How to read this:** for each operation we show the math, translate every
> symbol into plain words, then show the code that implements it.

### 4.1 &lt;First core equation — e.g. the loss / the model&gt;

<!-- fill: $$ equation $$, then symbol-by-symbol plain-words, then the NumPy line(s). -->

### 4.2 &lt;Second — e.g. the gradient / the update rule / the split criterion&gt;

<!-- fill: same pattern. This is often where GD from an earlier lesson is reused. -->

### 4.3 &lt;The output / decision rule — e.g. sigmoid → probability, argmax, threshold&gt;

<!-- fill: same pattern. End the section restating the punchline: seeing the
     mapping (not memorizing a library) is the transferable skill. -->

---

## 5. Faithful vs modernized — what we changed and why

<!-- fill: REQUIRED honesty section (house standard: "if something is
     simplified, say so"). List, as a numbered list, every place the code
     deviates from the original paper / the full canonical algorithm, and WHY
     (usually: beginner-readability or one-sitting scope). Then a short "kept
     faithfully" line naming what defines the algorithm and was preserved.
     Examples of what goes here: linear-SVM-by-subgradient instead of full SMO;
     closed-form vs iterative; Gaussian NB vs multinomial; k-means++ vs random
     init. If the lesson is a genuinely faithful full build with nothing
     simplified, say THAT explicitly in one line. -->

1. **&lt;What differs&gt;.** &lt;What the original did → what we do → why the change is harmless for the lesson's point&gt;.
2. **&lt;…&gt;.**

Kept faithfully: **&lt;the choices that make this algorithm itself&gt;**. Whenever this course simplifies something, you'll find a section exactly like this one telling you what and why.

---

## 6. Build it yourself

You'll work through the files in order: `model.py` (done — that was §3), then `train.py` to fit it, then `infer.py` to make predictions.

> **Already-trained parameters ship with this lesson.** The repo includes a
> ready-made `<model>.<pkl|npz>`, so `infer.py` and the API work the instant you
> clone — no waiting. We still walk through `train.py` next because *fitting it
> yourself* is the whole point; running it simply overwrites the artifact with
> your own freshly-learned parameters.

**Setup.** From inside the `&lt;lesson&gt;` folder:

```bash
pip install -r requirements.txt
```

### Training — `train.py`

<!-- fill: training has a fixed shape you'll see in every lesson — load data,
     set up the model, fit. Show the data loading + train/test split (SET THE
     SEED — house standard, reproducibility), then the fit call, then how the
     metric is computed on the HELD-OUT test set (honest evaluation — no
     leakage, ML_STANDARDS). Introduce it in pieces with prose between. -->

```python
# fill: dataset load + seeded train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = <Algorithm>()
model.fit(X_train, y_train)          # (or the GD loop, shown line by line)
score = <metric>(y_test, model.predict(X_test))   # measured on UNSEEN data
```

Run it:

```bash
python train.py <!-- fill: flags, e.g. --epochs 200 -->
```

Expected output (numbers vary slightly run-to-run):

```
<!-- fill: a REAL run's output — this must be genuine (house standard: "if it
     wasn't run, it doesn't ship"). Show the metric climbing / converging and
     the "saved parameters -> <artifact>" line. -->
```

<!-- fill: interpret the output in prose — what the metric means, why it's
     measured on the test set, what a realistic score is (don't oversell), and
     any honest caveat (e.g. "don't expect 100% — these classes overlap"). -->

### Inference — `infer.py`

<!-- fill: training's over; now USE it. Show load-params-and-cache
     (@lru_cache), then the predict() function that turns ONE feature vector
     into the contract's JSON shape (Part C). Keep predict() the single source
     of truth the API reuses. -->

```python
@lru_cache(maxsize=1)
def load_model():
    # fill: load the saved params into a fresh <Algorithm>; raise a clear
    # FileNotFoundError telling the user to run train.py if the artifact is missing.
    ...

def predict(features: list[float]) -> dict:
    # fill: returns the per-type contract shape from Part C.
    ...
```

Try it from the command line:

```bash
python infer.py '<!-- fill: an example feature vector, e.g. [5.1, 3.5, 1.4, 0.2] -->'
```

---

## 7. From model to API — `api.py`

A trained model sitting in a params file isn't useful to anyone else. The final
step of every lesson is to wrap it in a web API so any program (or person, via
`curl`) can send it a feature vector and get a prediction back. We use
**FastAPI**.

Every model in classical-ML-paper2code2api speaks the **same contract**, so
they're interchangeable (full spec in Part C):

```
POST /predict   (JSON: {"features": [ ... ]})  -> JSON <per-type result>
GET  /health                                   -> { status, model_loaded }
```

<!-- fill: show the heart of api.py — the /predict endpoint reading the JSON
     body and handing off to infer.predict(); note it's a THIN wrapper over the
     predict() from §6. Handle the two obvious failures gracefully: malformed
     input (HTTP 400) and model-not-trained (HTTP 503). Mention the startup hook
     that warm-loads params. -->

```python
@app.post("/predict")
async def predict_endpoint(payload: PredictRequest) -> dict:
    try:
        return predict(payload.features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))  # model not trained yet
```

Start the server:

```bash
uvicorn api:app --reload
```

Then open **http://127.0.0.1:8000/docs** for the interactive page, or use `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [<!-- fill --> ]}'
```

```jsonc
<!-- fill: an example response in the per-type contract shape from Part C. -->
```

![&lt;Algorithm&gt; single prediction detail](assets/prediction_detail.png)

*(Regenerate these figures any time with `python make_examples.py`.)*

---

## 8. In production

> TEMPLATE NOTE — **REQUIRED, NEW section for this course** (the CV template
> has no equivalent). Show the learner this algorithm doing *real work* in the
> world, not just on a toy dataset. Give **one or more concrete, named
> production examples** — real systems, products, or industries where this
> exact algorithm (or its direct descendants) ships today. Be specific and
> honest: if the pure algorithm is rarely deployed as-is but its *idea* lives on
> (e.g. the perceptron → online learning / neural nets), say exactly that. The
> curriculum draft has a vetted "In production" blurb for every lesson — start
> from it. 2–5 sentences or a short bulleted list; keep it real, not hype.

<!-- fill: e.g. "Logistic regression is the backbone of credit-default scoring
     at banks, click-through-rate prediction in ad systems, and clinical risk
     scores — still preferred wherever a decision must be explainable and
     audited." -->

---

## 9. The #1 gotcha: &lt;the mistake everyone makes&gt;

<!-- fill: the single most common way a beginner shoots themselves in the foot
     with THIS algorithm — stated as "you'd expect X, but you get Y, here's
     why, here's the fix." The CV analogue is the white-on-black inversion.
     Classical-ML examples: forgetting to SCALE/standardize features (kills kNN,
     SVM, k-means, PCA, GD convergence); the closed-set assumption; data
     leakage from scaling before the split; k-means sensitivity to
     initialization; extrapolation beyond the training range in regression.
     End with a > callout restating the rule. -->

> **Common mistake:** &lt;one-sentence restatement of the trap and its fix&gt;.

---

## 10. Exercises — try it yourself

<!-- fill: 4–6 exercises in ascending difficulty (easy → harder), each with a
     one-line *Lesson:* takeaway, exactly like the CV course. Good patterns:
     change a hyperparameter and observe; run it on your own / a second dataset;
     call the API and read the full probability/vote output; ablate a piece of
     the algorithm and watch it degrade; extend the API contract with a new
     route. Solutions are "an exercise in reading the code you already have." -->

1. **&lt;Easy&gt;.** &lt;task&gt;. *Lesson: &lt;takeaway&gt;.*
2. **&lt;Easy&gt;.** &lt;task&gt;. *Lesson: &lt;takeaway&gt;.*
3. **&lt;Medium&gt;.** Call your API, read the full &lt;probabilities/votes/path&gt; output. *Lesson: the model expresses graded uncertainty, not just one answer.*
4. **&lt;Medium/Harder&gt;.** &lt;ablate or extend&gt;. *Lesson: &lt;takeaway&gt;.*

> **Common mistake:** After editing `model.py`, you **must re-run
> `train.py`** — the shipped parameter artifact belongs to the old code and
> won't match a changed model. Delete `<model>.<pkl|npz>`, rerun `python
> train.py`, then test.

---

## 11. Recap & what's next

<!-- fill: bulleted recap of what the learner now understands (mirror the "What
     you'll learn" objectives, now in the past tense), then a one-line bridge to
     the next lesson that name-drops what it adds. House standard: recap + what's
     next, every lesson. -->

You just built &lt;algorithm&gt; from the ground up. Here's what you now understand:

- **&lt;objective 1, now mastered&gt;**
- **&lt;objective 2&gt;**
- **The full pipeline** — load → fit (&lt;metric&gt; on &lt;dataset&gt;) → infer → serve a `POST /predict` API — and the &lt;gotcha&gt; that trips people up.

**Next: [Lesson &lt;N+1&gt; · &lt;Name&gt;](../&lt;folder&gt;)** — &lt;one line on what it adds and why it follows from this one&gt;.

← Back to the [**course home**](../../README.md)

---

### Files in this lesson

| File | Purpose |
|---|---|
| `README.md` | This lesson |
| `model.py` | &lt;Algorithm&gt; from scratch (NumPy) — the reference implementation |
| `train.py` | Fit on &lt;dataset&gt;, save `<model>.<pkl|npz>` |
| `<model>.<pkl|npz>` | Pretrained parameters (ship with the repo) — `infer.py`/`api.py` work on clone; `train.py` overwrites it |
| `infer.py` | Preprocess + predict; usable standalone or as a library |
| `api.py` | FastAPI server exposing the shared `POST /predict` contract |
| `make_examples.py` | Generates the input/output example figures |
| `make_figures.py` | Generates the teaching diagrams (&lt;list this lesson's figures&gt;) |
| `requirements.txt` | Dependencies |
| `LICENSE-NOTES.md` | License status (safe to ship) |

<!-- ======================== END README SKELETON ======================== -->

---

## Part C · The shared API contract

Every model in the course speaks the same language so they're interchangeable —
the same principle as the CV course, but adapted for classical ML.

### [HOUSE DECISION — default] Tabular JSON, not multipart images

The CV course sends a **multipart image file** to `POST /predict`. Classical ML
is tabular, so this course's default is **JSON feature vectors in, JSON out**:

```
POST /predict   (JSON body: {"features": [f1, f2, ..., fn]})  -> JSON result
GET  /health                                                  -> { status, model_loaded }
```

- **`POST /predict`** accepts `{"features": [...]}` — one feature vector in the
  dataset's column order. (A lesson may document the expected feature names via
  an optional `GET /features` route.) It returns the per-type shape below.
- **`GET /health`** returns `{ "status": "ok", "model_loaded": <bool> }`, where
  `model_loaded` reflects whether the parameter artifact exists on disk.

> This is a **[HOUSE DECISION — default]**, marked so it can be swapped per
> lesson if a dataset genuinely calls for it (e.g. a text lesson might accept
> `{"text": "..."}`, or a batch route might accept a list of vectors). If you
> deviate, say so in the lesson's §7 the same way the course flags any
> contract change.

### Per-type return shapes

Match the lesson's task to one of these shapes (from the curriculum draft). The
type also determines what `model.py` exposes (`predict` / `predict_proba` /
`transform`).

| Task type | `POST /predict` returns | Notes |
|---|---|---|
| `classify` | `{ "label": <class>, "prob": <float> }` **or** `{ "label": <class>, "probs": { <class>: <float>, ... } }` | Use `prob` for the winning-class confidence; use `probs` when the model gives a full distribution (e.g. Naive Bayes, logistic on multiclass). A lesson may also add algorithm-specific extras — `neighbors` (kNN), `votes` (Random Forest), `path` (Decision Tree), `score` (SVM decision function), `confidence`/`margin` (AdaBoost). |
| `regress` | `{ "value": <float> }` | The predicted continuous target. |
| `cluster-assign` | `{ "cluster_id": <int>, "distance": <float> }` | Nearest centroid and the distance to it (k-Means). **Extends** the contract — flag it as a teaching moment in §7, as the CV course flags contract extensions. |
| `transform` | `{ "components": [<float>, ...] }` | The reduced/embedded vector (PCA). Also a contract **extension** — flag it. |

> TEMPLATE NOTE: `classify` and `regress` fit the mold cleanly (Lessons 1–11 in
> the draft). `cluster-assign` and `transform` (Lessons 12–13) deliberately
> extend it — that extension is itself a lesson, mirroring how the CV course
> treats `GET /classes` as "growing the contract per model."

---

## Deviations from the CV original (and why)

For the record, so future authors know what is intentional:

1. **Added §8 "In production"** — required by this course's brief; the CV
   template has no equivalent. Placed after the API section so the learner sees
   the algorithm working, *then* where it ships in the real world, before the
   gotcha/exercises/recap. This pushes the CV template's "gotcha", "exercises",
   and "recap" from §8/9/10 to §9/10/11.
2. **`POST /predict` is tabular JSON, not multipart image** — marked
   `[HOUSE DECISION — default]` in Part C, with per-type return shapes for
   classify/regress/cluster-assign/transform.
3. **Pure NumPy, not PyTorch** — `model.py` is from-scratch NumPy; the shipped
   artifact is `<model>.pkl`/`.npz`, not `<model>.pt`; `requirements.txt` drops
   `torch`/`torchvision`.
4. **Tabular sklearn datasets, not images** — `make_examples.py` /
   `make_figures.py` produce decision-boundary / scatter / line / tree / variance
   figures instead of image grids and feature maps.
5. **[paper] vs [textbook] tag** — the type tag in the metadata line drives §1's
   citation block (cite a paper, or note textbook roots) and is noted so §4
   ("equations to code") applies to both kinds.
6. **§3 heading is "step by step," not "layer by layer"** — classical algorithms
   aren't layered; otherwise the section's job (map the math onto `model.py`
   one-to-one, then show the fitted result) is unchanged.

## Open questions for a human decision

- **Feature scaling in the contract.** Many of these algorithms (kNN, SVM,
  k-means, PCA, GD) need standardized inputs. Decide whether `/predict` expects
  **raw** features (and the saved artifact carries the scaler) or
  **pre-scaled** features. Recommend: artifact owns the scaler so callers send
  raw values — but this needs a house ruling and should be stated once,
  course-wide.
- **`.pkl` vs `.npz` default.** Both are listed; pick a course-wide default
  (recommend `.npz` for array-only params, `.pkl` only where the artifact is a
  Python structure like a tree) so lessons are consistent.
- **Section-numbering vs the CV course.** Adding §8 shifts three section numbers
  relative to the CV lessons. That's fine within this course (internally
  consistent) but means cross-course "see §8" references won't line up — confirm
  that's acceptable, or renumber.
- **Batch prediction.** Whether to standardize a `POST /predict/batch` (list of
  vectors) route across lessons, or leave it as an optional per-lesson exercise.
