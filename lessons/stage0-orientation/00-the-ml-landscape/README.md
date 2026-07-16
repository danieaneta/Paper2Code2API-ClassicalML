# Lesson 0 · The Machine Learning Landscape — What ML Is, and the Map of It

> Stage · Orientation · No code — a ~10-minute read · Part of [classical-ML-paper2code2api](../../../README.md)

Before you build anything, you need the map. This is the one lesson in the course with no model, no `model.py`, and no `/predict` endpoint — just the mental picture that makes every lesson after it click. Spend ten minutes here and, when you build linear regression in Lesson 1, you'll be able to say exactly what *kind* of machine learning it is and where it sits in the field. No math dread, no prerequisites beyond curiosity — let's draw the map.

### What you'll learn

By the end of this lesson you'll be able to:

- **Say what machine learning actually is** — learning rules from data instead of hand-writing them — and why that shift matters.
- **Name the big split**: supervised learning (learning from labeled examples) vs. unsupervised learning (finding structure with no labels).
- **Place the four core tasks** on the map — regression, classification, clustering, and dimensionality reduction — and know which side of the split each lives on.
- **Speak the vocabulary** every later lesson assumes: features, labels, training vs. test data, model and parameters, generalization vs. overfitting.
- **Locate any of the 13 build lessons** on that map — so for every model you build, you can say "this is supervised regression" or "this is unsupervised clustering."

**Prerequisites:** none but curiosity and basic Python literacy. You won't even run code this lesson — that starts in Lesson 1.

---

## 1. What machine learning actually is

Imagine you're asked to build a spam filter. The old-fashioned way is to sit down and **write the rules yourself**: if the subject line contains "FREE MONEY," flag it; if it has five exclamation marks, flag it; if it mentions a Nigerian prince, flag it. You'd end up with a giant tangle of hand-coded `if` statements — and spammers would route around it the moment you shipped it. Change one word ("FR€E") and your rule misses. You'd be patching that list forever.

Machine learning flips the arrangement. Instead of you writing the rules, you **hand the machine a pile of examples and let it work out the rules itself.** Concretely: you collect a few thousand emails that real people have already sorted into "spam" and "not spam," and you feed those labeled examples to a *learning algorithm*. The algorithm studies which words, patterns, and combinations tend to show up in spam versus normal mail, and it produces a **model** — a set of learned numbers — that can score a brand-new email it has never seen.

That's the whole idea, and it's worth stating plainly:

> **Note — the core reframe.** Classical programming is *rules + data → answers*. You write the rules. Machine learning is *data + answers → rules*. You supply examples with their answers, and the machine learns the rules. Most of this course is a different, principled way of doing that — and its close cousin, *unsupervised* learning, does the same trick with **no answers at all** (*data → structure*), which you'll meet in §3.

Why does this matter? Because for a huge class of problems, **nobody actually knows the rules well enough to write them down.** What exact pixel pattern means "handwritten 7"? What precise combination of neighborhood facts sets a house's price? You can't enumerate those by hand — but you *can* gather examples, and let the data reveal the pattern. That single move, learning the rule from examples rather than hand-coding it, is what powers everything from your email spam folder to the recommendations on your favorite streaming service. The rest of this lesson is a map of the different *shapes* that move comes in.

---

## 2. Supervised learning: learning from labeled examples

The first and biggest region on the map is **supervised learning**. The name sounds fancy but the idea is simple: your training data comes **with the answers already attached**. Each example is a pair — some inputs, and the correct output for those inputs. The model's whole job is to learn the input → output mapping so well that it can produce the right answer for *new* inputs it never saw in training.

The "answer" attached to each example is called a **label** (you'll also hear **target**). In the spam example, the inputs are the email's words and the label is the human-assigned "spam" or "not spam." Because a person had to supply those labels — to *supervise* the learning — we call it supervised learning. This is where most of this course lives: eleven of the thirteen build lessons are supervised.

Supervised learning splits cleanly into two flavors, decided by *what kind of answer you're predicting*:

- **Regression — predict a number.** The label is a continuous quantity. *How much will this house sell for? How many units will we ship next month? What will the temperature be tomorrow?* The answer is a value on a scale, and being "close" counts. **Lesson 1, linear regression, is exactly this** — predicting a California house's value from eight facts about its neighborhood.

- **Classification — predict a category.** The label is one of a fixed set of classes. *Is this email spam or not? Which digit, 0–9, is in this image? Is this tumor benign or malignant?* The answer is a bucket, not a number on a scale. Most of the classifiers in Stages 2 and 3 do this.

The distinction is just "number vs. category," but it's the first thing to identify about any supervised problem, because it changes which models and which scoring methods make sense. A common beginner slip is to reach for the wrong one — trying to predict a house price with a classifier (forcing a smooth number into rigid buckets), or predicting spam-vs-not with a regression (getting back "0.63 of a spam," which means nothing). Match the tool to the shape of the answer.

Look at the **left panel** of the figure below. Every point is one example, plotted by two of its features. The *color and shape carry the known label*: blue circles are class A, red squares are class B, green triangles are class C. Because we already know each point's class, a supervised model can learn "points up here tend to be blue, points down there tend to be red" and then label a new point by where it falls. The answers were given; the model learns to reproduce them.

---

## 3. Unsupervised learning: finding structure without labels

Now cover up the answers. **Unsupervised learning** is what you do when your data has **no labels at all** — just the raw inputs, with nobody telling you the "right" output. There's nothing to reproduce, because there's no answer key. Instead, the model's job is to **find structure that's already hiding in the data**.

Look at the **right panel** of the same figure. It's the *exact same 300 points* as the left panel — but now they're all one neutral grey, with no legend. The labels haven't been deleted from reality; they've just been hidden from the algorithm. And yet your eye still sees roughly three clumps. That's the unsupervised task in a nutshell: *the groups are there; go find them without being told.*

Unsupervised learning has two core tasks in this course:

- **Clustering — group similar points.** Discover which examples naturally belong together, without any predefined categories. *Which of our customers behave alike, so we can market to them as segments?* You don't tell it the groups; it proposes them. **Lesson 12, k-means, does exactly this** — and it's essentially "find the clumps in that grey scatter."

- **Dimensionality reduction — compress many features into few.** Real data often has dozens or hundreds of features, most of them overlapping or redundant. Dimensionality reduction squeezes that down to a handful of new features that keep most of the information — great for visualizing high-dimensional data in 2-D, or for speeding up other models. **Lesson 13, PCA, is this** — taking 64-dimensional digit images down to a 2-D picture you can actually see.

> **Tip — the one-question test.** To tell which half of the map a problem is on, ask: *does my training data come with the answers?* If yes → supervised (and then: is the answer a number or a category?). If no, and you're hunting for structure → unsupervised. That single question places almost everything.

![Two side-by-side scatter plots of the same 300-point, three-cluster cloud. Left panel, titled "Supervised: labeled examples," colors each point by its known class — blue circles for class A, red squares for class B, green triangles for class C — with a legend headed "known label." Right panel, titled "Unsupervised: no labels," shows the identical points all in one neutral grey with no legend, the labels hidden so the groups must be found from the data itself.](assets/supervised_vs_unsupervised.png)

*The same data, twice. **Left (supervised):** every point arrives with its answer — class A (blue circles), B (red squares), C (green triangles) — and the model learns to reproduce those labels for new points. **Right (unsupervised):** identical points, labels hidden, all one grey; the algorithm has to discover the groups on its own. Supervised learning reproduces answers it was given; unsupervised learning finds structure nobody handed it.*

---

## 4. Two you'll hear about (but won't build here)

The map has two more regions. You should know their names so the picture feels complete — but this course deliberately doesn't build them, to stay lean and focused.

**Reinforcement learning** is learning from **reward and trial-and-error**. There's no fixed answer key; instead an *agent* takes actions in some environment, receives rewards or penalties, and gradually learns a strategy that earns the most reward over time. This is the flavor behind game-playing systems that master chess or Go, and behind robots learning to walk. It's a genuinely different setup — the data isn't a static labeled pile, it's generated by the agent's own actions — which is why it needs its own toolkit and sits outside this course.

**Semi-supervised learning** is the practical middle ground: **a little labeled data plus a lot of unlabeled data.** Labeling is expensive (someone has to do it by hand), while raw unlabeled data is often cheap and abundant. Semi-supervised methods squeeze extra signal out of the big unlabeled pile to improve a model trained on the small labeled one. It's a blend of the two halves you just met rather than a separate idea, so we name it and move on.

> **Note — scope.** This course covers **supervised** and **unsupervised** learning in depth. Reinforcement and semi-supervised learning are named here for a complete map, but we don't build them — keeping the course tight and the through-line clear.

---

## 5. The words every lesson assumes

A handful of terms show up in every lesson from here on. Here they are, in plain language, so nothing later trips you up. Read this as a quick glossary — you'll meet each one again in context.

- **Features** — the **inputs**: the measured facts you feed the model. For a house, its features might be median income, house age, and number of rooms. Each example is a list of feature values.

- **Labels / targets** — the **answers**, in supervised learning only. The known output attached to each training example: the house's actual price, or the email's true "spam / not spam." Unsupervised learning has no labels — that's exactly what makes it unsupervised.

- **Training data vs. test data** — you split your examples into two piles. The model **learns from the training pile**, and you **judge it on the test pile** — examples it never saw while learning. Why bother? Because scoring a model on the same data it studied is like grading a student on the exact questions they memorized: it tells you nothing about whether they actually *understand*. The test set is your honest measure.

- **Model and parameters** — the **model** is the thing that makes predictions; its **parameters** are the numbers it learns during training. "Training" or "fitting" a model *is* the process of finding good parameter values from the data. Linear regression, for instance, ends up as just nine numbers — that's the entire trained model.

- **Generalization vs. overfitting** — **generalization** is the goal: doing well on **new, unseen data**, not just the training examples. **Overfitting** is the failure mode where a model **memorizes** the training data — including its noise and quirks — and then flops on anything new. It's the student who memorized the practice test word-for-word and is lost the moment the real questions differ. Everything we do to evaluate honestly (that train/test split, and later cross-validation) exists to catch overfitting and measure real generalization.

> **Note — why the vocabulary is front-loaded.** These five terms recur in *every* build lesson. Getting them straight now means later lessons can move fast and talk about the interesting model-specific stuff without re-explaining the basics each time.

---

## 6. The map of this course

Here's the whole landscape on one page — machine learning's family tree, with the branches this course teaches drawn solid and the two we only mention drawn faint:

![A taxonomy tree titled "The machine learning landscape." A dark "Machine Learning" box at the top splits into two solid branches: a blue "Supervised (labeled data)" box branching to "Regression — predict a number" and "Classification — predict a category," and a green "Unsupervised (no labels)" box branching to "Clustering — group points" and "Dimensionality Reduction — compress features." Below, two greyed, dashed boxes labeled "Reinforcement Learning" and "Semi-supervised" are marked "(mentioned, not built in this course)."](assets/ml_taxonomy.png)

*The four solid leaves — regression, classification, clustering, dimensionality reduction — are precisely the tasks the 13 build lessons cover. Reinforcement learning and semi-supervised learning are greyed and dashed: on the map, out of scope.*

Now overlay that map onto the actual course. Every lesson sits on one *side* of the map, and every model you build lands on one of the four leaves (a couple of lessons — Gradient Descent and Regularization — are shared *tooling* for the supervised side rather than a task-leaf of their own):

| Stage | Lessons | Half of the map | Task on the map |
|---|---|---|---|
| **Stage 1 · Supervised Foundations** | 1–4 (Linear Regression, Gradient Descent, Logistic Regression, Regularization + CV) | Supervised | Regression, then the jump to Classification, plus the generalization toolkit |
| **Stage 2 · Classical Classifiers** | 5–8 (Perceptron, k-NN, Naive Bayes, SVM) | Supervised | Classification — four different paradigms |
| **Stage 3 · Trees & Ensembles** | 9–11 (Decision Tree, Random Forest, AdaBoost) | Supervised | Classification — trees and ensembles of them |
| **Stage 4 · Unsupervised Learning** | 12–13 (k-Means, PCA) | Unsupervised | Clustering, then Dimensionality Reduction |

Read top to bottom, the course walks the map left branch first: **Lessons 1–11 are all supervised** — starting with regression (predict a number), pivoting to classification (predict a category) in Lesson 3, and then piling up ever-stronger classifiers through Stages 2 and 3. Then **Lessons 12–13 cross to the unsupervised branch**, dropping the labels entirely to find structure.

The payoff of this lesson: from here on, every time you open a new lesson, you'll be able to point at this tree and say exactly where you are. Lesson 1? Supervised, regression, far-left leaf. Lesson 12? Unsupervised, clustering. The map is yours now.

> **Check yourself.** Which region of the map is each of these? (Answers below.)
> 1. Predicting tomorrow's temperature in degrees from today's weather readings.
> 2. Sorting a pile of news articles into groups by topic, with no topics defined in advance.
> 3. Deciding whether a credit-card transaction is fraudulent or legitimate.
>
> *Answers: (1) supervised — regression (a number); (2) unsupervised — clustering (no labels, find the groups); (3) supervised — classification (a category). If you got the split right, the map is doing its job.*

---

## 7. Recap & what's next

You didn't write a line of code — and that was the point. You now hold the frame that makes every model in this course make sense. Here's what you can now do:

- **Say what ML is** — learning rules from labeled or unlabeled examples instead of hand-coding them.
- **Name the big split** — supervised (data comes with answers) vs. unsupervised (no answers; find structure), with a nod to reinforcement and semi-supervised learning as the out-of-scope regions.
- **Place the four core tasks** — regression and classification on the supervised side, clustering and dimensionality reduction on the unsupervised side.
- **Speak the shared vocabulary** — features, labels, training vs. test data, model and parameters, generalization vs. overfitting.
- **Locate any lesson on the map** — Lessons 1–11 supervised, Lessons 12–13 unsupervised.

Now let's build. **Next: [Lesson 1 · Linear Regression](../../stage1-supervised-foundations/01-linear-regression)** — your first supervised model. It predicts a *number* (a house's value) from *labeled* examples, which puts it on the far-left leaf of the map you just learned: supervised, regression. You'll build it from scratch, train it on real California housing data, and serve it behind a `POST /predict` API. The map told you *what* it is; the next lesson shows you *how* it works.

← Back to the [**course home**](../../../README.md)

---

### Files in this lesson

This is the one no-code lesson — there's no `model.py`, `train.py`, `infer.py`, or `api.py` here, and no `/predict` endpoint. Just the reading and the two diagrams.

| File | Purpose |
|---|---|
| `README.md` | This primer (the lesson) |
| `make_figures.py` | Generates the two teaching diagrams |
| `assets/supervised_vs_unsupervised.png` | The labeled-vs-unlabeled scatter (§2–§3) |
| `assets/ml_taxonomy.png` | The map of the machine learning landscape (§6) |
| `requirements.txt` | Dependencies (figures only — no model to run) |
