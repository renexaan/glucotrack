# GlucoTrack

**An end-to-end machine learning study on 100,000 clinical diabetes records.**
Built for 23CSE301 Machine Learning, B.Tech CSE Year III, Academic Year 2026–27.

`Python 3.13` · `scikit-learn 1.9.1` · `pandas 3.0.6` · Apache-2.0

---

## What this project found

Three results are worth reading before anything else.

**1 · The top classifiers' headline scores are an artefact of the data, not a triumph of modelling.**

The Decision Tree achieved **precision of exactly 1.0000** — zero false positives across
18,298 non-diabetic test records. That looked too good, so we checked. Probing the raw
file directly:

| Check | Result |
|---|---|
| `P(diabetic \| blood_glucose_level > 200)` | **100.00%** — all 3,277 records |
| `P(diabetic \| hbA1c_level > 6.6)` | **100.00%** — all 3,895 records |
| Highest glucose among *any* non-diabetic | exactly **200** |
| Highest HbA1c among *any* non-diabetic | exactly **6.6** |

Those are hard ceilings, not tendencies. The labels were assigned by rule above those
cut-offs. Any model able to express an axis-aligned threshold gets perfect precision for
free — which is also why tree-based models beat the linear and distance-based ones. All ten
algorithms were then compared: the six best, from a single tree to a neural network, span
just 0.0011 in weighted F1 — five very different model families hit the same ceiling. The genuine
difficulty lives in the **86,042 patients below both thresholds, of whom 2.79% are still
diabetic**, and that is where every model's false negatives come from.

**2 · Accuracy is actively misleading on this dataset — and one model proves it.**

The classes split 91.5% / 8.5%, a ratio of **10.76 : 1**. A model that always predicts
"non-diabetic" scores 91.5% accuracy and identifies nobody. Gaussian Naive Bayes scores
**67.50% accuracy — below that baseline — while catching more diabetic patients than any
other model** (0.9594 recall, 69 missed out of 1,700).

**3 · Which model is "best" depends entirely on what an error costs.**

Four of the five classifiers are false-negative dominated, missing 33–42% of diabetic
patients while raising almost no false alarms. Naive Bayes is the exact inverse: 4.1%
missed, 35.1% false alarms. The model ranked **last** by weighted F1 misses the fewest
patients. In a screening context — where a missed diagnosis costs far more than a
follow-up test — the ranking metric and the clinically useful metric disagree.

---

## Tracks

| Track | Dataset | Target | Status | Review |
|---|---|---|---|---|
| **Classification** | Diabetes Clinical (100,000 × 16) | `diabetes` (binary) | Parts A and B complete (10 algorithms) | Part A → R1, Part B → R2 |
| **Regression** | Diabetes Prediction | `blood_glucose_level` | 10 algorithms complete | Review 1 |
| **Clustering** | Diabetes Prediction (100,000 × 9) | unsupervised | K-Means + Agglomerative complete | Review 2 |

---

## Classification results — Part A (five classical models)

Five algorithms, 80/20 stratified split, `random_state=42`, all scored on the same
held-out test set of 19,998 records (1,700 diabetic).

| Rank | Model | Accuracy | F1 (weighted) | ROC-AUC | Recall (diabetic) | Precision (diabetic) |
|---|---|---|---|---|---|---|
| 1 | **Decision Tree** | 0.9721 | **0.9696** | **0.9721** | 0.6718 | **1.0000** |
| 2 | Support Vector Machine | 0.9618 | 0.9585 | 0.9548 | 0.6188 | 0.9007 |
| 3 | Logistic Regression | 0.9612 | 0.9585 | 0.9579 | 0.6394 | 0.8696 |
| 4 | K-Nearest Neighbors | 0.9535 | 0.9499 | 0.8564 | 0.5829 | 0.8177 |
| 5 | Gaussian Naive Bayes | 0.6750 | 0.7467 | 0.9121 | **0.9594** | 0.2023 |

*Majority-class baseline accuracy: 0.9150*

**Error profile** — where the models actually differ:

| Model | Missed diabetics (FN) | False alarms (FP) | FN rate |
|---|---|---|---|
| Decision Tree | 558 | 0 | 32.8% |
| Logistic Regression | 613 | 163 | 36.1% |
| Support Vector Machine | 648 | 116 | 38.1% |
| K-Nearest Neighbors | 709 | 221 | 41.7% |
| Gaussian Naive Bayes | **69** | 6,430 | **4.1%** |

**Selected hyperparameters** (GridSearchCV, 3-fold stratified CV, scored on diabetic-class F1):

| Model | Configuration |
|---|---|
| Logistic Regression | `lbfgs`, `C=1.0` — not tuned |
| KNN | `k=3`, `weights=uniform`, Euclidean |
| Gaussian NB | no hyperparameters |
| Decision Tree | `entropy`, `max_depth=8`, `min_samples_leaf=5` → 25 leaves |
| SVC | `linear`, `C=10.0` → 2,081 support vectors (10.4%) |

**Cross-validation** (5-fold stratified, top two models) confirms the ranking is stable:

| Model | CV F1 (weighted) | Test F1 | Difference |
|---|---|---|---|
| Decision Tree | 0.9692 ± 0.0009 | 0.9696 | +0.0004 |
| Support Vector Machine | 0.9575 ± 0.0031 | 0.9585 | +0.0010 |

---

## Classification results — Part B and the full ten-algorithm comparison

Part B adds Random Forest, AdaBoost, Gradient Boosting, Bagging (Decision Tree base) and
an MLP on the same data, split and preprocessing. Each was fitted with library defaults,
then tuned with `GridSearchCV` (3-fold, scored on diabetic-class F1).

| Rank | Model | Part | Accuracy | F1 (weighted) | ROC-AUC | Recall (diabetic) | Precision (diabetic) |
|---|---|---|---|---|---|---|---|
| 1 | **Gradient Boosting** | B | 0.9721 | **0.9698** | **0.9772** | 0.6859 | 0.9798 |
| 2 | Random Forest | B | 0.9721 | 0.9697 | 0.9709 | 0.6759 | 0.9948 |
| 3 | MLP Neural Network | B | 0.9721 | 0.9696 | 0.9751 | 0.6753 | 0.9948 |
| 4 | AdaBoost | B | 0.9721 | 0.9696 | 0.9716 | 0.6724 | 0.9991 |
| 5 | Decision Tree | A | 0.9721 | 0.9696 | 0.9721 | 0.6718 | 1.0000 |
| 6 | Bagging | B | 0.9709 | 0.9687 | 0.9597 | 0.6853 | 0.9620 |
| 7 | Support Vector Machine | A | 0.9618 | 0.9585 | 0.9548 | 0.6188 | 0.9007 |
| 8 | Logistic Regression | A | 0.9612 | 0.9585 | 0.9579 | 0.6394 | 0.8696 |
| 9 | K-Nearest Neighbors | A | 0.9535 | 0.9499 | 0.8564 | 0.5829 | 0.8177 |
| 10 | Gaussian Naive Bayes | A | 0.6750 | 0.7467 | 0.9121 | **0.9594** | 0.2023 |

**Final model: Gradient Boosting** (`learning_rate=0.05`, `n_estimators=200`, `max_depth=5`).
It ranks first on weighted F1, diabetic F1 and ROC-AUC, and its 5-fold cross-validated
ROC-AUC is 0.9788 ± 0.0011 (Random Forest: 0.9721). The F1 margin over Random Forest is
inside one cross-validation standard deviation, so the decisive evidence is ranking
quality and recall, not a large performance gap.

**Tuning bought very little.** Diabetic-F1 changes from tuning: Bagging +0.0059, MLP
+0.0013, Random Forest +0.0005, AdaBoost +0.0001, Gradient Boosting −0.0010. Models are
already at the information ceiling of the data, so the search space barely matters.

---

## Clustering results

K-Means and Agglomerative clustering on the Diabetes Prediction dataset (label excluded
from every fitting step; used only for post-hoc interpretation). K-Means: k = 4, chosen
from the elbow curve together with silhouette, Davies-Bouldin and Calinski-Harabasz
across k = 2–10.

| Algorithm | Rows | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|
| K-Means (k=4) | 100,000 | 0.1842 | 1.5096 | 21,981 |
| K-Means (k=4), same subset | 2,000 | 0.1855 | 1.5152 | 452.6 |
| Agglomerative, Ward | 2,000 | 0.1593 | 1.6650 | 410.2 |
| Agglomerative, Complete | 2,000 | 0.3036 | 1.3300 | 201.3 |
| Agglomerative, Average | 2,000 | 0.3994 | 0.9886 | 298.8 |
| Agglomerative, Single | 2,000 | 0.4071 | 1.1163 | 297.6 |

Complete, Average and Single linkage put 88–89% of records in one giant cluster
(chaining), so their higher silhouette is an artefact of a few tiny outlier groups. On the
same 2,000 rows, K-Means beats Ward on all three metrics.

The four K-Means clusters are patient segments — young low-risk, middle-aged,
hypertensive, cardiac — and the diabetic share rises from 0.5% to 32.1% across them
although the label was never used. Cluster structure is weak overall (silhouette < 0.25
for balanced solutions): the features are mostly continuous with no natural gaps.

---

## Regression results

Ten algorithms predicting `blood_glucose_level`.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.1825 | 36.94 | 30.73 |
| Ridge | 0.1825 | 36.94 | 30.73 |
| **Lasso** | **0.1825** | **36.94** | 30.73 |
| **ElasticNet** | **0.1825** | **36.94** | **30.73** |
| Polynomial (degree 2) | 0.1811 | 36.97 | 30.75 |
| Gradient Boosting | 0.1810 | 36.97 | 30.73 |
| Decision Tree (tuned) | 0.1823 | 36.94 | 30.73 |
| Random Forest (tuned) | 0.1812 | 36.97 | 30.73 |
| SVR | 0.0936 | 38.90 | 30.01 |
| KNN Regressor | 0.0082 | 40.69 | 33.21 |

**Two observations.** First, R² of roughly 0.18 across every linear model is a real
finding rather than a failure — blood glucose genuinely is not well predicted by
demographics and comorbidities alone, and nine different algorithms converging on the same
ceiling is strong evidence of that. Second, **tuning rescued the Decision Tree from
R² = −0.6536 to 0.1823**, an improvement of 0.84. An untuned tree performed *worse than
predicting the mean*; constraining its depth fixed it entirely.

Cross-validated R² for the top two models (Lasso and ElasticNet): **0.1791 ± 0.0036**.

---

## How the pipeline avoids data leakage

This is the part most likely to be checked, so it is made explicit and then verified.

Operations are ordered by whether they **learn a parameter from the data**:

| Step | Operation | When | Safe because |
|---|---|---|---|
| 3.1 | Reconstruct `race` from 5 indicator columns | before split | deterministic reshape |
| 3.2 | Drop 14 exact duplicates | before split | row identity only |
| 3.4 | *Measure* IQR outliers | before split | reporting only; bounds refitted later |
| 3.5 | Engineer 3 features | before split | strictly row-wise |
| 3.6 | `train_test_split(stratify=y)` | — | the split itself |
| 3.7 | Impute, winsorise, encode, scale | **after split** | `.fit()` sees `X_train` only |

Section 3.7 closes with assertions that **prove** the discipline held:

```
train_mean   train_std    test_mean    test_std
     -0.0          1.0    -0.004833    0.999551   ← hbA1c_level
     -0.0          1.0    -0.008899    1.006024   ← blood_glucose_level
```

Training columns sit at exactly mean 0. Test columns **do not** — they were transformed
using the *training* mean and standard deviation, not their own. Had the scaler been
fitted on the full dataset, both blocks would read exactly zero, and the assertion fails
deliberately if that happens.

---

## Notable methodology decisions

**`location` is frequency-encoded, not one-hot.** With 54 levels, one-hot would grow the
design matrix from 24 to roughly 75 sparse columns carrying almost no class-rate
variation. That specifically damages KNN — near-orthogonal sparse dimensions dilute the
informative distance from HbA1c and glucose — and raises SVC cost. Target encoding was
rejected outright: it builds the feature from the label.

**Outliers were winsorised, not deleted.** IQR flagged 1,315 HbA1c and 2,038 glucose
outliers — and **100% of both groups are diabetic**, against an 8.50% baseline. Those
3,353 rows are 39% of every diabetic case in the dataset. They are extreme *because* the
patient is diabetic. Deleting them would have destroyed nearly 40% of the positive class.

**Three engineered features**, each validated against the target:

| Feature | Rationale | Evidence it works |
|---|---|---|
| `bmi_category` | Risk steps up at obesity rather than rising linearly; a linear model given raw BMI can only fit one slope | Prevalence 0.75% → 22.16% across WHO bands |
| `age_risk_interaction` | Linear models cannot represent a product of two inputs unless given it | r = **+0.4106**, higher than either parent column |
| `metabolic_risk_score` | Compresses five weak flags into one ordinal; helps Naive Bayes, which cannot combine co-occurring evidence | Prevalence 0.00% → 68.93% across scores 0–5 |

`metabolic_risk_score` uses **fixed clinical thresholds from the literature**, not
quantiles computed from this dataset — which is what keeps it leakage-free.

**Grid searches score on diabetic-class F1, not accuracy.** Under a 10.76 : 1 imbalance, a
grid optimised for accuracy would happily select a model that never predicts a diabetic
patient.

---

## Repository structure

```
glucotrack/
├── data/
│   ├── diabetes_dataset.csv              classification, 100,000 × 16
│   ├── diabetes_prediction_dataset.csv   regression
│   └── README.md                         column reference
├── notebooks/
│   ├── classification.ipynb              10 algorithms (Parts A and B) · runs in ~4 min
│   ├── clustering.ipynb                  K-Means + Agglomerative · PCA and t-SNE
│   └── regression.ipynb                  10 algorithms
├── models/                               serialised estimators (git-ignored)
├── requirements.txt                      pinned versions
└── README.md
```

Datasets are committed directly, so the notebooks run on a fresh clone with no download
step. Trained models are git-ignored because they rebuild from the notebooks.

---

## Setup

```bash
git clone https://github.com/renexaan/glucotrack.git
cd glucotrack

python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
jupyter lab notebooks/classification.ipynb
```

Then **Kernel → Restart Kernel and Run All Cells**. Every notebook runs top to bottom
with no manual intervention (classification takes about 4 minutes, the others about one).

To verify it end to end without opening Jupyter:

```bash
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3000 notebooks/classification.ipynb
```

**Python 3.13, not 3.14** — scikit-learn wheel coverage for 3.14 is still incomplete and
would force a source build.

---

## Reproducibility

`random_state = 42` is applied to the train/test split, every subsample, every
cross-validation fold, and every estimator that accepts a seed. The imputers, one-hot
encoder, frequency map, IQR bounds and `StandardScaler` are each fitted on `X_train`
exclusively.

---

## Limitations

Stated plainly, because they matter more than the headline numbers.

- **The classification dataset is substantially synthetic.** No non-diabetic record in
  100,000 rows exceeds glucose 200 or HbA1c 6.6. Each glycaemic marker takes only 18
  distinct values, and 25,495 records share a BMI of exactly 27.32. Treat 0.97 weighted F1
  as a property of this dataset, not an estimate of clinical performance.
- **Clustering structure is weak.** Silhouette is below 0.25 for every balanced solution,
  and the hierarchical analysis uses a 2,000-row subset (500 rows for the dendrogram)
  because linkage is quadratic in memory. K-Means is therefore compared with Ward on the
  same subset for fairness.
- **Part B tuning grids were small** (9–12 configurations) and the gains negligible, so a
  larger search is unlikely to change the ranking.
- **KNN and SVC were tuned and refit on stratified subsamples** (12k/40k and 6k/20k
  respectively), because kernel-SVM training scales between O(n²) and O(n³). Their scores
  are conservative lower bounds and the comparison is not strictly like-for-like.
- **No decision-threshold tuning.** Every model uses the default 0.5 cut-off. Since false
  negatives dominate, lowering it would trade precision for recall and would very likely
  produce a better screening model. This is the highest-value improvement available.
- **No class-imbalance correction** — no `class_weight="balanced"`, no resampling.
  Imbalance was handled through stratification and metric choice alone.
- **Logistic Regression was left untuned**, which was a scope decision rather than a
  principled one. Naive Bayes has no hyperparameter worth searching.
- **Target leakage is intrinsic to the classification problem.** HbA1c and blood glucose
  are the criteria by which diabetes is *diagnosed*, so the model reproduces a diagnostic
  rule rather than predicting future risk. This is distinct from train/test leakage, which
  the pipeline rules out: the pipeline is clean, but the features encode the answer.

## Next steps

1. Tune the decision threshold against a stated cost ratio between a missed diagnosis and
   a false alarm, chosen on a validation split — never the test set.
2. Apply `class_weight="balanced"` to the models that support it.
3. Re-run KNN and SVC on the full training split to quantify the subsampling cost.
4. Build a leakage-free early-warning variant excluding both glycaemic markers, removing
   the intrinsic leakage and the synthetic labelling rule together. Expect much worse
   metrics — and a far more honest model.

---

## Declaration of AI assistance

Per §7.5 of the course guidelines, this project used **Claude (Anthropic)** as an AI
assistant.

**Used for:** code scaffolding — pipeline structure, plotting boilerplate, the evaluation
harness — along with repository setup and documentation.

**Not used for:** the analysis and interpretation. Observations, conclusions and the
reasoning behind each modelling decision are the team's own.

No code was copied from StackOverflow, blogs or other repositories.

---

## Course context

| | |
|---|---|
| **Module** | 23CSE301 Machine Learning |
| **Programme** | B.Tech Computer Science and Engineering, Year III |
| **Assessment** | Two reviews, 25 marks each |
| **Review 1** | Full Regression track + Classification Part A |
| **Review 2** | Classification Part B + full Clustering track |
| **Licence** | Apache-2.0 |
