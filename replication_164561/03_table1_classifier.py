"""Table 1 of the paper: classifier accuracy in predicting treatment status
and demographic characteristics from the open-ended response text.

The paper uses BERT embeddings + a neural network. We use a TF-IDF +
Logistic Regression pipeline instead, which (a) requires no pre-trained model
downloads, (b) is fast and reproducible, and (c) is a reasonable benchmark
for bag-of-words text predictability. Absolute accuracies will differ
slightly from BERT, but the *ordering* (which dimensions are predictable from
the text) should be the same, which is the paper's qualitative claim.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import OUTDIR

df = pd.read_parquet(OUTDIR / "study1_cleaned.parquet")

median_age = df["age"].median()
median_income = df["income_midpoint"].median()

df["high_income"] = (df["income_midpoint"] >= median_income).astype(int)
df["old"] = (df["age"] >= median_age).astype(int)
df["after_crisis"] = df["Tr"].astype(int)
df["biden"] = (df["vote_trump"] == 0).astype(int)

DIMENSIONS = [
    ("After crisis", "after_crisis"),
    ("Biden voter", "biden"),
    ("College", "college"),
    ("High income", "high_income"),
    ("Male", "male"),
    ("White", "white"),
    ("Old", "old"),
]

SEED = 42
text = df["story_cleaned"].fillna("").astype(str).to_numpy(dtype=object)


def evaluate(y):
    y = np.asarray(y).astype(int)
    base_rate = max(np.mean(y), 1 - np.mean(y))
    X_tr, X_te, y_tr, y_te = train_test_split(
        text, y, test_size=0.20, random_state=SEED, stratify=y
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2),
            min_df=3, max_df=0.9, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    acc = float(np.mean(preds == y_te))
    # t-test vs base rate via binomial
    n = len(y_te)
    k = int(np.sum(preds == y_te))
    # two-sided test of accuracy == base rate, matching the paper's Table 1 note
    p_val = stats.binomtest(k, n, p=base_rate, alternative="two-sided").pvalue
    return acc, base_rate, p_val, n


rows = []
print(f"{'Dimension':<15} {'Accuracy':>9} {'Rate':>9} {'p-value':>10}")
print("-" * 48)
for name, col in DIMENSIONS:
    acc, rate, p, n = evaluate(df[col].values)
    rows.append({"dimension": name, "accuracy": acc, "rate": rate, "p_value": p, "n_test": n})
    pstr = f"{p:.3f}" if p >= 0.001 else "<0.001"
    print(f"{name:<15} {acc:>9.3f} {rate:>9.3f} {pstr:>10}")

pd.DataFrame(rows).to_csv(OUTDIR / "table1_classifier.csv", index=False)
print(f"\nSaved to {OUTDIR/'table1_classifier.csv'}")
