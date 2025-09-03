import os
import re
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Tiny numeric features (optional)
def title_stats(X):
    s = pd.Series(X).fillna("")
    def digit_ratio(t): return (sum(ch.isdigit() for ch in t)/len(t)) if t else 0.0
    def upper_ratio(t): return (sum(ch.isupper() for ch in t)/len(t)) if t else 0.0
    def word_count(t): return len(t.split())
    def max_word_len(t): return max((len(w) for w in t.split()), default=0)
    def has_storage_unit(t): return int(bool(re.search(r"\b(\d+\s?(gb|tb|mb))\b", t, re.I)))
    feats = pd.DataFrame({
        "len_chars": s.map(len).astype(float),
        "word_count": s.map(word_count).astype(float),
        "digit_ratio": s.map(digit_ratio).astype(float),
        "upper_ratio": s.map(upper_ratio).astype(float),
        "max_word_len": s.map(max_word_len).astype(float),
        "has_storage_unit": s.map(has_storage_unit).astype(float),
    })
    return feats.values


def build_preprocessor():
    tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                                 min_df=3, max_features=30000, strip_accents="unicode")
    tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3,4),
                                 min_df=5, max_features=30000)
    stats = Pipeline([
        ("fx", FunctionTransformer(title_stats, validate=False)),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    return ColumnTransformer([
        ("w", tfidf_word, "title"),
        ("c", tfidf_char, "title"),
        ("s", stats, "title"),
    ], remainder="drop")


def get_model(name: str):
    name = name.lower()
    if name == "linear_svc":   return LinearSVC()
    if name == "logreg":       return LogisticRegression(max_iter=1000)
    if name == "nb":           return MultinomialNB()
    if name == "rf":           return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "dt":           return DecisionTreeClassifier(random_state=42)
    raise ValueError(f"Unknown --algo '{name}'. Use one of: linear_svc, logreg, nb, rf, dt")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        'product ID': 'product_id',
        'Product Title': 'title',
        'Merchant ID': 'merchant_id',
        ' Category Label': 'category',
        '_Product Code': 'product_code',
        'Number_of_Views': 'views',
        'Merchant Rating': 'merchant_rating',
        ' Listing Date  ': 'listing_date',
    })


def normalize_labels(s: pd.Series) -> pd.Series:
    norm = {"CPU":"CPUs", "Mobile Phone":"Mobile Phones", "fridge":"Fridges"}
    return s.astype(str).str.strip().map(lambda x: norm.get(x, x))


def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    # 1 Load & clean
    df = pd.read_csv(args.csv)
    df = normalize_columns(df).dropna(subset=["title", "category"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = normalize_labels(df["category"])

    X = df[["title"]]
    y = df["category"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2 Pipeline
    preproc = build_preprocessor()
    clf = get_model(args.algo)
    pipe = Pipeline([("prep", preproc), ("clf", clf)])

    # 3 Train & validate
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_valid)
    acc = accuracy_score(y_valid, pred)
    rep = classification_report(y_valid, pred, digits=3)

    # 4 Save report
    report_path = os.path.join(args.report_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Algo: {args.algo}\nAccuracy: {acc:.4f}\n\n")
        f.write(rep)
    print(f"Accuracy={acc:.4f}  |  Saved report -> {report_path}")

    # 5 Confusion matrix
    classes = sorted(y.unique().tolist())
    cm = confusion_matrix(y_valid, pred, labels=classes)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, aspect="auto")
    plt.title(f"Confusion Matrix — {args.algo}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.colorbar(); plt.tight_layout()
    cm_path = os.path.join(args.report_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix -> {cm_path}")

    # 6 Retrain on full data & export .pkl
    final_pipe = Pipeline([("prep", preproc), ("clf", get_model(args.algo))])
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, args.out)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/products.csv", help="Path to products.csv")
    p.add_argument("--out", default="models/product_category_model.pkl", help="Output .pkl path")
    p.add_argument("--report_dir", default="reports", help="Dir for reports/plots")
    p.add_argument("--algo", default="linear_svc",
                   help="Classifier: linear_svc | logreg | nb | rf | dt")
    main(p.parse_args())

