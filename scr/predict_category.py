import re
import joblib
import pandas as pd

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

def main():
    # Load trained pipeline
    pipe = joblib.load("models/product_category_model.pkl")

    print("Type product titles (or 'exit' to quit).")
    while True:
        try:
            title = input("> ").strip()
        except EOFError:
            break
        if not title or title.lower() == "exit":
            break

        df = pd.DataFrame({"title": [title]})
        pred = pipe.predict(df)[0]
        print(f"Predicted category: {pred}\n" + "-"*50)

if __name__ == "__main__":
    main()



