# Product Category Classifier (Title → Category)

End-to-end ML project that predicts a product **category** from its **title**.  
The solution automates product onboarding for an e-commerce platform, reducing manual work and classification errors.

---

## TL;DR

- **Data**: `data/products.csv` (~30-35k rows)  
- **Target**: `category` • **Input**: `title`  
- **Preprocessing**: TF-IDF (word 1–2; char 3–4) + **6 numeric title features**  
- **Models tried**: LogisticRegression, MultinomialNB, DecisionTree, RandomForest, LinearSVC  
- **Final default**: **LinearSVC**  
- **Artifacts**:  
  - `models/product_category_model.pkl`  
  - `reports/classification_report.txt`  
  - `reports/confusion_matrix.png`  
- **Testing**: `src/predict_category.py` (interactive / single title / file)

---

## 1) Goal & Why it matters

Automatically suggesting the correct category for each new product title speeds up catalog operations, reduces costs, and improves search/browsing quality for users.

---

## 2) Dataset & Normalization

File: `data/products.csv`. On load, headers are normalized:

| Original header         | Used name      |
|-------------------------|----------------|
| `product ID`            | `product_id`   |
| `Product Title`         | `title`        |
| `Merchant ID`           | `merchant_id`  |
| ` Category Label`       | `category`     |
| `_Product Code`         | `product_code` |
| `Number_of_Views`       | `views`        |
| `Merchant Rating`       | `merchant_rating` |
| ` Listing Date  `       | `listing_date` |

Cleaning:
- keep rows with both `title` and `category`;
- strip whitespace;
- unify near-duplicate labels (e.g., `CPU → CPUs`, `Mobile Phone → Mobile Phones`, `fridge → Fridges`).

---

## 3) Architecture & Feature Engineering

Preprocessing uses a **ColumnTransformer** with three branches:

1. **Word TF-IDF**: n-grams (1–2), up to 30-35k features, `strip_accents="unicode"`  
2. **Character TF-IDF**: n-grams (3–4), up to 30-35k features  
3. **Custom `title_stats`** (6 numeric features, **fixed order**):
   - `len_chars`
   - `word_count`
   - `digit_ratio`
   - `upper_ratio`
   - `max_word_len`
   - `has_storage_unit` (GB/TB/MB flag)

> Important: `title_stats` is used **in training and prediction**. To load the serialized pipeline, the same function (same **6 features** in the **same order**) is defined in `predict_category.py`.

---

## 4) Models & Choice

Tried: Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, Linear SVC.  
**LinearSVC** offered the best speed/accuracy trade-off on validation, so it’s the default in `train_model.py`.

---
## 5) Training (train_model.py)
The script cleans the data, builds the preprocessing pipeline, trains the selected model, and exports the artifacts.

```bash
python src/train_model.py \
  --csv data/products.csv \
  --out models/product_category_model.pkl \
  --report_dir reports \
  --algo linear_svc

---

### 6) Prediction (predict_category.py)
Three usage modes:
- **Interactive** (starts a prompt and returns the category immediately):
  ```bash
  python src/predict_category.py

## 7) Results
- Validation performance is reported in `reports/classification_report.txt`.
- The confusion matrix is available at `reports/confusion_matrix.png`.

> In the current runs, accuracy was around **~0.99** (may vary by split/seed).


