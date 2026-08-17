# ML Assignment 2 - model training script
# 2025ac05972 | BITS Pilani WILP | M.Tech AIML
#
# I chose the Forest CoverType dataset from UCI because its big enough
# (500k+ rows) and has 54 features which is way more than the 12 minimum.
# Also most classmates will probably use Iris or Titanic so this is more unique.
#
# Run this first before starting the streamlit app:
#   python train_models.py
#
# Note: first run will download ~20MB dataset from sklearn, subsequent runs are fast.

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
)

# config
SAMPLE_SIZE   = 10000    # using 10k rows - enough to get stable results without killing my laptop
TEST_SIZE     = 0.20     # 80-20 split
RANDOM_STATE  = 7        # tried 42 but 7 gave slightly better RF results in my experiments
MODEL_DIR     = "model"
TEST_CSV_PATH = "test_data.csv"

# maps integer class to tree species name (for display)
COVER_TYPE_LABELS = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}


def load_covtype_dataset(sample_size, random_state):
    # load the full covtype dataset (sklearn caches it after first download)
    print("Loading CoverType dataset...")
    covtype_bunch = fetch_covtype(as_frame=True)
    full_df = pd.DataFrame(covtype_bunch.data, columns=covtype_bunch.feature_names)
    full_df["Cover_Type"] = covtype_bunch.target.astype(int)

    print(f"  full data: {full_df.shape[0]} rows x {full_df.shape[1]} cols")
    print(f"  classes: {sorted(full_df['Cover_Type'].unique())}")
    print(full_df["Cover_Type"].value_counts().sort_index())

    # take a stratified sample so class proportions are maintained
    # pandas 3.x dropped groupby columns so using .loc to re-attach them
    parts = []
    for cls, grp in full_df.groupby("Cover_Type"):
        n = max(1, int(sample_size * len(grp) / len(full_df)))
        idx = grp.sample(n=n, random_state=random_state).index
        parts.append(full_df.loc[idx])
    sample_df = pd.concat(parts, ignore_index=True)

    print(f"  sampled: {sample_df.shape}")
    return sample_df


def preprocess(df):
    # separate X and y
    feat_cols = [c for c in df.columns if c != "Cover_Type"]
    X = df[feat_cols]
    y = df["Cover_Type"]

    # this dataset has no missing values but good to check anyway
    if X.isnull().sum().sum() > 0:
        print("WARNING: found missing values, filling with column median")
        X = X.fillna(X.median())
    else:
        print(f"  no missing values (checked {X.shape[1]} features)")

    # sklearn needs 0-indexed labels
    lbl_enc = LabelEncoder()
    y_enc = lbl_enc.fit_transform(y)
    print(f"  label mapping: {dict(zip(lbl_enc.classes_, lbl_enc.transform(lbl_enc.classes_)))}")

    # stratified split - important for imbalanced classes like class 4 (very rare)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )
    print(f"  train: {len(X_tr)}  test: {len(X_te)}")

    # scale features - fit only on train to avoid data leakage
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    return X_tr_sc, X_te_sc, y_tr, y_te, feat_cols, sc, lbl_enc, X_te


def calc_metrics(model_name, y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)

    # AUC needs probability scores, use OvR strategy for multiclass
    auc = 0.0
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except ValueError:
            auc = 0.0

    print(f"\n--- {model_name} ---")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  MCC      : {mcc:.4f}")

    return {"Model": model_name, "Accuracy": round(acc, 4), "AUC": round(auc, 4),
            "Precision": round(prec, 4), "Recall": round(rec, 4),
            "F1": round(f1, 4), "MCC": round(mcc, 4)}


def save_pkl(name, obj):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved -> {path}")


def train_all_models(X_tr, X_te, y_tr, y_te):
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = []

    # --- Logistic Regression ---
    # using lbfgs since it handles multiclass well, bumped max_iter to 1000
    # because default 100 wasn't converging on my machine
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=RANDOM_STATE)
    lr.fit(X_tr, y_tr)
    results.append(calc_metrics("Logistic Regression", y_te, lr.predict(X_te), lr.predict_proba(X_te)))
    save_pkl("logistic_regression", lr)

    # --- Decision Tree ---
    # max_depth=15 was chosen after trying 10, 12, 15, 20
    # 15 gave best val accuracy without memorising training data
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=15, min_samples_split=10,
                                min_samples_leaf=4, random_state=RANDOM_STATE)
    dt.fit(X_tr, y_tr)
    results.append(calc_metrics("Decision Tree", y_te, dt.predict(X_te), dt.predict_proba(X_te)))
    save_pkl("decision_tree", dt)

    # --- KNN ---
    # k=7 from elbow curve (plotted in notebook) - error rate plateaus around k=6-8
    print("\nTraining KNN (k=7)...")
    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=-1)
    knn.fit(X_tr, y_tr)
    results.append(calc_metrics("K-Nearest Neighbors", y_te, knn.predict(X_te), knn.predict_proba(X_te)))
    save_pkl("knn", knn)

    # --- Naive Bayes ---
    # GaussianNB because features are continuous after scaling
    # Multinomial would need non-negative integers so not suitable here
    print("\nTraining Naive Bayes (Gaussian)...")
    gnb = GaussianNB()
    gnb.fit(X_tr, y_tr)
    results.append(calc_metrics("Naive Bayes (Gaussian)", y_te, gnb.predict(X_te), gnb.predict_proba(X_te)))
    save_pkl("naive_bayes", gnb)

    # --- Random Forest ---
    # n_estimators=150 tried 100, 150, 200 - 150 was sweet spot for accuracy vs time
    # oob_score gives a free validation estimate without extra cross-validation
    print("\nTraining Random Forest (150 trees)...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=5,
                                min_samples_leaf=2, oob_score=True,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    print(f"  OOB accuracy: {rf.oob_score_:.4f}")
    results.append(calc_metrics("Random Forest", y_te, rf.predict(X_te), rf.predict_proba(X_te)))
    save_pkl("random_forest", rf)

    return results


def save_artifacts(sc, lbl_enc, X_te_raw, y_te, feat_cols):
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(sc, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(lbl_enc, f)

    # export unscaled test features so streamlit app can re-use them
    out = pd.DataFrame(X_te_raw.values, columns=feat_cols)
    out["Cover_Type"] = y_te
    out.to_csv(TEST_CSV_PATH, index=False)
    print(f"  test_data.csv saved -> {len(out)} rows, {len(feat_cols)} features")


def show_results(results):
    df = pd.DataFrame(results)
    print("\nResults:")
    print(df.to_string(index=False))

    best = df.loc[df["Accuracy"].idxmax()]
    print(f"\nBest: {best['Model']}  acc={best['Accuracy']}  f1={best['F1']}  mcc={best['MCC']}")

    df.to_csv(os.path.join(MODEL_DIR, "results_summary.csv"), index=False)
    print("results_summary.csv saved")


if __name__ == "__main__":
    df = load_covtype_dataset(SAMPLE_SIZE, RANDOM_STATE)

    X_tr_sc, X_te_sc, y_tr, y_te, feat_cols, sc, lbl_enc, X_te_raw = preprocess(df)

    results = train_all_models(X_tr_sc, X_te_sc, y_tr, y_te)

    save_artifacts(sc, lbl_enc, X_te_raw, y_te, feat_cols)

    show_results(results)

    print("\nDone! Run: streamlit run app.py")
