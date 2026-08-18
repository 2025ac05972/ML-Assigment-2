# Random Forest (Ensemble) model for Forest CoverType classification
# 2025ac05972 | BITS Pilani WILP | ML Assignment 2

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest.pkl")

def build_model():
    # 150 trees - tried 100, 150, 200; 150 was sweet spot for accuracy vs training time
    # oob_score=True gives a free out-of-bag validation estimate
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        oob_score=True,
        random_state=7,
        n_jobs=-1,
    )

def train(X_train, y_train):
    model = build_model()
    model.fit(X_train, y_train)
    print(f"  OOB accuracy: {model.oob_score_:.4f}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    return model

def load():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }
