# K-Nearest Neighbors model for Forest CoverType classification
# 2025ac05972 | BITS Pilani WILP | ML Assignment 2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "knn.pkl")

def build_model():
    # k=7 from elbow curve - error rate plateaus around k=6-8
    return KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=-1)

def train(X_train, y_train):
    model = build_model()
    model.fit(X_train, y_train)
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
