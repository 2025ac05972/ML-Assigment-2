# Streamlit app for ML Assignment 2
# 2025ac05972 | BITS Pilani WILP
# run with: streamlit run app.py

import gzip
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# page setup - wide layout looks better for comparison tables
st.set_page_config(
    page_title="Forest Cover Type | ML Classifier",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# green forest theme - kept colours consistent with the tree/nature theme
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .page-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1B5E20;
            text-align: center;
            letter-spacing: 0.5px;
            padding-bottom: 0.4rem;
            border-bottom: 4px solid #43A047;
            margin-bottom: 1.6rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
            border: 1px solid #A5D6A7;
            border-radius: 12px;
            padding: 1.1rem 0.8rem;
            text-align: center;
            box-shadow: 0 3px 8px rgba(0,0,0,0.08);
            margin-bottom: 0.6rem;
        }
        .metric-card .icon  { font-size: 1.6rem; margin-bottom: 0.2rem; }
        .metric-card .value { font-size: 2rem; font-weight: 700; color: #1B5E20; }
        .metric-card .label { font-size: 0.78rem; color: #555; text-transform: uppercase; letter-spacing: 0.8px; }
        .info-banner {
            background: #E3F2FD;
            border-left: 5px solid #1976D2;
            padding: 0.8rem 1.2rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            font-size: 0.92rem;
        }
        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #2E7D32;
            margin-top: 1.2rem;
            margin-bottom: 0.6rem;
        }
        [data-testid="stSidebar"] { background-color: #F9FBE7; }
        .sidebar-brand {
            font-size: 1.35rem;
            font-weight: 800;
            color: #1B5E20;
            text-align: center;
            padding: 0.5rem 0 0.3rem;
        }
        .sidebar-badge {
            background: #C8E6C9;
            color: #1B5E20;
            border-radius: 20px;
            padding: 0.15rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 0.4rem;
        }
        div.stButton > button {
            background: linear-gradient(135deg, #2E7D32, #43A047);
            color: white;
            border: none;
            border-radius: 9px;
            padding: 0.55rem 2.2rem;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        div.stButton > button:hover { opacity: 0.88; }
        .winner-box {
            background: linear-gradient(135deg, #FFF9C4, #FFFDE7);
            border: 2px solid #F9A825;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            text-align: center;
            font-size: 1.15rem;
            font-weight: 700;
            color: #E65100;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# constants
MODEL_DIR = "model"

model_files = {
    "Logistic Regression":    "logistic_regression.pkl",
    "Decision Tree":          "decision_tree.pkl",
    "K-Nearest Neighbors":    "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes.pkl",
    "Random Forest":          "random_forest.pkl",
}

# 0-indexed because sklearn LabelEncoder maps 1->0, 2->1, etc.
cover_labels = {
    0: "Spruce/Fir",
    1: "Lodgepole Pine",
    2: "Ponderosa Pine",
    3: "Cottonwood/Willow",
    4: "Aspen",
    5: "Douglas-fir",
    6: "Krummholz",
}

metric_icons = {
    "Accuracy":  "🎯",
    "AUC Score": "📐",
    "Precision": "🔬",
    "Recall":    "🔄",
    "F1 Score":  "⚖️",
    "MCC Score": "🧮",
}

# --- sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌲 CoverType<br>Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center"><span class="sidebar-badge">M.Tech AIML</span></div>', unsafe_allow_html=True)
    st.markdown("---")

    current_page = st.radio(
        "Navigation",
        [
            "🏠  Overview",
            "📊  Evaluate Model",
            "📈  Compare All Models",
            "ℹ️  Dataset & Info",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <small>
        <b>Assignment</b> : ML – 2<br>
        <b>Dataset</b>    : Forest CoverType<br>
        <b>Source</b>     : UCI / sklearn<br>
        <b>Models</b>     : 5 classifiers<br>
        <b>Metrics</b>    : 6 per model
        </small>
        """,
        unsafe_allow_html=True,
    )

# --- helpers ---
@st.cache_resource(show_spinner=False)
def load_scaler():
    p = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_model(name):
    p = os.path.join(MODEL_DIR, model_files[name])
    p_gz = p + ".gz"
    if os.path.exists(p_gz):
        with gzip.open(p_gz, "rb") as f:
            return pickle.load(f)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def calc_metrics(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)

    auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except ValueError:
            auc = 0.0

    return {
        "Accuracy":  round(acc, 4),
        "AUC Score": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
        "F1 Score":  round(f1, 4),
        "MCC Score": round(mcc, 4),
    }


def plot_cm(y_true, y_pred):
    lbls = sorted(set(y_true) | set(y_pred))
    tick_names = [cover_labels.get(l, str(l)) for l in lbls]
    cm = confusion_matrix(y_true, y_pred, labels=lbls)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=tick_names, yticklabels=tick_names,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def upload_csv(key=""):
    uploaded = st.file_uploader(
        "Upload **test_data.csv** (generated by `train_models.py`)",
        type=["csv"],
        key=f"uploader_{key}",
        help="File must contain 54 feature columns and a 'Cover_Type' target column.",
    )
    if uploaded is None:
        return None, None

    try:
        df = pd.read_csv(uploaded)
    except Exception as err:
        st.error(f"Could not parse CSV: {err}")
        return None, None

    if "Cover_Type" not in df.columns:
        st.error("Column 'Cover_Type' not found. Please upload test_data.csv from train_models.py")
        return None, None

    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"].values
    st.success(f"Loaded {len(df):,} rows x {X.shape[1]} features")
    return X, y


# ============================================================
# PAGE: OVERVIEW
# ============================================================
if current_page == "🏠  Overview":
    st.markdown('<div class="page-title">🌲 Forest Cover Type Classification</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(
            """
            <div class="info-banner">
            This dashboard demonstrates end-to-end ML classification using five
            algorithms trained on the <b>Forest CoverType</b> dataset from the
            UCI Machine Learning Repository.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            ### Problem Statement
            Predict the dominant tree species in a 30x30 m forest patch
            using 54 cartographic measurements collected across Roosevelt National
            Forest in northern Colorado, USA.

            ### How to Use
            1. Run `python train_models.py` to train models and save pkl files.
            2. Upload `test_data.csv` in the Evaluate or Compare pages.
            3. Pick a model and click Run Evaluation.
            4. Check metrics, confusion matrix, and classification report.
            """
        )

    with col_right:
        st.markdown("### Dataset at a Glance")
        glance = {
            "Property": ["Source", "Task", "Records (sample)", "Total Features",
                         "Numeric", "Binary", "Classes"],
            "Value": ["UCI ML Repository", "Multi-class Classification",
                      "10 000", "54", "10", "44", "7 cover types"],
        }
        st.table(pd.DataFrame(glance).set_index("Property"))

        st.markdown("### Cover Type Labels")
        ldf = pd.DataFrame([(k+1, v) for k, v in cover_labels.items()],
                           columns=["Class ID", "Cover Type"])
        st.dataframe(ldf.set_index("Class ID"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Models Available")
    minfo = {
        "Model": list(model_files.keys()),
        "Category": ["Linear", "Tree-Based", "Instance-Based", "Probabilistic", "Ensemble"],
        "Key Hyperparameter": [
            "C=1.0, solver=lbfgs", "max_depth=15", "k=7, euclidean",
            "var_smoothing=1e-9", "n_estimators=150, max_depth=20",
        ],
    }
    st.dataframe(pd.DataFrame(minfo).set_index("Model"), use_container_width=True)


# ============================================================
# PAGE: EVALUATE MODEL
# ============================================================
elif current_page == "📊  Evaluate Model":
    st.markdown('<div class="page-title">📊 Single Model Evaluation</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Step 1 — Upload Test Data</div>', unsafe_allow_html=True)
    X_test, y_test = upload_csv("eval")

    if X_test is not None:
        st.markdown('<div class="section-header">Step 2 — Select Model</div>', unsafe_allow_html=True)
        chosen = st.selectbox(
            "Choose a classification model:",
            options=list(model_files.keys()),
            index=4,
            help="Random Forest is selected by default (highest expected accuracy).",
        )

        st.markdown('<div class="section-header">Step 3 — Run Evaluation</div>', unsafe_allow_html=True)
        if st.button("🚀  Run Evaluation"):
            scaler = load_scaler()
            model  = load_model(chosen)

            if model is None:
                st.error("Model file not found. Run train_models.py first.")
                st.stop()

            with st.spinner(f"Evaluating {chosen}..."):
                X_sc = scaler.transform(X_test) if scaler is not None else X_test.values
                y_pred = model.predict(X_sc)
                y_prob = model.predict_proba(X_sc) if hasattr(model, "predict_proba") else None
                metrics = calc_metrics(y_test, y_pred, y_prob)

            st.markdown(f"## Results — {chosen}")
            st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)

            cols = st.columns(6, gap="small")
            for col, (m_name, m_val) in zip(cols, metrics.items()):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="icon">{metric_icons[m_name]}</div>
                            <div class="value">{m_val:.4f}</div>
                            <div class="label">{m_name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            col_cm, col_rep = st.columns([1, 1], gap="large")

            with col_cm:
                st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
                fig = plot_cm(y_test, y_pred)
                st.pyplot(fig)
                plt.close(fig)

            with col_rep:
                st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
                rpt = (pd.DataFrame(classification_report(y_test, y_pred, zero_division=0, output_dict=True))
                       .T.drop(index=["accuracy"], errors="ignore").round(4))
                st.dataframe(rpt.style.background_gradient(cmap="Greens", axis=0), use_container_width=True)

            with st.expander("View sample predictions (first 20 rows)"):
                pv = X_test.copy().head(20)
                pv["True Label"] = y_test[:20]
                pv["Predicted"]  = y_pred[:20]
                pv["Correct"]    = pv["True Label"] == pv["Predicted"]
                st.dataframe(pv[["True Label", "Predicted", "Correct"]], use_container_width=True)

    else:
        st.markdown(
            '<div class="info-banner">Upload test_data.csv above to get started.</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: COMPARE ALL MODELS
# ============================================================
elif current_page == "📈  Compare All Models":
    st.markdown('<div class="page-title">📈 Multi-Model Comparison</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Upload Test Data</div>', unsafe_allow_html=True)
    X_cmp, y_cmp = upload_csv("cmp")

    if X_cmp is not None:
        if st.button("🔄  Evaluate All 5 Models"):
            scaler = load_scaler()
            X_sc = scaler.transform(X_cmp) if scaler is not None else X_cmp.values

            all_results  = []
            pbar = st.progress(0, text="Evaluating models...")
            total = len(model_files)

            for i, name in enumerate(model_files):
                mdl = load_model(name)
                if mdl is None:
                    st.warning(f"Skipped {name} - model file not found")
                    pbar.progress((i+1)/total)
                    continue
                yp = mdl.predict(X_sc)
                yprob = mdl.predict_proba(X_sc) if hasattr(mdl, "predict_proba") else None
                m = calc_metrics(y_cmp, yp, yprob)
                m["Model"] = name
                all_results.append(m)
                pbar.progress((i+1)/total, text=f"Done: {name}")

            pbar.empty()

            if not all_results:
                st.error("No results - run train_models.py first")
                st.stop()

            cols = ["Model", "Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
            rdf = pd.DataFrame(all_results)[cols].sort_values("Accuracy", ascending=False).reset_index(drop=True)

            st.markdown('<div class="section-header">Performance Comparison</div>', unsafe_allow_html=True)
            st.dataframe(
                rdf.set_index("Model")
                .style.highlight_max(axis=0, color="#C8E6C9")
                .highlight_min(axis=0, color="#FFCDD2")
                .format("{:.4f}"),
                use_container_width=True,
            )

            palette = ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A"]

            st.markdown('<div class="section-header">Accuracy Comparison</div>', unsafe_allow_html=True)
            fig_bar, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(rdf["Model"], rdf["Accuracy"], color=palette[:len(rdf)], edgecolor="white", width=0.55)
            ax.set_ylim(0.0, 1.08)
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("Accuracy by Classifier", fontsize=14, fontweight="bold")
            ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10, fontweight="bold")
            ax.axhline(0.8, color="#E53935", linestyle="--", linewidth=1, alpha=0.7, label="0.80 threshold")
            ax.legend(fontsize=10)
            plt.xticks(rotation=20, ha="right", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

            st.markdown('<div class="section-header">F1 Score Comparison</div>', unsafe_allow_html=True)
            fig_f1, ax2 = plt.subplots(figsize=(10, 5))
            ax2.barh(rdf["Model"][::-1], rdf["F1 Score"][::-1], color=palette[:len(rdf)], edgecolor="white", height=0.5)
            ax2.set_xlim(0.0, 1.08)
            ax2.set_xlabel("F1 Score (weighted)", fontsize=12)
            ax2.set_title("F1 Score by Classifier", fontsize=14, fontweight="bold")
            for bar, val in zip(ax2.patches, rdf["F1 Score"][::-1]):
                ax2.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                         f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_f1)
            plt.close(fig_f1)

            winner = rdf.iloc[0]
            st.markdown(
                f"""<div class="winner-box">
                🏆 Best: {winner['Model']}
                &nbsp;|&nbsp; Acc: {winner['Accuracy']:.4f}
                &nbsp;|&nbsp; F1: {winner['F1 Score']:.4f}
                &nbsp;|&nbsp; MCC: {winner['MCC Score']:.4f}
                </div>""",
                unsafe_allow_html=True,
            )

    else:
        st.info("Upload test_data.csv to compute live results. Reference values from training run shown below.")
        sample_df = pd.DataFrame({
            "Model":     ["Random Forest", "K-Nearest Neighbors", "Logistic Regression",
                          "Decision Tree", "Naive Bayes (Gaussian)"],
            "Accuracy":  [0.7795, 0.7410, 0.7305, 0.7120, 0.1045],
            "AUC Score": [0.9212, 0.8797, 0.8720, 0.8232, 0.6723],
            "Precision": [0.7780, 0.7383, 0.7236, 0.7053, 0.5196],
            "Recall":    [0.7795, 0.7410, 0.7305, 0.7120, 0.1045],
            "F1 Score":  [0.7679, 0.7346, 0.7188, 0.7052, 0.0827],
            "MCC Score": [0.6371, 0.5783, 0.5580, 0.5273, 0.0748],
        })
        st.dataframe(
            sample_df.set_index("Model").style.highlight_max(axis=0, color="#C8E6C9").format("{:.4f}"),
            use_container_width=True,
        )


# ============================================================
# PAGE: DATASET & INFO
# ============================================================
elif current_page == "ℹ️  Dataset & Info":
    st.markdown('<div class="page-title">ℹ️ Dataset & Project Information</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Dataset Details", "Model Details", "Academic Info"])

    with tab1:
        st.markdown(
            """
            ### Forest CoverType Dataset

            | Attribute         | Detail                                           |
            |-------------------|--------------------------------------------------|
            | Full Name         | Forest Cover Type Dataset                        |
            | Source            | UCI Machine Learning Repository                  |
            | Records (full)    | 581,012                                          |
            | Records (sampled) | 10,000 (stratified by cover type)                |
            | Features          | 54 cartographic variables                        |
            | Target            | Cover_Type (7 classes, integer 1-7)              |
            | Task              | Multi-class classification                       |
            | Missing Values    | None                                             |

            **Quantitative Features (10):** Elevation, Aspect, Slope, distances to
            hydrology/roads/fire points, and three hillshade readings.

            **Wilderness Area (4 binary):** Rawah, Neota, Comanche Peak, Cache la Poudre.

            **Soil Type (40 binary):** One-hot soil indicators (Soil_Type_1 to Soil_Type_40).

            I picked this dataset because it has far more features than the 12-feature minimum,
            has real-world relevance, and is less commonly used in student projects compared to Iris/Titanic.
            """
        )

    with tab2:
        st.markdown(
            """
            ### Model Configurations

            | Model                 | Key Hyperparameters                            |
            |-----------------------|------------------------------------------------|
            | Logistic Regression   | C=1.0, solver=lbfgs, max_iter=1000             |
            | Decision Tree         | max_depth=15, min_samples_split=10             |
            | K-Nearest Neighbors   | k=7, metric=Euclidean                          |
            | Naive Bayes (Gaussian)| var_smoothing=1e-9                             |
            | Random Forest         | n_estimators=150, max_depth=20, oob_score=True |

            ### Preprocessing Steps
            1. Stratified 10k sample to keep class proportions intact.
            2. Checked for missing values (none found).
            3. LabelEncoder: original labels 1-7 re-mapped to 0-6.
            4. 80/20 stratified train/test split.
            5. StandardScaler fitted on training set only (no leakage).

            ### Evaluation Metrics
            - **Accuracy** - proportion of correct predictions
            - **AUC (OvR weighted)** - area under ROC for each class vs rest
            - **Precision / Recall / F1 (weighted)** - standard classification metrics
            - **MCC** - Matthews Correlation Coefficient, handles class imbalance well
            """
        )

    with tab3:
        st.markdown(
            """
            ### Academic Details

            | Field           | Value                       |
            |-----------------|-----------------------------|
            | Programme       | M.Tech AIML / DSE           |
            | Course          | Machine Learning            |
            | Assignment No.  | 2                           |
            | Student ID      | 2025ac05972                 |
            | Submission Date | 18-Aug-2026                 |
            | Institution     | BITS Pilani - WILP          |

            ### Stack
            Python 3.10+ | scikit-learn 1.4+ | Streamlit 1.35+ | Pandas | NumPy | Matplotlib | Seaborn

            ### Repo Structure
            ```
            forest-cover-type-classifier/
            ├── app.py
            ├── train_models.py
            ├── requirements.txt
            ├── README.md
            ├── test_data.csv
            ├── model/
            │   ├── logistic_regression.pkl
            │   ├── decision_tree.pkl
            │   ├── knn.pkl
            │   ├── naive_bayes.pkl
            │   ├── random_forest.pkl
            │   ├── scaler.pkl
            │   └── label_encoder.pkl
            └── notebooks/
                └── model_training.ipynb
            ```
            """
        )
