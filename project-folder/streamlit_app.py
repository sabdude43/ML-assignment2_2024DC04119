import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent


def rel(p: str) -> str:
    return str(BASE_DIR / p)


st.set_page_config(page_title="Breast Cancer Models", layout="wide")

st.title("Breast Cancer Classification Models")

tabs = st.tabs(["Models", "Predict", "Results", "About"])

# --- Models tab: show metrics CSV if present
with tabs[0]:
    st.header("Model Comparison")
    try:
        df = pd.read_csv(rel("results/model_comparison.csv"), index_col=0)
        st.dataframe(df.round(4))
    except Exception:
        st.error("results/model_comparison.csv not found. Run train_models.py")


# --- Predict tab
with tabs[1]:
    st.header("Predict / Evaluate")

    uploaded = st.file_uploader("Upload test CSV (features + optional target)", type=["csv"]) 

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ]

    # multiselect used for both evaluation and single-sample
    selected_models = st.multiselect("Select model(s)", model_names, default=model_names)

    # Evaluate uploaded dataset
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.write(df_up.head())
            if "target" in df_up.columns:
                y = df_up["target"]
                X = df_up.drop(columns=["target"])
            elif "diagnosis" in df_up.columns:
                y = df_up["diagnosis"]
                X = df_up.drop(columns=["diagnosis"])
            else:
                if os.path.exists(rel("data/y_test.csv")):
                    y = pd.read_csv(rel("data/y_test.csv")).squeeze()
                    X = df_up
                else:
                    st.error("No target column found; cannot evaluate")
                    X = None
                    y = None

            if X is not None:
                scaler = joblib.load(rel("model/scaler.pkl"))
                try:
                    Xs = scaler.transform(X)
                except Exception:
                    Xs = X

                models = {
                    name: joblib.load(rel(f"model/{name.lower().replace(' ', '_')}_model.pkl"))
                    for name in model_names
                }

                results = {}
                for name in selected_models:
                    m = models[name]
                    try:
                        yp = m.predict(Xs)
                        prob = m.predict_proba(Xs)[:, 1]
                    except Exception:
                        yp = m.predict(X)
                        try:
                            prob = m.predict_proba(X)[:, 1]
                        except Exception:
                            prob = np.zeros(len(yp))
                    results[name] = {
                        "accuracy": accuracy_score(y, yp),
                        "precision": precision_score(y, yp, zero_division=0),
                        "recall": recall_score(y, yp, zero_division=0),
                        "f1": f1_score(y, yp, zero_division=0),
                        "auc": (roc_auc_score(y, prob) if len(prob) == len(yp) else np.nan),
                        "y_pred": yp,
                    }

                res_df = pd.DataFrame(results).T.drop(columns=["y_pred"]).round(4)
                st.dataframe(res_df)

                for name, v in results.items():
                    st.subheader(name)
                    cm = confusion_matrix(y, v["y_pred"]) 
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    st.pyplot(fig)
                    st.text(classification_report(y, v["y_pred"], zero_division=0))

        except Exception as e:
            st.error(f"Failed to evaluate uploaded file: {e}")

    # Single-sample prediction inputs
    try:
        X_test = pd.read_csv(rel("data/X_test.csv"))
        feats = X_test.columns.tolist()
        sample = {}
        cols = st.columns(3)
        for i, f in enumerate(feats):
            col = cols[i % 3]
            with col:
                sample[f] = st.slider(f, float(X_test[f].min()), float(X_test[f].max()), float(X_test[f].mean()))

        if st.button("Predict sample"):
            scaler = joblib.load(rel("model/scaler.pkl"))
            input_df = pd.DataFrame([sample])
            try:
                ins = scaler.transform(input_df)
            except Exception:
                ins = input_df

            models = {
                name: joblib.load(rel(f"model/{name.lower().replace(' ', '_')}_model.pkl"))
                for name in model_names
            }

            cols_out = st.columns(3)
            preds = {}
            for idx, name in enumerate(model_names):
                if name not in selected_models:
                    continue
                m = models[name]
                try:
                    p = m.predict(ins)[0]
                    prob = m.predict_proba(ins)[0]
                except Exception:
                    p = m.predict(input_df)[0]
                    prob = m.predict_proba(input_df)[0]
                preds[name] = (p, prob)
                col = cols_out[idx % 3]
                with col:
                    st.markdown(f"**{name}**")
                    st.metric("Malignant", f"{prob[0]*100:.2f}%")
                    st.metric("Benign", f"{prob[1]*100:.2f}%")

            if preds:
                votes = [v[0] for v in preds.values()]
                st.metric("Models predicting benign", sum(votes))

    except FileNotFoundError:
        st.info("No X_test.csv found in data/. Single-sample inputs hidden.")


# TAB 3
with tabs[2]:
    st.header("Results & Analysis")
    try:
        if os.path.exists(rel('results/model_comparison.png')):
            st.image(rel('results/model_comparison.png'))
        if os.path.exists(rel('results/roc_curves.png')):
            st.image(rel('results/roc_curves.png'))
        if os.path.exists(rel('results/confusion_matrices.png')):
            st.image(rel('results/confusion_matrices.png'))
        df = pd.read_csv(rel('results/model_comparison.csv'), index_col=0)
        st.dataframe(df.round(4))
    except Exception:
        st.error('Results not found. Run train_models.py')


with tabs[3]:
    st.header('About')
    st.markdown('Simple Streamlit app for evaluation and prediction.')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


# Base directory for relative assets
BASE_DIR = Path(__file__).resolve().parent


def rel(path: str) -> str:
    return str(BASE_DIR / path)


st.set_page_config(page_title="Breast Cancer Classification Models",
                   page_icon="🏥", layout="wide")


st.markdown("""
<style>
 .main { padding: 20px; }
 .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


st.title("🏥 Breast Cancer Classification Models")
st.markdown("**Multi-Model Comparison & Prediction System**")
st.markdown("Binary classification of breast cancer tumors using 6 different ML models")


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "🔮 Make Predictions",
    "📈 Results & Analysis",
    "ℹ️ About",
])


# TAB 1
with tab1:
    st.header("Model Performance Comparison")
    try:
        results_df = pd.read_csv(rel("results/model_comparison.csv"), index_col=0)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Performance Metrics Table")
            st.dataframe(results_df.round(4), use_container_width=True)
        with col2:
            st.subheader("Key Statistics")
            st.metric("Best Accuracy", f"{results_df['accuracy'].max():.4f}",
                      results_df['accuracy'].idxmax())
            st.metric("Best AUC", f"{results_df['auc'].max():.4f}",
                      results_df['auc'].idxmax())
            st.metric("Best F1-Score", f"{results_df['f1'].max():.4f}",
                      results_df['f1'].idxmax())

        st.subheader("Metric Comparison Charts")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4))
            results_df['accuracy'].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Accuracy')
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 4))
            results_df['auc'].plot(kind='bar', ax=ax, color='coral')
            ax.set_title('AUC')
            st.pyplot(fig)
        with c3:
            fig, ax = plt.subplots(figsize=(8, 4))
            results_df['f1'].plot(kind='bar', ax=ax, color='mediumseagreen')
            ax.set_title('F1')
            st.pyplot(fig)
    except FileNotFoundError:
        st.error("Results file not found. Please run train_models.py first.")


# TAB 2
with tab2:
    st.header("Make Predictions with Trained Models")
    st.markdown("Upload a test CSV to evaluate models or enter a single sample below.")

    uploaded_file = st.file_uploader("Upload CSV (features and optional 'target' column)", type=["csv"])

    def _normalize_target(y):
        try:
            if y.dtype == object:
                s = y.astype(str).str.strip().str.lower()
                mapping = {"m": 0, "malignant": 0, "b": 1, "benign": 1, "0": 0, "1": 1}
                s = s.map(lambda v: mapping.get(v, v))
                return pd.to_numeric(s, errors='coerce').astype(int)
            return pd.to_numeric(y, errors='coerce').astype(int)
        except Exception:
            return y

    # Evaluation from uploaded CSV
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head())

            if 'target' in uploaded_df.columns:
                y_true = uploaded_df['target']
                X_eval = uploaded_df.drop(columns=['target'])
            elif 'diagnosis' in uploaded_df.columns:
                y_true = uploaded_df['diagnosis']
                X_eval = uploaded_df.drop(columns=['diagnosis'])
            else:
                if os.path.exists(rel('data/y_test.csv')):
                    y_true = pd.read_csv(rel('data/y_test.csv')).squeeze()
                    X_eval = uploaded_df
                else:
                    st.error("No target column and data/y_test.csv not found. Cannot evaluate.")
                    X_eval = None
                    y_true = None

            if X_eval is not None:
                y_true = _normalize_target(y_true)

                model_options = [
                    "Logistic Regression", "Decision Tree", "KNN",
                    "Naive Bayes", "Random Forest", "XGBoost"
                ]

                models_selected = st.multiselect("Select model(s) to evaluate", model_options, default=model_options)

                if st.button("Evaluate on uploaded data"):
                    try:
                        scaler = joblib.load(rel('model/scaler.pkl'))
                        models_dict = {
                            "Logistic Regression": joblib.load(rel('model/logistic_regression_model.pkl')),
                            "Decision Tree": joblib.load(rel('model/decision_tree_model.pkl')),
                            "KNN": joblib.load(rel('model/knn_model.pkl')),
                            "Naive Bayes": joblib.load(rel('model/naive_bayes_model.pkl')),
                            "Random Forest": joblib.load(rel('model/random_forest_model.pkl')),
                            "XGBoost": joblib.load(rel('model/xgboost_model.pkl')),
                        }

                        try:
                            X_scaled = scaler.transform(X_eval)
                        except Exception:
                            X_scaled = X_eval

                        eval_results = {}
                        for name, model in models_dict.items():
                            if len(models_selected) > 0 and name not in models_selected:
                                continue
                            try:
                                y_pred = model.predict(X_scaled)
                                y_prob = model.predict_proba(X_scaled)[:, 1]
                            except Exception:
                                y_pred = model.predict(X_eval)
                                try:
                                    y_prob = model.predict_proba(X_eval)[:, 1]
                                except Exception:
                                    y_prob = np.zeros(len(y_pred))

                            eval_results[name] = {
                                'accuracy': accuracy_score(y_true, y_pred),
                                'precision': precision_score(y_true, y_pred, zero_division=0),
                                'recall': recall_score(y_true, y_pred, zero_division=0),
                                'f1': f1_score(y_true, y_pred, zero_division=0),
                                'auc': (roc_auc_score(y_true, y_prob) if len(y_prob) == len(y_pred) else np.nan),
                                'y_pred': y_pred,
                            }

                        res_df = pd.DataFrame.from_dict(eval_results, orient='index').drop(columns=['y_pred'])
                        st.subheader('Evaluation Metrics')
                        st.dataframe(res_df.round(4))

                        for name, v in eval_results.items():
                            st.subheader(f"Confusion Matrix: {name}")
                            cm = confusion_matrix(y_true, v['y_pred'])
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            st.text(classification_report(y_true, v['y_pred'], zero_division=0))

                    except FileNotFoundError as e:
                        st.error(f"Model files not found: {e}")

        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")


    # Single-sample input and prediction
    try:
        X_test = pd.read_csv(rel('data/X_test.csv'))
        feature_names = X_test.columns.tolist()
        st.subheader('Feature Input (single sample)')
        cols = st.columns(3)
        user_input = {}
        for i, feat in enumerate(feature_names):
            col = cols[i % 3]
            with col:
                minv = float(X_test[feat].min())
                maxv = float(X_test[feat].max())
                meanv = float(X_test[feat].mean())
                step = (maxv - minv) / 100 if maxv > minv else 0.01
                user_input[feat] = st.slider(feat, min_value=minv, max_value=maxv, value=meanv, step=step)

        model_options = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
        models_to_predict = st.multiselect('Select model(s) for prediction', model_options, default=model_options)

        if st.button('Get Predictions'):
            try:
                scaler = joblib.load(rel('model/scaler.pkl'))
                input_df = pd.DataFrame([user_input])
                try:
                    input_scaled = scaler.transform(input_df)
                except Exception:
                    input_scaled = input_df

                models_dict = {
                    'Logistic Regression': joblib.load(rel('model/logistic_regression_model.pkl')),
                    'Decision Tree': joblib.load(rel('model/decision_tree_model.pkl')),
                    'KNN': joblib.load(rel('model/knn_model.pkl')),
                    'Naive Bayes': joblib.load(rel('model/naive_bayes_model.pkl')),
                    'Random Forest': joblib.load(rel('model/random_forest_model.pkl')),
                    'XGBoost': joblib.load(rel('model/xgboost_model.pkl')),
                }

                st.subheader('Model Predictions')
                out_cols = st.columns(3)
                preds = {}
                for idx, (name, m) in enumerate(models_dict.items()):
                    if len(models_to_predict) > 0 and name not in models_to_predict:
                        continue
                    col = out_cols[idx % 3]
                    try:
                        p = m.predict(input_scaled)[0]
                        prob = m.predict_proba(input_scaled)[0]
                    except Exception:
                        p = m.predict(input_df)[0]
                        prob = m.predict_proba(input_df)[0]
                    preds[name] = {'pred': p, 'prob': prob}
                    with col:
                        st.markdown(f'### {name}')
                        diag = '🔴 Malignant' if p == 0 else '🟢 Benign'
                        st.markdown(f'**Prediction:** {diag}')
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric('Malignant', f"{prob[0]*100:.2f}%")
                        with c2:
                            st.metric('Benign', f"{prob[1]*100:.2f}%")

                if preds:
                    votes = [v['pred'] for v in preds.values()]
                    cons = sum(votes) / len(votes)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric('Models Predicting Benign', sum(votes))
                    with c2:
                        st.metric('Models Predicting Malignant', len(votes) - sum(votes))
                    with c3:
                        st.metric('Consensus', '🟢 BENIGN' if cons >= 0.5 else '🔴 MALIGNANT')

            except FileNotFoundError as e:
                st.error(f"Model files not found: {e}")

    except FileNotFoundError:
        st.error('Dataset file not found. Please run load_dataset.py first.')


# TAB 3
with tab3:
    st.header('Detailed Results & Analysis')
    try:
        if os.path.exists(rel('results/model_comparison.png')):
            st.image(rel('results/model_comparison.png'), use_container_width=True)
        if os.path.exists(rel('results/roc_curves.png')):
            st.image(rel('results/roc_curves.png'), use_container_width=True)
        if os.path.exists(rel('results/confusion_matrices.png')):
            st.image(rel('results/confusion_matrices.png'), use_container_width=True)

        results_df = pd.read_csv(rel('results/model_comparison.csv'), index_col=0)
        st.subheader('Detailed Metrics')
        st.dataframe(results_df.round(4), use_container_width=True)
        csv = results_df.to_csv()
        st.download_button('📥 Download Results as CSV', data=csv, file_name='model_comparison_results.csv', mime='text/csv')
    except FileNotFoundError:
        st.error('Results files not found. Please run train_models.py first.')


# TAB 4
with tab4:
    st.header('About This Application')
    st.subheader('📋 Project Overview')
    st.markdown('This application compares 6 ML models on the Breast Cancer dataset.')
    st.subheader('🤖 Models Implemented')
    st.markdown('- Logistic Regression\n- Decision Tree\n- KNN\n- Naive Bayes\n- Random Forest\n- XGBoost')
    st.divider()
    st.markdown('🔬 ML Assignment 2 | Breast Cancer Classification | February 2026')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


# Base directory for relative assets (makes paths work regardless of CWD)
BASE_DIR = Path(__file__).resolve().parent


def rel(path: str) -> str:
    return str(BASE_DIR / path)


# Configure page
st.set_page_config(
    page_title="Breast Cancer Classification Models",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown(
    """
<style>
    .main {
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Title and description
st.title("🏥 Breast Cancer Classification Models")
st.markdown("**Multi-Model Comparison & Prediction System**")
st.markdown("Binary classification of breast cancer tumors using 6 different ML models")


# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "🔮 Make Predictions",
    "📈 Results & Analysis",
    "ℹ️ About",
])


# ============================================================================
# TAB 1: Model Comparison
# ============================================================================
with tab1:
    st.header("Model Performance Comparison")
    try:
        results_df = pd.read_csv(rel("results/model_comparison.csv"), index_col=0)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Performance Metrics Table")
            st.dataframe(results_df.round(4), use_container_width=True)

        with col2:
            st.subheader("Key Statistics")
            st.metric("Best Accuracy", f"{results_df['accuracy'].max():.4f}", results_df['accuracy'].idxmax())
            st.metric("Best AUC", f"{results_df['auc'].max():.4f}", results_df['auc'].idxmax())
            st.metric("Best F1-Score", f"{results_df['f1'].max():.4f}", results_df['f1'].idxmax())

        st.subheader("Metric Comparison Charts")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            results_df['accuracy'].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
            ax.set_ylabel('Accuracy')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0.0, 1.0])
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            results_df['auc'].plot(kind='bar', ax=ax, color='coral')
            ax.set_title('AUC Score Comparison', fontweight='bold', fontsize=12)
            ax.set_ylabel('AUC')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0.0, 1.0])
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(8, 5))
            results_df['f1'].plot(kind='bar', ax=ax, color='mediumseagreen')
            ax.set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
            ax.set_ylabel('F1-Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0.0, 1.0])
            st.pyplot(fig)

    except FileNotFoundError:
        st.error("Results file not found. Please run train_models.py first.")


# ============================================================================
# TAB 2: Make Predictions and Evaluation
# ============================================================================
with tab2:
    st.header("Make Predictions with Trained Models")

    st.markdown(
        """
Enter feature values for a sample to get predictions from models,
or upload a test CSV (features and optional target) to evaluate.
"""
    )

    # --- Upload test CSV (features only) for evaluation ---
    uploaded_file = st.file_uploader(
        "Upload test CSV (CSV with features and optional 'target' or 'diagnosis' column)", type=["csv"]
    )

    def _normalize_target(y_series: pd.Series) -> pd.Series:
        try:
            if y_series.dtype == object:
                s = y_series.astype(str).str.strip().str.lower()
                mapping = {"m": 0, "malignant": 0, "b": 1, "benign": 1, "0": 0, "1": 1}
                s = s.map(lambda v: mapping.get(v, v))
                return pd.to_numeric(s, errors="coerce").astype(int)
            return pd.to_numeric(y_series, errors="coerce").astype(int)
        except Exception:
            return y_series

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Test Data (first 5 rows)")
            st.dataframe(uploaded_df.head())

            if "target" in uploaded_df.columns:
                y_true = uploaded_df["target"]
                X_eval = uploaded_df.drop(columns=["target"])
            elif "diagnosis" in uploaded_df.columns:
                y_true = uploaded_df["diagnosis"]
                X_eval = uploaded_df.drop(columns=["diagnosis"])
            else:
                # Fallback: try to load y_test from data folder
                if os.path.exists(rel("data/y_test.csv")):
                    y_true = pd.read_csv(rel("data/y_test.csv")).squeeze()
                    X_eval = uploaded_df
                else:
                    st.error("No target column in uploaded file and data/y_test.csv not found. Cannot evaluate.")
                    X_eval = None
                    y_true = None

            if X_eval is not None:
                y_true = _normalize_target(y_true)

                model_options = [
                    "Logistic Regression",
                    "Decision Tree",
                    "KNN",
                    "Naive Bayes",
                    "Random Forest",
                    "XGBoost",
                ]

                models_selected = st.multiselect(
                    "Select model(s) to evaluate (choose one or more)", model_options, default=model_options
                )

                if st.button("Evaluate on uploaded test data"):
                    try:
                        scaler = joblib.load(rel("model/scaler.pkl"))

                        models_dict = {
                            "Logistic Regression": joblib.load(rel("model/logistic_regression_model.pkl")),
                            "Decision Tree": joblib.load(rel("model/decision_tree_model.pkl")),
                            "KNN": joblib.load(rel("model/knn_model.pkl")),
                            "Naive Bayes": joblib.load(rel("model/naive_bayes_model.pkl")),
                            "Random Forest": joblib.load(rel("model/random_forest_model.pkl")),
                            "XGBoost": joblib.load(rel("model/xgboost_model.pkl")),
                        }

                        # Try using scaled features; fallback to raw if model expects raw
                        try:
                            X_scaled = scaler.transform(X_eval)
                        except Exception:
                            X_scaled = X_eval

                        eval_results = {}
                        for name, model in models_dict.items():
                            if len(models_selected) > 0 and name not in models_selected:
                                continue

                            try:
                                y_pred = model.predict(X_scaled)
                                y_prob = model.predict_proba(X_scaled)[:, 1]
                            except Exception:
                                y_pred = model.predict(X_eval)
                                try:
                                    y_prob = model.predict_proba(X_eval)[:, 1]
                                except Exception:
                                    y_prob = np.zeros(len(y_pred))

                            acc = accuracy_score(y_true, y_pred)
                            prec = precision_score(y_true, y_pred, zero_division=0)
                            rec = recall_score(y_true, y_pred, zero_division=0)
                            f1 = f1_score(y_true, y_pred, zero_division=0)
                            try:
                                auc = roc_auc_score(y_true, y_prob)
                            except Exception:
                                auc = np.nan

                            eval_results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "y_pred": y_pred}

                        res_df = pd.DataFrame.from_dict(eval_results, orient="index").drop(columns=["y_pred"])
                        st.subheader("Evaluation Metrics")
                        st.dataframe(res_df.round(4))

                        for name, v in eval_results.items():
                            st.subheader(f"Confusion Matrix & Report: {name}")
                            cm = confusion_matrix(y_true, v["y_pred"]) 
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig)
                            st.text(classification_report(y_true, v["y_pred"], zero_division=0))

                    except FileNotFoundError as e:
                        st.error(f"Model files not found: {e}. Please ensure models are trained.")

        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")


    # --- Single-sample sliders and predictions ---
    try:
        X_test = pd.read_csv(rel("data/X_test.csv"))
        feature_names = X_test.columns.tolist()

        st.subheader("Feature Input (single sample)")
        col_inputs = st.columns(3)
        user_input = {}
        for idx, feature in enumerate(feature_names):
            col_idx = idx % 3
            with col_inputs[col_idx]:
                min_val = float(X_test[feature].min())
                max_val = float(X_test[feature].max())
                mean_val = float(X_test[feature].mean())
                step = (max_val - min_val) / 100 if max_val > min_val else 0.01
                user_input[feature] = st.slider(feature, min_value=min_val, max_value=max_val, value=mean_val, step=step)

        models_to_predict = st.multiselect(
            "Select model(s) for prediction (choose one or more)",
            ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"],
            default=["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"],
        )

        if st.button("🔮 Get Predictions", use_container_width=True):
            try:
                scaler = joblib.load(rel("model/scaler.pkl"))
                input_df = pd.DataFrame([user_input])
                try:
                    input_scaled = scaler.transform(input_df)
                except Exception:
                    input_scaled = input_df

                models_dict = {
                    "Logistic Regression": joblib.load(rel("model/logistic_regression_model.pkl")),
                    "Decision Tree": joblib.load(rel("model/decision_tree_model.pkl")),
                    "KNN": joblib.load(rel("model/knn_model.pkl")),
                    "Naive Bayes": joblib.load(rel("model/naive_bayes_model.pkl")),
                    "Random Forest": joblib.load(rel("model/random_forest_model.pkl")),
                    "XGBoost": joblib.load(rel("model/xgboost_model.pkl")),
                }

                st.subheader("Model Predictions")
                cols = st.columns(3)
                prediction_results = {}
                for idx, (model_name, model) in enumerate(models_dict.items()):
                    if len(models_to_predict) > 0 and model_name not in models_to_predict:
                        continue
                    col = cols[idx % 3]
                    try:
                        pred = model.predict(input_scaled)[0]
                        prob = model.predict_proba(input_scaled)[0]
                    except Exception:
                        pred = model.predict(input_df)[0]
                        prob = model.predict_proba(input_df)[0]

                    prediction_results[model_name] = {"prediction": pred, "prob": prob}

                    with col:
                        st.markdown(f"### {model_name}")
                        diagnosis = "🔴 Malignant" if pred == 0 else "🟢 Benign"
                        st.markdown(f"**Prediction:** {diagnosis}")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Malignant", f"{prob[0]*100:.2f}%")
                        with c2:
                            st.metric("Benign", f"{prob[1]*100:.2f}%")

                # Consensus
                if prediction_results:
                    preds = [v["prediction"] for v in prediction_results.values()]
                    consensus = sum(preds) / len(preds)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Models Predicting Benign", sum(preds))
                    with col2:
                        st.metric("Models Predicting Malignant", len(preds) - sum(preds))
                    with col3:
                        consensus_diag = "🟢 BENIGN" if consensus >= 0.5 else "🔴 MALIGNANT"
                        st.metric("Consensus", consensus_diag)

            except FileNotFoundError as e:
                st.error(f"Model files not found: {e}. Please ensure all models are trained.")

    except FileNotFoundError:
        st.error("Dataset file not found. Please run load_dataset.py first.")


# ============================================================================
# TAB 3: Results & Analysis
# ============================================================================
with tab3:
    st.header("Detailed Results & Analysis")
    try:
        if os.path.exists(rel("results/model_comparison.png")):
            st.subheader("Model Performance Metrics")
            st.image(rel("results/model_comparison.png"), use_container_width=True)

        if os.path.exists(rel("results/roc_curves.png")):
            st.subheader("ROC Curves Comparison")
            st.image(rel("results/roc_curves.png"), use_container_width=True)

        if os.path.exists(rel("results/confusion_matrices.png")):
            st.subheader("Confusion Matrices")
            st.image(rel("results/confusion_matrices.png"), use_container_width=True)

        st.subheader("Detailed Metrics")
        results_df = pd.read_csv(rel("results/model_comparison.csv"), index_col=0)
        st.dataframe(results_df.round(4), use_container_width=True)

        csv = results_df.to_csv()
        st.download_button(label="📥 Download Results as CSV", data=csv, file_name="model_comparison_results.csv", mime="text/csv")

    except FileNotFoundError:
        st.error("Results files not found. Please run train_models.py first.")


# ============================================================================
# TAB 4: About
# ============================================================================
with tab4:
    st.header("About This Application")
    st.subheader("📋 Project Overview")
    st.markdown(
        """
This application demonstrates the implementation and comparison of 6 different
machine learning classification models on the Breast Cancer Wisconsin dataset.

**Objective:** Classify breast cancer tumors as malignant or benign using diagnostic features.
"""
    )

    st.subheader("📊 Dataset Information")
    st.markdown(
        """
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository
- **Samples:** 569 total (455 training, 114 testing)
- **Features:** 30 diagnostic measurements
- **Classes:** Binary (Malignant/Benign)
"""
    )

    st.subheader("🤖 Models Implemented")
    models_info = {
        "Logistic Regression": "Linear probabilistic classifier",
        "Decision Tree": "Tree-based classifier with max_depth=10",
        "K-Nearest Neighbor": "Instance-based classifier with k=5",
        "Naive Bayes": "Gaussian Naive Bayes classifier",
        "Random Forest": "Ensemble with 100 trees, max_depth=15",
        "XGBoost": "Gradient boosting ensemble with 100 estimators",
    }

    for model_name, description in models_info.items():
        st.markdown(f"- **{model_name}:** {description}")

    st.subheader("📈 Evaluation Metrics")
    st.markdown(
        """
- **Accuracy:** Overall correctness of predictions
- **AUC:** Area under the ROC curve
- **Precision:** True positives / Predicted positives
- **Recall:** True positives / Actual positives
- **F1-Score:** Harmonic mean of Precision and Recall
- **MCC:** Matthews Correlation Coefficient
"""
    )

    st.divider()
    st.markdown("---")
    st.markdown("🔬 ML Assignment 2 | Breast Cancer Classification | February 2026")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Base directory for relative assets (makes paths work regardless of CWD)
BASE_DIR = Path(__file__).resolve().parent

def rel(path: str) -> str:
    return str(BASE_DIR / path)

# Configure page
st.set_page_config(
    page_title="Breast Cancer Classification Models",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏥 Breast Cancer Classification Models")
st.markdown("**Multi-Model Comparison & Prediction System**")
st.markdown("Binary classification of breast cancer tumors using 6 different ML models")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🔮 Make Predictions", "📈 Results & Analysis", "ℹ️ About"])

# ============================================================================
# TAB 1: Model Comparison
# ============================================================================
with tab1:
    st.header("Model Performance Comparison")
    
    # Load results CSV
    try:
        results_df = pd.read_csv(rel('results/model_comparison.csv'), index_col=0)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Performance Metrics Table")
            st.dataframe(results_df.round(4), use_container_width=True)
        
        with col2:
            st.subheader("Key Statistics")
            st.metric("Best Accuracy", f"{results_df['accuracy'].max():.4f}", 
                     results_df['accuracy'].idxmax())
            st.metric("Best AUC", f"{results_df['auc'].max():.4f}", 
                     results_df['auc'].idxmax())
            st.metric("Best F1-Score", f"{results_df['f1'].max():.4f}", 
                     results_df['f1'].idxmax())
        
        # Comparison charts
        st.subheader("Metric Comparison Charts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            results_df['accuracy'].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
            ax.set_ylabel('Accuracy')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0.85, 1.0])
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            results_df['auc'].plot(kind='bar', ax=ax, color='coral')
            ax.set_title('AUC Score Comparison', fontweight='bold', fontsize=12)
            ax.set_ylabel('AUC')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0.9, 1.0])
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots(figsize=(8, 5))
            results_df['f1'].plot(kind='bar', ax=ax, color='mediumseagreen')
            ax.set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
            ax.set_ylabel('F1-Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0.9, 1.0])
            st.pyplot(fig)
    
    except FileNotFoundError:
        st.error("Results file not found. Please run train_models.py first.")

# ============================================================================
# TAB 2: Make Predictions and Evaluation
# ============================================================================
with tab2:
    st.header("Make Predictions with Trained Models")

    st.markdown(
        """
    Enter feature values for a sample to get predictions from all models,
    or upload a test CSV (features only) to evaluate models and display metrics.
    """
    )

    # --- Upload test CSV (features only) for evaluation ---
    uploaded_file = st.file_uploader(
        "Upload test CSV (features only). If it contains a target column name it 'target' or 'diagnosis', it will be used.",
        type=["csv"],
    )

    def _normalize_target(y_series: pd.Series) -> pd.Series:
        try:
            if y_series.dtype == object:
                s = y_series.astype(str).str.strip().str.lower()
                mapping = {"m": 0, "malignant": 0, "b": 1, "benign": 1, "0": 0, "1": 1}
                s = s.map(lambda v: mapping.get(v, v))
                return pd.to_numeric(s, errors="coerce").astype(int)
            return pd.to_numeric(y_series, errors="coerce").astype(int)
        except Exception:
            return y_series

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Test Data (first 5 rows)")
            st.dataframe(uploaded_df.head())

            if "target" in uploaded_df.columns:
                y_true = uploaded_df["target"]
                X_eval = uploaded_df.drop(columns=["target"])
            elif "diagnosis" in uploaded_df.columns:
                y_true = uploaded_df["diagnosis"]
                X_eval = uploaded_df.drop(columns=["diagnosis"])
            else:
                # Fallback: try to load y_test from data folder
                if os.path.exists(rel("data/y_test.csv")):
                    y_true = pd.read_csv(rel("data/y_test.csv")).squeeze()
                    X_eval = uploaded_df
                else:
                    st.error(
                        "No target column in uploaded file and `data/y_test.csv` not found. Cannot evaluate."
                    )
                    X_eval = None
                    y_true = None

            if X_eval is not None:
                y_true = _normalize_target(y_true)

                model_options = [
                    "Logistic Regression",
                    "Decision Tree",
                    "KNN",
                    "Naive Bayes",
                    "Random Forest",
                    "XGBoost",
                ]

                models_selected = st.multiselect(
                    "Select model(s) to evaluate (choose one or more)",
                    model_options,
                    default=model_options,
                )

                if st.button("Evaluate on uploaded test data"):
                    try:
                        scaler = joblib.load(rel("model/scaler.pkl"))

                        # Load models
                        models_dict = {
                            "Logistic Regression": joblib.load(
                                rel("model/logistic_regression_model.pkl")
                            ),
                            "Decision Tree": joblib.load(rel("model/decision_tree_model.pkl")),
                            "KNN": joblib.load(rel("model/knn_model.pkl")),
                            "Naive Bayes": joblib.load(rel("model/naive_bayes_model.pkl")),
                            "Random Forest": joblib.load(rel("model/random_forest_model.pkl")),
                            "XGBoost": joblib.load(rel("model/xgboost_model.pkl")),
                        }

                        X_scaled = scaler.transform(X_eval)

                        eval_results = {}

                        for name, model in models_dict.items():
                            if len(models_selected) > 0 and name not in models_selected:
                                continue

                            # Try predictions on scaled data, fallback to raw
                            try:
                                y_pred = model.predict(X_scaled)
                                y_prob = model.predict_proba(X_scaled)[:, 1]
                            except Exception:
                                y_pred = model.predict(X_eval)
                                try:
                                    y_prob = model.predict_proba(X_eval)[:, 1]
                                except Exception:
                                    y_prob = np.zeros(len(y_pred))

                            acc = accuracy_score(y_true, y_pred)
                            prec = precision_score(y_true, y_pred, zero_division=0)
                            rec = recall_score(y_true, y_pred, zero_division=0)
                            f1 = f1_score(y_true, y_pred, zero_division=0)
                            try:
                                auc = roc_auc_score(y_true, y_prob)
                            except Exception:
                                auc = np.nan

                            eval_results[name] = {
                                "accuracy": acc,
                                "precision": prec,
                                "recall": rec,
                                "f1": f1,
                                "auc": auc,
                                "y_pred": y_pred,
                            }

                        # Display aggregated metrics
                        res_df = pd.DataFrame.from_dict(eval_results, orient="index").drop(
                            columns=["y_pred"]
                        )
                        st.subheader("Evaluation Metrics")
                        st.dataframe(res_df.round(4))

                        # Show confusion matrix and classification report for each evaluated model
                        for name, v in eval_results.items():
                            st.subheader(f"Confusion Matrix & Report: {name}")
                            cm = confusion_matrix(y_true, v["y_pred"])
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig)
                            st.text(classification_report(y_true, v["y_pred"], zero_division=0))

                    except FileNotFoundError as e:
                        st.error(f"Model files not found: {e}. Please ensure models are trained.")

        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    # --- Single-sample sliders and predictions (existing behavior) ---
    try:
        X_test = pd.read_csv(rel("data/X_test.csv"))
        feature_names = X_test.columns.tolist()

        st.subheader("Feature Input (single sample)")

        # Create input fields for features
        col_inputs = st.columns(3)
        user_input = {}

        for idx, feature in enumerate(feature_names):
            col_idx = idx % 3
            with col_inputs[col_idx]:
                min_val = float(X_test[feature].min())
                max_val = float(X_test[feature].max())
                mean_val = float(X_test[feature].mean())

                user_input[feature] = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100 if max_val > min_val else 0.01,
                )

        models_to_predict = st.multiselect(
            "Select model(s) for prediction (choose one or more)",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost",
            ],
            default=[
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost",
            ],
        )

        if st.button("🔮 Get Predictions", use_container_width=True):
            try:
                scaler = joblib.load(rel("model/scaler.pkl"))

                # Prepare input
                input_df = pd.DataFrame([user_input])
                input_scaled = scaler.transform(input_df)

                # Load models
                models_dict = {
                    "Logistic Regression": joblib.load(
                        rel("model/logistic_regression_model.pkl")
                    ),
                    "Decision Tree": joblib.load(rel("model/decision_tree_model.pkl")),
                    "KNN": joblib.load(rel("model/knn_model.pkl")),
                    "Naive Bayes": joblib.load(rel("model/naive_bayes_model.pkl")),
                    "Random Forest": joblib.load(rel("model/random_forest_model.pkl")),
                    "XGBoost": joblib.load(rel("model/xgboost_model.pkl")),
                }

                st.subheader("Model Predictions")

                # Display predictions in columns
                col1, col2, col3 = st.columns(3)
                columns = [col1, col2, col3]

                prediction_results = {}

                for idx, (model_name, model) in enumerate(models_dict.items()):
                    if len(models_to_predict) > 0 and model_name not in models_to_predict:
                        continue
                    col = columns[idx % 3]

                    # Make prediction
                    try:
                        prediction = model.predict(input_scaled)[0]
                        probability = model.predict_proba(input_scaled)[0]
                    except Exception:
                        prediction = model.predict(input_df)[0]
                        probability = model.predict_proba(input_df)[0]

                    prediction_results[model_name] = {
                        "prediction": prediction,
                        "malignant_prob": probability[0],
                        "benign_prob": probability[1],
                    }

                    with col:
                        st.markdown(f"### {model_name}")

                        diagnosis = "🔴 Malignant" if prediction == 0 else "🟢 Benign"
                        st.markdown(f"**Prediction:** {diagnosis}")

                        # Show probabilities
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            st.metric("Malignant", f"{probability[0]*100:.2f}%")
                        with col_prob2:
                            st.metric("Benign", f"{probability[1]*100:.2f}%")

                # Consensus prediction
                st.subheader("Model Consensus")
                predictions = [v["prediction"] for v in prediction_results.values()]
                consensus = sum(predictions) / len(predictions)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Models Predicting Benign", sum(predictions))
                with col2:
                    st.metric("Models Predicting Malignant", len(predictions) - sum(predictions))
                with col3:
                    consensus_diagnosis = "🟢 BENIGN" if consensus >= 0.5 else "🔴 MALIGNANT"
                    st.metric("Consensus", consensus_diagnosis)

            except FileNotFoundError as e:
                st.error(f"Model files not found: {e}. Please ensure all models are trained.")

    except FileNotFoundError:
        st.error("Dataset file not found. Please run load_dataset.py first.")

# ============================================================================
# TAB 3: Results & Analysis
# ============================================================================
with tab3:
    st.header("Detailed Results & Analysis")
    
    try:
        # Check if visualization files exist
        if os.path.exists(rel('results/model_comparison.png')):
            st.subheader("Model Performance Metrics")
            st.image(rel('results/model_comparison.png'), use_container_width=True)

        if os.path.exists(rel('results/roc_curves.png')):
            st.subheader("ROC Curves Comparison")
            st.image(rel('results/roc_curves.png'), use_container_width=True)

        if os.path.exists(rel('results/confusion_matrices.png')):
            st.subheader("Confusion Matrices")
            st.image(rel('results/confusion_matrices.png'), use_container_width=True)
        
        # Load and display detailed metrics
        st.subheader("Detailed Metrics")
        results_df = pd.read_csv(rel('results/model_comparison.csv'), index_col=0)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Download option
        csv = results_df.to_csv()
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
    
    except FileNotFoundError:
        st.error("Results files not found. Please run train_models.py first.")

# ============================================================================
# TAB 4: About
# ============================================================================
with tab4:
    st.header("About This Application")
    
    st.subheader("📋 Project Overview")
    st.markdown("""
    This application demonstrates the implementation and comparison of 6 different 
    machine learning classification models on the Breast Cancer Wisconsin dataset.
    
    **Objective:** Classify breast cancer tumors as malignant or benign using diagnostic features.
    """)
    
    st.subheader("📊 Dataset Information")
    st.markdown("""
    - **Dataset:** Breast Cancer Wisconsin (Diagnostic)
    - **Source:** UCI Machine Learning Repository
    - **Samples:** 569 total (455 training, 114 testing)
    - **Features:** 30 diagnostic measurements
    - **Classes:** Binary (Malignant/Benign)
    """)
    
    st.subheader("🤖 Models Implemented")
    models_info = {
        "Logistic Regression": "Linear probabilistic classifier",
        "Decision Tree": "Tree-based classifier with max_depth=10",
        "K-Nearest Neighbor": "Instance-based classifier with k=5",
        "Naive Bayes": "Gaussian Naive Bayes classifier",
        "Random Forest": "Ensemble with 100 trees, max_depth=15",
        "XGBoost": "Gradient boosting ensemble with 100 estimators"
    }
    
    for model_name, description in models_info.items():
        st.markdown(f"- **{model_name}:** {description}")
    
    st.subheader("📈 Evaluation Metrics")
    st.markdown("""
    - **Accuracy:** Overall correctness of predictions
    - **AUC:** Area under the ROC curve
    - **Precision:** True positives / Predicted positives
    - **Recall:** True positives / Actual positives
    - **F1-Score:** Harmonic mean of Precision and Recall
    - **MCC:** Matthews Correlation Coefficient
    """)
    
    st.subheader("🏆 Best Model")
    st.markdown("""
    **Logistic Regression** achieved the best overall performance with:
    - Accuracy: 0.9825
    - Precision: 0.9861
    - Recall: 0.9861
    - F1-Score: 0.9861
    - AUC: 0.9954
    - MCC: 0.9623
    """)
    
    st.subheader("👨‍💻 Technologies Used")
    st.markdown("""
    - **Python 3.8+**
    - **Scikit-learn:** ML algorithms
    - **XGBoost:** Gradient boosting
    - **Streamlit:** Web application framework
    - **Pandas:** Data manipulation
    - **Matplotlib & Seaborn:** Visualization
    """)
    
    st.divider()
    st.markdown("---")
    st.markdown("🔬 ML Assignment 2 | Breast Cancer Classification | February 2026")

