import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent

def rel(p: str) -> str:
    return str(BASE_DIR / p)

st.set_page_config(page_title="Breast Cancer Models", layout="wide")
st.title("Breast Cancer Classification Models")

# Minimal tabs
tab1, tab2, tab3, tab4 = st.tabs(["Models", "Predict", "Results", "About"])

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}

def load_models(names=None):
    names = names or list(MODEL_FILES.keys())
    models = {}
    for n in names:
        p = rel(f"model/{MODEL_FILES[n]}")
        if os.path.exists(p):
            try:
                models[n] = joblib.load(p)
            except Exception:
                models[n] = None
        else:
            models[n] = None
    return models

with tab1:
    st.header("Model Comparison")
    path = rel("results/model_comparison.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        sel = st.multiselect("Filter models", df.index.tolist(), default=df.index.tolist())
        df_show = df.loc[sel] if sel else df
        st.dataframe(df_show.round(4))
    else:
        st.error("Run train_models.py to generate results/ files")

with tab2:
    st.header("Predict / Evaluate")
    uploaded = st.file_uploader("Upload CSV (features + optional target)", type=["csv"])
    model_names = list(MODEL_FILES.keys())
    selected = st.multiselect("Select model(s)", model_names, default=model_names)

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.dataframe(df_up.head())
        if 'target' in df_up.columns:
            y = df_up['target']
            X = df_up.drop(columns=['target'])
        elif 'diagnosis' in df_up.columns:
            y = df_up['diagnosis']
            X = df_up.drop(columns=['diagnosis'])
        else:
            y = None
            X = df_up

        if X is not None and y is not None:
            models = load_models(selected)
            out = {}
            for name, m in models.items():
                if m is None or name not in selected:
                    continue
                try:
                    yp = m.predict(X.values)
                except Exception:
                    yp = m.predict(X)
                try:
                    prob = m.predict_proba(X.values)[:, 1]
                except Exception:
                    prob = np.zeros(len(yp))
                out[name] = {
                    'accuracy': accuracy_score(y, yp),
                    'precision': precision_score(y, yp, zero_division=0),
                    'recall': recall_score(y, yp, zero_division=0),
                    'f1': f1_score(y, yp, zero_division=0),
                    'auc': roc_auc_score(y, prob) if len(prob) == len(yp) else np.nan,
                    'y_pred': yp,
                }
            if out:
                st.dataframe(pd.DataFrame(out).T.round(4))
                for k, v in out.items():
                    st.write(f"Confusion Matrix: {k}")
                    cm = confusion_matrix(y, v['y_pred'])
                    st.write(cm)
                    st.text(classification_report(y, v['y_pred'], zero_division=0))
        else:
            st.info('Provide target column named "target" or "diagnosis" in upload')

with tab3:
    st.header('Results & Analysis')
    img = rel('results/model_comparison.png')
    if os.path.exists(img):
        st.image(img)
    else:
        st.info('No result images found')

with tab4:
    st.header('About')
    st.write('Upload a CSV to evaluate models or use single-sample input in the Predict tab.')

