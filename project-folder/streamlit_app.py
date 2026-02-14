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
        results_df = pd.read_csv('results/model_comparison.csv', index_col=0)
        
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
# TAB 2: Make Predictions
# ============================================================================
with tab2:
    st.header("Make Predictions with Trained Models")
    
    st.markdown("""
    Enter feature values for a sample to get predictions from all 6 models.
    The feature values should be normalized diagnostic measurements from breast cancer imaging.
    """)
    
    # Load sample data to show feature names and ranges
    try:
        X_test = pd.read_csv('data/X_test.csv')
        feature_names = X_test.columns.tolist()
        
        st.subheader("Feature Input")
        
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
                    step=(max_val - min_val) / 100
                )
        
        if st.button("🔮 Get Predictions from All Models", use_container_width=True):
            # Load scaler and models
            try:
                scaler = joblib.load('model/scaler.pkl')
                
                # Prepare input
                input_df = pd.DataFrame([user_input])
                input_scaled = scaler.transform(input_df)
                
                # Load models
                models_dict = {
                    'Logistic Regression': joblib.load('model/logistic_regression_model.pkl'),
                    'Decision Tree': joblib.load('model/decision_tree_model.pkl'),
                    'KNN': joblib.load('model/knn_model.pkl'),
                    'Naive Bayes': joblib.load('model/naive_bayes_model.pkl'),
                    'Random Forest': joblib.load('model/random_forest_model.pkl'),
                    'XGBoost': joblib.load('model/xgboost_model.pkl'),
                }
                
                st.subheader("Model Predictions")
                
                # Display predictions in columns
                col1, col2, col3 = st.columns(3)
                columns = [col1, col2, col3]
                
                prediction_results = {}
                
                for idx, (model_name, model) in enumerate(models_dict.items()):
                    col = columns[idx % 3]
                    
                    # Make prediction
                    if model_name in ['Logistic Regression', 'KNN', 'Naive Bayes']:
                        prediction = model.predict(input_scaled)[0]
                        probability = model.predict_proba(input_scaled)[0]
                    else:
                        prediction = model.predict(input_df)[0]
                        probability = model.predict_proba(input_df)[0]
                    
                    prediction_results[model_name] = {
                        'prediction': prediction,
                        'malignant_prob': probability[0],
                        'benign_prob': probability[1]
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
                predictions = [v['prediction'] for v in prediction_results.values()]
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
        if os.path.exists('results/model_comparison.png'):
            st.subheader("Model Performance Metrics")
            st.image('results/model_comparison.png', use_container_width=True)
        
        if os.path.exists('results/roc_curves.png'):
            st.subheader("ROC Curves Comparison")
            st.image('results/roc_curves.png', use_container_width=True)
        
        if os.path.exists('results/confusion_matrices.png'):
            st.subheader("Confusion Matrices")
            st.image('results/confusion_matrices.png', use_container_width=True)
        
        # Load and display detailed metrics
        st.subheader("Detailed Metrics")
        results_df = pd.read_csv('results/model_comparison.csv', index_col=0)
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

