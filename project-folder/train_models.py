"""
Train 6 Classification Models on Breast Cancer Dataset
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load preprocessed data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Testing set size: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Standardize features (important for many algorithms)
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved to model/scaler.pkl")

# Dictionary to store models and results
models = {}
results = {}

# 1. LOGISTIC REGRESSION
print("\n" + "=" * 80)
print("1. LOGISTIC REGRESSION")
print("=" * 80)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
lr_mcc = matthews_corrcoef(y_test, y_pred_lr)

models['Logistic Regression'] = lr_model
results['Logistic Regression'] = {
    'accuracy': lr_accuracy,
    'precision': lr_precision,
    'recall': lr_recall,
    'f1': lr_f1,
    'auc': lr_auc,
    'mcc': lr_mcc,
    'y_pred': y_pred_lr,
    'y_pred_proba': y_pred_proba_lr
}

print(f"Accuracy:  {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")
print(f"F1-Score:  {lr_f1:.4f}")
print(f"ROC-AUC:   {lr_auc:.4f}")
print(f"MCC Score: {lr_mcc:.4f}")

# 2. DECISION TREE CLASSIFIER
print("\n" + "=" * 80)
print("2. DECISION TREE CLASSIFIER")
print("=" * 80)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)  # Decision Tree doesn't require scaling
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)
dt_auc = roc_auc_score(y_test, y_pred_proba_dt)
dt_mcc = matthews_corrcoef(y_test, y_pred_dt)

models['Decision Tree'] = dt_model
results['Decision Tree'] = {
    'accuracy': dt_accuracy,
    'precision': dt_precision,
    'recall': dt_recall,
    'f1': dt_f1,
    'auc': dt_auc,
    'mcc': dt_mcc,
    'y_pred': y_pred_dt,
    'y_pred_proba': y_pred_proba_dt
}

print(f"Accuracy:  {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall:    {dt_recall:.4f}")
print(f"F1-Score:  {dt_f1:.4f}")
print(f"ROC-AUC:   {dt_auc:.4f}")
print(f"MCC Score: {dt_mcc:.4f}")

# 3. K-NEAREST NEIGHBOR CLASSIFIER
print("\n" + "=" * 80)
print("3. K-NEAREST NEIGHBOR CLASSIFIER")
print("=" * 80)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)
knn_auc = roc_auc_score(y_test, y_pred_proba_knn)
knn_mcc = matthews_corrcoef(y_test, y_pred_knn)

models['KNN'] = knn_model
results['KNN'] = {
    'accuracy': knn_accuracy,
    'precision': knn_precision,
    'recall': knn_recall,
    'f1': knn_f1,
    'auc': knn_auc,
    'mcc': knn_mcc,
    'y_pred': y_pred_knn,
    'y_pred_proba': y_pred_proba_knn
}

print(f"Accuracy:  {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall:    {knn_recall:.4f}")
print(f"F1-Score:  {knn_f1:.4f}")
print(f"ROC-AUC:   {knn_auc:.4f}")
print(f"MCC Score: {knn_mcc:.4f}")

# 4. GAUSSIAN NAIVE BAYES CLASSIFIER
print("\n" + "=" * 80)
print("4. GAUSSIAN NAIVE BAYES CLASSIFIER")
print("=" * 80)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)
y_pred_proba_nb = nb_model.predict_proba(X_test_scaled)[:, 1]

nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)
nb_auc = roc_auc_score(y_test, y_pred_proba_nb)
nb_mcc = matthews_corrcoef(y_test, y_pred_nb)

models['Naive Bayes'] = nb_model
results['Naive Bayes'] = {
    'accuracy': nb_accuracy,
    'precision': nb_precision,
    'recall': nb_recall,
    'f1': nb_f1,
    'auc': nb_auc,
    'mcc': nb_mcc,
    'y_pred': y_pred_nb,
    'y_pred_proba': y_pred_proba_nb
}

print(f"Accuracy:  {nb_accuracy:.4f}")
print(f"Precision: {nb_precision:.4f}")
print(f"Recall:    {nb_recall:.4f}")
print(f"F1-Score:  {nb_f1:.4f}")
print(f"ROC-AUC:   {nb_auc:.4f}")
print(f"MCC Score: {nb_mcc:.4f}")

# 5. RANDOM FOREST CLASSIFIER
print("\n" + "=" * 80)
print("5. RANDOM FOREST CLASSIFIER")
print("=" * 80)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
rf_mcc = matthews_corrcoef(y_test, y_pred_rf)

models['Random Forest'] = rf_model
results['Random Forest'] = {
    'accuracy': rf_accuracy,
    'precision': rf_precision,
    'recall': rf_recall,
    'f1': rf_f1,
    'auc': rf_auc,
    'mcc': rf_mcc,
    'y_pred': y_pred_rf,
    'y_pred_proba': y_pred_proba_rf
}

print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_auc:.4f}")
print(f"MCC Score: {rf_mcc:.4f}")

# 6. XGBOOST CLASSIFIER
print("\n" + "=" * 80)
print("6. XGBOOST CLASSIFIER")
print("=" * 80)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
xgb_mcc = matthews_corrcoef(y_test, y_pred_xgb)

models['XGBoost'] = xgb_model
results['XGBoost'] = {
    'accuracy': xgb_accuracy,
    'precision': xgb_precision,
    'recall': xgb_recall,
    'f1': xgb_f1,
    'auc': xgb_auc,
    'mcc': xgb_mcc,
    'y_pred': y_pred_xgb,
    'y_pred_proba': y_pred_proba_xgb
}

print(f"Accuracy:  {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall:    {xgb_recall:.4f}")
print(f"F1-Score:  {xgb_f1:.4f}")
print(f"ROC-AUC:   {xgb_auc:.4f}")
print(f"MCC Score: {xgb_mcc:.4f}")

# Save all models
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

for model_name, model in models.items():
    filepath = f'model/{model_name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(model, filepath)
    print(f"Saved: {filepath}")

# Create comprehensive results summary
print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame(results).T
summary_df = summary_df[['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']]
print("\n", summary_df.round(4))

# Save summary to CSV
summary_df.to_csv('results/model_comparison.csv')
print("\nResults saved to results/model_comparison.csv")

# Find best models
print("\n" + "=" * 80)
print("BEST MODELS")
print("=" * 80)
print(f"Best Accuracy:  {summary_df['accuracy'].idxmax()} ({summary_df['accuracy'].max():.4f})")
print(f"Best Precision: {summary_df['precision'].idxmax()} ({summary_df['precision'].max():.4f})")
print(f"Best Recall:    {summary_df['recall'].idxmax()} ({summary_df['recall'].max():.4f})")
print(f"Best F1-Score:  {summary_df['f1'].idxmax()} ({summary_df['f1'].max():.4f})")
print(f"Best ROC-AUC:   {summary_df['auc'].idxmax()} ({summary_df['auc'].max():.4f})")
print(f"Best MCC Score: {summary_df['mcc'].idxmax()} ({summary_df['mcc'].max():.4f})")

# Generate visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

for metric, pos in zip(metrics, positions):
    ax = axes[pos]
    summary_df[metric].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.85, 1.0])

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/model_comparison.png")

# Remove extra subplot logic removed since we now have 6 subplots

# 2. ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for (model_name, model_results), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, model_results['y_pred_proba'])
    auc_score = model_results['auc']
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2.5, color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: results/roc_curves.png")

# 3. Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')

model_list = list(results.keys())
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

for (model_name, model_results), pos in zip(results.items(), positions):
    ax = axes[pos]
    cm = confusion_matrix(y_test, model_results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{model_name}\n(Accuracy: {model_results["accuracy"]:.4f})', fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: results/confusion_matrices.png")

# Generate detailed classification reports
print("\nDetailed Classification Reports:")
for model_name, model_results in results.items():
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, model_results['y_pred'], 
                              target_names=['Malignant (0)', 'Benign (1)']))

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\nFiles saved:")
print("  Models: model/*.pkl")
print("  Results: results/model_comparison.csv")
print("  Visualizations: results/*.png")
