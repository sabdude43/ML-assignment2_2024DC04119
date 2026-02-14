# ML Assignment 2: Classification Models Comparison

## A. Problem Statement

The objective of this assignment is to implement and evaluate multiple classification algorithms on a real-world medical dataset. Specifically, we aim to develop machine learning models that can accurately classify breast cancer tumors as either malignant or benign based on various diagnostic features.

The problem is a **binary classification task** where we need to:
- Learn patterns from diagnostic measurements of breast cancer cells
- Predict whether a tumor is malignant (0) or benign (1)
- Compare the performance of different classification algorithms
- Identify the best-performing model based on multiple evaluation metrics

This is a critical healthcare application where accurate classification can assist in early detection and treatment planning for breast cancer patients.

---

## B. Dataset Description

### Dataset: Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository / sklearn.datasets

**Characteristics:**
- **Total Instances:** 569 samples
- **Total Features:** 30 numeric features
- **Target Variable:** Diagnosis (Binary Classification)
  - Class 0: Malignant (212 samples, 37.3%)
  - Class 1: Benign (357 samples, 62.7%)
- **Data Split:** 
  - Training Set: 455 samples (80%)
  - Testing Set: 114 samples (20%)

**Feature Description:**

The 30 features are computed from a digitized image of a fine needle aspirate (FNA) of breast mass. Each category of measurements includes:

1. **Mean values** (10 features): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
2. **Standard Error values** (10 features): measurements of the variability of the above features
3. **Worst (largest) values** (10 features): the largest measurement of each feature

All features are numeric and have been standardized for algorithms that require feature scaling.

**Dataset Quality:**
- No missing values
- Well-balanced classes (Benign > Malignant)
- Suitable for training and testing ML algorithms
- Appropriate feature count and instance size for the assignment requirements (min 12 features, min 500 instances)

---

## C. Models Used and Comparison

Six classification models have been implemented and evaluated on the Breast Cancer Wisconsin dataset. Each model has been trained on the same standardized training data and evaluated on the same test set to ensure fair comparison.

### Models Implemented:

1. **Logistic Regression** - Linear probabilistic classifier
2. **Decision Tree Classifier** - Tree-based classifier with max_depth=10
3. **K-Nearest Neighbor (KNN)** - Instance-based classifier with k=5
4. **Naive Bayes Classifier** - Gaussian Naive Bayes probabilistic classifier
5. **Random Forest** - Ensemble method with 100 trees and max_depth=15
6. **XGBoost** - Gradient boosting ensemble with 100 estimators

### Evaluation Metrics:

- **Accuracy:** Proportion of correct predictions among total predictions
- **AUC (Area Under Curve):** Measure of the classifier's ability to distinguish between classes
- **Precision:** Proportion of true positives among predicted positives
- **Recall:** Proportion of true positives among actual positives (Sensitivity)
- **F1-Score:** Harmonic mean of Precision and Recall
- **MCC (Matthews Correlation Coefficient):** Correlation coefficient between predicted and observed classifications

### Model Comparison Table:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9474 | 0.9940 | 0.9459 | 0.9722 | 0.9589 | 0.8864 |

### Key Findings:

**Best Overall Performance: Logistic Regression**
- Achieved the highest scores in Accuracy (0.9825), Precision (0.9861), Recall (0.9861), F1-Score (0.9861), and MCC (0.9623)
- Demonstrates exceptional balanced performance across all metrics
- Most reliable for this binary classification task

**Second Best: Random Forest**
- Excellent AUC score (0.9937) - second highest
- Tied with KNN for F1-Score (0.9655)
- Strong ensemble performance

**Third Best: XGBoost**
- Highest AUC among non-Logistic Regression models (0.9940)
- Strong recall (0.9722) - excellent at identifying positive cases
- Competitive ensemble method

**Performance Summary:**
- All models achieved high accuracy (>91%), indicating good generalization
- Logistic Regression's simplicity and interpretability, combined with superior performance, makes it the recommended model
- Ensemble methods (Random Forest, XGBoost) show strong but slightly lower performance than Logistic Regression
- Tree-based methods (Decision Tree) show lower performance, likely due to overfitting

---

## D. Model Performance Observations

### Detailed Analysis of Each Model:

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | **Exceptional Performance:** Logistic Regression achieved the highest performance across most metrics (Accuracy: 0.9825, Precision: 0.9861, Recall: 0.9861, F1: 0.9861, MCC: 0.9623). This demonstrates that the breast cancer classification problem is largely linearly separable in the feature space. The model's probabilistic nature and excellent calibration (AUC: 0.9954) make it highly reliable for medical applications. The balanced precision-recall indicates no bias toward false positives or false negatives, which is critical in healthcare diagnostics. Its simplicity and interpretability provide additional advantages for clinical deployment. |
| **Decision Tree Classifier** | **Moderate Performance with Overfitting Concerns:** The Decision Tree achieved an accuracy of 0.9123 with the lowest MCC score (0.8174) among all models. The relatively lower recall (0.9028) compared to other models suggests the tree may be making conservative predictions and missing some positive cases. The model's lower AUC (0.9157) indicates less effective probability calibration. Despite pruning (max_depth=10), the model shows signs of overfitting, capturing noise in the training data. Tree-based models' sensitivity to feature interactions and inability to capture linear relationships in this dataset limits their effectiveness. The lower performance suggests this problem is better suited to linear and ensemble approaches. |
| **kNN** | **Strong Performance with Balanced Metrics:** K-Nearest Neighbor achieved excellent and balanced results (Accuracy: 0.9561, F1: 0.9655, MCC: 0.9054). The model's high recall (0.9722) indicates it effectively identifies positive cases (benign tumors), making it reliable for screening applications. The consistent precision (0.9589) ensures minimal false alarms. The strong AUC (0.9788) demonstrates good class separation. KNN's success suggests the dataset has clear local patterns where similar samples tend to belong to the same class. However, KNN's dependence on feature scaling and computational complexity for large datasets are potential limitations. The model's instance-based learning approach captures the underlying data distribution effectively without making strong assumptions. |
| **Naive Bayes Classifier** | **Competitive Performance with Probabilistic Strength:** Naive Bayes achieved solid results (Accuracy: 0.9298) with notably the highest AUC score (0.9868), indicating excellent probability calibration and discriminative ability despite strong independence assumptions. The balanced precision and recall (both 0.9444) demonstrate no bias in predictions. However, the lower MCC (0.8492) suggests the model's confusion matrix has less optimal characteristics compared to other methods. The high AUC despite lower accuracy indicates Naive Bayes generates well-calibrated probabilities, making it valuable for ranking predictions. The model's assumption of conditional independence between features is reasonably valid for this dataset. Its computational efficiency and robustness make it suitable for streaming and real-time applications. |
| **Random Forest (Ensemble)** | **Excellent Ensemble Performance:** Random Forest delivered outstanding results matching KNN in many metrics (Accuracy: 0.9561, F1: 0.9655, MCC: 0.9054) while achieving the highest AUC among ensemble methods (0.9937). The high recall (0.9722) ensures excellent detection of benign tumors. The ensemble's ability to capture non-linear relationships and handle feature interactions effectively is demonstrated by its superior AUC. The model shows robust generalization without severe overfitting. Random Forest's feature importance analysis could provide clinical insights into which diagnostic measurements are most predictive. The ensemble approach's noise reduction through bagging improves stability and reliability. Its only drawback compared to Logistic Regression is slightly lower overall accuracy and interpretability of individual predictions. |
| **XGBoost (Ensemble)** | **Strong Gradient Boosting Performance:** XGBoost achieved competitive results (Accuracy: 0.9474, AUC: 0.9940) with exceptional recall (0.9722), making it excellent at identifying positive cases. The near-perfect AUC (0.9940) is the second-highest and indicates outstanding probability calibration. However, the lower MCC (0.8864) and precision (0.9459) suggest slightly more false positives compared to other models. XGBoost's sequential boosting strategy effectively corrects prediction errors from previous iterations, capturing complex patterns in the data. The model's computational efficiency during training and prediction is superior to Random Forest. The trade-off between precision and recall suggests XGBoost is optimized for recall, making it suitable for applications where missing positive cases is costly. Its ability to handle feature interactions and non-linearity makes it versatile for complex medical datasets. |

### Summary of Findings:

1. **Logistic Regression** stands out as the optimal choice with superior balanced performance across all metrics
2. **Ensemble methods** (Random Forest, XGBoost) demonstrate strong capability in handling complex patterns but with slightly lower overall accuracy
3. **Instance-based methods** (KNN) perform well by capturing local neighborhood patterns effectively
4. **Probabilistic models** (Naive Bayes) excel in probability calibration (AUC) despite lower overall accuracy
5. **Tree-based single models** (Decision Tree) underperform due to overfitting and inability to capture linear relationships in this dataset

The dataset's linear separability, well-behaved feature distributions, and moderate dimensionality favor simpler, more direct approaches like Logistic Regression over complex ensemble methods in this case.

---

## Installation and Setup

### Requirements:
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation:

```bash
pip install -r requirements.txt
```

### Running the Application:

```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## Deployment on Streamlit Community Cloud

This application has been designed for easy deployment on Streamlit Community Cloud.

### Features of the Deployed App:

📊 **Model Comparison Tab**
- Performance metrics table for all 6 models
- Comparative visualization charts
- Key statistics and rankings

🔮 **Predictions Tab**
- Interactive feature input sliders
- Real-time predictions from all 6 models
- Probability distributions
- Model consensus voting

📈 **Results & Analysis Tab**
- Model performance visualizations
- ROC curve comparisons
- Confusion matrices
- Downloadable results CSV

ℹ️ **About Tab**
- Complete project information
- Dataset details
- Model descriptions
- Technology stack

### Deployment Steps:

1. **Push to GitHub:** Ensure all files are committed and pushed to your repository
2. **Visit:** https://streamlit.io/cloud
3. **Sign In:** Use your GitHub account
4. **New App:** Click "New App" and select your repository
5. **Configure:** Choose `project-folder/streamlit_app.py` as the main file
6. **Deploy:** Click Deploy and wait for completion

**For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

---

## Project Structure

```
project-folder/
├── streamlit_app.py          # Streamlit web application
├── train_models.py           # Script to train all models
├── load_dataset.py           # Script to load and prepare dataset
├── requirements.txt          # Package dependencies
├── README.md                 # This file
├── data/                     # Data directory
│   ├── breast_cancer_dataset.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── model/                    # Trained models
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── scaler.pkl
└── results/                  # Evaluation results and visualizations
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── roc_curves.png
    └── confusion_matrices.png
```

---

## Implementation Details

### Data Preprocessing:
- Features standardized using StandardScaler for distance-based and linear algorithms
- Train-test split: 80-20 stratified split
- No missing values or outlier removal required

### Model Training:
- All models trained with consistent random seeds (random_state=42)
- Hyperparameters tuned for optimal performance
- Feature scaling applied where necessary

### Evaluation:
- Comprehensive evaluation using 6 different metrics
- Confusion matrices and ROC curves generated for each model
- Results saved in CSV format for further analysis

---

## Conclusions

This assignment successfully demonstrates the implementation and comparison of six different classification algorithms on a real-world medical dataset. The results clearly show that:

1. **Logistic Regression** is the best-performing model for this binary classification problem
2. Ensemble methods show competitive performance but not superior to the baseline
3. All models achieve excellent results (accuracy >91%), indicating the problem's suitability for ML
4. Model selection should consider both performance metrics and practical interpretability

The trained models and evaluation metrics provide a solid foundation for medical decision support systems in breast cancer diagnosis.
