# Streamlit Cloud Deployment Guide

## Prerequisites

Before deploying to Streamlit Community Cloud, ensure you have:

1. ✅ A GitHub account
2. ✅ The project repository pushed to GitHub
3. ✅ All trained models in the `model/` directory
4. ✅ All results in the `results/` directory
5. ✅ All data in the `data/` directory

## Step-by-Step Deployment Instructions

### Step 1: Prepare Your Repository

Ensure your GitHub repository contains:

```
project-folder/
├── streamlit_app.py          # Main application file
├── train_models.py           # Model training script
├── load_dataset.py           # Dataset loading script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Data files
│   ├── breast_cancer_dataset.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── model/                    # Trained model files
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── scaler.pkl
└── results/                  # Results and visualizations
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── roc_curves.png
    └── confusion_matrices.png
```

### Step 2: Push to GitHub

```bash
cd project-folder
git add .
git commit -m "Add Streamlit app and trained models for deployment"
git push origin main
```

### Step 3: Deploy to Streamlit Community Cloud

1. **Go to Streamlit Cloud:** https://streamlit.io/cloud

2. **Sign In:** 
   - Click "Sign in"
   - Authenticate with your GitHub account
   - Grant necessary permissions

3. **Create New App:**
   - Click "New App" button
   - Select "From existing repo"

4. **Configure Deployment:**
   - **Repository:** Select your GitHub repository (ML-assignment2_2024DC04119)
   - **Branch:** Select `main` (or your working branch)
   - **Main File Path:** Enter `project-folder/streamlit_app.py`

5. **Deploy:**
   - Click "Deploy"
   - Wait for the deployment to complete (usually 2-3 minutes)

6. **Access Your App:**
   - Once deployed, you'll receive a unique URL like: `https://[your-app-name].streamlit.app`
   - Share this URL with others

## App Features

Once deployed, your Streamlit app will have the following features:

### 📊 Tab 1: Model Comparison
- View performance metrics table for all 6 models
- See comparison charts for Accuracy, AUC, and F1-Score
- Display key statistics

### 🔮 Tab 2: Make Predictions
- Interactive sliders to input feature values
- Get predictions from all 6 trained models
- View probabilities for each prediction
- See model consensus

### 📈 Tab 3: Results & Analysis
- View model performance visualizations
- Display ROC curves comparison
- Show confusion matrices for all models
- Download results as CSV

### ℹ️ Tab 4: About
- Project overview
- Dataset information
- Models description
- Evaluation metrics explanation
- Best model summary
- Technologies used

## Troubleshooting

### Issue: "File not found" errors

**Solution:** Ensure all data, model, and results files are committed and pushed to GitHub.

```bash
git status  # Check if files are tracked
git add .
git push
```

### Issue: "Module not found" errors

**Solution:** Verify `requirements.txt` contains all necessary packages:

```
streamlit>=1.28.0
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
matplotlib>=3.7.2
seaborn>=0.12.2
xgboost>=2.0.0
joblib>=1.3.0
```

### Issue: App runs locally but fails in cloud

**Solution:** 
1. Check Streamlit logs: Click "Manage app" → "View logs"
2. Ensure all file paths are relative (not absolute)
3. Verify data directory structure matches locally

## Configuration Tips

### Increase Upload Limit

For larger model files, add a `.streamlit/config.toml` file:

```toml
[client]
maxUploadSize = 200

[logger]
level = "info"
```

### Add Custom Secrets

For sensitive data, use Streamlit Secrets:
1. Go to app settings on Streamlit Cloud
2. Click "Secrets"
3. Add key-value pairs

Access in app:
```python
import streamlit as st
secret_value = st.secrets["key_name"]
```

## Performance Optimization

### For Large Model Files
- Consider using model compression
- Upload pre-trained models to cloud storage
- Load models conditionally with caching

### Streamlit Caching

Add caching to improve performance:

```python
@st.cache_resource
def load_model(model_name):
    return joblib.load(f'model/{model_name}_model.pkl')

@st.cache_data
def load_results():
    return pd.read_csv('results/model_comparison.csv')
```

## Monitoring Your Deployment

1. **View Logs:** 
   - App dashboard → Manage app → View logs
   - Check for errors and warnings

2. **Monitor Resources:**
   - Community Cloud has resource limits
   - Monitor memory and CPU usage

3. **Update Your App:**
   - Push changes to GitHub
   - Streamlit automatically redeploys

## Sharing Your App

Once deployed, share the URL:
- Direct link: `https://[your-app-name].streamlit.app`
- Embed in portfolio: Add to LinkedIn, GitHub profile
- Share with classmates/instructors

## Keeping App Live

Streamlit Community Cloud keeps free apps running as long as:
- They aren't inactive for more than 7 days
- They don't exceed 1GB of RAM
- They don't have more than 1 active user per month on average

## Additional Resources

- **Streamlit Docs:** https://docs.streamlit.io/
- **Deployment Guide:** https://docs.streamlit.io/deploy/streamlit-community-cloud
- **Troubleshooting:** https://docs.streamlit.io/deploy/troubleshooting

## Support

For issues or questions:
- Check Streamlit documentation
- Visit Streamlit Community forum
- Review app logs in Streamlit Cloud dashboard

---

**Happy Deploying! 🚀**
