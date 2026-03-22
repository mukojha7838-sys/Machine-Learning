
# Telco Customer Churn Prediction

## Project Overview
Binary classification model to predict whether a telecom customer will churn based on their service usage, contract type, billing and demographic information.

**Type:** Binary Classification  
**Dataset:** Telco Customer Churn — BlastChar (Kaggle)  
**Tool:** Python, Google Colab, Jupyter Notebook  

---

## Problem Statement
Customer churn is one of the most critical business problems for telecom companies. Acquiring a new customer costs 5-7x more than retaining an existing one. This model identifies at-risk customers so the business can proactively offer retention incentives.

---

## Dataset
- **Rows:** 7,043
- **Columns:** 21
- **Target:** Churn (Yes = 1, No = 0)
- **Class Distribution:** No Churn ~73%, Churn ~27%

---

## ML Pipeline

| Step | Action |
|---|---|
| Data Inspection | Shape, dtypes, nulls, duplicates, describe |
| Missing Values | Found hidden blank strings in TotalCharges — filled with median |
| Dropped Columns | customerID, gender, PhoneService — no predictive value |
| EDA | Correlation heatmap — tenure is strongest predictor of churn |
| Encoding | Binary mapping (Yes/No → 1/0), One Hot for Contract & PaymentMethod |
| Scaling | StandardScaler on tenure, MonthlyCharges, TotalCharges |
| Class Imbalance | SMOTE applied on training data only |
| Train/Test Split | 80/20, random_state=42 |

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.77 | 0.55 | 0.76 | **0.64** |
| Decision Tree | 0.72 | 0.48 | 0.59 | 0.53 |
| Random Forest | 0.78 | 0.58 | 0.63 | 0.60 |

**Winner: Logistic Regression** — Best F1 Score and highest Recall (0.76)

---

## Key Findings
- **tenure** is the strongest predictor — longer customers are less likely to churn
- **MonthlyCharges** and **PaperlessBilling** positively correlate with churn
- **Partner** and **Dependents** negatively correlate — settled customers stay longer
- Hyperparameter tuning: Best params C=10, solver=lbfgs, max_iter=500

---

## Key Learnings
- `isnull()` does not catch hidden blank strings — always verify with `value_counts()`
- Recall matters more than Precision for churn prediction
- Dropping irrelevant columns improves model clarity
- 3-value columns (No/Yes/No internet service) can be simplified to binary

---

## Tech Stack
Python | Pandas | NumPy | Scikit-learn | Seaborn | Matplotlib | Imbalanced-learn | Google Colab

---

## Next Improvements
- Try XGBoost and LightGBM
- Use Optuna for hyperparameter tuning
- Add Feature Importance analysis
- Deploy as web app using Streamlit
