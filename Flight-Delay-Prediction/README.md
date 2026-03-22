# Flight Delay Prediction

## Project Overview
Binary classification model to predict whether a flight will be delayed or on time based on airline, departure time, day of week, origin airport, destination airport and flight duration.

**Type:** Binary Classification  
**Dataset:** Airlines Dataset to Predict a Delay — Jim Schacko (Kaggle)  
**Tool:** Python, Jupyter Notebook  

---

## Problem Statement
Flight delays cause significant disruption for passengers and operational costs for airlines. This model predicts whether a flight will be delayed, allowing airlines to proactively manage resources and inform passengers in advance.

---

## Dataset
- **Rows:** 539,383
- **Columns:** 9
- **Target:** Delay (1 = Delayed, 0 = On Time)
- **Class Distribution:** Imbalanced — more on-time flights than delayed

---

## ML Pipeline

| Step | Action |
|---|---|
| Data Inspection | Shape, dtypes, nulls, duplicates, describe |
| Missing Values | None found |
| Dropped Columns | id — no predictive value |
| EDA | Correlation heatmap — Time shows weak positive correlation with Delay |
| Outlier Treatment | Capped Length column using IQR method |
| Feature Engineering | Created IsWeekend (from DayOfWeek) and TimeOfDay (from Time) |
| Encoding | One Hot Encoding for Airline, AirportFrom, AirportTo, TimeOfDay |
| Scaling | StandardScaler on Flight, DayOfWeek, Time, Length |
| Class Imbalance | SMOTE applied on training data only |
| Train/Test Split | 80/20, random_state=42 |

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.64 | 0.60 | 0.61 | **0.60** |
| Decision Tree | 0.61 | 0.57 | 0.48 | 0.52 |
| Random Forest | 0.62 | 0.57 | 0.57 | 0.57 |

**Winner: Logistic Regression** — Best F1 Score and Recall

---

## Key Findings
- No strong correlations found — dataset lacks strong predictive signal
- Time of day has weak positive correlation with delay (0.15)
- Day of week has almost no correlation with delay (-0.026)
- Hyperparameter tuning: Best params C=10, solver=liblinear
- Tuning did not improve test F1 — data quality is the bottleneck

---

## Key Learnings
- Always identify issues during inspection — fix in later steps
- Outlier treatment decision depends on whether data is real or corrupt
- Feature engineering requires domain knowledge — think real world context
- Class imbalance must be addressed — accuracy alone is misleading
- F1 Score is the right metric for imbalanced classification problems

---

## Tech Stack
Python | Pandas | NumPy | Scikit-learn | Seaborn | Matplotlib | Imbalanced-learn | Jupyter Notebook

---

## Next Improvements
- Add weather data as features — strongest predictor of flight delays
- Add historical delay rate per airline and airport
- Try XGBoost and LightGBM
- Use Optuna for hyperparameter tuning

