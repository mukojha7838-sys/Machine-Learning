# Car Sales Prediction

## Project Overview
A regression model built using a Neural Network (Deep Learning) to predict the car purchase amount a customer is likely to spend based on their demographic and financial information.

**Type:** Regression  
**Dataset:** Car Purchasing Data  
**Tool:** Python, Jupyter Notebook, TensorFlow/Keras  

---

## Problem Statement
Car dealerships need to estimate how much a customer is willing to spend before approaching them. This model predicts the car purchase amount based on customer age, salary, credit card debt, net worth and gender — enabling smarter sales targeting.

---

## Dataset
- **Target Variable:** Car Purchase Amount
- **Features Used:** Gender, Age, Annual Salary, Credit Card Debt, Net Worth
- **Dropped Columns:** Customer Name, Customer e-mail, Country — no predictive value

---

## ML Pipeline

| Step | Action |
|---|---|
| Data Inspection | head(), tail(), pairplot for visual exploration |
| Feature Selection | Dropped Name, Email, Country columns |
| Scaling | MinMaxScaler applied to both X and y |
| Train/Test Split | 70/30 split |
| Model | Neural Network — 2 hidden layers (50 neurons each, ReLU activation) |
| Output Layer | 1 neuron, Linear activation (regression) |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Training | 100 epochs, batch size 50, 20% validation split |

---

## Model Architecture

```
Input Layer  →  5 features
Hidden Layer 1  →  50 neurons, ReLU
Hidden Layer 2  →  50 neurons, ReLU
Output Layer  →  1 neuron, Linear
```

---

## Key Learnings
- First introduction to Neural Networks and Deep Learning
- MinMaxScaler used instead of StandardScaler — scales both features and target to 0-1 range
- Training and validation loss curves used to monitor model learning
- Neural networks are powerful for regression on continuous target variables

---

## Tech Stack
Python | Pandas | NumPy | TensorFlow | Keras | Scikit-learn | Seaborn | Matplotlib

---

## Next Improvements
- Add more features to improve prediction accuracy
- Try other regression models (Linear Regression, Random Forest Regressor) for comparison
- Perform proper EDA and correlation analysis
- Add hyperparameter tuning for neural network layers and neurons

