# 📊 Customer Churn Analysis and Prediction – Telco Dataset

## 📄 Project Overview

This project analyzes customer churn behavior for a telecom company using real-world customer data. The objective is to:
- Identify key drivers of customer churn
- Predict churn using machine learning models
- Recommend strategies to improve customer retention

---

## 📊 Dataset

- 📁 Source: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 📌 Records: 7,043 customers
- 🧩 Features include:
  - Customer demographics
  - Services used (e.g., Internet, phone)
  - Contract and billing details
  - Churn label (Yes/No)

---

## 🧹 Data Cleaning & Preprocessing

- Converted `TotalCharges` from object to float and handled missing values
- Dropped `customerID` as it's non-informative
- Encoded binary categorical features using label encoding
- One-hot encoded multi-category features (e.g., `Contract`, `InternetService`)
- Final dataset: All numeric, model-ready

---

## 📊 Exploratory Data Analysis (EDA)

- 🔍 Month-to-month contract customers churn more often
- 📉 Tenure is strongly negatively correlated with churn
- 💳 Electronic check users have highest churn
- 📡 Fiber optic customers churn more than DSL

### 📷 Sample Visuals

![Churn Distribution](images/churn_distribution.png)
![Churn vs Contract](images/contract_churn.png)

---

## 🤖 Machine Learning Models

### 1. Logistic Regression
- Accuracy: 79%
- Interpretable coefficients

### 2. Random Forest
- Accuracy: 84%
- Top features: Contract type, tenure, payment method, internet service

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
📈 Evaluation
Metrics used: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Best model: Random Forest with ~84% test accuracy

🧠 Business Recommendations
📈 Encourage long-term contracts to reduce churn

💳 Incentivize auto-pay and credit card payment methods

👋 Improve onboarding and early engagement for new customers

🛠️ Address fiber optic service issues to retain subscribers

🚀 How to Run
Clone this repository

Install dependencies (if needed)

Run churn_model.ipynb in Jupyter Notebook

bash
Copy
Edit
pip install pandas matplotlib seaborn scikit-learn
📁 Project Structure
kotlin
Copy
Edit
Telco-Churn-Project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── images/
│   └── churn_distribution.png
│   └── contract_churn.png
├── churn_model.ipynb
├── churn_model.py
└── README.md
📌 Résumé Summary
Built a Random Forest model on 7K+ telecom customers to predict churn with 84% accuracy. Generated actionable business insights that support strategic decision-making in customer retention.

🙌 Acknowledgments
Dataset from Kaggle

Built using Python, Pandas, Matplotlib, Seaborn, Scikit-learn

