# Customer Churn Analysis and Prediction - Telecom Dataset
## 📄 Project Overview

This project analyzes customer churn behavior for a telecom company using real-world data.  
It aims to:
- Identify key factors leading to churn
- Predict churn using machine learning models
- Provide actionable insights to improve customer retention
## 📊 Dataset

- 📁 Source: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 📌 Records: 7,043 customers
- 🧩 Features include: Demographics, Contract details, Payment info, Service usage, and Churn status
## 🧹 Data Cleaning

- Converted `TotalCharges` from string to numeric
- Handled missing values (~11 nulls in TotalCharges)
- Dropped `customerID` (non-informative)
- Encoded categorical variables using one-hot and label encoding
## 📊 Exploratory Analysis

Key findings:
- 📉 Customers with month-to-month contracts churn the most
- 🕑 Low-tenure customers are more likely to churn
- 💳 Customers paying via electronic check have higher churn
- 📡 Fiber optic users churn more than DSL customers

Used Seaborn and Matplotlib for visual insights.
![Churn Distribution](images/churn_distribution.png)
![Contract vs Churn](images/contract_churn.png)
## 🤖 Machine Learning Models

### 1. Logistic Regression
- Accuracy: 79%
- Good interpretability

### 2. Random Forest
- Accuracy: 84%
- Top features: Contract type, Tenure, Internet Service, Payment Method

```python
# Sample code block (optional)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

---

## 📈 7. **Evaluation Metrics**

```markdown
## 📈 Evaluation

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Used Scikit-learn for classification report.
## 🧠 Business Recommendations

- Promote long-term contracts to reduce churn
- Improve onboarding and engagement for new users (<12 months)
- Encourage auto-payment methods to reduce electronic check churn
- Investigate dissatisfaction among fiber optic users
## 📁 File Structure
## 🚀 How to Run

1. Clone the repo
2. Install requirements (if applicable)
3. Open `churn_model.ipynb` in Jupyter Notebook
## 🙌 Acknowledgments

- Dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Built with Python, Pandas, Scikit-learn, Seaborn, Matplotlib
> 📌 Résumé Summary:  
> Built a Random Forest model on 7K+ telecom customers to predict churn with 84% accuracy. Generated actionable business recommendations from EDA and model insights.

