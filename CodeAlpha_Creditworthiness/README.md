# Creditworthiness Prediction Using Machine Learning

## 📌 Objective
The goal of this project is to predict an individual's **creditworthiness** using their past financial and demographic data.  
This helps financial institutions and lenders make informed decisions about loan approvals.

---

## 🚀 Machine Learning Models Used
The project uses three classification algorithms:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

The best model is selected based on **ROC-AUC** score and saved as `best_model.pkl`.

---

## 🧠 Features Used in the Dataset
The dataset (generated synthetically if not present) includes realistic financial indicators:

### **Financial Features**
- Income  
- Total Debt  
- Number of Late Payments  
- Payment History Score  
- Credit Utilization  

### **Demographic Features**
- Age  
- Employment Years  
- Loan Purpose  

### **Engineered Features**
- Debt-to-Income Ratio  
- Average Payment Delay  
- Credit Score Category (High / Medium / Low)

Feature engineering improves model performance and interpretability.

---

## 📊 Evaluation Metrics
Each model is evaluated using:

- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrix**

These metrics help understand how well the model predicts both classes, especially the minority class.

---

## 🛠 How to Run the Project

### **1. Create & Activate Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
