# 🚀 Fraud Detection in Insurance Claims  

## 📌 Project Overview  
This project focuses on detecting fraudulent insurance claims using machine learning techniques. The dataset contains various policyholder details, claim information, and incident reports. The goal is to preprocess the data and build a model that classifies claims as fraudulent or non-fraudulent.  

## 📊 Dataset Details  
- The dataset consists of multiple categorical and numerical features related to insurance claims.  
- The target variable (`fraud_reported`) indicates whether a claim is fraudulent (`1`) or not (`0`).  
- Some features include policy details, incident descriptions, and claim amounts.  

## 🛠️ Data Preprocessing  
Several preprocessing steps were applied to clean and prepare the dataset for modeling:  

1. **Handling Missing Values**  
   - Replaced `?` with `NaN` and imputed missing categorical values with `'Unknown'`.  

2. **Feature Engineering**  
   - Converted `fraud_reported` from categorical (`Y/N`) to binary (`1/0`).  
   - Converted `policy_bind_date` and `incident_date` into `datetime` format.  
   - Dropped unnecessary columns (`policy_number`, `insured_zip`, `incident_location`, etc.).  
   - Applied one-hot encoding to categorical features.  

3. **Data Export**  
   - The cleaned dataset is saved as `insurance_claims_preprocessed.csv`.  

## 📂 Project Structure  
```
📁 Fraud-Detection-Insurance
│── 📂 data
│   ├── insurance_claims.csv  # Raw dataset
│   ├── insurance_claims_preprocessed.csv  # Cleaned dataset
│── 📂 src
│   ├── preprocess.py  # Data preprocessing script
│   ├── model_train.py  # Model training and evaluation
│── README.md  # Project documentation
│── requirements.txt  # Dependencies
```
## 📈 Expected Outcome  
The project aims to train a machine learning model that can effectively classify fraudulent and non-fraudulent insurance claims. Performance metrics such as accuracy, precision, recall, and F1-score will be evaluated.  

## 📌 Future Improvements  
- Feature selection and dimensionality reduction for better model performance.  
- Implementing advanced models like ensemble learning and deep learning.  
- Deploying the model using Flask or FastAPI for real-time fraud detection.  

---
