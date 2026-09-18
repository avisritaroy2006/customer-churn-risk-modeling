# Customer Churn Risk Modeling Using Machine Learning

## Project Overview

This project focuses on predicting customer churn risk using machine learning classification techniques.

The project addresses class imbalance using SMOTE and evaluates multiple machine learning algorithms. Random Forest is further optimized using GridSearchCV to improve model performance.

## Objectives

- Analyze customer data to identify churn patterns
- Develop machine learning models for churn risk prediction
- Handle class imbalance using SMOTE
- Compare multiple classification algorithms
- Optimize the Random Forest model using GridSearchCV
- Evaluate model performance using multiple classification metrics
- Identify important features associated with churn
- Predict churn probability for new customers

## Dataset

The dataset contains 50,000 customer records.

The target variable represents whether a customer churned.

The dataset has a highly imbalanced target distribution, making appropriate evaluation metrics and class-imbalance handling important.

## Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

Random Forest was further optimized using GridSearchCV.

## Methodology

Data Loading
↓
Data Preprocessing
↓
Train-Test Split
↓
SMOTE
↓
Model Training
↓
Model Evaluation
↓
Random Forest Hyperparameter Optimization
↓
Feature Importance Analysis
↓
Churn Risk Prediction

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Because the dataset is imbalanced, Precision, Recall, F1-score and ROC-AUC are considered alongside accuracy.

## Feature Importance

The optimized Random Forest model identified important predictive features including:

- Tenure months
- Monthly usage hours
- Premium plan
- Customer support calls

Feature importance visualization is included in the `images` folder.

## Model

The trained Random Forest model is saved using Joblib:

`customer_churn_prediction.joblib`

This allows the trained model to be reused for prediction without retraining.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib

## Project Structure

```text
customer-churn-risk-modeling/
│
├── customer_churn_dataset.xlsx
├── customer_churn_prediction.py
├── customer_churn_prediction.joblib
├── README.md
├── requirements.txt
│
└── images/
    ├── confusion_matrix.png
    └── feature_importance.png