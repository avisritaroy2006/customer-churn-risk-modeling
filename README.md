# Customer Churn Risk Modeling Using Machine Learning

## Project Overview

This project focuses on predicting customer churn risk using machine learning classification techniques.

The project addresses class imbalance using SMOTE and evaluates multiple machine learning algorithms. Random Forest is further optimized using GridSearchCV to improve model performance.

## Objectives

* Analyze customer data to identify churn patterns
* Develop machine learning models for churn risk prediction
* Handle class imbalance using SMOTE
* Compare multiple classification algorithms
* Optimize the Random Forest model using GridSearchCV
* Evaluate model performance using multiple classification metrics
* Identify important features associated with churn
* Predict churn probability for new customers

## Dataset

The dataset contains **50,000 customer records**.

The target variable represents whether a customer churned.

The dataset has a highly imbalanced target distribution, making appropriate evaluation metrics and class-imbalance handling important.

## Machine Learning Models

The following classification models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

Random Forest was further optimized using GridSearchCV.

## Methodology

**Data Loading → Data Preprocessing → Train-Test Split → SMOTE → Model Training → Model Evaluation → Random Forest Hyperparameter Optimization → Feature Importance Analysis → Churn Risk Prediction**

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

Because the dataset is imbalanced, Precision, Recall, F1-score, and ROC-AUC are considered alongside accuracy.

## Feature Importance

The optimized Random Forest model identified important predictive features including:

* Tenure months
* Monthly usage hours
* Premium plan
* Customer support calls

The feature importance visualization is included in the `images` folder.

## Model

The trained Random Forest model is saved using Joblib:

`customer_churn_prediction.joblib`

This allows the trained model to be reused for prediction without retraining.

## Visualizations

The project includes the following visualizations:

* Customer churn distribution
* Correlation matrix
* Logistic Regression confusion matrix
* Tuned Random Forest confusion matrix
* Tuned Random Forest feature importance

All visualizations are available in the `images` folder.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib

## Project Structure

```text
customer-churn-risk-modeling/
│
├── customer_churn_dataset.csv
├── customer_churn_prediction.py
├── customer_churn_prediction.joblib
├── README.md
├── requirements.txt
│
└── images/
    ├── churn_distribution.png
    ├── correlation_matrix.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/avisritaroy2006/customer-churn-risk-modeling.git
```

### 2. Navigate to the project directory

```bash
cd customer-churn-risk-modeling
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Python program

```bash
python customer_churn_prediction.py
```

## Author

**Avisrita Roy**

B.Tech CSE (AI & ML)
