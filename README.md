# Customer Churn Risk Modeling

## Project Overview

Customer churn prediction is a machine learning classification problem that aims to identify customers who are likely to stop using a service.

This project develops a **Customer Churn Risk Modeling system** using multiple machine learning classification algorithms. Since the dataset contains a highly imbalanced target variable, **SMOTE (Synthetic Minority Over-sampling Technique)** is used to improve the representation of the minority churn class.

Several classification models are evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC. Hyperparameter tuning is then performed on the Random Forest model using GridSearchCV.

---

## Objectives

* Predict whether a customer is likely to churn.
* Handle severe class imbalance using SMOTE.
* Compare multiple machine learning classification models.
* Evaluate models using suitable classification metrics.
* Optimize the Random Forest model using GridSearchCV.
* Identify the most important features contributing to customer churn.
* Save the trained model for future predictions.

---

## Dataset

The dataset used in this project is:

`customer_churn_dataset.csv`

### Dataset Information

* **Total records:** 50,000
* **Missing values:** 0
* **Target variable:** `churn`

### Class Distribution

* **Non-churn customers (0):** 48,967 (97.934%)
* **Churn customers (1):** 1,033 (2.066%)

Because the dataset is highly imbalanced, SMOTE is applied to the training data.

---

## Machine Learning Models

The following classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

---

## Methodology

The project follows these steps:

1. Load the customer churn dataset.
2. Perform data inspection and preprocessing.
3. Separate features and target variable.
4. Split the dataset into training and testing sets.
5. Apply SMOTE to the training data to handle class imbalance.
6. Train multiple classification models.
7. Evaluate the models using classification metrics.
8. Perform hyperparameter tuning on Random Forest using GridSearchCV.
9. Evaluate the tuned Random Forest model.
10. Generate confusion matrix and feature importance visualizations.
11. Save the trained model using Joblib.
12. Use the saved model for predicting churn risk for new customers.

---

## Model Performance

The evaluated models produced the following results on the test set:

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   79.51% |     7.79% | 82.13% |   14.23% |  0.8921 |
| Decision Tree       |   94.70% |    22.30% | 62.80% |   32.91% |  0.7909 |
| Random Forest       |   95.37% |    25.76% | 65.70% |   37.01% |  0.9426 |
| SVM                 |   88.74% |    14.46% | 90.34% |   24.93% |  0.9410 |
| KNN                 |   92.82% |    18.50% | 72.46% |   29.47% |  0.8817 |

Because the dataset is highly imbalanced, accuracy alone does not fully represent model performance. Precision, recall, F1-score, and ROC-AUC are therefore considered together.

---

## Tuned Random Forest

Random Forest hyperparameters were optimized using **GridSearchCV**.

### Best Parameters

```text
max_depth = 20
max_features = sqrt
min_samples_leaf = 1
min_samples_split = 5
n_estimators = 200
```

### Best Cross-Validation Score

```text
0.9713968448460714
```

Approximately **97.14%**.

### Tuned Random Forest Test Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 95.07% |
| Precision | 24.28% |
| Recall    | 65.22% |
| F1-score  | 35.39% |
| ROC-AUC   | 0.9496 |

---

## Tuned Random Forest Confusion Matrix

The confusion matrix for the tuned Random Forest is:

```text
[[9372  421]
 [  72  135]]
```

Where:

* **True Negatives (TN):** 9,372
* **False Positives (FP):** 421
* **False Negatives (FN):** 72
* **True Positives (TP):** 135

Visualization:

![Tuned Random Forest Confusion Matrix](images/confusion_matrix.png)

---

## Feature Importance

The tuned Random Forest identified the following feature importance values:

| Feature                | Importance |
| ---------------------- | ---------: |
| tenure_months          |   0.610211 |
| monthly_usage_hours    |   0.227358 |
| is_premium_plan        |   0.073300 |
| customer_support_calls |   0.057832 |
| payment_failures       |   0.018736 |
| has_multiple_devices   |   0.012562 |

Visualization:

![Feature Importance](images/feature_importance.png)

---

## Visualizations

### 1. Churn Distribution

Shows the distribution of churn and non-churn customers.

![Churn Distribution](images/churn_distribution.png)

### 2. Correlation Matrix

Shows the relationships between numerical features in the dataset.

![Correlation Matrix](images/correlation_matrix.png)

### 3. Logistic Regression Confusion Matrix

Shows the classification results of the Logistic Regression model.

![Logistic Regression Confusion Matrix](images/confusion_matrix_logistic_regression.png)

### 4. Tuned Random Forest Confusion Matrix

Shows the classification results of the optimized Random Forest model.

![Random Forest Confusion Matrix](images/confusion_matrix.png)

### 5. Feature Importance

Shows the relative importance of features used by the tuned Random Forest model.

![Feature Importance](images/feature_importance.png)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib
* GridSearchCV
* SMOTE

---

## Project Structure

```text
customer-churn-risk-modeling/
│
├── images/
│   ├── churn_distribution.png
│   ├── correlation_matrix.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── customer_churn_dataset.csv
├── customer_churn_prediction.py
├── customer_churn_prediction.joblib
├── README.md
└── requirements.txt
```

---

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

The program will train the models, display evaluation results, generate visualizations, save the trained Random Forest model, and allow predictions for new customer data.

---

## Author

**Avisrita Roy**

B.Tech CSE (AI & ML)
