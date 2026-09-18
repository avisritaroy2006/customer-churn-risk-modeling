import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,)
from sklearn.metrics import(confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib


df=pd.read_csv("customer_churn_dataset.csv")
print(df)                                      #describe dataset   
print(df.info())
print(df.shape)                                  
print(df.size)
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())


#training

X=df.drop("churn",axis=1)
y=df["churn"]
print(X)
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


smote=SMOTE(random_state=42)
X_train_SMOTE,y_train_SMOTE=smote.fit_resample(X_train,y_train)



print("Before SMOTE:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

print("After SMOTE:")
print(y_train_SMOTE.value_counts())
print(y_train_SMOTE.value_counts(normalize=True) * 100)

model=LogisticRegression()
model.fit(X_train_SMOTE,y_train_SMOTE)
y_pred=model.predict(X_test)





#model evaluation

print("accuracy score",accuracy_score(y_test,y_pred))
print("precision score",precision_score(y_test,y_pred))
print("recall score",recall_score(y_test,y_pred))
print("f1 score",f1_score(y_test,y_pred))
y_prob=model.predict_proba(X_test)[:,1]
print("roc-auc score",roc_auc_score(y_test,y_prob))
print(df["churn"].value_counts())
print(df["churn"].value_counts(normalize=True) * 100)

print("Before SMOTE:")
print(y_train.value_counts(normalize=True) * 100)

print("After SMOTE:")
print(y_train_SMOTE.value_counts(normalize=True) * 100)

print("accuracy score",accuracy_score(y_test,y_pred))
print("precision score",precision_score(y_test,y_pred))
print("recall score",recall_score(y_test,y_pred))
print("f1 score",f1_score(y_test,y_pred))
y_prob=model.predict_proba(X_test)[:,1]
print("roc-auc score",roc_auc_score(y_test,y_prob))





df["churn"].value_counts().plot(kind="bar")
plt.xlabel("churn")
plt.ylabel("count")
plt.title("customer churn")
plt.savefig("images/churn_distribution.png", bbox_inches="tight")
plt.show()

corr=df.corr()
plt.figure(figsize=(8,6))
plt.imshow(corr,cmap="coolwarm",interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(df.columns)),df.columns,rotation=90)
plt.yticks(range(len(df.columns)),df.columns)
plt.tight_layout()
plt.title("correlation matrix")
plt.show()

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.savefig("images/correlation_matrix.png", bbox_inches="tight")
plt.show()

cm=confusion_matrix(y_test,y_pred)
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")

plt.savefig("images/confusion_matrix_logistic_regression.png", bbox_inches="tight")
plt.show()



dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_SMOTE,y_train_SMOTE)
y_pred_dt=dt_model.predict(X_test)


print("accuracy score",accuracy_score(y_test,y_pred_dt))
print("precision score",precision_score(y_test,y_pred_dt))
print("recall score",recall_score(y_test,y_pred_dt))
print("f1 score",f1_score(y_test,y_pred_dt))
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
print("roc-auc score",roc_auc_score(y_test,y_prob_dt))

cm_dt=confusion_matrix(y_test,y_pred_dt)
print(cm_dt)



rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train_SMOTE,y_train_SMOTE)
y_pred_rf=rf_model.predict(X_test)

print("Random Forest Results")

print("accuracy score", accuracy_score(y_test, y_pred_rf))

print("precision score", precision_score(y_test, y_pred_rf))

print("recall score", recall_score(y_test, y_pred_rf))

print("f1 score", f1_score(y_test, y_pred_rf))

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("roc-auc score", roc_auc_score(y_test, y_prob_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)




#scaling

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train_SMOTE)
X_test_scaled=scaler.transform(X_test)

#create svm model

svm_model=SVC(probability=True,random_state=42)

#train

svm_model.fit(X_train_scaled,y_train_SMOTE)

#prediction

y_pred_svm=svm_model.predict(X_test_scaled)

#evaluation

print("SVM Results")

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))

y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob_svm))
cm_svm = confusion_matrix(y_test, y_pred_svm)

print("Confusion Matrix:")
print(cm_svm)




knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train_SMOTE)
y_pred_knn=knn_model.predict(X_test_scaled)

print("KNN Results")

print("Accuracy:", accuracy_score(y_test, y_pred_knn))

print("Precision:", precision_score(y_test, y_pred_knn))

print("Recall:", recall_score(y_test, y_pred_knn))

print("F1 Score:", f1_score(y_test, y_pred_knn))

# Probability for ROC-AUC
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)

print("Confusion Matrix:")
print(cm_knn)




# Model Comparison

results = {
    "Logistic Regression": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    ],

    "Decision Tree": [
        accuracy_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_dt),
        roc_auc_score(y_test, y_prob_dt)
    ],

    "Random Forest": [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf),
        roc_auc_score(y_test, y_prob_rf)
    ],

    "SVM": [
        accuracy_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_svm),
        recall_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_svm),
        roc_auc_score(y_test, y_prob_svm)
    ],

    "KNN": [
        accuracy_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_knn),
        roc_auc_score(y_test, y_prob_knn)
    ]
}


comparison = pd.DataFrame(
    results,
    index=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
).T

print(comparison)

param_grid={
    "n_estimators":[100,200],
        "max_depth":[10,20],
        "min_samples_split":[2,5],
        "min_samples_leaf":[1],
        "max_features":["sqrt"]
}
grid_search=GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=1
)
grid_search.fit(X_train_SMOTE,y_train_SMOTE)
print(grid_search.best_params_)
print(grid_search.best_score_)

best_rf=grid_search.best_estimator_
y_pred_tuned=best_rf.predict(X_test)
y_prob_tuned=best_rf.predict_proba(X_test)[:,1]
print("tunned random forest result")

print("Accuracy:",
      accuracy_score(y_test, y_pred_tuned))

print("Precision:",
      precision_score(y_test, y_pred_tuned))

print("Recall:",
      recall_score(y_test, y_pred_tuned))

print("F1 Score:",
      f1_score(y_test, y_pred_tuned))

print("ROC-AUC:",
      roc_auc_score(y_test, y_prob_tuned))

print("Confusion Matrix:")

cm_tuned = confusion_matrix(y_test, y_pred_tuned)
print(cm_tuned)

plt.figure(figsize=(6,5))
sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned Random Forest")

plt.savefig("images/confusion_matrix.png", bbox_inches="tight")
plt.show()

importance=best_rf.feature_importances_
feature_importance=pd.DataFrame({
    "Feature":X.columns,
    "Importance":best_rf.feature_importances_
})
feature_importance=feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print(feature_importance.head(10))

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"].head(10)[::-1],
    feature_importance["Importance"].head(10)[::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importance - Tuned Random Forest")

plt.savefig("images/feature_importance.png", bbox_inches="tight")
plt.show()


joblib.dump(best_rf,"customer_churn_prediction.joblib")

tenure = int(input("Enter tenure in months: "))
usage = float(input("Enter monthly usage hours: "))
premium = int(input("Is premium plan? (1=Yes, 0=No): "))
support = int(input("Enter number of support calls: "))
payment = int(input("Enter number of payment failures: "))
devices = int(input("Has multiple devices? (1=Yes, 0=No): "))

new_customer_data = pd.DataFrame(0, index=[0], columns=X.columns)

new_customer_data["tenure_months"] = tenure
new_customer_data["monthly_usage_hours"] = usage
new_customer_data["is_premium_plan"] = premium
new_customer_data["customer_support_calls"] = support
new_customer_data["payment_failures"] = payment
new_customer_data["has_multiple_devices"] = devices

new_customer_data = new_customer_data[X.columns]

prediction = best_rf.predict(new_customer_data)
probability = best_rf.predict_proba(new_customer_data)

print("Prediction:", prediction)
print("Probability:", probability)

if prediction[0] == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")