import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib 
import json

#Loading the data
df = pd.read_csv("data/customers.csv")

#Feature selection
#Dropping customer_id, company and churned
feature_cols = [
    "tenure_months", "monthly_charges", "total_charges", "mrr",
    "logins_last_30d", "support_tickets_last_90d", "features_used_pct",
    "contract", "payment_method", "internet_service", "tech_support", "online_security"
]

#Text categories to numbers
label_encoders = {}
df_model = df[feature_cols].copy()

#Converting the categorical features to number
for col in ["contract", "payment_method", "internet_service", "tech_support", "online_security"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

X = df_model
y = df["churned"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random forest 
model = RandomForestClassifier(
    n_estimators=100,     
    max_depth=8,          
    random_state=42,
    class_weight="balanced"  #Handle class imbalance
)
model.fit(X_train, y_train)

#Evaluation
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.1%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

#Checking feature importance
importance = dict(zip(feature_cols, model.feature_importances_))
importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
print("Feature Importance (Churn):")
for feat, imp in importance_sorted.items():
    print(f"  {feat}: {imp:.3f}")

# Saving the model
joblib.dump(model, "models/churn_model.joblib")

# Saving the label encoders (needed at runtime to encode new data the same way)
joblib.dump(label_encoders, "models/label_encoders.joblib")

# Saving the feature column order (must match at prediction time)
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)