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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probabilities for the "churned" class

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)        # Of all predicted churns, how many actually churned?
rec = recall_score(y_test, y_pred)            # Of all actual churns, how many did we catch?
f1 = f1_score(y_test, y_pred)                 # Balance between precision and recall
cm = confusion_matrix(y_test, y_pred)         # [[TN, FP], [FN, TP]]

print(f"\nAccuracy:  {acc:.1%}")
print(f"Precision: {prec:.1%}")
print(f"Recall:    {rec:.1%}")
print(f"F1 Score:  {f1:.1%}")
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

#Checking feature importance
importance = dict(zip(feature_cols, model.feature_importances_))
importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
print("Feature Importance (Churn):")
for feat, imp in importance_sorted.items():
    print(f"  {feat}: {imp:.3f}")

# ══════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ══════════════════════════════════════════════════════════════

# Save the model
joblib.dump(model, "models/churn_model.joblib")

# Save the label encoders
joblib.dump(label_encoders, "models/label_encoders.joblib")

# Save the feature column order
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# Save evaluation metrics (NEW — professor asked for this)
eval_metrics = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "confusion_matrix": {
        "true_negative": int(cm[0][0]),   # correctly predicted "stays"
        "false_positive": int(cm[0][1]),  # predicted churn but actually stayed (wasted effort)
        "false_negative": int(cm[1][0]),  # predicted stay but actually churned (MISSED — costly!)
        "true_positive": int(cm[1][1])    # correctly predicted churn (caught it!)
    },
    "test_size": len(y_test),
    "train_size": len(y_train)
}

with open("models/eval_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)

print("\n✅ Model, encoders, features, and evaluation metrics saved to /models/")


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