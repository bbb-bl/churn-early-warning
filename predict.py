import pandas as pd
import numpy as np
import joblib
import json

#Loading the model, encoders and features
def load_model():
    model = joblib.load("models/churn_model.joblib")
    encoders = joblib.load("models/label_encoders.joblib")
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, encoders, feature_cols

#Predicting churn 
def predict_churn(df: pd.DataFrame, model, encoders, feature_cols) -> pd.DataFrame:

    #Categorical encoding
    df_encoded = df[feature_cols].copy()
    for col, le in encoders.items():
        df_encoded[col] = le.transform(df_encoded[col]) #Just transform

    #Get churn probabilities (column 1 - Churn probability)
    probabilities = model.predict_proba(df_encoded)[:, 1]

    #Risk tier assignment using simple tresholds
    def get_tier(prob):
        if prob >= 0.60:
            return "High"
        elif prob >= 0.35:
            return "Medium"
        else:
            return "Low"

    #Risk drivers from feature importance and individual value
    top_drivers = []
    for idx in range(len(df)):
        drivers = get_risk_drivers(df.iloc[idx], probabilities[idx])
        top_drivers.append(drivers[0] if drivers else "Unknown")

    #Assigning recommended actions
    actions = [
        get_recommended_action(df.iloc[idx], probabilities[idx])
        for idx in range(len(df))
    ]

    #Adding columns to the original Dataframe
    result = df.copy()
    result["churn_probability"] = (probabilities * 100).round(1)
    result["risk_tier"] = [get_tier(p) for p in probabilities]
    result["top_risk_driver"] = top_drivers
    result["recommended_action"] = actions

    return result

#Identifying why specific customer is at risk based on rules
def get_risk_drivers(customer_row, churn_prob) -> list:

    drivers = []

    if customer_row["contract"] == "Month-to-month":
        drivers.append("Month-to-month contract (no commitment)")

    if customer_row["tenure_months"] < 6:
        drivers.append("Very new customer (< 6 months)")
    elif customer_row["tenure_months"] < 12:
        drivers.append("Short tenure (< 12 months)")

    if customer_row["logins_last_30d"] < 5:
        drivers.append("Very low engagement (< 5 logins/month)")
    elif customer_row["logins_last_30d"] < 10:
        drivers.append("Low engagement (< 10 logins/month)")

    if customer_row["support_tickets_last_90d"] > 5:
        drivers.append("High support ticket volume (> 5 tickets)")
    elif customer_row["support_tickets_last_90d"] > 3:
        drivers.append("Elevated support tickets (> 3 tickets)")

    if customer_row["features_used_pct"] < 25:
        drivers.append("Very low feature adoption (< 25%)")
    elif customer_row["features_used_pct"] < 40:
        drivers.append("Low feature adoption (< 40%)")

    if customer_row["monthly_charges"] > 90:
        drivers.append("High monthly cost (may seek cheaper alternative)")

    if customer_row["payment_method"] == "Electronic check":
        drivers.append("Electronic check payment (historically high churn)")

    if customer_row["tech_support"] == "No":
        drivers.append("No tech support subscription")

    if not drivers:
        drivers.append("No clear single risk factor")

    return drivers

#Decision support logic to match risk profile with an action
def get_recommended_action(customer_row, churn_prob) -> str:

    if churn_prob < 0.35:
        return "Monitor — No immediate action needed"

    # High risk + low engagement → re-onboarding
    if customer_row["logins_last_30d"] < 10 and customer_row["features_used_pct"] < 40:
        return "Schedule re-onboarding call — Low engagement & feature adoption"

    # High risk + many support tickets → escalate
    if customer_row["support_tickets_last_90d"] > 4:
        return "Escalate to Account Manager — Unresolved support issues"

    # High risk + month-to-month + high value → offer discount/upgrade
    if customer_row["contract"] == "Month-to-month" and customer_row["mrr"] > 200:
        return "Offer annual contract discount — High-value at-risk account"

    # High risk + month-to-month → lock in contract
    if customer_row["contract"] == "Month-to-month":
        return "Propose annual contract — Reduce month-to-month churn risk"

    # High risk + high cost → value demonstration
    if customer_row["monthly_charges"] > 80:
        return "Schedule ROI review — Demonstrate value to justify cost"

    # Default high-risk action
    return "Proactive check-in call — Assess satisfaction and needs"

#Getting all drivers for app.py readibility
def get_all_risk_drivers(customer_row, churn_prob) -> list:
    return get_risk_drivers(customer_row, churn_prob)
