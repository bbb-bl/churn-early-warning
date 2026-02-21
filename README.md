# 🚨 Customer Churn Early Warning System

A Streamlit-based decision-support prototype for Customer Success Managers. Predicts which customers are at risk of churning and recommends specific retention actions.

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

**Step 1:** Generate the dataset (only needed once):
```bash
python generate_data.py
```

**Step 2:** Train the model (only needed once):
```bash
python train_model.py
```

**Step 3:** Launch the app:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Streamlit prototype (front-end)
├── predict.py              # Prediction logic (separated from prototype)
├── train_model.py          # Offline model training script
├── generate_data.py        # Synthetic data generation
├── requirements.txt        # Python dependencies
├── data/
│   └── customers.csv       # Customer dataset
└── models/
    ├── churn_model.joblib   # Trained Random Forest model
    ├── label_encoders.joblib # Categorical encoders
    └── feature_cols.json    # Feature column order
```

## Architecture

- **Offline Layer:** `train_model.py` trains a Random Forest classifier and exports it via joblib
- **Prediction Logic:** `predict.py` handles model loading, scoring, risk tier assignment, and action recommendations
- **Prototype Layer:** `app.py` is the Streamlit front-end with dashboard, risk table, and customer deep-dive

---
PDAI 2026 — ESADE Masters in Business Analytics
