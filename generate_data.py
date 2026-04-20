"""
generate_data.py — Creates a realistic synthetic customer churn dataset.
Run this ONCE to create data/customers.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)
N = 800  # Number of customers

# ── Generate customer attributes ──
customer_ids = [f"CUST-{i:04d}" for i in range(1, N + 1)]

tenure = np.random.exponential(scale=24, size=N).clip(1, 72).astype(int)
monthly_charges = np.random.normal(65, 25, N).clip(20, 120).round(2)
contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.50, 0.30, 0.20])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N, p=[0.35, 0.20, 0.25, 0.20]
)
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20])
tech_support = np.random.choice(["Yes", "No", "No internet"], N, p=[0.30, 0.50, 0.20])
online_security = np.random.choice(["Yes", "No", "No internet"], N, p=[0.30, 0.50, 0.20])

# Usage metrics (SaaS-style)
logins_last_30d = np.random.poisson(lam=15, size=N).clip(0, 60)
support_tickets_last_90d = np.random.poisson(lam=2, size=N).clip(0, 15)
features_used_pct = np.random.beta(3, 3, N).round(2) * 100  # 0-100%

# Total charges = monthly * tenure (with some noise)
total_charges = (monthly_charges * tenure * np.random.uniform(0.9, 1.1, N)).round(2)

# ── Generate churn labels (realistic probabilities) ──
# Higher churn probability for: short tenure, month-to-month, low usage, many support tickets
churn_prob = (
    0.15  # base rate
    + 0.20 * (contract == "Month-to-month")
    - 0.10 * (contract == "Two year")
    - 0.003 * tenure  # longer tenure = lower churn
    + 0.002 * monthly_charges  # higher price = slightly more churn
    - 0.003 * logins_last_30d  # more usage = lower churn
    + 0.02 * support_tickets_last_90d  # more tickets = higher churn
    - 0.002 * features_used_pct  # more features used = lower churn
    + 0.08 * (payment_method == "Electronic check")  # known churn signal
    + 0.05 * (tech_support == "No")
    + 0.05 * (online_security == "No")
    + np.random.normal(0, 0.05, N)  # noise
)
churn_prob = churn_prob.clip(0.02, 0.95)
churn = (np.random.random(N) < churn_prob).astype(int)

# ── Company names (for realism) ──
companies = [
    "Acme Corp", "TechVibe", "DataFlow Inc", "CloudNine", "Pixel Labs",
    "Streamline Co", "NovaTech", "BrightPath", "Quantum SaaS", "Apex Digital",
    "BlueShift", "CoreStack", "FusionHub", "NetPulse", "Skyward Tech",
    "ZenithOS", "Catalyst AI", "Prism Analytics", "Vortex Cloud", "Helix Software"
]
company = np.random.choice(companies, N)

# MRR (Monthly Recurring Revenue) per customer
mrr = (monthly_charges * np.random.uniform(1, 5, N)).round(0).astype(int)

# ── Build DataFrame ──
df = pd.DataFrame({
    "customer_id": customer_ids,
    "company": company,
    "tenure_months": tenure,
    "contract": contract,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "mrr": mrr,
    "payment_method": payment_method,
    "internet_service": internet_service,
    "tech_support": tech_support,
    "online_security": online_security,
    "logins_last_30d": logins_last_30d,
    "support_tickets_last_90d": support_tickets_last_90d,
    "features_used_pct": features_used_pct.round(1),
    "churned": churn,
})

df.to_csv("data/customers.csv", index=False)
print(f"✅ Generated {N} customers. Churn rate: {churn.mean():.1%}")
print(df.head())
print(f"\nSaved to data/customers.csv")
