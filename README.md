# 🚨 Customer Churn Early Warning System

**🔗 Live demo:** https://churn-early-warning-axkh9aqqobzd2b44pj74ck.streamlit.app/

A Streamlit-based decision-support prototype for Customer Success teams. Combines a classical ML risk model (Random Forest) with two LLM-powered features — an AI Account Strategist that generates retention plans from customer profiles, and an "Ask Your Data" natural-language query interface built on the tool-use / function-calling pattern.

---

## Features

The app has four tabs:

**1. Dashboard** — Portfolio-level view: risk tier breakdown, revenue-at-risk, high-risk customers.

**2. Customer Deep-Dive** — Per-customer risk drivers + **AI Account Strategist**: sends the full customer profile to an LLM (Llama 3.3 70B via Groq), which returns structured JSON with urgency, root cause, a named retention strategy, 3 action steps with owners and timelines, talking points, and warnings. Python parses the JSON and renders each piece as a Streamlit component.

**3. Ask Your Data** — Natural-language query interface built with the **tool-use pattern**. Four Python functions are exposed as tools (`get_high_risk_customers`, `get_portfolio_summary`, `get_customers_by_filter`, `compare_segments`). The LLM decides which function to call based on the user's question, the function executes on the real dataframe, and the LLM writes a natural-language answer from the result. The tool choice is displayed in the UI for transparency.

**4. Model Evaluation** — Confusion matrix, precision (36.8%), recall (34.1%), F1 (35.4%), accuracy (68.1%). Errors are framed in business terms: false negatives (missed churners → lost revenue) are weighted more heavily than false positives (unnecessary check-ins → still builds goodwill). Includes model rationale and roadmap.

---

## Architecture

- **Offline Layer** — `train_model.py` trains the Random Forest, exports with joblib
- **Prediction Logic** — `predict.py` handles scoring, risk-tier assignment, rule-based baseline recommendations
- **LLM Layer** — `llm_utils.py` isolates all LLM logic: prompts, tool schemas, JSON parsing, tool-call execution
- **UI Layer** — `app.py` is the Streamlit front-end (4 tabs)

Keeping LLM logic in its own module mirrors the same separation-of-concerns pattern used for `predict.py` — the UI stays thin, and LLM behavior can be tested and swapped independently.

---

## Setup
