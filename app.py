import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from predict import load_model, predict_churn, get_all_risk_drivers

st.set_page_config(
    page_title="Churn Early Warning System",
    layout="wide"  #full width of the screen
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Bigger, bolder title */
    h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: #1B2A4A !important;
        margin-bottom: 0.2rem !important;
    }
    
    /* Subtitle / caption styling */
    .stCaption {
        font-size: 1.1rem !important;
        color: #64748B !important;
    }
    
    /* Tab buttons - bigger and more distinct */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-bottom: 12px;
        border-bottom: 2px solid #E2E8F0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px 8px 0 0 !important;
        color: #64748B !important;
        background-color: #F0F2F6 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        background-color: #1B2A4A !important;
        border-bottom: 3px solid #2E7D5B !important;
    }
    
    /* Subheader styling */
    h2 {
        color: #1B2A4A !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: #1B2A4A !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


#Cacheing data
@st.cache_data  
def load_data():
    df = pd.read_csv("data/customers.csv")
    model, encoders, feature_cols = load_model()
    df_scored = predict_churn(df, model, encoders, feature_cols)
    return df_scored

df = load_data()


#Sidebar filters below
st.sidebar.title("Filters")
st.sidebar.markdown("Filter your customer portfolio based on risk tier, contract type and MRR")

#Risk tier filter
risk_filter = st.sidebar.multiselect(
    "Risk Tier",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)

#Contract type filter
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["contract"].unique().tolist(),
    default=df["contract"].unique().tolist()
)

#MRR range filter
mrr_min, mrr_max = int(df["mrr"].min()), int(df["mrr"].max())
mrr_range = st.sidebar.slider(
    "MRR Range ($)",
    min_value=mrr_min,
    max_value=mrr_max,
    value=(mrr_min, mrr_max)
)

# Toggle: show only high-risk
high_risk_only = st.sidebar.toggle("🔴 Only showing High Risk", value=False)

#Apply filters
df_filtered = df[
    (df["risk_tier"].isin(risk_filter)) &
    (df["contract"].isin(contract_filter)) &
    (df["mrr"] >= mrr_range[0]) &
    (df["mrr"] <= mrr_range[1])
]

if high_risk_only:
    df_filtered = df_filtered[df_filtered["risk_tier"] == "High"]


#Dashboards and Customer Deep-Dive
st.title("Customer Churn Early Warning System")
st.caption("Decision-support tool for Customer Success Managers to Prioritize, understand, and act on customer churn risk.")

tab_dashboard, tab_deepdive, tab_query, tab_model = st.tabs(["Dashboard", "Customer Deep-Dive", "Ask Your Data", "Model Evaluation"])

#Dashboard tab
with tab_dashboard:

    #Metrics top row
    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df_filtered)
    high_risk_count = len(df_filtered[df_filtered["risk_tier"] == "High"])
    revenue_at_risk = df_filtered[df_filtered["risk_tier"] == "High"]["mrr"].sum()
    avg_churn_prob = df_filtered["churn_probability"].mean()

    with col1:
        st.metric("Total Customers", f"{total_customers}")
    with col2:
        st.metric("🔴 High Risk", f"{high_risk_count}",
                   delta=f"{high_risk_count/max(total_customers,1)*100:.0f}%",
                   delta_color="inverse")  # Red when high = bad
    with col3:
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}/mo")
    with col4:
        st.metric("Avg Churn Probability", f"{avg_churn_prob:.1f}%")

    st.markdown("---")

    #Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Risk Distribution")
        risk_counts = df_filtered["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        fig_risk = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map={"High": "#C4314B", "Medium": "#D4943A", "Low": "#2E7D5B"},
            labels={"x": "Risk Tier", "y": "Number of Customers"},
        )
        fig_risk.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_risk, use_container_width=True)

    with chart_col2:
        st.subheader("Churn Probability vs. Tenure")
        fig_scatter = px.scatter(
            df_filtered,
            x="tenure_months",
            y="churn_probability",
            color="risk_tier",
            color_discrete_map={"High": "#C4314B", "Medium": "#D4943A", "Low": "#2E7D5B"},
            hover_data=["company", "mrr", "contract"],
            labels={"tenure_months": "Tenure (months)", "churn_probability": "Churn Probability (%)"},
            size="mrr",
            size_max=15,
        )
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    #Risk Table sorted with impartance of action
    st.subheader("Customer Risk Worklist")
    st.caption("Sorted by churn probability. This is your daily action list.")

    #Prepare display table
    display_cols = [
        "customer_id", "company", "churn_probability", "risk_tier",
        "mrr", "contract", "top_risk_driver", "recommended_action"
    ]
    df_display = df_filtered[display_cols].sort_values("churn_probability", ascending=False)

    #Table formatting for readibility
    st.dataframe(
        df_display,
        column_config={
            "customer_id": "Customer ID",
            "company": "Company",
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Risk %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "risk_tier": "Risk Tier",
            "mrr": st.column_config.NumberColumn("MRR ($)", format="$%d"),
            "contract": "Contract",
            "top_risk_driver": "Top Risk Driver",
            "recommended_action": "Recommended Action",
        },
        hide_index=True,
        use_container_width=True,
        height=450,
    )

    #Portfolio Summary Statistics
    with st.expander("📈 Portfolio Summary Statistics"):
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            st.write("**By Contract Type:**")
            st.dataframe(
                df_filtered.groupby("contract")["churn_probability"].mean().round(1).reset_index()
                .rename(columns={"churn_probability": "Avg Churn %"}),
                hide_index=True
            )
        with sum_col2:
            st.write("**By Payment Method:**")
            st.dataframe(
                df_filtered.groupby("payment_method")["churn_probability"].mean().round(1).reset_index()
                .rename(columns={"churn_probability": "Avg Churn %"}),
                hide_index=True
            )
        with sum_col3:
            st.write("**By Internet Service:**")
            st.dataframe(
                df_filtered.groupby("internet_service")["churn_probability"].mean().round(1).reset_index()
                .rename(columns={"churn_probability": "Avg Churn %"}),
                hide_index=True
            )


#Deep-Dive tab
with tab_deepdive:

    st.subheader("Select a customer to see their risk profile")

    #Customer selector with high risk first
    df_sorted = df_filtered.sort_values("churn_probability", ascending=False)
    customer_options = [
        f"{row['customer_id']} — {row['company']} — {row['risk_tier']} Risk ({row['churn_probability']}%)"
        for _, row in df_sorted.iterrows() #_ignore index uses row
    ]

    if len(customer_options) == 0:
        st.warning("No customers match your current filters")
        st.stop()

    selected = st.selectbox("Choose a customer:", customer_options)

    #Customer ID extraction from the selection string
    selected_id = selected.split(" — ")[0]
    customer = df[df["customer_id"] == selected_id].iloc[0]

    st.markdown("---")

    #Customer Profile
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

    with header_col1:
        st.markdown(f"### {customer['company']}")
        st.markdown(f"**Customer ID:** {customer['customer_id']}")
        st.markdown(f"**Contract:** {customer['contract']} | **Tenure:** {customer['tenure_months']} months")

    with header_col2:
        #Churn risk with a visual
        st.metric("Churn Risk", f"{customer['churn_probability']}%")
        risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(customer["risk_tier"], "⚪")
        st.markdown(f"**Risk Tier:** {risk_color} {customer['risk_tier']}")

    with header_col3:
        st.metric("Monthly Revenue", f"${customer['mrr']:,.0f}")
        st.metric("Monthly Charges", f"${customer['monthly_charges']:,.2f}")

    st.markdown("---")

    #Behavioral Signals
    st.subheader("Behavioral Signals")
    signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)

    with signal_col1:
        logins = customer["logins_last_30d"]
        delta_label = "Good" if logins >= 15 else ("Concerning" if logins >= 8 else "Critical")
        st.metric("Logins (30d)", logins, delta=delta_label,
                   delta_color="normal" if logins >= 15 else ("off" if logins >= 8 else "inverse"))

    with signal_col2:
        tickets = customer["support_tickets_last_90d"]
        delta_label = "Normal" if tickets <= 2 else ("Elevated" if tickets <= 4 else "High")
        st.metric("Support Tickets (90d)", tickets, delta=delta_label,
                   delta_color="normal" if tickets <= 2 else ("off" if tickets <= 4 else "inverse"))

    with signal_col3:
        features = customer["features_used_pct"]
        delta_label = "Good" if features >= 50 else ("Low" if features >= 30 else "Very Low")
        st.metric("Feature Adoption", f"{features}%", delta=delta_label,
                   delta_color="normal" if features >= 50 else ("off" if features >= 30 else "inverse"))

    with signal_col4:
        st.metric("Payment Method", customer["payment_method"])
        st.metric("Internet Service", customer["internet_service"])

    st.markdown("---")

    #Risk Drivers
    st.subheader("Why This Customer Is At Risk")

    all_drivers = get_all_risk_drivers(customer, customer["churn_probability"] / 100)

    if all_drivers and all_drivers[0] != "No clear single risk factor":
        for i, driver in enumerate(all_drivers):
            severity = "🔴" if i == 0 else "🟡"
            st.markdown(f"{severity} **{driver}**")
    else:
        st.success("No major risk factors identified. This customer appears stable.")

    #Feature importance chart for this customer
    st.subheader("Risk Factor Breakdown")
    feature_labels = {
        "contract": "Contract Type",
        "monthly_charges": "Monthly Cost",
        "features_used_pct": "Feature Adoption",
        "mrr": "Revenue (MRR)",
        "total_charges": "Total Charges",
        "tenure_months": "Tenure",
        "logins_last_30d": "Login Frequency",
        "support_tickets_last_90d": "Support Tickets",
        "payment_method": "Payment Method",
        "internet_service": "Internet Service",
        "tech_support": "Tech Support",
        "online_security": "Online Security"
    }

    #Use global feature importances from the model
    model, _, _ = load_model()
    import json
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    importance_df = pd.DataFrame({
        "Feature": [feature_labels.get(f, f) for f in feature_cols],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#2E7D5B", "#D4943A", "#C4314B"],
    )
    fig_importance.update_layout(height=400, showlegend=False,
                                  coloraxis_showscale=False)
    st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("---")

    #Recommended action and revenue impact
    st.subheader("Recommended Action")

    action = customer["recommended_action"]
    if customer["risk_tier"] == "High":
        st.error(f"**{action}**")
    elif customer["risk_tier"] == "Medium":
        st.warning(f"**{action}**")
    else:
        st.success(f"**{action}**")

    #Action context
    with st.expander("Why this action?"):
        st.write(f"""
        This recommendation is based on the customer's risk profile:
        - **Churn Probability:** {customer['churn_probability']}%
        - **Contract Type:** {customer['contract']}
        - **Engagement Level:** {customer['logins_last_30d']} logins/month, {customer['features_used_pct']}% features used
        - **Support Health:** {customer['support_tickets_last_90d']} tickets in the last 90 days
        - **Revenue at Stake:** ${customer['mrr']:,.0f}/month

        The system prioritizes actions that address the primary risk driver while considering
        the customer's value and likelihood of responding to intervention.
""")

# ══════════════════════════════════════════════════════════
    # AI ACCOUNT STRATEGIST — LLM Feature (Category A+B)
    # ══════════════════════════════════════════════════════════
    st.subheader("AI Account Strategist")
    st.caption("AI-generated personalized retention strategy based on this customer's data.")

    # Use session_state to persist the strategy across reruns
    if "last_strategy_customer" not in st.session_state:
        st.session_state.last_strategy_customer = None
        st.session_state.last_strategy_result = None

    if st.button("Generate Retention Strategy", key="strategist_btn", type="primary"):
        with st.spinner("Analyzing customer profile and generating strategy..."):
            from llm_utils import generate_retention_strategy

            customer_data = {
                "company": customer["company"],
                "mrr": customer["mrr"],
                "tenure_months": customer["tenure_months"],
                "contract": customer["contract"],
                "monthly_charges": customer["monthly_charges"],
                "logins_last_30d": customer["logins_last_30d"],
                "features_used_pct": customer["features_used_pct"],
                "support_tickets_last_90d": customer["support_tickets_last_90d"],
                "payment_method": customer["payment_method"],
                "tech_support": customer["tech_support"],
                "churn_probability": customer["churn_probability"],
                "risk_tier": customer["risk_tier"],
            }

            result = generate_retention_strategy(customer_data, all_drivers)

            # Save to session_state so it survives reruns
            st.session_state.last_strategy_customer = customer["customer_id"]
            st.session_state.last_strategy_result = result

    # Display strategy if we have one for THIS customer
    if (st.session_state.last_strategy_customer == customer["customer_id"] 
        and st.session_state.last_strategy_result is not None):
        
        result = st.session_state.last_strategy_result

        if result["success"]:
            strategy = result["strategy"]

            urgency_colors = {"critical": "🔴", "high": "🟠", "medium": "🟡"}
            urgency_icon = urgency_colors.get(strategy.get("urgency", ""), "⚪")
            st.markdown(f"**Urgency:** {urgency_icon} {strategy.get('urgency', 'N/A').upper()}")
            st.markdown(f"**Strategy:** {strategy.get('strategy_name', 'N/A')}")
            st.markdown(f"**Root Cause:** {strategy.get('root_cause', 'N/A').replace('$', '\\$')}")

            st.markdown("---")

            st.markdown("**Action Plan:**")
            actions = strategy.get("actions", [])
            for action in actions:
                with st.container():
                    st.markdown(
                        f"**Step {action.get('step', '?')}:** {action.get('action', 'N/A')}  \n"
                        f"👤 *Owner:* {action.get('owner', 'N/A')} · "
                        f"⏱️ *Timeline:* {action.get('timeline', 'N/A')}  \n"
                        f"✅ *Expected Outcome:* {action.get('expected_outcome', 'N/A')}"
                    )
                    st.markdown("")

            talking_points = strategy.get("talking_points", [])
            if talking_points:
                st.markdown("**Talking Points for Customer Call:**")
                for point in talking_points:
                    st.markdown(f"- {point}")

            st.markdown("---")
            risk_warning = strategy.get("risk_if_no_action", "")
            if risk_warning:
                st.error(f"**If no action is taken:** {risk_warning.replace('$', '\\$')}")

        else:
            st.error(f"Strategy generation failed: {result.get('error', 'Unknown error')}")

    st.markdown("---")

    #Revenue impact estimate
    st.subheader("Revenue Impact")
    impact_col1, impact_col2 = st.columns(2)
    with impact_col1:
        annual_revenue = customer["mrr"] * 12
        st.metric("Annual Revenue at Risk", f"${annual_revenue:,.0f}")
    with impact_col2:
        #Rough estimate with 40% of revenue at risk saved
        saved_estimate = annual_revenue * 0.4 * (customer["churn_probability"] / 100)
        st.metric("Est. Revenue Saved by Acting", f"${saved_estimate:,.0f}",
                   delta="if intervention succeeds")

# ══════════════════════════════════════════════════════════════
# ASK YOUR DATA TAB — LLM Feature 
# ══════════════════════════════════════════════════════════════
with tab_query:

    st.subheader("Ask Your Data")
    st.caption("Ask questions about your customer portfolio in plain English. Powered by AI")

    st.markdown("""
    **Example questions you can ask:**
    - *Which high-risk customers have the highest revenue?*
    - *Compare month-to-month vs annual contract churn rates*
    - *What are the common traits of my top 5 riskiest customers?*
    - *Which customers should I focus on this week?*
    - *How many customers have low feature adoption and high support tickets?*
    """)

    # Initialize chat history in session state
    if "nl_query_history" not in st.session_state:
        st.session_state.nl_query_history = []

    # Question input
    user_question = st.text_input("Ask a question about your customers:", 
                                   placeholder="e.g., Which customers have the highest MRR but are at high risk?",
                                   key="nl_query_input")

    if st.button("Ask", key="nl_query_btn", type="primary") and user_question:
        with st.spinner("🔍 Call 1: Planning analysis... 📊 Call 2: Querying your data..."):
            from llm_utils import natural_language_query
            try:
                answer = natural_language_query(user_question, df_filtered)
                # Save to history
                st.session_state.nl_query_history.append({
                    "question": user_question,
                    "answer": answer
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display conversation history (most recent first)
    for i, qa in enumerate(reversed(st.session_state.nl_query_history)):
        st.markdown(f"**You:** {qa['question']}")
        st.markdown(qa["answer"].replace("$", "\\$"))
        if i < len(st.session_state.nl_query_history) - 1:
            st.markdown("---")

# ══════════════════════════════════════════════════════════════
# MODEL EVALUATION TAB — Professor feedback: deeper evaluation
# ══════════════════════════════════════════════════════════════
with tab_model:

    st.subheader("Model Performance & Evaluation")
    st.caption("Transparency: how well does the churn prediction model perform, and what are the business implications?")

    # Load saved evaluation metrics
    import json as json_lib
    with open("models/eval_metrics.json") as f:
        metrics = json_lib.load(f)

    # ── Metrics row ──
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    with m_col2:
        st.metric("Precision", f"{metrics['precision']:.1%}")
    with m_col3:
        st.metric("Recall", f"{metrics['recall']:.1%}")
    with m_col4:
        st.metric("F1 Score", f"{metrics['f1_score']:.1%}")

    st.markdown("---")

    # ── Confusion matrix visual ──
    st.subheader("Confusion Matrix")

    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["true_negative"], cm["false_positive"]],
         [cm["false_negative"], cm["true_positive"]]],
        index=["Actually Stayed", "Actually Churned"],
        columns=["Predicted Stay", "Predicted Churn"]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale=["#2E7D5B", "#D4943A", "#C4314B"],
        labels=dict(color="Count"),
        aspect="auto"
    )
    fig_cm.update_layout(height=350, title="")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # ── Business impact explanation ──
    st.subheader("Business Impact of Prediction Errors")

    impact_col1, impact_col2 = st.columns(2)

    with impact_col1:
        st.error(f"""
        **False Negatives: {cm['false_negative']} customers**

        Customer churned but we predicted they'd stay.
        **The CSM was never warned — revenue lost.**

        This is the costly mistake. Each missed churner
        means lost MRR with no chance to intervene.
        """)

    with impact_col2:
        st.warning(f"""
        **False Positives: {cm['false_positive']} customers**

        We predicted churn but the customer stayed.
        **CSM makes an unnecessary check-in call.**

        Low cost — and the outreach can still
        strengthen the customer relationship.
        """)

    st.markdown("---")

    # ── Model details ──
    st.subheader("Model Specification")

    spec_col1, spec_col2 = st.columns(2)

    with spec_col1:
        st.markdown(f"""
        **Algorithm:** Random Forest Classifier

        **Hyperparameters:**
        - Trees: 100
        - Max depth: 8
        - Class weight: balanced (compensates for fewer churners)

        **Training data:** {metrics['train_size']} customers (80% split)

        **Test data:** {metrics['test_size']} customers (20% split)
        """)

    with spec_col2:
        st.markdown(f"""
        **Why Random Forest?**
        - Handles mixed feature types (numerical + categorical)
        - Provides feature importance rankings
        - Resistant to overfitting with proper depth limits
        - Good baseline for prototype; production would compare
          against XGBoost, Logistic Regression, and ensemble methods

        **Why balanced class weights?**
        - Dataset has ~28.6% churn rate (imbalanced)
        - Without balancing, model would default to predicting "stay"
        """)

    st.markdown("---")

    # ── Known limitations ──
    st.subheader("Known Limitations & Roadmap")
    st.markdown(f"""
    **Current recall is {metrics['recall']:.0%}** — the model catches only {metrics['recall']:.0%} of actual churners.
    This is expected for a prototype trained on synthetic data. Improvements for production:

    1. **Real data** — Train on actual company CRM exports with historical churn labels
    2. **Tune for recall** — In churn prevention, missing a churner (FN) costs more than a false alarm (FP).
       Future versions would optimize the probability threshold to maximize recall.
    3. **More algorithms** — Compare Random Forest against XGBoost, Logistic Regression, and gradient boosting
    4. **Feature engineering** — Add time-series features (login trend, MRR change over time)
    5. **User-uploaded data** — Allow each company to upload their own CSV, map columns, and retrain the model
    """)
