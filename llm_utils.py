# llm_utils.py — All LLM logic lives here (Groq API)
# This file handles: AI Account Strategist, Natural Language Query

import os
import json
from groq import Groq

def get_groq_client():
    """Create a Groq client. Tries st.secrets first (for Streamlit Cloud), 
    then falls back to environment variable."""
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("No GROQ_API_KEY found. Set it in Streamlit secrets or environment.")
    
    return Groq(api_key=api_key)


def generate_retention_strategy(customer_data: dict, risk_drivers: list) -> dict:
    """
    AI Account Strategist — Category A+B
    Takes customer data + risk drivers, returns a structured retention strategy.
    
    Category A (sophisticated prompt): injects all customer data, specifies JSON output,
    includes persona and business rules.
    Category B (post-processing): parses the JSON, validates structure, handles errors.
    """
    
    client = get_groq_client()
    
    # ── BUILD THE SOPHISTICATED PROMPT (Category A) ──
    prompt = f"""You are a senior SaaS Customer Success strategist with 15 years of experience 
preventing customer churn. You analyze customer data and create actionable retention plans.

CUSTOMER PROFILE:
- Company: {customer_data['company']}
- Monthly Revenue (MRR): ${customer_data['mrr']:,.0f}
- Tenure: {customer_data['tenure_months']} months
- Contract: {customer_data['contract']}
- Monthly Charges: ${customer_data['monthly_charges']:,.2f}
- Logins (last 30 days): {customer_data['logins_last_30d']}
- Feature Adoption: {customer_data['features_used_pct']}%
- Support Tickets (last 90 days): {customer_data['support_tickets_last_90d']}
- Payment Method: {customer_data['payment_method']}
- Tech Support: {customer_data['tech_support']}
- Churn Probability: {customer_data['churn_probability']}%
- Risk Tier: {customer_data['risk_tier']}

IDENTIFIED RISK DRIVERS:
{chr(10).join(f'- {driver}' for driver in risk_drivers)}

Based on this data, create a personalized retention strategy.

RULES:
- Be specific to THIS customer's data. Do not give generic advice.
- If MRR is below $200, never suggest discounts — suggest value demonstration instead.
- If logins are low, prioritize re-engagement before anything else.
- If support tickets are high, address frustration first before upselling.
- Maximum 3 action steps. Each must be concrete and immediately actionable.

Respond ONLY with valid JSON in exactly this format, no other text.

EXAMPLE of perfect output (for a different customer — do NOT copy this content):
{{
    "urgency": "high",
    "root_cause": "Customer has extremely low feature adoption (18%) combined with a month-to-month contract, suggesting they haven't found enough value to commit long-term.",
    "strategy_name": "Value Discovery Sprint",
    "actions": [
        {{
            "step": 1,
            "action": "Schedule a 30-minute product walkthrough focused on the 3 features most relevant to their industry",
            "owner": "CSM",
            "timeline": "Within 48 hours",
            "expected_outcome": "Customer discovers at least 2 features they weren't using that solve their pain points"
        }},
        {{
            "step": 2,
            "action": "Set up weekly check-in calls for the next month to track feature adoption progress",
            "owner": "CSM",
            "timeline": "Starting next week",
            "expected_outcome": "Feature adoption increases from 18% to at least 40% within 30 days"
        }},
        {{
            "step": 3,
            "action": "Present ROI report showing time/cost savings from adopted features, then propose annual contract",
            "owner": "Account Manager",
            "timeline": "After 30 days",
            "expected_outcome": "Customer converts to annual contract based on demonstrated value"
        }}
    ],
    "talking_points": [
        "We noticed you might not be using some features that could save your team significant time",
        "Companies similar to yours typically see a 30% efficiency gain from our advanced reporting tools"
    ],
    "risk_if_no_action": "Without intervention, this $350/mo account will likely churn within 60 days — a $4,200 annual revenue loss. The month-to-month contract means there is no commitment barrier to leaving."
}}

Now generate a strategy for the customer above. Output ONLY valid JSON, nothing else:"""

    # ── CALL THE LLM ──
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a JSON-only response bot. Never include markdown, backticks, or explanations. Only output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3  # Low temperature = more focused, less random
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # ── POST-PROCESS THE RESPONSE (Category B) ──
        # Clean up common LLM issues: sometimes they wrap JSON in backticks
        if raw_response.startswith("```"):
            raw_response = raw_response.split("```")[1]
            if raw_response.startswith("json"):
                raw_response = raw_response[4:]
            raw_response = raw_response.strip()
        
        # Parse the JSON
        strategy = json.loads(raw_response)
        
        # Validate required fields exist
        required_fields = ["urgency", "root_cause", "strategy_name", "actions", "talking_points", "risk_if_no_action"]
        for field in required_fields:
            if field not in strategy:
                strategy[field] = "Not available"
        
        # Validate each action has required subfields
        if isinstance(strategy["actions"], list):
            for action in strategy["actions"]:
                for key in ["step", "action", "owner", "timeline", "expected_outcome"]:
                    if key not in action:
                        action[key] = "Not specified"
        
        return {"success": True, "strategy": strategy}
    
    except json.JSONDecodeError:
        return {
            "success": False, 
            "error": "LLM returned invalid JSON. Try again.",
            "raw": raw_response if 'raw_response' in dir() else "No response"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    
def natural_language_query(question: str, df) -> str:
    """
    Natural Language Query — Category C (Tool Use / Multi-Call)
    
    TRUE tool-use pattern:
    - We define Python functions as "tools" and describe them to the LLM
    - LLM Call 1: receives the question + tool definitions, DECIDES which tool to call
    - We execute that tool (Python function) with the LLM's chosen parameters
    - LLM Call 2: receives the tool result and generates a natural language answer
    """
    
    try:
        client = get_groq_client()
    except Exception as e:
        return f"⚠️ Could not connect to AI service: {str(e)}"
    
    if len(df) == 0:
        return "⚠️ No customers match your current filters. Try adjusting the sidebar filters."

    # ══════════════════════════════════════════════════════════
    # STEP 1: Define the tools (Python functions the LLM can call)
    # ══════════════════════════════════════════════════════════
    
    def get_high_risk_customers(min_mrr=0, limit=10):
        """Returns high-risk customers, optionally filtered by minimum MRR."""
        filtered = df[df["risk_tier"] == "High"]
        if min_mrr > 0:
            filtered = filtered[filtered["mrr"] >= min_mrr]
        filtered = filtered.sort_values("churn_probability", ascending=False).head(limit)
        rows = []
        for _, r in filtered.iterrows():
            rows.append(f"{r['company']} | MRR: ${r['mrr']:.0f} | Churn: {r['churn_probability']}% | "
                       f"Tenure: {r['tenure_months']}mo | Contract: {r['contract']} | "
                       f"Logins: {r['logins_last_30d']} | Features: {r['features_used_pct']}%")
        return "\n".join(rows) if rows else "No high-risk customers found matching criteria."
    
    def get_portfolio_summary():
        """Returns overall portfolio statistics."""
        total = len(df)
        high = len(df[df["risk_tier"] == "High"])
        medium = len(df[df["risk_tier"] == "Medium"])
        low = len(df[df["risk_tier"] == "Low"])
        return (f"Total customers: {total} | High risk: {high} | Medium: {medium} | Low: {low}\n"
                f"Total MRR: ${df['mrr'].sum():,.0f}/mo | MRR at risk (high): ${df[df['risk_tier']=='High']['mrr'].sum():,.0f}/mo\n"
                f"Avg churn probability: {df['churn_probability'].mean():.1f}%\n"
                f"Avg logins: {df['logins_last_30d'].mean():.1f} | Avg feature adoption: {df['features_used_pct'].mean():.1f}%\n"
                f"Contract breakdown: {df['contract'].value_counts().to_dict()}")
    
    def get_customers_by_filter(risk_tier="all", contract="all", sort_by="churn_probability", limit=10):
        """Returns customers filtered and sorted by specified criteria."""
        filtered = df.copy()
        if risk_tier != "all":
            filtered = filtered[filtered["risk_tier"] == risk_tier]
        if contract != "all":
            filtered = filtered[filtered["contract"] == contract]
        
        if sort_by in filtered.columns:
            ascending = sort_by in ["logins_last_30d", "features_used_pct"]  # low = bad for these
            filtered = filtered.sort_values(sort_by, ascending=ascending).head(limit)
        
        rows = []
        for _, r in filtered.iterrows():
            rows.append(f"{r['company']} | MRR: ${r['mrr']:.0f} | Churn: {r['churn_probability']}% | "
                       f"Tier: {r['risk_tier']} | Tenure: {r['tenure_months']}mo | "
                       f"Contract: {r['contract']} | Logins: {r['logins_last_30d']} | "
                       f"Features: {r['features_used_pct']}% | Tickets: {r['support_tickets_last_90d']}")
        return "\n".join(rows) if rows else "No customers found matching criteria."

    def compare_segments(group_by="contract"):
        """Compares customer segments by a given field (contract, risk_tier, payment_method)."""
        if group_by not in df.columns:
            return f"Cannot group by '{group_by}'. Available: contract, risk_tier, payment_method, internet_service"
        
        summary = df.groupby(group_by).agg(
            count=("customer_id", "count"),
            avg_churn=("churn_probability", "mean"),
            avg_mrr=("mrr", "mean"),
            avg_logins=("logins_last_30d", "mean"),
            avg_features=("features_used_pct", "mean")
        ).round(1)
        
        rows = []
        for idx, r in summary.iterrows():
            rows.append(f"{idx}: {int(r['count'])} customers | Avg churn: {r['avg_churn']}% | "
                       f"Avg MRR: ${r['avg_mrr']:.0f} | Avg logins: {r['avg_logins']} | "
                       f"Avg features: {r['avg_features']}%")
        return "\n".join(rows)

    # Map function names to actual functions
    available_tools = {
        "get_high_risk_customers": get_high_risk_customers,
        "get_portfolio_summary": get_portfolio_summary,
        "get_customers_by_filter": get_customers_by_filter,
        "compare_segments": compare_segments,
    }

    # ══════════════════════════════════════════════════════════
    # STEP 2: Tell the LLM about the tools (OpenAI-compatible format)
    # ══════════════════════════════════════════════════════════
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_high_risk_customers",
                "description": "Get a list of high-risk customers sorted by churn probability. Can filter by minimum MRR.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_mrr": {"type": "number", "description": "Minimum MRR to filter by. Use 0 for no filter."},
                        "limit": {"type": "integer", "description": "Max number of customers to return. Default 10."}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_portfolio_summary",
                "description": "Get overall portfolio statistics: total customers, risk distribution, MRR totals, averages.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_customers_by_filter",
                "description": "Get customers filtered by risk tier and/or contract type, sorted by a chosen field.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "risk_tier": {"type": "string", "description": "Filter by risk tier: 'High', 'Medium', 'Low', or 'all'"},
                        "contract": {"type": "string", "description": "Filter by contract: 'Month-to-month', 'One year', 'Two year', or 'all'"},
                        "sort_by": {"type": "string", "description": "Column to sort by: 'churn_probability', 'mrr', 'logins_last_30d', 'features_used_pct', 'support_tickets_last_90d'"},
                        "limit": {"type": "integer", "description": "Max results to return. Default 10."}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_segments",
                "description": "Compare customer segments by grouping on a field. Shows averages for churn, MRR, logins, features per group.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group_by": {"type": "string", "description": "Field to group by: 'contract', 'risk_tier', 'payment_method', 'internet_service'"}
                    },
                    "required": ["group_by"]
                }
            }
        }
    ]

    # ══════════════════════════════════════════════════════════
    # STEP 3: LLM CALL 1 — LLM decides which tool to call
    # ══════════════════════════════════════════════════════════
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a data analyst for a Customer Success team. Use the available tools to answer questions about customer churn data. Always use a tool — never guess."},
                {"role": "user", "content": question}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
            temperature=0.1
        )
        
        message = response.choices[0].message
        
        # ══════════════════════════════════════════════════════
        # STEP 4: Execute the tool the LLM chose
        # ══════════════════════════════════════════════════════
        
        if message.tool_calls:
            tool_call = message.tool_calls[0]  # Take the first tool call
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Run the actual Python function
            if function_name in available_tools:
                tool_result = available_tools[function_name](**function_args)
            else:
                tool_result = f"Unknown tool: {function_name}"
            
            # ══════════════════════════════════════════════════
            # STEP 5: LLM CALL 2 — Answer with the tool result
            # ══════════════════════════════════════════════════
            
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst. Answer the user's question based on the tool results. Be specific, reference actual company names and numbers. Keep it concise."},
                    {"role": "user", "content": question},
                    message,  # The assistant's message with tool_calls
                    {"role": "tool", "content": tool_result, "tool_call_id": tool_call.id}
                ],
                max_tokens=600,
                temperature=0.2
            )
            
            answer = final_response.choices[0].message.content.strip()
            
            # Return with transparency: show which tool was called
            return (f"**Tool used:** `{function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})`\n\n"
                    f"---\n\n{answer}")
        
        else:
            # LLM didn't call a tool — just gave a direct answer
            return message.content.strip() if message.content else "No response generated."
    
    except Exception as e:
        return f"⚠️ AI query failed: {str(e)}. Please try again."