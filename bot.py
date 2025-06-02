import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import io

# Load API key from .env or fallback (use your key securely in production)
load_dotenv()
api_key = "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"

# Initialize Groq client
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# Get enrichment from LLaMA3
def get_company_summary(company_name):
    prompt = f"""
Please ONLY respond with a valid JSON object with EXACTLY these keys:

{{
  "name": "",
  "industry": "",
  "headquarters": "",
  "founded": "",
  "employee_count": "",
  "key_products_or_services": "",
  "summary": "",
  "website": "",
  "ai_automation_idea": ""
}}

Provide realistic answers for the company named "{company_name}".
Make the summary detailed (2-3 sentences).
If any info is unknown, use "unknown".
DO NOT add anything outside this JSON.
"""
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()

        json_start = reply.find("{")
        json_end = reply.rfind("}") + 1

        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON object found in response.")

        json_string = reply[json_start:json_end]
        return json.loads(json_string)

    except Exception as e:
        return {"name": company_name, "error": str(e)}

# --- Streamlit UI ---
st.title("🤖 AI Company Enrichment Tool (LLaMA3)")
uploaded_file = st.file_uploader("Upload CSV with 'company_name' column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "company_name" not in df.columns:
        st.error("❌ CSV must contain a 'company_name' column.")
    else:
        st.success("✅ File uploaded. Ready to enrich.")
        if st.button("Run Enrichment"):
            enriched_data = []
            with st.spinner("Fetching data from LLaMA3..."):
                for name in df["company_name"]:
                    enriched = get_company_summary(str(name))
                    enriched_data.append(enriched)

            enriched_df = pd.DataFrame(enriched_data)
            st.success("✅ Enrichment complete!")
            st.dataframe(enriched_df)

            # Download button
            csv = enriched_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Enriched CSV",
                data=csv,
                file_name="enriched_companies.csv",
                mime="text/csv"
            )
