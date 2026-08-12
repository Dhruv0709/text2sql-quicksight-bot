import streamlit as st
import pandas as pd
import os
import snowflake.connector
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

st.set_page_config(page_title="GenAI Text2SQL Analytics", layout="wide")

st.title("🤖 Text2SQL Analytics Engine (Groq Powered)")
st.caption("Ask natural language questions to query your Snowflake tables directly.")

@st.cache_resource
def get_snowflake_conn():
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    database = os.getenv("SNOWFLAKE_DATABASE", "QUICKSIGHT")
    schema = os.getenv("SNOWFLAKE_SCHEMA", "GENIE")

    return snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )

conn = None
try:
    conn = get_snowflake_conn()
    st.success("✅ Connected to Snowflake successfully!")
except Exception as e:
    st.error(f"❌ Snowflake Credentials/Connection Error: {e}")

GLOSSARY_PROMPT = """
You are an expert Data Engineer generating Snowflake SQL queries for QUICKSIGHT.GENIE schema based on these official synonyms:

AVAILABLE TABLES & MAPPINGS:
1. QUICKSIGHT.GENIE.EVENTS (Synonyms: Lead Site Visit, SV, Walkin)
   - Columns:
     * event_id (Synonyms: Sv_event_id, Walkin_id, Site_visit_id)
     * anarock_id (Synonyms: lead_id, customer_id)
     * event_start_time (Synonyms: SV_date, Event_date, Walkin_date, Site_visit_Date)
     * Status (Synonyms: New, Planned, Sv_planned, Site_visit_planned, Missed, SV_Missied, Done)
     * lead_status (Values: 'Junk', 'Failed', 'New', 'Fresh', 'In Call Center')
       - 'j &F', 'junk & failed', 'JNF' -> lead_status IN ('Junk', 'Failed')
     * agent_id (Synonyms: Bot_id, Walkin_genie_id, Lead_genie_id)
     * assisted (TRUE = human helped, FALSE = planned on Genie)

2. QUICKSIGHT.GENIE.BOOKING (ALWAYS SINGULAR 'BOOKING')
   - Columns: booking_id, lead_id, project_id, booking_date, amount.

3. QUICKSIGHT.GENIE.AI_CALLS
   - Columns: user_answered (answered/picked), duration, attempts, direction, dial_time.

4. QUICKSIGHT.GENIE.CP_LEADS (Synonyms: CP Leads, Partner Leads, Broker Leads)
   - Columns: id, anarock_id (lead_id), user_id, agent_id, project_id.

CRITICAL FILTERING RULES:
1. DO NOT ADD DATE OR TIME FILTERS (e.g. event_start_time, created_at, booking_date) UNLESS THE USER EXPLICITLY MENTIONS A DATE RANGE IN THEIR PROMPT.
2. "SV" or "Site Visit" means: Status = 'Done' in QUICKSIGHT.GENIE.EVENTS.

INSTRUCTIONS:
1. Output ONLY the raw executable Snowflake SQL query without markdown code blocks.
"""

def generate_sql(prompt_text):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None, "GROQ_API_KEY is missing in .env file"

        client = Groq(api_key=api_key.strip())
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": GLOSSARY_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0
        )
        
        sql = response.choices[0].message.content.strip()
        
        if "```" in sql:
            sql = sql.split("```")[1]
            if sql.startswith("sql"): 
                sql = sql[3:]
        return sql.strip(), None

    except Exception as e:
        return None, str(e)

user_query = st.text_input("💬 Ask a question about your data:", placeholder="e.g., uniqe event count who did sv in j &F lead status")

if user_query:
    st.subheader("⚙️ Generated SQL Query")
    
    with st.spinner("Generating query via Groq..."):
        generated_sql, err = generate_sql(user_query)

    if err:
        st.error(f"❌ Error: {err}")
    else:
        st.code(generated_sql, language="sql")
        
        st.subheader("📊 Query Results")
        if conn is None:
            st.error("❌ Cannot execute query because Snowflake connection is not active.")
        else:
            try:
                with conn.cursor() as cur:
                    cur.execute(generated_sql)
                    df = cur.fetch_pandas_all()
                    
                    if len(df) == 1 and len(df.columns) == 1:
                        col_name = df.columns[0]
                        val = df.iloc[0, 0]
                        st.metric(label=col_name, value=f"{val:,}")
                    else:
                        st.dataframe(df, use_container_width=True)
                        
                    st.success("Query executed successfully!")
            except Exception as e:
                st.error(f"Error executing query: {e}")
