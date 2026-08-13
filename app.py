import os
import pandas as pd
import streamlit as st
import snowflake.connector
from groq import Groq

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="GenAI Text2SQL Analytics", layout="wide")

st.title("🤖 Text2SQL AI Analytics Assistant")
st.caption("Ask questions in plain English to get instant tabular results, insights, and interactive visuals.")

# ---------------------------------------------------------
# 2. Snowflake Connection Setup (Auto-Reconnect handling)
# ---------------------------------------------------------
@st.cache_resource(ttl=3600)  # Refreshes connection every hour to prevent session expiry
def get_snowflake_conn():
    user = st.secrets["SNOWFLAKE_USER"]
    password = st.secrets["SNOWFLAKE_PASSWORD"]
    account = st.secrets["SNOWFLAKE_ACCOUNT"]
    warehouse = st.secrets.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    database = st.secrets.get("SNOWFLAKE_DATABASE", "QUICKSIGHT")
    schema = st.secrets.get("SNOWFLAKE_SCHEMA", "GENIE")

    return snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        host="fjbhcos-fe80032.snowflakecomputing.com",
        warehouse=warehouse,
        database=database,
        schema=schema,
        client_session_keep_alive=True
    )

try:
    conn = get_snowflake_conn()
    st.success("Connected to Snowflake successfully!")
except Exception as e:
    st.error(f"Snowflake Connection Error: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. Groq Client Initialization
# ---------------------------------------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is missing from Secrets!")
    st.stop()

client = Groq(api_key=groq_api_key)

# ---------------------------------------------------------
# 4. Text2SQL Generation Function
# ---------------------------------------------------------
def generate_sql_query(user_query):
    system_prompt = f"""
    You are an expert Snowflake SQL assistant. 
    Convert the user's natural language question into a valid Snowflake SQL query.
    
    Database Context:
    - Database: {st.secrets.get("SNOWFLAKE_DATABASE", "QUICKSIGHT")}
    - Schema: {st.secrets.get("SNOWFLAKE_SCHEMA", "GENIE")}
    
    Rules:
    1. Return ONLY the raw SQL query. Do not wrap it in markdown code blocks like ```sql ... ```.
    2. Do not include any explanations or extra text.
    3. Make sure table and column identifiers match Snowflake casing/syntax.
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------
# 5. Natural Language Summary Generator
# ---------------------------------------------------------
def generate_text_summary(user_query, df):
    # Take first 5 rows for quick context to keep summary fast
    sample_data = df.head(5).to_string()
    
    prompt = f"""
    User Asked: "{user_query}"
    Data Returned (sample):
    {sample_data}
    Total Rows: {len(df)}

    Provide a concise, 1-line plain language summary answering the user's question directly based on this data.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------
# 6. User Input & Query Execution
# ---------------------------------------------------------
user_input = st.text_input("💬 Ask a question about your data:", placeholder="e.g., unique event count top_reason wise")

if user_input:
    with st.spinner("Generating SQL query..."):
        try:
            generated_sql = generate_sql_query(user_input)
            with st.expander("🔍 View Generated SQL Query", expanded=False):
                st.code(generated_sql, language="sql")
        except Exception as e:
            st.error(f"Error generating SQL: {e}")
            st.stop()

    with st.spinner("Executing query on Snowflake..."):
        try:
            cursor = conn.cursor()
            cursor.execute(generated_sql)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
        except Exception as e:
            st.error(f"Error executing query: {e}")
            st.stop()

    # ---------------------------------------------------------
    # 7. Smart AI Output: Summary + Table + Auto-Chart
    # ---------------------------------------------------------
    if not df.empty:
        # 1. Natural Language Answer
        summary = generate_text_summary(user_input, df)
        st.info(f"💡 **AI Answer:** {summary}")

        # 2. Case: Single Metric (e.g. Total Count)
        if len(df) == 1 and len(df.columns) == 1:
            metric_val = df.iloc[0, 0]
            st.metric(label=df.columns[0], value=f"{metric_val:,}" if isinstance(metric_val, (int, float)) else str(metric_val))

        # 3. Case: Tabular & Visual Data
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📋 Query Results")
                st.dataframe(df, use_container_width=True)

            with col2:
                st.subheader("📊 Visual Representation")
                
                # Determine categorical vs numeric columns for smart plot
                cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
                num_cols = df.select_dtypes(include=['number']).columns.tolist()

                x_col = cat_cols[0] if cat_cols else df.columns[0]
                y_col = num_cols[0] if num_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

                # Render Bar Chart
                chart_df = df.set_index(x_col)[[y_col]]
                st.bar_chart(chart_df, use_container_width=True)

        # CSV Download Option
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv_data, file_name="query_results.csv", mime="text/csv")

    else:
        st.warning("Query executed successfully, but returned 0 rows.")
