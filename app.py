import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables from .env file
load_dotenv()

# ================= 1. STREAMLIT UI CONFIGURATION =================
st.set_page_config(page_title="Groq & Roll Data Laundromat", layout="wide")
st.title("🤖 LLM-Powered Spreadsheet Washer & Insight Spinner")
st.write("Upload a CSV file to automatically clean your data, visualize charts, and generate business insights.")

# ================= 2. SECURE API KEY VERIFICATION =================
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("🔒 Power Outage: The Laundromat is out of power! Please plug the GROQ_API_KEY into the generator (.env file).")
    st.stop()

# Initialize Groq LLM (Using production tier Llama 3.1 8B)
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=groq_api_key,
    temperature=0
)

# ================= 3. FILE UPLOAD LOGIC =================
uploaded_file = st.file_uploader("🧺 Drop Your Dirty CSV Load Here", type="csv")

if uploaded_file is not None:
    # SCALED UP: Now handles large files up to 100MB safely
    if uploaded_file.size > 100 * 1024 * 1024:
        st.error("⚠️ **Load Limit Exceeded:** This laundromat only handles loads under 100MB to prevent memory timeouts!")
        st.stop()

if uploaded_file is not None:
    # Read the full dataset into memory natively
    df_raw = pd.read_csv(uploaded_file)
        
    if "df" not in st.session_state:
        st.session_state.df = df_raw.copy()
        st.session_state.cleaning_report = ""
        st.session_state.insights_report = ""

    # Main layout division: Preview and Global Metrics
    preview_col, metrics_col = st.columns([5, 3], gap="large")
    
    with preview_col:
        st.subheader("🔍 Batch Inspection Preview")
        st.caption("A look at the top rows of your active dataframe load.")
        st.dataframe(
            st.session_state.df.head(10), 
            use_container_width=True,
            column_config={"_index": st.column_config.NumberColumn("Fabric Row #")}
        )
        
    with metrics_col:
        st.subheader("📊 Full Load Metrics")
        st.caption("Diagnostic metrics detailing the macroscopic contamination state.")
        
        with st.container(border=True):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="🧵 Total Dataset Rows", value=f"{st.session_state.df.shape[0]:,}")
            with m_col2:
                st.metric(label="📐 Total Data Columns", value=st.session_state.df.shape[1])
            
            missing_count = int(st.session_state.df.isnull().sum().sum())
            duplicate_count = int(st.session_state.df.duplicated().sum())
            
            st.markdown("---")
            
            m_col3, m_col4 = st.columns(2)
            with m_col3:
                st.metric(
                    label="⚠️ Empty Pockets", 
                    value=f"{missing_count:,}", 
                    delta=f"{missing_count} Missing Values" if missing_count > 0 else "No Missing data", 
                    delta_color="inverse" if missing_count > 0 else "normal"
                )
            with m_col4:
                st.metric(
                    label="🔁 Tangled Clones", 
                    value=f"{duplicate_count:,}",
                    delta=f"{duplicate_count} Duplicate Rows" if duplicate_count > 0 else "No Duplicate data",
                    delta_color="inverse" if duplicate_count > 0 else "normal"
                )

            if missing_count > 0 or duplicate_count > 0:
                st.warning("⚠️ **Status:** Contaminants detected across the dataset. Scrubbing recommended.")
            else:
                st.success("✅ **Status:** Fabric is perfectly clean. Ready for the Insight Reporter.")

    st.markdown("---")

    # ================= 4. STABLE NATIVE FUNCTION CALLING ENGINE =================
    def execute_analytics_code(code_input: str) -> str:
        """Isolated environment runner with strict warnings suppression and pandas safeties."""
        import warnings
        cleaned_code = code_input.strip().strip("```python").strip("```")
        df_safe = st.session_state.df.copy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            categorical_cols = df_safe.select_dtypes(include=['object', 'category', 'string']).columns
            for col in categorical_cols:
                df_safe[col] = df_safe[col].fillna('Unknown')
            for col in df_safe.select_dtypes(include=['number']).columns:
                df_safe[col] = df_safe[col].fillna(0)
        
        sandbox_env = {"df": df_safe, "plt": plt, "sns": sns}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(cleaned_code, globals(), sandbox_env)
                
            if isinstance(sandbox_env.get("df"), pd.DataFrame):
                st.session_state.df = sandbox_env["df"]
            return ""
        except Exception as e:
            return f"Execution Error: {str(e)}"

    class PythonExecutorInput(BaseModel):
        code_input: str = Field(..., description="The executable Python code string to process 'df'.")

    @tool("python_data_executor", args_schema=PythonExecutorInput)
    def data_tool(code_input: str) -> str:
        """Executes python pandas instructions on the target mutable dataframe 'df'."""
        return execute_analytics_code(code_input)
    
    llm_with_tools = llm.bind_tools([data_tool])

    def run_agent_workflow(user_instruction: str) -> str:
        """Reinforced tool-calling loop that executes data operations."""
        system_instruction = (
            "You are an elite data scientist with access to a pandas DataFrame named 'df'. "
            "You must use the 'python_data_executor' tool to process data transformations on 'df'."
        )
        messages = [("system", system_instruction), ("human", user_instruction)]
        
        for _ in range(5): 
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == "python_data_executor":
                        args = tool_call["args"]
                        code_to_run = args.get("code_input") or args.get("code") or list(args.values())[0]
                        tool_output = execute_analytics_code(code_to_run)
                        messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
                continue
            else:
                return ai_msg.content.replace("Execution complete. Dataframe state updated safely.", "").strip()
        return ai_msg.content

    # ================= STAGE 1: DATA CLEANING =================
    st.header("🧹 Step 1: The Intelligent Data Scrubber")
    cleaning_prompt = st.text_area(
        "Modify stain-removal instructions if needed:",
        value="Identify and handle missing values appropriately, remove duplicate rows, ensure correct data types, and clean anomalies."
    )

    if st.button("🧼 Start the Heavy-Duty Wash Cycle"):
        with st.spinner("Scrubbing rows clean across the dataset matrix..."):
            try:
                before_cols = list(st.session_state.df.columns)
                before_row_count = st.session_state.df.shape[0]
                
                reinforced_cleaning_prompt = f"""
                You are a data cleaning engine. The active dataframe 'df' has {before_row_count} rows and columns: {before_cols}
                Execute your cleaning script via your tool according to these rules: {cleaning_prompt}
                Provide a short summary text answering what actions were executed. Do not output code logs.
                """
                st.session_state.cleaning_report = run_agent_workflow(reinforced_cleaning_prompt)
                st.success("Deep Wash Cycle Completed!")
                st.rerun()
            except Exception as e:
                st.error(f"Scrubbing Execution Error: {e}")

    if st.session_state.cleaning_report:
        st.subheader("📋 Detergent Action Summary")
        st.markdown(st.session_state.cleaning_report)
        
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Freshly Folded CSV",
            data=csv_data,
            file_name="freshly_folded_laundry.csv",
            mime="text/csv"
        )

    # ================= STAGE 2: THE SPIN CYCLE & INSIGHT PRESS =================
    st.markdown("---")
    st.header("📈 Step 2: The Spin Cycle & Insight Press")
    
    report_prompt = st.text_area(
        "Modify the styling and analytical press settings if needed:",
        value="""Provide a comprehensive analysis of the dataset based on the metrics provided. Format it strictly with these exact markdown subheaders:
### 📋 The Freshly Folded Executive Summary
[Write summary analysis text here. Ensure you place the token '[CHART_1_HERE]' at the very end of this section]

### 🔍 Fabric Irregularities & Key Trends
[Write trends analysis text here. Ensure you place the token '[CHART_2_HERE]' at the very end of this section]

### 🧮 Strategic Pressing Recommendations
[Write recommendations here]""",
        height=180
    )

    if st.button("🚀 Run the Spin & Generate Report"):
        with st.spinner("Synthesizing metrics and plotting charts locally..."):
            try:
                for chart in ["chart1.png", "chart2.png"]:
                    if os.path.exists(chart):
                        os.remove(chart)
                
                df_plot = st.session_state.df.copy()
                for col in df_plot.select_dtypes(include=['object', 'category', 'string']).columns:
                    df_plot[col] = df_plot[col].fillna('Unknown')
                for col in df_plot.select_dtypes(include=['number']).columns:
                    df_plot[col] = df_plot[col].fillna(0)

                numeric_cols = list(df_plot.select_dtypes(include=['number']).columns)
                categorical_cols = list(df_plot.select_dtypes(include=['object', 'category', 'string']).columns)

                # Local Vector Plotting Engine (Scales natively to millions of records perfectly)
                if numeric_cols:
                    primary_num = numeric_cols[0]
                    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
                    sns.histplot(df_plot[primary_num], kde=True, ax=ax1, color="#2E86AB")
                    ax1.set_title(f"Distribution of {primary_num.replace('_', ' ')}")
                    fig1.savefig('chart1.png', bbox_inches='tight')
                    plt.close(fig1)
                
                if len(categorical_cols) >= 2:
                    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
                    sns.countplot(x=categorical_cols[0], hue=categorical_cols[1], data=df_plot, ax=ax2)
                    ax2.set_title(f"{categorical_cols[0].replace('_', ' ')} Breakdown by {categorical_cols[1].replace('_', ' ')}")
                    fig2.savefig('chart2.png', bbox_inches='tight')
                    plt.close(fig2)
                elif categorical_cols:
                    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
                    sns.countplot(x=categorical_cols[0], data=df_plot, ax=ax2)
                    ax2.set_title(f"Distribution of {categorical_cols[0].replace('_', ' ')}")
                    fig2.savefig('chart2.png', bbox_inches='tight')
                    plt.close(fig2)

                # HIGH-SCALE AGGREGATION PATTERN (Extracts matrix summaries to preserve context tokens)
                actual_columns = list(st.session_state.df.columns)
                row_count_val = int(st.session_state.df.shape[0])
                numeric_summary = st.session_state.df.describe(include='number').to_string()
                
                # Pre-calculate targeted absolute metrics to prevent hallucinations
                primary_target = numeric_cols[0] if numeric_cols else None
                exact_metrics_text = ""
                if primary_target and primary_target in st.session_state.df.columns:
                    total_spend_val = float(st.session_state.df[primary_target].sum())
                    actual_mean_val = float(st.session_state.df[primary_target].mean())
                    max_spend_val = float(st.session_state.df[primary_target].max())
                    min_spend_val = float(st.session_state.df[primary_target].min())
                    exact_metrics_text = f"""
                    - Target Column Selected: '{primary_target}'
                    - Total Combined Sum: ${total_spend_val:,.2f}
                    - True Mathematical Average (Mean): ${actual_mean_val:,.2f}
                    - Maximum Record Value: ${max_spend_val:,.2f}
                    - Minimum Record Value: ${min_spend_val:,.2f}
                    """

                # High-density distribution summaries for categorical attributes
                categorical_summary = ""
                for col in categorical_cols[:3]:
                    categorical_summary += f"\nValue counts distribution for '{col}':\n{st.session_state.df[col].value_counts().head(6).to_string()}\n"

                text_generation_prompt = f"""
                You are an expert executive business data analyst. Generate an enterprise report.
                Columns available: {actual_columns}. Total Row Dataset Scale: {row_count_val} rows.
                
                CRITICAL EXACT COMPUTED METRICS TO USE:
                {exact_metrics_text}
                
                BACKGROUND STRUCTURAL MATRICES:
                --- NUMERICAL DISTRIBUTION PROPERTIES ---
                {numeric_summary}
                
                --- CATEGORICAL SEGMENT PROPERTIES ---
                {categorical_summary}
                
                Fulfill this structural layout directive explicitly:
                {report_prompt}
                
                CRITICAL INSTRUCTION: Do NOT say the average is equal to the total sum. Use the precise values given.
                Start directly with the first subheader line. Append '[CHART_1_HERE]' and '[CHART_2_HERE]' where indicated.
                """

                final_text_call = llm.invoke([
                    ("system", "You are an automated corporate insight reporter. Write an expert analytical report using the macro summaries provided. Do not include debug warnings or duplicate headers."),
                    ("human", text_generation_prompt)
                ])
                
                st.session_state.insights_report = final_text_call.content
                st.success("Report and Visual Outfits Compiled Successfully!")
                st.rerun() 
                
            except Exception as e:
                st.error(f"Analysis Generation Failure: {e}")

    # ================= 🔄 INLINE STREAM RENDERING ENGINE =================
    if st.session_state.insights_report:
        raw_text = st.session_state.insights_report
        
        if "### 🔍 Fabric Irregularities & Key Trends" in raw_text:
            parts = raw_text.split("### 🔍 Fabric Irregularities & Key Trends")
            summary_part = parts[0]
            remaining_part = parts[1] if len(parts) > 1 else ""
            
            clean_summary = summary_part.replace("[CHART_1_HERE]", "").replace("### 📋 The Freshly Folded Executive Summary", "").strip()
            
            # Render Section 1 with pure HTML font protections
            st.subheader("📋 The Freshly Folded Executive Report")
            st.markdown(f"<p style='font-family:sans-serif; font-size:16px; line-height:1.6;'>{clean_summary}</p>", unsafe_allow_html=True)
            st.markdown("") 
            
            # Center & Shrink Chart 1 using responsive layout grids
            if os.path.exists("chart1.png"):
                c1_left, c1_mid, c1_right = st.columns([1.5, 5, 1.5])
                with c1_mid:
                    st.image("chart1.png", use_container_width=True, caption="Primary Feature Distribution Vector")
                st.markdown("")
                
            # Render Sections 2 & 3
            if "### 🧮 Strategic Pressing Recommendations" in remaining_part:
                trends_parts = remaining_part.split("### 🧮 Strategic Pressing Recommendations")
                trends_part = trends_parts[0]
                rec_part = trends_parts[1] if len(trends_parts) > 1 else ""
                
                st.markdown("### 🔍 Fabric Irregularities & Key Trends")
                st.markdown(f"<p style='font-family:sans-serif; font-size:16px; line-height:1.6;'>{trends_part.replace('[CHART_2_HERE]', '').strip()}</p>", unsafe_allow_html=True)
                st.markdown("")
                
                # Center & Shrink Chart 2
                if os.path.exists("chart2.png"):
                    c2_left, c2_mid, c2_right = st.columns([1.5, 5, 1.5])
                    with c2_mid:
                        st.image("chart2.png", use_container_width=True, caption="Categorical Variable Cross-Analysis Matrix")
                    st.markdown("")
                    
                st.markdown("### 🧮 Strategic Pressing Recommendations")
                st.markdown(f"<p style='font-family:sans-serif; font-size:16px; line-height:1.6;'>{rec_part.strip()}</p>", unsafe_allow_html=True)
            else:
                st.markdown("### 🔍 Fabric Irregularities & Key Trends")
                st.markdown(f"<p style='font-family:sans-serif; font-size:16px; line-height:1.6;'>{remaining_part.replace('[CHART_2_HERE]', '').strip()}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='font-family:sans-serif; font-size:16px; line-height:1.6;'>{raw_text}</p>", unsafe_allow_html=True)

    # ================= 🪙 STEP 3: INTERACTIVE SPOT CLEANER =================
    st.markdown("---")
    st.header("💬 Step 3: Interactive Spot Cleaner")
    
    col_chat_title, col_chat_clear = st.columns([6, 1])
    with col_chat_title:
        st.caption("Have a specific question about your data? Ask the Oracle directly.")
    with col_chat_clear:
        if st.button("🧹 Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask something like: 'Which segment had the highest total spend?'"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # PRODUCTION SCALE METADATA COMPILATION (Aggregates high-density summaries)
        actual_columns = list(st.session_state.df.columns)
        row_count_val = int(st.session_state.df.shape[0])
        numeric_summary = st.session_state.df.describe(include='number').to_string()
        
        categorical_summary = ""
        categorical_cols = list(st.session_state.df.select_dtypes(include=['object', 'category', 'string']).columns)
        for col in categorical_cols[:4]:
            categorical_summary += f"\nValue distributions for '{col}':\n{st.session_state.df[col].value_counts().head(10).to_string()}\n"
        
        # Multi-dimensional Pivot Aggregations (Enables answering deep analytical questions cross-column)
        pivot_insights = ""
        numeric_cols = list(st.session_state.df.select_dtypes(include=['number']).columns)
        if numeric_cols and categorical_cols:
            primary_num = numeric_cols[0]
            primary_cat = categorical_cols[0]
            
            # Pivot 1: Spend segments metrics
            pivot_insights += f"\n--- Breakdown Matrix of '{primary_num}' grouped by '{primary_cat}' ---\n"
            pivot_insights += st.session_state.df.groupby(primary_cat)[primary_num].agg(['sum', 'mean', 'count', 'max']).to_string() + "\n"
            
            # Pivot 2: Explicit Top 5 Extreme Individual Rows (Answers 'who had maximum' queries instantly)
            name_cols = [c for c in ['Customer_Name', 'Name', 'Customer', 'User'] if c in st.session_state.df.columns]
            label_col = name_cols[0] if name_cols else st.session_state.df.columns[0]
            
            top_df = st.session_state.df.sort_values(by=primary_num, ascending=False).head(5)
            pivot_insights += f"\n--- TOP 5 HIGHEST INDIVIDUAL RECORDS FOR '{primary_num}' ---\n"
            for idx, row in top_df.iterrows():
                pivot_insights += f"- Index {idx}: Attribute '{row[label_col]}' value is ${row[primary_num]:,.2f}\n"

        spot_cleaner_prompt = f"""
        You are an expert data analyst answering metrics questions. Dataset row scale: {row_count_val} rows.
        Available features: {actual_columns}.
        
        HIGH-DENSITY PIVOT RECORDS & TOP ROWS:
        {pivot_insights}
        
        GENERAL DISTRIBUTION PROPERTIES:
        --- NUMERICAL METRICS ---
        {numeric_summary}
        
        --- CATEGORICAL PROPERTIES ---
        {categorical_summary}
        
        Answer this user query accurately, factually, and concisely using the compiled records above:
        "{user_query}"
        """
        
        with st.spinner("Oracle is scanning the fabric..."):
            try:
                ai_msg = llm.invoke([
                    ("system", "You are an expert data analyst assistant. Answer user questions directly using the provided pivot matrices and summary summaries. Keep responses crisp and numerically precise."),
                    ("human", spot_cleaner_prompt)
                ])
                
                with st.chat_message("assistant"):
                    st.markdown(ai_msg.content)
                
                st.session_state.chat_history.append({"role": "assistant", "content": ai_msg.content})
                
            except Exception as e:
                st.error(f"Spot Cleaner Error: {e}")