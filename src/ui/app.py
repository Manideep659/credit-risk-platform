import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from src.talk_to_data.chatbot import OfflineCreditRiskChatbot

# Configure Page Layout and Tab Title
st.set_page_config(
    page_title="Apex Bank Credit Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Slate Blue & Slate Light Banking CSS Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;700&display=swap');
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #F8FAFC !important;
    }
    
    /* Title text styling */
    h1, h2, h3 {
        color: #1E3A8A !important;
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Custom metric card wrapper */
    .metric-card {
        background-color: white;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 4px;
        font-family: 'Outfit', sans-serif;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    /* Risk outcome callouts */
    .risk-banner {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
        color: white;
        margin-bottom: 20px;
    }
    
    /* Custom divider line */
    .premium-hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(37, 99, 235, 0), rgba(37, 99, 235, 0.75), rgba(37, 99, 235, 0));
        margin: 25px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model metrics
def load_metrics(metrics_path="models/metrics.json"):
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {
        "roc_auc": 0.761189,
        "precision": 0.167366,
        "recall": 0.690634,
        "f1_score": 0.269438
    }

# Navigation in Sidebar
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 20px;'><h2 style='color:#F8FAFC !important; margin:0;'>🏦 Apex Bank</h2><p style='color:#64748B; margin:0; font-size:0.85rem;'>Risk Platform v1.0</p></div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "NAVIGATION MENU",
    ["1. Executive Dashboard", 
     "2. EDA Insights", 
     "3. Credit Underwriting", 
     "4. SHAP Explainability", 
     "5. Automated Policy Rules", 
     "6. Relational SQL Chatbot"]
)

# Add sidebar footer info
st.sidebar.markdown("<div style='position: fixed; bottom: 20px; font-size: 0.8rem; color:#64748B;'>🔒 Secure Underwriting Session</div>", unsafe_allow_html=True)

# ==============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==============================================================================
if page == "1. Executive Dashboard":
    st.markdown("<h1>📊 Apex Bank Credit Risk Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Real-time baseline evaluation metrics, portfolio distributions, and analytical totals.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='premium-hr'>", unsafe_allow_html=True)
    
    # Ingest baseline metrics
    metrics = load_metrics()
    
    # 1. Show Model Evaluation Metrics
    st.markdown("<h3>🎯 Underwriting Model Baseline Performance</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['roc_auc']:.4f}</div><div class='metric-label'>ROC AUC Score</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['recall']*100:.2f}%</div><div class='metric-label'>Model Recall</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['precision']*100:.2f}%</div><div class='metric-label'>Model Precision</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['f1_score']:.4f}</div><div class='metric-label'>F1 Score</div></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Show Portfolio Totals
    st.markdown("<h3>📁 Portfolio and Database Statistics</h3>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown("<div class='metric-card'><div class='metric-value' style='color:#0D9488;'>307,511</div><div class='metric-label'>Total Ingested Applications</div></div>", unsafe_allow_html=True)
    with col6:
        st.markdown("<div class='metric-card'><div class='metric-value' style='color:#DC2626;'>8.07%</div><div class='metric-label'>Base Default Rate</div></div>", unsafe_allow_html=True)
    with col7:
        st.markdown("<div class='metric-card'><div class='metric-value' style='color:#0D9488;'>1,716,428</div><div class='metric-label'>Linked Credit Bureau Records</div></div>", unsafe_allow_html=True)
    with col8:
        st.markdown("<div class='metric-card'><div class='metric-value' style='color:#F59E0B;'>128</div><div class='metric-label'>Indexed Active Variables</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Target Distribution & Missing Data Visuals
    st.markdown("<h3>📊 Core Portfolio Analytical Views</h3>", unsafe_allow_html=True)
    col9, col10 = st.columns(2)
    with col9:
        target_img = "notebooks/charts/target_distribution.png"
        if os.path.exists(target_img):
            st.image(target_img, caption="Loan Default Balance (Imbalanced Classification)", use_container_width=True)
        else:
            st.warning("Target distribution chart not found. Run compiler to generate.")
    with col10:
        missing_img = "notebooks/charts/missing_values.png"
        if os.path.exists(missing_img):
            st.image(missing_img, caption="Missingness concentrations comparison across datasets", use_container_width=True)
        else:
            st.warning("Missingness concentrations chart not found. Run compiler to generate.")

# ==============================================================================
# PAGE 2: EDA INSIGHTS
# ==============================================================================
elif page == "2. EDA Insights":
    st.markdown("<h1>📊 Portfolio Exploratory Data Analysis & Cohorts</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Inspect financial characteristics, risk metrics, and customer demographics generated from live data.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='premium-hr'>", unsafe_allow_html=True)
    
    tab_dem, tab_fin, tab_bureau, tab_corr = st.tabs([
        "👥 Demographics (Age & Gender)", 
        "💰 Financial Metrics (Income & Credit)", 
        "🏢 Bureau Credit History", 
        "🔗 Feature Correlations"
    ])
    
    with tab_dem:
        col_age, col_gen = st.columns(2)
        with col_age:
            age_chart = "notebooks/charts/age_analysis.png"
            if os.path.exists(age_chart):
                st.image(age_chart, caption="Default Rates declines monotonically as borrower age increases", use_container_width=True)
            else:
                st.warning("Age analysis chart not found.")
        with col_gen:
            gender_chart = "notebooks/charts/gender_analysis.png"
            if os.path.exists(gender_chart):
                st.image(gender_chart, caption="Male applicants display a higher default rate than Female applicants", use_container_width=True)
            else:
                st.warning("Gender analysis chart not found.")
                
    with tab_fin:
        col_inc, col_cred = st.columns(2)
        with col_inc:
            inc_chart = "notebooks/charts/income_distribution.png"
            if os.path.exists(inc_chart):
                st.image(inc_chart, caption="Income Distributions and Log Boxplots by Target Status", use_container_width=True)
            else:
                st.warning("Income distribution chart not found.")
        with col_cred:
            cred_chart = "notebooks/charts/credit_distribution.png"
            if os.path.exists(cred_chart):
                st.image(cred_chart, caption="Requested Credit Sizes Distributions by Target Status", use_container_width=True)
            else:
                st.warning("Credit distribution chart not found.")
                
    with tab_bureau:
        bureau_chart = "notebooks/charts/bureau_analysis.png"
        if os.path.exists(bureau_chart):
            st.image(bureau_chart, caption="Defaulters accumulate more concurrent active bureau loans and close fewer historical accounts", use_container_width=True)
        else:
            st.warning("Bureau analysis chart not found.")
            
    with tab_corr:
        corr_chart = "notebooks/charts/correlation_heatmap.png"
        if os.path.exists(corr_chart):
            st.image(corr_chart, caption="Correlations between Target Default status and Key variables (External Sources are strongest)", use_container_width=True)
        else:
            st.warning("Correlation heatmap not found.")

# ==============================================================================
# PAGE 3: CREDIT RISK PREDICTION
# ==============================================================================
elif page == "3. Credit Underwriting":
    st.markdown("<h1>⚖️ Real-Time Credit Underwriting Decisions</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Expose applicant properties to the trained LightGBM model to evaluate real-time default risk and pricing bounds.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='premium-hr'>", unsafe_allow_html=True)
    
    model_path = "models/model.pkl"
    processed_path = "data/processed_train.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(processed_path):
        st.error("Model and preprocessed data files are required to run underwriters. Train model first.")
    else:
        # Load classifier and a baseline row to fetch non-exposed features
        clf = joblib.load(model_path)
        df_base = pd.read_csv(processed_path)
        
        # UI Input layout
        st.markdown("### 📝 Enter Borrower and Credit Details")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            gender = st.selectbox("CODE_GENDER (Applicant Gender)", ["Female (F)", "Male (M)"])
            age = st.slider("AGE_YEARS (Borrower Age in positive years)", 20.0, 70.0, 35.0, 0.5)
            income = st.number_input("AMT_INCOME_TOTAL (Total Annual Income)", min_value=5000.0, max_value=5000000.0, value=150000.0, step=5000.0)
            
        with col_in2:
            credit = st.number_input("AMT_CREDIT (Requested Credit Loan Size)", min_value=10000.0, max_value=10000000.0, value=450000.0, step=10000.0)
            annuity = st.number_input("AMT_ANNUITY (Annual Installment Repayment)", min_value=1000.0, max_value=500000.0, value=25000.0, step=1000.0)
            ext_1 = st.slider("EXT_SOURCE_1 Rating Score", 0.0, 1.0, 0.5, 0.01)
            
        with col_in3:
            ext_2 = st.slider("EXT_SOURCE_2 Rating Score", 0.0, 1.0, 0.5, 0.01)
            ext_3 = st.slider("EXT_SOURCE_3 Rating Score", 0.0, 1.0, 0.5, 0.01)
            active_loans = st.number_input("active_credit_count (Current Active Bureau Accounts)", min_value=0, max_value=50, value=1, step=1)
            closed_loans = st.number_input("closed_credit_count (Completed Bureau Accounts)", min_value=0, max_value=50, value=2, step=1)
            
        # Trigger assessment
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚖️ EVALUATE LOAN APPLICATION", use_container_width=True):
            with st.spinner("Processing underwriting metrics..."):
                # Fetch baseline median row
                # We drop Target and Unique Key
                X_template = df_base.drop(columns=['TARGET', 'SK_ID_CURR'])
                
                # Fetch median of all columns to establish fallback values for remaining 110 less important variables
                median_row = X_template.median().to_dict()
                
                # Overwrite manual features exposed in the UI
                # CODE_GENDER (Female -> 0, Male -> 1)
                median_row['CODE_GENDER'] = 1 if "Male" in gender else 0
                # DAYS_BIRTH (Age in negative days)
                median_row['DAYS_BIRTH'] = -age * 365.0
                median_row['AMT_INCOME_TOTAL'] = income
                median_row['AMT_CREDIT'] = credit
                median_row['AMT_ANNUITY'] = annuity
                median_row['EXT_SOURCE_1'] = ext_1
                median_row['EXT_SOURCE_2'] = ext_2
                median_row['EXT_SOURCE_3'] = ext_3
                median_row['active_credit_count'] = active_loans
                median_row['closed_credit_count'] = closed_loans
                median_row['total_previous_loans'] = active_loans + closed_loans
                
                # Convert back to DataFrame matching trained features ordering exactly
                X_eval = pd.DataFrame([median_row])[X_template.columns]
                
                # Run prediction
                prob_default = clf.predict_proba(X_eval)[0, 1]
                prob_pct = prob_default * 100
                
                st.markdown("<hr class='premium-hr'>", unsafe_allow_html=True)
                st.markdown("### 📊 Underwriting Decision Outcome")
                
                # Render outcome banner by risk bounds
                if prob_pct < 30.0:
                    st.markdown(f"<div class='risk-banner' style='background-color:#10B981;'>APPROVED — LOW DEFAULT RISK ({prob_pct:.2f}% Probability)</div>", unsafe_allow_html=True)
                    st.success("Applicant satisfies all threshold criteria. Standard interest rate pricing is approved.")
                elif prob_pct < 60.0:
                    st.markdown(f"<div class='risk-banner' style='background-color:#F59E0B;'>CONDITIONAL APPROVAL — MANUAL REVIEW ({prob_pct:.2f}% Probability)</div>", unsafe_allow_html=True)
                    st.warning("Applicant shows moderate risk signals. Conditional approval granted subject to secondary income proof verification.")
                else:
                    st.markdown(f"<div class='risk-banner' style='background-color:#EF4444;'>REJECTED — HIGH RISK DEVIATION ({prob_pct:.2f}% Probability)</div>", unsafe_allow_html=True)
                    st.error("Applicant has breached risk tolerance bounds. Credit application rejected due to excessive default likelihood.")

# ==============================================================================
# PAGE 4: SHAP EXPLAINABILITY
# ==============================================================================
elif page == "4. SHAP Explainability":
    st.markdown("<h1>🔍 Model Explainability & SHAP Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Audit mathematical model feature contributions to understand global and local decision boundaries.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='premium-hr'>", unsafe_allow_html=True)
    
    tab_glob, tab_bees, tab_loc = st.tabs([
        "🏆 Global Feature Importance",
        "🐝 Global Beeswarm Summaries",
        "🎯 Local Applicant Waterfall"
    ])
    
    with tab_glob:
        col_img1, col_txt1 = st.columns([0.6, 0.4])
        with col_img1:
            glob_chart = "documents/shap/shap_importance_plot.png"
            if os.path.exists(glob_chart):
                st.image(glob_chart, caption="Average impact magnitude on model outputs", use_container_width=True)
            else:
                st.warning("SHAP global importance plot not found. Run explainability script.")
        with col_txt1:
            st.markdown("### Global Feature Contribution")
            st.write("This bar chart ranks the top 20 features by their average absolute impact on the model's output in log-odds scale.")
            st.markdown("""
            *   **External Scores Dominance**: The external score variables (`EXT_SOURCE_2`, `EXT_SOURCE_3`, and `EXT_SOURCE_1`) are the leading indicators of defaults.
            *   **Financial Demographics**: Goods price and borrower's gender show strong overall contributions.
            *   **Ingested Bureau Metrics**: Engineered credit bureau accounts (`active_credit_count`, `closed_credit_count`, `total_previous_loans`) appear inside the top 20, confirming prior credit bureau habits are significant risk factors.
            """)
            
    with tab_bees:
        col_img2, col_txt2 = st.columns([0.6, 0.4])
        with col_img2:
            summary_chart = "documents/shap/shap_summary_plot.png"
            if os.path.exists(summary_chart):
                st.image(summary_chart, caption="Summary Beeswarm: impact of feature values", use_container_width=True)
            else:
                st.warning("SHAP summary beeswarm plot not found. Run explainability script.")
        with col_txt2:
            st.markdown("### Beeswarm Impact Breakdown")
            st.write("This Beeswarm summary dot plot maps individual applicant records. Red represents high feature values, and blue represents low values.")
            st.markdown("""
            *   **Negative Impact (Lower Risk)**: High values of `EXT_SOURCE` scores (red dots) shift SHAP values strongly to the left (negative region), indicating a near-zero default risk.
            *   **Positive Impact (Higher Risk)**: Low values of `EXT_SOURCE` scores (blue dots) shift SHAP values strongly to the right (positive region), indicating a heavy default likelihood.
            *   **Gender and Age Splits**: Female gender (value 0, blue) reduces default probability, while younger applicants (red dots on negative days, meaning lower ages) increase risk.
            """)
            
    with tab_loc:
        col_img3, col_txt3 = st.columns([0.6, 0.4])
        with col_img3:
            waterfall_chart = "documents/shap/shap_waterfall_applicant_0.png"
            if os.path.exists(waterfall_chart):
                st.image(waterfall_chart, caption="Additive explanation waterfall for Applicant Index 0", use_container_width=True)
            else:
                st.warning("SHAP waterfall plot not found. Run explainability script.")
        with col_txt3:
            st.markdown("### Individual Applicant Waterfall Audit")
            st.write("This plot shows an individual client's underwritten risk path. Underwriters can track how different features push the final prediction score away from the base expected rate.")
            st.markdown("""
            *   **Base Value**: The starting reference value (unbiased population average log-odds, approx. -2.32).
            *   **Feature Offsets**: Each row shows the specific additive value that the applicant's unique parameters (like their external scores, age, or income) contributed to their final default probability score.
            *   **Regulatory Auditing**: This provides a fully auditable decision record for compliance, allowing underwriters to explain exactly *why* a particular borrower was rejected or conditionally approved.
            """)

# ==============================================================================
# PAGE 5: AUTOMATED POLICY RULES
# ==============================================================================
elif page == "5. Automated Policy Rules":
    st.markdown("<h1>📜 Transparent Credit Policy Business Rules</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Auditable business rules generated via a depth-3 Decision Tree Classifier for direct policy engine deployment.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='premium-hr'>", unsafe_allow_html=True)
    
    col_tree, col_rules = st.columns([0.55, 0.45])
    
    with col_tree:
        st.markdown("### 🌲 Visual Policy Rules Tree Map")
        tree_img = "documents/rules/tree.png"
        if os.path.exists(tree_img):
            st.image(tree_img, caption="Automated Decision Tree structure (max_depth=3)", use_container_width=True)
        else:
            st.warning("Decision Tree visual map not found. Run rule generator script.")
            
    with col_rules:
        st.markdown("### 📜 Deployment Business Parameters")
        rules_path = "documents/rules/business_rules.txt"
        if os.path.exists(rules_path):
            with open(rules_path, 'r', encoding='utf-8') as f:
                rules_content = f.read()
            st.text_area("Extracted Human-Readable Rules", rules_content, height=500)
        else:
            st.warning("Extracted business rules parameters text file not found.")

# ==============================================================================
# PAGE 6: TALK-TO-DATA CHATBOT
# ==============================================================================
elif page == "6. Relational SQL Chatbot":
    st.markdown("<h1>💬 Relational SQL Database Chatbot (Offline Mode)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Query preprocessed credit risk application database tables using a secure, offline SQL translation interface.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='premium-hr'>", unsafe_allow_html=True)
    
    db_path = "data/credit_risk.db"
    
    if not os.path.exists(db_path):
        st.error("SQLite database credit_risk.db not found. Load database first.")
    else:
        # Initialize chatbot
        chatbot = OfflineCreditRiskChatbot(db_path=db_path)
        
        st.markdown("### ❓ Select or Ask a Business Question")
        
        # Example questions
        examples = [
            "How many customers defaulted?",
            "Average income of defaulters?",
            "Which gender defaults more?",
            "What is the average credit amount?",
            "Which age group has highest default rate?"
        ]
        
        # Visual select dropdown
        selected_q = st.selectbox("📌 Select a sample query shortcut...", ["-- Select Question --"] + examples)
        
        # Text input query
        st.markdown("<p style='font-size: 0.9rem; font-weight:600; color:#475569;'>Or type your own question here:</p>", unsafe_allow_html=True)
        custom_q = st.text_input("NL Query Input Box", placeholder="e.g. Total defaults?", label_visibility="collapsed")
        
        # Evaluate query trigger
        query_to_run = None
        if custom_q.strip() != "":
            query_to_run = custom_q
        elif selected_q != "-- Select Question --":
            query_to_run = selected_q
            
        if query_to_run:
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                with st.spinner("Executing relational secure query..."):
                    sql, df_results, explanation = chatbot.ask(query_to_run)
                    
                    st.markdown("### 🔏 Generated Secure SQL")
                    st.code(sql, language="sql")
                    
                    st.markdown("### 📊 Raw Database Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    st.markdown("### 💡 Business Analyst Interpretation")
                    st.info(explanation)
                    
            except Exception as e:
                st.error(f"Execution Error: {e}")
