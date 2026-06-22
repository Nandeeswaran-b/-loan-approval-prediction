import os
import json
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Dream Finance - Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the UI
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        width: 100%;
        border-radius: 8px;
        padding: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1565c0;
        margin-bottom: 10px;
    }
    .title-text {
        color: #1e3d59;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .subtitle-text {
        color: #17b978;
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if model files exist
def check_model_files():
    return os.path.exists("loan_model.pkl") and os.path.exists("model_metadata.json")

# Helper function to load model and metadata
@st.cache_resource
def load_model_and_metadata():
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

# App Title & Layout
st.markdown("<h1 class='title-text'>🏦 Dream Housing Finance</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Automated Loan Eligibility Prediction System</p>", unsafe_allow_html=True)

# ----------------- MODEL LOADER & TRAINER -----------------
model_loaded = False
model = None
metadata = None

if check_model_files():
    model, metadata = load_model_and_metadata()
    model_loaded = True
else:
    st.warning("⚠️ Machine Learning model is not trained yet. You can train it below.")
    if st.button("🚀 Train Model Now"):
        with st.spinner("Downloading dataset and training classifier (Logistic Regression, Trees, XGBoost)..."):
            try:
                import importlib
                import train_model
                importlib.reload(train_model)
                train_model.train_and_save_model()
                st.success("🎉 Model trained successfully! Reloading application...")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error during training: {e}")

# ----------------- TABBED LAYOUT -----------------
tab_predict, tab_explore = st.tabs(["🔮 Loan Eligibility Predictor", "📊 Interactive Data Explorer"])

# ==================== TAB 1: PREDICTOR ====================
with tab_predict:
    col_form, col_metrics = st.columns([1.8, 1.2])

    with col_form:
        st.subheader("📋 Loan Application Details")
        st.write("Please fill in the applicant's details below to verify eligibility:")
        
        with st.form("loan_application_form"):
            # Personal details
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Marital Status", ["Yes", "No"], help="Is the applicant married?")
            with c2:
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], help="Number of dependent family members.")
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                
            # Employment & Financial details
            c3, c4 = st.columns(2)
            with c3:
                self_employed = st.selectbox("Self Employed", ["Yes", "No"], help="Is the applicant self-employed?")
                applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=100)
            with c4:
                coapplicant_income = st.number_input("Co-Applicant Monthly Income ($)", min_value=0, value=0, step=100, help="Income of spouse or partner.")
                
            # Loan details
            c5, c6 = st.columns(2)
            with c5:
                loan_amount = st.number_input("Loan Amount in Thousands ($)", min_value=1, value=150, step=5, help="e.g. 150 = $150,000")
            with c6:
                term_options = [360, 480, 300, 240, 180, 120, 84, 60, 36, 12]
                loan_term = st.selectbox("Loan Term (Months)", term_options, index=0)
                
            # Credit history and Property location
            c7, c8 = st.columns(2)
            with c7:
                credit_history_label = st.selectbox(
                    "Credit History", 
                    ["Good / Cleared Debts", "Poor / Outstanding Debts"],
                    help="Has the applicant cleared past debts?"
                )
                credit_history = 1.0 if credit_history_label == "Good / Cleared Debts" else 0.0
            with c8:
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
                
            submit_btn = st.form_submit_button("🔍 Predict Eligibility")

        # Handle submission
        if submit_btn:
            if not model_loaded:
                st.error("Please train the model first by clicking 'Train Model Now' on the right.")
            else:
                # Financial Calculations (Risk Indicators)
                total_income = float(applicant_income) + float(coapplicant_income)
                loan_amount_dollars = float(loan_amount) * 1000
                
                # Estimate monthly payment (EMI) at standard 6% interest rate
                r = 0.06 / 12  # monthly interest rate
                n = float(loan_term)
                if total_income > 0:
                    emi = loan_amount_dollars * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
                    dti = (emi / total_income) * 100
                else:
                    emi = 0
                    dti = 0
                
                # Map inputs to DataFrame
                input_df = pd.DataFrame([{
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "ApplicantIncome": float(applicant_income),
                    "CoapplicantIncome": float(coapplicant_income),
                    "LoanAmount": float(loan_amount),
                    "Loan_Amount_Term": float(loan_term),
                    "Credit_History": str(float(credit_history)),
                    "Property_Area": property_area
                }])
                
                # Format features
                input_df['Credit_History'] = input_df['Credit_History'].astype(object)
                
                # Predict
                try:
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    confidence = probabilities[1] if prediction == 1 else probabilities[0]
                    
                    st.markdown("### 📊 Prediction Result")
                    
                    # 1. Prediction Outcome Alerts
                    if prediction == 1:
                        st.success(f"🎉 **LOAN APPROVED**")
                        st.info(f"The model is **{confidence*100:.2f}%** confident that the applicant meets eligibility criteria.")
                    else:
                        st.error(f"❌ **LOAN REJECTED**")
                        st.warning(f"The model is **{confidence*100:.2f}%** confident that the applicant does NOT meet eligibility criteria.")
                        
                    # 2. Risk Metrics & Analysis Output
                    st.markdown("#### 🔍 Financial Health & Risk Analysis")
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.metric(label="Estimated Monthly Payment (EMI)", value=f"${emi:,.2f}")
                    with col_r2:
                        st.metric(label="Debt-to-Income Ratio (DTI)", value=f"{dti:.1f}%")
                        
                    # Threshold warning for DTI (standard banking guidelines flag DTI > 40%)
                    if dti > 40:
                        st.warning(f"⚠️ **High DTI Warning:** The monthly payment represents **{dti:.1f}%** of total income. Most lenders flag values above 40% as high risk.")
                    else:
                        st.success(f"✅ **Acceptable Debt-to-Income Ratio:** The DTI of **{dti:.1f}%** is within standard lending bounds.")
                        
                    # 3. Export Prediction Report
                    report_text = f"""==================================================
DREAM HOUSING FINANCE - LOAN APPLICATION REPORT
==================================================
PREDICTION STATUS: {'APPROVED' if prediction == 1 else 'REJECTED'}
CONFIDENCE LEVEL: {confidence * 100:.2f}%

--- APPLICANT PROFILE ---
Gender: {gender}
Married: {married}
Dependents: {dependents}
Education: {education}
Self Employed: {self_employed}
Applicant Monthly Income: ${applicant_income:,.2f}
Co-Applicant Monthly Income: ${coapplicant_income:,.2f}
Total Combined Income: ${total_income:,.2f}

--- LOAN APPLICATION DETAILS ---
Requested Loan Amount: ${loan_amount_dollars:,.2f}
Loan Term: {loan_term} Months
Credit History Status: {'Good (Cleared past debts)' if credit_history == 1.0 else 'Poor (Outstanding debts)'}
Property Area: {property_area}

--- RISK SUMMARY ---
Estimated Monthly Payment (EMI): ${emi:,.2f} (Estimated at 6% APR)
Debt-to-Income (DTI) Ratio: {dti:.2f}%
Risk Designation: {'HIGH DEBT-TO-INCOME RISK' if dti > 40 else 'LOW/MODERATE DEBT RISK'}
=================================================="""

                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Prediction Report",
                        data=report_text,
                        file_name=f"loan_report_{gender}_{loan_amount}k.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error predicting loan status: {e}")

    with col_metrics:
        st.subheader("📈 Model Insights")
        st.write("Information about the underlying Machine Learning model:")
        
        if model_loaded:
            metrics = metadata["metrics"]
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏆 Selected Model: Tuned {metadata['best_model_name']}</h4>
                <p>Accuracy: <b>{metrics['Accuracy']*100:.1f}%</b></p>
                <p>F1-Score: <b>{metrics['F1_Score']*100:.1f}%</b></p>
                <p>Recall: <b>{metrics['Recall']*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("🔍 **Key Features Driving Approvals**")
            feature_importances = metadata["feature_importances"]
            
            if feature_importances:
                top_features = list(feature_importances.keys())[:7]
                top_scores = list(feature_importances.values())[:7]
                
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.barplot(x=top_scores, y=top_features, ax=ax, palette="viridis")
                ax.set_title("Feature Importance Breakdown", fontsize=10, fontweight="bold")
                ax.set_xlabel("Relative Importance Score", fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Feature importance display not supported for this model.")
        else:
            st.info("Model details will appear here once the model is trained.")
            
        st.markdown("""
        ---
        ### 📁 Technical Details:
        *   **Data Imputers:** Median (Numeric), Mode (Categorical)
        *   **Scaling:** Standard Scaler
        *   **Encoding:** One-Hot Encoding
        *   **Test-Split Ratio:** 80% Train, 20% Test
        """)

# ==================== TAB 2: EXPLORER ====================
with tab_explore:
    st.subheader("📊 Interactive Data Explorer")
    st.write("Explore the underlying training dataset directly within your browser:")
    
    csv_path = os.path.join("dataset", "loan_data.csv")
    if not os.path.exists(csv_path):
        st.warning("⚠️ Training dataset file not found. Please run or train the model to download the dataset.")
    else:
        df_raw = pd.read_csv(csv_path)
        
        # Summary KPI Cards
        total_records = df_raw.shape[0]
        total_cols = df_raw.shape[1]
        
        # Calculate approvals rate
        if "Loan_Status" in df_raw.columns:
            approvals = df_raw[df_raw['Loan_Status'] == 'Y'].shape[0]
            approval_rate = (approvals / total_records) * 100
        else:
            approvals = 0
            approval_rate = 0.0
            
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="Total Application Records", value=f"{total_records}")
        with kpi2:
            st.metric(label="Total Attributes (Columns)", value=f"{total_cols}")
        with kpi3:
            st.metric(label="Historical Approval Rate", value=f"{approval_rate:.1f}%")
            
        # Section 1: Raw Data Sample
        st.markdown("---")
        with st.expander("🔍 View Raw Dataset Table (First 50 Rows)"):
            st.dataframe(df_raw.head(50), use_container_width=True)
            
        # Section 2: Summary Statistics
        with st.expander("🔢 Statistical Summary Table"):
            st.dataframe(df_raw.describe(), use_container_width=True)
            
        # Section 3: Dynamic Distribution Visualization
        st.markdown("---")
        st.write("🎯 **Interactive Distribution Plotter**")
        st.write("Select any column below to dynamically plot its distributions:")
        
        # Allow selection from numeric and key categorical columns
        plottable_cols = [
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
            "Loan_Amount_Term", "Credit_History", "Property_Area", 
            "Gender", "Married", "Education", "Loan_Status"
        ]
        selected_col = st.selectbox("Select column to plot:", plottable_cols, index=2)
        
        fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
        if df_raw[selected_col].dtype in ['int64', 'float64'] and selected_col != 'Credit_History':
            # Histogram for continuous numeric data
            sns.histplot(df_raw[selected_col].dropna(), kde=True, ax=ax_dist, color="#1565c0", bins=30)
            ax_dist.set_title(f"Continuous Numerical Distribution of {selected_col}", fontsize=12, fontweight='bold')
            ax_dist.set_xlabel(selected_col)
            ax_dist.set_ylabel("Frequency")
        else:
            # Countplot for categorical / discrete data
            # Format Credit_History for cleaner labels
            plot_series = df_raw[selected_col].astype(str).replace({'1.0': 'Good/Cleared', '0.0': 'Poor/Outstanding', 'nan': 'Missing'})
            sns.countplot(x=plot_series, ax=ax_dist, palette="Set2", order=plot_series.value_counts().index)
            ax_dist.set_title(f"Categorical Frequency Counts of {selected_col}", fontsize=12, fontweight='bold')
            ax_dist.set_xlabel(selected_col)
            ax_dist.set_ylabel("Count")
            
        plt.tight_layout()
        st.pyplot(fig_dist)
