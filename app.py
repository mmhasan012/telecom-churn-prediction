import streamlit as st
import pandas as pd
import joblib

# 1. Advanced Page Configuration (Wide Layout)
st.set_page_config(page_title="Enterprise Churn Analytics", layout="wide", initial_sidebar_state="expanded")

# 2. Header Section
st.title("📡 Enterprise Telecom Analytics Dashboard")
st.markdown("### Subscriber Attrition & Risk Diagnostics")
st.markdown("Analyze comprehensive user profiles to generate real-time retention intelligence.")
st.divider()

# 3. Load Model
@st.cache_resource
def load_model():
    return joblib.load('telco_churn_model.pkl')

model = load_model()

# 4. User Interface: Tabbed Navigation for Detailed Inputs
tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🛠️ Services & Usage", "💳 Billing & Contract"])

with tab1:
    st.subheader("Customer Demographics")
    col1, col2, col3, col4 = st.columns(4)
    gender = col1.selectbox("Gender", ["Male", "Female"])
    senior = col2.selectbox("Senior Citizen", ["No", "Yes"])
    partner = col3.selectbox("Partner", ["No", "Yes"])
    dependents = col4.selectbox("Dependents", ["No", "Yes"])

with tab2:
    st.subheader("Subscribed Services")
    col1, col2, col3 = st.columns(3)
    phone = col1.selectbox("Phone Service", ["No", "Yes"])
    mult_lines = col2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = col3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    col4, col5, col6 = st.columns(3)
    security = col4.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = col5.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = col6.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    
    col7, col8 = st.columns(2)
    tech_support = col7.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming = col8.selectbox("Streaming Services (TV/Movies)", ["No", "Yes", "No internet service"])

with tab3:
    st.subheader("Financial Overview")
    col1, col2, col3 = st.columns(3)
    contract = col1.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = col2.selectbox("Paperless Billing", ["No", "Yes"])
    payment = col3.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    tenure = col4.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = col5.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=65.0)
    total_charges = col6.number_input("Total Lifetime Charges ($)", min_value=0.0, max_value=9000.0, value=780.0)

st.divider()

# 5. Data Mapping Dictionaries (Matching LabelEncoder output)
# Simplified mapping logic for cleaner code
yn_map = {"No": 0, "Yes": 1}
yni_map = {"No": 0, "Yes": 2, "No internet service": 1}

# 6. Prediction Execution
if st.button("Generate Risk Profile", type="primary", use_container_width=True):
    
    # Constructing the precise array for LightGBM
    input_dict = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': yn_map[partner],
        'Dependents': yn_map[dependents],
        'tenure': tenure,
        'PhoneService': yn_map[phone],
        'MultipleLines': 0 if mult_lines == "No" else (2 if mult_lines == "Yes" else 1),
        'InternetService': 0 if internet == "DSL" else (1 if internet == "Fiber optic" else 2),
        'OnlineSecurity': yni_map[security],
        'OnlineBackup': yni_map[backup],
        'DeviceProtection': yni_map[protection],
        'TechSupport': yni_map[tech_support],
        'StreamingTV': yni_map[streaming],
        'StreamingMovies': yni_map[streaming], 
        'Contract': 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
        'PaperlessBilling': yn_map[paperless],
        'PaymentMethod': 2 if payment == "Electronic check" else (3 if payment == "Mailed check" else (0 if payment == "Bank transfer (automatic)" else 1)),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([input_dict])
    churn_prob = model.predict_proba(input_df)[0][1]
    
    # 7. Dynamic Output Display with UI Metrics
    st.subheader("Diagnostic Results")
    
    col_metric, col_visual = st.columns([1, 2])
    
    with col_metric:
        st.metric(label="Calculated Attrition Risk", value=f"{churn_prob:.1%}", delta="High Risk" if churn_prob > 0.5 else "Stable", delta_color="inverse")
    
    with col_visual:
        st.markdown("**Risk Gauge:**")
        st.progress(float(churn_prob))
        
        if churn_prob > 0.50:
            st.error("⚠️ Action Required: This profile matches historical markers for immediate service cancellation.")
        else:
            st.success(" Secure: This profile demonstrates strong retention metrics.")
