import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")
st.title(" Telecom Customer Risk Profiling")
st.markdown("Adjust the core business drivers below to predict subscriber retention likelihood.")

# 2. Load the pre-trained model
@st.cache_resource
def load_model():
    # Ensure this file is uploaded to your GitHub repository alongside app.py
    model = joblib.load('telco_churn_model.pkl')
    return model

model = load_model()

# 3. User Input Interface
st.header("Subscriber Diagnostics")
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=65.0)
    total_charges = st.number_input("Total Lifetime Charges ($)", min_value=0.0, max_value=9000.0, value=780.0)

with col2:
    # Mapped to match Scikit-Learn's alphabetical LabelEncoder output
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    contract_dict = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tech_dict = {"No": 0, "Yes": 2, "No internet service": 1} 

# 4. Prediction Logic & Background Padding
if st.button("Predict Risk Score", type="primary"):
    
    # The Baseline Customer Array (Pads the features not shown in the UI)
    input_dict = {
        'gender': 1,              # Default: Male
        'SeniorCitizen': 0,       # Default: No
        'Partner': 0,             # Default: No
        'Dependents': 0,          # Default: No
        'tenure': tenure,         # UI Input
        'PhoneService': 1,        # Default: Yes
        'MultipleLines': 0,       # Default: No
        'InternetService': 1,     # Default: Fiber optic (High Churn Risk Group)
        'OnlineSecurity': 0,      # Default: No
        'OnlineBackup': 0,        # Default: No
        'DeviceProtection': 0,    # Default: No
        'TechSupport': tech_dict[tech_support], # UI Input
        'StreamingTV': 0,         # Default: No
        'StreamingMovies': 0,     # Default: No
        'Contract': contract_dict[contract_type], # UI Input
        'PaperlessBilling': 1,    # Default: Yes
        'PaymentMethod': 2,       # Default: Electronic check
        'MonthlyCharges': monthly_charges, # UI Input
        'TotalCharges': total_charges      # UI Input
    }
    
    # Convert to DataFrame to ensure exact column alignment for LightGBM
    input_df = pd.DataFrame([input_dict])
    
    # Generate Probability Score
    churn_prob = model.predict_proba(input_df)[0][1]
    
    st.divider()
    
    # Display Business Logic Output
    if churn_prob > 0.50:
        st.error(f"⚠️ High Risk of Attrition (Probability: {churn_prob:.1%})")
        st.write("**Strategy:** Flagged for proactive retention campaign. Consider offering a discounted annual contract upgrade or complimentary technical support.")
    else:
        st.success(f" Stable Subscriber (Probability: {churn_prob:.1%})")
        st.write("**Strategy:** Standard lifecycle management.")
