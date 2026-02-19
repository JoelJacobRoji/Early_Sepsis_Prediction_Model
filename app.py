import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="SEWS - Sepsis Early Warning System",
    page_icon="🏥",
    layout="wide", # Switched to wide for the dashboard look
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
    /* 1. Deep Blue Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
    }

    /* 2. Darker Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 3. Global Text Color (White for labels, headers, and metrics) */
    h1, h2, h3, p, label, .stMarkdown, div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }

    /* 4. Forcing Input Boxes to be White */
    input[type="number"], input[type="text"] {
        background-color: #ffffff !important;
        color: #000000 !important; 
        border-radius: 5px !important;
    }
    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-radius: 5px !important;
    }
    button[kind="stepUp"], button[kind="stepDown"], div[data-baseweb="button"] {
        background-color: #e2e8f0 !important; 
        color: #000000 !important;
    }
    div[data-baseweb="button"] svg {
        fill: #000000 !important;
    }

    /* 5. Glassmorphism effect for the input groups */
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    /* 6. High-Contrast Analyze Button */
    .stButton>button {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 15px !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }

    /* 7. Expander Styling */
    .streamlit-expanderHeader {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ==========================================
# 2. MODEL LOADING & EXPECTED COLUMNS
# ==========================================
@st.cache_resource
def load_model():
    try:
        with open('xgboost_sepsis_balanced_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run '10_retrain_balanced.py' to generate 'xgboost_sepsis_balanced_model.pkl'.")
        return None

model = load_model()

# Define the exact 40 columns the model was trained on
EXPECTED_COLUMNS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
    'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 
    'HospAdmTime', 'ICULOS'
]

# ==========================================
# 3. SIDEBAR - PATIENT DEMOGRAPHICS
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.header("Patient Demographics")
    st.write("Enter basic patient details:")
    
    age = st.slider("Age", 18, 100, 65)
    gender = st.radio("Gender", ["Male", "Female"])
    st.divider()
    hosp_time = st.number_input("Hours since Admission", 0, 500, 24)
    icu_stay = st.number_input("ICU Length of Stay (Hours)", 0, 500, 12)

# ==========================================
# 4. MAIN DASHBOARD - VITALS & LABS
# ==========================================
st.title("🏥 Sepsis Early Warning System")
st.markdown("### AI-Powered Real-Time ICU Monitor")
st.info("⚠️ This tool predicts sepsis onset **6 hours before** clinical recognition.")

st.subheader("1. Vital Signs (Real-Time)")
col1, col2 = st.columns(2)

with col1:
    hr = st.number_input("Heart Rate (BPM)", 40, 200, 85)
    temp = st.number_input("Temperature (°C)", 34.0, 42.0, 37.0, step=0.1)
    sbp = st.number_input("Systolic BP (mm Hg)", 50, 250, 120)
    resp = st.number_input("Respiration Rate (breaths/min)", 8, 60, 18)

with col2:
    o2sat = st.number_input("O2 Saturation (%)", 50, 100, 98)
    dbp = st.number_input("Diastolic BP (mm Hg)", 30, 150, 80)
    map_val = st.number_input("Mean Arterial Pressure (MAP)", 40, 150, 93)

st.subheader("2. Laboratory Values (Optional)")
with st.expander("➕ Add Blood Test Results (Simulate Septic Shock)", expanded=False):
    st.markdown("Leaving these empty (0) assumes no test was ordered.")
    c1, c2, c3 = st.columns(3)
    with c1:
        lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 0.0) # Normal < 2
    with c2:
        wbc = st.number_input("WBC Count (K/uL)", 0.0, 50.0, 0.0) # Normal 4-11
    with c3:
        creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 10.0, 0.0) # Normal 0.6-1.2

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
st.divider()

if st.button("Analyze Patient Risk", type="primary", use_container_width=True):
    if model:
        # 1. Initialize empty patient row with -1 (Missing Value)
        input_data = pd.DataFrame(columns=EXPECTED_COLUMNS)
        input_data.loc[0] = -1 
        
        # 2. Map Inputs to Features
        input_data['HR'] = hr
        input_data['Temp'] = temp
        input_data['SBP'] = sbp
        input_data['MAP'] = map_val
        input_data['DBP'] = dbp
        input_data['Resp'] = resp
        input_data['O2Sat'] = o2sat
        
        input_data['Age'] = age
        input_data['Gender'] = 1 if gender == "Female" else 0
        input_data['HospAdmTime'] = -hosp_time # Often negative in dataset
        input_data['ICULOS'] = icu_stay
        
        # 3. Map Labs (Only if entered)
        if lactate > 0: input_data['Lactate'] = lactate
        if wbc > 0: input_data['WBC'] = wbc
        if creatinine > 0: input_data['Creatinine'] = creatinine

        # 4. Get Prediction
        probability = model.predict_proba(input_data)[0][1]
        risk_score = probability * 100
        
        # 5. Display Results
        st.write("### Risk Assessment Result:")
        
        if risk_score > 40:
            st.error(f"⚠️ HIGH RISK DETECTED")
            st.metric(label="Sepsis Probability", value=f"{risk_score:.1f}%", delta="CRITICAL")
            st.progress(int(risk_score))
            st.markdown("""
            **Clinical Recommendation:**
            * 🚨 **Immediate Sepsis Bundle Activation Required**
            * 🩸 Order Lactate & Blood Cultures
            * 💊 Start Broad-Spectrum Antibiotics within 1 hour
            * 💧 Initiate IV Fluid Resuscitation (30mL/kg)
            """)
            
        elif risk_score > 15:
            st.warning(f"⚖️ MODERATE RISK")
            st.metric(label="Sepsis Probability", value=f"{risk_score:.1f}%", delta="Elevated")
            st.progress(int(risk_score))
            st.markdown("""
            **Clinical Recommendation:**
            * ⚠️ **Increase Monitoring Frequency**
            * 🩸 Check WBC & inflammatory markers
            * 🩺 Re-evaluate SOFA Score in 2 hours
            """)
            
        else:
            st.success(f"✅ LOW RISK")
            st.metric(label="Sepsis Probability", value=f"{risk_score:.1f}%", delta="-Stable")
            st.progress(int(risk_score))
            st.markdown("**Clinical Recommendation:** Patient is stable. Continue standard ICU monitoring.")