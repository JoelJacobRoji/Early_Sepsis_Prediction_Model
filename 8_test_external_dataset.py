import pandas as pd
import numpy as np
import pickle

# 1. Load the MIMIC Data
print("Loading MIMIC data...")
# Read only first 100,000 rows to ensure it runs fast on your laptop
df = pd.read_csv('mimic_demo_raw.csv', low_memory=False, nrows=100000)

# --- FIX: FORCE UPPERCASE COLUMNS ---
# This fixes the 'KeyError: VALUENUM' by standardizing all names
df.columns = df.columns.str.upper()
print(f"Columns standardized. Found: {list(df.columns)}") 
# ------------------------------------

# 2. Pivot the Data (Long -> Wide Format)
print("Pivoting data to format patients (this takes ~10 seconds)...")
# We pivot so each row is one hour of patient time
pivot_df = df.pivot_table(index=['SUBJECT_ID', 'CHARTTIME'], columns='ITEMID', values='VALUENUM')

# 3. Rename MIMIC IDs to Model Feature Names
# 220045=HR, 220047=O2Sat, 223761=TempF, 220179=SBP, 220210=Resp
column_map = {
    220045: 'HR',
    220047: 'O2Sat',
    223761: 'Temp',
    220179: 'SBP',
    220210: 'Resp'
}
pivot_df.rename(columns=column_map, inplace=True)

# 4. Add Missing Columns (Fill with -1)
# The model expects 40 features. We fill the ones we don't have with -1.
required_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 
                 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
                 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
                 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
                 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
                 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 
                 'HospAdmTime', 'ICULOS']

for col in required_cols:
    if col not in pivot_df.columns:
        pivot_df[col] = -1

# Prepare final matrix
X_external = pivot_df[required_cols].fillna(-1)
print(f"External Test Set Created: {X_external.shape}")

# 5. Load Your Model
print("\nLoading production model...")
# Change this line:
with open('xgboost_sepsis_balanced_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 6. Run Predictions
print("Diagnosing external patients...")
y_prob_external = model.predict_proba(X_external)[:, 1]

# 7. Generate Validation Report
print("\n" + "="*40)
print("EXTERNAL VALIDATION REPORT (MIMIC-III)")
print("="*40)
print(f"Total Patient Hours Analyzed: {len(X_external)}")
print(f"Average Predicted Sepsis Risk: {np.mean(y_prob_external):.4f} ({(np.mean(y_prob_external)*100):.2f}%)")

# Count how many were flagged as High Risk (> 0.45)
high_risk_count = (y_prob_external > 0.45).sum()
print(f"High Risk Alerts Triggered: {high_risk_count}")

if np.mean(y_prob_external) > 0:
    print("\n✅ SUCCESS: Model is successfully generating predictions on new data.")
    print("Note: AUROC cannot be calculated because this demo dataset lacks 'True Sepsis' labels.")
else:
    print("\n❌ WARNING: Model output is zero. Check feature mapping.")