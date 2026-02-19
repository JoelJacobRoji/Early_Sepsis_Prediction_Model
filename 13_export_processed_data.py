import os
import glob
import pandas as pd
from tqdm import tqdm

# 1. SETUP
print("Starting Data Factory...")
data_path = './sepsis_data'
# Get ALL patient files
all_files = glob.glob(os.path.join(data_path, "**", "*.psv"), recursive=True)
print(f"Found {len(all_files)} patient files.")

processed_data = []

# 2. PROCESSING LOOP (The Cleaning Machine)
print("Preprocessing all 40,000 patients (This may take 1-2 minutes)...")

for file in tqdm(all_files):
    try:
        # Load raw data
        df = pd.read_csv(file, sep='|')
        
        # --- STEP 1: FILL MISSING VALUES ---
        # Forward Fill: If HR is missing at hour 5, use hour 4's value.
        # Backward Fill: If hour 1 is missing, use hour 2's value.
        # Fill -1: If a value is completely missing for a patient, set to -1.
        df = df.ffill().bfill().fillna(-1)
        
        # --- STEP 2: CREATE TARGET LABEL ---
        # We want to predict sepsis 6 hours in ADVANCE.
        # So we shift the 'SepsisLabel' column UP by 6 rows.
        df['Target_Label_6h'] = df['SepsisLabel'].shift(-6)
        
        # Remove the rows at the end that don't have a future label anymore
        df.dropna(subset=['Target_Label_6h'], inplace=True)
        
        # Drop the original label (since we are predicting the future one)
        df.drop(columns=['SepsisLabel'], inplace=True)
        
        # Add Patient ID (optional, helps identify rows later)
        patient_id = os.path.basename(file).split('.')[0]
        df['Patient_ID'] = patient_id
        
        # Append to master list
        if not df.empty:
            processed_data.append(df)
            
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

# 3. COMBINE AND SAVE
print("Merging data...")
full_df = pd.concat(processed_data, ignore_index=True)

# Make sure the target is an integer (0 or 1)
full_df['Target_Label_6h'] = full_df['Target_Label_6h'].astype(int)

print(f"Final Dataset Shape: {full_df.shape}")
print("Saving to CSV (this is the big file you wanted)...")

full_df.to_csv('final_processed_sepsis_data.csv', index=False)

print("\n✅ SUCCESS: 'final_processed_sepsis_data.csv' has been created!")