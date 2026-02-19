import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys

# --- CONFIGURATION ---
PROCESSED_CSV = 'final_processed_sepsis_data.csv'

# 1. LOAD THE DATA
print(f"Loading '{PROCESSED_CSV}'...")
try:
    df = pd.read_csv(PROCESSED_CSV)
except FileNotFoundError:
    print(f"❌ Error: '{PROCESSED_CSV}' not found!")
    print("   Please run 'python 13_export_processed_data.py' first.")
    sys.exit()

print(f"Dataset Loaded. Total Rows: {len(df)}")

# 2. SPLIT BY PATIENT ID
print("Splitting data by Patient ID...")
unique_patients = df['Patient_ID'].unique()
train_ids, test_ids = train_test_split(unique_patients, test_size=0.2, random_state=42)

train_df = df[df['Patient_ID'].isin(train_ids)]
test_df = df[df['Patient_ID'].isin(test_ids)]

print(f"Training on {len(train_ids)} patients.")
print(f"Testing on {len(test_ids)} patients.")

# 3. BALANCE THE TRAINING SET
print("Balancing Training Data...")
sepsis_train = train_df[train_df['Target_Label_6h'] == 1]
healthy_train = train_df[train_df['Target_Label_6h'] == 0]

# Downsample Healthy to 3x Sepsis
healthy_sample = healthy_train.sample(n=len(sepsis_train) * 3, random_state=42)
balanced_train = pd.concat([sepsis_train, healthy_sample]).sample(frac=1, random_state=42)

print(f"Balanced Training Rows: {len(balanced_train)}")

# 4. PREPARE INPUTS
drop_cols = ['Target_Label_6h', 'Patient_ID']
X_train = balanced_train.drop(columns=drop_cols)
y_train = balanced_train['Target_Label_6h']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['Target_Label_6h']

# 5. TRAIN MODEL
print("\nTraining XGBoost Model...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 6. EVALUATE & SAVE
print("\n" + "="*40)
print("FINAL RESULTS")
print("="*40)

y_pred = model.predict(X_test)

# Calculate Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"FINAL ACCURACY: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix Numbers:")
print(cm)

# Save Model
with open('xgboost_sepsis_balanced_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✅ Model saved!")

# 7. GENERATE THE PICTURE (With Accuracy in Title)
print("Generating Confusion Matrix Plot...")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Predicted Healthy', 'Predicted Sepsis'],
            yticklabels=['Actual Healthy', 'Actual Sepsis'])
plt.xlabel('Model Prediction')
plt.ylabel('Reality')
# This line puts the accuracy right on the chart
plt.title(f'Final Model Performance\nAccuracy: {acc*100:.2f}%')
plt.tight_layout()
plt.show()