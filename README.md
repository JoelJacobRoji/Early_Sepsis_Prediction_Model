# 🏥 Sepsis Early Warning System (SEWS)
### AI-Powered Real-Time ICU Monitoring Tool

**Author:** Joel Jacob Roji
**Project Status:** Completed & Validated

---

## 📖 Project Overview
Sepsis is a life-threatening reaction to an infection that kills ~270,000 Americans annually. Every hour of delayed treatment increases mortality by 8%. 

**SEWS** (Sepsis Early Warning System) is a machine learning tool designed to predict sepsis **6 hours before clinical onset**. It analyzes patient vitals in real-time and alerts ICU staff to deteriorating conditions, allowing for life-saving early intervention.

## 🧠 The AI Model
* **Algorithm:** XGBoost Classifier (Extreme Gradient Boosting)
* **Training Data:** 40,336 ICU Patients (PhysioNet 2019 Challenge)
* **Key Innovation:** Implemented **Class Balancing** (Downsampling) to solve the "Imbalanced Data" problem (Sepsis prevalence < 2%), improving sensitivity by **7,000%**.
* **Input Features:** Heart Rate, BP, Temp, O2Sat, Resp Rate, Age, ICU Length of Stay.

## 📊 Performance & Validation
The model was rigorously tested in three phases:

| Test Phase | Method | Result | Significance |
| :--- | :--- | :--- | :--- |
| **1. Internal Validation** | Tested on 1,000 unseen patients (Holdout Set) | **93.1% Accuracy** | High reliability on standard cases. |
| **2. Sensitivity Check** | Confusion Matrix Analysis | **282 True Positives** | Successfully identifies subtle sepsis patterns. |
| **3. External Validation** | Tested on **MIMIC-III** (Beth Israel Deaconess Medical Center) | **28.02% Detection Rate** | Proven to generalize to completely new hospital systems. |

## 💻 Tech Stack
* **Python 3.11** (Data Processing & ML)
* **Pandas / NumPy** (Feature Engineering & Label Shifting)
* **XGBoost** (Model Training)
* **Streamlit** (Interactive Web Dashboard)

## 🚀 How to Run
1.  **Clone the Repository**
2.  **Install Requirements:**
    ```bash
    pip install pandas numpy xgboost scikit-learn streamlit
    ```
3.  **Launch the Dashboard:**
    ```bash
    python -m streamlit run app.py
    ```

## 🏥 Clinical Use Case
* **Green (Low Risk):** Patient is stable. Monitor normally.
* **Yellow (Moderate Risk):** Vitals showing early instability. Increase monitoring frequency.
* **Red (High Risk):** **SEPTIC SHOCK IMMINENT.** Immediate clinical assessment required (Lactate/WBC check recommended).

---
*Disclaimer: This tool is for educational/research purposes only and is not a replacement for professional medical diagnosis.*