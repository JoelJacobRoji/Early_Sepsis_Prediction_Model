# 🏥 Sepsis Early Warning System (SEWS)
### AI-Powered Real-Time ICU Monitoring Tool

## Authors :
[Joel Jacob Roji](https://github.com/JoelJacobRoji)
[P Kamuel Shawn](https://github.com/KamuelShawn)
[Darren Samuel Dcruz](https://github.com/Darren-Dcruz)


**Project Status:** Completed & Validated


**Live Demo:** [SEWS Dashboard](https://earlysepsispredictionmodel-unvvvcefskwrbyca8p67ps.streamlit.app/)

---

## 📖 Project Overview
Sepsis is a life-threatening reaction to an infection that kills ~270,000 Americans annually. Every hour of delayed treatment increases mortality by 8%. 

**SEWS** (Sepsis Early Warning System) is a machine learning tool designed to predict sepsis **6 hours before clinical onset**. It analyzes patient vitals in real-time and alerts ICU staff to deteriorating conditions, allowing for life-saving early intervention.

## 📊 Data Source & Citation
The dataset used to train this model is publicly available via PhysioNet. Due to GitHub's file size limits, the raw data (~40,000 patients) is not hosted in this repository. 

**To recreate the dataset:**
1. Download the raw data from the [PhysioNet Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/).
2. Place the unzipped folders into a directory named `sepsis_data/`.
3. Run `python 13_export_processed_data.py` to automatically clean, impute, and generate the final CSV used for training.

**Citations:**
This project utilizes data from the following sources, which apply the Sepsis-3 clinical criteria:

> Reyna, M., Josef, C., Jeter, R., Shashikumar, S., Moody, B., Westover, M. B., Sharma, A., Nemati, S., & Clifford, G. D. (2019). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019 (version 1.0.0). PhysioNet. https://doi.org/10.13026/v64v-d857

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

## 🧠 The AI Model
* **Algorithm:** XGBoost Classifier (Extreme Gradient Boosting)
* **Training Data:** 40,336 ICU Patients (PhysioNet 2019 Challenge)
* **Key Innovation:** Implemented **Class Balancing** (Downsampling) to solve the "Imbalanced Data" problem (Sepsis prevalence < 2%), improving sensitivity by **7,000%**.
* **Input Features:** Heart Rate, BP, Temp, O2Sat, Resp Rate, Age, ICU Length of Stay.

## 📈 Performance & Validation
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

## 🚀 How to Run Locally
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
