# Loan Approval Prediction Streamlit Web Application

🚀 **Live Demo:** [https://loan-approval-prediction-lfoh.onrender.com](https://loan-approval-prediction-lfoh.onrender.com)

This repository contains an enhanced, production-ready Machine Learning project that predicts loan eligibility using customer application profiles. It features advanced **domain feature engineering**, an optimized **Random Forest classifier** pipeline, an interactive **What-If Loan Optimizer simulator**, and a bulk **Batch Prediction Explorer** for bulk processing and CSV export.

---

## 📂 Project Structure

```text
loan-approval-prediction/
├── .gitignore                 # Prevents committing local virtual env & cache files
├── dataset/
│   └── loan_data.csv          # Raw customer application dataset
├── app.py                     # Streamlit web application dashboard (Predictor, Explorer, Batch Processor)
├── train_model.py             # ML pipeline (Data download, Feature engineering, Tuning, Serialization)
├── loan_model.pkl             # Serialized Scikit-Learn pipeline (Preprocessors + Tuned Random Forest Classifier)
├── model_metadata.json        # Saved performance scores and feature importances
├── requirements.txt           # Python library dependencies
├── render.yaml                # Render Blueprint configuration for cloud deployment
└── README.md                  # Setup, local execution, and deployment guide
```

---

## 📊 Dataset & Advanced Feature Engineering

The baseline dataset represents historical loan applications for **Dream Housing Finance**. In Phase 2, we engineered the following custom financial risk indicators:

1.  **Total_Income:** Combined monthly income of the primary applicant and the co-applicant.
2.  **Income_to_Loan_Ratio:** The monthly income relative to the requested loan amount.
3.  **EMI (Estimated Monthly Installment):** Calculated at a standard 6% interest rate (0.5% monthly APR) over the selected loan term.
4.  **DTI (Debt-to-Income Ratio):** The percentage of combined monthly income represented by the estimated EMI payment.

---

## 📈 Model Performance & Selection

The pipeline trains and compares four classifiers using an 80-20 train-test split:
1.  **Logistic Regression**
2.  **Decision Tree**
3.  **Random Forest** (Selected & Tuned)
4.  **XGBoost**

With our advanced engineered features, the tuned **Random Forest Classifier** achieved the highest F1-Score:
*   **Accuracy:** 86.99%
*   **Precision:** 84.85%
*   **Recall:** 98.82%
*   **F1-Score:** 91.30%

---

## 💡 Custom Dashboard Features

The Streamlit web application dashboard includes three main explorers:

### 1. 🔮 Loan Eligibility Predictor
Input individual applicant profiles (gender, marital status, income, loan amount, credit standing, etc.) to get an instant prediction.
*   **What-If Loan Optimizer:** If an application is rejected, the simulator runs background scenarios to recommend actionable adjustments (e.g., specific reductions in loan amount, required co-applicant income, or extending terms) to reverse the model's decision and obtain approval.

### 2. 📊 Interactive Data Explorer
View the raw training dataset, explore statistical summary metrics, and dynamically plot attributes (e.g. ApplicantIncome, LoanAmount, Property_Area distributions) to discover trends.

### 3. 📤 Batch Prediction Explorer
Upload a CSV file containing multiple applicant records to process bulk data.
*   **Template Downloader:** Download a sample template CSV to construct bulk records.
*   **Bulk Analytics:** Shows aggregate KPIs (Total Applicants, Approval Count, Rejection Count, Approval Rate).
*   **Export Report:** Download a complete bulk prediction report containing predictions and confidence levels for all applications.

---

## 💻 How to Run Locally

### Prerequisites
Make sure Python (version 3.8 or higher) is installed on your computer.

### 1. Clone the Directory & Install Dependencies
Navigate to the project folder in your command prompt/terminal and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
Run the training script to retrain the pipeline and update `loan_model.pkl` and `model_metadata.json`:
```bash
python train_model.py
```

### 3. Launch the Streamlit Web Application
Run the Streamlit server locally:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser at `http://localhost:8501`.

---

## 🚀 How to Deploy on Render

### Option A: Using the Render Blueprint (`render.yaml`)
1. Push this project folder to your GitHub repository.
2. Log in to [Render.com](https://render.com/).
3. In the Render Dashboard, click **New +** and select **Blueprint**.
4. Connect your GitHub account and select your project repository.
5. Render will automatically detect the `render.yaml` file, provision a Python Web Service, install dependencies, train the model, and run the Streamlit app!

### Option B: Manual Setup on Render Dashboard
1. Create a **New Web Service** on Render and link it to your GitHub repository.
2. Configure these settings:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
3. Click **Deploy Web Service**. Render will assign you a public URL (e.g. `https://your-app-name.onrender.com`) to access your app from anywhere!
