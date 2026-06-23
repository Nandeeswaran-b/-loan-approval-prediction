# Loan Approval Prediction Streamlit Web Application

🚀 **Live Demo:** [https://loan-approval-prediction.onrender.com](https://loan-approval-prediction.onrender.com)

This repository contains a complete, production-ready Machine Learning project that predicts loan eligibility using a customer application profile. It includes a comprehensive model training pipeline and a web application built using **Streamlit**, designed for easy local execution and automated cloud deployment on **Render.com**.

---

## 📂 Project Structure

```text
loan-approval-prediction/
├── dataset/
│   └── loan_data.csv          # Raw customer application dataset
├── app.py                     # Streamlit web application dashboard
├── train_model.py             # ML pipeline (Data download, Preprocessing, Comparison, Tuning, Saving)
├── loan_model.pkl             # Serialized Scikit-Learn pipeline (Preprocessors + Tuned Classifier)
├── model_metadata.json        # Saved performance scores and feature importances
├── requirements.txt           # Python library dependencies
├── render.yaml                # Render Blueprint configuration for cloud deployment
└── README.md                  # Setup, local execution, and deployment guide
```

---

## 📊 Dataset Description

The dataset represents loan applications for **Dream Housing Finance**. It includes demographic, financial, and credit history variables for each applicant:

1.  **Gender:** Male / Female
2.  **Married:** Yes / No
3.  **Dependents:** 0, 1, 2, 3+
4.  **Education:** Graduate / Not Graduate
5.  **Self_Employed:** Yes / No
6.  **ApplicantIncome:** Primary applicant monthly income ($)
7.  **CoapplicantIncome:** Co-applicant monthly income ($)
8.  **LoanAmount:** Requested loan amount in thousands ($)
9.  **Loan_Amount_Term:** Term of the loan in months (e.g., 360 months = 30 years)
10. **Credit_History:** Credit history meets guidelines (1.0 = Good/Cleared, 0.0 = Poor/Outstanding)
11. **Property_Area:** Urban / Semiurban / Rural
12. **Loan_Status (Target Variable):** Y (Approved) / N (Rejected)

---

## 📈 Model Comparison & Pipeline

The pipeline compares four different classifiers using an 80-20 train-test split:
1.  **Logistic Regression**
2.  **Decision Tree**
3.  **Random Forest**
4.  **XGBoost**

### Data Preprocessing
-   **Missing Values:** Numeric columns are imputed with the **median**; categorical columns are imputed with the **mode** (most frequent value).
-   **Feature Scaling:** Numeric inputs are scaled using `StandardScaler` to ensure equal feature weight.
-   **Categorical Encoding:** Text categories are encoded into binary fields using `OneHotEncoder`.
-   All preprocessing steps and the model are combined into a single, cohesive Scikit-Learn **Pipeline** object, ensuring zero data leakage and making serialization seamless.

---

## 💻 How to Run Locally

### Prerequisites
Make sure Python (version 3.8 or higher) is installed on your computer.

### 1. Clone the Directory & Install Dependencies
Navigate to the project folder in your command prompt/terminal and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
Run the training script to fetch the dataset, train/compare models, and create `loan_model.pkl` and `model_metadata.json`:
```bash
python train_model.py
```
*Note: If you skip this, you can also train the model directly from the Streamlit web interface using the "Train Model Now" button.*

### 3. Launch the Streamlit Web Application
Run the Streamlit server locally:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser at `http://localhost:8501`.

---

## 🚀 How to Deploy on Render

Render.com is a cloud hosting provider that makes deploying Python web services simple. We have included a `render.yaml` configuration to allow automated, one-click deployments.

### Option A: Using the Render Blueprint (`render.yaml`)
1. Push this project folder to your GitHub repository.
2. Log in to [Render.com](https://render.com/).
3. In the Render Dashboard, click **New +** and select **Blueprint**.
4. Connect your GitHub account and select your project repository.
5. Render will automatically detect the `render.yaml` file, provision a Python Web Service, install dependencies, train the model, and run the Streamlit app!

### Option B: Manual Setup on Render Dashboard
If you prefer configuring the web service manually:
1. Create a **New Web Service** on Render and link it to your GitHub repository.
2. Configure these settings:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
3. Click **Deploy Web Service**. Render will assign you a public URL (e.g. `https://your-app-name.onrender.com`) to access your app from anywhere!
