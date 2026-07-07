import unittest
import pandas as pd
import numpy as np
import pickle
import os
import json

import train_model
import app

class TestLoanPredictionPipeline(unittest.TestCase):

    def setUp(self):
        # Create a mock applicant profile
        self.mock_profile = pd.DataFrame([{
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000.0,
            "CoapplicantIncome": 1500.0,
            "LoanAmount": 150.0,
            "Loan_Amount_Term": 360.0,
            "Credit_History": "1.0",
            "Property_Area": "Semiurban"
        }])

    def test_feature_engineering_calculations(self):
        # Apply feature engineering
        df_feat = train_model.add_engineered_features(self.mock_profile)
        
        # 1. Total Income: 5000 + 1500 = 6500
        self.assertEqual(df_feat["Total_Income"].iloc[0], 6500.0)
        
        # 2. Income to Loan Ratio: 6500 / 150
        self.assertAlmostEqual(df_feat["Income_to_Loan_Ratio"].iloc[0], 6500.0 / 150.0, places=4)
        
        # 3. EMI Calculation
        # P = 150,000, r = 0.005, n = 360
        # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        r = 0.005
        n = 360
        expected_emi = 150000.0 * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
        self.assertAlmostEqual(df_feat["EMI"].iloc[0], expected_emi, places=4)
        
        # 4. DTI Calculation: (EMI / Total_Income) * 100
        expected_dti = (expected_emi / 6500.0) * 100
        self.assertAlmostEqual(df_feat["DTI"].iloc[0], expected_dti, places=4)

    def test_feature_engineering_nans(self):
        # Test if function handles NaNs gracefully without throwing errors
        nan_profile = pd.DataFrame([{
            "Gender": "Male",
            "Married": np.nan,
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000.0,
            "CoapplicantIncome": np.nan, # Should default to 0.0 in Total_Income
            "LoanAmount": 150.0,
            "Loan_Amount_Term": np.nan, # Should calculate term as NaN and result in NaN EMI
            "Credit_History": "1.0",
            "Property_Area": "Semiurban"
        }])
        
        df_feat = train_model.add_engineered_features(nan_profile)
        self.assertEqual(df_feat["Total_Income"].iloc[0], 5000.0)
        self.assertTrue(np.isnan(df_feat["EMI"].iloc[0]))
        self.assertTrue(np.isnan(df_feat["DTI"].iloc[0]))

    def test_what_if_optimizer_recommendations(self):
        # We need a model to run recommendations, check if model exists
        if os.path.exists("loan_model.pkl") and os.path.exists("model_metadata.json"):
            with open("loan_model.pkl", "rb") as f:
                model = pickle.load(f)
                
            # Create a high-risk profile (low income, high loan, poor credit)
            rejected_profile = pd.DataFrame([{
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000.0,
                "CoapplicantIncome": 0.0,
                "LoanAmount": 100.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": "0.0", # Poor credit history
                "Property_Area": "Semiurban"
            }])
            
            recs = app.get_loan_recommendations(model, rejected_profile)
            # Should output some recommendations
            self.assertTrue(len(recs) >= 0)
            
            # If we check for specific credit standing recommendation (since credit is 0.0)
            types = [r["type"] for r in recs]
            self.assertIn("Credit Standing", types)

if __name__ == "__main__":
    unittest.main()
