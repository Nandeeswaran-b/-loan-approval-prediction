import os
import json
import pickle
import urllib.request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def download_dataset():
    """
    Downloads the Analytics Vidhya Loan dataset into the dataset/ directory.
    """
    dataset_url = "https://raw.githubusercontent.com/sahutkarsh/loan-prediction-analytics-vidhya/master/train.csv"
    data_dir = "dataset"
    file_path = os.path.join(data_dir, "loan_data.csv")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created folder: {data_dir}")
        
    print(f"Downloading dataset from: {dataset_url}")
    try:
        urllib.request.urlretrieve(dataset_url, file_path)
        print(f"Dataset successfully saved to: {file_path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        
    return file_path

def add_engineered_features(df):
    df = df.copy()
    # 1. Total Income
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome'].fillna(0.0)
    
    # 2. Income to Loan Ratio
    df['Income_to_Loan_Ratio'] = df['Total_Income'] / (df['LoanAmount'] + 1e-5)
    
    # 3. Estimated Monthly Installment (EMI)
    # 6% APR -> 0.5% monthly rate (0.005)
    r = 0.005
    term = df['Loan_Amount_Term'].apply(lambda x: x if x > 0 else np.nan)
    pow_r = (1 + r) ** term
    df['EMI'] = (df['LoanAmount'] * 1000) * r * pow_r / (pow_r - 1)
    
    # 4. Debt to Income Ratio (DTI)
    df['DTI'] = (df['EMI'] / (df['Total_Income'] + 1e-5)) * 100
    
    return df

def train_and_save_model():
    # 1. Download and load data
    file_path = download_dataset()
    if not os.path.exists(file_path):
        print("Error: Could not retrieve dataset. Aborting training.")
        return
        
    df = pd.read_csv(file_path)
    
    # Drop Loan_ID
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
        
    # Treat Credit_History as object type (categorical indicator)
    df['Credit_History'] = df['Credit_History'].astype(object)
    
    # Apply feature engineering
    df = add_engineered_features(df)
    
    # 2. Separate features (X) and target (y)
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"].map({"Y": 1, "N": 0})
    
    # Compute modes/medians for default fallback values
    defaults = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(X[col].median())
        else:
            defaults[col] = str(X[col].mode().iloc[0])
    defaults["CoapplicantIncome"] = 0.0 # Standard default
    
    # 3. Train-Test Split (80-20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Pipelines
    numeric_features = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Total_Income", "Income_to_Loan_Ratio", "EMI", "DTI"
    ]
    categorical_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area"]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit the preprocessor on training data to transform features
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    
    # 5. Model Comparison
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    comparison_results = {}
    
    print("\n--- MODEL COMPARISON ---")
    for name, model_instance in models.items():
        clf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_instance)
        ])
        clf_pipeline.fit(X_train, y_train)
        
        y_pred = clf_pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        comparison_results[name] = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1_Score": float(f1)
        }
        print(f"{name}: Accuracy={acc:.4f}, F1-Score={f1:.4f}")
        
    # Select best model (by F1-score)
    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]["F1_Score"])
    print(f"\nBest Model Selected: {best_model_name}")
    
    # 6. Hyperparameter Tuning
    print(f"Tuning hyperparameters for {best_model_name}...")
    if best_model_name == "Logistic Regression":
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
        raw_clf = LogisticRegression(random_state=42, max_iter=1000)
    elif best_model_name == "Decision Tree":
        param_grid = {
            'classifier__max_depth': [3, 5, 7, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
        raw_clf = DecisionTreeClassifier(random_state=42)
    elif best_model_name == "Random Forest":
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 8, None],
            'classifier__min_samples_split': [2, 5]
        }
        raw_clf = RandomForestClassifier(random_state=42)
    else: # XGBoost
        param_grid = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [3, 4, 6],
            'classifier__learning_rate': [0.01, 0.05, 0.1]
        }
        raw_clf = XGBClassifier(random_state=42, eval_metric='logloss')
        
    tuning_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', raw_clf)
    ])
    
    grid_search = GridSearchCV(
        tuning_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_tuned_pipeline = grid_search.best_estimator_
    print(f"Best Tuned Parameters: {grid_search.best_params_}")
    
    # Evaluate Tuned Model
    y_pred_tuned = best_tuned_pipeline.predict(X_test)
    acc_t = accuracy_score(y_test, y_pred_tuned)
    prec_t = precision_score(y_test, y_pred_tuned)
    rec_t = recall_score(y_test, y_pred_tuned)
    f1_t = f1_score(y_test, y_pred_tuned)
    
    print("\n--- TUNED MODEL EVALUATION ---")
    print(f"Accuracy:  {acc_t:.4f}")
    print(f"Precision: {prec_t:.4f}")
    print(f"Recall:    {rec_t:.4f}")
    print(f"F1-Score:  {f1_t:.4f}")
    
    # 7. Extract Feature Importances
    feature_importance_dict = {}
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback names
        feature_names = [f"feature_{i}" for i in range(X_train_preprocessed.shape[1])]
        
    clf = best_tuned_pipeline.named_steps['classifier']
    
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        feature_importance_dict = dict(zip(feature_names, [float(val) for val in importances]))
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
        importances = importances / np.sum(importances) # Normalize
        feature_importance_dict = dict(zip(feature_names, [float(val) for val in importances]))
        
    # Clean up feature names for visualization display
    # (Scikit-Learn preprocessor prefixes feature names with cat__ or num__)
    cleaned_importances = {}
    for name, val in feature_importance_dict.items():
        clean_name = name.replace("cat__", "").replace("num__", "")
        cleaned_importances[clean_name] = val
        
    # Sort feature importances
    sorted_importances = dict(sorted(cleaned_importances.items(), key=lambda item: item[1], reverse=True))
    
    # 8. Save Pipeline and Metadata
    with open("loan_model.pkl", "wb") as f:
        pickle.dump(best_tuned_pipeline, f)
    print("Saved tuned model pipeline to: loan_model.pkl")
    
    metadata = {
        "best_model_name": best_model_name,
        "metrics": {
            "Accuracy": float(acc_t),
            "Precision": float(prec_t),
            "Recall": float(rec_t),
            "F1_Score": float(f1_t)
        },
        "feature_importances": sorted_importances,
        "comparison_results": comparison_results,
        "defaults": defaults
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Saved training metadata to: model_metadata.json")

if __name__ == "__main__":
    train_and_save_model()
