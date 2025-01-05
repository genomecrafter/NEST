import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data_cleaned = pd.read_csv("processed_data.csv")

target_column = "Study Recruitment Rate"  
if target_column in data_cleaned.columns:
    X = data_cleaned.drop(columns=[target_column])
    y = data_cleaned[target_column]

    date_columns = X.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            X[col] = pd.to_datetime(X[col], errors='coerce')
            X[col] = (X[col] - X[col].min()).dt.days  
        except Exception as e:
            print(f"Skipping column {col} due to error: {e}")

    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X[categorical_cols])
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, pd.DataFrame(X_encoded)], axis=1)

    if X.select_dtypes(include=['object']).shape[1] > 0:
        print("Warning: Still contains object columns!")
        print(X.select_dtypes(include=['object']).columns)

    X = X.astype('float64', errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42)
rf_model.fit(X_train, y_train)

explainer = shap.Explainer(rf_model, X_train)

shap_values = explainer(X_train, check_additivity=False)

shap.summary_plot(shap_values, X_train, plot_type="bar")

shap.summary_plot(shap_values, X_train)

shap.dependence_plot(0, shap_values.values, X_train) 

