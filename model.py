import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import time

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data Preprocessing Complete.")
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
    
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    start_time = time.time() 
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)
    end_time = time.time() 
    print(f"Time taken for RandomizedSearchCV: {end_time - start_time:.2f} seconds")

    best_rf = random_search.best_estimator_

    y_pred = best_rf.predict(X_test)

    print("Best Parameters:", random_search.best_params_)

    # RMSE 
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")

    # MAE 
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae}")

    # R² 
    r2 = r2_score(y_test, y_pred)
    print(f"R²: {r2}")

    evs = explained_variance_score(y_test, y_pred)
    print(f"Explained Variance Score: {evs}")

    y_bins = pd.cut(y_test, bins=3, labels=["Low", "Medium", "High"])
    y_pred_bins = pd.cut(y_pred, bins=3, labels=["Low", "Medium", "High"])

    # Accuracy, Precision, Recall, F1-score
    accuracy = accuracy_score(y_bins, y_pred_bins)
    precision = precision_score(y_bins, y_pred_bins, average='weighted', zero_division=0)
    recall = recall_score(y_bins, y_pred_bins, average='weighted', zero_division=0)
    f1 = f1_score(y_bins, y_pred_bins, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

else:
    print("Target column not found in the dataset.")
