import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_excel("usecase_4_.xlsx")  

# Step 1: Handle missing values
# Drop columns with too many missing values
threshold = 0.4  # Allow up to 40% missing values
data_cleaned = data.loc[:, data.isnull().mean() < threshold]

# Impute missing values for categorical columns
imputer = SimpleImputer(strategy='most_frequent')  
data_cleaned.loc[:, 'Secondary Outcome Measures'] = imputer.fit_transform(data_cleaned[['Secondary Outcome Measures']]).ravel()

# Impute missing values for numerical columns
numerical_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='median')
data_cleaned.loc[:, numerical_columns] = imputer.fit_transform(data_cleaned[numerical_columns])

# Handle specific missing value cases
if 'Study Duration (days)' in data_cleaned.columns:
    data_cleaned['Study Duration (days)'] = data_cleaned['Study Duration (days)'].fillna(
        data_cleaned['Study Duration (days)'].median()
    )

# Step 2: Encode categorical data with a cardinality check
categorical_columns = data_cleaned.select_dtypes(include=['object', 'category']).columns

for col in categorical_columns:
    unique_count = data_cleaned[col].nunique()
    if unique_count > 50:  # Set a threshold for high cardinality
        # Convert mixed types to strings before encoding
        data_cleaned[col] = data_cleaned[col].astype(str)
        le = LabelEncoder()
        data_cleaned[col] = le.fit_transform(data_cleaned[col])
    else:
        # Ensure only categorical values for one-hot encoding
        data_cleaned = pd.get_dummies(data_cleaned, columns=[col], drop_first=True)

# If there are still mixed-type columns, handle them as strings
mixed_type_columns = data_cleaned.select_dtypes(include=['object']).columns
for col in mixed_type_columns:
    data_cleaned[col] = data_cleaned[col].astype(str)
    le = LabelEncoder()
    data_cleaned[col] = le.fit_transform(data_cleaned[col])

# Step 3: Standardize numerical features
scaler = StandardScaler()
numerical_features = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numerical_features] = scaler.fit_transform(data_cleaned[numerical_features])

# Step 4: Split the data into training and testing sets
target_column = "Study Recruitment Rate"  # Replace with your target column name
if target_column in data_cleaned.columns:
    X = data_cleaned.drop(columns=[target_column])
    y = data_cleaned[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data Preprocessing Complete.")
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
    
    # Save the cleaned data to a new CSV file
    data_cleaned.to_csv("processed_data.csv", index=False)
    print("Processed data saved to 'processed_data11.csv'.")
else:
    print("Target column not found in the dataset.")
