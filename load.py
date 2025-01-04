import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "usecase_4_.xlsx"
data = pd.read_excel(file_path)

print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values in Each Column:")
print(data.isnull().sum())

if 'Study Recruitment Rate' in data.columns:
    print("\nDistribution of Study Recruitment Rate:")
    sns.histplot(data['Study Recruitment Rate'], kde=True, bins=20)
    plt.title("Distribution of Study Recruitment Rate")
    plt.xlabel("Study Recruitment Rate")
    plt.ylabel("Frequency")
    plt.show()

date_columns = ['First Posted', 'Results First Posted', 'Last Update Posted']
for col in date_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
        print(f"\nConverted {col} to datetime. Missing values after conversion:")
        print(data[col].isnull().sum())

if 'First Posted' in data.columns and 'Last Update Posted' in data.columns:
    data['Study Duration (days)'] = (data['Last Update Posted'] - data['First Posted']).dt.days
    print("\nStudy Duration (days):")
    print(data['Study Duration (days)'].describe())

    if 'Study Recruitment Rate' in data.columns:
        print("\nScatter Plot: Study Recruitment Rate vs Study Duration (days)")
        sns.scatterplot(x='Study Duration (days)', y='Study Recruitment Rate', data=data)
        plt.title("Study Recruitment Rate vs Study Duration")
        plt.xlabel("Study Duration (days)")
        plt.ylabel("Study Recruitment Rate")
        plt.show()

if 'Locations' in data.columns:
    print("\nSample Locations Data:")
    print(data['Locations'].head())
    print("\nNumber of unique location entries:")
    print(data['Locations'].nunique())

if 'Study Recruitment Rate' in data.columns and 'Study Duration (days)' in data.columns:
    print("\nCorrelation between Study Recruitment Rate and Study Duration (days):")
    print(data[['Study Recruitment Rate', 'Study Duration (days)']].corr())

numerical_columns = ['Study Recruitment Rate', 'Study Duration (days)']  
numerical_data = data[numerical_columns].dropna()
if not numerical_data.empty:
    print("\nCorrelation Matrix:")
    corr_matrix = numerical_data.corr()
    print(corr_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
