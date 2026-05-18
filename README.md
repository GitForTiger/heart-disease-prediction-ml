# Heart Disease Prediction using Machine Learning

## 📌 Overview
This project aims to predict whether a person is likely to have heart disease based on medical and lifestyle-related attributes such as age, cholesterol level, chest pain type, blood pressure, maximum heart rate, and exercise-induced angina.

## 📊 Dataset
The dataset contains 918 records with the following features:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope
- Heart Disease

## ⚙️ Steps Performed
- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Checked duplicate and missing values
- Visualized numerical and categorical features
- Replaced unrealistic zero values in cholesterol and resting blood pressure
- Correlation analysis using heatmap
- One-hot encoding of categorical variables
- Feature scaling using StandardScaler

## 🚧 Current Status
Data cleaning, EDA, and preprocessing are completed. Model training and evaluation are in progress.

## 🛠️ Tools Used
- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## 📁 Files
- `Heart Disease Predictor.ipynb` - Jupyter Notebook containing code and analysis
- `Heartt.csv` - Dataset used for prediction

## 🎯 Target Variable
The target variable is `HeartDisease`, where:
- `0` means no heart disease
- `1` means presence of heart disease