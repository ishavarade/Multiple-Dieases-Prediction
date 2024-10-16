# -*- coding: utf-8 -*-
"""DSML Kidney Disease - Random Forest Classifier"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import streamlit as st

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set()
plt.style.use('ggplot')

# Load dataset
df = pd.read_csv('kidney_disease.csv')

# Initial dataset checks
df.head()
df['classification'].value_counts()

# Drop unnecessary columns
df.drop('id', axis=1, inplace=True)

# Rename columns for consistency
df.columns = ['age', 'Blood_pressure', 'Specefic_gravity', 'albumin', 'sugar', 'red_blood_cells', 
              'pus_cell', 'pus_cell_clumps', 'Bacteria', 'Blood_glucose_random', 'Blood_urea', 
              'serum_creatinine', 'sodium', 'pottasium', 'haemoglobin', 'packed_cell_volume', 
              'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 
              'coronary_artery_disease', 'appetite', 'peda_edema', 'anaemia', 'classification']

# Convert numeric columns to appropriate data types
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

# Categorize columns into categorical and numerical
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
numeric_cols = [col for col in df.columns if df[col].dtype != 'object']

# Clean specific categorical columns
df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
df['coronary_artery_disease'].replace(to_replace={'\tno': 'no'}, inplace=True)
df['classification'].replace(to_replace={'ckd\t': 'ckd'}, inplace=True)

# Encode target column
df['classification'] = df['classification'].map({'ckd': 0, 'notckd': 1})
df['classification'] = pd.to_numeric(df['classification'], errors='coerce')

# Exploratory Data Analysis (EDA)

# Data Preprocessing

# Handle missing values
def random_sampling(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

def inpute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

for col in numeric_cols:
    random_sampling(col)

for col in categorical_cols:
    inpute_mode(col)

# Check for any remaining missing values
df.isnull().sum()

# Label encoding for categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Model Building - Random Forest Classifier

# Define features and target
X = df.drop('classification', axis=1)
Y = df['classification']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rand_clf = RandomForestClassifier(criterion='gini', max_depth=10, max_features="sqrt", 
                                  min_samples_leaf=1, min_samples_split=7, n_estimators=400)
rand_clf.fit(X_train, Y_train)

def predict_kidney_disease():
    # Streamlit app UI
    st.title('Kidney Disease Prediction')

    st.write("Enter the patient's health metrics to predict if they have chronic kidney disease.")

    # Taking user inputs from the front-end
    age = st.number_input('Age', format="%.2f", min_value=0.0)
    Blood_pressure = st.number_input('Blood Pressure', format="%.2f", min_value=0.0)
    Specefic_gravity = st.number_input('Specific Gravity', format="%.2f", min_value=0.0)
    albumin = st.number_input('Albumin', format="%.2f", min_value=0.0)
    sugar = st.number_input('Sugar', format="%.2f", min_value=0.0)
    red_blood_cells = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
    pus_cell = st.selectbox('Pus Cell', ['normal', 'abnormal'])
    pus_cell_clumps = st.selectbox('Pus Cell Clumps', ['not_present', 'present'])
    Bacteria = st.selectbox('Bacteria', ['not_present', 'present'])
    Blood_glucose_random = st.number_input('Blood Glucose Random', format="%.2f", min_value=0.0)
    Blood_urea = st.number_input('Blood Urea', format="%.2f", min_value=0.0)
    serum_creatinine = st.number_input('Serum Creatinine', format="%.2f", min_value=0.0)
    sodium = st.number_input('Sodium', format="%.2f", min_value=0.0)
    pottasium = st.number_input('Potassium', format="%.2f", min_value=0.0)
    haemoglobin = st.number_input('Haemoglobin', format="%.2f", min_value=0.0)
    packed_cell_volume = st.number_input('Packed Cell Volume', format="%.2f", min_value=0.0)
    white_blood_cell_count = st.number_input('White Blood Cell Count', format="%.2f", min_value=0.0)
    red_blood_cell_count = st.number_input('Red Blood Cell Count', format="%.2f", min_value=0.0)
    hypertension = st.selectbox('Hypertension', ['yes', 'no'])
    diabetes_mellitus = st.selectbox('Diabetes Mellitus', ['yes', 'no'])
    coronary_artery_disease = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
    appetite = st.selectbox('Appetite', ['good', 'poor'])
    peda_edema = st.selectbox('Pedal Edema', ['yes', 'no'])
    anaemia = st.selectbox('Anaemia', ['yes', 'no'])

    # Button to predict
    if st.button('Predict'):
        # Mapping categorical inputs to numeric values
        red_blood_cells = 1 if red_blood_cells == 'normal' else 0
        pus_cell = 1 if pus_cell == 'normal' else 0
        pus_cell_clumps = 1 if pus_cell_clumps == 'present' else 0
        Bacteria = 1 if Bacteria == 'present' else 0
        hypertension = 1 if hypertension == 'yes' else 0
        diabetes_mellitus = 1 if diabetes_mellitus == 'yes' else 0
        coronary_artery_disease = 1 if coronary_artery_disease == 'yes' else 0
        appetite = 1 if appetite == 'good' else 0
        peda_edema = 1 if peda_edema == 'yes' else 0
        anaemia = 1 if anaemia == 'yes' else 0

        # Create input array for prediction
        input_data = np.asarray([
            age, Blood_pressure, Specefic_gravity, albumin, sugar, red_blood_cells, 
            pus_cell, pus_cell_clumps, Bacteria, Blood_glucose_random, Blood_urea, 
            serum_creatinine, sodium, pottasium, haemoglobin, packed_cell_volume, 
            white_blood_cell_count, red_blood_cell_count, hypertension, diabetes_mellitus, 
            coronary_artery_disease, appetite, peda_edema, anaemia
        ])
        input_data_reshape = input_data.reshape(1, -1)

        # Make prediction
        prediction = rand_clf.predict(input_data_reshape)

        if prediction[0] == 1:
            st.write('The patient **has chronic kidney disease (CKD)**.')
        else:
            st.write('The patient **does not have chronic kidney disease**.')

