

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

# Title and description
st.title("Heart Disease Prediction")
st.markdown("""
This app predicts the risk of heart disease based on several health metrics like age, cholesterol levels, blood pressure, etc.
You can enter the health metrics manually to predict the risk of heart disease.
""")

# Sidebar for user input
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=35, key='age')
    cholesterol = st.sidebar.number_input('Cholesterol', min_value=100, max_value=500, value=200, key='cholesterol')
    systolic_bp = st.sidebar.number_input('Systolic Blood Pressure', min_value=80, max_value=200, value=120, key='systolic_bp')
    diastolic_bp = st.sidebar.number_input('Diastolic Blood Pressure', min_value=50, max_value=120, value=80, key='diastolic_bp')
    active_hours = st.sidebar.number_input('Active Hours per Day', min_value=0, max_value=24, value=6, key='active_hours')
    sedentary_hours = st.sidebar.number_input('Sedentary Hours per Day', min_value=0, max_value=24, value=18, key='sedentary_hours')
    
    data = {
        'Age': age,
        'Cholesterol': cholesterol,
        'Systolic Blood Pressure': systolic_bp,
        'Diastolic Blood Pressure': diastolic_bp,
        'Active Hours': active_hours,
        'Sedentary Hours': sedentary_hours
    }
    features = pd.DataFrame(data, index=[0])
    return features


def predict_disease(dataframe):

    scaler = RobustScaler()
    model = LogisticRegression()


    train_data = pd.DataFrame({
        'Age': np.random.randint(30, 80, 100),
        'Cholesterol': np.random.randint(150, 300, 100),
        'Systolic Blood Pressure': np.random.randint(110, 180, 100),
        'Diastolic Blood Pressure': np.random.randint(60, 120, 100),
        'Active Hours': np.random.randint(1, 10, 100),
        'Sedentary Hours': np.random.randint(10, 20, 100)
    })

    train_labels = np.random.randint(0, 2, 100)

# Fit scaler and model using hypothetical data
    scaler.fit(train_data)
    train_data_scaled = scaler.transform(train_data)
    model.fit(train_data_scaled, train_labels)

    # Make prediction based on manually input data
    if st.button("Predict from Manual Input"):
    # Scale the user input based on the scaler used for training data
        input_scaled = scaler.transform(dataframe)

    # Use the trained model to predict the result for the manual input
        prediction = model.predict(input_scaled)

        if prediction == 1:
            st.write("Prediction: High risk of heart disease")
        else:
            st.write("Prediction: Low risk of heart disease")



# Collect user input
#input_data = user_input_features()

# Display the input data for review
#st.write("### Input Health Metrics:")
#st.write(input_data)

# Model initialization
#scaler = RobustScaler()
#model = LogisticRegression()

# Hypothetical training data (for example purposes)
# You would normally train your model with real data
# Make prediction based on manually input data
if st.button("Predict from Manual Input"):
    # Scale the user input based on the scaler used for training data
    input_scaled = scaler.transform(input_data)

    # Use the trained model to predict the result for the manual input
    prediction = model.predict(input_scaled)

    if prediction == 1:
        st.write("Prediction: High risk of heart disease")
    else:
        st.write("Prediction: Low risk of heart disease")












