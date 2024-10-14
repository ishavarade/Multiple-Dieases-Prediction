import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data.csv')

# Preprocessing
columns_to_drop = ['id', 'Unnamed: 32']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})

selected_columns = [
    'texture_mean', 'smoothness_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'texture_se', 'smoothness_se', 'symmetry_se', 'fractal_dimension_se'
]

# Seperate the features and target
X = df[selected_columns]
y = df['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Logistic Regression Model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)

# Train the Random Forest model using the training data
rf_model.fit(X_train, y_train)


def predict_disease():
# Streamlit app UI
    st.title('Breast Cancer Prediction')

    st.write("Enter the tumor characteristics to predict if it's malignant or benign.")

# Taking user inputs from the front-end with specific format to prevent rounding
    texture_mean = st.number_input('Texture Mean', format="%.5f", min_value=0.0)
    smoothness_mean = st.number_input('Smoothness Mean', format="%.5f", min_value=0.0)
    symmetry_mean = st.number_input('Symmetry Mean', format="%.5f", min_value=0.0)
    fractal_dimension_mean = st.number_input('Fractal Dimension Mean', format="%.5f", min_value=0.0)
    texture_se = st.number_input('Texture SE', format="%.5f", min_value=0.0)
    smoothness_se = st.number_input('Smoothness SE', format="%.5f", min_value=0.0)
    symmetry_se = st.number_input('Symmetry SE', format="%.5f", min_value=0.0)
    fractal_dimension_se = st.number_input('Fractal Dimension SE', format="%.5f", min_value=0.0)

# Button to predict
    if st.button('Predict'):
    # Create input array for prediction
        input_data = np.asarray([texture_mean, smoothness_mean, symmetry_mean, fractal_dimension_mean,
                             texture_se, smoothness_se, symmetry_se, fractal_dimension_se])
        input_data_reshape = input_data.reshape(1, -1)

    # Make prediction
        prediction =rf_model.predict(input_data_reshape)

        if prediction[0] == 0:
            st.write('The breast cancer is **malignant**.')
        else:
            st.write('The breast cancer is **benign**.')
