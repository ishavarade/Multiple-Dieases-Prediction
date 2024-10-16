import streamlit as st
import pickle
import numpy as np
import heart_disease_sakshi as hd
import breast_cancer_isha as bc
import dsml_kidney as kd

# Navigation mechanism (via URL or session state)
if 'page' not in st.session_state:
    st.session_state.page = 'index'

# Function to navigate between pages
def navigate(page):
    st.session_state.page = page

# Index Page (main landing page)
if st.session_state.page == 'index':
    st.title('Welcome to the Disease Prediction App')

    st.write('Select a disease to learn more and make a prediction:')
    
    # Disease options
    if st.button('Heart Disease'):
        navigate('heart_info')
    if st.button('Breast Cancer'):
        navigate('breast_info')
    if st.button('Kidney Disease'):
        navigate('kidney_info')

# Information Pages
elif st.session_state.page == 'heart_info':
    st.title('Heart Disease Information')
    st.write("""
        Heart disease refers to various types of heart conditions, such as coronary artery disease, arrhythmias, and more.
        Common risk factors include high cholesterol, high blood pressure, smoking, diabetes, and family history.
    """)
    
    if st.button('Proceed to Heart Disease Prediction'):
        navigate('heart_predict')

elif st.session_state.page == 'breast_info':
    st.title('Breast Cancer Information')
    st.write("""
        Breast cancer is a disease in which malignant (cancer) cells form in the tissues of the breast. 
        Factors such as age, genetics, and lifestyle can influence the risk of developing breast cancer.
    """)
    
    if st.button('Proceed to Breast Cancer Prediction'):
        navigate('breast_predict')

elif st.session_state.page == 'kidney_info':
    st.title('Kidney Disease Information')
    st.write("""
        Kidney disease occurs when your kidneys are damaged and cannot filter blood the way they should.
        Risk factors include high blood pressure, diabetes, and a family history of kidney failure.
    """)
    
    if st.button('Proceed to Kidney Disease Prediction'):
        navigate('kidney_predict')

# Prediction Pages
elif st.session_state.page == 'heart_predict':
    st.title('Heart Disease Prediction')
    
    features = hd.user_input_features()
    
    if st.button('Predict Heart Disease'):
        hd.predict_disease(features)


elif st.session_state.page == 'breast_predict':
    st.title('Breast Cancer Prediction')
    
    bc.predict_disease()

elif st.session_state.page == 'kidney_predict':
    st.title('Kidney Disease Prediction')
    
    kd.predict_kidney_disease()

# Back Button to Index Page
if st.button('Go Back to Index'):
    navigate('index')

