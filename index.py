import streamlit as st

# Set page title
st.set_page_config(page_title="HealthCure - An All-in-One Medical Solution", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .header {
            font-size: 50px;
            font-weight: bold;
            text-align: center;
        }
        .subheader {
            font-size: 24px;
            text-align: center;
        }
        .description {
            text-align: center;
            font-size: 18px;
        }
        .detection-title {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
        }
        .detection-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 301px;
            height: 168px;
            object-fit: cover;
        }
        .disease-container {
            display: flex;
            justify-content: space-around;
            margin-top: 30px;
        }
        .disease-card {
            text-align: center;
            width: 100%;
        }
        .detection-text {
            font-size: 18px;
            text-align: center;
        }
        .navbar {
            background-color: #333;
            padding: 10px;
            text-align: right;
        }
        .navbar a {
            color: white;
            margin-left: 20px;
            text-decoration: none;
            font-weight: bold;
        }
        .main-content {
            width: 100%;
            display: flex;
            justify-content: space-evenly;
        }
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
    <div class="navbar">
        <a href="#">Covid</a>
        <a href="#">Brain Tumor</a>
        <a href="#">Breast Cancer</a>
        <a href="#">Alzheimer</a>
        <a href="#">Diabetes</a>
        <a href="#">Pneumonia</a>
        <a href="#">Heart Disease</a>
    </div>
""", unsafe_allow_html=True)

# Header section
st.markdown('<p class="header">HealthCure</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">An All-in-One Medical Solution</p>', unsafe_allow_html=True)
st.markdown('<p class="description">HealthCure is an all-in-one medical solution app that offers disease detection, including Heart Attack, Kidney Disease, and Breast Cancer Detection.</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">3 Disease Detections</p>', unsafe_allow_html=True)

# Display three detection cards occupying the full page width
st.markdown('<div class="main-content">', unsafe_allow_html=True)

with st.container():
    # Full width column layout with fixed image size
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="detection-title">Heart Attack Detection</p>', unsafe_allow_html=True)
        st.image("C:/TY/DSML/templates/heart.jpeg", use_column_width=True)  # Adjust image path
        st.markdown('<p class="detection-text">Early detection of heart attacks to improve survival rates.</p>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="detection-title">Kidney Disease Detection</p>', unsafe_allow_html=True)
        st.image("C:/TY/DSML/templates/kidney.jpeg", use_column_width=True)  # Adjust image path
        st.markdown('<p class="detection-text">Monitor kidney health and detect early signs of kidney disease.</p>', unsafe_allow_html=True)

    with col3:
        st.markdown('<p class="detection-title">Breast Cancer Detection</p>', unsafe_allow_html=True)
        st.image("C:/TY/DSML/templates/breast.jpeg", use_column_width=True)  # Adjust image path
        st.markdown('<p class="detection-text">Detect breast cancer early and improve treatment outcomes.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer section
st.markdown('<p class="description">HealthCure brings multiple health detections under one platform for better accessibility and early diagnosis.</p>', unsafe_allow_html=True)
