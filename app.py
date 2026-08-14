import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("model.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Large Sales Prediction",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📊 Large Sales Data - Revenue Prediction")

st.write(
    "Enter the sales information below to predict the expected revenue."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Sales Information")


# Region
region = st.sidebar.selectbox(
    "Region",
    [
        "North",
        "South",
        "East",
        "West"
    ]
)


# Salesperson
salesperson = st.sidebar.text_input(
    "Salesperson",
    value=""
)


# Product Category
product_category = st.sidebar.text_input(
    "Product Category",
    value=""
)


# Units Sold
units_sold = st.sidebar.number_input(
    "Units Sold",
    min_value=0,
    value=1,
    step=1
)


# ==========================================
# PREDICTION
# ==========================================

if st.sidebar.button("Predict Revenue"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Region": [region],
        "Salesperson": [salesperson],
        "Product_Category": [product_category],
        "Units_Sold": [units_sold]
    })


    # Make prediction
    prediction = model.predict(input_data)


    # Display result
    st.subheader("💰 Predicted Revenue")

    st.success(
        f"₹ {prediction[0]:,.2f}"
    )


    # Display entered information
    st.subheader("📋 Sales Information")

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ==========================================
# INFORMATION
# ==========================================

st.divider()

st.subheader("About the Model")

st.write(
    """
    This application uses a Random Forest Regression model
    trained on the Large Sales Data dataset.

    The model predicts Revenue based on the sales information
    provided by the user.
    """
)
