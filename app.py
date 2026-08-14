import streamlit as st
import pandas as pd
import joblib


# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="Sales Revenue Prediction",
    page_icon="📊",
    layout="wide"
)


# =====================================
# TITLE
# =====================================

st.title("📊 Sales Revenue Prediction")

st.write(
    "Enter sales information to predict Revenue."
)


# =====================================
# LOAD MODEL
# =====================================

try:

    model = joblib.load("model.pkl")

except Exception as e:

    st.error("Unable to load model.pkl")

    st.code(str(e))

    st.stop()


# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("Sales Information")


region = st.sidebar.selectbox(
    "Region",
    [
        "North",
        "South",
        "East",
        "West"
    ]
)


salesperson = st.sidebar.text_input(
    "Salesperson",
    "John"
)


product_category = st.sidebar.text_input(
    "Product Category",
    "Electronics"
)


units_sold = st.sidebar.number_input(
    "Units Sold",
    min_value=0,
    value=10,
    step=1
)


# =====================================
# PREDICTION
# =====================================

if st.sidebar.button("Predict Revenue"):

    input_data = pd.DataFrame({

        "Region": [region],

        "Salesperson": [salesperson],

        "Product_Category": [product_category],

        "Units_Sold": [units_sold]

    })


    try:

        prediction = model.predict(
            input_data
        )


        st.subheader("💰 Predicted Revenue")


        st.success(
            f"₹ {prediction[0]:,.2f}"
        )


        st.subheader("Input Information")


        st.dataframe(
            input_data,
            use_container_width=True
        )


    except Exception as e:

        st.error("Prediction error")

        st.code(str(e))
