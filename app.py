import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Sales Revenue Prediction",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📊 Sales Revenue Prediction")

st.write(
    "Enter the sales information to predict Revenue."
)


# ==========================================
# FIND MODEL FILE
# ==========================================

model_path = os.path.join(
    os.path.dirname(__file__),
    "model.pkl"
)


# ==========================================
# CHECK MODEL FILE
# ==========================================

if not os.path.exists(model_path):

    st.error("❌ model.pkl not found!")

    st.write("Files found in the application folder:")

    st.write(
        os.listdir(os.path.dirname(__file__))
    )

    st.stop()


# ==========================================
# LOAD MODEL
# ==========================================

try:

    model = joblib.load(model_path)

    st.success(
        "✅ Model loaded successfully!"
    )

except Exception as e:

    st.error(
        "❌ Error loading model.pkl"
    )

    st.exception(e)

    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "Sales Information"
)


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
    value="John"
)


product_category = st.sidebar.text_input(
    "Product Category",
    value="Electronics"
)


units_sold = st.sidebar.number_input(
    "Units Sold",
    min_value=0,
    value=10,
    step=1
)


# ==========================================
# PREDICTION
# ==========================================

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


        # ==================================
        # RESULT
        # ==================================

        st.subheader(
            "💰 Predicted Revenue"
        )

        st.success(
            f"₹ {prediction[0]:,.2f}"
        )


        # ==================================
        # INPUT DATA
        # ==================================

        st.subheader(
            "📋 Sales Information"
        )

        st.dataframe(
            input_data,
            use_container_width=True
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed"
        )

        st.exception(e)
