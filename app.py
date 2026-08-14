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
import os
import joblib
import streamlit as st

model_path = os.path.join(
    os.path.dirname(__file__),
    "model.pkl"
)

if not os.path.exists(model_path):
    st.error("❌ model.pkl not found!")

    st.write("Files available in app folder:")

    st.write(os.listdir(os.path.dirname(__file__)))

    st.stop()

model = joblib.load(model_path)

st.success("✅ model.pkl loaded successfully!")
    st.success("Model loaded successfully!")

except Exception as e:
    st.error("❌ Unable to load model.pkl")

    st.write("### Exact Error:")
    st.exception(e)

    st.write("### Model file information:")

    import os

    if os.path.exists("model.pkl"):
        st.write("model.pkl exists ✅")
        st.write(
            "File size:",
            os.path.getsize("model.pkl"),
            "bytes"
        )
    else:
        st.write("model.pkl NOT FOUND ❌")

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
