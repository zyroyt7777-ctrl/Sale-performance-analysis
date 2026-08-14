import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# PAGE CONFIGURATION
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
    "Predict Revenue using your Large Sales dataset."
)


# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "large_sales_data.csv"
    )


df = load_data()


# ==========================================
# SHOW DATASET
# ==========================================

st.subheader("📋 Sales Dataset")

st.write(
    f"Total records: {len(df)}"
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ==========================================
# PREPARE DATA
# ==========================================

df = df.dropna(
    subset=["Revenue"]
)


X = df.drop(
    columns=["Revenue"]
)

y = df["Revenue"]


# ==========================================
# FIND CATEGORICAL COLUMNS
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_columns
        )

    ],

    remainder="passthrough"
)


# ==========================================
# MODEL
# ==========================================

model = RandomForestRegressor(

    n_estimators=100,

    random_state=42,

    n_jobs=-1
)


# ==========================================
# PIPELINE
# ==========================================

pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        model
    )

])


# ==========================================
# TRAIN MODEL
# ==========================================

with st.spinner(
    "Training model..."
):

    pipeline.fit(
        X,
        y
    )


st.success(
    "✅ Model trained successfully!"
)


# ==========================================
# INPUT SECTION
# ==========================================

st.sidebar.header(
    "Enter Sales Information"
)


# Get values directly from dataset
# This prevents incorrect category names

for column in categorical_columns:

    values = (
        df[column]
        .dropna()
        .unique()
        .tolist()
    )

    if len(values) > 0:

        selected_value = st.sidebar.selectbox(
            column,
            values
        )

        # Store selected value
        if "inputs" not in st.session_state:
            st.session_state.inputs = {}

        st.session_state.inputs[
            column
        ] = selected_value


# Numerical columns
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


for column in numerical_columns:

    default_value = float(
        df[column].median()
    )

    value = st.sidebar.number_input(
        column,
        value=default_value
    )

    if "inputs" not in st.session_state:
        st.session_state.inputs = {}

    st.session_state.inputs[
        column
    ] = value


# ==========================================
# PREDICTION
# ==========================================

if st.sidebar.button(
    "🔮 Predict Revenue"
):

    input_data = pd.DataFrame(
        [st.session_state.inputs]
    )


    try:

        prediction = pipeline.predict(
            input_data
        )


        st.subheader(
            "💰 Predicted Revenue"
        )


        st.success(
            f"₹ {prediction[0]:,.2f}"
        )


        st.subheader(
            "📋 Input Data"
        )


        st.dataframe(
            input_data,
            use_container_width=True
        )


    except Exception as e:

        st.error(
            "Prediction failed"
        )

        st.exception(e)

