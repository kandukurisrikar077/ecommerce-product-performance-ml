
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "best_model.pkl"
ENCODER_DIR = BASE_DIR / "label_encoder.pkl"
FEATURE_DIR = BASE_DIR / "feature_names.pkl"

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR)
    label_encoder = joblib.load(ENCODER_DIR)
    features = joblib.load(FEATURE_DIR)

    return model, label_encoder, features

model, label_encoder, features = load_artifacts()

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Product Performance Predictor",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Product Performance Predictor")

st.write(
    "Predict e-commerce product performance "
    "using a trained Machine Learning model."
)

st.divider()

# ==========================================
# USER INPUTS
# ==========================================

st.subheader("Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=5000.0
    )

    discount = st.slider(
        "Discount Rate",
        0.0, 1.0, 0.2
    )

    rating = st.slider(
        "Product Rating",
        1.0, 5.0, 4.0
    )

    reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        value=500
    )

with col2:
    stock = st.slider(
        "Stock Availability",
        0.0, 1.0, 0.8
    )

    delivery = st.number_input(
        "Days to Deliver",
        min_value=0,
        value=5
    )

    returns = st.slider(
        "Return Rate",
        0.0, 1.0, 0.1
    )

    category = st.number_input(
        "Category ID",
        min_value=0,
        value=10
    )

# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Product Performance", type="primary"):

    input_df = pd.DataFrame(
        [[
            price,
            discount,
            rating,
            reviews,
            stock,
            delivery,
            returns,
            category
        ]],
        columns=features
    )

    try:
        # Pipeline handles scaling internally
        prediction = model.predict(input_df)

        # Convert encoded output to original label
        result = label_encoder.inverse_transform(prediction)[0]

        st.divider()
        st.subheader("Prediction Result")

        if result == "High":
            st.success("Product Performance: HIGH")

        elif result == "Moderate":
            st.info("Product Performance: MODERATE")

        else:
            st.warning("Product Performance: LOW")

        st.caption(
            "Prediction generated using the trained "
            "Logistic Regression pipeline."
        )

        with st.expander("View Input Details"):
            st.dataframe(input_df, width="stretch")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
