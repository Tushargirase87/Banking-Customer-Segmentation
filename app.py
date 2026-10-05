import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Banking Customer Segmentation",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# ADMIN LOGIN SESSION
# =========================================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False


# =========================================================
# LOAD MODEL
# =========================================================

kmeans = joblib.load(
    "Model/K-Means_Model.pkl"
)

scaler = joblib.load(
    "Model/Scaler.pkl"
)


# =========================================================
# CLUSTER NAMES
# =========================================================

cluster_name = {
    0: "High Value",
    1: "Low Activity",
    2: "Medium Value"
}


cluster_description = {
    0: "High income and spending customers with strong financial activity.",
    1: "Customers with relatively lower transaction and spending activity.",
    2: "Customers showing moderate income, spending and banking activity."
}


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "Data/Banking_Segmentation_Result.csv"
)


# =========================================================
# MODEL FEATURES
# =========================================================

features = [
    "Annual_Income",
    "Account_Balance",
    "Transactions_Per_Month",
    "Loan_Amount",
    "Monthly_Spending"
]


# =========================================================
# CREATE CLUSTER COLUMN
# =========================================================

df_scaled = scaler.transform(
    df[features]
)

df["Cluster"] = kmeans.predict(
    df_scaled
)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

pages = [
    "🏠 Home",
    "🎯 Customer Prediction",
    "📊 Customer Dashboard",
    "🔐 Admin Login"
]


# Show Customer Management only after admin login
if st.session_state.admin_logged_in:

    pages.append(
        "👤 Customer Management"
    )

navigation_key = (
    "page_admin"
    if st.session_state.admin_logged_in
    else "page_user"
)

default_page = (
    "👤 Customer Management"
    if st.session_state.admin_logged_in
    else "🏠 Home"
)

page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(default_page),
    key=navigation_key
)


# =========================================================
# SIDEBAR INFORMATION
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Project"
)

st.sidebar.caption(
    "Algorithm: K-Means Clustering"
)