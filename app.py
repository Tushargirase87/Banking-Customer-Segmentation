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

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title(
        "🏦 Banking Customer Segmentation"
    )

    st.subheader(
        "Customer Segmentation using K-Means Clustering"
    )

    st.write(
        "This application uses Machine Learning to group "
        "banking customers based on their financial and "
        "transactional behaviour."
    )

    st.divider()

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Total Customers",
            len(df)
        )

    with col2:

        st.metric(
            "🎯 Number of Clusters",
            df["Cluster"].nunique()
        )

    with col3:

        st.metric(
            "💰 Avg Income",
            f"₹{df['Annual_Income'].mean():,.0f}"
        )

    with col4:

        st.metric(
            "💳 Avg Spending",
            f"₹{df['Monthly_Spending'].mean():,.0f}"
        )

    st.divider()

    # =====================================================
    # PROJECT OVERVIEW
    # =====================================================

    st.subheader(
        "📌 Project Overview"
    )

    st.write(
        """
        The Banking Customer Segmentation project uses the
        K-Means clustering algorithm to identify groups of
        customers with similar financial and behavioural
        characteristics.

        The model considers factors such as:

        • Annual Income

        • Account Balance

        • Transactions Per Month

        • Loan Amount

        • Monthly Spending
        """
    )

    st.info(
        "💡 Use the sidebar to predict a new customer's "
        "segment or explore the customer dashboard."
    )


# =========================================================
# CUSTOMER PREDICTION PAGE
# =========================================================

elif page == "🎯 Customer Prediction":

    st.title(
        "🎯 Customer Segment Prediction"
    )

    st.write(
        "Enter customer information below to predict "
        "the customer's segment."
    )

    st.divider()

    # =====================================================
    # CUSTOMER INPUT
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "🎂 Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        annual_income = st.number_input(
            "💰 Annual Income",
            min_value=0,
            value=50000,
            step=1000
        )

        account_balance = st.number_input(
            "🏦 Account Balance",
            min_value=0,
            value=15000,
            step=1000
        )

        transactions = st.number_input(
            "💳 Transactions Per Month",
            min_value=0,
            value=15,
            step=1
        )

    with col2:

        credit_score = st.number_input(
            "📊 Credit Score",
            min_value=300,
            max_value=900,
            value=700,
            step=1
        )

        loan_amount = st.number_input(
            "💵 Loan Amount",
            min_value=0,
            value=20000,
            step=1000
        )

        monthly_spending = st.number_input(
            "🛒 Monthly Spending",
            min_value=0,
            value=5000,
            step=500
        )

    st.divider()

    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    predict_button = st.button(
        "🔮 Predict Customer Segment",
        use_container_width=True
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        new_customer = pd.DataFrame({

            "Annual_Income": [
                annual_income
            ],

            "Account_Balance": [
                account_balance
            ],

            "Transactions_Per_Month": [
                transactions
            ],

            "Loan_Amount": [
                loan_amount
            ],

            "Monthly_Spending": [
                monthly_spending
            ]
        })


        # Scale customer data

        new_customer_scaled = scaler.transform(
            new_customer
        )


        # Predict cluster

        predicted_cluster = kmeans.predict(
            new_customer_scaled
        )[0]


        # Get segment

        predict_segment = cluster_name[
            predicted_cluster
        ]


        # Save prediction in session

        st.session_state[
            "predicted_cluster"
        ] = predicted_cluster


        st.session_state[
            "predict_segment"
        ] = predict_segment


        # Save customer information

        st.session_state[
            "customer_data"
        ] = {

            "Age": age,

            "Annual_Income": annual_income,

            "Account_Balance": account_balance,

            "Credit_Score": credit_score,

            "Transactions_Per_Month": transactions,

            "Loan_Amount": loan_amount,

            "Monthly_Spending": monthly_spending
        }
    
    # =====================================================
    # SHOW RESULT
    # =====================================================

    if "predicted_cluster" in st.session_state:

        predicted_cluster = (
            st.session_state[
                "predicted_cluster"
            ]
        )

        predict_segment = (
            st.session_state[
                "predict_segment"
            ]
        )

        customer_data = (
            st.session_state[
                "customer_data"
            ]
        )

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.success(
                f"🎯 Customer Segment: "
                f"{predict_segment}"
            )

        with result_col2:

            st.info(
                f"🔢 Cluster: "
                f"{predicted_cluster}"
            )

        st.write(
            f"📌 **Customer Profile:** "
            f"{cluster_description[predicted_cluster]}"
        )

        st.divider()

        # =================================================
        # CUSTOMER INFORMATION
        # =================================================

        st.subheader(
            "📋 Customer Information"
        )

        customer_summary = pd.DataFrame({

            "Feature": [

                "Age",

                "Annual Income",

                "Account Balance",

                "Credit Score",

                "Transactions Per Month",

                "Loan Amount",

                "Monthly Spending"
            ],

            "Value": [

                customer_data["Age"],

                f"₹{customer_data['Annual_Income']:,.0f}",

                f"₹{customer_data['Account_Balance']:,.0f}",

                customer_data["Credit_Score"],

                customer_data[
                    "Transactions_Per_Month"
                ],

                f"₹{customer_data['Loan_Amount']:,.0f}",

                f"₹{customer_data['Monthly_Spending']:,.0f}"
            ]
        })

        st.dataframe(
            customer_summary,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # =================================================
        # SAVE CUSTOMER
        # =================================================

        save_customer = st.button(
            "💾 Save Customer to Dataset",
            use_container_width=True
        )

        if save_customer:

            # =============================================
            # GENERATE CUSTOMER ID
            # =============================================

            existing_ids = (
                df["Customer_ID"]
                .astype(str)
            )

            numeric_ids = pd.to_numeric(

                existing_ids.str.extract(
                    r"(\d+)",
                    expand=False
                ),

                errors="coerce"
            )

            if numeric_ids.notna().any():

                next_id = (
                    int(numeric_ids.max()) + 1
                )

            else:

                next_id = 1


            customer_id = (
                f"{next_id:05d}"
            )


            # =============================================
            # CREATE NEW CUSTOMER
            # =============================================

            new_row = pd.DataFrame({

                "Customer_ID": [
                    customer_id
                ],

                "Age": [
                    customer_data["Age"]
                ],

                "Annual_Income": [
                    customer_data["Annual_Income"]
                ],

                "Account_Balance": [
                    customer_data["Account_Balance"]
                ],

                "Credit_Score": [
                    customer_data["Credit_Score"]
                ],

                "Transactions_Per_Month": [
                    customer_data[
                        "Transactions_Per_Month"
                    ]
                ],

                "Loan_Amount": [
                    customer_data["Loan_Amount"]
                ],

                "Monthly_Spending": [
                    customer_data[
                        "Monthly_Spending"
                    ]
                ],

                "cluster": [
                    predicted_cluster
                ],

                "Customer_Segment": [
                    predict_segment
                ]
            })


            # =============================================
            # ADD CUSTOMER
            # =============================================

            df = pd.concat(
                [
                    df,
                    new_row
                ],
                ignore_index=True
            )


            # =============================================
            # SAVE DATASET
            # =============================================

            try:

                df.to_csv(
                    "Data/Banking_Segmentation_Result.csv",
                    index=False
                )

                st.success(
                    f"✅ Customer {customer_id} "
                    f"saved successfully!"
                )

            except PermissionError:

                st.error(
                    "⚠️ Unable to save the customer record. "
                    "Please close the dataset file in Excel "
                    "or any other application and try again."
                )
