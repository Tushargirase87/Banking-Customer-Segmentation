# 🏦 Banking Customer Segmentation Using K-Means

A Machine Learning project that groups banking customers into meaningful segments based on their financial behavior and banking activity. The project uses **K-Means Clustering** and provides an interactive **Streamlit web application** for customer prediction, visualization, and management.

## 📌 Project Overview

Banks serve customers with different income levels, account balances, transaction patterns, loan amounts, and spending habits. Understanding these differences helps banks make better business decisions and improve customer engagement.

This project applies the K-Means clustering algorithm to group customers into three segments:

* **High Value** – Customers with relatively high income and spending activity.
* **Medium Value** – Customers showing moderate financial and banking activity.
* **Low Activity** – Customers with relatively lower transaction and spending activity.

The project also includes an interactive Streamlit application where users can predict a customer's segment, explore dashboard insights, and manage customer records through an admin-protected page.

## 🎯 Project Objectives

* Analyze customer financial and banking activity.
* Apply K-Means clustering to identify customer segments.
* Use feature scaling before clustering.
* Compare clustering configurations using the Elbow Method and Silhouette Score.
* Build an interactive Streamlit application.
* Visualize customer distribution and income-versus-spending patterns.
* Add new customer records and manage saved customer data.
* Present cluster summaries and business insights.

## 📂 Project Structure

```text
Banking-Customer-Segmentation/
│
├── Data/
│   └── Banking_Segmentation_Result.csv
│
├── Model/
│   ├── K-Means_Model.pkl
│   └── Scaler.pkl
│
├── Notebook/
│   └── Banking_Customer_Segmentation.ipynb
│
├── .streamlit/
│   └── secrets.toml          # Local only; not committed to GitHub
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### 📁 Folder and File Description

| File / Folder | Description |
|---|---|
| `Data/` | Stores the customer segmentation dataset and saved customer records. |
| `Model/` | Contains the trained K-Means model and feature scaler. |
| `Notebook/` | Contains the Jupyter Notebook used for data analysis and model development. |
| `.streamlit/` | Stores local Streamlit secrets for admin authentication. |
| `app.py` | Runs the interactive Streamlit web application. |
| `requirements.txt` | Lists the Python libraries required to run the project. |
| `README.md` | Documents the project, features, setup, and usage. |
| `.gitignore` | Prevents selected local and sensitive files from being committed to GitHub.