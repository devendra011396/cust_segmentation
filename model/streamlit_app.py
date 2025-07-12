import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and preprocessor
model = joblib.load("model/customer_segment_model.joblib")
preprocessor = joblib.load("model/scaler.joblib")

# Load training data for summary/visualization
df_train = pd.read_csv("dataset/customer_data.csv").drop(columns=["CustomerID", "Last_Purchase_Date"])
df_train["Cluster"] = model.predict(preprocessor.transform(df_train))

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation Dashboard")

# Sidebar input
st.sidebar.header("📥 Enter Customer Details")
age = st.sidebar.slider("Age", 18, 100, 30)
income = st.sidebar.slider("Annual Income (₹)", 10000, 200000, 50000, step=1000)
score = st.sidebar.slider("Spending Score", 0, 100, 50)
freq = st.sidebar.slider("Purchase Frequency", 1, 30, 10)
spend = st.sidebar.slider("Total Spending (₹)", 0.0, 50000.0, 10000.0)
family = st.sidebar.slider("Family Size", 1, 10, 3)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Widowed", "Divorced"])

if st.sidebar.button("Predict Segment"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Annual_Income": income,
        "Spending_Score": score,
        "Purchase_Frequency": freq,
        "Total_Spending": spend,
        "Family_Size": family,
        "Gender": gender,
        "Marital_Status": marital
    }])

    transformed = preprocessor.transform(input_df)
    cluster = model.predict(transformed)[0]
    st.success(f"The customer belongs to **Segment {cluster}**")

# Tabs for additional features
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data Sample", "📊 Cluster Summary", "📉 Distribution", "📂 CSV Upload", "🧠 Segment Logic"
])

with tab1:
    st.subheader("📁 Sample of Training Data")
    st.dataframe(df_train.head(10))

with tab2:
    st.subheader("📊 Cluster-Wise Summary")
    st.dataframe(df_train.groupby("Cluster").mean(numeric_only=True))

with tab3:
    st.subheader("📉 Cluster Distribution")
    cluster_counts = df_train["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.pie(cluster_counts, labels=[f"Cluster {i}" for i in cluster_counts.index],
           autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

with tab4:
    st.subheader("📂 Upload CSV for Bulk Prediction")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        try:
            user_data = pd.read_csv(uploaded)
            required_cols = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency",
                             "Total_Spending", "Family_Size", "Gender", "Marital_Status"]
            missing_cols = set(required_cols) - set(user_data.columns)
            if missing_cols:
                st.error(f"❌ Error: Missing columns: {missing_cols}")
            else:
                predictions = model.predict(preprocessor.transform(user_data))
                user_data["Predicted_Cluster"] = predictions
                st.write(user_data)
                csv_out = user_data.to_csv(index=False).encode("utf-8")
                st.download_button("Download Result CSV", csv_out, "predicted_clusters.csv")
        except Exception as e:
            st.error(f"❌ Failed to process uploaded CSV: {e}")


with tab5:
    st.subheader("🧠 Segment Logic (Assumed Interpretation)")
    st.markdown("""
- **Segment 0:** Young, high-spending, frequent buyers.
- **Segment 1:** Older customers, less frequent, moderate income.
- **Segment 2:** High income, high spenders, likely families.
- **Segment 3:** Low income, low engagement.
""")
