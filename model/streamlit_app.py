import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and preprocessor
model = joblib.load("model/customer_segment_model.joblib")
preprocessor = joblib.load("model/scaler.joblib")

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation Dashboard")

# Upload data section (must be uploaded to use the app)
uploaded_data = st.sidebar.file_uploader("📂 Upload a CSV file to begin", type=["csv"])

if uploaded_data:
    try:
        df = pd.read_csv(uploaded_data)
        required_cols = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency",
                         "Total_Spending", "Family_Size", "Gender", "Marital_Status"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            st.error(f"❌ Error: Missing columns: {missing_cols}")
        else:
            df["Cluster"] = model.predict(preprocessor.transform(df))

            # Sidebar input
            st.sidebar.header("📥 Predict Segment for a New Customer")
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
                prediction = model.predict(preprocessor.transform(input_df))[0]
                st.success(f"The customer belongs to **Segment {prediction}**")

            # Main dashboard
            tab1, tab2, tab3, tab4 = st.tabs([
                "📁 Uploaded Data", "📊 Cluster Summary", "📉 Distribution", "🧠 Segment Logic"
            ])

            with tab1:
                st.subheader("📁 Uploaded Customer Data with Cluster Labels")
                st.dataframe(df)
                # 💾 Add download button
                csv_clustered = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                label="📥 Download Clustered Data as CSV",
                data=csv_clustered,
                file_name="clustered_customers.csv",
                mime="text/csv")

            with tab2:
                st.subheader("📊 Cluster-Wise Summary")
                st.dataframe(df.groupby("Cluster").mean(numeric_only=True))

            with tab3:
                st.subheader("📉 Cluster Distribution")
                cluster_counts = df["Cluster"].value_counts().sort_index()
                fig, ax = plt.subplots()
                ax.pie(cluster_counts, labels=[f"Cluster {i}" for i in cluster_counts.index],
                       autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

            with tab4:
                st.subheader("🧠 Segment Logic (Assumed Interpretation)")
                for cluster_id in sorted(df["Cluster"].unique()):
                cluster_df = df[df["Cluster"] == cluster_id]
                st.markdown(f"### 🔹 Segment {cluster_id}")
                st.write(f"🧮 **Total Customers:** {len(cluster_df)}")
                st.write(f"📊 **Avg Age:** {cluster_df['Age'].mean():.1f}")
                st.write(f"💸 **Avg Income:** ₹{cluster_df['Annual_Income'].mean():,.0f}")
                st.write(f"💳 **Avg Spending Score:** {cluster_df['Spending_Score'].mean():.1f}")
                st.write(f"🛍️ **Avg Total Spending:** ₹{cluster_df['Total_Spending'].mean():,.0f}")
                st.write(f"👨‍👩‍👧‍👦 **Avg Family Size:** {cluster_df['Family_Size'].mean():.1f}")
                st.markdown("---")
    except Exception as e:
        st.error(f"❌ Failed to process uploaded CSV: {e}")
else:
    st.warning("📂 Please upload a customer dataset CSV file to begin using the dashboard.")
