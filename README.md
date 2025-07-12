# customerSegmentation project

A smart and interactive web application for performing **customer segmentation** using **KMeans clustering**.  
Built with **Streamlit**, this app allows users to upload their own customer dataset, analyze clusters dynamically, and explore rich insights through visual dashboards.

---

## 🚀 Features

- 🔍 Upload your own dataset with custom mapping (coming soon)
- 📊 Visualize and filter cluster segments in real-time
- 📈 Radar chart of average metrics per segment
- 📉 Compare features with interactive scatter plots
- 📥 Bulk CSV upload and downloadable predictions
- 🧠 AI-driven dynamic interpretation of each segment

---

## 📁 Expected Columns

Your uploaded dataset should contain the following columns:

```
Age, Annual_Income, Spending_Score, Purchase_Frequency,
Total_Spending, Family_Size, Gender, Marital_Status
```

If the dataset structure differs, upcoming versions will allow column mapping or retraining.

---

## 🧪 How It Works

1. Upload a customer dataset CSV.
2. The app preprocesses the data and predicts a segment using a trained KMeans model.
3. Explore visual insights like:
   - Cluster distribution
   - Feature trends
   - Segment explanations

---

## 📦 Tech Stack

- `Python`
- `Pandas`, `Scikit-learn`, `Matplotlib`, `Seaborn`
- `Streamlit` for UI
- `Joblib` for model persistence

---

## 👨‍💻 Authors

- **Devendra Gurav**
- **Pankaj Bhandari**

---

## 🏁 Run Locally

```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
streamlit run model/streamlit_app.py
```

---

📫 For suggestions or improvements, feel free to reach out or fork the repo!
