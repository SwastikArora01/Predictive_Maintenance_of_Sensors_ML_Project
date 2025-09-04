# ⚙️ Predictive Maintenance Dashboard  

A **Streamlit-based web application** that leverages machine learning models to provide **predictive maintenance insights** for lab sensors operating under various conditions.  

This dashboard helps monitor sensor readings, predict **Remaining Useful Life (RUL)**, detect anomalies, and determine when maintenance is required — all with interactive data visualizations.  

---

## 🚀 Features  

- 📂 **Historical Data View** – Explore past sensor readings  
- 🔧 **Custom Input Data** – Enter or auto-generate sensor values  
- 📊 **Prediction Results** –  
  - Remaining Useful Life (RUL) in hours  
  - Maintenance status (Normal / Needs Maintenance)  
  - Anomaly detection (Normal / Anomaly)  
- 📈 **Data Visualizations** –  
  - Histograms of sensor readings  
  - Scatter plots vs. operational hours  
  - Boxplots by maintenance status  
  - Correlation heatmaps  
  - Feature importance for RUL prediction  

---

## 🛠️ Tech Stack  

- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Machine Learning:** Scikit-learn (Regression, Classification, Clustering)  
- **Data Handling:** pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning Models Used:**  
  - **Random Forest Regressor** → for RUL prediction  
  - **Random Forest Classifier** → for maintenance status prediction  
  - **KMeans Clustering** → for anomaly detection

---

## 📦 Installation  

1. Clone the repository:  
```bash
git clone https://github.com/your-username/predictive-maintenance-dashboard.git
cd predictive-maintenance-dashboard
