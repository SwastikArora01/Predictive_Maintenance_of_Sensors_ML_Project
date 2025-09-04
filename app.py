import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# RIGHT HERE, before you import or render anything else:
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    css = css_path.read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.warning("Couldn’t find style.css")

from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

# ======================
# Load Models & Data
# ======================
# Load data
data = pd.read_csv('balanced_simulated_data.csv')
data.fillna(method='ffill', inplace=True)

# Load saved models
with open('reg_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

features = ['sensor_temp', 'sensor_vib', 'sensor_voltage', 'operational_hours']

# ======================
# Distance Threshold for Anomaly Detection
# ======================
# Calculate distances of all data points to their cluster centers
distances = cdist(data[features], kmeans_model.cluster_centers_)
distance_threshold = np.percentile(distances.min(axis=1), 90)  # 90th percentile

# ======================
# Prediction Function
# ======================
def predict_maintenance(features_input):
    input_scaled = scaler.transform([features_input])
    rul_pred = reg_model.predict(input_scaled)
    maint_pred = clf_model.predict(input_scaled)

    cluster_pred = kmeans_model.predict(input_scaled)
    distance = cdist(input_scaled, kmeans_model.cluster_centers_)[0, cluster_pred[0]]

    # Hybrid anomaly detection
    anomaly = 'Anomaly' if (rul_pred[0] < 200 or distance > distance_threshold) else 'Normal'

    return {
        'RUL Prediction': rul_pred[0],
        'Maintenance Prediction': 'Needs Maintenance' if maint_pred[0] == 1 else 'Normal',
        'Anomaly Detection': anomaly
    }


# ======================
# Streamlit UI
# ======================


with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Historical Data", "Input Data", "Results", "Visualizations"],
        icons=["house", "table", "input-cursor", "check2-circle", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
    )

# Home Page
if selected == "Home":
    st.title("Welcome to the Predictive Maintenance Dashboard")
    st.markdown("""
    This application provides predictive maintenance insights for lab sensors operating under various environmental conditions .  
    It uses machine learning models to analyze sensor readings and predict:
    - **Remaining Useful Life (RUL)** in hours,
    - **Maintenance requirements** (whether equipment needs maintenance),
    - **Anomaly detection** based on abnormal sensor behavior.

    Use the navigation menu to explore:
    - Historical sensor data and trends,
    - Predictive insights (RUL & maintenance status),
    - Data visualizations for better understanding of sensor behavior.
    """)


# Historical Data
elif selected == "Historical Data":
    st.title("📂 Historical Data")
    st.write(data)

# Input Data
elif selected == "Input Data":
    st.title("🔧 Input Features")
    st.markdown("Use sliders or random generation to provide input values.")

    if 'generated_values' not in st.session_state:
        st.session_state['generated_values'] = None

    # Generate random values within fixed ranges
    if st.button('Generate Random Values'):
        temp = np.random.uniform(0, 100)          # Sensor Temp: 0–100 °C
        vib = np.random.uniform(0, 2)             # Sensor Vib: 0–2 g
        volt = np.random.uniform(210, 240)        # Sensor Voltage: 210–240 V
        hours = np.random.uniform(100, 3000)      # Operational Hours: 100–3000
        st.session_state['generated_values'] = [temp, vib, volt, hours]
        st.success("Random values generated successfully!")

    if st.session_state['generated_values'] is not None:
        st.write("**Generated Values:**")
        st.write(f"Sensor Temp: {st.session_state['generated_values'][0]:.2f}")
        st.write(f"Sensor Vib: {st.session_state['generated_values'][1]:.2f}")
        st.write(f"Sensor Voltage: {st.session_state['generated_values'][2]:.2f}")
        st.write(f"Operational Hours: {st.session_state['generated_values'][3]:.2f}")
        if st.button('Use Generated Values'):
            st.session_state['input_features'] = st.session_state['generated_values']
            st.success("Values saved. Go to Results page.")

    st.markdown("**Or manually input values:**")
    sensor_temp = st.slider('Sensor Temp (°C)', 0, 100, 40)          # Slider 0–100
    sensor_vib = st.slider('Sensor Vib (g)', 0.0, 2.0, 0.5)          # Slider 0–2
    sensor_voltage = st.slider('Sensor Voltage (V)', 210, 240, 220)  # Slider 210–240
    operational_hours = st.slider('Operational Hours', 100, 3000, 500) # Slider 100–3000

    if st.button('Submit'):
        st.session_state['input_features'] = [sensor_temp, sensor_vib, sensor_voltage, operational_hours]
        st.success("Manual input submitted! Go to Results page.")

# Results Page
elif selected == "Results":
    st.title("📊 Prediction Results")
    if 'input_features' not in st.session_state:
        st.warning("Please provide input data in the 'Input Data' section.")
    else:
        input_features = st.session_state['input_features']
        prediction = predict_maintenance(input_features)
        st.write(f"**Remaining Useful Life (RUL):** {prediction['RUL Prediction']:.2f} hours")
        st.write(f"**Maintenance Status:** {prediction['Maintenance Prediction']}")
        st.write(f"**Anomaly Detection:** {prediction['Anomaly Detection']}")
        if prediction['Maintenance Prediction'] == 'Needs Maintenance':
            st.error('⚠️ Maintenance required!')
        if prediction['Anomaly Detection'] == 'Anomaly':
            st.warning('⚠️ Anomaly detected!')

# Visualizations
elif selected == "Visualizations":
    st.title("📊 Data Visualizations")

    # Histograms
    st.subheader("Histogram of Sensor Readings")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data['sensor_temp'], bins=30, ax=axs[0], kde=True)
    axs[0].set_title('Sensor Temp')
    sns.histplot(data['sensor_vib'], bins=30, ax=axs[1], kde=True)
    axs[1].set_title('Sensor Vib')
    sns.histplot(data['sensor_voltage'], bins=30, ax=axs[2], kde=True)
    axs[2].set_title('Sensor Voltage')
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("Scatter Plot of Sensors vs Operational Hours")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(data['operational_hours'], data['sensor_temp'], alpha=0.5)
    axs[0].set_title('Operational Hours vs Temp')
    axs[1].scatter(data['operational_hours'], data['sensor_vib'], alpha=0.5)
    axs[1].set_title('Operational Hours vs Vib')
    axs[2].scatter(data['operational_hours'], data['sensor_voltage'], alpha=0.5)
    axs[2].set_title('Operational Hours vs Voltage')
    st.pyplot(fig)

    # Boxplots for Sensor Distributions by Maintenance Status
    st.subheader("Boxplot of Sensors by Maintenance Status")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(data=data, x='maintenance', y='sensor_temp', ax=axs[0])
    axs[0].set_title('Sensor Temp by Maintenance')
    sns.boxplot(data=data, x='maintenance', y='sensor_vib', ax=axs[1])
    axs[1].set_title('Sensor Vib by Maintenance')
    sns.boxplot(data=data, x='maintenance', y='sensor_voltage', ax=axs[2])
    axs[2].set_title('Sensor Voltage by Maintenance')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Feature Importance for RUL Prediction
    st.subheader("Feature Importance for RUL Prediction")
    importances = reg_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', ax=ax)
    ax.set_title('Feature Importance (RUL)')
    st.pyplot(fig)






