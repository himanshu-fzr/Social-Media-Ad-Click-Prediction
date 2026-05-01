# app.py - Social Media Ad Click Predictor
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time

# Page config
st.set_page_config(page_title="Ad Click Predictor", page_icon="📱", layout="wide")

# Custom CSS for colors (matching your image's style)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-size: 1.2rem;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .big-prob {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 20px;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    h1 {
        color: #1e3c72;
        text-align: center;
    }
    hr {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
with open('ad_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('ad_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load original data for test accuracy
df = pd.read_csv('ad_click_data.csv')
X = df.drop('Click', axis=1)
y = df['Click']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)

# Title
st.markdown("<h1>📱 Social Media Ad Click Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Advanced Logistic Regression Model with Interpretable Insights</p>", unsafe_allow_html=True)
st.markdown("---")

# Two columns
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown("### 👤 User Profile & Ad Settings")
    age = st.slider("**Age**", 18, 70, 30)
    gender = st.radio("**Gender**", ["Male", "Female"], horizontal=True)
    gender_val = 1 if gender == "Male" else 0
    income = st.slider("**Annual Income (k USD)**", 20, 200, 50)
    time_spent = st.slider("**Time spent on platform (min/day)**", 0, 300, 60)
    prev_clicks = st.slider("**Previous ad clicks (30 days)**", 0, 50, 5)
    device = st.radio("**Device type**", ["Mobile", "Desktop"], horizontal=True)
    device_val = 1 if device == "Mobile" else 0
    ad_topic = st.selectbox("**Ad topic**", ["Technology", "Fashion", "Sports", "Food"])
    topic_map = {"Technology":0, "Fashion":1, "Sports":2, "Food":3}
    topic_val = topic_map[ad_topic]
    spending_score = st.slider("**Spending Score (0-100)**", 0, 100, 50)
    engagement_rate = st.slider("**Engagement Rate (0-100)**", 0, 100, 50)
    ad_frequency = st.slider("**Ad Frequency (ads seen per day)**", 1, 30, 10)
    
    # Predict button
    predict_click = st.button("🔮 PREDICT CLICK LIKELIHOOD", use_container_width=True)

with right:
    st.markdown("### 📊 Model Performance")
    st.metric("Test Accuracy", f"{test_acc:.2%}", delta=None)
    st.metric("Training Samples", "30,000", delta=None)
    
    # Show coefficients (feature importance)
    feature_names = X.columns.tolist()
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df = coef_df.sort_values("Coefficient", ascending=True)
    
    st.markdown("### 🔥 Top Feature Contributions")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#4CAF50' if c > 0 else '#FF6B6B' for c in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Coefficient (impact on click)")
    ax.set_title("Logistic Regression Coefficients")
    st.pyplot(fig)

# Prediction result area
if predict_click:
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender_val,
        'Income_kUSD': income,
        'TimeSpent_min': time_spent,
        'PrevClicks': prev_clicks,
        'DeviceType': device_val,
        'AdTopic': topic_val,
        'SpendingScore': spending_score,
        'EngagementRate': engagement_rate,
        'AdFrequency': ad_frequency
    }])
    
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]  # probability of click
    prediction = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")
    
    # Big probability display
    if prob >= 0.5:
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #00b09b, #96c93d); padding: 1.5rem; border-radius: 20px; text-align: center;'>
            <h2 style='color:white;'>✅ HIGH LIKELIHOOD OF CLICK</h2>
            <p style='font-size:3rem; color:white; font-weight:bold;'>{prob*100:.1f}%</p>
            <p style='color:white;'>This user is likely to click the ad.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #ff6b6b, #f0c27b); padding: 1.5rem; border-radius: 20px; text-align: center;'>
            <h2 style='color:white;'>⚠️ LOW LIKELIHOOD OF CLICK</h2>
            <p style='font-size:3rem; color:white; font-weight:bold;'>{prob*100:.1f}%</p>
            <p style='color:white;'>Try different targeting (device, topic, or frequency).</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show probability chart
    fig2, ax2 = plt.subplots()
    ax2.barh(["Click Probability"], [prob*100], color='#4CAF50')
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Probability (%)")
    ax2.set_title("Click Probability")
    for i, v in enumerate([prob*100]):
        ax2.text(v + 2, i, f"{v:.1f}%", va='center')
    st.pyplot(fig2)

    # Download button for prediction report
    report_df = input_data.copy()
    report_df['Predicted_Click_Probability'] = prob
    report_df['Predicted_Label'] = ['Click' if prediction==1 else 'No Click']
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=csv,
        file_name=f"ad_click_prediction_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# Additional: Confusion matrix expander
with st.expander("📊 Model Confusion Matrix (on test set)"):
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")
    st.pyplot(fig3)

st.markdown("---")
st.caption("💡 Model trained on 30,000 synthetic user profiles. Features: Age, Gender, Income, Time spent, Previous clicks, Device type, Ad topic, Spending Score, Engagement Rate, Ad Frequency.")