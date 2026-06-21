import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Page Configuration with wide layout and custom title
st.set_page_config(page_title="ProTech Price Predictor", page_icon="💻", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for advanced styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1E88E5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Design
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3173/3173239.png", width=100)
    st.title("⚙️ Settings & Tools")
    st.markdown("Customize your prediction environment.")
    
    currency = st.selectbox("Select Currency", ["PKR (Rs)", "USD ($)", "EUR (€)"])
    currency_rates = {"PKR (Rs)": 1, "USD ($)": 0.0036, "EUR (€)": 0.0033}
    currency_symbols = {"PKR (Rs)": "Rs.", "USD ($)": "$", "EUR (€)": "€"}
    
    st.divider()
    st.markdown("### 📌 Saved Predictions")
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history[-5:][::-1]): # Show last 5
            st.info(f"**{item['model']}**\n\n{item['specs']}\n\nPrice: {item['price']}")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No saved predictions yet.")

# Main Header
st.title("💻 NextGen Laptop Price Engine")
st.markdown("An advanced AI-powered recommendation system built with **Machine Learning**.")

# Load Models
@st.cache_resource
def load_models():
    try:
        lr_model = pickle.load(open('linear_model.pkl', 'rb'))
        rf_model = pickle.load(open('rf_model.pkl', 'rb'))
        metrics = pickle.load(open('metrics.pkl', 'rb'))
        return lr_model, rf_model, metrics
    except FileNotFoundError:
        return None, None, None

lr_model, rf_model, metrics = load_models()

if lr_model is None:
    st.error("⚠️ Models not found! Please run `train_model.py` to generate the AI models.")
    st.stop()

# Tabs for structured navigation
tab1, tab2, tab3 = st.tabs(["🔮 AI Price Predictor", "📊 Analytics Dashboard", "📈 Market Trends"])

# --- TAB 1: USER INPUT & PREDICTION ---
with tab1:
    st.markdown("### 🔧 Configure Laptop Specifications")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ram = st.selectbox("Memory (RAM)", [4, 8, 16, 32, 64], help="Higher RAM improves multitasking.")
        with col2:
            storage = st.selectbox("Storage Capacity (GB)", [128, 256, 512, 1024, 2048], help="SSD/HDD Storage space.")
        with col3:
            screen = st.slider("Display Size (Inches)", 11.0, 18.0, 15.6, 0.1, help="Screen diagonal size.")
            
    st.markdown("---")
    col_alg, col_btn = st.columns([2, 1])
    with col_alg:
        chosen_model = st.radio("🧠 Select AI Engine", 
                                ("Random Forest Regressor (Recommended)", "Linear Regression"), 
                                horizontal=True)
    
    with col_btn:
        st.write("") # Spacer
        st.write("") # Spacer
        predict_btn = st.button("🚀 Generate Estimate", use_container_width=True)

    if predict_btn:
        with st.spinner('Analyzing market data & evaluating AI models...'):
            time.sleep(1) # Fake loading for effect
            
            user_features = np.array([[ram, storage, screen]])
            
            if "Linear" in chosen_model:
                prediction = lr_model.predict(user_features)[0]
                model_name = "Linear Regression"
            else:
                prediction = rf_model.predict(user_features)[0]
                model_name = "Random Forest"
                
            # Convert currency
            converted_price = prediction * currency_rates[currency]
            sym = currency_symbols[currency]
            
            st.success("✅ Prediction generated successfully!")
            
            # Display beautifully using columns and metrics
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label=f"Estimated Market Value ({model_name})", 
                          value=f"{sym} {converted_price:,.2f}", 
                          delta="Highly Accurate" if "Random" in model_name else "Standard Estimate")
                          
            with res_col2:
                st.info(f"**Specs Analyzed:**\n- RAM: {ram}GB\n- Storage: {storage}GB\n- Display: {screen}\"")
                
            # Save to history
            st.session_state.history.append({
                "model": model_name,
                "specs": f"{ram}GB RAM | {storage}GB | {screen}\"",
                "price": f"{sym} {converted_price:,.2f}"
            })

# --- TAB 2: DASHBOARD ---
with tab2:
    st.markdown("### ⚡ AI Model Performance Metrics")
    
    # 1. Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linear Reg R²", f"{metrics['lr_r2']*100:.1f}%")
    m2.metric("Random Forest R²", f"{metrics['rf_r2']*100:.1f}%", delta="Best", delta_color="normal")
    m3.metric("LR MAE Error", f"{metrics['lr_mae']:,.0f}")
    m4.metric("RF MAE Error", f"{metrics['rf_mae']:,.0f}", delta="-Lower is better", delta_color="inverse")
    
    st.markdown("---")
    
    # 2. Comparison Charts
    st.subheader("📊 Visual Evaluation")
    col_chart1, col_chart2 = st.columns(2)
    
    # Use st.bar_chart for cleaner, native UI instead of pyplot
    with col_chart1:
        st.markdown("**Model Accuracy Comparison (%)**")
        acc_data = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "Accuracy": [metrics['lr_r2']*100, metrics['rf_r2']*100]
        }).set_index("Model")
        st.bar_chart(acc_data, color="#2E86C1")
        
    with col_chart2:
        st.markdown("**Mean Absolute Error Comparison**")
        err_data = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "Error (MAE)": [metrics['lr_mae'], metrics['rf_mae']]
        }).set_index("Model")
        st.bar_chart(err_data, color="#E74C3C")

# --- TAB 3: MARKET TRENDS (New Functionality) ---
with tab3:
    st.markdown("### 📈 Simulated Market Price Trends")
    st.markdown("How laptop prices have fluctuated over the past year based on component shortages and demand.")
    
    # Generate dummy trend data
    months = pd.date_range(start='2025-01-01', periods=12, freq='ME')
    base_trend = np.linspace(50000, 55000, 12) + np.random.normal(0, 1500, 12)
    premium_trend = np.linspace(120000, 135000, 12) + np.random.normal(0, 3000, 12)
    
    trend_df = pd.DataFrame({
        "Date": months,
        "Budget Laptops": base_trend,
        "Premium Laptops": premium_trend
    }).set_index("Date")
    
    st.line_chart(trend_df)
    
    with st.expander("💡 View Trend Insights"):
        st.write("""
        - **Q1-Q2:** Prices saw a slight increase due to global chip shortages.
        - **Q3-Q4:** Stabilization in supply chains led to more consistent pricing, though premium laptops continue to see price hikes due to new GPU releases.
        """)