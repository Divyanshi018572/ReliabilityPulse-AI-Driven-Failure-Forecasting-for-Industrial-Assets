import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import path_utils

# Page Configuration
st.set_page_config(
    page_title="ReliabilityPulse | AI-Driven Maintenance", 
    layout="wide", 
    page_icon="⚡", 
    initial_sidebar_state="expanded"
)

# Premium Custom CSS (Glassmorphism + Neon Industrial Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #0B0E14;
        color: #E6E6E6;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(22, 27, 34, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Premium Metric Card */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: #00D4FF;
        transform: translateY(-5px);
    }

    /* Custom Progress Bar Color */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00D4FF, #B026FF);
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #161B22;
        border-radius: 8px 8px 0px 0px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1F242D;
        color: #00D4FF;
    }

    .stTabs [aria-selected="true"] {
        background-color: #21262D !important;
        border-bottom: 2px solid #00D4FF !important;
    }
    
    .stImage > img {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper to load models
@st.cache_resource
def load_all_resources():
    try:
        models = {
            "XGBoost": joblib.load(path_utils.get_model_path('xgboost_model.pkl')),
            "Random Forest": joblib.load(path_utils.get_model_path('random_forest.pkl')),
            "Decision Tree": joblib.load(path_utils.get_model_path('decision_tree.pkl'))
        }
        scaler = joblib.load(path_utils.get_model_path('scaler.pkl'))
        return models, scaler
    except Exception as e:
        return None, None

all_models, scaler = load_all_resources()
models_ready = all_models is not None

# SIDEBAR: Control Center
with st.sidebar:
    st.image(path_utils.get_output_path('failure_distribution.png'), use_container_width=True)
    st.title("🛡️ Control Center")
    st.caption("Industrial Reliability Forecasting")
    
    st.markdown("---")
    selected_model_name = st.selectbox("AI Model", ["XGBoost", "Random Forest", "Decision Tree"])
    st.markdown("---")
    st.header("Sensor Telemetry")
    
    with st.expander("🌡️ Temperature", expanded=True):
        input_air_temp = st.slider("Ambient [K]", 295.0, 305.0, 300.0, 0.1)
        input_proc_temp = st.slider("Process [K]", 305.0, 315.0, 310.0, 0.1)
    
    with st.expander("⚙️ Mechanical", expanded=True):
        input_rpm = st.number_input("Speed [rpm]", 1200, 2800, 1500)
        input_torque = st.number_input("Torque [Nm]", 3.0, 76.0, 40.0, 0.1)
    
    with st.expander("🛠️ Tool Wear", expanded=True):
        input_type = st.selectbox("Market Grade", ["L", "M", "H"])
        input_tool_wear = st.slider("Duration [min]", 0, 250, 100)

# MAIN: ReliabilityPulse
st.title("⚡ ReliabilityPulse")

# Fragment implementation for stable diagnostics
@st.fragment
def run_stable_diagnostics():
    if not models_ready:
        st.warning("Core engine is offline. Please ensure models/ directory is populated.")
        return

    # Internal Logic
    type_map = {"L": 0, "M": 1, "H": 2}
    type_val = type_map[input_type]
    temp_diff = input_proc_temp - input_air_temp
    power = input_torque * (input_rpm * 2 * np.pi / 60)
    tool_wear_torque = input_tool_wear * input_torque

    input_df = pd.DataFrame([[
        type_val, input_air_temp, input_proc_temp, input_rpm, input_torque, input_tool_wear, 
        temp_diff, power, tool_wear_torque
    ]], columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                'temp_diff', 'power', 'tool_wear_torque'])

    input_scaled = scaler.transform(input_df)
    model = all_models[selected_model_name]
    prob = model.predict_proba(input_scaled)[0, 1]

    if prob < 0.2: status, color, s_icon = "OPTIMAL", "#00FF41", "✅"
    elif prob < 0.5: status, color, s_icon = "MONITOR", "#FFB300", "⚠️"
    elif prob < 0.8: status, color, s_icon = "WARNING", "#FF3D00", "🚨"
    else: status, color, s_icon = "CRITICAL", "#D50000", "🛑"

    # Stability Columns (3 metrics)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Failure Probability", f"{prob*100:.1f}%")
    with c2: st.metric("System Health", status)
    with c3: st.metric("Tool Stress", f"{tool_wear_torque:.0f}")

    st.write("---")
    st.markdown(f"#### Asset Health: {s_icon} **{status}**")
    st.progress(min(int(prob * 100), 100))
    
    st.info(f"**Diagnostic Metric**: `Power: {power:.1f}W` | `Delta: {temp_diff:.1f}K` | `Stress: {tool_wear_torque:.1f}`")

# Fragment implementation for stable visuals
@st.fragment
def render_visual_engine():
    st.header("📈 Visual Data Engine")
    st.info("Directly analyzing 10 key diagnostic signatures from the AI4I 2020 dataset.")
    
    # 1. Row: Population Distributions
    st.write("### 📊 Population Diagnostics")
    v_c1, v_c2 = st.columns(2)
    with v_c1:
        st.image(path_utils.get_output_path('failure_distribution.png'), use_container_width=True)
        st.caption("Target Class Distribution (SMOTE Balanced)")
    with v_c2:
        st.image(path_utils.get_output_path('failure_rate_by_type.png'), use_container_width=True)
        st.caption("Failure Propensity by Market Grade (L/M/H)")

    st.write("---")

    # 2. Row: Sensor Envelopes
    st.write("### 🌡️ Sensor Envelopes")
    v_c3, v_c4 = st.columns(2)
    with v_c3:
        st.image(path_utils.get_output_path('numeric_distributions.png'), use_container_width=True)
        st.caption("Global Sensor Distributions")
    with v_c4:
        st.image(path_utils.get_output_path('sensor_boxplots.png'), use_container_width=True)
        st.caption("Outlier Detection & Variance Matrix")

    st.write("---")

    # 3. Row: Decision Intelligence
    st.write("### 🔍 Decision Intelligence")
    v_c5, v_c6 = st.columns(2)
    with v_c5:
        st.image(path_utils.get_output_path('feature_importance.png'), use_container_width=True)
        st.caption("AI Feature Ranking (Top Predictors)")
    with v_c6:
        st.image(path_utils.get_output_path('correlation_heatmap.png'), use_container_width=True)
        st.caption("Feature Multicollinearity Matrix")

    st.write("---")

    # 4. Row: Failure Modes
    st.write("### ⚙️ Failure Mechanism Analysis")
    v_c7, v_c8 = st.columns(2)
    with v_c7:
        st.image(path_utils.get_output_path('sub_label_counts.png'), use_container_width=True)
        st.caption("Failure Mode Signatures (HDF/PWF/OSF)")
    with v_c8:
        st.image(path_utils.get_output_path('anomaly_scores.png'), use_container_width=True)
        st.caption("Isolation Forest Anomaly Scores")

    st.write("---")

    # 5. Row: Performance Benchmarks
    st.write("### 🎯 Model Performance Benchmarks")
    v_c9, v_c10 = st.columns(2)
    with v_c9:
        st.image(path_utils.get_output_path('roc_curve_comparison.png'), use_container_width=True)
        st.caption("Multi-Model ROC Comparison")
    with v_c10:
        st.image(path_utils.get_output_path('confusion_matrix_xgboost.png'), use_container_width=True)
        st.caption("XGBoost Confusion Landscape")

# Tabs
tab_predict, tab_viz, tab_docs = st.tabs(["🚀 Real-time Diagnostics", "📈 Visual Data Engine", "📁 Asset Specs"])

with tab_predict:
    render_results = render_results = run_stable_diagnostics()

with tab_viz:
    render_visual_engine()

with tab_docs:
    st.markdown("""
    ### 🏭 ReliabilityPulse Architecture
    **ReliabilityPulse** is an industrial AI suite designed for zero-downtime manufacturing. 
    It leverages Gradient Boosted Trees and physics-informed feature engineering to forecast failures within a 95% AUC precision.
    """)

# Footer
st.markdown("---")
st.write(f"Built by **Divyanshi Singh** | [LinkedIn](https://www.linkedin.com/in/divyanshi-singh-/) | [GitHub](https://github.com/Divyanshi018572)")
