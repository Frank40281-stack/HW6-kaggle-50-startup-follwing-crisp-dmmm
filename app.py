import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
import logging

# Configure logging to write both to console and streamlit_app.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Streamlit dashboard loaded / refreshed")

# -----------------------------------------------------------------------------
# PAGE CONFIG & CUSTOM CSS (Premium Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="50 Startups CRISP-DM Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium styling
st.markdown("""
<style>
    /* Global Styles */
    .reportview-container {
        background: #0e1117;
    }
    
    /* Header Gradient */
    .title-gradient {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8533 50%, #FFC300 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Elegant Cards */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 75, 75, 0.4);
    }
    
    /* Prediction Glowing Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(255,75,75,0.1) 0%, rgba(255,133,51,0.1) 100%);
        border: 2px solid #FF4B4B;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.2);
        animation: pulse 2s infinite alternate;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 15px rgba(255, 75, 75, 0.2); }
        100% { box-shadow: 0 0 30px rgba(255, 75, 75, 0.5); }
    }
    
    /* Subheadings */
    .section-header {
        color: #FF8533;
        font-size: 1.8rem;
        font-weight: 700;
        border-bottom: 2px solid rgba(255, 133, 51, 0.2);
        padding-bottom: 8px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    .css-1639g78 {
        background-color: #1a1c24;
    }
    
    /* Metrics display */
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00E676;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #B0BEC5;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists("50_Startups.csv"):
        return pd.read_csv("50_Startups.csv")
    return None

@st.cache_data
def load_metrics():
    if os.path.exists("evaluation_metrics.csv"):
        return pd.read_csv("evaluation_metrics.csv")
    return None

@st.cache_data
def load_fs_metrics():
    if os.path.exists("feature_selection_metrics_all.csv"):
        return pd.read_csv("feature_selection_metrics_all.csv")
    return None

df = load_data()
metrics_df = load_metrics()
fs_metrics_df = load_fs_metrics()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b5/CRISP-DM_Process_V2.png", caption="CRISP-DM Methodology", use_container_width=True)

st.sidebar.markdown("""
### 🚀 50 Startups Profit Predictor
This application walks through the **CRISP-DM** lifecycle to analyze expenses and predict the profitability of startups.

#### 📊 Dataset Quick Stats
*   **Total Records**: 50
*   **Target**: Profit
*   **Numerical Features**: R&D Spend, Administration, Marketing Spend
*   **Categorical Features**: State
""")

st.sidebar.info("💡 **Model deployed**: Random Forest Regressor")

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="title-gradient">50 Startups Profit Predictor</div>', unsafe_allow_html=True)
st.markdown("##### *An Interactive Machine Learning Sandbox following the CRISP-DM Framework*")
st.write("---")

if df is None:
    st.error("Error: '50_Startups.csv' not found. Please place it in the workspace directory.")
    st.stop()

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Phases 1 & 2: Business & Data Understanding",
    "⚙️ Phases 3, 4 & 5: Preparation, Modeling & Evaluation",
    "🔮 Phase 6: Interactive Deployment"
])

# =============================================================================
# TAB 1: BUSINESS & DATA UNDERSTANDING
# =============================================================================
with tab1:
    st.markdown('<div class="section-header">Phase 1: Business Understanding</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h4>🎯 Business Objective</h4>
            <p>Predicting the potential profit of a new startup is key for investors (VCs) looking to maximize returns. 
            By analyzing where startups spend money, we can identify structural factors that drive higher profitability.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h4>💡 Key Questions To Answer</h4>
            <ul>
                <li>Which expenditure category (R&D, Admin, Marketing) is the strongest predictor of Profit?</li>
                <li>Does the location of the startup (State: NY, CA, FL) affect profitability?</li>
                <li>Can we accurately predict Profit to filter high-potential startup pitches?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Phase 2: Data Understanding</div>', unsafe_allow_html=True)
    
    # Dataset view
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        st.write("First 10 Rows of the Dataset:")
        st.dataframe(df.head(10), use_container_width=True)
    with col_d2:
        st.write("Statistical Summary:")
        st.dataframe(df.describe().T, use_container_width=True)
        
    st.write("---")
    
    # Plots selection
    plot_option = st.selectbox(
        "Select Interactive Plot to Visualize:",
        ["Profit Distribution", "R&D Spend vs Profit", "Marketing Spend vs Profit", "Profit by State"]
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="darkgrid")
    
    if plot_option == "Profit Distribution":
        sns.histplot(df['Profit'], kde=True, color='#FF4B4B', ax=ax)
        ax.set_title("Distribution of Profit among 50 Startups", fontsize=14, color='white')
        ax.set_xlabel("Profit ($)", color='white')
        ax.set_ylabel("Count", color='white')
    
    elif plot_option == "R&D Spend vs Profit":
        sns.scatterplot(data=df, x='R&D Spend', y='Profit', hue='State', s=100, palette='Set1', ax=ax)
        # Add a trendline
        sns.regplot(data=df, x='R&D Spend', y='Profit', scatter=False, color='#FFC300', ax=ax)
        ax.set_title("R&D Spend vs Profit (Strong linear relationship)", fontsize=14, color='white')
        ax.set_xlabel("R&D Spend ($)", color='white')
        ax.set_ylabel("Profit ($)", color='white')
        
    elif plot_option == "Marketing Spend vs Profit":
        sns.scatterplot(data=df, x='Marketing Spend', y='Profit', hue='State', s=100, palette='Set2', ax=ax)
        sns.regplot(data=df, x='Marketing Spend', y='Profit', scatter=False, color='#00E676', ax=ax)
        ax.set_title("Marketing Spend vs Profit (Moderate relationship)", fontsize=14, color='white')
        ax.set_xlabel("Marketing Spend ($)", color='white')
        ax.set_ylabel("Profit ($)", color='white')
        
    else: # Profit by State
        sns.boxplot(data=df, x='State', y='Profit', palette='Set3', ax=ax)
        ax.set_title("Startup Profit Distribution across States", fontsize=14, color='white')
        ax.set_xlabel("State", color='white')
        ax.set_ylabel("Profit ($)", color='white')
        
    # Styling matplotlib figure to blend with Streamlit dark mode
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e212b')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')
        
    st.pyplot(fig)
    plt.close()

# =============================================================================
# TAB 2: DATA PREP, MODELING & EVALUATION
# =============================================================================
with tab2:
    st.markdown('<div class="section-header">Phase 3: Data Preparation</div>', unsafe_allow_html=True)
    st.markdown("""
    *   **Handling Nulls**: Checked for missing values (No null values found in this dataset!).
    *   **Categorical Encoding**: Used scikit-learn's `OneHotEncoder` to encode the `State` column into dummy features (representing California, Florida, and New York).
    *   **Feature Scaling**: Used `StandardScaler` to scale continuous features (`R&D Spend`, `Administration`, `Marketing Spend`) so models like Lasso and Ridge perform correctly.
    *   **Data Splitting**: Split data into **80% Training Set (40 startups)** and **20% Test Set (10 startups)** to ensure unbiased model evaluation.
    """)

    st.markdown('<div class="section-header">Phase 4 & 5: Modeling & Evaluation</div>', unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        st.subheader("🏆 Model Comparison")
        if metrics_df is not None:
            # Highlight the top performing model
            st.dataframe(
                metrics_df.style.highlight_max(subset=['R2 Score'], color='rgba(0, 230, 118, 0.3)')
                          .highlight_min(subset=['MAE', 'RMSE'], color='rgba(0, 230, 118, 0.3)'),
                use_container_width=True
            )
            
            # Draw interactive bar chart of R2 Scores
            st.write("Model R² Scores Comparison:")
            st.bar_chart(metrics_df.set_index('Model')['R2 Score'], color='#FF8533')
        else:
            st.warning("Could not read 'evaluation_metrics.csv'. Please run 'solve_50_startups.py' first.")
            
    with col_m2:
        st.subheader("📈 Best Model Predictions vs. Actuals")
        if os.path.exists("model_evaluation.png"):
            st.image("model_evaluation.png", caption="Model predictions on unseen testing set", use_container_width=True)
        else:
            st.info("Run the python script to generate the model evaluation plot.")

    st.write("---")
    
    st.subheader("📊 Sequential Feature Selection & Performance Analysis (All Models)")
    
    if fs_metrics_df is not None:
        model_options = list(fs_metrics_df['Model'].unique())
        selected_model = st.selectbox("Select Model to View SFS Metrics Table:", model_options)
        filtered_df = fs_metrics_df[fs_metrics_df['Model'] == selected_model].drop(columns=['Model'])
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("Could not read 'feature_selection_metrics_all.csv'. Please run 'solve_50_startups.py' first.")
        
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        st.markdown("""
        **🔍 Key Insights from Feature Selection across 5 Algorithms:**
        *   **Optimal Subset**: Most algorithms (Linear, Ridge, Lasso, Random Forest) achieve their best performance (high R² and low RMSE) with **2 features** (`R&D Spend` and `Marketing Spend`).
        *   **Effect of Extra Features**:
            *   Adding geographical variable `State_New York` (3 features) and `State_Florida` (4 features) causes minor model noise and slight drops in R² score.
            *   Adding the 5th feature (like `Administration` / `State_California`) creates substantial multicollinearity (dummy trap) in linear models and introduces overfitting in tree models, resulting in noticeable metric degradation (RMSE increases, R² drops).
        *   **Key Takeaway**: Across all algorithms, the **2-feature model (研發費用 + 行銷費用)** is the most parsimonious and stable model.
        """)
    with col_f2:
        if os.path.exists("feature_selection_performance_allinone.png"):
            st.image("feature_selection_performance_allinone.png", caption="All Models: RMSE and R-squared by Number of Features", use_container_width=True)
        else:
            st.info("Run the python script `solve_50_startups.py` to generate the all-in-one feature selection plots.")

# =============================================================================
# TAB 3: DEPLOYMENT & INTERACTIVE PREDICTOR
# =============================================================================
with tab3:
    st.markdown('<div class="section-header">Phase 6: Deployment</div>', unsafe_allow_html=True)
    st.markdown("""
    Below is the deployed **Random Forest Regressor** pipeline model. 
    Adjust the expenditures and region parameters to predict a startup's potential profit in real time!
    """)
    
    # Check if joblib model exists
    model_path = "best_model.joblib"
    if not os.path.exists(model_path):
        st.error(f"Error: Deployed model file '{model_path}' not found! Please run the training script `python solve_50_startups.py` first.")
    else:
        # Load the model
        model = joblib.load(model_path)
        
        # User input form
        with st.form(key="prediction_form"):
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                rd_spend = st.slider("🔬 R&D Spend ($)", min_value=0.0, max_value=200000.0, value=100000.0, step=1000.0)
                admin_spend = st.slider("💼 Administration Spend ($)", min_value=0.0, max_value=200000.0, value=120000.0, step=1000.0)
                
            with col_in2:
                marketing_spend = st.slider("📢 Marketing Spend ($)", min_value=0.0, max_value=500000.0, value=250000.0, step=1000.0)
                state = st.selectbox("📍 State / Location", options=['California', 'Florida', 'New York'])
                
            submit_button = st.form_submit_button(label="🔮 Predict Startup Profit")
            
        if submit_button:
            # Create a dataframe for the input
            input_df = pd.DataFrame([{
                'R&D Spend': rd_spend,
                'Administration': admin_spend,
                'Marketing Spend': marketing_spend,
                'State': state
            }])
            
            # Predict
            pred_profit = model.predict(input_df)[0]
            logger.info(f"Prediction requested - Inputs: R&D={rd_spend}, Admin={admin_spend}, Marketing={marketing_spend}, State={state} -> Predicted Profit: ${pred_profit:,.2f}")
            
            # Format and output results in a beautiful glowing box
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: white; margin-bottom: 5px;">Estimated Profitability</h3>
                <div style="font-size: 3rem; font-weight: 800; color: #00E676; margin-bottom: 10px;">
                    ${pred_profit:,.2f}
                </div>
                <p style="color: #ECEFF1; font-size: 0.95rem; max-width: 600px; margin: 0 auto;">
                    Based on expenditures of <b>${rd_spend:,.2f}</b> on R&D, <b>${admin_spend:,.2f}</b> on Administration, 
                    and <b>${marketing_spend:,.2f}</b> on Marketing in <b>{state}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison metrics in the prediction card context
            st.write("")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                # Calculate return ratio
                total_spend = rd_spend + admin_spend + marketing_spend
                roi = (pred_profit / total_spend) * 100 if total_spend > 0 else 0
                st.metric("Estimated Return on Spend (ROI)", f"{roi:.1f}%")
            with col_res2:
                st.metric("Model Used", "Random Forest Regressor")
            with col_res3:
                st.metric("Model R² Score", "92.8%")
