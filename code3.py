import json
import os
import hashlib
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.set_page_config(
    page_title="No Code Platform For Machine Learning",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Background Image (replace with your JPG link) ----------
bg_image_url = "https://raw.githubusercontent.com/ZhiXuan-cod/dsp/blob/main/4882066.jpg"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- User Management ----------
class UserManager:
    def __init__(self):
        self.users_file = "users.json"
        self.load_users()
    
    def load_users(self):
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            else:
                self.users = {}
        except:
            self.users = {}
    
    def save_users(self):
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            return True
        except:
            return False
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username, password, user_info, role="user"):
        if username in self.users:
            return False, "Username already exists"
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        self.users[username] = {
            'password_hash': self.hash_password(password),
            'info': user_info,
            'role': role,
            'history': [],
            'created_at': datetime.now().isoformat(),
            'last_login': datetime.now().isoformat()
        }
        if self.save_users():
            return True, "Sign-up successful!"
        return False, "Error saving user data"
    
    def login(self, username, password):
        if username not in self.users:
            return False, "User does not exist"
        if self.users[username]['password_hash'] == self.hash_password(password):
            self.users[username]['last_login'] = datetime.now().isoformat()
            self.save_users()
            return True, "Login successful!"
        return False, "Incorrect password."
    
    def get_user_info(self, username):
        return self.users.get(username, {}).get('info', {})
    
    def get_user_role(self, username):
        return self.users.get(username, {}).get('role', 'user')
    
    def add_history(self, username, action):
        if username in self.users:
            self.users[username]['history'].append({
                'action': action,
                'timestamp': datetime.now().isoformat()
            })
            self.save_users()
            return True
        return False

user_manager = UserManager()

# TPOT availability
try:
    from tpot import TPOTClassifier, TPOTRegressor
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; padding: 1rem; margin-bottom: 2rem; }
    .sub-header { font-size: 1.5rem; color: #3949AB; margin-top: 1.5rem; margin-bottom: 1rem; }
    .card { background-color: #f5f5f5; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; border-left: 5px solid #1E88E5; }
    .success-box { background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .warning-box { background-color: #FFF3E0; border-left: 5px solid #FF9800; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .metric-card { background-color: white; border-radius: 10px; padding: 1rem; margin: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'eda_report' not in st.session_state:
    st.session_state.eda_report = None
if 'page' not in st.session_state:
    st.session_state.page = "📊 Data Upload"
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None

# Login / Signup
if not st.session_state.logged_in:
    st.markdown('<h1 class="main-header">🤖 No-Code Machine Learning Platform</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if username and password:
                        success, msg = user_manager.login(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            user_manager.add_history(username, "Logged in")
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill in both fields")
        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("Choose a username")
                new_pass = st.text_input("Choose a password", type="password")
                confirm_pass = st.text_input("Confirm password", type="password")
                full_name = st.text_input("Full name (optional)")
                email = st.text_input("Email (optional)")
                submitted = st.form_submit_button("Sign Up", use_container_width=True)
                if submitted:
                    if not new_user or not new_pass:
                        st.warning("Username and password are required")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match")
                    else:
                        user_info = {"full_name": full_name, "email": email}
                        success, msg = user_manager.register(new_user, new_pass, user_info)
                        if success:
                            st.success(msg + " You can now log in.")
                        else:
                            st.error(msg)
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
    st.markdown(f"**Welcome, {st.session_state.username}!**")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        for key in ['data', 'target_column', 'problem_type', 'model', 'predictions', 
                    'test_data', 'training_complete', 'eda_report', 'page', 'X_scaled']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Select a step:",
        ["📊 Data Upload", "🔍 Exploratory Analysis", "🤖 Model Training", 
         "📈 Model Evaluation", "🔮 Make Predictions", "💾 Export Results",
         "👥 User List"],
        key="page"
    )
    
    st.markdown("---")
    st.markdown("### Platform Info")
    st.info("""
    This platform enables:
    - CSV data upload
    - Automated EDA
    - AutoML with TPOT (Classification/Regression)
    - Clustering (K‑Means, Agglomerative, DBSCAN)
    - Model evaluation
    - No-code predictions
    """)
    
    if not TPOT_AVAILABLE:
        st.error("⚠️ TPOT not installed. Install with: `pip install tpot`")
        st.code("pip install tpot", language="bash")

# Page 1: Data Upload
if page == "📊 Data Upload":
    st.markdown('<h2 class="sub-header">📊 Upload Your Dataset</h2>', unsafe_allow_html=True)
    # ... (unchanged, see full code above) ...

# Page 2: Exploratory Analysis
elif page == "🔍 Exploratory Analysis":
    # ... (unchanged) ...

# Page 3: Model Training
elif page == "🤖 Model Training":
    # ... (unchanged) ...

# Page 4: Model Evaluation
elif page == "📈 Model Evaluation":
    # ... (unchanged) ...

# Page 5: Make Predictions
elif page == "🔮 Make Predictions":
    # ... (unchanged) ...

# Page 6: Export Results
elif page == "💾 Export Results":
    # ... (unchanged) ...

# NEW PAGE: User List
elif page == "👥 User List":
    st.markdown('<h2 class="sub-header">👥 Registered Users</h2>', unsafe_allow_html=True)
    
    users_data = user_manager.users
    if not users_data:
        st.info("No users have registered yet.")
    else:
        user_list = []
        for username, details in users_data.items():
            user_list.append({
                "Username": username,
                "Role": details.get("role", "user"),
                "Full Name": details.get("info", {}).get("full_name", ""),
                "Email": details.get("info", {}).get("email", ""),
                "Created At": details.get("created_at", ""),
                "Last Login": details.get("last_login", "")
            })
        df_users = pd.DataFrame(user_list)
        st.dataframe(df_users, use_container_width=True)
        
        st.markdown("### 📥 Export User List as JSON")
        json_str = json.dumps(users_data, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="registered_users.json",
            mime="application/json",
            use_container_width=True
        )
        
        with st.expander("Show raw JSON"):
            st.json(users_data)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>🤖 No-Code Machine Learning Platform | Developed with Streamlit & TPOT</p>
<p>📍 Universiti Malaysia Pahang Al-Sultan Abdullah | Data Science Project I</p>
</div>
""", unsafe_allow_html=True)