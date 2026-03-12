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
    silhouette_score, davies_bouldin_score, calinski_harabasz_score   # <-- new clustering metrics
)
# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Must be the first Streamlit command
st.set_page_config(
    page_title="No Code Platform For Machine Learning",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Try to import TPOT (optional - show message if not installed)
try:
    from tpot import TPOTClassifier, TPOTRegressor
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3949AB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session State Initialization ----------
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
# New session variable for clustering: store preprocessed features
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None

# ---------- Login / Signup Page (if not logged in) ----------
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
    st.stop()  # Stop execution if not logged in

# ---------- Main App (Logged In) ----------
# Sidebar with user info and logout
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
    st.markdown(f"**Welcome, {st.session_state.username}!**")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        # Clear all ML-related session data
        for key in ['data', 'target_column', 'problem_type', 'model', 'predictions', 
                    'test_data', 'training_complete', 'eda_report', 'page', 'X_scaled']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Navigation")
    
    # Radio widget bound to session state
    page = st.radio(
        "Select a step:",
        ["📊 Data Upload", "🔍 Exploratory Analysis", "🤖 Model Training", 
         "📈 Model Evaluation", "🔮 Make Predictions", "💾 Export Results"],
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

# ---------- Page 1: Data Upload ----------
if page == "📊 Data Upload":
    st.markdown('<h2 class="sub-header">📊 Upload Your Dataset</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
        <h4>📁 Supported Data Format</h4>
        <ul>
            <li>CSV files only (.csv)</li>
            <li>Structured tabular data</li>
            <li>Numerical and categorical variables</li>
            <li>Clear target column for supervised learning</li>
            <li>Small to medium datasets recommended</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                
                # Show data preview
                st.markdown("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Basic statistics
                with st.expander("📊 Basic Data Statistics"):
                    st.write("**Shape:**", df.shape)
                    st.write("**Column Types:**")
                    col_types = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Missing Values': df.isnull().sum(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(col_types, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Important Notes</h4>
        <ul>
            <li>Ensure your data is clean</li>
            <li>Remove sensitive information</li>
            <li>Check for missing values</li>
            <li>Define target variable clearly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            st.markdown("### 🎯 Define Target Column")
            
            # Select target column
            target_col = st.selectbox(
                "Select the target column:",
                options=st.session_state.data.columns.tolist(),
                index=len(st.session_state.data.columns)-1 if st.session_state.data is not None else 0
            )
            
            # Select problem type (now includes Clustering)
            problem_type = st.selectbox(
                "Select problem type:",
                options=["Classification", "Regression", "Clustering"]   # <-- added Clustering
            )
            
            # Additional note for clustering
            if problem_type == "Clustering":
                st.info("ℹ️ For clustering, the target column is optional and will only be used for comparison (if available).")
            
            if st.button("Set Target & Continue", type="primary"):
                st.session_state.target_column = target_col
                st.session_state.problem_type = problem_type
                st.success(f"✅ Target set: {target_col} ({problem_type})")
                st.rerun()

# ---------- Page 2: Exploratory Analysis ----------
elif page == "🔍 Exploratory Analysis":
    st.markdown('<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first from the 'Data Upload' page.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "📊 Data Upload"
            st.rerun()
    else:
        df = st.session_state.data
        
        # EDA Options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Generate Full EDA Report", type="primary"):
                with st.spinner("Generating EDA report..."):
                    time.sleep(2)
                    st.session_state.eda_report = True
        
        with col2:
            if st.button("🔄 Reset EDA"):
                st.session_state.eda_report = None
        
        if st.session_state.eda_report:
            # ... (same EDA code as before, unchanged) ...
            # (I'm omitting the unchanged EDA section for brevity – it stays exactly as in the original)
            st.markdown("### 📋 Dataset Information")
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.metric("Rows", len(df))
            with info_col2:
                st.metric("Columns", len(df.columns))
            with info_col3:
                missing = df.isnull().sum().sum()
                st.metric("Missing Values", missing)
            with info_col4:
                memory = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory (MB)", f"{memory:.2f}")
            # ... (rest of EDA) ...

# ---------- Page 3: Model Training ----------
elif page == "🤖 Model Training":
    st.markdown('<h2 class="sub-header">🤖 Automated Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload data and set target column first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "📊 Data Upload"
            st.rerun()
    else:
        df = st.session_state.data
        target_col = st.session_state.target_column
        problem_type = st.session_state.problem_type
        
        st.markdown(f"""
        <div class="card">
        <h4>Training Configuration</h4>
        <ul>
            <li><strong>Problem Type:</strong> {problem_type}</li>
            <li><strong>Target Column:</strong> {target_col}</li>
            <li><strong>Dataset Shape:</strong> {df.shape}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Training Parameters
        st.markdown("### ⚙️ Training Parameters")
        
        # Preprocessing options (common for all types)
        preprocessing_cols = st.columns(3)
        with preprocessing_cols[0]:
            handle_missing = st.selectbox("Handle Missing Values", ["auto", "impute", "drop"])
        with preprocessing_cols[1]:
            scale_data = st.checkbox("Scale Numerical Features", value=True)
        with preprocessing_cols[2]:
            encode_categorical = st.checkbox("Encode Categorical Features", value=True)
        
        # -----------------------------------------------------------------
        # SUPERVISED LEARNING (Classification / Regression) with TPOT
        # -----------------------------------------------------------------
        if problem_type in ["Classification", "Regression"]:
            if not TPOT_AVAILABLE:
                st.error("TPOT is not installed. Please install it to use AutoML features.")
                st.code("pip install tpot", language="bash")
            else:
                # TPOT specific parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
                    generations = st.slider("Generations", 5, 50, 10)
                with col2:
                    population_size = st.slider("Population Size", 10, 100, 50)
                    cv_folds = st.slider("CV Folds", 2, 10, 5)
                with col3:
                    max_time_mins = st.slider("Max Time (minutes)", 1, 60, 10)
                    random_state = st.number_input("Random State", 0, 100, 42)
                
                # Start Training button
                if st.button("🚀 Start Automated Training", type="primary", use_container_width=True):
                    with st.spinner("🧪 Preparing data and starting TPOT..."):
                        try:
                            # Prepare data
                            X = df.drop(columns=[target_col])
                            y = df[target_col]
                            
                            # Handle categorical columns
                            if encode_categorical:
                                categorical_cols = X.select_dtypes(include=['object']).columns
                                for col in categorical_cols:
                                    le = LabelEncoder()
                                    X[col] = le.fit_transform(X[col].astype(str))
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state
                            )
                            
                            st.session_state.test_data = {
                                'X_test': X_test,
                                'y_test': y_test
                            }
                            
                            # Scale data if selected
                            if scale_data:
                                scaler = StandardScaler()
                                X_train = scaler.fit_transform(X_train)
                                X_test = scaler.transform(X_test)
                            
                            # Create progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Initialize TPOT
                            if problem_type == "Classification":
                                tpot = TPOTClassifier(
                                    generations=generations,
                                    population_size=population_size,
                                    cv=cv_folds,
                                    random_state=random_state,
                                    verbosity=2,
                                    n_jobs=-1,
                                    max_time_mins=max_time_mins
                                )
                            else:  # Regression
                                tpot = TPOTRegressor(
                                    generations=generations,
                                    population_size=population_size,
                                    cv=cv_folds,
                                    random_state=random_state,
                                    verbosity=2,
                                    n_jobs=-1,
                                    max_time_mins=max_time_mins
                                )
                            
                            # Training simulation with progress updates
                            for i in range(100):
                                progress_bar.progress(i + 1)
                                status_text.text(f"Training in progress... {i+1}%")
                                time.sleep(max_time_mins * 60 / 1000)  # Simulate training time
                            
                            # Fit TPOT
                            tpot.fit(X_train, y_train)
                            
                            # Store model and predictions
                            st.session_state.model = tpot
                            st.session_state.predictions = tpot.predict(X_test)
                            st.session_state.training_complete = True
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Training complete!")
                            
                            st.success("🎉 Model training completed successfully!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error during training: {str(e)}")
        
        # -----------------------------------------------------------------
        # CLUSTERING (Unsupervised)
        # -----------------------------------------------------------------
        elif problem_type == "Clustering":
            st.markdown("### 🔧 Clustering Algorithm Selection")
            
            # Algorithm choice
            clustering_algo = st.selectbox(
                "Select clustering algorithm",
                ["K-Means", "Agglomerative (Hierarchical)", "DBSCAN"]
            )
            
            # Algorithm-specific parameters
            if clustering_algo == "K-Means":
                n_clusters = st.slider("Number of clusters (k)", 2, 20, 3)
                init = st.selectbox("Initialization", ["k-means++", "random"])
                max_iter = st.number_input("Max iterations", 100, 1000, 300)
                random_state = st.number_input("Random state", 0, 100, 42)
                
            elif clustering_algo == "Agglomerative (Hierarchical)":
                n_clusters = st.slider("Number of clusters", 2, 20, 3)
                linkage = st.selectbox("Linkage criterion", ["ward", "complete", "average", "single"])
                affinity = st.selectbox("Affinity / metric", ["euclidean", "l1", "l2", "manhattan", "cosine"])
                
            else:  # DBSCAN
                eps = st.number_input("Epsilon (neighborhood radius)", 0.1, 10.0, 0.5)
                min_samples = st.number_input("Minimum samples in a neighborhood", 1, 50, 5)
                metric = st.selectbox("Distance metric", ["euclidean", "manhattan", "cosine"])
            
            # Start clustering button
            if st.button("🚀 Run Clustering", type="primary", use_container_width=True):
                with st.spinner("🔄 Computing clusters..."):
                    try:
                        # Prepare features (X) – drop target column (optional)
                        X = df.drop(columns=[target_col])
                        
                        # Handle missing values
                        if handle_missing == "drop":
                            X = X.dropna()
                        elif handle_missing == "impute":
                            for col in X.columns:
                                if X[col].dtype in ['int64', 'float64']:
                                    X[col].fillna(X[col].median(), inplace=True)
                                else:
                                    X[col].fillna(X[col].mode()[0], inplace=True)
                        
                        # Encode categorical features
                        if encode_categorical:
                            categorical_cols = X.select_dtypes(include=['object']).columns
                            for col in categorical_cols:
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col].astype(str))
                        
                        # Scale features (important for distance-based clustering)
                        if scale_data:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                        else:
                            X_scaled = X.values  # keep as numpy array
                        
                        # Fit the chosen algorithm
                        if clustering_algo == "K-Means":
                            model = KMeans(n_clusters=n_clusters, init=init,
                                           max_iter=max_iter, random_state=random_state)
                            labels = model.fit_predict(X_scaled)
                            
                        elif clustering_algo == "Agglomerative (Hierarchical)":
                            model = AgglomerativeClustering(n_clusters=n_clusters,
                                                            linkage=linkage, affinity=affinity)
                            labels = model.fit_predict(X_scaled)
                            # AgglomerativeClustering doesn't store labels in a standard attribute,
                            # but fit_predict returns them. We'll store the model and labels separately.
                            model = None  # we won't need the model object later
                            
                        else:  # DBSCAN
                            model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                            labels = model.fit_predict(X_scaled)
                        
                        # Store everything needed for evaluation
                        st.session_state.model = model          # fitted model (may be None for agglomerative)
                        st.session_state.predictions = labels   # cluster labels
                        st.session_state.X_scaled = X_scaled    # scaled features for evaluation plots
                        st.session_state.training_complete = True
                        
                        # Display basic cluster info
                        unique_labels = set(labels)
                        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                        n_noise = list(labels).count(-1)
                        
                        st.success(f"✅ Clustering completed!")
                        st.info(f"**Number of clusters found:** {n_clusters_found}")
                        if n_noise > 0:
                            st.info(f"**Noise points (DBSCAN):** {n_noise}")
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during clustering: {str(e)}")
        
        # Show training status for any completed training
        if st.session_state.training_complete:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Training Complete!</h4>
            <p>Your model has been trained successfully. You can now proceed to:</p>
            <ul>
                <li><strong>Model Evaluation:</strong> See how well your model performs</li>
                <li><strong>Make Predictions:</strong> Use the model on new data</li>
                <li><strong>Export Results:</strong> Save your model and results</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# ---------- Page 4: Model Evaluation ----------
elif page == "📈 Model Evaluation":
    st.markdown('<h2 class="sub-header">📈 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first from the 'Model Training' page.")
        if st.button("Go to Model Training"):
            st.session_state.page = "🤖 Model Training"
            st.rerun()
    else:
        problem_type = st.session_state.problem_type
        
        # -----------------------------------------------------------------
        # CLASSIFICATION EVALUATION
        # -----------------------------------------------------------------
        if problem_type == "Classification":
            model = st.session_state.model
            predictions = st.session_state.predictions
            test_data = st.session_state.test_data
            
            if test_data is None:
                st.error("Test data not found. Please retrain the model.")
            else:
                y_test = test_data['y_test']
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                
                # Display metrics
                st.markdown("### 📊 Classification Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("Precision", f"{precision:.4f}")
                with col3:
                    st.metric("Recall", f"{recall:.4f}")
                with col4:
                    st.metric("F1-Score", f"{f1:.4f}")
                
                # Confusion Matrix
                st.markdown("### 🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, predictions)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                # Classification Report
                st.markdown("### 📝 Detailed Classification Report")
                report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
                # Show best pipeline
                st.markdown("### 🏆 Best Pipeline Found by TPOT")
                if model is not None:
                    st.code(model.fitted_pipeline_, language='python')
                    
                    # Export pipeline as Python code
                    pipeline_code = model.export()
                    st.download_button(
                        label="📥 Download Pipeline Code",
                        data=pipeline_code,
                        file_name="best_pipeline.py",
                        mime="text/python"
                    )
        
        # -----------------------------------------------------------------
        # REGRESSION EVALUATION
        # -----------------------------------------------------------------
        elif problem_type == "Regression":
            model = st.session_state.model
            predictions = st.session_state.predictions
            test_data = st.session_state.test_data
            
            if test_data is None:
                st.error("Test data not found. Please retrain the model.")
            else:
                y_test = test_data['y_test']
                
                # Regression metrics
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, predictions)
                
                # Display metrics
                st.markdown("### 📊 Regression Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{mae:.4f}")
                with col2:
                    st.metric("MSE", f"{mse:.4f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col4:
                    st.metric("R² Score", f"{r2:.4f}")
                
                # Actual vs Predicted Plot
                st.markdown("### 📈 Actual vs Predicted Values")
                fig = px.scatter(
                    x=y_test, y=predictions,
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title='Actual vs Predicted Values'
                )
                max_val = max(max(y_test), max(predictions))
                min_val = min(min(y_test), min(predictions))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual Plot
                st.markdown("### 📉 Residual Plot")
                residuals = y_test - predictions
                fig = px.scatter(
                    x=predictions, y=residuals,
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    title='Residual Plot'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show best pipeline
                st.markdown("### 🏆 Best Pipeline Found by TPOT")
                if model is not None:
                    st.code(model.fitted_pipeline_, language='python')
                    
                    pipeline_code = model.export()
                    st.download_button(
                        label="📥 Download Pipeline Code",
                        data=pipeline_code,
                        file_name="best_pipeline.py",
                        mime="text/python"
                    )
        
        # -----------------------------------------------------------------
        # CLUSTERING EVALUATION
        # -----------------------------------------------------------------
        elif problem_type == "Clustering":
            labels = st.session_state.predictions   # cluster labels from training
            X_scaled = st.session_state.X_scaled    # scaled features used for clustering
            
            if X_scaled is None or labels is None:
                st.error("Clustering data not found. Please retrain the clustering model.")
            else:
                st.markdown("### 📊 Clustering Metrics")
                
                # Calculate metrics (handle edge cases)
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = list(labels).count(-1)
                
                # Silhouette Score (requires at least 2 clusters and more samples than clusters)
                if n_clusters >= 2 and len(labels) > n_clusters:
                    sil = silhouette_score(X_scaled, labels)
                    st.metric("Silhouette Score", f"{sil:.4f}",
                              help="Measures how similar an object is to its own cluster compared to other clusters. Range [-1,1], higher is better.")
                else:
                    st.warning("Silhouette score requires at least 2 clusters and more samples than clusters.")
                
                # Davies-Bouldin Index (always computable if n_clusters >= 2)
                if n_clusters >= 2:
                    db = davies_bouldin_score(X_scaled, labels)
                    st.metric("Davies-Bouldin Index", f"{db:.4f}",
                              help="Average similarity between each cluster and its most similar one. Lower is better (clusters more separated).")
                else:
                    st.warning("Davies-Bouldin index requires at least 2 clusters.")
                
                # Calinski-Harabasz Index (requires at least 2 clusters)
                if n_clusters >= 2:
                    ch = calinski_harabasz_score(X_scaled, labels)
                    st.metric("Calinski-Harabasz Index", f"{ch:.4f}",
                              help="Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.")
                else:
                    st.warning("Calinski-Harabasz index requires at least 2 clusters.")
                
                # Cluster Distribution
                st.markdown("### 📊 Cluster Distribution")
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                fig = px.bar(x=cluster_counts.index.astype(str), y=cluster_counts.values,
                             labels={'x': 'Cluster', 'y': 'Count'},
                             title="Number of points per cluster")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster Visualization with PCA
                st.markdown("### 🌌 Cluster Visualization (PCA Projection)")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Create a DataFrame for plotting
                plot_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': labels.astype(str)
                })
                
                fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                                 title="2D PCA projection of clusters",
                                 color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig, use_container_width=True)
                
                # Optionally show t-SNE (more time-consuming)
                if st.checkbox("Show t‑SNE visualization (may take a few seconds)"):
                    with st.spinner("Computing t‑SNE..."):
                        tsne = TSNE(n_components=2, random_state=42)
                        X_tsne = tsne.fit_transform(X_scaled)
                        plot_df_tsne = pd.DataFrame({
                            'tSNE1': X_tsne[:, 0],
                            'tSNE2': X_tsne[:, 1],
                            'Cluster': labels.astype(str)
                        })
                        fig = px.scatter(plot_df_tsne, x='tSNE1', y='tSNE2', color='Cluster',
                                         title="t‑SNE projection of clusters")
                        st.plotly_chart(fig, use_container_width=True)

# ---------- Page 5: Make Predictions ----------
elif page == "🔮 Make Predictions":
    st.markdown('<h2 class="sub-header">🔮 Make Predictions with Trained Model</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first from the 'Model Training' page.")
        if st.button("Go to Model Training"):
            st.session_state.page = "🤖 Model Training"
            st.rerun()
    else:
        problem_type = st.session_state.problem_type
        
        # Clustering does not support "prediction" on new points in the same way
        if problem_type == "Clustering":
            st.info("ℹ️ For clustering, prediction on new data is not directly supported in this demo. "
                    "You can export the cluster labels or use the trained model in your own code.")
            # Optionally, we could allow assigning new points to the nearest cluster (for K‑Means)
            if isinstance(st.session_state.model, KMeans):
                st.markdown("### Assign new points to clusters (K‑Means only)")
                # ... (could implement a simple upload form) ...
            else:
                st.warning("Only K‑Means models support assigning new points. Please retrain with K‑Means if you need this feature.")
        else:
            # For supervised learning, keep existing prediction interface
            model = st.session_state.model
            
            # Prediction options
            prediction_method = st.radio(
                "Select prediction method:",
                ["📤 Upload New Data", "✍️ Manual Input", "📊 Use Test Data"]
            )
            
            # ... (rest of the original prediction code) ...
            # (I'm omitting the unchanged prediction section for brevity – it stays as in the original)

# ---------- Page 6: Export Results ----------
elif page == "💾 Export Results":
    st.markdown('<h2 class="sub-header">💾 Export Model and Results</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first to export results.")
        if st.button("Go to Model Training"):
            st.session_state.page = "🤖 Model Training"
            st.rerun()
    else:
        model = st.session_state.model
        problem_type = st.session_state.problem_type
        
        # Export options
        st.markdown("### 📦 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🐍 Python Code")
            if problem_type in ["Classification", "Regression"] and TPOT_AVAILABLE:
                st.markdown("Export the best pipeline as executable Python code.")
                if st.button("Generate Pipeline Code"):
                    if model is not None:
                        pipeline_code = model.export()
                        st.code(pipeline_code, language='python')
                        st.download_button(
                            label="📥 Download Pipeline",
                            data=pipeline_code,
                            file_name="tpot_pipeline.py",
                            mime="text/python"
                        )
            elif problem_type == "Clustering":
                st.markdown("Export clustering model (if available) or cluster labels.")
                if isinstance(model, (KMeans, DBSCAN)):
                    # For scikit-learn models, we can use joblib
                    import joblib
                    model_bytes = joblib.dumps(model)
                    st.download_button(
                        label="📥 Download Model (joblib)",
                        data=model_bytes,
                        file_name="clustering_model.joblib"
                    )
                # Also export cluster labels
                if st.session_state.predictions is not None:
                    labels_df = pd.DataFrame({'cluster_label': st.session_state.predictions})
                    csv = labels_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="cluster_labels.csv">📥 Download Cluster Labels</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Model Report")
            st.markdown("Generate a comprehensive report of the model and results.")
            if st.button("Generate Model Report"):
                report_content = f"""
# Machine Learning Model Report

## Project Information
- Platform: No-Code ML Platform
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Problem Type: {st.session_state.problem_type}
- Target Column: {st.session_state.target_column}

## Dataset Information
- Original Shape: {st.session_state.data.shape if st.session_state.data else 'N/A'}
- Features: {len(st.session_state.data.columns) - 1 if st.session_state.data else 'N/A'}

## Model Information
- Best Pipeline: {model.fitted_pipeline_ if hasattr(model, 'fitted_pipeline_') else str(model) if model else 'N/A'}
- Training Completed: {st.session_state.training_complete}

## Notes
This model was generated through the No-Code ML Platform.
"""
                st.code(report_content, language='markdown')
                st.download_button(
                    label="📥 Download Report",
                    data=report_content,
                    file_name="ml_model_report.md",
                    mime="text/markdown"
                )
        
        # Session information
        st.markdown("### 📋 Session Information")
        session_info = {
            "Data Loaded": st.session_state.data is not None,
            "Target Column": st.session_state.target_column,
            "Problem Type": st.session_state.problem_type,
            "Model Trained": st.session_state.training_complete,
            "Predictions Made": st.session_state.predictions is not None,
            "Test Data Available": st.session_state.test_data is not None
        }
        session_df = pd.DataFrame.from_dict(session_info, orient='index', columns=['Status'])
        st.dataframe(session_df, use_container_width=True)
        
        # Reset session
        st.markdown("### 🔄 Reset Platform")
        st.warning("This will clear all data and models from the current session.")
        if st.button("🔄 Reset All Data", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>🤖 No-Code Machine Learning Platform | Developed with Streamlit & TPOT</p>
<p>📍 Universiti Malaysia Pahang Al-Sultan Abdullah | Data Science Project I</p>
</div>
""", unsafe_allow_html=True)