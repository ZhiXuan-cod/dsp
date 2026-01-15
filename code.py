import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import time
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# Try to import TPOT (optional - show message if not installed)
try:
    from tpot import TPOTClassifier, TPOTRegressor
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="No-Code ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize session state
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

# Main title
st.markdown('<h1 class="main-header">🤖 No-Code Machine Learning Platform</h1>', unsafe_allow_html=True)
st.markdown("**A web-based platform for accessible machine learning without coding**")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
    st.markdown("### Navigation")
    
    page = st.radio(
        "Select a step:",
        ["📊 Data Upload", "🔍 Exploratory Analysis", "🤖 Model Training", 
         "📈 Model Evaluation", "🔮 Make Predictions", "💾 Export Results"]
    )
    
    st.markdown("---")
    st.markdown("### Platform Info")
    st.info("""
    This platform enables:
    - CSV data upload
    - Automated EDA
    - AutoML with TPOT
    - Model evaluation
    - No-code predictions
    """)
    
    if not TPOT_AVAILABLE:
        st.error("⚠️ TPOT not installed. Install with: `pip install tpot`")
        st.code("pip install tpot", language="bash")

# Page 1: Data Upload
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
            
            # Select problem type
            problem_type = st.selectbox(
                "Select problem type:",
                options=["Classification", "Regression"]
            )
            
            if st.button("Set Target & Continue", type="primary"):
                st.session_state.target_column = target_col
                st.session_state.problem_type = problem_type
                st.success(f"✅ Target set: {target_col} ({problem_type})")
                st.rerun()

# Page 2: Exploratory Analysis
elif page == "🔍 Exploratory Analysis":
    st.markdown('<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first from the 'Data Upload' page.")
        if st.button("Go to Data Upload"):
            st.switch_page("📊 Data Upload")
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
            # Basic Information
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
            
            # Data Types
            st.markdown("### 🏷️ Data Types")
            dtype_df = pd.DataFrame(df.dtypes.value_counts()).reset_index()
            dtype_df.columns = ['Data Type', 'Count']
            fig = px.pie(dtype_df, values='Count', names='Data Type', 
                        title="Distribution of Data Types")
            st.plotly_chart(fig, use_container_width=True)
            
            # Missing Values
            st.markdown("### ⚠️ Missing Values Analysis")
            missing_series = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_series.index,
                'Missing_Count': missing_series.values,
                'Missing_Percentage': (missing_series.values / len(df)) * 100
            }).sort_values('Missing_Percentage', ascending=False)
            
            missing_df = missing_df[missing_df['Missing_Count'] > 0]
            
            if len(missing_df) > 0:
                fig = px.bar(missing_df, x='Column', y='Missing_Percentage',
                           title="Missing Values by Column (%)",
                           color='Missing_Percentage')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Numerical Columns Analysis
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                st.markdown("### 📊 Numerical Columns Analysis")
                
                selected_num_col = st.selectbox("Select numerical column:", numerical_cols)
                
                if selected_num_col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig = px.histogram(df, x=selected_num_col, 
                                         title=f"Distribution of {selected_num_col}",
                                         nbins=50)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Box plot
                        fig = px.box(df, y=selected_num_col, 
                                    title=f"Box Plot of {selected_num_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    stats = df[selected_num_col].describe()
                    st.dataframe(stats, use_container_width=True)
            
            # Categorical Columns Analysis
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.markdown("### 📊 Categorical Columns Analysis")
                
                selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
                
                if selected_cat_col:
                    value_counts = df[selected_cat_col].value_counts().head(20)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   title=f"Top Categories in {selected_cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(names=value_counts.index, values=value_counts.values,
                                   title=f"Distribution of {selected_cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Matrix (for numerical columns)
            if len(numerical_cols) > 1:
                st.markdown("### 🔗 Correlation Matrix")
                corr_matrix = df[numerical_cols].corr()
                
                fig = px.imshow(corr_matrix,
                              labels=dict(color="Correlation"),
                              x=corr_matrix.columns,
                              y=corr_matrix.columns,
                              title="Correlation Heatmap")
                fig.update_layout(width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Target Variable Analysis (if set)
            if st.session_state.target_column:
                st.markdown(f"### 🎯 Analysis of Target: {st.session_state.target_column}")
                target_col = st.session_state.target_column
                
                if target_col in df.columns:
                    if df[target_col].dtype in ['int64', 'float64']:
                        # Regression target
                        fig = px.histogram(df, x=target_col, 
                                         title=f"Distribution of Target ({target_col})")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Classification target
                        value_counts = df[target_col].value_counts()
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                       title=f"Class Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.pie(names=value_counts.index, values=value_counts.values,
                                       title=f"Class Proportions")
                            st.plotly_chart(fig, use_container_width=True)

# Page 3: Model Training
elif page == "🤖 Model Training":
    st.markdown('<h2 class="sub-header">🤖 Automated Model Training with TPOT</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload data and set target column first.")
        if st.button("Go to Data Upload"):
            st.switch_page("📊 Data Upload")
    else:
        if not TPOT_AVAILABLE:
            st.error("TPOT is not installed. Please install it to use AutoML features.")
            st.code("pip install tpot", language="bash")
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
            
            # Preprocessing options
            st.markdown("### 🔧 Preprocessing Options")
            
            preprocessing_cols = st.columns(3)
            
            with preprocessing_cols[0]:
                handle_missing = st.selectbox(
                    "Handle Missing Values",
                    ["auto", "impute", "drop"]
                )
            
            with preprocessing_cols[1]:
                scale_data = st.checkbox("Scale Numerical Features", value=True)
            
            with preprocessing_cols[2]:
                encode_categorical = st.checkbox("Encode Categorical Features", value=True)
            
            # Start Training
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
            
            # Show training status
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

# Page 4: Model Evaluation
elif page == "📈 Model Evaluation":
    st.markdown('<h2 class="sub-header">📈 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first from the 'Model Training' page.")
        if st.button("Go to Model Training"):
            st.switch_page("🤖 Model Training")
    else:
        model = st.session_state.model
        predictions = st.session_state.predictions
        test_data = st.session_state.test_data
        
        if test_data is None:
            st.error("Test data not found. Please retrain the model.")
        else:
            y_test = test_data['y_test']
            problem_type = st.session_state.problem_type
            
            # Calculate metrics
            if problem_type == "Classification":
                # Classification metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted')
                recall = recall_score(y_test, predictions, average='weighted')
                f1 = f1_score(y_test, predictions, average='weighted')
                
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
                report = classification_report(y_test, predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
            else:  # Regression
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
                    x=y_test,
                    y=predictions,
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title='Actual vs Predicted Values'
                )
                
                # Add diagonal line
                max_val = max(max(y_test), max(predictions))
                min_val = min(min(y_test), min(predictions))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual Plot
                st.markdown("### 📉 Residual Plot")
                residuals = y_test - predictions
                
                fig = px.scatter(
                    x=predictions,
                    y=residuals,
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    title='Residual Plot'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
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

# Page 5: Make Predictions
elif page == "🔮 Make Predictions":
    st.markdown('<h2 class="sub-header">🔮 Make Predictions with Trained Model</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first from the 'Model Training' page.")
        if st.button("Go to Model Training"):
            st.switch_page("🤖 Model Training")
    else:
        model = st.session_state.model
        
        # Prediction options
        prediction_method = st.radio(
            "Select prediction method:",
            ["📤 Upload New Data", "✍️ Manual Input", "📊 Use Test Data"]
        )
        
        if prediction_method == "📤 Upload New Data":
            st.markdown("### 📤 Upload New Data for Prediction")
            
            new_file = st.file_uploader(
                "Upload new CSV file for predictions",
                type=['csv'],
                key="prediction_file"
            )
            
            if new_file is not None:
                try:
                    new_df = pd.read_csv(new_file)
                    
                    # Check if columns match training data
                    original_cols = st.session_state.data.drop(
                        columns=[st.session_state.target_column]
                    ).columns.tolist()
                    
                    missing_cols = set(original_cols) - set(new_df.columns)
                    extra_cols = set(new_df.columns) - set(original_cols)
                    
                    if missing_cols:
                        st.warning(f"⚠️ Missing columns: {missing_cols}")
                    
                    if extra_cols:
                        st.info(f"ℹ️ Extra columns will be ignored: {extra_cols}")
                    
                    # Align columns
                    new_df = new_df.reindex(columns=original_cols, fill_value=0)
                    
                    # Show preview
                    st.markdown("### 📋 Data Preview")
                    st.dataframe(new_df.head(), use_container_width=True)
                    
                    if st.button("🔮 Make Predictions", type="primary"):
                        with st.spinner("Making predictions..."):
                            predictions = model.predict(new_df)
                            
                            # Create results dataframe
                            results_df = new_df.copy()
                            results_df['Predictions'] = predictions
                            
                            st.success(f"✅ Predictions complete for {len(predictions)} samples!")
                            
                            # Show results
                            st.markdown("### 📊 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        elif prediction_method == "✍️ Manual Input":
            st.markdown("### ✍️ Enter Values Manually")
            
            if st.session_state.data is not None:
                # Get feature columns (excluding target)
                feature_cols = st.session_state.data.drop(
                    columns=[st.session_state.target_column]
                ).columns.tolist()
                
                # Create input form
                input_data = {}
                cols = st.columns(3)
                
                for i, col_name in enumerate(feature_cols):
                    with cols[i % 3]:
                        if st.session_state.data[col_name].dtype in ['int64', 'float64']:
                            # Numerical input
                            min_val = float(st.session_state.data[col_name].min())
                            max_val = float(st.session_state.data[col_name].max())
                            mean_val = float(st.session_state.data[col_name].mean())
                            
                            input_data[col_name] = st.number_input(
                                col_name,
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val
                            )
                        else:
                            # Categorical input
                            unique_vals = st.session_state.data[col_name].unique()[:10]
                            input_data[col_name] = st.selectbox(col_name, unique_vals)
                
                if st.button("🔮 Predict", type="primary"):
                    # Prepare input for prediction
                    input_df = pd.DataFrame([input_data])
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    st.markdown("### 🎯 Prediction Result")
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>Predicted {st.session_state.target_column}: {prediction}</h3>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:  # Use Test Data
            st.markdown("### 📊 Predictions on Test Data")
            
            if st.session_state.test_data is not None:
                X_test = st.session_state.test_data['X_test']
                y_test = st.session_state.test_data['y_test']
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Create comparison dataframe
                comparison_df = X_test.copy()
                comparison_df['Actual'] = y_test.values
                comparison_df['Predicted'] = predictions
                
                # Show first 20 rows
                st.dataframe(comparison_df.head(20), use_container_width=True)
                
                # Download comparison
                csv = comparison_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="test_predictions.csv">📥 Download Test Predictions</a>'
                st.markdown(href, unsafe_allow_html=True)

# Page 6: Export Results
elif page == "💾 Export Results":
    st.markdown('<h2 class="sub-header">💾 Export Model and Results</h2>', unsafe_allow_html=True)
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first to export results.")
        if st.button("Go to Model Training"):
            st.switch_page("🤖 Model Training")
    else:
        model = st.session_state.model
        
        # Export options
        st.markdown("### 📦 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🐍 Python Code")
            st.markdown("Export the best pipeline as executable Python code.")
            
            if st.button("Generate Pipeline Code"):
                if model is not None:
                    pipeline_code = model.export()
                    st.code(pipeline_code, language='python')
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Pipeline",
                        data=pipeline_code,
                        file_name="tpot_pipeline.py",
                        mime="text/python"
                    )
        
        with col2:
            st.markdown("#### 📊 Model Report")
            st.markdown("Generate a comprehensive report of the model and results.")
            
            if st.button("Generate Model Report"):
                # Create a simple report
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
                - Best Pipeline: {model.fitted_pipeline_ if model else 'N/A'}
                - Training Completed: {st.session_state.training_complete}
                
                ## Notes
                This model was generated using TPOT AutoML through the No-Code ML Platform.
                """
                
                st.code(report_content, language='markdown')
                
                # Download button
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>🤖 No-Code Machine Learning Platform | Developed with Streamlit & TPOT</p>
<p>📍 Universiti Malaysia Pahang Al-Sultan Abdullah | Data Science Project I</p>
</div>
""", unsafe_allow_html=True)