# Page configuration MUST BE FIRST
import streamlit as st
st.set_page_config(
    page_title="Employee Layoff Analytics & Prediction Dashboard",
    page_icon="🤖",
    layout="wide"
)

# THEN import other libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost for better accuracy
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# App title
st.title("🤖 Employee Layoff Analytics & Prediction Dashboard")

# Function to generate enhanced sample data with realistic patterns
@st.cache_data
def generate_enhanced_sample_data():
    np.random.seed(42)
    n_rows = 3000
    
    companies = ['TechCorp', 'FinServe', 'HealthPlus', 'RetailGiant', 'AutoMakers']
    industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Automotive']
    cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai']
    roles = ['Software Engineer', 'Data Analyst', 'Product Manager', 'HR Manager', 'Sales Executive']
    
    # Create realistic patterns with stronger correlations
    df = pd.DataFrame({
        'Employee_ID': [f'EMP{str(i).zfill(5)}' for i in range(1, n_rows+1)],
        'Company': np.random.choice(companies, n_rows, p=[0.25, 0.2, 0.2, 0.2, 0.15]),
        'Industry': np.random.choice(industries, n_rows, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        'City': np.random.choice(cities, n_rows),
        'Role': np.random.choice(roles, n_rows, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        'Age': np.random.normal(35, 8, n_rows).clip(22, 60).astype(int),
        'Gender': np.random.choice(['Male', 'Female'], n_rows, p=[0.6, 0.4]),
        'Experience_Years': np.random.exponential(5, n_rows).clip(1, 25).astype(int),
        'Salary_LPA': np.random.lognormal(2.5, 0.4, n_rows).clip(4, 50),
        'Performance_Rating': np.random.choice(['Exceeds', 'Meets', 'Below'], n_rows, p=[0.2, 0.6, 0.2]),
        'Tenure_In_Company': np.random.exponential(3, n_rows).clip(1, 15).astype(int),
        'Projects_Completed': np.random.poisson(8, n_rows),
        'Promotions': np.random.binomial(3, 0.3, n_rows),
        'Layoff_Reason': np.random.choice(['Cost Cutting', 'Restructuring', 'Performance Issues', 
                                          'Company Closure', 'Automation'], n_rows),
        'Layoff_Year': np.random.choice([2021, 2022, 2023], n_rows, p=[0.3, 0.4, 0.3]),
    })
    
    # Create layoff patterns with strong correlations
    df['Laid_Off'] = 0
    
    # Strong predictors - more realistic patterns
    performance_weights = {'Exceeds': 0.05, 'Meets': 0.3, 'Below': 0.85}
    df['Performance_Weight'] = df['Performance_Rating'].map(performance_weights)
    
    # Industry risk
    industry_risk = {'Technology': 0.7, 'Finance': 0.4, 'Healthcare': 0.2, 'Retail': 0.8, 'Automotive': 0.5}
    df['Industry_Risk'] = df['Industry'].map(industry_risk)
    
    # Company-specific risks
    company_risk = {'TechCorp': 0.8, 'FinServe': 0.3, 'HealthPlus': 0.2, 'RetailGiant': 0.9, 'AutoMakers': 0.4}
    df['Company_Risk'] = df['Company'].map(company_risk)
    
    # Year risk
    year_risk = {2021: 0.3, 2022: 0.7, 2023: 0.5}
    df['Year_Risk'] = df['Layoff_Year'].map(year_risk)
    
    # Calculate layoff probability with clear patterns
    df['Layoff_Probability'] = (
        0.35 * df['Performance_Weight'] +
        0.25 * df['Industry_Risk'] +
        0.15 * df['Company_Risk'] +
        0.10 * df['Year_Risk'] +
        0.05 * (df['Salary_LPA'] > df['Salary_LPA'].median()) +
        0.05 * (df['Experience_Years'] > 15) +
        0.05 * (df['Projects_Completed'] < 5) +
        np.random.normal(0, 0.05, n_rows)
    )
    
    df['Layoff_Probability'] = df['Layoff_Probability'].clip(0, 1)
    
    # Generate layoffs with clear threshold
    threshold = np.random.uniform(0.3, 0.5, n_rows)
    df['Laid_Off'] = (df['Layoff_Probability'] > threshold).astype(int)
    
    # Ensure balanced dataset
    target_rate = 0.45
    current_rate = df['Laid_Off'].mean()
    
    if current_rate < target_rate - 0.05:
        # Add more layoffs from high probability candidates
        n_needed = int((target_rate - current_rate) * len(df))
        candidates = df[df['Laid_Off'] == 0].nlargest(n_needed, 'Layoff_Probability')
        df.loc[candidates.index, 'Laid_Off'] = 1
    elif current_rate > target_rate + 0.05:
        # Remove layoffs from low probability candidates
        n_remove = int((current_rate - target_rate) * len(df))
        candidates = df[df['Laid_Off'] == 1].nsmallest(n_remove, 'Layoff_Probability')
        df.loc[candidates.index, 'Laid_Off'] = 0
    
    # Drop temporary columns
    df = df.drop(['Performance_Weight', 'Industry_Risk', 'Company_Risk', 
                  'Year_Risk', 'Layoff_Probability'], axis=1)
    
    return df

# Feature engineering function (used by both training and prediction)
def feature_engineering(df, median_salary=None):
    """Add engineered features to the dataframe."""
    df_engineered = df.copy()
    
    # Calculate median salary if not provided
    if median_salary is None:
        median_salary = df_engineered['Salary_LPA'].median()
    
    # Feature engineering
    df_engineered['Salary_Experience_Ratio'] = df_engineered['Salary_LPA'] / (df_engineered['Experience_Years'] + 1)
    df_engineered['High_Performer'] = (df_engineered['Performance_Rating'] == 'Exceeds').astype(int)
    df_engineered['Low_Performer'] = (df_engineered['Performance_Rating'] == 'Below').astype(int)
    df_engineered['Productivity_Score'] = df_engineered['Projects_Completed'] / (df_engineered['Experience_Years'] + 1)
    df_engineered['Stagnant_Career'] = ((df_engineered['Experience_Years'] > 10) & (df_engineered['Promotions'] < 2)).astype(int)
    df_engineered['High_Cost_Employee'] = (df_engineered['Salary_LPA'] > median_salary).astype(int)
    df_engineered['Recent_Hire'] = (df_engineered['Tenure_In_Company'] < 2).astype(int)
    
    # Create Age_Group
    age_bins = [20, 30, 40, 50, 60]
    age_labels = ['20-30', '30-40', '40-50', '50-60']
    df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    return df_engineered, median_salary

# Enhanced preprocessing with feature engineering
def preprocess_data_for_ml(df, for_training=True):
    """Preprocess data for machine learning with feature engineering"""
    
    # Apply feature engineering
    df_engineered, median_salary = feature_engineering(df)
    
    # Define feature sets
    numeric_features = [
        'Age', 'Experience_Years', 'Salary_LPA', 'Tenure_In_Company',
        'Projects_Completed', 'Promotions', 'Salary_Experience_Ratio', 
        'Productivity_Score'
    ]
    
    categorical_features = ['Gender', 'Performance_Rating', 'Industry', 'Role', 'Age_Group']
    
    binary_features = ['High_Performer', 'Low_Performer', 'Stagnant_Career', 
                      'High_Cost_Employee', 'Recent_Hire']
    
    all_features = numeric_features + categorical_features + binary_features
    
    if for_training:
        X = df_engineered[all_features].copy()
        y = df_engineered['Laid_Off'].copy()
        return X, y, median_salary, all_features
    else:
        X = df_engineered[all_features].copy()
        return X, median_salary, all_features

# Improved model training with better preprocessing
def train_improved_model(X, y, model_type='XGBoost'):
    """Train improved model with better accuracy"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify column types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define models with optimized parameters
    if model_type == 'XGBoost' and XGB_AVAILABLE:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        )
    else:
        # Default to Random Forest if XGBoost not available
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    
    # Train final model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'cv_mean_f1': cv_scores.mean(),
        'cv_std_f1': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'pipeline': pipeline,
        'model_type': model_type,
        'feature_names': list(X.columns)  # Store the feature names
    }
    
    # Get feature importance if available
    try:
        if model_type in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                importance = pipeline.named_steps['classifier'].feature_importances_
                
                # Get feature names after preprocessing
                preprocessor = pipeline.named_steps['preprocessor']
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = cat_encoder.get_feature_names_out(categorical_features)
                all_features = list(numeric_features) + list(cat_features)
                
                if len(importance) == len(all_features):
                    metrics['feature_importance'] = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False).head(15)
    except Exception as e:
        metrics['feature_importance'] = None
    
    return metrics

# Function for single prediction
def predict_single_employee(pipeline, input_data, median_salary, feature_names):
    """Predict for a single employee"""
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply feature engineering with the same median salary
    input_df_engineered, _ = feature_engineering(input_df, median_salary)
    
    # Make sure we have all expected features
    for feature in feature_names:
        if feature not in input_df_engineered.columns:
            # Add missing features with default values
            if feature in ['High_Performer', 'Low_Performer', 'Stagnant_Career', 
                          'High_Cost_Employee', 'Recent_Hire']:
                input_df_engineered[feature] = 0
            else:
                input_df_engineered[feature] = 0
    
    # Ensure correct column order
    input_df_engineered = input_df_engineered[feature_names]
    
    # Make prediction
    try:
        prediction = pipeline.predict(input_df_engineered)[0]
        prediction_proba = pipeline.predict_proba(input_df_engineered)[0]
        return prediction, prediction_proba
    except Exception as e:
        # Fallback to default prediction if error occurs
        st.error(f"Prediction error: {str(e)}")
        return 0, [0.5, 0.5]

# Generate sample data
df = generate_enhanced_sample_data()

# Sidebar for filters only (no CSV upload)
with st.sidebar:
    st.header("🔍 Filters")
    
    # Year filter
    years = sorted(df['Layoff_Year'].unique())
    selected_years = st.multiselect(
        "Select Years",
        options=years,
        default=years
    )
    
    # Reason filter
    reasons = df['Layoff_Reason'].unique()
    selected_reasons = st.multiselect(
        "Select Layoff Reasons",
        options=reasons,
        default=reasons
    )
    
    # Industry filter
    industries = df['Industry'].unique()
    selected_industries = st.multiselect(
        "Select Industries",
        options=industries,
        default=industries
    )
    
    # Additional filters
    st.subheader("Additional Filters")
    
    age_range = st.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )
    
    experience_range = st.slider(
        "Experience Range (Years)",
        min_value=int(df['Experience_Years'].min()),
        max_value=int(df['Experience_Years'].max()),
        value=(int(df['Experience_Years'].min()), int(df['Experience_Years'].max()))
    )
    
    salary_range = st.slider(
        "Salary Range (LPA)",
        min_value=float(df['Salary_LPA'].min()),
        max_value=float(df['Salary_LPA'].max()),
        value=(float(df['Salary_LPA'].min()), float(df['Salary_LPA'].max()))
    )

# Apply filters
filtered_df = df[
    (df['Layoff_Year'].isin(selected_years)) &
    (df['Layoff_Reason'].isin(selected_reasons)) &
    (df['Industry'].isin(selected_industries)) &
    (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
    (df['Experience_Years'] >= experience_range[0]) & (df['Experience_Years'] <= experience_range[1]) &
    (df['Salary_LPA'] >= salary_range[0]) & (df['Salary_LPA'] <= salary_range[1])
]

# Main dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "🏢 Industry Analysis", "👥 Demographic Insights", 
    "💰 Salary Analysis", "🤖 ML Prediction"
])

with tab1:
    # Simple metrics without complex styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_employees = len(filtered_df)
        st.metric("Total Employees", f"{total_employees:,}")
    
    with col2:
        laid_off_count = filtered_df['Laid_Off'].sum()
        st.metric("Employees Laid Off", f"{laid_off_count:,}")
    
    with col3:
        layoff_rate = (laid_off_count / total_employees * 100) if total_employees > 0 else 0
        st.metric("Layoff Rate", f"{layoff_rate:.1f}%")
    
    with col4:
        avg_salary = filtered_df['Salary_LPA'].mean()
        st.metric("Avg Salary (LPA)", f"₹{avg_salary:.1f}")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Layoffs by Year")
        yearly_layoffs = filtered_df.groupby('Layoff_Year')['Laid_Off'].sum().reset_index()
        fig = px.bar(yearly_layoffs, x='Layoff_Year', y='Laid_Off',
                    title="Layoffs by Year")
        fig.update_layout(xaxis_title="Year", yaxis_title="Number of Layoffs")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Layoffs by Reason")
        reason_counts = filtered_df['Layoff_Reason'].value_counts().reset_index()
        reason_counts.columns = ['Layoff_Reason', 'Count']
        fig = px.pie(reason_counts, values='Count', names='Layoff_Reason',
                    title="Distribution of Layoff Reasons")
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Layoffs by Industry")
        industry_layoffs = filtered_df.groupby('Industry')['Laid_Off'].sum().reset_index()
        industry_layoffs = industry_layoffs.sort_values('Laid_Off', ascending=False)
        fig = px.bar(industry_layoffs, x='Industry', y='Laid_Off',
                    title="Layoffs by Industry")
        fig.update_layout(xaxis_title="Industry", yaxis_title="Number of Layoffs")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Layoffs by Company")
        company_layoffs = filtered_df.groupby('Company')['Laid_Off'].sum().reset_index()
        company_layoffs = company_layoffs.sort_values('Laid_Off', ascending=False).head(10)
        fig = px.bar(company_layoffs, x='Company', y='Laid_Off',
                    title="Top 10 Companies by Layoffs")
        fig.update_layout(xaxis_title="Company", yaxis_title="Number of Layoffs")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Industry-Wide Analysis")
    
    # Industry statistics
    industry_stats = filtered_df.groupby('Industry').agg({
        'Laid_Off': ['sum', 'count', 'mean'],
        'Salary_LPA': 'mean',
        'Experience_Years': 'mean'
    }).round(2)
    
    industry_stats.columns = ['Laid_Off_Count', 'Total_Employees', 'Layoff_Rate', 
                             'Avg_Salary', 'Avg_Experience']
    industry_stats = industry_stats.sort_values('Layoff_Rate', ascending=False)
    
    st.dataframe(industry_stats, use_container_width=True)
    
    # Industry comparison chart
    fig = px.bar(industry_stats.reset_index(), x='Industry', y='Layoff_Rate',
                title='Layoff Rate by Industry',
                color='Layoff_Rate',
                color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Demographic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender analysis
        gender_stats = filtered_df.groupby('Gender').agg({
            'Laid_Off': ['sum', 'count', 'mean']
        }).round(3)
        gender_stats.columns = ['Laid_Off_Count', 'Total', 'Layoff_Rate']
        
        fig = px.bar(gender_stats.reset_index(), x='Gender', y='Layoff_Rate',
                    title='Layoff Rate by Gender')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        fig = px.histogram(filtered_df, x='Age', color='Laid_Off',
                          title="Age Distribution by Layoff Status",
                          barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Salary Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary distribution
        fig = px.histogram(filtered_df, x='Salary_LPA', nbins=30,
                          title="Salary Distribution (LPA)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Salary vs Layoff status
        fig = px.box(filtered_df, x='Laid_Off', y='Salary_LPA',
                    title="Salary by Layoff Status")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Machine Learning Prediction Model")
    
    # Model Training Section
    st.write("The model uses advanced features to predict layoff risk with high accuracy.")
    
    # Model selection
    model_options = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
    if XGB_AVAILABLE:
        model_options.insert(0, 'XGBoost')
    
    model_type = st.selectbox("Select Model", model_options)
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model with enhanced features..."):
            # Preprocess data
            X, y, median_salary, feature_names = preprocess_data_for_ml(filtered_df)
            
            # Train model
            metrics = train_improved_model(X, y, model_type)
            
            # Store in session state
            st.session_state['metrics'] = metrics
            st.session_state['model_trained'] = True
            st.session_state['model_type'] = model_type
            st.session_state['median_salary'] = median_salary
            st.session_state['feature_names'] = feature_names
            st.session_state['training_df'] = filtered_df
            
            st.success(f"✅ {model_type} model trained successfully!")
    
    # Display results if model trained
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        metrics = st.session_state['metrics']
        
        st.subheader("Model Performance")
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        with col5:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        
        st.write(f"Cross-Validation F1 Score: {metrics['cv_mean_f1']:.3f} ± {metrics['cv_std_f1']:.3f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig = px.imshow(
            metrics['confusion_matrix'],
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Laid Off', 'Laid Off'],
            y=['Not Laid Off', 'Laid Off']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC curve (AUC = {roc_auc:.3f})',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random',
                                line=dict(color='gray', width=1, dash='dash')))
        
        fig.update_layout(
            title=f'ROC Curve - {model_type}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        if 'feature_importance' in metrics and metrics['feature_importance'] is not None:
            st.subheader("Top Important Features")
            fig = px.bar(metrics['feature_importance'], x='Importance', y='Feature',
                        orientation='h',
                        title="Feature Importance",
                        color='Importance')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Single Employee Prediction
    st.markdown("---")
    st.subheader("Predict Layoff Risk for Single Employee")
    
    if 'model_trained' not in st.session_state:
        st.info("👈 Please train a model first.")
    else:
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", 22, 60, 35)
                gender = st.selectbox("Gender", ["Male", "Female"])
                experience = st.slider("Experience (Years)", 1, 25, 5)
                salary = st.slider("Salary (LPA)", 4.0, 50.0, 15.0, 0.5)
            
            with col2:
                performance = st.selectbox("Performance Rating", ["Exceeds", "Meets", "Below"])
                industry = st.selectbox("Industry", sorted(df['Industry'].unique()))
                role = st.selectbox("Role", sorted(df['Role'].unique()))
                tenure = st.slider("Tenure in Company (Years)", 1, 15, 3)
                projects = st.slider("Projects Completed", 0, 30, 8)
                promotions = st.slider("Promotions", 0, 5, 1)
            
            submitted = st.form_submit_button("🔮 Predict Layoff Risk", use_container_width=True)
            
            if submitted:
                # Create input data
                input_data = {
                    'Age': age,
                    'Gender': gender,
                    'Experience_Years': experience,
                    'Salary_LPA': salary,
                    'Performance_Rating': performance,
                    'Industry': industry,
                    'Role': role,
                    'Tenure_In_Company': tenure,
                    'Projects_Completed': projects,
                    'Promotions': promotions
                }
                
                try:
                    # Get stored values
                    pipeline = st.session_state['metrics']['pipeline']
                    median_salary = st.session_state['median_salary']
                    feature_names = st.session_state['feature_names']
                    
                    # Make prediction
                    prediction, prediction_proba = predict_single_employee(
                        pipeline, input_data, median_salary, feature_names
                    )
                    
                    # Display results
                    risk_percentage = prediction_proba[1] * 100
                    
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    if risk_percentage >= 70:
                        st.error(f"**🚨 High Risk of Layoff ({risk_percentage:.1f}%)**")
                        st.write("**Recommendations:**")
                        st.write("1. Immediate performance review")
                        st.write("2. Skill development training")
                        st.write("3. Consider role adjustment")
                    elif risk_percentage >= 40:
                        st.warning(f"**⚠️ Medium Risk of Layoff ({risk_percentage:.1f}%)**")
                        st.write("**Recommendations:**")
                        st.write("1. Monitor performance closely")
                        st.write("2. Provide coaching and support")
                        st.write("3. Regular check-ins")
                    else:
                        st.success(f"**✅ Low Risk of Layoff ({risk_percentage:.1f}%)**")
                        st.write("**Recommendations:**")
                        st.write("1. Continue current performance")
                        st.write("2. Seek growth opportunities")
                        st.write("3. Regular skill updates")
                    
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_percentage,
                        title = {'text': "Layoff Risk Score"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_percentage
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Employee Layoff Analytics & Prediction Dashboard*")

# Installation note
if not XGB_AVAILABLE:
    st.sidebar.info("💡 Install XGBoost for better accuracy: `pip install xgboost`")