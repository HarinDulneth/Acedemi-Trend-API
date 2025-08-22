# app.py - Complete Career Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')


#Import your existing modules
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model import CareerPredictionModel
from handler import StudentPredictionAPI


class CareerPredictionSystem:
    """
    Complete Career Prediction System with Streamlit Interface
    """

    def __init__(self):
        """Initialize the system"""
        self.api = StudentPredictionAPI()
        self.data_loader = None
        self.feature_engineer = None
        self.model = None
        self.system_ready = False
        
    def setup_system(self):
        """Setup and train the complete system"""
        if 'system_setup' not in st.session_state:
            st.session_state.system_setup = False
            
        if not st.session_state.system_setup:
            with st.spinner("Setting up the career prediction system..."):
                try:
                    # 1. Load and process data
                    st.info("📊 Loading datasets...")
                    self.data_loader = DataLoader()
                    datasets = self.data_loader.load_all_datasets()
                    comprehensive_df = self.data_loader.create_comprehensive_dataset()
                    
                    # 2. Feature engineering
                    st.info("🔧 Engineering features...")
                    self.feature_engineer = FeatureEngineer()
                    engineered_df = self.feature_engineer.engineer_features(comprehensive_df)
                    
                    # 3. Prepare for modeling
                    st.info("🎯 Preparing features for modeling...")
                    features_df, target_series = self.feature_engineer.prepare_features_for_modeling(
                        engineered_df, target_column='starting_salary_lkr'
                    )
                    
                    # 4. Feature selection
                    st.info("✨ Selecting best features...")
                    selected_features_df, selected_feature_names = self.feature_engineer.select_best_features(
                        features_df, target_series, k=20
                    )
                    
                    # 5. Train model
                    st.info("🤖 Training prediction models...")
                    self.model = CareerPredictionModel(random_state=42)
                    model_results = self.model.evaluate_models(selected_features_df, target_series)
                    
                    # 6. Setup API
                    st.info("🚀 Setting up prediction API...")
                    self.api.load_model(self.feature_engineer, self.model)
                    
                    # Store in session state
                    st.session_state.comprehensive_df = comprehensive_df
                    st.session_state.model_results = model_results
                    st.session_state.selected_feature_names = selected_feature_names
                    st.session_state.system_setup = True
                    st.session_state.api = self.api
                    
                    st.success("✅ System setup completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ System setup failed: {str(e)}")
                    return False
                    
        else:
            # Load from session state
            self.api = st.session_state.api
            
        return True

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Career Prediction System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .prediction-result {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🎓 Career Outcome & Salary Prediction System
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">
            AI-Powered Career Guidance for Computer Science Students
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    system = CareerPredictionSystem()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📊 System Analytics", "🔮 Student Prediction", "📈 Data Insights", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page(system)
    elif page == "📊 System Analytics":
        show_analytics_page(system)
    elif page == "🔮 Student Prediction":
        show_prediction_page(system)
    elif page == "📈 Data Insights":
        show_insights_page(system)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(system):
    """Display the home page"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Welcome to the Career Prediction System")
        
        st.markdown("""
        This advanced AI system helps Computer Science students predict their career outcomes 
        and starting salaries based on their academic performance, experience, and skills.
        
        ### 🎯 Key Features:
        - **Salary Prediction**: Get accurate salary predictions based on your profile
        - **Career Insights**: Receive personalized career guidance and recommendations
        - **Performance Analytics**: Understand what factors influence career success
        - **Real-time Processing**: Get instant predictions with confidence intervals
        
        ### 📋 How it Works:
        1. **Input Your Data**: Provide your academic and experience information
        2. **AI Analysis**: Our machine learning models analyze your profile
        3. **Get Predictions**: Receive salary predictions with actionable insights
        4. **Follow Recommendations**: Implement suggestions to improve your prospects
        """)
        
        # Setup system button
        if st.button("🔧 Initialize System", type="primary", use_container_width=True):
            system.setup_system()
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        # Sample metrics (you can replace with real data)
        st.metric("Students Analyzed", "2,500+", "12%")
        st.metric("Prediction Accuracy", "87%", "3%")
        st.metric("Career Pathways", "5", "")
        st.metric("Average Salary", "LKR 85,000", "8%")
        
        st.markdown("### 🎯 Supported Pathways")
        pathways = [
            "🤖 Artificial Intelligence",
            "📊 Data Science", 
            "🔒 Cyber Security",
            "🧮 Scientific Computing",
            "💻 Standard Computing"
        ]
        
        for pathway in pathways:
            st.markdown(f"- {pathway}")

def show_prediction_page(system):
    """Display the student prediction page"""
    st.markdown("## 🔮 Student Career Prediction")
    
    # Check if system is ready
    if not system.setup_system():
        st.warning("Please initialize the system first from the Home page.")
        return
    
    # Get input schema
    try:
        schema = st.session_state.api.get_input_schema()
        required_fields = schema['required_fields']
        optional_fields = schema['optional_fields']
        
        # Create input form
        with st.form("student_prediction_form"):
            st.markdown("### 📝 Required Information")
            
            col1, col2 = st.columns(2)
            student_data = {}
            
            with col1:
                student_data['student_id'] = st.text_input(
                    "Student ID", 
                    help="Your unique student identifier"
                )
                
                student_data['gender'] = st.selectbox(
                    "Gender",
                    options=['Male', 'Female']
                )
                
                student_data['age_at_enrollment'] = st.number_input(
                    "Age at Enrollment",
                    min_value=17, max_value=30, value=19
                )
                
                student_data['province'] = st.selectbox(
                    "Province",
                    options=['Western', 'Central', 'Southern', 'Northern', 'Eastern', 
                            'North Western', 'North Central', 'Uva', 'Sabaragamuwa']
                )
                
                student_data['district'] = st.text_input(
                    "District",
                    help="Your district of origin"
                )
            
            with col2:
                student_data['z_score_AL'] = st.number_input(
                    "A/L Z-Score",
                    min_value=1.0, max_value=2.5, value=1.8, step=0.01,
                    help="Your A/L Z-score (1.0 to 2.5)"
                )
                
                student_data['pathway'] = st.selectbox(
                    "Academic Pathway",
                    options=['Artificial Intelligence', 'Data Science', 'Cyber Security', 
                            'Scientific Computing', 'Standard']
                )
                
                student_data['intake_year'] = st.number_input(
                    "Intake Year",
                    min_value=2018, max_value=2025, value=2022
                )
                
                student_data['current_semester'] = st.number_input(
                    "Current Semester",
                    min_value=1, max_value=8, value=6
                )
            
            # Optional fields
            st.markdown("### 📊 Optional Information (for better accuracy)")
            
            with st.expander("Academic Performance"):
                col3, col4 = st.columns(2)
                with col3:
                    # Fixed: Use proper default value and checkbox to handle optional input
                    include_gpa = st.checkbox("Include Current GPA")
                    if include_gpa:
                        current_gpa = st.number_input(
                            "Current GPA",
                            min_value=0.0, max_value=4.0, value=3.0, step=0.01,
                            help="Your current cumulative GPA"
                        )
                        student_data['current_gpa'] = current_gpa
                
            with st.expander("Experience & Skills"):
                col5, col6 = st.columns(2)
                with col5:
                    internships = st.number_input("Completed Internships", min_value=0, max_value=10, value=0)
                    if internships > 0:
                        student_data['completed_internships'] = internships
                        
                        # Internship ratings
                        st.markdown("**Internship Ratings (1-5)**")
                        ratings = []
                        for i in range(int(internships)):
                            rating = st.slider(f"Internship {i+1} Rating", 1, 5, 4, key=f"rating_{i}")
                            ratings.append(rating)
                        student_data['internship_ratings'] = ratings
                        
                        total_months = st.number_input("Total Internship Months", min_value=1, max_value=24, value=3)
                        student_data['total_internship_months'] = total_months
                
                with col6:
                    projects = st.number_input("Completed Projects", min_value=0, max_value=20, value=0)
                    if projects > 0:
                        student_data['completed_projects'] = projects
                        
                        tech_input = st.text_area(
                            "Technologies Used",
                            placeholder="e.g., Python, React, SQL, Machine Learning",
                            help="Comma-separated list of technologies"
                        )
                        if tech_input:
                            student_data['project_technologies'] = [t.strip() for t in tech_input.split(',')]
                    
                    certs = st.number_input("Professional Certifications", min_value=0, max_value=10, value=0)
                    if certs > 0:
                        student_data['certifications_earned'] = certs
            
            # Fixed: Ensure submit button is properly within the form
            submitted = st.form_submit_button(
                "🔮 Predict Career Outcomes", 
                type="primary",
                use_container_width=True
            )
            
        # Handle form submission outside the form context
        if submitted:
            if not student_data.get('student_id'):
                st.error("⚠️ Please provide a Student ID")
            else:
                with st.spinner("🔄 Analyzing your profile and generating predictions..."):
                    try:
                        # Make prediction
                        result = st.session_state.api.predict(student_data)
                        
                        if 'error' in result:
                            st.error(f"❌ Prediction failed: {result['message']}")
                        else:
                            show_prediction_results(result, student_data)
                            
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        # Add debug info
                        with st.expander("Debug Information"):
                            st.write("Student data:", student_data)
                            st.write("Error details:", str(e))
    
    except Exception as e:
        st.error(f"❌ System error: {str(e)}")
        # Add debug info
        with st.expander("Debug Information"):
            st.write("Error details:", str(e))

def show_prediction_results(result, student_data):
    """Display prediction results"""
    st.markdown("## 🎯 Your Career Prediction Results")
    
    # Main prediction
    salary_pred = result['predicted_salary']
    
    st.markdown(f"""
    <div class="prediction-result">
        <h2 style="text-align: center; margin-bottom: 1rem;">💰 Predicted Starting Salary</h2>
        <h1 style="text-align: center; color: #2E86C1;">
            LKR {salary_pred['amount']:,.0f}
        </h1>
        <p style="text-align: center; margin-top: 1rem;">
            <strong>Confidence Range:</strong> 
            LKR {salary_pred['confidence_interval']['lower']:,.0f} - 
            LKR {salary_pred['confidence_interval']['upper']:,.0f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    profile = result['student_profile']
    
    with col1:
        st.metric("Experience Score", f"{profile['experience_score']:.1f}/10")
    with col2:
        st.metric("Academic Level", profile['academic_performance'])
    with col3:
        st.metric("Pathway", profile['pathway'])
    with col4:
        st.metric("Progress", profile['completion_status'])
    
    # Insights and recommendations
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("### 💡 Key Insights")
        for insight in result['insights']:
            st.markdown(f"• {insight}")
    
    with col6:
        st.markdown("### 🎯 Recommendations")
        for recommendation in result['recommendations']:
            st.markdown(f"• {recommendation}")
    
    # Salary comparison chart
    st.markdown("### 📊 Salary Comparison")
    
    # Create comparison data (you can make this more sophisticated)
    pathways = ['Artificial Intelligence', 'Data Science', 'Cyber Security', 'Scientific Computing', 'Standard']
    avg_salaries = [95000, 90000, 85000, 80000, 75000]  # Sample data
    
    fig = go.Figure()
    
    # Add average salaries
    fig.add_trace(go.Bar(
        x=pathways,
        y=avg_salaries,
        name='Average Salary',
        marker_color='lightblue'
    ))
    
    # Highlight user's pathway
    user_pathway = student_data['pathway']
    if user_pathway in pathways:
        idx = pathways.index(user_pathway)
        colors = ['lightblue'] * len(pathways)
        colors[idx] = 'darkblue'
        
        fig.data[0].marker.color = colors
        
        # Add user's prediction
        fig.add_trace(go.Scatter(
            x=[user_pathway],
            y=[salary_pred['amount']],
            mode='markers',
            name='Your Prediction',
            marker=dict(color='red', size=15, symbol='star')
        ))
    
    fig.update_layout(
        title="Salary Comparison by Pathway",
        xaxis_title="Pathway",
        yaxis_title="Starting Salary (LKR)",
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_analytics_page(system):
    """Display system analytics"""
    st.markdown("## 📊 System Analytics")
    
    if not system.setup_system():
        st.warning("⚠️ Please initialize the system first from the Home page.")
        return
    
    # Get data from session state
    if 'comprehensive_df' in st.session_state:
        df = st.session_state.comprehensive_df
        model_results = st.session_state.model_results
        
        # Model performance
        st.markdown("### 🤖 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not model_results.empty:
                fig = px.bar(
                    model_results, 
                    x='model', 
                    y='test_r2',
                    title="Model R² Scores",
                    color='test_r2',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if not model_results.empty:
                fig = px.bar(
                    model_results, 
                    x='model', 
                    y='test_rmse',
                    title="Model RMSE Scores",
                    color='test_rmse',
                    color_continuous_scale='viridis_r'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Data insights
        st.markdown("### 📈 Dataset Overview")
        
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            st.metric("Total Students", len(df))
        with col4:
            placed_count = df['placed'].sum() if 'placed' in df.columns else 0
            st.metric("Placed Students", placed_count)
        with col5:
            if 'starting_salary_lkr' in df.columns and 'placed' in df.columns:
                avg_salary = df[df['placed'] == True]['starting_salary_lkr'].mean()
                st.metric("Average Salary", f"LKR {avg_salary:,.0f}" if not pd.isna(avg_salary) else "N/A")
            else:
                st.metric("Average Salary", "N/A")
        with col6:
            placement_rate = (placed_count / len(df) * 100) if len(df) > 0 else 0
            st.metric("Placement Rate", f"{placement_rate:.1f}%")
    else:
        st.info("📊 No data available. Please initialize the system first.")

def show_insights_page(system):
    """Display data insights"""
    st.markdown("## 📈 Data Insights")
    
    if not system.setup_system():
        st.warning("⚠️ Please initialize the system first from the Home page.")
        return
    
    if 'comprehensive_df' in st.session_state:
        df = st.session_state.comprehensive_df
        
        # Salary distribution by pathway
        if 'pathway' in df.columns and 'starting_salary_lkr' in df.columns and 'placed' in df.columns:
            placed_df = df[df['placed'] == True]
            
            if not placed_df.empty:
                st.markdown("### 💰 Salary Distribution by Pathway")
                fig = px.box(
                    placed_df, 
                    x='pathway', 
                    y='starting_salary_lkr',
                    title="Starting Salary Distribution by Academic Pathway"
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # GPA vs Salary correlation
        if 'cumulative_gpa' in df.columns and 'starting_salary_lkr' in df.columns and 'placed' in df.columns:
            st.markdown("### 📚 GPA vs Starting Salary")
            placed_df = df[(df['placed'] == True) & (df['cumulative_gpa'].notna())]
            
            if not placed_df.empty:
                fig = px.scatter(
                    placed_df,
                    x='cumulative_gpa',
                    y='starting_salary_lkr',
                    color='pathway' if 'pathway' in placed_df.columns else None,
                    title="Correlation between GPA and Starting Salary",
                    trendline='ols'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### 📊 Additional Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'gender' in df.columns:
                gender_dist = df['gender'].value_counts()
                fig = px.pie(
                    values=gender_dist.values,
                    names=gender_dist.index,
                    title="Gender Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'pathway' in df.columns:
                pathway_dist = df['pathway'].value_counts()
                fig = px.bar(
                    x=pathway_dist.index,
                    y=pathway_dist.values,
                    title="Students by Pathway"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No data available. Please initialize the system first.")

def show_about_page():
    """Display about page"""
    st.markdown("## ℹ️ About the System")
    
    st.markdown("""
    ### 🎓 Career Outcome & Salary Prediction System
    
    This system uses advanced machine learning algorithms to predict career outcomes and starting salaries 
    for Computer Science students based on their academic performance, experience, and skills.
    
    #### 🧠 Machine Learning Models:
    - **Random Forest**: Ensemble method for robust predictions
    - **Gradient Boosting**: Advanced boosting for high accuracy
    - **Linear Regression**: Baseline model for interpretability
    - **XGBoost**: State-of-the-art gradient boosting
    
    #### 📊 Key Features:
    - Real-time salary predictions with confidence intervals
    - Personalized career insights and recommendations
    - Interactive data visualizations
    - Support for 5 different academic pathways
    
    #### 🎯 Pathways Supported:
    1. **Artificial Intelligence** - Machine Learning, Deep Learning, NLP
    2. **Data Science** - Analytics, Big Data, Statistics
    3. **Cyber Security** - Information Security, Ethical Hacking
    4. **Scientific Computing** - Computational Methods, Modeling
    5. **Standard Computing** - General Software Development
    
    #### 📈 Prediction Factors:
    - Academic performance (GPA, A/L results)
    - Practical experience (internships, projects)
    - Skills and certifications
    - Geographic and demographic factors
    - Academic pathway specialization
    
    ---
    
    **Built with**: Python, Streamlit, Scikit-learn, Plotly
    
    **Version**: 1.0.0
    """)

if __name__ == "__main__":
    main()