import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from job_salary_prediction.feature_engineering import FeatureEngineer
from job_salary_prediction.model import CareerPredictionModel
import matplotlib.pyplot as plt
import io
import base64

# Helper: Create base features from student input
def create_base_features(student_data):
    current_year = datetime.now().year
    years_since_intake = current_year - student_data['intake_year']
    base_features = {
        'student_id': student_data['student_id'],
        'gender': student_data['gender'],
        'age_at_enrollment': student_data['age_at_enrollment'],
        'province': student_data['province'],
        'district': student_data['district'],
        'z_score_AL': student_data['z_score_AL'],
        'pathway': student_data['pathway'],
        'intake_year': student_data['intake_year'],
        'years_since_intake': years_since_intake,
    }
    return pd.DataFrame([base_features])

# Helper: Estimate missing features
def estimate_missing_features(df, student_data):
    if 'current_gpa' in student_data and student_data['current_gpa'] is not None:
        df['cumulative_gpa'] = student_data['current_gpa']
    else:
        base_gpa = 2.0 + (student_data['z_score_AL'] - 1.45) * 1.5
        current_semester = student_data.get('current_semester', 1)
        semester_factor = min(current_semester / 8.0, 1.0)
        df['cumulative_gpa'] = base_gpa * (0.9 + 0.1 * semester_factor)
        df['cumulative_gpa'] = np.clip(df['cumulative_gpa'], 2.0, 4.0)
    if 'completed_internships' in student_data:
        df['internship_count'] = student_data['completed_internships']
        if 'internship_ratings' in student_data and student_data['internship_ratings']:
            df['avg_internship_rating'] = np.mean(student_data['internship_ratings'])
        else:
            df['avg_internship_rating'] = min(3.0 + df['cumulative_gpa'].iloc[0] * 0.5, 5.0)
        if 'total_internship_months' in student_data:
            df['total_internship_days'] = student_data['total_internship_months'] * 30
            df['avg_internship_duration'] = df['total_internship_days'] / max(df['internship_count'].iloc[0], 1)
        else:
            df['avg_internship_duration'] = 90
            df['total_internship_days'] = df['internship_count'] * 90
    else:
        current_semester = student_data.get('current_semester', 1)
        if current_semester >= 6:
            df['internship_count'] = max(1, current_semester - 5)
        else:
            df['internship_count'] = 0
        df['avg_internship_rating'] = 4.0
        df['avg_internship_duration'] = 90
        df['total_internship_days'] = df['internship_count'] * 90
    if 'completed_projects' in student_data:
        df['project_count'] = student_data['completed_projects']
        df['completed_projects'] = student_data['completed_projects']
        if 'project_technologies' in student_data and student_data['project_technologies']:
            df['total_technologies'] = len(set(student_data['project_technologies']))
        else:
            df['total_technologies'] = min(df['project_count'].iloc[0] * 2 + 3, 15)
    else:
        current_semester = student_data.get('current_semester', 1)
        df['project_count'] = max(1, current_semester // 2)
        df['completed_projects'] = df['project_count']
        df['total_technologies'] = df['project_count'] * 2 + 2
    df['avg_project_duration'] = 45
    if 'certifications_earned' in student_data:
        df['certification_count'] = student_data['certifications_earned']
    else:
        high_cert_pathways = ['Artificial Intelligence', 'Data Science', 'Cyber Security']
        current_semester = student_data.get('current_semester', 1)
        if student_data['pathway'] in high_cert_pathways:
            df['certification_count'] = max(0, current_semester // 3)
        else:
            df['certification_count'] = max(0, current_semester // 4)
    if 'capstone_domain' in student_data and student_data['capstone_domain']:
        df['domain'] = student_data['capstone_domain']
        df['technologies_used'] = 'Python,JavaScript,SQL'
        df['outcome'] = 'Completed'
    else:
        current_semester = student_data.get('current_semester', 1)
        if current_semester >= 7:
            df['domain'] = 'AI/ML' if student_data['pathway'] == 'Artificial Intelligence' else 'Web Development'
            df['technologies_used'] = 'Python,JavaScript,SQL'
            df['outcome'] = 'Completed' if current_semester >= 8 else 'In Progress'
        else:
            df['domain'] = None
            df['technologies_used'] = None
            df['outcome'] = None
    df['placed'] = False
    df['starting_salary_lkr'] = None
    df['company_name'] = None
    df['job_title'] = None
    df['employment_type'] = None
    return df

# Helper: Apply feature engineering
def apply_feature_engineering(df, feature_engineer):
    return feature_engineer.engineer_features(df)

# Helper: Prepare for model
def prepare_for_model(df, feature_engineer):
    if hasattr(feature_engineer, 'selected_features') and feature_engineer.selected_features:
        required_features = feature_engineer.selected_features
    else:
        required_features = [
            'cumulative_gpa', 'z_score_AL', 'age_at_enrollment', 'years_since_intake',
            'internship_count', 'project_count', 'certification_count',
            'experience_score', 'high_achiever', 'has_internship'
        ]
    for feature in required_features:
        if feature not in df.columns:
            if 'count' in feature or 'score' in feature:
                df[feature] = 0
            elif 'gpa' in feature.lower():
                df[feature] = 3.0
            elif feature.startswith('has_') or feature.startswith('is_'):
                df[feature] = 0
            else:
                df[feature] = 0
    available_features = [f for f in required_features if f in df.columns]
    model_ready_df = df[available_features].copy()
    if hasattr(feature_engineer, 'label_encoders'):
        for col, encoder in feature_engineer.label_encoders.items():
            if col in model_ready_df.columns:
                try:
                    model_ready_df[col] = encoder.transform(model_ready_df[col].astype(str))
                except ValueError:
                    model_ready_df[col] = 0
    model_ready_df = model_ready_df.fillna(0)
    return model_ready_df

# Helper: Generate insights
def generate_insights(processed_df, student_data):
    insights = []
    gpa = processed_df.get('cumulative_gpa', [3.0])[0]
    experience_score = processed_df.get('experience_score', [0])[0]
    internship_count = processed_df.get('internship_count', [0])[0]
    if gpa >= 3.5:
        insights.append("Strong academic performance will positively impact your career prospects")
    elif gpa <= 2.5:
        insights.append("Consider focusing on improving academic performance in remaining semesters")
    if experience_score >= 5:
        insights.append("Excellent practical experience profile - you're well-prepared for the job market")
    elif experience_score <= 2:
        insights.append("Consider gaining more practical experience through internships and projects")
    high_demand = ['Artificial Intelligence', 'Data Science', 'Cyber Security']
    if student_data['pathway'] in high_demand:
        insights.append("Your pathway is in high demand in the current job market")
    if internship_count == 0 and student_data['current_semester'] >= 5:
        insights.append("Consider applying for internships to gain industry experience")
    elif internship_count >= 2:
        insights.append("Multiple internships demonstrate strong industry engagement")
    return insights

# Helper: Generate recommendations
def generate_recommendations(processed_df, student_data):
    recommendations = []
    semester = student_data['current_semester']
    experience_score = processed_df.get('experience_score', [0])[0]
    cert_count = processed_df.get('certification_count', [0])[0]
    if semester <= 4:
        recommendations.append("Focus on building a strong foundation and maintaining good grades")
        recommendations.append("Start exploring internship opportunities for upcoming breaks")
    elif semester <= 6:
        recommendations.append("Apply for internships to gain practical experience")
        recommendations.append("Begin working on substantial projects in your specialization area")
    else:
        recommendations.append("Focus on completing final projects and capstone work")
        recommendations.append("Start applying for full-time positions")
    if experience_score < 3:
        recommendations.append("Participate in more coding competitions and hackathons")
        recommendations.append("Contribute to open-source projects to build your portfolio")
    if cert_count == 0:
        pathway_certs = {
            'Artificial Intelligence': ['AWS Machine Learning', 'Google Cloud ML', 'TensorFlow Developer'],
            'Data Science': ['Google Data Analytics', 'Microsoft Azure Data Scientist', 'Tableau Desktop'],
            'Cyber Security': ['CompTIA Security+', 'CISSP', 'CEH'],
            'Scientific Computing': ['MATLAB Certification', 'Python Scientific Computing'],
            'Standard': ['AWS Solutions Architect', 'Oracle Java', 'Microsoft Azure Fundamentals']
        }
        certs = pathway_certs.get(student_data['pathway'], ['Industry-relevant certifications'])
        recommendations.append(f"Consider pursuing certifications like: {', '.join(certs[:2])}")
    return recommendations

# Helper: Categorize GPA
def categorize_gpa(gpa):
    if gpa >= 3.7:
        return "Excellent"
    elif gpa >= 3.3:
        return "Good"
    elif gpa >= 2.7:
        return "Average"
    else:
        return "Below Average"

def generate_salary_growth_plot(feature_engineer, trained_model, data_loader):
    # Load and prepare data
    datasets = data_loader.load_all_datasets()
    comprehensive_df = data_loader.create_comprehensive_dataset()
    df = comprehensive_df.copy()
    # Only keep rows with current_semester and pathway
    df = df[df['current_semester'].notna()]
    # Predict salary for each student
    from job_salary_prediction.helpers import (
        create_base_features, estimate_missing_features, apply_feature_engineering, prepare_for_model
    )
    df['predicted_salary'] = None
    for idx, student in df.iterrows():
        student_data = student.to_dict()
        processed = create_base_features(student_data)
        processed = estimate_missing_features(processed, student_data)
        processed = apply_feature_engineering(processed, feature_engineer)
        processed = prepare_for_model(processed, feature_engineer)
        salary = trained_model.predict(processed)[0]
        df.at[idx, 'predicted_salary'] = salary
    # Group by current_semester and calculate average salary
    avg_salary_by_semester = df.groupby('current_semester')['predicted_salary'].mean()
    # Plot
    plt.figure(figsize=(8, 5))
    avg_salary_by_semester.plot(marker='o')
    plt.title('Average Predicted Salary Growth by Semester')
    plt.xlabel('Current Semester')
    plt.ylabel('Predicted Salary (LKR)')
    plt.grid(True)
    # Save plot to a PNG image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64 