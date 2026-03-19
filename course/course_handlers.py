import pandas as pd
import numpy as np
from pathlib import Path
import os

def run_course_enrollment_prediction():
    """Run course enrollment prediction and return detailed results"""
    try:
        # Load existing predictions and return detailed view
        result = load_existing_predictions()
        result["view_type"] = "detailed"
        result["description"] = "Complete course enrollment predictions with all records"
        return result
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running course enrollment prediction: {str(e)}"
        }

def load_course_enrollment_summary():
    """Load course enrollment prediction summary statistics"""
    try:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        predictions_path = Path(current_dir) / 'course_enrollment_prediction' / 'data' / 'processed' / 'predictions.csv'
        
        if not predictions_path.exists():
            return {
                "status": "error",
                "message": f"Predictions file not found at: {predictions_path}"
            }
        
        predictions_df = pd.read_csv(predictions_path)
        
        # Helper function to convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Calculate summary statistics
        summary_stats = {
            "total_predictions": int(len(predictions_df)),
            "unique_universities": int(len(predictions_df['university'].unique())),
            "unique_courses": int(len(predictions_df['course_name'].unique())),
            "models_used": [str(x) for x in predictions_df['model'].unique()],
            "years_predicted": [int(x) for x in sorted(predictions_df['year'].unique())],
            "avg_enrollment_pred": float(predictions_df['enrollments_pred'].mean()) if 'enrollments_pred' in predictions_df.columns else 0.0,
            "avg_application_pred": float(predictions_df['applications_pred'].mean()) if 'applications_pred' in predictions_df.columns else 0.0,
            "max_enrollment_pred": float(predictions_df['enrollments_pred'].max()) if 'enrollments_pred' in predictions_df.columns else 0.0,
            "min_enrollment_pred": float(predictions_df['enrollments_pred'].min()) if 'enrollments_pred' in predictions_df.columns else 0.0
        }
        
        # Top universities by average enrollment
        top_universities = predictions_df.groupby('university')['enrollments_pred'].mean().sort_values(ascending=False).head(10)
        top_universities_dict = {str(uni): float(avg) for uni, avg in top_universities.items()}
        
        # Top courses by average enrollment
        top_courses = predictions_df.groupby('course_name')['enrollments_pred'].mean().sort_values(ascending=False).head(10)
        top_courses_dict = {str(course): float(avg) for course, avg in top_courses.items()}
        
        # Model performance summary - convert MultiIndex to proper format
        model_summary = predictions_df.groupby('model').agg({
            'enrollments_pred': ['mean', 'count']
        }).round(2)
        
        # Convert MultiIndex DataFrame to JSON-serializable format
        model_summary_dict = {}
        for model in model_summary.index:
            model_summary_dict[str(model)] = {
                'mean_enrollment': float(model_summary.loc[model, ('enrollments_pred', 'mean')]),
                'prediction_count': int(model_summary.loc[model, ('enrollments_pred', 'count')])
            }
        
        result = {
            "status": "success",
            "message": "Course enrollment prediction summary loaded successfully",
            "view_type": "summary",
            "description": "High-level statistics and overview of course enrollment predictions",
            "summary_statistics": summary_stats,
            "top_universities_by_enrollment": top_universities_dict,
            "top_courses_by_enrollment": top_courses_dict,
            "model_performance_summary": model_summary_dict,
            "source": "pre-generated predictions.csv"
        }
        
        # Convert any remaining numpy types
        result = convert_numpy_types(result)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading course enrollment summary: {str(e)}"
        }

def load_filtered_course_predictions(year=None, university=None, course=None, model=None):
    """Load filtered course enrollment predictions based on criteria"""
    try:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        predictions_path = Path(current_dir) / 'course_enrollment_prediction' / 'data' / 'processed' / 'predictions.csv'
        
        if not predictions_path.exists():
            return {
                "status": "error",
                "message": f"Predictions file not found at: {predictions_path}"
            }
        
        predictions_df = pd.read_csv(predictions_path)
        
        # Apply filters
        filtered_df = predictions_df.copy()
        
        if year is not None:
            filtered_df = filtered_df[filtered_df['year'] == int(year)]
        
        if university is not None:
            filtered_df = filtered_df[filtered_df['university'].str.contains(university, case=False, na=False)]
        
        if course is not None:
            filtered_df = filtered_df[filtered_df['course_name'].str.lower() == course.lower()]
        
        if model is not None:
            filtered_df = filtered_df[filtered_df['model'].str.contains(model, case=False, na=False)]
        
        # Convert to JSON-serializable format
        filtered_predictions = []
        for _, row in filtered_df.iterrows():
            filtered_predictions.append({
                'year': int(row['year']),
                'university': str(row['university']),
                'course_name': str(row['course_name']),
                'enrollments_pred': float(row.get('enrollments_pred', 0)) if pd.notna(row.get('enrollments_pred')) else 0.0,
                'applications_pred': float(row.get('applications_pred', 0)) if pd.notna(row.get('applications_pred')) else 0.0,
                'model': str(row['model'])
            })
        
        return {
            "status": "success",
            "message": f"Filtered course enrollment predictions loaded successfully",
            "view_type": "filtered",
            "description": f"Course enrollment predictions filtered by criteria",
            "filters_applied": {
                "year": year,
                "university": university,
                "course": course,
                "model": model
            },
            "predictions": filtered_predictions,
            "total_filtered_records": len(filtered_predictions),
            "source": "pre-generated predictions.csv"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading filtered course enrollment predictions: {str(e)}"
        }

def load_existing_predictions():
    """Load existing course enrollment predictions from CSV file"""
    try:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        predictions_path = Path(current_dir) / 'course_enrollment_prediction' / 'data' / 'processed' / 'predictions.csv'
        
        if not predictions_path.exists():
            return {
                "status": "error",
                "message": f"Predictions file not found at: {predictions_path}"
            }
        
        predictions_df = pd.read_csv(predictions_path)
        
        # Convert predictions to JSON-serializable format
        predictions_json = []
        for _, row in predictions_df.iterrows():
            predictions_json.append({
                'year': int(row['year']),
                'university': str(row['university']),
                'course_name': str(row['course_name']),
                'enrollments_pred': float(row.get('enrollments_pred', 0)) if pd.notna(row.get('enrollments_pred')) else 0.0,
                'applications_pred': float(row.get('applications_pred', 0)) if pd.notna(row.get('applications_pred')) else 0.0,
                'model': str(row['model'])
            })
        
        return {
            "status": "success",
            "message": "Existing course enrollment predictions loaded successfully",
            "predictions": predictions_json,
            "total_predictions": len(predictions_json),
            "models_used": [str(x) for x in predictions_df['model'].unique()],
            "years_predicted": [int(x) for x in sorted(predictions_df['year'].unique())],
            "source": "pre-generated predictions.csv"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading existing predictions: {str(e)}"
        }

def load_course_historical_data():
    """Load historical enrollments and applications data from raw data files"""
    try:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_data_path = Path(current_dir) / 'course_enrollment_prediction' / 'data' / 'raw'
        
        # Load enrollments data
        enrollments_path = raw_data_path / 'enrollments.csv'
        if not enrollments_path.exists():
            return {
                "status": "error",
                "message": f"Enrollments file not found at: {enrollments_path}"
            }
        
        enrollments_df = pd.read_csv(enrollments_path)
        
        # Load applications data (combine both files)
        applications_2016_2023_path = raw_data_path / 'Application_2016-2023.csv'
        applications_2005_2015_path = raw_data_path / 'Application_2005-2015.csv'
        
        applications_df = None
        
        if applications_2016_2023_path.exists():
            applications_df = pd.read_csv(applications_2016_2023_path)
        
        if applications_2005_2015_path.exists():
            if applications_df is not None:
                # Combine both application files
                applications_2005_2015 = pd.read_csv(applications_2005_2015_path)
                applications_df = pd.concat([applications_2005_2015, applications_df], ignore_index=True)
            else:
                applications_df = pd.read_csv(applications_2005_2015_path)
        
        # Convert enrollments data to JSON-serializable format
        enrollments_data = []
        for _, row in enrollments_df.iterrows():
            enrollments_data.append({
                'university': str(row['university']),
                'course_name': str(row['course_name']),
                'year': int(row['year']),
                'enrollments': int(row['enrollments']),
                'avg_start_sal': float(row['avg_start_sal']) if pd.notna(row['avg_start_sal']) else None,
                'graduate_employment_rate': float(row['graduate_employment_rate']) if pd.notna(row['graduate_employment_rate']) else None
            })
        
        # Convert applications data to JSON-serializable format
        applications_data = []
        if applications_df is not None:
            for _, row in applications_df.iterrows():
                applications_data.append({
                    'university': str(row['university']),
                    'course_name': str(row['course_name']),
                    'district': str(row['district']) if 'district' in row and pd.notna(row['district']) else None,
                    'year': int(row['year']),
                    'applications': int(row['applications']) if pd.notna(row['applications']) else 0,
                    'cutoff_mark': float(row['cutoff_mark']) if 'cutoff_mark' in row and pd.notna(row['cutoff_mark']) else None
                })
        
        # Calculate summary statistics
        summary_stats = {
            "enrollments": {
                "total_records": len(enrollments_data),
                "unique_universities": int(len(enrollments_df['university'].unique())),
                "unique_courses": int(len(enrollments_df['course_name'].unique())),
                "years_covered": [int(x) for x in sorted(enrollments_df['year'].unique())],
                "total_enrollments": int(enrollments_df['enrollments'].sum()),
                "avg_enrollments_per_year": float(enrollments_df.groupby('year')['enrollments'].sum().mean())
            },
            "applications": {
                "total_records": len(applications_data),
                "unique_universities": int(len(applications_df['university'].unique())) if applications_df is not None else 0,
                "unique_courses": int(len(applications_df['course_name'].unique())) if applications_df is not None else 0,
                "years_covered": [int(x) for x in sorted(applications_df['year'].unique())] if applications_df is not None else [],
                "total_applications": int(applications_df['applications'].sum()) if applications_df is not None else 0,
                "avg_applications_per_year": float(applications_df.groupby('year')['applications'].sum().mean()) if applications_df is not None else 0
            }
        }
        
        # Top universities by total enrollments
        top_universities_enrollments = enrollments_df.groupby('university')['enrollments'].sum().sort_values(ascending=False).head(10)
        top_universities_enrollments_dict = {str(uni): int(total) for uni, total in top_universities_enrollments.items()}
        
        # Top courses by total enrollments
        top_courses_enrollments = enrollments_df.groupby('course_name')['enrollments'].sum().sort_values(ascending=False).head(10)
        top_courses_enrollments_dict = {str(course): int(total) for course, total in top_courses_enrollments.items()}
        
        # Top universities by total applications (if available)
        top_universities_applications = {}
        top_courses_applications = {}
        if applications_df is not None:
            top_universities_applications_data = applications_df.groupby('university')['applications'].sum().sort_values(ascending=False).head(10)
            top_universities_applications = {str(uni): int(total) for uni, total in top_universities_applications_data.items()}
            
            top_courses_applications_data = applications_df.groupby('course_name')['applications'].sum().sort_values(ascending=False).head(10)
            top_courses_applications = {str(course): int(total) for course, total in top_courses_applications_data.items()}
        
        return {
            "status": "success",
            "message": "Historical course enrollment and application data loaded successfully",
            "system": "University Course Enrollment Prediction System",
            "data_source": "Raw data files from course_enrollment_prediction/data/raw/",
            "summary_statistics": summary_stats,
            "top_universities_by_enrollments": top_universities_enrollments_dict,
            "top_courses_by_enrollments": top_courses_enrollments_dict,
            "top_universities_by_applications": top_universities_applications,
            "top_courses_by_applications": top_courses_applications,
            "enrollments_data": enrollments_data,
            "applications_data": applications_data,
            "files_loaded": {
                "enrollments": "enrollments.csv",
                "applications_2016_2023": "Application_2016-2023.csv" if applications_2016_2023_path.exists() else None,
                "applications_2005_2015": "Application_2005-2015.csv" if applications_2005_2015_path.exists() else None
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading historical course data: {str(e)}"
        } 

def run_course_enrollment_prediction_with_years(forecast_years):
    """Run course enrollment prediction for a user-specified number of years."""
    try:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        predictions_path = Path(current_dir) / 'course_enrollment_prediction' / 'data' / 'processed' / 'predictions.csv'
        if not predictions_path.exists():
            return {
                "status": "error",
                "message": f"Predictions file not found at: {predictions_path}"
            }
        predictions_df = pd.read_csv(predictions_path)
        # Only keep the last N years for each university/course/model
        max_year = predictions_df['year'].max()
        min_year = max_year - forecast_years + 1
        filtered_df = predictions_df[predictions_df['year'] >= min_year]
        # Convert to JSON-serializable format
        predictions_json = []
        for _, row in filtered_df.iterrows():
            predictions_json.append({
                'year': int(row['year']),
                'university': str(row['university']),
                'course_name': str(row['course_name']),
                'enrollments_pred': float(row.get('enrollments_pred', 0)) if pd.notna(row.get('enrollments_pred')) else 0.0,
                'applications_pred': float(row.get('applications_pred', 0)) if pd.notna(row.get('applications_pred')) else 0.0,
                'model': str(row['model'])
            })
        return {
            "status": "success",
            "message": f"Course enrollment predictions for last {forecast_years} years loaded successfully",
            "forecast_years": forecast_years,
            "predictions": predictions_json,
            "total_predictions": len(predictions_json),
            "models_used": [str(x) for x in filtered_df['model'].unique()],
            "years_predicted": [int(x) for x in sorted(filtered_df['year'].unique())],
            "source": "pre-generated predictions.csv"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running course enrollment prediction for years: {str(e)}"
        }

def load_filtered_course_predictions_with_years(forecast_years, year=None, university=None, course=None, model=None):
    """Load filtered course enrollment predictions for a user-specified number of years and filters."""
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        predictions_path = Path(current_dir) / 'course_enrollment_prediction' / 'data' / 'processed' / 'predictions.csv'
        if not predictions_path.exists():
            return {
                "status": "error",
                "message": f"Predictions file not found at: {predictions_path}"
            }
        predictions_df = pd.read_csv(predictions_path)
        # Only keep the last N years
        max_year = predictions_df['year'].max()
        min_year = max_year - forecast_years + 1
        filtered_df = predictions_df[predictions_df['year'] >= min_year]
        # Apply additional filters
        if year is not None:
            filtered_df = filtered_df[filtered_df['year'] == int(year)]
        if university is not None:
            filtered_df = filtered_df[filtered_df['university'].str.contains(university, case=False, na=False)]
        if course is not None:
            filtered_df = filtered_df[filtered_df['course_name'].str.lower() == course.lower()]
        if model is not None:
            filtered_df = filtered_df[filtered_df['model'].str.contains(model, case=False, na=False)]
        # Convert to JSON-serializable format
        filtered_predictions = []
        for _, row in filtered_df.iterrows():
            filtered_predictions.append({
                'year': int(row['year']),
                'university': str(row['university']),
                'course_name': str(row['course_name']),
                'enrollments_pred': float(row.get('enrollments_pred', 0)) if pd.notna(row.get('enrollments_pred')) else 0.0,
                'applications_pred': float(row.get('applications_pred', 0)) if pd.notna(row.get('applications_pred')) else 0.0,
                'model': str(row['model'])
            })
        return {
            "status": "success",
            "message": f"Filtered course enrollment predictions for last {forecast_years} years loaded successfully",
            "forecast_years": forecast_years,
            "filters_applied": {
                "year": year,
                "university": university,
                "course": course,
                "model": model
            },
            "predictions": filtered_predictions,
            "total_filtered_records": len(filtered_predictions),
            "source": "pre-generated predictions.csv"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading filtered course enrollment predictions for years: {str(e)}"
        }