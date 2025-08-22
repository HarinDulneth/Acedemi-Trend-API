# student_input_handler.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
from datetime import datetime, date
warnings.filterwarnings('ignore')

from .helpers import (
    create_base_features,
    estimate_missing_features,
    apply_feature_engineering,
    prepare_for_model,
    generate_insights,
    generate_recommendations,
    categorize_gpa
)

class StudentInputHandler:
    """
    Handles input data from current students and prepares it for prediction
    """
    
    def __init__(self, feature_engineer, trained_model):
        """
        Initialize with trained feature engineer and model
        
        Args:
            feature_engineer: Trained FeatureEngineer instance
            trained_model: Trained CareerPredictionModel instance
        """
        self.feature_engineer = feature_engineer
        self.trained_model = trained_model
        self.required_fields = self._define_required_fields()
        self.optional_fields = self._define_optional_fields()
        
    def _define_required_fields(self) -> Dict[str, Dict[str, Any]]:
        """Define required fields for student input"""
        return {
            'student_id': {
                'type': str,
                'description': 'Unique student identifier'
            },
            'gender': {
                'type': str,
                'options': ['Male', 'Female'],
                'description': 'Student gender'
            },
            'age_at_enrollment': {
                'type': int,
                'min': 17,
                'max': 30,
                'description': 'Age when enrolled at university'
            },
            'province': {
                'type': str,
                'options': ['Western', 'Central', 'Southern', 'Northern', 'Eastern', 
                           'North Western', 'North Central', 'Uva', 'Sabaragamuwa'],
                'description': 'Province of origin'
            },
            'district': {
                'type': str,
                'description': 'District of origin'
            },
            'z_score_AL': {
                'type': float,
                'min': 1.0,
                'max': 2.5,
                'description': 'A/L Z-score (1.0 to 2.5)'
            },
            'pathway': {
                'type': str,
                'options': ['Artificial Intelligence', 'Data Science', 'Cyber Security', 
                           'Scientific Computing', 'Standard'],
                'description': 'Academic pathway/specialization'
            },
            'intake_year': {
                'type': int,
                'min': 2018,
                'max': 2025,
                'description': 'Year of university intake'
            },
            'current_semester': {
                'type': int,
                'min': 1,
                'max': 8,
                'description': 'Current semester (1-8)'
            }
        }
    
    def _define_optional_fields(self) -> Dict[str, Dict[str, Any]]:
        """Define optional fields that enhance prediction accuracy"""
        return {
            'current_gpa': {
                'type': float,
                'min': 0.0,
                'max': 4.0,
                'description': 'Current cumulative GPA (if available)'
            },
            'completed_internships': {
                'type': int,
                'min': 0,
                'max': 10,
                'description': 'Number of completed internships'
            },
            'internship_ratings': {
                'type': list,
                'description': 'List of internship performance ratings (1-5)'
            },
            'total_internship_months': {
                'type': int,
                'min': 0,
                'max': 24,
                'description': 'Total months of internship experience'
            },
            'completed_projects': {
                'type': int,
                'min': 0,
                'max': 20,
                'description': 'Number of completed projects'
            },
            'project_technologies': {
                'type': list,
                'description': 'List of technologies used in projects'
            },
            'certifications_earned': {
                'type': int,
                'min': 0,
                'max': 15,
                'description': 'Number of professional certifications'
            },
            'capstone_domain': {
                'type': str,
                'options': ['AI/ML', 'Web Development', 'Mobile Development', 
                           'Data Analytics', 'Cybersecurity', 'IoT', 'Other'],
                'description': 'Capstone project domain (if completed)'
            },
            'leadership_roles': {
                'type': int,
                'min': 0,
                'max': 5,
                'description': 'Number of leadership positions held'
            },
            'extracurricular_activities': {
                'type': int,
                'min': 0,
                'max': 10,
                'description': 'Number of extracurricular activities'
            }
        }
    
    def get_input_form_schema(self) -> Dict[str, Any]:
        """
        Generate a schema for creating input forms
        
        Returns:
            Dictionary containing field definitions for UI generation
        """
        return {
            'required_fields': self.required_fields,
            'optional_fields': self.optional_fields,
            'form_title': 'Student Career Prediction Input',
            'form_description': 'Provide your current academic and experience information for career outcome prediction'
        }
    
    def validate_input(self, student_data: Dict[str, Any]) -> Tuple[bool, list[str]]:
        """
        Validate student input data
        
        Args:
            student_data: Dictionary containing student information
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field_name, field_config in self.required_fields.items():
            if field_name not in student_data:
                errors.append(f"Required field '{field_name}' is missing")
                continue
                
            value = student_data[field_name]
            
            # Type validation
            if field_config['type'] == int and not isinstance(value, int):
                try:
                    student_data[field_name] = int(value)
                except ValueError:
                    errors.append(f"Field '{field_name}' must be an integer")
                    continue
            elif field_config['type'] == float and not isinstance(value, (int, float)):
                try:
                    student_data[field_name] = float(value)
                except ValueError:
                    errors.append(f"Field '{field_name}' must be a number")
                    continue
            
            # Range validation
            if 'min' in field_config and value < field_config['min']:
                errors.append(f"Field '{field_name}' must be at least {field_config['min']}")
            if 'max' in field_config and value > field_config['max']:
                errors.append(f"Field '{field_name}' must be at most {field_config['max']}")
            
            # Options validation
            if 'options' in field_config and value not in field_config['options']:
                errors.append(f"Field '{field_name}' must be one of: {field_config['options']}")
        
        # Validate optional fields if provided
        for field_name, field_config in self.optional_fields.items():
            if field_name in student_data:
                value = student_data[field_name]
                
                if field_config['type'] == int and not isinstance(value, int):
                    try:
                        student_data[field_name] = int(value)
                    except ValueError:
                        errors.append(f"Optional field '{field_name}' must be an integer")
                elif field_config['type'] == float and not isinstance(value, (int, float)):
                    try:
                        student_data[field_name] = float(value)
                    except ValueError:
                        errors.append(f"Optional field '{field_name}' must be a number")
        
        return len(errors) == 0, errors
    
    def process_student_input(self, student_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process and transform student input into model-ready format
        
        Args:
            student_data: Raw student input data
            
        Returns:
            DataFrame ready for model prediction
        """
        is_valid, errors = self.validate_input(student_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")
        processed_data = create_base_features(student_data)
        processed_data = estimate_missing_features(processed_data, student_data)
        processed_data = apply_feature_engineering(processed_data, self.feature_engineer)
        processed_data = prepare_for_model(processed_data, self.feature_engineer)
        return processed_data
    
    def predict_career_outcomes(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict career outcomes for a student
        
        Args:
            student_data: Student input data
            
        Returns:
            Dictionary containing predictions and insights
        """
        try:
            processed_df = self.process_student_input(student_data)
            
            # Make prediction
            salary_prediction = self.trained_model.predict(processed_df)[0]
            
            # Calculate confidence intervals (simplified approach)
            base_uncertainty = 50000  # Base uncertainty in LKR
            experience_factor = processed_df.get('experience_score', [0])[0]
            gpa_factor = processed_df.get('cumulative_gpa', [3.0])[0]
            
            # Lower uncertainty for students with more experience and higher GPA
            uncertainty = base_uncertainty * (1.2 - 0.1 * experience_factor - 0.1 * gpa_factor)
            uncertainty = max(uncertainty, 20000)  # Minimum uncertainty
            
            # Generate insights and recommendations
            insights = generate_insights(processed_df, student_data)
            recommendations = generate_recommendations(processed_df, student_data)
            
            return {
                'predicted_salary': {
                    'amount': float(salary_prediction),
                    'currency': 'LKR',
                    'confidence_interval': {
                        'lower': float(salary_prediction - uncertainty),
                        'upper': float(salary_prediction + uncertainty)
                    }
                },
                'insights': insights,
                'recommendations': recommendations,
                'student_profile': {
                    'experience_score': float(processed_df.get('experience_score', [0])[0]),
                    'academic_performance': categorize_gpa(processed_df.get('cumulative_gpa', [3.0])[0]),
                    'pathway': student_data['pathway'],
                    'completion_status': f"{student_data['current_semester']}/8 semesters"
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'message': 'Unable to generate prediction. Please check your input data.'
            }


class StudentPredictionAPI:
    """
    API wrapper for student career prediction
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the API
        
        Args:
            model_path: Path to saved model files
        """
        self.input_handler = None
        self.model_loaded = False
        
    def load_model(self, feature_engineer, trained_model):
        """
        Load trained model and feature engineer
        
        Args:
            feature_engineer: Trained FeatureEngineer instance
            trained_model: Trained CareerPredictionModel instance
        """
        self.input_handler = StudentInputHandler(feature_engineer, trained_model)
        self.model_loaded = True
        
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input form schema for UI generation"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        return self.input_handler.get_input_form_schema()
    
    def predict(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for student
        
        Args:
            student_data: Student input data
            
        Returns:
            Prediction results with insights and recommendations
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        return self.input_handler.predict_career_outcomes(student_data)


def filter_job_salary_predictions(feature_engineer, trained_model, filters: dict):
    import pandas as pd
    from job_salary_prediction.data_loader import DataLoader
    from job_salary_prediction.helpers import (
        create_base_features, estimate_missing_features, apply_feature_engineering, prepare_for_model
    )
    # Load and prepare data
    data_loader = DataLoader(data_directory='job_salary_prediction')
    datasets = data_loader.load_all_datasets()
    comprehensive_df = data_loader.create_comprehensive_dataset()
    df = comprehensive_df.copy()
    # Apply filters
    if 'pathway' in filters and filters['pathway']:
        df = df[df['pathway'] == filters['pathway']]
    if 'min_gpa' in filters:
        df = df[df['cumulative_gpa'] >= float(filters['min_gpa'])]
    if 'max_gpa' in filters:
        df = df[df['cumulative_gpa'] <= float(filters['max_gpa'])]
    results = []
    for _, student in df.iterrows():
        student_data = student.to_dict()
        processed = create_base_features(student_data)
        processed = estimate_missing_features(processed, student_data)
        processed = apply_feature_engineering(processed, feature_engineer)
        processed = prepare_for_model(processed, feature_engineer)
        salary = trained_model.predict(processed)[0]
        results.append({
            'student_id': student_data.get('student_id'),
            'pathway': student_data.get('pathway'),
            'cumulative_gpa': student_data.get('cumulative_gpa'),
            'predicted_salary': salary
        })
    return results

# Example usage function
def example_usage():
    """
    Example of how to use the StudentInputHandler
    """
    # Example student data
    sample_student_data = {
        'student_id': 'CS2021001',
        'gender': 'Male',
        'age_at_enrollment': 19,
        'province': 'Western',
        'district': 'Colombo',
        'z_score_AL': 1.85,
        'pathway': 'Artificial Intelligence',
        'intake_year': 2021,
        'current_semester': 6,
        'current_gpa': 3.2,
        'completed_internships': 1,
        'internship_ratings': [4.5],
        'total_internship_months': 3,
        'completed_projects': 4,
        'project_technologies': ['Python', 'TensorFlow', 'React', 'SQL'],
        'certifications_earned': 1
    }
    
    print("Sample Student Input:")
    print("=" * 50)
    for key, value in sample_student_data.items():
        print(f"{key}: {value}")
    
    print("\nThis data would be processed and used for prediction")
    print("The system will estimate missing features and generate predictions")

if __name__ == "__main__":
    example_usage()