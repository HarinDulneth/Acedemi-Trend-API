# data_loader.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Handles loading and initial processing of career prediction datasets
    """
    
    def __init__(self, data_directory: str = "."):
        """
        Initialize DataLoader with data directory path
        
        Args:
            data_directory: Path to directory containing CSV files
        """
        self.data_directory = data_directory
        self.datasets = {}
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV datasets required for the career prediction system
        
        Returns:
            Dictionary of dataset name to DataFrame mappings
        """
        # Define expected CSV files and their corresponding keys
        csv_files = {
            'students': 'kelaniya_students.csv',
            'courses': 'kelaniya_courses.csv',
            'gpa': 'kelaniya_gpa.csv',
            'internships': 'kelaniya_internships.csv',
            'capstone': 'kelaniya_capstone.csv',
            'projects': 'kelaniya_projects.csv',
            'certifications': 'kelaniya_certifications.csv',
            'placement': 'kelaniya_placement.csv',
            'industry': 'kelaniya_industry.csv',
            'industry_trends': 'kelaniya_industry_trends.csv'
        }
        
        print("Loading datasets...")
        for key, filename in csv_files.items():
            filepath = os.path.join(self.data_directory, filename)
            try:
                self.datasets[key] = pd.read_csv(filepath)
                print(f"[OK] Loaded {filename}: {self.datasets[key].shape}")
            except FileNotFoundError:
                print(f"[WARN] {filename} not found")
                self.datasets[key] = pd.DataFrame()
            except Exception as e:
                print(f"[ERROR] {filename}: {str(e)}")
                self.datasets[key] = pd.DataFrame()
        
        return self.datasets
    
    def create_comprehensive_dataset(self) -> pd.DataFrame:
        """
        Merge all datasets to create a comprehensive dataset for analysis
        
        Returns:
            Merged DataFrame with all relevant information
        """
        if not self.datasets:
            self.load_all_datasets()
        
        print("\nCreating comprehensive dataset...")
        
        # Start with students as the base dataset
        merged_df = self.datasets['students'].copy()
        print(f"Base students dataset: {merged_df.shape}")
        
        # Merge GPA data (get latest GPA for each student)
        if not self.datasets['gpa'].empty:
            latest_gpa = self.datasets['gpa'].loc[
                self.datasets['gpa'].groupby('student_id')['semester_number'].idxmax()
            ][['student_id', 'cumulative_gpa']]
            merged_df = merged_df.merge(latest_gpa, on='student_id', how='left')
            print(f"After GPA merge: {merged_df.shape}")
        
        # Aggregate internship data
        if not self.datasets['internships'].empty:
            internship_agg = self._aggregate_internships()
            merged_df = merged_df.merge(internship_agg, on='student_id', how='left')
            print(f"After internships merge: {merged_df.shape}")
        
        # Aggregate project data
        if not self.datasets['projects'].empty:
            project_agg = self._aggregate_projects()
            merged_df = merged_df.merge(project_agg, on='student_id', how='left')
            print(f"After projects merge: {merged_df.shape}")
        
        # Aggregate certification data
        if not self.datasets['certifications'].empty:
            cert_agg = self._aggregate_certifications()
            merged_df = merged_df.merge(cert_agg, on='student_id', how='left')
            print(f"After certifications merge: {merged_df.shape}")
        
        # Merge capstone data
        if not self.datasets['capstone'].empty:
            merged_df = merged_df.merge(
                self.datasets['capstone'][['student_id', 'domain', 'technologies_used', 'outcome']], 
                on='student_id', how='left'
            )
            print(f"After capstone merge: {merged_df.shape}")
        
        # Merge placement data
        if not self.datasets['placement'].empty:
            merged_df = merged_df.merge(self.datasets['placement'], on='student_id', how='left')
            print(f"After placement merge: {merged_df.shape}")
        
        print(f"Final comprehensive dataset: {merged_df.shape}")
        return merged_df
    
    def _aggregate_internships(self) -> pd.DataFrame:
        """Aggregate internship data per student"""
        internship_df = self.datasets['internships'].copy()
        
        # Convert dates
        internship_df['start_date'] = pd.to_datetime(internship_df['start_date'])
        internship_df['end_date'] = pd.to_datetime(internship_df['end_date'])
        internship_df['internship_duration'] = (
            internship_df['end_date'] - internship_df['start_date']
        ).dt.days
        
        # Aggregate by student
        agg_data = internship_df.groupby('student_id').agg({
            'employer': 'count',  # Count of internships
            'performance_rating': 'mean',  # Average rating
            'internship_duration': ['sum', 'mean']  # Total and average duration
        }).round(2)
        
        # Flatten column names
        agg_data.columns = [
            'internship_count', 'avg_internship_rating', 
            'total_internship_days', 'avg_internship_duration'
        ]
        
        return agg_data.reset_index()
    
    def _aggregate_projects(self) -> pd.DataFrame:
        """Aggregate project data per student"""
        project_df = self.datasets['projects'].copy()
        
        # Convert dates and calculate duration
        project_df['start_date'] = pd.to_datetime(project_df['start_date'])
        project_df['end_date'] = pd.to_datetime(project_df['end_date'])
        project_df['project_duration'] = (
            project_df['end_date'] - project_df['start_date']
        ).dt.days
        
        # Count technologies per project
        project_df['tech_count'] = project_df['technologies_used'].str.split(',').str.len()
        
        # Aggregate by student
        agg_data = project_df.groupby('student_id').agg({
            'project_id': 'count',  # Total projects
            'project_duration': 'mean',  # Average duration
            'tech_count': 'sum',  # Total technologies used
            'outcome': lambda x: (x == 'Completed').sum()  # Completed projects
        }).round(2)
        
        # Flatten column names
        agg_data.columns = [
            'project_count', 'avg_project_duration', 
            'total_technologies', 'completed_projects'
        ]
        
        return agg_data.reset_index()
    
    def _aggregate_certifications(self) -> pd.DataFrame:
        """Aggregate certification data per student"""
        cert_df = self.datasets['certifications'].copy()
        
        # Count certifications per student
        cert_agg = cert_df.groupby('student_id').size().reset_index(name='certification_count')
        
        return cert_agg
    
    def get_dataset_info(self) -> None:
        """Print information about loaded datasets"""
        if not self.datasets:
            print("No datasets loaded. Call load_all_datasets() first.")
            return
        
        print("\nDataset Information:")
        print("=" * 50)
        for name, df in self.datasets.items():
            if not df.empty:
                print(f"{name.upper()}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  Columns: {list(df.columns)}")
            else:
                print(f"{name.upper()}: Empty dataset")
            print("-" * 30)

