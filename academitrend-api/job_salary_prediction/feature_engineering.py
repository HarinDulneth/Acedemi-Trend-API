import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Advanced feature engineering for career prediction system
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.feature_importance = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features from raw data
        
        Args:
            df: Raw merged dataset
            
        Returns:
            DataFrame with engineered features
        """
        engineered_df = df.copy()
        
        # 1. Academic Performance Features
        engineered_df = self._create_academic_features(engineered_df)
        
        # 2. Demographic and Geographic Features
        engineered_df = self._create_demographic_features(engineered_df)
        
        # 3. Experience and Skills Features
        engineered_df = self._create_experience_features(engineered_df)
        
        # 4. Pathway-specific Features
        engineered_df = self._create_pathway_features(engineered_df)
        
        # 5. Interaction Features
        engineered_df = self._create_interaction_features(engineered_df)
        
        # 6. Time-based Features
        engineered_df = self._create_time_features(engineered_df)
        
        return engineered_df
    
    def _create_academic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create academic performance related features"""
        
        # GPA categories
        df['gpa_category'] = pd.cut(df['cumulative_gpa'], 
                                   bins=[0, 2.5, 3.0, 3.5, 4.0], 
                                   labels=['Below_Average', 'Average', 'Good', 'Excellent'])
        
        # Academic excellence indicators
        df['high_achiever'] = (df['cumulative_gpa'] >= 3.5).astype(int)
        df['academic_struggle'] = (df['cumulative_gpa'] <= 2.5).astype(int)
        
        # A/L performance indicators
        df['al_excellence'] = (df['z_score_AL'] >= 1.8).astype(int)
        df['al_performance_category'] = pd.cut(df['z_score_AL'], 
                                              bins=[1.0, 1.6, 1.8, 2.0], 
                                              labels=['Good', 'Very_Good', 'Excellent'])
        
        # Academic consistency (difference between AL and university performance)
        df['performance_consistency'] = df['cumulative_gpa'] / 2.0 - (df['z_score_AL'] - 1.45) / 0.55
        df['improved_performance'] = (df['performance_consistency'] > 0).astype(int)
        
        return df
    
    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic and geographic features"""
        
        # Age-related features
        df['mature_student'] = (df['age_at_enrollment'] >= 21).astype(int)
        df['traditional_age'] = (df['age_at_enrollment'] <= 19).astype(int)
        
        # Geographic advantage features
        western_provinces = ['Western', 'Southern', 'Central']
        df['from_developed_province'] = df['province'].isin(western_provinces).astype(int)
        
        # Urban vs rural (based on district populations and development)
        major_districts = ['Colombo', 'Gampaha', 'Kandy', 'Galle', 'Kurunegala']
        df['from_major_district'] = df['district'].isin(major_districts).astype(int)
        
        # Distance-based features (from data generation logic)
        close_districts = ['Colombo', 'Gampaha', 'Kalutara']
        df['close_to_university'] = df['district'].isin(close_districts).astype(int)
        
        return df
    
    def _create_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create experience and practical skills features"""
        
        # Fill missing values for aggregated features
        experience_cols = ['internship_count', 'project_count', 'certification_count',
                          'avg_internship_rating', 'total_internship_days', 
                          'avg_project_duration', 'total_technologies', 'completed_projects']
        
        for col in experience_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Experience level indicators
        df['has_internship'] = (df.get('internship_count', 0) > 0).astype(int)
        df['multiple_internships'] = (df.get('internship_count', 0) > 1).astype(int)
        df['excellent_intern'] = (df.get('avg_internship_rating', 0) >= 4.5).astype(int)
        
        # Project engagement features
        df['active_project_developer'] = (df.get('project_count', 0) >= 3).astype(int)
        df['project_completion_rate'] = np.where(df.get('project_count', 0) > 0,
                                                df.get('completed_projects', 0) / df.get('project_count', 1), 0)
        df['high_project_completer'] = (df['project_completion_rate'] >= 0.8).astype(int)
        
        # Technical diversity
        df['tech_diversity'] = df.get('total_technologies', 0) / (df.get('project_count', 1) + 1)
        df['diverse_tech_skills'] = (df['tech_diversity'] >= 3).astype(int)
        
        # Certification value
        df['certified_professional'] = (df.get('certification_count', 0) > 0).astype(int)
        df['highly_certified'] = (df.get('certification_count', 0) >= 2).astype(int)
        
        # Overall experience score
        df['experience_score'] = (
            df['has_internship'] * 2 +
            df['multiple_internships'] * 1 +
            df['excellent_intern'] * 2 +
            df['active_project_developer'] * 2 +
            df['high_project_completer'] * 1 +
            df['diverse_tech_skills'] * 1 +
            df['certified_professional'] * 1
        )
        
        return df
    
    def _create_pathway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pathway-specific features"""
        
        # Pathway demand indicators (based on industry trends)
        high_demand_pathways = ['Artificial Intelligence', 'Data Science', 'Cyber Security']
        df['high_demand_pathway'] = df['pathway'].isin(high_demand_pathways).astype(int)
        
        # Pathway-specific skill alignment
        pathway_tech_alignment = {
            'Artificial Intelligence': ['Python', 'TensorFlow', 'PyTorch', 'Machine Learning'],
            'Data Science': ['Python', 'Pandas', 'R', 'SQL'],
            'Cyber Security': ['Python', 'Wireshark', 'Kali Linux', 'Security'],
            'Scientific Computing': ['MATLAB', 'Python', 'NumPy'],
            'Standard': ['Java', 'Spring', 'React', 'SQL']
        }
        
        # Calculate alignment score (this would need the actual technologies from projects)
        df['pathway_alignment_score'] = 1.0  # Placeholder - would calculate based on actual tech match
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        
        # Academic-Experience interactions
        df['gpa_experience_interaction'] = df['cumulative_gpa'] * df['experience_score']
        df['al_gpa_interaction'] = df['z_score_AL'] * df['cumulative_gpa']
        
        # Geographic-Academic interactions
        df['location_advantage_academic'] = df['from_developed_province'] * df['cumulative_gpa']
        df['urban_academic_advantage'] = df['from_major_district'] * df['high_achiever']
        
        # Pathway-Performance interactions
        df['pathway_performance'] = df['high_demand_pathway'] * df['cumulative_gpa']
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # Years since intake effects
        df['recent_graduate'] = (df['years_since_intake'] <= 1).astype(int)
        df['experienced_graduate'] = (df['years_since_intake'] >= 4).astype(int)
        
        # Intake year effects (market conditions)
        df['post_covid_intake'] = (df['intake_year'] >= 2021).astype(int)
        df['pre_covid_intake'] = (df['intake_year'] <= 2019).astype(int)
        
        return df
    
    def prepare_features_for_modeling(self, df: pd.DataFrame, target_column: str = 'starting_salary_lkr') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning modeling
        
        Args:
            df: DataFrame with engineered features
            target_column: Name of target variable
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Filter to placed students only for salary prediction
        if 'placed' in df.columns:
            modeling_df = df[df['placed'] == True].copy()
        else:
            modeling_df = df.copy()
        
        # Handle target variable
        if target_column in modeling_df.columns:
            target = modeling_df[target_column].copy()
            # Remove rows with missing target
            valid_indices = target.notna()
            modeling_df = modeling_df[valid_indices]
            target = target[valid_indices]
        else:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Select features for modeling
        features_df = self._select_modeling_features(modeling_df)
        
        # Encode categorical variables
        features_df = self._encode_categorical_features(features_df)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        return features_df, target
    
    def _select_modeling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for modeling"""
        
        # Define feature categories
        academic_features = [
            'cumulative_gpa', 'z_score_AL', 'high_achiever', 'academic_struggle',
            'al_excellence', 'performance_consistency', 'improved_performance'
        ]
        
        demographic_features = [
            'gender', 'age_at_enrollment', 'mature_student', 'traditional_age',
            'from_developed_province', 'from_major_district', 'close_to_university'
        ]
        
        experience_features = [
            'internship_count', 'project_count', 'certification_count',
            'has_internship', 'multiple_internships', 'excellent_intern',
            'active_project_developer', 'project_completion_rate', 'high_project_completer',
            'tech_diversity', 'diverse_tech_skills', 'certified_professional',
            'experience_score'
        ]
        
        pathway_features = [
            'pathway', 'high_demand_pathway', 'pathway_alignment_score'
        ]
        
        interaction_features = [
            'gpa_experience_interaction', 'al_gpa_interaction',
            'location_advantage_academic', 'urban_academic_advantage',
            'pathway_performance'
        ]
        
        time_features = [
            'intake_year', 'years_since_intake', 'recent_graduate',
            'experienced_graduate', 'post_covid_intake'
        ]
        
        # Combine all feature categories
        all_features = (academic_features + demographic_features + 
                       experience_features + pathway_features + 
                       interaction_features + time_features)
        
        # Select available features
        available_features = [f for f in all_features if f in df.columns]
        
        return df[available_features]
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding and one-hot encoding"""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = set(df[col].astype(str).unique())
                known_values = set(self.label_encoders[col].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Add new categories to encoder
                    all_values = list(known_values) + list(new_values)
                    self.label_encoders[col].classes_ = np.array(all_values)
                
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        
        return df
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the best features using statistical methods
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            Tuple of (selected_features_df, feature_names)
        """
        # Use SelectKBest with f_regression for feature selection
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        # Calculate feature importance scores
        feature_scores = self.feature_selector.scores_
        self.feature_importance = dict(zip(X.columns, feature_scores))
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 15 Most Important Features:")
        print("-" * 40)
        for i, (feature, score) in enumerate(sorted_features[:15]):
            print(f"{i+1:2d}. {feature:<30} {score:8.2f}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index), self.selected_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance
    
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2, 
                                 interaction_only: bool = True) -> pd.DataFrame:
        """
        Create polynomial features for enhanced model performance
        
        Args:
            X: Feature matrix
            degree: Degree of polynomial features
            interaction_only: Whether to include only interaction terms
            
        Returns:
            DataFrame with polynomial features
        """
        # Select only numeric features for polynomial transformation
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_features]
        
        # Limit to most important features to avoid curse of dimensionality
        if len(numeric_features) > 10:
            # Select top 10 numeric features based on correlation with target if available
            X_numeric = X_numeric.iloc[:, :10]
        
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, 
                                include_bias=False)
        X_poly = poly.fit_transform(X_numeric)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(X_numeric.columns)
        
        # Combine with original categorical features
        categorical_features = X.select_dtypes(exclude=[np.number]).columns
        if len(categorical_features) > 0:
            X_categorical = X[categorical_features]
            X_combined = np.hstack([X_poly, X_categorical.values])
            all_feature_names = list(feature_names) + list(categorical_features)
        else:
            X_combined = X_poly
            all_feature_names = list(feature_names)
        
        return pd.DataFrame(X_combined, columns=all_feature_names, index=X.index)