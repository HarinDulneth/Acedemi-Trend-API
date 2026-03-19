import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """Enhanced feature engineering class for enrollment and application prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV file with error handling."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {file_path} with shape {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, numeric_method: str = 'median', 
                            categorical_method: str = 'mode') -> pd.DataFrame:
        """Enhanced missing value handling."""
        df = df.copy()
        
        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if numeric_method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif numeric_method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif numeric_method == 'forward_fill':
                    df[col] = df[col].ffill()
                    df[col].fillna(df[col].median(), inplace=True)  # Fallback
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if categorical_method == 'mode':
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                    df[col].fillna(mode_val, inplace=True)
                else:
                    df[col].fillna(categorical_method, inplace=True)
        
        return df
    
    def load_and_clean_data(self, file_path: str, required_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load and clean a dataset with enhanced error handling."""
        try:
            df = self.load_csv(file_path)
            if df is None:
                return None
            
            # Standardize column names
            df = self.standardize_columns(df)
            
            # Ensure year is integer if present
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Check required columns
            if required_cols:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in {file_path}: {missing_cols}")
                    return None
            
            logger.info(f"Successfully processed {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def load_datasets(self) -> Tuple[Optional[pd.DataFrame], ...]:
        """Load all required datasets."""
        datasets = {}
        
        # Define file paths and required columns
        dataset_configs = {
            'merged_apps': {
                'path': 'data/processed/merged_applications.csv',
                'required_cols': ['university', 'course_name', 'year', 'applications', 'cutoff_mark']
            },
            'enrollments': {
                'path': 'data/raw/enrollments.csv',
                'required_cols': ['university', 'course_name', 'year', 'enrollments']
            },
            'rankings': {
                'path': 'data/raw/rank_by_university.csv',
                'required_cols': ['university', 'year', 'rank']
            },
            'job_market': {
                'path': 'data/raw/job_market_demand_by_field.csv',
                'required_cols': ['field', 'year', 'job_market_demand_by_field']
            },
            'international': {
                'path': 'data/raw/international_trend.csv',
                'required_cols': ['course_name', 'field', 'year', 'international_education_trend']
            }
        }
        
        for name, config in dataset_configs.items():
            datasets[name] = self.load_and_clean_data(config['path'], config['required_cols'])
        
        return tuple(datasets.values())
    
    def merge_datasets(self, merged_apps: pd.DataFrame, enrollments: pd.DataFrame, 
                      rankings: pd.DataFrame, job_market: pd.DataFrame, 
                      international: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Enhanced dataset merging with better error handling."""
        try:
            # Check if all datasets are available
            if any(df is None for df in [merged_apps, enrollments, rankings, job_market, international]):
                logger.error("One or more datasets failed to load.")
                return None
            
            # Rename job market demand column
            if 'job_market_demand_by_field' in job_market.columns:
                job_market = job_market.rename(columns={'job_market_demand_by_field': 'job_market_demand'})
            
            # Start with enrollments as base
            df = enrollments.copy()
            
            # Merge with applications
            df = pd.merge(df, merged_apps, on=['university', 'course_name', 'year'], how='outer')
            
            # Merge with rankings
            df = pd.merge(df, rankings, on=['university', 'year'], how='left')
            
            # Create course-field mapping
            course_field_map = international[['course_name', 'field']].drop_duplicates()
            df = pd.merge(df, course_field_map, on=['course_name'], how='left')
            
            # Merge with international trends
            df = pd.merge(df, international[['course_name', 'year', 'international_education_trend']], 
                         on=['course_name', 'year'], how='left')
            
            # Merge with job market demand
            df = pd.merge(df, job_market[['field', 'year', 'job_market_demand']], 
                         on=['field', 'year'], how='left')
            
            # Fill missing values for target variables
            df['enrollments'] = df['enrollments'].fillna(0)
            df['applications'] = df['applications'].fillna(0)
            
            logger.info(f"Merged dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            return None
    
    def encode_categorical_features(self, df: pd.DataFrame, target_cols: List[str] = ['enrollments', 'applications']) -> pd.DataFrame:
        """Enhanced categorical encoding with multiple strategies - FIXED VERSION."""
        df = df.copy()
        
        # High cardinality categorical columns for target encoding
        high_cardinality_cols = ['university', 'course_name']
        
        # Low cardinality categorical columns for one-hot encoding
        low_cardinality_cols = ['field']
        
        # Target encoding for high cardinality columns - FIXED
        for col in high_cardinality_cols:
            if col in df.columns:
                for target_col in target_cols:
                    if target_col in df.columns and df[target_col].notna().sum() > 0:
                        try:
                            # Calculate group means properly
                            group_means = df.groupby(col)[target_col].mean()
                            global_mean = df[target_col].mean()
                            
                            # Use simple target encoding (can be improved with regularization later)
                            df[f'{col}_{target_col}_target_encoded'] = df[col].map(group_means).fillna(global_mean)
                            
                            logger.info(f"Target encoded '{col}' based on {target_col}")
                        except Exception as e:
                            logger.warning(f"Failed to target encode {col} for {target_col}: {e}")
                            # Fall back to label encoding
                            if col not in self.label_encoders:
                                self.label_encoders[col] = LabelEncoder()
                            df[f'{col}_label_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # One-hot encoding for low cardinality columns
        for col in low_cardinality_cols:
            if col in df.columns:
                # Limit to top categories to prevent too many features
                top_categories = df[col].value_counts().head(10).index
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'other')
                
                encoded_cols = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                df = pd.concat([df, encoded_cols], axis=1)
                logger.info(f"One-hot encoded '{col}' into {len(encoded_cols.columns)} columns")
        
        # Label encoding for remaining categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['university', 'course_name']:  # Keep these for temporal features
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                try:
                    df[f'{col}_label_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                except Exception as e:
                    logger.warning(f"Failed to label encode {col}: {e}")
        
        # Drop original categorical columns except those needed for temporal features
        drop_cols = [col for col in low_cardinality_cols if col in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        return df
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced temporal feature engineering."""
        df = df.copy()
        df = df.sort_values(['university', 'course_name', 'year'])
        
        # Define columns for temporal features
        temporal_cols = ['enrollments', 'applications', 'cutoff_mark', 'rank', 
                        'job_market_demand', 'international_education_trend']
        
        # Add additional temporal columns if they exist
        additional_cols = ['infrastructure_and_facility_ratings', 'page_followers']
        temporal_cols.extend([col for col in additional_cols if col in df.columns])
        
        groupby_cols = ['university', 'course_name']
        
        for col in temporal_cols:
            if col in df.columns:
                try:
                    # Lag features (1, 2, 3 years)
                    for lag in [1, 2, 3]:
                        df[f'{col}_lag{lag}'] = df.groupby(groupby_cols)[col].shift(lag)
                    
                    # Rolling statistics
                    for window in [2, 3, 5]:
                        df[f'{col}_rolling_mean_{window}yr'] = df.groupby(groupby_cols)[col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
                        df[f'{col}_rolling_std_{window}yr'] = df.groupby(groupby_cols)[col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).std()
                        )
                    
                    # Growth rates and trends
                    df[f'{col}_growth_rate'] = df.groupby(groupby_cols)[col].pct_change()
                    df[f'{col}_growth_rate_3yr'] = df.groupby(groupby_cols)[col].pct_change(periods=3)
                    
                    # Trend direction (increasing/decreasing)
                    df[f'{col}_trend_direction'] = df.groupby(groupby_cols)[col].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
                except Exception as e:
                    logger.warning(f"Failed to create temporal features for {col}: {e}")
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Year-based features
        if 'year' in df.columns:
            df['year_squared'] = df['year'] ** 2
            df['years_since_start'] = df['year'] - df['year'].min()
        
        logger.info("Completed temporal feature engineering")
        return df
    
    def engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced interaction feature engineering."""
        df = df.copy()
        
        # Ratio features
        ratio_pairs = [
            ('applications', 'cutoff_mark', 'app_to_cutoff_ratio'),
            ('applications', 'rank', 'app_to_rank_ratio'),
            ('enrollments', 'applications', 'enrollment_rate'),
            ('cutoff_mark', 'rank', 'cutoff_to_rank_ratio'),
            ('job_market_demand', 'international_education_trend', 'job_to_international_ratio')
        ]
        
        for col1, col2, ratio_name in ratio_pairs:
            if col1 in df.columns and col2 in df.columns:
                try:
                    df[ratio_name] = df[col1] / df[col2].replace(0, np.nan)
                    df[ratio_name] = df[ratio_name].clip(lower=-1e6, upper=1e6)
                except Exception as e:
                    logger.warning(f"Failed to create ratio feature {ratio_name}: {e}")
        
        # Multiplicative interactions
        interaction_pairs = [
            ('rank', 'cutoff_mark', 'rank_cutoff_interaction'),
            ('job_market_demand', 'international_education_trend', 'market_international_interaction'),
            ('applications_lag1', 'cutoff_mark', 'past_app_cutoff_interaction')
        ]
        
        for col1, col2, interaction_name in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                try:
                    df[interaction_name] = df[col1] * df[col2]
                except Exception as e:
                    logger.warning(f"Failed to create interaction feature {interaction_name}: {e}")
        
        # Competitive features
        if 'rank' in df.columns:
            try:
                df['is_top_10'] = (df['rank'] <= 10).astype(int)
                df['is_top_50'] = (df['rank'] <= 50).astype(int)
                df['rank_category'] = pd.cut(df['rank'], bins=[0, 10, 50, 100, np.inf], 
                                           labels=['top10', 'top50', 'top100', 'others'])
            except Exception as e:
                logger.warning(f"Failed to create rank-based features: {e}")
        
        # Demand-supply indicators
        if 'applications' in df.columns and 'enrollments' in df.columns:
            try:
                df['excess_demand'] = df['applications'] - df['enrollments']
                df['demand_supply_ratio'] = df['applications'] / df['enrollments'].replace(0, np.nan)
            except Exception as e:
                logger.warning(f"Failed to create demand-supply features: {e}")
        
        logger.info("Completed interaction feature engineering")
        return df
    
    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering techniques."""
        df = df.copy()
        
        # Polynomial features for key numeric columns
        poly_cols = ['cutoff_mark', 'rank', 'applications', 'enrollments']
        for col in poly_cols:
            if col in df.columns:
                try:
                    df[f'{col}_squared'] = df[col] ** 2
                    df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
                except Exception as e:
                    logger.warning(f"Failed to create polynomial features for {col}: {e}")
        
        # Percentile ranks within groups
        groupby_cols = ['university', 'year']
        rank_cols = ['applications', 'enrollments', 'cutoff_mark']
        
        for col in rank_cols:
            if col in df.columns:
                try:
                    df[f'{col}_percentile_rank'] = df.groupby(groupby_cols)[col].rank(pct=True)
                except Exception as e:
                    logger.warning(f"Failed to create percentile rank for {col}: {e}")
        
        # Market position features
        if 'applications' in df.columns:
            try:
                df['market_share'] = df.groupby('year')['applications'].transform(lambda x: x / x.sum())
                df['applications_z_score'] = df.groupby('year')['applications'].transform(lambda x: (x - x.mean()) / x.std())
            except Exception as e:
                logger.warning(f"Failed to create market position features: {e}")
        
        # Seasonal/cyclical features
        if 'year' in df.columns:
            try:
                df['year_sin'] = np.sin(2 * np.pi * df['year'] / 10)  # 10-year cycle
                df['year_cos'] = np.cos(2 * np.pi * df['year'] / 10)
            except Exception as e:
                logger.warning(f"Failed to create cyclical features: {e}")
        
        logger.info("Completed advanced feature engineering")
        return df
    
    def select_features(self, df: pd.DataFrame, target_cols: List[str] = ['enrollments', 'applications'], 
                       correlation_threshold: float = 0.95, k_best: int = 50) -> Tuple[pd.DataFrame, Dict]:
        """Enhanced feature selection with multiple techniques - FIXED VERSION."""
        
        # Separate features and targets
        feature_cols = [col for col in df.columns if col not in target_cols + ['university', 'course_name']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Remove constant features
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        selection_results = {}
        
        for target_col in target_cols:
            if target_col not in df.columns:
                continue
                
            y = df[target_col].fillna(df[target_col].median())
            
            try:
                # Remove highly correlated features
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
                X_reduced = X.drop(columns=to_drop)
                
                # Statistical feature selection
                k_select = min(k_best, X_reduced.shape[1])
                selector = SelectKBest(score_func=f_regression, k=k_select)
                X_selected = selector.fit_transform(X_reduced, y)
                selected_features = X_reduced.columns[selector.get_support()].tolist()
                
                # Mutual information feature selection
                mi_k = min(k_best//2, len(selected_features))
                if mi_k > 0:
                    mi_selector = SelectKBest(score_func=mutual_info_regression, k=mi_k)
                    mi_selector.fit(X_reduced[selected_features], y)
                    final_features = [selected_features[i] for i in range(len(selected_features)) if mi_selector.get_support()[i]]
                else:
                    final_features = selected_features[:k_best//2]
                
                selection_results[target_col] = {
                    'dropped_correlated': to_drop,
                    'selected_features': final_features,
                    'feature_scores': dict(zip(selected_features, selector.scores_[selector.get_support()]))
                }
                
            except Exception as e:
                logger.warning(f"Feature selection failed for {target_col}: {e}")
                # Fallback: select top features by variance
                feature_vars = X.var().sort_values(ascending=False)
                fallback_features = feature_vars.head(min(k_best, len(feature_vars))).index.tolist()
                selection_results[target_col] = {'selected_features': fallback_features}
        
        # Combine selected features from all targets
        all_selected = set()
        for result in selection_results.values():
            all_selected.update(result.get('selected_features', []))
        
        # Keep essential columns
        essential_cols = ['university', 'course_name', 'year'] + target_cols
        final_cols = list(all_selected) + [col for col in essential_cols if col in df.columns]
        
        df_selected = df[final_cols]
        
        logger.info(f"Selected {len(all_selected)} features from {len(feature_cols)} original features")
        
        # Create correlation matrix for visualization
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
        corr_matrix = df_selected[numeric_cols].corr()
        
        return df_selected, selection_results, corr_matrix
    
    def validate_features(self, df: pd.DataFrame, target_cols: List[str] = ['enrollments', 'applications']) -> Dict:
        """Validate feature quality and predictive power."""
        validation_results = {}
        
        for target_col in target_cols:
            if target_col not in df.columns:
                continue
                
            # Prepare data
            feature_cols = [col for col in df.columns if col not in target_cols + ['university', 'course_name']]
            X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
            y = df[target_col].fillna(df[target_col].median())
            
            # Remove rows where target is 0 or missing for better validation
            valid_idx = (y > 0) & (y.notna())
            if valid_idx.sum() < 10:  # Not enough valid samples
                logger.warning(f"Not enough valid samples for {target_col}")
                validation_results[target_col] = {'r2_score': 0, 'mse': float('inf')}
                continue
            
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]
            
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
                
                # Quick model validation
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                validation_results[target_col] = {
                    'r2_score': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'feature_importance': dict(zip(X.columns, model.feature_importances_))
                }
                
            except Exception as e:
                logger.warning(f"Validation failed for {target_col}: {e}")
                validation_results[target_col] = {'r2_score': 0, 'mse': float('inf')}
        
        return validation_results
    
    def save_dataframe(self, df: pd.DataFrame, file_path: str) -> None:
        """Save dataframe with error handling."""
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            df.to_csv(file_path, index=False)
            logger.info(f"Saved dataframe to {file_path}")
        except Exception as e:
            logger.error(f"Error saving dataframe to {file_path}: {e}")
    
    def feature_engineering_pipeline(self) -> Optional[pd.DataFrame]:
        """Complete feature engineering pipeline."""
        logger.info("Starting enhanced feature engineering pipeline")
        
        # Load datasets
        merged_apps, enrollments, rankings, job_market, international = self.load_datasets()
        if any(df is None for df in [merged_apps, enrollments, rankings, job_market, international]):
            logger.error("Feature engineering aborted due to missing datasets.")
            return None
        
        # Merge datasets
        df = self.merge_datasets(merged_apps, enrollments, rankings, job_market, international)
        if df is None:
            return None
        
        logger.info(f"Initial dataset shape: {df.shape}")
        
        # Feature engineering steps
        df = self.encode_categorical_features(df)
        logger.info(f"After categorical encoding: {df.shape}")
        
        df = self.engineer_temporal_features(df)
        logger.info(f"After temporal features: {df.shape}")
        
        df = self.engineer_interaction_features(df)
        logger.info(f"After interaction features: {df.shape}")
        
        df = self.engineer_advanced_features(df)
        logger.info(f"After advanced features: {df.shape}")
        
        # Handle missing values
        df = self.handle_missing_values(df, numeric_method='median')
        
        # Feature selection
        df_selected, selection_results, corr_matrix = self.select_features(df)
        logger.info(f"After feature selection: {df_selected.shape}")
        
        # Validate features
        validation_results = self.validate_features(df_selected)
        
        # Log validation results
        for target, results in validation_results.items():
            logger.info(f"{target} - R² Score: {results.get('r2_score', 0):.4f}, MSE: {results.get('mse', 0):.4f}")

        # Create visualizations
        logger.info("Creating feature importance visualizations...")
        save_feature_importance_plots(validation_results, 'plots')
        save_combined_feature_importance_plot(validation_results, 'plots')
        save_correlation_heatmap(corr_matrix, 'plots')
        logger.info("Visualizations saved to 'plots' directory")
        
        # Save results
        self.save_dataframe(df_selected, 'data/processed/final_dataset.csv')
        
        # Save feature importance
        feature_importance_df = pd.DataFrame([
            {'target': target, 'feature': feature, 'importance': importance}
            for target, results in validation_results.items()
            for feature, importance in results.get('feature_importance', {}).items()
        ])
        self.save_dataframe(feature_importance_df, 'data/processed/feature_importance.csv')
        
        # Save selection results
        selection_summary = pd.DataFrame([
            {'target': target, 'selected_features': len(results.get('selected_features', [])), 
             'dropped_correlated': len(results.get('dropped_correlated', []))}
            for target, results in selection_results.items()
        ])
        self.save_dataframe(selection_summary, 'data/processed/feature_selection_summary.csv')

         # Save correlation matrix
        self.save_dataframe(corr_matrix, 'data/processed/correlation_matrix.csv')
        
        logger.info("Enhanced feature engineering pipeline completed successfully")
        return df_selected

def save_feature_importance_plots(validation_results: Dict, output_dir: str = 'plots', 
                                top_n: int = 20) -> None:
    """Standalone function to save feature importance plots."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')  # Use default if seaborn-v0_8 not available
    
    for target, results in validation_results.items():
        feature_importance = results.get('feature_importance', {})
        
        if not feature_importance:
            print(f"No feature importance data for {target}")
            continue
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = [item[0] for item in top_features]
        importances = [item[1] for item in top_features]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importances, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(features))))
        
        # Customize the plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importances for {target.title()}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(importance + max(importances) * 0.01, i, f'{importance:.4f}', 
                   va='center', ha='left', fontsize=9)
        
        # Invert y-axis to show highest importance at top
        ax.invert_yaxis()
        
        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        filename = f'feature_importance_{target}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved feature importance plot for {target} to {filepath}")

def save_combined_feature_importance_plot(validation_results: Dict, 
                                        output_dir: str = 'plots', top_n: int = 15) -> None:
    """Save a combined feature importance plot for all targets."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for combined plot
    all_features = {}
    targets = list(validation_results.keys())
    
    for target, results in validation_results.items():
        feature_importance = results.get('feature_importance', {})
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        all_features[target] = dict(sorted_features[:top_n])
    
    if not all_features:
        print("No feature importance data to plot")
        return
    
    # Get union of all top features
    all_feature_names = set()
    for features_dict in all_features.values():
        all_feature_names.update(features_dict.keys())
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame(index=sorted(all_feature_names))
    
    for target in targets:
        importance_df[target] = importance_df.index.map(all_features[target]).fillna(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(importance_df) * 0.3)))
    
    # Create grouped bar plot
    importance_df.plot(kind='barh', ax=ax, width=0.8)
    
    # Customize the plot
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Comparison Across Targets', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Customize legend
    ax.legend(title='Target Variables', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save the plot
    filepath = os.path.join(output_dir, 'combined_feature_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved combined feature importance plot to {filepath}")

def save_correlation_heatmap(corr_matrix: pd.DataFrame, 
                           output_dir: str = 'plots', top_n: int = 30) -> None:
    """Save correlation heatmap of top features."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select top features based on average correlation
    if corr_matrix.shape[0] > top_n:
        # Calculate average absolute correlation for each feature
        avg_corr = corr_matrix.abs().mean().sort_values(ascending=False)
        top_features = avg_corr.head(top_n).index
        corr_subset = corr_matrix.loc[top_features, top_features]
    else:
        corr_subset = corr_matrix
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    
    sns.heatmap(corr_subset, mask=mask, annot=True, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
               fmt='.2f', ax=ax)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    filepath = os.path.join(output_dir, 'feature_correlation_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved correlation heatmap to {filepath}")


def main():
    """Main function to run the enhanced feature engineering pipeline."""
    engineer = EnhancedFeatureEngineer()
    final_dataset = engineer.feature_engineering_pipeline()
    
    if final_dataset is not None:
        print(f"Final dataset shape: {final_dataset.shape}")
        print(f"Final dataset columns: {list(final_dataset.columns)}")
        print("\nDataset info:")
        print(final_dataset.info())
        print("\nFirst few rows:")
        print(final_dataset.head())
    else:
        print("Feature engineering failed!")

if __name__ == "__main__":
    main()