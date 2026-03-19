import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from src.utils.helper_functions import handle_missing_values, standardize_columns

# Set up logging (will be configured in main function)
logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration with proper directory creation."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_cleaning.log'),
            logging.StreamHandler()
        ]
    )

class ApplicationCleaner:
    """
    A comprehensive data cleaner for applications dataset.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cleaner with configuration options.
        
        Parameters:
        - config (dict): Configuration dictionary with cleaning parameters
        """
        self.config = config or self._get_default_config()
        self.cleaning_stats = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for data cleaning."""
        return {
            'missing_value_threshold': 0.5,  # Drop columns with >50% missing values
            'duplicate_subset': None,  # Columns to check for duplicates
            'standardize_text': True,
            'remove_outliers': True,
            'outlier_method': 'iqr',  # 'iqr' or 'zscore'
            'outlier_threshold': 3.0,
            'date_columns': [],  # List of date columns to parse
            'categorical_columns': ['university', 'course_name'],
            'numerical_columns': [],
        }
    
    def clean_applications(self, merged_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Clean the merged applications dataset with comprehensive preprocessing.
        
        Parameters:
        - merged_df (pd.DataFrame): Merged applications DataFrame.
        
        Returns:
        - pd.DataFrame: Cleaned DataFrame or None if cleaning fails.
        """
        if merged_df is None or merged_df.empty:
            logger.error("Input dataset is missing or empty.")
            return None
        
        logger.info(f"Starting data cleaning for dataset with shape: {merged_df.shape}")
        
        try:
            # Create a copy to avoid modifying original data
            df = merged_df.copy()
            
            # Store original statistics
            self.cleaning_stats['original_shape'] = df.shape
            self.cleaning_stats['original_missing'] = df.isnull().sum().sum()
            
            # Step 1: Handle missing values
            df = self._handle_missing_values(df)
            
            # Step 2: Remove duplicates
            df = self._remove_duplicates(df)
            
            # Step 3: Standardize text columns
            if self.config['standardize_text']:
                df = self._standardize_text_columns(df)
            
            # Step 4: Parse date columns
            df = self._parse_date_columns(df)
            
            # Step 5: Clean numerical columns
            df = self._clean_numerical_columns(df)
            
            # Step 6: Remove outliers
            if self.config['remove_outliers']:
                df = self._remove_outliers(df)
            
            # Step 7: Final validation
            df = self._validate_data(df)
            
            # Store final statistics
            self.cleaning_stats['final_shape'] = df.shape
            self.cleaning_stats['final_missing'] = df.isnull().sum().sum()
            
            logger.info(f"Data cleaning completed. Final shape: {df.shape}")
            self._log_cleaning_summary()
            
            return df
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            return None
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with multiple strategies."""
        logger.info("Handling missing values...")
        
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > self.config['missing_value_threshold']].index
        
        if len(cols_to_drop) > 0:
            logger.warning(f"Dropping columns with >{self.config['missing_value_threshold']*100}% missing values: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
        
        # Handle remaining missing values using built-in pandas methods
        try:
            # Try to use the helper function if it exists and works
            df = handle_missing_values(df)
        except (TypeError, AttributeError) as e:
            logger.warning(f"Helper function failed ({str(e)}), using built-in methods")
            # Fallback to built-in pandas methods
            df = df.fillna(method='ffill')
        
        # For categorical columns, fill remaining NaNs with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        # For numerical columns, fill remaining NaNs with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median: {median_val}")
        
        self.cleaning_stats['columns_dropped_missing'] = len(cols_to_drop)
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        logger.info("Removing duplicates...")
        
        initial_rows = len(df)
        subset_cols = self.config['duplicate_subset']
        
        if subset_cols:
            # Remove duplicates based on specific columns
            df = df.drop_duplicates(subset=subset_cols, keep='first')
        else:
            # Remove complete duplicates
            df = df.drop_duplicates(keep='first')
        
        duplicates_removed = initial_rows - len(df)
        self.cleaning_stats['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df
    
    def _standardize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text columns."""
        logger.info("Standardizing text columns...")
        
        columns_to_standardize = [col for col in self.config['categorical_columns'] if col in df.columns]
        
        if columns_to_standardize:
            try:
                # Try to use the helper function if it exists and works
                df = standardize_columns(df, columns=columns_to_standardize)
            except (TypeError, AttributeError) as e:
                logger.warning(f"Helper function failed ({str(e)}), using built-in methods")
                # Fallback to built-in standardization
                for col in columns_to_standardize:
                    if col in df.columns:
                        # Basic standardization
                        df[col] = df[col].astype(str).str.strip().str.title()
        
        # Additional text cleaning
        for col in columns_to_standardize:
            if col in df.columns:
                # Remove extra whitespace and convert to title case
                df[col] = df[col].astype(str).str.strip().str.title()
                # Remove special characters (optional)
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        
        return df
    
    def _parse_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and validate date columns."""
        logger.info("Parsing date columns...")
        
        for col in self.config['date_columns']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Log any dates that couldn't be parsed
                    invalid_dates = df[col].isnull().sum()
                    if invalid_dates > 0:
                        logger.warning(f"Found {invalid_dates} invalid dates in column '{col}'")
                except Exception as e:
                    logger.error(f"Error parsing date column '{col}': {str(e)}")
        
        return df
    
    def _clean_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numerical columns."""
        logger.info("Cleaning numerical columns...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Remove infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in column '{col}', replacing with NaN")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Ensure non-negative values for specific columns
            if col in ['applications']:  # Applications should be non-negative
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in column '{col}', setting to 0")
                    df[col] = df[col].clip(lower=0)
            
            # Validate year column
            if col == 'year':
                # Check for reasonable year range (e.g., 2000-2030)
                invalid_years = ((df[col] < 2000) | (df[col] > 2030)).sum()
                if invalid_years > 0:
                    logger.warning(f"Found {invalid_years} invalid years in column '{col}'")
            
            # Validate cutoff_mark column
            if col == 'cutoff_mark':
                # Cutoff marks should typically be between 0 and 4 (assuming Z-score system)
                invalid_cutoffs = ((df[col] < 0) | (df[col] > 4)).sum()
                if invalid_cutoffs > 0:
                    logger.warning(f"Found {invalid_cutoffs} potentially invalid cutoff marks")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from numerical columns."""
        logger.info("Removing outliers...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numerical_cols:
            if self.config['outlier_method'] == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif self.config['outlier_method'] == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > self.config['outlier_threshold']
            
            else:
                continue
            
            col_outliers = outlier_mask.sum()
            if col_outliers > 0:
                logger.info(f"Removing {col_outliers} outliers from column '{col}'")
                df = df[~outlier_mask]
                outliers_removed += col_outliers
        
        self.cleaning_stats['outliers_removed'] = outliers_removed
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform final data validation."""
        logger.info("Performing final data validation...")
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            logger.warning(f"Removing {empty_rows} completely empty rows")
            df = df.dropna(how='all')
        
        # Check data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column should be categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    logger.info(f"Converting column '{col}' to categorical")
                    df[col] = df[col].astype('category')
        
        return df
    
    def _log_cleaning_summary(self):
        """Log summary of cleaning operations."""
        logger.info("=== Data Cleaning Summary ===")
        logger.info(f"Original shape: {self.cleaning_stats['original_shape']}")
        logger.info(f"Final shape: {self.cleaning_stats['final_shape']}")
        logger.info(f"Rows removed: {self.cleaning_stats['original_shape'][0] - self.cleaning_stats['final_shape'][0]}")
        logger.info(f"Columns dropped (missing): {self.cleaning_stats.get('columns_dropped_missing', 0)}")
        logger.info(f"Duplicates removed: {self.cleaning_stats.get('duplicates_removed', 0)}")
        logger.info(f"Outliers removed: {self.cleaning_stats.get('outliers_removed', 0)}")
        logger.info(f"Missing values reduced from {self.cleaning_stats['original_missing']} to {self.cleaning_stats['final_missing']}")
        logger.info("===========================")

def load_data_safely(file_path: str) -> Optional[pd.DataFrame]:
    """
    Safely load data from CSV file with error handling.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    
    Returns:
    - pd.DataFrame or None: Loaded DataFrame or None if loading fails.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None

def save_data_safely(df: pd.DataFrame, file_path: str) -> bool:
    """
    Safely save DataFrame to CSV file.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - file_path (str): Path to save the CSV file.
    
    Returns:
    - bool: True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved cleaned data to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        return False

def main():
    """Main function to orchestrate the data cleaning process."""
    
    # Set up logging first
    setup_logging()
    
    # Configuration - Updated to match your data structure
    config = {
        'missing_value_threshold': 0.5,
        'duplicate_subset': ['university', 'course_name', 'district', 'year'],  # Check for duplicates across these key columns
        'standardize_text': True,
        'remove_outliers': True,
        'outlier_method': 'iqr',
        'date_columns': [],  # No date columns in your data
        'categorical_columns': ['university', 'course_name', 'district'],  # Text columns to standardize
        'numerical_columns': ['year', 'applications', 'cutoff_mark'],  # Numerical columns to clean
    }
    
    # File paths
    input_file = 'data/processed/merged_applications.csv'
    output_file = 'data/processed/cleaned_applications.csv'
    
    try:
        # Load data
        merged_applications = load_data_safely(input_file)
        
        if merged_applications is None:
            logger.error("Failed to load input data. Exiting.")
            return
        
        # Initialize cleaner
        cleaner = ApplicationCleaner(config)
        
        # Clean data
        cleaned_applications = cleaner.clean_applications(merged_applications)
        
        if cleaned_applications is None:
            logger.error("Data cleaning failed. Exiting.")
            return
        
        # Save cleaned data
        if save_data_safely(cleaned_applications, output_file):
            logger.info("Data cleaning process completed successfully.")
        else:
            logger.error("Failed to save cleaned data.")
    
    except Exception as e:
        logger.error(f"Unexpected error in main process: {str(e)}")

if __name__ == "__main__":
    main()