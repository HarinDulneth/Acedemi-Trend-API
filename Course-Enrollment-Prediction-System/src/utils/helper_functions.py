import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_csv(file_path, delimiter=',', encoding='utf-8'):
    """
    Load a CSV file into a pandas DataFrame.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    - delimiter (str): Delimiter used in the CSV file (default: ',').
    - encoding (str): Encoding of the CSV file (default: 'utf-8').
    
    Returns:
    - pd.DataFrame: Loaded DataFrame or None if error occurs.
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        logging.info(f"Successfully loaded {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def standardize_columns(df):
    """
    Standardize column names by converting to lowercase and replacing spaces with underscores.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    
    Returns:
    - pd.DataFrame: DataFrame with standardized column names.
    """
    df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    return df

def handle_missing_values(df, numeric_method='mean', categorical_method='unknown'):
    """
    Handle missing values in numerical and categorical columns.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - numeric_method (str): Method for numerical columns ('mean', 'median', 'ffill', 'bfill').
    - categorical_method (str): Method for categorical columns ('unknown', 'mode').
    
    Returns:
    - pd.DataFrame: DataFrame with handled missing values.
    """
    # Numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_method in ['mean', 'median']:
        for col in numeric_cols:
            if numeric_method == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    elif numeric_method in ['ffill', 'bfill']:
        df[numeric_cols] = df[numeric_cols].fillna(method=numeric_method)
    else:
        logging.warning(f"Unknown numeric imputation method: {numeric_method}")
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if categorical_method == 'unknown':
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    elif categorical_method == 'mode':
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        logging.warning(f"Unknown categorical imputation method: {categorical_method}")
    
    return df

def save_dataframe(df, file_path):
    """
    Save a DataFrame to a CSV file with logging.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - file_path (str): Path to save the CSV file.
    
    Returns:
    - bool: True if saved successfully, False otherwise.
    """
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Saved DataFrame to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving {file_path}: {e}")
        return False

def compute_correlation_matrix(df, numeric_only=True):
    """
    Compute the correlation matrix for numerical columns.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - numeric_only (bool): Include only numerical columns (default: True).
    
    Returns:
    - pd.DataFrame: Correlation matrix.
    """
    if numeric_only:
        df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = df.corr()
    logging.info("Computed correlation matrix")
    return corr_matrix

def plot_distribution(data, column, title, output_path=None):
    """
    Plot a histogram for a given column.
    
    Parameters:
    - data (pd.DataFrame or pd.Series): Data to plot.
    - column (str): Column name to plot (if DataFrame).
    - title (str): Plot title.
    - output_path (str, optional): Path to save the plot.
    
    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    if isinstance(data, pd.DataFrame):
        sns.histplot(data[column], bins=30)
    else:
        sns.histplot(data, bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    if output_path:
        plt.savefig(output_path)
        logging.info(f"Saved plot to {output_path}")
    plt.close()

def scale_features(df, columns):
    """
    Scale numerical features using StandardScaler.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to scale.
    
    Returns:
    - pd.DataFrame: DataFrame with scaled columns.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    if all(col in df.columns for col in columns):
        df_scaled[columns] = scaler.fit_transform(df[columns])
        logging.info(f"Scaled columns: {columns}")
    else:
        logging.warning("Some columns not found in DataFrame")
    return df_scaled, scaler  

if __name__ == "__main__":
    # Example usage
    df = load_csv('data/raw/enrollments.csv')
    if df is not None:
        df = standardize_columns(df)
        df = handle_missing_values(df, numeric_method='mean', categorical_method='unknown')
        corr_matrix = compute_correlation_matrix(df)
        save_dataframe(corr_matrix, 'data/processed/sample_correlation_matrix.csv')
        plot_distribution(df, 'enrollments', 'Enrollment Distribution', 
                         'visualizations/sample_distribution.png')