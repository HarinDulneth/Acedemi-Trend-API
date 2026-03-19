import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_applications(app_2005_2015, app_2016_2023):
    """
    Merge two application datasets into a single DataFrame.
    
    Parameters:
    - app_2005_2015 (pd.DataFrame): Applications from 2005-2015.
    - app_2016_2023 (pd.DataFrame): Applications from 2016-2023.
    
    Returns:
    - pd.DataFrame: Merged applications DataFrame.
    """
    if app_2005_2015 is None or app_2016_2023 is None:
        logging.error("One or both application datasets are missing.")
        return None
    
    # Check if columns match
    if list(app_2005_2015.columns) != list(app_2016_2023.columns):
        logging.error("Column mismatch between the two datasets.")
        return None
    
    # Concatenate the datasets
    merged_df = pd.concat([app_2005_2015, app_2016_2023], ignore_index=True)
    logging.info(f"Merged applications dataset shape: {merged_df.shape}")
    return merged_df

# Example usage
if __name__ == "__main__":
    # Load datasets
    app_2005_2015_df = pd.read_csv('data/raw/Application_2005-2015.csv')
    app_2016_2023_df = pd.read_csv('data/raw/Application_2016-2023.csv')
    
    # Merge datasets
    merged_applications = merge_applications(app_2005_2015_df, app_2016_2023_df)
    
    if merged_applications is not None:
        merged_applications.to_csv('data/processed/merged_applications.csv', index=False)
        logging.info("Merged applications saved to data/processed/merged_applications.csv")
        