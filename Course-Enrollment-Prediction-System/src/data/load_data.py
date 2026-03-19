import pandas as pd
import logging
from src.utils.helper_functions import load_csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Load all raw datasets
    rank_df = load_csv('data/raw/rank_by_university.csv')
    enrollments_df = load_csv('data/raw/enrollments.csv')
    job_market_df = load_csv('data/raw/job_market_demand_by_field.csv')
    app_2005_2015_df = load_csv('data/raw/Application_2005-2015.csv')
    international_trend_df = load_csv('data/raw/international_trend.csv')
    app_2016_2023_df = load_csv('data/raw/Application_2016-2023.csv')