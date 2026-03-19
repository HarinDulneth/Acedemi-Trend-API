import pickle
from job_salary_prediction.data_loader import DataLoader
from job_salary_prediction.feature_engineering import FeatureEngineer
from job_salary_prediction.model import CareerPredictionModel
import os
import numpy as np
import json

def directional_accuracy(y_true, y_pred):
    """
    Computes the percentage of times the predicted direction matches the actual direction.
    """
    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(np.diff(y_pred))
    correct = actual_direction == predicted_direction
    return float(np.mean(correct) * 100)

if __name__ == '__main__':
    print('=== Job Salary Prediction Model Training & Saving Script ===')

    # 1. Load and merge data
    print('Loading and merging datasets...')
    data_loader = DataLoader(data_directory='job_salary_prediction')
    datasets = data_loader.load_all_datasets()
    comprehensive_df = data_loader.create_comprehensive_dataset()
    print(f'Comprehensive dataset shape: {comprehensive_df.shape}')

    # 2. Feature engineering
    print('Performing feature engineering...')
    feature_engineer = FeatureEngineer()
    engineered_df = feature_engineer.engineer_features(comprehensive_df)

    # 3. Prepare features and target
    print('Preparing features and target for modeling...')
    features_df, target_series = feature_engineer.prepare_features_for_modeling(
        engineered_df, target_column='starting_salary_lkr'
    )
    print(f'Features shape: {features_df.shape}, Target shape: {target_series.shape}')

    # 4. Feature selection
    print('Selecting best features...')
    selected_features_df, selected_feature_names = feature_engineer.select_best_features(
        features_df, target_series, k=20
    )
    print(f'Selected features: {selected_feature_names}')

    # 5. Train the model
    print('Training the model...')
    model = CareerPredictionModel(random_state=42)
    model.evaluate_models(selected_features_df, target_series)
    print('Model training complete.')

    # 5b. Compute and save directional accuracy
    print('Computing directional accuracy...')
    y_pred = model.predict(selected_features_df)
    da = directional_accuracy(target_series.values, y_pred)
    print(f'Directional Accuracy: {da:.2f}%')
    da_json_path = 'job_salary_prediction/directional_accuracy.json'
    with open(da_json_path, 'w') as f:
        json.dump({'directional_accuracy_percent': da}, f, indent=2)
    print(f'Directional accuracy saved to {da_json_path}')

    # 6. Save the feature engineer and model
    fe_path = 'job_salary_prediction/saved_feature_engineer.pkl'
    model_path = 'job_salary_prediction/saved_trained_model.pkl'
    print(f'Saving feature engineer to {fe_path}')
    with open(fe_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    print(f'Saving trained model to {model_path}')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print('All done! You can now use the API for predictions.') 