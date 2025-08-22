
# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model import CareerPredictionModel
import warnings
import os
warnings.filterwarnings('ignore')

def main():
    """
    Main function to orchestrate the career prediction system
    """
    print("=" * 60)
    print("CAREER PATHWAY & EMPLOYMENT OUTCOME PREDICTION SYSTEM")
    print("=" * 60)
    
    # 1. Data Loading Phase
    print("\n1. DATA LOADING PHASE")
    print("-" * 30)
    
    data_loader = DataLoader()
    datasets = data_loader.load_all_datasets()
    data_loader.get_dataset_info()
    
    # Create comprehensive dataset
    comprehensive_df = data_loader.create_comprehensive_dataset()
    
    print(f"\nComprehensive dataset created with {comprehensive_df.shape[0]} students")
    print(f"Total features: {comprehensive_df.shape[1]}")
    
    # 2. Feature Engineering Phase
    print("\n2. FEATURE ENGINEERING PHASE")
    print("-" * 30)
    
    feature_engineer = FeatureEngineer()
    
    # Engineer features
    print("Engineering features...")
    engineered_df = feature_engineer.engineer_features(comprehensive_df)
    print(f"Engineered dataset shape: {engineered_df.shape}")
    
    # Prepare features for modeling
    print("\nPreparing features for modeling...")
    try:
        features_df, target_series = feature_engineer.prepare_features_for_modeling(
            engineered_df, target_column='starting_salary_lkr'
        )
        print(f"Modeling dataset: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        print(f"Target variable: {target_series.name} (valid samples: {target_series.notna().sum()})")
        
    except Exception as e:
        print(f"Error in feature preparation: {str(e)}")
        return
    
    # Feature selection
    print("\nSelecting best features...")
    selected_features_df, selected_feature_names = feature_engineer.select_best_features(
        features_df, target_series, k=20
    )
    
    # 3. Model Building and Evaluation Phase
    print("\n3. MODEL BUILDING & EVALUATION PHASE")
    print("-" * 30)
    
    # Initialize model
    model = CareerPredictionModel(random_state=42)
    
    # Evaluate multiple models
    print("\nEvaluating multiple candidate models...")
    model_results = model.evaluate_models(selected_features_df, target_series)
    
    if not model_results.empty:
        print("\nModel Evaluation Results:")
        print(model_results.round(3))
        
        # 4. Hyperparameter Tuning Phase
        print("\n4. HYPERPARAMETER TUNING PHASE")
        print("-" * 30)
        
        try:
            tuning_results = model.tune_hyperparameters(selected_features_df, target_series)
            print("\nHyperparameter tuning completed!")
            
        except Exception as e:
            print(f"Hyperparameter tuning failed: {str(e)}")
        
        # 5. Analysis and Insights Phase
        print("\n5. ANALYSIS & INSIGHTS PHASE")
        print("-" * 30)
        
        # Feature importance analysis
        try:
            feature_importance = model.get_feature_importance(selected_feature_names)
            print("\nTop 10 Most Important Features for Salary Prediction:")
            print(feature_importance.head(10))
            
        except Exception as e:
            print(f"Feature importance analysis failed: {str(e)}")
        
        # Model summary
        model_summary = model.get_model_summary()
        print(f"\nBest performing model: {model_summary.get('best_model', 'Unknown')}")
        print(f"Total models evaluated: {model_summary.get('total_models_evaluated', 0)}")
        
        # 6. Visualization Phase
        print("\n6. VISUALIZATION PHASE")
        print("-" * 30)
        
        create_visualizations(comprehensive_df, model_results, feature_importance if 'feature_importance' in locals() else None)
        
        # 7. Prediction Examples
        print("\n7. PREDICTION EXAMPLES")
        print("-" * 30)
        
        # Make sample predictions
        sample_predictions = make_sample_predictions(model, selected_features_df, target_series)
        
        print("\nSample Salary Predictions:")
        for i, pred in enumerate(sample_predictions[:5]):
            print(f"Student {i+1}: LKR {pred:,.0f}")
        
    else:
        print("No models were successfully evaluated!")
    
    print("\n" + "=" * 60)
    print("CAREER PREDICTION SYSTEM ANALYSIS COMPLETED")
    print("=" * 60)

def create_visualizations(df: pd.DataFrame, model_results: pd.DataFrame, feature_importance: pd.DataFrame = None):
    """
    Create visualizations for the analysis
    
    Args:
        df: Comprehensive dataset
        model_results: Model evaluation results
        feature_importance: Feature importance scores
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Model Performance Comparison
    if not model_results.empty:
        plt.subplot(2, 3, 1)
        plt.bar(range(len(model_results)), model_results['test_rmse'])
        plt.xticks(range(len(model_results)), model_results['model'], rotation=45, ha='right')
        plt.title('Model Performance Comparison (Test RMSE)')
        plt.ylabel('RMSE')
        
        # R² comparison
        plt.subplot(2, 3, 2)
        plt.bar(range(len(model_results)), model_results['test_r2'])
        plt.xticks(range(len(model_results)), model_results['model'], rotation=45, ha='right')
        plt.title('Model Performance Comparison (R²)')
        plt.ylabel('R² Score')
    
    # 2. Salary Distribution by Pathway
    if 'starting_salary_lkr' in df.columns and 'pathway' in df.columns:
        plt.subplot(2, 3, 3)
        placed_students = df[df['placed'] == True]
        if not placed_students.empty:
            sns.boxplot(data=placed_students, x='pathway', y='starting_salary_lkr')
            plt.xticks(rotation=45, ha='right')
            plt.title('Salary Distribution by Pathway')
            plt.ylabel('Starting Salary (LKR)')
    
    # 3. Feature Importance
    if feature_importance is not None and not feature_importance.empty:
        plt.subplot(2, 3, 4)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance Score')
    
    # 4. GPA vs Salary correlation
    if 'cumulative_gpa' in df.columns and 'starting_salary_lkr' in df.columns:
        plt.subplot(2, 3, 5)
        placed_students = df[(df['placed'] == True) & (df['cumulative_gpa'].notna())]
        if not placed_students.empty:
            plt.scatter(placed_students['cumulative_gpa'], placed_students['starting_salary_lkr'], alpha=0.6)
            plt.xlabel('Cumulative GPA')
            plt.ylabel('Starting Salary (LKR)')
            plt.title('GPA vs Starting Salary')
            
            # Add trend line
            z = np.polyfit(placed_students['cumulative_gpa'], placed_students['starting_salary_lkr'], 1)
            p = np.poly1d(z)
            plt.plot(placed_students['cumulative_gpa'], p(placed_students['cumulative_gpa']), "r--", alpha=0.8)
    
    # 5. Placement Rate by Province
    if 'province' in df.columns and 'placed' in df.columns:
        plt.subplot(2, 3, 6)
        placement_by_province = df.groupby('province')['placed'].mean().sort_values(ascending=False)
        plt.bar(range(len(placement_by_province)), placement_by_province.values)
        plt.xticks(range(len(placement_by_province)), placement_by_province.index, rotation=45, ha='right')
        plt.title('Placement Rate by Province')
        plt.ylabel('Placement Rate')
    
    plt.tight_layout()
    plt.savefig('career_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'career_prediction_analysis.png'")

def make_sample_predictions(model: CareerPredictionModel, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Make sample predictions using the trained model
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target variable
        
    Returns:
        Array of sample predictions
    """
    # Select random samples
    sample_indices = np.random.choice(X.index, size=min(10, len(X)), replace=False)
    sample_features = X.loc[sample_indices]
    
    # Make predictions
    predictions = model.predict(sample_features)
    
    return predictions

if __name__ == "__main__":
    main()