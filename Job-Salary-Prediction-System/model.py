# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

class CareerPredictionModel:
    """
    Advanced machine learning models for career outcome prediction
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model with configuration
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.model_results = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize multiple candidate models for comparison
        
        Returns:
            Dictionary of model name to model object mappings
        """
        self.models = {
            # Linear Models
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            
            # Tree-based Models
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            ),
            'xgboost': xgb.XGBRegressor(
                random_state=self.random_state, eval_metric='rmse'
            ),
            'lightgbm': lgb.LGBMRegressor(
                random_state=self.random_state, verbose=-1
            ),
            
            # Support Vector Machine
            'svr_rbf': SVR(kernel='rbf'),
            'svr_linear': SVR(kernel='linear'),
            
            # Neural Network
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500, 
                random_state=self.random_state
            )
        }
        
        return self.models
    
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series, 
                       test_size: float = 0.2, cv_folds: int = 5) -> pd.DataFrame:
        """
        Evaluate all models using cross-validation and test set
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with model evaluation results
        """
        print("Evaluating models...")
        print("=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Initialize models if not done
        if not self.models:
            self.initialize_models()
        
        results = []
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            try:
                # Create pipeline with scaling
                if name in ['svr_rbf', 'svr_linear', 'mlp']:
                    # These models need scaling
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                else:
                    # Tree-based models don't need scaling
                    pipeline = Pipeline([
                        ('model', model)
                    ])
                
                # Cross-validation
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, cv=cv_folds, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                cv_rmse = np.sqrt(-cv_scores)
                
                # Fit and predict on test set
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_mae = mean_absolute_error(y_test, y_pred)
                test_r2 = r2_score(y_test, y_pred)
                
                # Store results
                result = {
                    'model': name,
                    'cv_rmse_mean': cv_rmse.mean(),
                    'cv_rmse_std': cv_rmse.std(),
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                }
                results.append(result)
                
                # Store model and pipeline for later use
                self.model_results[name] = {
                    'pipeline': pipeline,
                    'metrics': result
                }
                
                print(f"  CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
                print(f"  Test RMSE: {test_rmse:.2f}")
                print(f"  Test R²: {test_r2:.3f}")
                
            except Exception as e:
                print(f"  Error evaluating {name}: {str(e)}")
                continue
        
        # Convert to DataFrame and sort by test RMSE
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('test_rmse')
            
            # Identify best model
            self.best_model_name = results_df.iloc[0]['model']
            self.best_model = self.model_results[self.best_model_name]['pipeline']
            
            print(f"\nBest Model: {self.best_model_name}")
            print("=" * 50)
        
        return results_df
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                           model_name: str = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for the best model or specified model
        
        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name of model to tune (uses best model if None)
            
        Returns:
            Dictionary with tuning results
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name is None:
            raise ValueError("No best model found. Run evaluate_models first.")
        
        print(f"Tuning hyperparameters for {model_name}...")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7, -1],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__num_leaves': [31, 50, 100]
            },
            'ridge': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'model__alpha': [0.01, 0.1, 1.0, 10.0]
            },
            'svr_rbf': {
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return {}
        
        # Get the base pipeline
        base_pipeline = self.model_results[model_name]['pipeline']
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_pipeline, param_grids[model_name],
            cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Best CV RMSE: {results['best_score']:.2f}")
        
        return results
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from the best model
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run evaluate_models first.")
        
        # Get the actual model from pipeline
        model = self.best_model.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_)
        else:
            print("Model doesn't support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.best_model is None:
            raise ValueError("No best model found. Train a model first.")
        
        return self.best_model.predict(X)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of model evaluation results
        
        Returns:
            Dictionary with model summary information
        """
        if not self.model_results:
            return {}
        
        summary = {
            'best_model': self.best_model_name,
            'total_models_evaluated': len(self.model_results),
            'model_rankings': []
        }
        
        # Rank models by test RMSE
        for name, results in self.model_results.items():
            summary['model_rankings'].append({
                'model': name,
                'test_rmse': results['metrics']['test_rmse'],
                'test_r2': results['metrics']['test_r2']
            })
        
        summary['model_rankings'] = sorted(
            summary['model_rankings'], 
            key=lambda x: x['test_rmse']
        )
        
        return summary

