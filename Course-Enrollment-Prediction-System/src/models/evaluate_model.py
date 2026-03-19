import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential  # type: ignore
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from src.utils.helper_functions import load_csv, scale_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation class with advanced metrics and visualization."""
    
    def __init__(self, models_dir: str = 'models/trained_models'):
        self.models_dir = Path(models_dir)
        self.results_dir = Path('results')
        self.figures_dir = Path('results/figures')
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories for results and figures."""
        time_series_models = ['prophet', 'arima', 'sarima']
        dirs = [self.results_dir, self.figures_dir, 
                self.figures_dir / 'individual_models', 
                self.figures_dir / 'comparisons']
        for model_type in time_series_models:
            dirs.append(self.figures_dir / 'individual_models' / model_type)
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_models(self) -> None:
        """Load all trained models from the models directory."""
        logger.info("Loading trained models...")
        
        model_files = list(self.models_dir.glob('*.pkl'))
        loaded_count = 0
        
        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                self.models[model_file.stem] = model
                loaded_count += 1
                
                # Load scalers for LSTM models
                if 'lstm_scalers' in model_file.stem:
                    self.scalers[model_file.stem] = model
                    
            except Exception as e:
                logger.warning(f"Failed to load model {model_file.stem}: {e}")
        
        logger.info(f"Successfully loaded {loaded_count} models")
    
    def prepare_data_for_evaluation(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model evaluation (same as training preparation)."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Define columns to exclude
        base_exclude = ['year', 'university', 'course_name']
        exclude_cols = base_exclude + [target_col]
        
        # Prevent data leakage for applications prediction
        if target_col == 'applications':
            leakage_cols = [col for col in df.columns 
                          if 'application' in col.lower() and col != 'applications']
            exclude_cols.extend(leakage_cols)
        
        # Select features
        available_exclude = [col for col in exclude_cols if col in df.columns]
        X = df.drop(columns=available_exclude)
        X = X.select_dtypes(include=[np.number])
        
        # Handle problematic values
        X = self._clean_numeric_data(X)
        y = df[target_col].copy()
        
        # Remove rows with missing target values
        valid_mask = ~y.isna()
        X, y = X[valid_mask], y[valid_mask]
        
        logger.info(f"Prepared evaluation data for {target_col}: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _clean_numeric_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric data by handling infinite and extreme values."""
        X = X.copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Clip extreme values
        max_val = 1e6
        X = X.clip(lower=-max_val, upper=max_val)
        
        # Handle missing values (simple median imputation)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Handle any remaining NaN or inf values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions for metric calculation")
            return {'error': 'no_valid_predictions'}
        
        metrics = {}
        
        try:
            # Basic regression metrics
            metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
            metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
            
            # MAPE (handling division by zero)
            mape_mask = y_true_clean != 0
            if mape_mask.any():
                metrics['mape'] = mean_absolute_percentage_error(
                    y_true_clean[mape_mask], y_pred_clean[mape_mask]
                ) * 100
            else:
                metrics['mape'] = np.nan
            
            # Additional metrics
            metrics['max_error'] = np.max(np.abs(y_true_clean - y_pred_clean))
            metrics['mean_residual'] = np.mean(y_true_clean - y_pred_clean)
            metrics['std_residual'] = np.std(y_true_clean - y_pred_clean)
            
            # Directional accuracy (for time series)
            if len(y_true_clean) > 1:
                true_direction = np.diff(y_true_clean) > 0
                pred_direction = np.diff(y_pred_clean) > 0
                metrics['directional_accuracy'] = np.mean(true_direction == pred_direction) * 100
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def evaluate_traditional_models(self, df: pd.DataFrame, target_col: str) -> Dict[str, Dict]:
        """Evaluate Random Forest and XGBoost models with cross-validation."""
        logger.info(f"Evaluating traditional ML models for {target_col}")
        
        results = {}
        X, y = self.prepare_data_for_evaluation(df, target_col)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name in ['random_forest', 'xgboost']:
            model_key = f'{model_name}_{target_col}'
            
            if model_key not in self.models:
                logger.warning(f"Model {model_key} not found")
                continue
            
            model = self.models[model_key]
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X, y, cv=tscv, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                
                # Full dataset prediction
                y_pred = model.predict(X)
                metrics = self.calculate_comprehensive_metrics(y.values, y_pred)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                    top_features = dict(sorted(feature_importance.items(), 
                                             key=lambda x: x[1], reverse=True)[:10])
                else:
                    top_features = {}
                
                results[model_name] = {
                    'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
                    'cv_rmse_std': np.sqrt(cv_scores.std()),
                    'metrics': metrics,
                    'feature_importance': top_features,
                    'predictions': y_pred.tolist(),
                    'actuals': y.values.tolist()
                }
                
                logger.info(f"{model_name} {target_col} - CV RMSE: {results[model_name]['cv_rmse_mean']:.2f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} for {target_col}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def prepare_lstm_data_for_evaluation(self, df: pd.DataFrame, target_col: str, 
                                       sequence_length: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare LSTM data for evaluation (matching training preparation)."""
        X, y = [], []
        
        # Get grouping columns (same logic as training)
        if target_col == 'applications':
            group_cols = ['university_applications_target_encoded', 'course_name_applications_target_encoded']
        else:
            group_cols = ['university_enrollments_target_encoded', 'course_name_enrollments_target_encoded']
        
        # Check if group columns exist
        valid_group_cols = [col for col in group_cols if col in df.columns]
        
        if not valid_group_cols:
            # Fallback grouping
            fallback_cols = []
            if 'university' in df.columns:
                fallback_cols.append('university')
            if 'course_name' in df.columns:
                fallback_cols.append('course_name')
            group_cols = fallback_cols if fallback_cols else ['dummy_group']
            
            if group_cols == ['dummy_group']:
                df = df.copy()
                df['dummy_group'] = 'all'
        
        # Load scalers for this target
        scaler_key = f'lstm_scalers_{target_col}'
        if scaler_key in self.scalers:
            scalers_dict = self.scalers[scaler_key]
        else:
            logger.warning(f"No scalers found for {target_col}, using default scaling")
            scalers_dict = {}
        
        for group_key, group_df in df.groupby(group_cols):
            if len(group_df) <= sequence_length:
                continue
            
            group_df = group_df.sort_values('year')
            
            # Prepare features (same as training)
            exclude_cols = [target_col, 'year', 'university', 'course_name']
            if target_col == 'applications':
                exclude_cols.extend([col for col in df.columns 
                                   if 'application' in col.lower() and col != 'applications'])
            
            features = group_df.drop(columns=[col for col in exclude_cols if col in df.columns])
            features = features.select_dtypes(include=[np.number])
            features = self._clean_numeric_data(features)
            
            if features.empty:
                continue
            
            # Scale features
            scaler_group_key = f"{target_col}_{group_key}"
            if scaler_group_key in scalers_dict:
                scaler = scalers_dict[scaler_group_key]
                scaled_features = scaler.transform(features)
            else:
                # Fallback: create new scaler
                scaled_features, _ = scale_features(features, features.columns.tolist())
            
            target_array = group_df[target_col].values
            
            # Create sequences
            for i in range(len(scaled_features) - sequence_length):
                X.append(scaled_features[i:i + sequence_length])
                y.append(target_array[i + sequence_length])
        
        if not X:
            raise ValueError(f"No valid sequences created for {target_col} evaluation")
        
        return np.array(X), np.array(y)
    
    def evaluate_lstm_models(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Evaluate LSTM models with time series validation."""
        logger.info(f"Evaluating LSTM model for {target_col}")
        
        model_key = f'lstm_{target_col}'
        
        if model_key not in self.models:
            logger.warning(f"LSTM model {model_key} not found")
            return {'error': 'model_not_found'}
        
        model = self.models[model_key]
        
        try:
            # Prepare LSTM data
            X, y = self.prepare_lstm_data_for_evaluation(df, target_col)
            
            # Time series split for LSTM evaluation
            n_samples = len(X)
            train_size = int(0.8 * n_samples)
            
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Predictions
            y_pred_train = model.predict(X_train).flatten()
            y_pred_test = model.predict(X_test).flatten()
            
            # Calculate metrics
            train_metrics = self.calculate_comprehensive_metrics(y_train, y_pred_train)
            test_metrics = self.calculate_comprehensive_metrics(y_test, y_pred_test)
            
            results = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'train_predictions': y_pred_train.tolist(),
                'test_predictions': y_pred_test.tolist(),
                'train_actuals': y_train.tolist(),
                'test_actuals': y_test.tolist(),
                'train_size': len(y_train),
                'test_size': len(y_test)
            }
            
            logger.info(f"LSTM {target_col} - Test RMSE: {test_metrics.get('rmse', 'N/A'):.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating LSTM for {target_col}: {e}")
            return {'error': str(e)}
    
    def evaluate_time_series_models(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Evaluate Prophet, ARIMA, and SARIMA models."""
        logger.info(f"Evaluating time series models for {target_col}")
        
        results = {}
        
        # Prepare time series data
        ts_data = self._prepare_time_series_data(df, target_col)
        
        # Evaluate each model type
        for model_type in ['prophet', 'arima', 'sarima']:
            model_key = f'{model_type}_{target_col}'
            
            if model_key not in self.models:
                logger.warning(f"Model {model_key} not found")
                continue
            
            models_dict = self.models[model_key]
            
            if isinstance(models_dict, dict) and models_dict:
                results[model_type] = self._evaluate_grouped_time_series_models(
                    models_dict, ts_data, model_type
                )
            else:
                logger.warning(f"No valid models found for {model_type} {target_col}")
                results[model_type] = {'error': 'no_valid_models'}
        
        return results
    
    def _prepare_time_series_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare time series data for evaluation."""
        # Get grouping columns
        if target_col == 'applications':
            group_cols = ['university_applications_target_encoded', 'course_name_applications_target_encoded']
        else:
            group_cols = ['university_enrollments_target_encoded', 'course_name_enrollments_target_encoded']
        
        valid_group_cols = [col for col in group_cols if col in df.columns]
        
        if not valid_group_cols:
            # Fallback grouping
            if 'university' in df.columns and 'course_name' in df.columns:
                group_cols = ['university', 'course_name']
            else:
                group_cols = ['dummy_group']
                df = df.copy()
                df['dummy_group'] = 'all'
        else:
            group_cols = valid_group_cols
        
        ts_data = []
        for group_key, group_df in df.groupby(group_cols):
            if len(group_df) < 3:
                continue
            
            group_ts = group_df[['year', target_col]].copy().sort_values('year')
            group_ts['ds'] = pd.to_datetime(group_ts['year'].astype(str) + '-01-01')
            group_ts['y'] = group_ts[target_col]
            group_ts['group_key'] = str(group_key)
            ts_data.append(group_ts[['ds', 'y', 'group_key']])
        
        if ts_data:
            return pd.concat(ts_data, ignore_index=True)
        else:
            # Create aggregate data
            agg_data = df.groupby('year')[target_col].mean().reset_index()
            agg_data['ds'] = pd.to_datetime(agg_data['year'].astype(str) + '-01-01')
            agg_data['y'] = agg_data[target_col]
            agg_data['group_key'] = 'aggregate'
            return agg_data[['ds', 'y', 'group_key']]
    
    def _evaluate_grouped_time_series_models(self, models_dict: Dict, ts_data: pd.DataFrame, 
                                           model_type: str) -> Dict[str, Any]:
        """Evaluate grouped time series models (Prophet, ARIMA, SARIMA)."""
        group_results = {}
        all_predictions = []
        all_actuals = []
        
        for group_key, group_df in ts_data.groupby('group_key'):
            if group_key not in models_dict:
                continue
            
            model = models_dict[group_key]
            group_data = group_df.sort_values('ds').reset_index(drop=True)
            
            try:
                if model_type == 'prophet':
                    predictions = self._evaluate_prophet_model(model, group_data)
                else:  # ARIMA or SARIMA
                    predictions = self._evaluate_arima_sarima_model(model, group_data)
                
                if predictions is not None:
                    actuals = group_data['y'].values
                    metrics = self.calculate_comprehensive_metrics(actuals, predictions)
                    
                    group_results[group_key] = {
                        'metrics': metrics,
                        'predictions': predictions.tolist(),
                        'actuals': actuals.tolist()
                    }
                    
                    all_predictions.extend(predictions)
                    all_actuals.extend(actuals)
                
            except Exception as e:
                logger.warning(f"Error evaluating {model_type} for group {group_key}: {e}")
                continue
        
        # Overall metrics
        overall_metrics = {}
        if all_predictions and all_actuals:
            overall_metrics = self.calculate_comprehensive_metrics(
                np.array(all_actuals), np.array(all_predictions)
            )
        
        return {
            'group_results': group_results,
            'overall_metrics': overall_metrics,
            'num_groups': len(group_results)
        }
    
    def _evaluate_prophet_model(self, model: Prophet, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Evaluate a single Prophet model."""
        try:
            forecast = model.predict(data[['ds']])
            return forecast['yhat'].values
        except Exception as e:
            logger.warning(f"Prophet evaluation error: {e}")
            return None
    
    def _evaluate_arima_sarima_model(self, model: Any, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Evaluate a single ARIMA/SARIMA model."""
        try:
            # For fitted ARIMA/SARIMA models, get fitted values
            if hasattr(model, 'fittedvalues'):
                fitted_values = model.fittedvalues
                # Check if it's already a numpy array or pandas Series
                if isinstance(fitted_values, np.ndarray):
                    return fitted_values
                elif hasattr(fitted_values, 'values'):
                    return fitted_values.values
                else:
                    return np.array(fitted_values)
                    
            elif hasattr(model, 'predict'):
                predictions = model.predict(start=0, end=len(data)-1)
                # Handle the return type properly
                if isinstance(predictions, np.ndarray):
                    return predictions
                elif hasattr(predictions, 'values'):
                    return predictions.values
                else:
                    return np.array(predictions)
            else:
                logger.warning("Model has no fittedvalues or predict method")
                return None
                
        except Exception as e:
            logger.warning(f"ARIMA/SARIMA evaluation error: {e}")
            return None
    
    def create_evaluation_visualizations(self, target_col: str) -> None:
        """Create comprehensive visualization plots for model evaluation."""
        logger.info(f"Creating evaluation visualizations for {target_col}")
        
        if target_col not in self.evaluation_results:
            logger.warning(f"No evaluation results found for {target_col}")
            return
        
        results = self.evaluation_results[target_col]
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        self._plot_model_performance_comparison(results, target_col)
        
        # 2. Prediction vs Actual plots for each model (including time series)
        self._plot_predictions_vs_actuals(results, target_col)
        
        # 3. Residual analysis (including time series)
        self._plot_residual_analysis(results, target_col)
        
        # 4. Feature importance (for tree-based models)
        self._plot_feature_importance(results, target_col)
        
        # 5. Time series specific plots
        self._plot_time_series_forecasts(results, target_col)
        
        logger.info(f"Visualizations saved for {target_col}")
    
    def _plot_model_performance_comparison(self, results: Dict, target_col: str) -> None:
        """Create model performance comparison plots."""
        metrics_to_plot = ['rmse', 'mae', 'r2', 'mape']
        model_names = []
        metric_values = {metric: [] for metric in metrics_to_plot}
        
        # Collect metrics from all models
        for model_type, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                model_names.append(model_type.replace('_', ' ').title())
                
                # Get metrics from appropriate source
                metrics = None
                if 'metrics' in model_results:
                    metrics = model_results['metrics']
                elif 'test_metrics' in model_results:
                    metrics = model_results['test_metrics']
                elif 'overall_metrics' in model_results:
                    metrics = model_results['overall_metrics']
                
                if metrics:
                    for metric in metrics_to_plot:
                        value = metrics.get(metric, np.nan)
                        metric_values[metric].append(value)
                else:
                    # If no metrics found, add NaN values
                    for metric in metrics_to_plot:
                        metric_values[metric].append(np.nan)
        
        if not model_names:
            logger.warning(f"No valid model results found for comparison plot ({target_col})")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Performance Comparison - {target_col.title()}', fontsize=16)
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            values = metric_values[metric]
            
            # Filter out NaN values
            valid_indices = [j for j, v in enumerate(values) if not np.isnan(v)]
            valid_names = [model_names[j] for j in valid_indices]
            valid_values = [values[j] for j in valid_indices]
            
            if valid_values:
                bars = ax.bar(valid_names, valid_values)
                ax.set_title(f'{metric.upper()}')
                ax.set_ylabel(metric.upper())
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, valid_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'comparisons' / f'model_comparison_{target_col}.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions_vs_actuals(self, results: Dict, target_col: str) -> None:
        """Create prediction vs actual scatter plots for each model."""
        for model_type, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                
                # Handle traditional ML and LSTM models
                predictions, actuals = None, None
                
                if 'predictions' in model_results and 'actuals' in model_results:
                    predictions = np.array(model_results['predictions'])
                    actuals = np.array(model_results['actuals'])
                elif 'test_predictions' in model_results and 'test_actuals' in model_results:
                    predictions = np.array(model_results['test_predictions'])
                    actuals = np.array(model_results['test_actuals'])
                # Handle time series models (Prophet, ARIMA, SARIMA)
                elif 'group_results' in model_results:
                    all_predictions = []
                    all_actuals = []
                    
                    for group_key, group_data in model_results['group_results'].items():
                        if 'predictions' in group_data and 'actuals' in group_data:
                            all_predictions.extend(group_data['predictions'])
                            all_actuals.extend(group_data['actuals'])
                    
                    if all_predictions and all_actuals:
                        predictions = np.array(all_predictions)
                        actuals = np.array(all_actuals)
                
                if predictions is not None and actuals is not None:
                    self._create_prediction_scatter_plot(
                        actuals, predictions, model_type, target_col
                    )
    
    def _create_prediction_scatter_plot(self, actuals: np.ndarray, predictions: np.ndarray, 
                                      model_type: str, target_col: str) -> None:
        """Create a single prediction vs actual scatter plot."""
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(actuals, predictions, alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_val = min(np.min(actuals), np.min(predictions))
        max_val = max(np.max(actuals), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(actuals, predictions)
        
        plt.xlabel(f'Actual {target_col.title()}')
        plt.ylabel(f'Predicted {target_col.title()}')
        plt.title(f'{model_type.replace("_", " ").title()} - Predictions vs Actuals\n'
                 f'{target_col.title()} (R² = {r2:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(actuals, predictions)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'individual_models' / 
                   f'{model_type}_{target_col}_predictions_vs_actuals.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residual_analysis(self, results: Dict, target_col: str) -> None:
        """Create residual analysis plots."""
        for model_type, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                
                # Handle traditional ML and LSTM models
                predictions, actuals = None, None
                
                if 'predictions' in model_results and 'actuals' in model_results:
                    predictions = np.array(model_results['predictions'])
                    actuals = np.array(model_results['actuals'])
                elif 'test_predictions' in model_results and 'test_actuals' in model_results:
                    predictions = np.array(model_results['test_predictions'])
                    actuals = np.array(model_results['test_actuals'])
                # Handle time series models (Prophet, ARIMA, SARIMA)
                elif 'group_results' in model_results:
                    all_predictions = []
                    all_actuals = []
                    
                    for group_key, group_data in model_results['group_results'].items():
                        if 'predictions' in group_data and 'actuals' in group_data:
                            all_predictions.extend(group_data['predictions'])
                            all_actuals.extend(group_data['actuals'])
                    
                    if all_predictions and all_actuals:
                        predictions = np.array(all_predictions)
                        actuals = np.array(all_actuals)
                
                if predictions is not None and actuals is not None:
                    residuals = actuals - predictions
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Residuals vs Predicted
                    ax1.scatter(predictions, residuals, alpha=0.6)
                    ax1.axhline(y=0, color='r', linestyle='--')
                    ax1.set_xlabel('Predicted Values')
                    ax1.set_ylabel('Residuals')
                    ax1.set_title(f'Residuals vs Predicted - {model_type.title()}')
                    ax1.grid(True, alpha=0.3)
                    
                    # Histogram of residuals
                    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
                    ax2.set_xlabel('Residuals')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title(f'Distribution of Residuals - {model_type.title()}')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.suptitle(f'Residual Analysis - {target_col.title()}')
                    plt.tight_layout()
                    plt.savefig(self.figures_dir / 'individual_models' / 
                            f'{model_type}_{target_col}_residual_analysis.png', 
                            dpi=300, bbox_inches='tight')
                    plt.close()

    def _plot_time_series_forecasts(self, results: Dict, target_col: str) -> None:
        """Create time series forecast plots for Prophet, ARIMA, and SARIMA models."""
        for model_type in ['prophet', 'arima', 'sarima']:
            if model_type in results and 'group_results' in results[model_type]:
                group_results = results[model_type]['group_results']
                
                # Create individual plots for each group
                for group_key, group_data in group_results.items():
                    if 'predictions' in group_data and 'actuals' in group_data:
                        predictions = np.array(group_data['predictions'])
                        actuals = np.array(group_data['actuals'])
                        
                        plt.figure(figsize=(12, 6))
                        
                        # Create time index (assuming yearly data)
                        time_index = range(len(actuals))
                        
                        plt.plot(time_index, actuals, 'o-', label='Actual', linewidth=2, markersize=6)
                        plt.plot(time_index, predictions, 's-', label='Predicted', linewidth=2, markersize=6)
                        
                        plt.xlabel('Time Period')
                        plt.ylabel(f'{target_col.title()}')
                        plt.title(f'{model_type.upper()} Forecast - {target_col.title()}\nGroup: {group_key}')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Add metrics text
                        if 'metrics' in group_data:
                            metrics = group_data['metrics']
                            rmse = metrics.get('rmse', 'N/A')
                            r2 = metrics.get('r2', 'N/A')
                            plt.text(0.02, 0.98, f'RMSE: {rmse:.3f}\nR²: {r2:.3f}', 
                                    transform=plt.gca().transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        
                        plt.tight_layout()
                        
                        # Save individual group plot
                        safe_group_key = str(group_key).replace('/', '_').replace('\\', '_')
                        plt.savefig(self.figures_dir / 'individual_models' / model_type / 
                                f'{model_type}_{target_col}_forecast_{safe_group_key}.png', 
                                dpi=300, bbox_inches='tight')
                        plt.close()
                
                # Create combined plot for all groups
                if len(group_results) > 1:
                    self._plot_combined_time_series_forecasts(group_results, model_type, target_col)

    def _plot_combined_time_series_forecasts(self, group_results: Dict, model_type: str, target_col: str) -> None:
        """Create a combined plot showing all groups for a time series model."""
        fig, axes = plt.subplots(min(len(group_results), 4), 1, figsize=(12, 4 * min(len(group_results), 4)))
        if len(group_results) == 1:
            axes = [axes]
        
        fig.suptitle(f'{model_type.upper()} Forecasts - {target_col.title()} (All Groups)', fontsize=16)
        
        for idx, (group_key, group_data) in enumerate(list(group_results.items())[:4]):  # Limit to 4 groups
            if 'predictions' in group_data and 'actuals' in group_data:
                predictions = np.array(group_data['predictions'])
                actuals = np.array(group_data['actuals'])
                
                ax = axes[idx] if len(group_results) > 1 else axes[0]
                time_index = range(len(actuals))
                
                ax.plot(time_index, actuals, 'o-', label='Actual', linewidth=2, markersize=4)
                ax.plot(time_index, predictions, 's-', label='Predicted', linewidth=2, markersize=4)
                
                ax.set_title(f'Group: {group_key}')
                ax.set_ylabel(f'{target_col.title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add metrics
                if 'metrics' in group_data:
                    metrics = group_data['metrics']
                    rmse = metrics.get('rmse', 'N/A')
                    ax.text(0.02, 0.98, f'RMSE: {rmse:.2f}', 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        if len(group_results) > 1:
            axes[-1].set_xlabel('Time Period')
        else:
            axes[0].set_xlabel('Time Period')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'individual_models' / 
                f'{model_type}_{target_col}_combined_forecasts.png', 
                dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, results: Dict, target_col: str) -> None:
        """Create feature importance plots for tree-based models."""
        for model_type in ['random_forest', 'xgboost']:
            if model_type in results and 'feature_importance' in results[model_type]:
                importance_dict = results[model_type]['feature_importance']
                
                if importance_dict:
                    features = list(importance_dict.keys())
                    importances = list(importance_dict.values())
                    
                    plt.figure(figsize=(12, 8))
                    bars = plt.barh(features, importances)
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top 10 Feature Importance - {model_type.replace("_", " ").title()}\n'
                             f'{target_col.title()}')
                    plt.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, importance in zip(bars, importances):
                        width = bar.get_width()
                        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                                f'{importance:.3f}', ha='left', va='center')
                    
                    plt.tight_layout()
                    plt.savefig(self.figures_dir / 'individual_models' / 
                               f'{model_type}_{target_col}_feature_importance.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
    
    def generate_evaluation_report(self, target_col: str) -> str:
        """Generate a comprehensive evaluation report."""
        if target_col not in self.evaluation_results:
            return f"No evaluation results available for {target_col}"
        
        results = self.evaluation_results[target_col]
        report = []
        
        report.append(f"# Model Evaluation Report - {target_col.title()}")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("## Summary")
        report.append(f"- Target Variable: {target_col}")
        report.append(f"- Models Evaluated: {len([k for k, v in results.items() if 'error' not in v])}")
        report.append(f"- Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model performance summary
        report.append("## Model Performance Summary")
        report.append("")
        
        for model_type, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                report.append(f"### {model_type.replace('_', ' ').title()}")
                
                # Get metrics
                metrics = None
                if 'metrics' in model_results:
                    metrics = model_results['metrics']
                elif 'test_metrics' in model_results:
                    metrics = model_results['test_metrics']
                elif 'overall_metrics' in model_results:
                    metrics = model_results['overall_metrics']
                
                if metrics and 'error' not in metrics:
                    report.append(f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                    report.append(f"- MAE: {metrics.get('mae', 'N/A'):.4f}")
                    report.append(f"- R²: {metrics.get('r2', 'N/A'):.4f}")
                    report.append(f"- MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                    
                    if 'directional_accuracy' in metrics:
                        report.append(f"- Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
                
                # Cross-validation results for traditional models
                if 'cv_rmse_mean' in model_results:
                    report.append(f"- Cross-Validation RMSE: {model_results['cv_rmse_mean']:.4f} "
                                f"(±{model_results['cv_rmse_std']:.4f})")
                
                # Feature importance for tree-based models
                if 'feature_importance' in model_results and model_results['feature_importance']:
                    report.append("- Top 5 Features:")
                    for i, (feature, importance) in enumerate(list(model_results['feature_importance'].items())[:5]):
                        report.append(f"  {i+1}. {feature}: {importance:.4f}")
                
                report.append("")
        
        # Best model recommendation
        report.append("## Model Recommendation")
        best_model = self._find_best_model(results)
        if best_model:
            report.append(f"**Recommended Model: {best_model['name'].replace('_', ' ').title()}**")
            report.append(f"- RMSE: {best_model['rmse']:.4f}")
            report.append(f"- R²: {best_model['r2']:.4f}")
            report.append("")
            report.append("**Rationale:**")
            report.append(f"- Lowest RMSE among all evaluated models")
            report.append(f"- Good balance between accuracy and generalization")
        else:
            report.append("Unable to determine best model due to insufficient data.")
        
        report.append("")
        report.append("## Notes")
        report.append("- All metrics calculated on the full dataset or appropriate test splits")
        report.append("- Cross-validation used for traditional ML models where applicable")
        report.append("- Time series validation used for sequential models")
        report.append("- Feature importance shown for tree-based models only")
        
        return "\n".join(report)
    
    def _find_best_model(self, results: Dict) -> Optional[Dict]:
        """Find the best performing model based on RMSE."""
        best_model = None
        best_rmse = float('inf')
        
        for model_type, model_results in results.items():
            if isinstance(model_results, dict) and 'error' not in model_results:
                
                # Get metrics
                metrics = None
                if 'metrics' in model_results:
                    metrics = model_results['metrics']
                elif 'test_metrics' in model_results:
                    metrics = model_results['test_metrics']
                elif 'overall_metrics' in model_results:
                    metrics = model_results['overall_metrics']
                
                if metrics and 'rmse' in metrics:
                    rmse = metrics['rmse']
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = {
                            'name': model_type,
                            'rmse': rmse,
                            'r2': metrics.get('r2', 0)
                        }
        
        return best_model
    
    def save_evaluation_results(self, target_col: str) -> None:
        """Save evaluation results to JSON file."""
        if target_col not in self.evaluation_results:
            logger.warning(f"No evaluation results to save for {target_col}")
            return
        
        results_file = self.results_dir / f'evaluation_results_{target_col}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.evaluation_results[target_col])
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def run_comprehensive_evaluation(self, df: pd.DataFrame, target_cols: List[str]) -> Dict[str, Any]:
        """Run comprehensive evaluation for all target columns."""
        logger.info("Starting comprehensive model evaluation")
        
        # Load all models
        self.load_models()
        
        if not self.models:
            logger.error("No models found for evaluation")
            return {}
        
        results_summary = {}
        
        for target_col in target_cols:
            logger.info(f"Evaluating models for {target_col}")
            
            try:
                # Initialize results for this target
                self.evaluation_results[target_col] = {}
                
                # Evaluate traditional ML models
                traditional_results = self.evaluate_traditional_models(df, target_col)
                self.evaluation_results[target_col].update(traditional_results)
                
                # Evaluate LSTM models
                lstm_results = self.evaluate_lstm_models(df, target_col)
                self.evaluation_results[target_col]['lstm'] = lstm_results
                
                # Evaluate time series models
                ts_results = self.evaluate_time_series_models(df, target_col)
                self.evaluation_results[target_col].update(ts_results)
                
                # Create visualizations
                self.create_evaluation_visualizations(target_col)
                
                # Generate and save report
                report = self.generate_evaluation_report(target_col)
                report_file = self.results_dir / f'evaluation_report_{target_col}.txt'
                with open(report_file, 'w') as f:
                    f.write(report)
                
                # Save results
                self.save_evaluation_results(target_col)
                
                # Summary for this target
                best_model = self._find_best_model(self.evaluation_results[target_col])
                results_summary[target_col] = {
                    'best_model': best_model['name'] if best_model else 'N/A',
                    'best_rmse': best_model['rmse'] if best_model else 'N/A',
                    'models_evaluated': len([k for k, v in self.evaluation_results[target_col].items() 
                                           if isinstance(v, dict) and 'error' not in v])
                }
                
                logger.info(f"Completed evaluation for {target_col}")
                
            except Exception as e:
                logger.error(f"Error evaluating models for {target_col}: {e}")
                results_summary[target_col] = {'error': str(e)}
        
        logger.info("Comprehensive evaluation completed")
        return results_summary


def main():
    """Main function to run model evaluation."""
    
    # Configuration
    DATA_FILE = 'data/processed/final_dataset.csv'
    TARGET_COLS = ['applications', 'enrollments']
    MODELS_DIR = 'models/trained_models'
    
    logger.info("Starting model evaluation process")
    
    try:
        # Load data
        logger.info(f"Loading data from {DATA_FILE}")
        df = load_csv(DATA_FILE)
        
        if df is None or df.empty:
            logger.error("Failed to load data or data is empty")
            return
        
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(models_dir=MODELS_DIR)
        
        # Run comprehensive evaluation
        results_summary = evaluator.run_comprehensive_evaluation(df, TARGET_COLS)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        for target_col, summary in results_summary.items():
            print(f"\n{target_col.upper()}:")
            if 'error' in summary:
                print(f"  Error: {summary['error']}")
            else:
                print(f"  Best Model: {summary['best_model']}")
                print(f"  Best RMSE: {summary['best_rmse']:.4f}" if summary['best_rmse'] != 'N/A' else "  Best RMSE: N/A")
                print(f"  Models Evaluated: {summary['models_evaluated']}")
        
        print(f"\nDetailed results saved in: {evaluator.results_dir}")
        print(f"Visualizations saved in: {evaluator.figures_dir}")
        
        logger.info("Model evaluation process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main evaluation process: {e}")
        raise


if __name__ == "__main__":
    main()