import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from prophet import Prophet
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import pickle
import os
import json
from pathlib import Path
import warnings
from src.utils.helper_functions import load_csv, save_dataframe, handle_missing_values, scale_features

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    # Random Forest
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # XGBoost
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    # LSTM
    lstm_params = {
        'units': [50, 100, 150],
        'dropout': [0.1, 0.2, 0.3],
        'batch_size': [16, 32, 64],
        'epochs': 50
    }
    
    # Prophet
    prophet_params = {
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 10.0, 0.1],
        'holidays_prior_scale': [10.0, 1.0, 0.1]
    }

class ModelTrainer:
    """Enhanced model trainer with better organization and error handling."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.metrics = {}
        self.scalers = {}
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories."""
        dirs = ['models/trained_models', 'logs', 'results/metrics']
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'enrollments') -> Tuple[pd.DataFrame, pd.Series]:
        """Enhanced data preparation with better validation and feature selection."""
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
            logger.info(f"Excluded {len(leakage_cols)} potential leakage columns for applications")
        
        # Select features
        available_exclude = [col for col in exclude_cols if col in df.columns]
        X = df.drop(columns=available_exclude)
        X = X.select_dtypes(include=[np.number])
        
        # Handle problematic values
        X = self._clean_numeric_data(X, target_col)
        y = df[target_col].copy()
        
        # Remove rows with missing target values
        valid_mask = ~y.isna()
        X, y = X[valid_mask], y[valid_mask]
        
        logger.info(f"Prepared data for {target_col}: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _clean_numeric_data(self, X: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Clean numeric data by handling infinite and extreme values."""
        X = X.copy()
        
        # Handle infinite values
        inf_mask = np.isinf(X.values)
        if inf_mask.any():
            inf_cols = X.columns[inf_mask.any(axis=0)].tolist()
            logger.warning(f"Replacing infinite values in {len(inf_cols)} columns for {target_col}")
            X = X.replace([np.inf, -np.inf], np.nan)
        
        # Clip extreme values
        max_val = 1e6
        extreme_mask = (X.abs() > max_val).any()
        if extreme_mask.any():
            extreme_cols = X.columns[extreme_mask].tolist()
            logger.warning(f"Clipping extreme values in {len(extreme_cols)} columns for {target_col}")
            X = X.clip(lower=-max_val, upper=max_val)
        
        # Handle missing values
        X = handle_missing_values(X, numeric_method='median')
        
        return X
    
    def prepare_time_series_data(self, df: pd.DataFrame, target_col: str = 'enrollments') -> pd.DataFrame:
        """Prepare data for time series models with proper grouping."""
        group_cols = self._get_group_columns(df, target_col)
        
        # If no valid group columns found, use fallback grouping
        if not group_cols:
            logger.warning(f"No target-encoded columns found for {target_col}, using fallback grouping")
            group_cols = self._get_fallback_group_columns(df)
        
        ts_data = []
        for group_key, group_df in df.groupby(group_cols):
            if len(group_df) < 3:  # Skip groups with insufficient data
                continue
                
            group_ts = group_df[['year', target_col]].copy().sort_values('year')
            group_ts['ds'] = pd.to_datetime(group_ts['year'].astype(str) + '-01-01')
            group_ts['y'] = group_ts[target_col]
            group_ts['group_key'] = str(group_key)
            ts_data.append(group_ts[['ds', 'y', 'group_key']])
        
        if not ts_data:
            # If still no valid groups, create a single group with all data
            logger.warning(f"No valid time series groups found, creating single aggregate group for {target_col}")
            agg_data = df.groupby('year')[target_col].mean().reset_index()
            agg_data['ds'] = pd.to_datetime(agg_data['year'].astype(str) + '-01-01')
            agg_data['y'] = agg_data[target_col]
            agg_data['group_key'] = 'aggregate'
            ts_data = [agg_data[['ds', 'y', 'group_key']]]
            
        result = pd.concat(ts_data, ignore_index=True)
        logger.info(f"Prepared time series data for {target_col}: {len(ts_data)} groups")
        return result
    
    def _get_group_columns(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Get appropriate grouping columns based on target variable."""
        if target_col == 'applications':
            base_cols = ['university_applications_target_encoded', 'course_name_applications_target_encoded']
        else:
            base_cols = ['university_enrollments_target_encoded', 'course_name_enrollments_target_encoded']
        
        # Return only columns that actually exist in the dataframe
        valid_cols = [col for col in base_cols if col in df.columns]
        
        if not valid_cols:
            logger.warning(f"Target-encoded columns not found for {target_col}: {base_cols}")
        
        return valid_cols
    
    def _get_fallback_group_columns(self, df: pd.DataFrame) -> List[str]:
        """Get fallback grouping columns when target-encoded columns are not available."""
        # Look for original categorical columns
        fallback_cols = []
        
        # Check for university and course_name columns
        if 'university' in df.columns:
            fallback_cols.append('university')
        if 'course_name' in df.columns:
            fallback_cols.append('course_name')
            
        # If still empty, look for any categorical-like columns
        if not fallback_cols:
            for col in df.columns:
                if df[col].dtype == 'object' and col not in ['year']:
                    fallback_cols.append(col)
                    break
        
        # If still no columns, use a single group
        if not fallback_cols:
            logger.warning("No suitable grouping columns found, will use aggregate grouping")
            
        return fallback_cols
    
    def prepare_lstm_data(self, df: pd.DataFrame, sequence_length: int = 3, 
                         target_col: str = 'enrollments') -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced LSTM data preparation with better sequence handling."""
        X, y = [], []
        group_cols = self._get_group_columns(df, target_col)
        
        # If no valid group columns found, use fallback grouping
        if not group_cols:
            logger.warning(f"No target-encoded columns found for {target_col}, using fallback grouping for LSTM")
            group_cols = self._get_fallback_group_columns(df)
            
        # If still no group columns, treat entire dataset as one group
        if not group_cols:
            logger.warning(f"No grouping columns available, treating entire dataset as single group for LSTM {target_col}")
            group_iterator = [('single_group', df)]
        else:
            group_iterator = df.groupby(group_cols)
        
        for group_key, group_df in group_iterator:
            if len(group_df) <= sequence_length:
                continue
                
            group_df = group_df.sort_values('year')
            
            # Prepare features (excluding target and metadata)
            exclude_cols = [target_col, 'year', 'university', 'course_name']
            if target_col == 'applications':
                exclude_cols.extend([col for col in df.columns 
                                   if 'application' in col.lower() and col != 'applications'])
            
            features = group_df.drop(columns=[col for col in exclude_cols if col in df.columns])
            features = features.select_dtypes(include=[np.number])
            features = self._clean_numeric_data(features, target_col)
            
            if features.empty:
                continue
            
            # Scale features for this group
            scaled_features, scaler = scale_features(features, features.columns.tolist())
            self.scalers[f"{target_col}_{group_key}"] = scaler
            
            features_array = scaled_features.values
            target_array = group_df[target_col].values
            
            # Create sequences
            for i in range(len(features_array) - sequence_length):
                X.append(features_array[i:i + sequence_length])
                y.append(target_array[i + sequence_length])
        
        if not X:
            raise ValueError(f"No valid sequences created for {target_col}")
        
        # Save scalers
        with open(f'models/trained_models/lstm_scalers_{target_col}.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        X, y = np.array(X), np.array(y)
        logger.info(f"Prepared LSTM data for {target_col}: {X.shape} sequences")
        return X, y
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, target_col: str) -> RandomForestRegressor:
        """Train Random Forest with enhanced hyperparameter tuning."""
        logger.info(f"Training Random Forest for {target_col}")
        
        # Use time series split for temporal data
        cv = TimeSeriesSplit(n_splits=3)
        
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            model, 
            self.config.rf_params, 
            cv=cv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Store metrics
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        metrics = self._calculate_metrics(y, y_pred)
        self.metrics[f'random_forest_{target_col}'] = {
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_,
            **metrics
        }
        
        logger.info(f"Random Forest {target_col} - Best params: {grid_search.best_params_}")
        return best_model
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, target_col: str) -> XGBRegressor:
        """Train XGBoost with enhanced hyperparameter tuning."""
        logger.info(f"Training XGBoost for {target_col}")
        
        cv = TimeSeriesSplit(n_splits=3)
        
        model = XGBRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            model, 
            self.config.xgb_params, 
            cv=cv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Store metrics
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        metrics = self._calculate_metrics(y, y_pred)
        self.metrics[f'xgboost_{target_col}'] = {
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_,
            **metrics
        }
        
        logger.info(f"XGBoost {target_col} - Best params: {grid_search.best_params_}")
        return best_model
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, target_col: str) -> Sequential:
        """Train LSTM with improved architecture and callbacks."""
        logger.info(f"Training LSTM for {target_col}")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_score = float('inf')
        best_model = None
        best_params = None
        
        for units in self.config.lstm_params['units']:
            for dropout in self.config.lstm_params['dropout']:
                for batch_size in self.config.lstm_params['batch_size']:
                    
                    model = Sequential([
                        LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2]), 
                             return_sequences=True),
                        Dropout(dropout),
                        LSTM(units // 2, return_sequences=False),
                        Dropout(dropout),
                        Dense(50, activation='relu'),
                        Dense(1)
                    ])
                    
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    # Enhanced callbacks
                    callbacks = [
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
                    ]
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=self.config.lstm_params['epochs'],
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    val_loss = min(history.history['val_loss'])
                    if val_loss < best_score:
                        best_score = val_loss
                        best_model = model
                        best_params = {'units': units, 'dropout': dropout, 'batch_size': batch_size}
        
        # Store metrics
        y_pred = best_model.predict(X).flatten()
        metrics = self._calculate_metrics(y, y_pred)
        self.metrics[f'lstm_{target_col}'] = {
            'best_params': best_params,
            'val_loss': best_score,
            **metrics
        }
        
        logger.info(f"LSTM {target_col} - Best params: {best_params}")
        return best_model
    
    def train_prophet(self, df: pd.DataFrame, target_col: str = 'enrollments') -> Dict:
        """Train Prophet models with enhanced parameter tuning."""
        logger.info(f"Training Prophet models for {target_col}")
        
        ts_df = self.prepare_time_series_data(df, target_col)
        models = {}
        
        for group_key, group_df in ts_df.groupby('group_key'):
            if len(group_df) < 4:  # Need minimum data for Prophet
                continue
            
            data = group_df[['ds', 'y']].reset_index(drop=True)
            
            # Split for validation
            train_size = max(3, len(data) - 1)
            train_data = data.iloc[:train_size]
            
            best_score = float('inf')
            best_params = None
            best_model = None
            
            # Hyperparameter tuning
            for cps in self.config.prophet_params['changepoint_prior_scale']:
                for sps in self.config.prophet_params['seasonality_prior_scale']:
                    for hps in self.config.prophet_params['holidays_prior_scale']:
                        try:
                            model = Prophet(
                                changepoint_prior_scale=cps,
                                seasonality_prior_scale=sps,
                                holidays_prior_scale=hps,
                                yearly_seasonality=True,
                                daily_seasonality=False,
                                weekly_seasonality=False
                            )
                            model.fit(train_data)
                            
                            # Validate if we have test data
                            if len(data) > train_size:
                                test_data = data.iloc[train_size:]
                                forecast = model.predict(test_data[['ds']])
                                error = mean_absolute_error(test_data['y'], forecast['yhat'])
                                
                                if error < best_score:
                                    best_score = error
                                    best_params = (cps, sps, hps)
                                    best_model = model
                        except Exception as e:
                            logger.warning(f"Prophet training failed for group {group_key}: {e}")
                            continue
            
            # Train final model on all data
            if best_params:
                final_model = Prophet(
                    changepoint_prior_scale=best_params[0],
                    seasonality_prior_scale=best_params[1],
                    holidays_prior_scale=best_params[2],
                    yearly_seasonality=True,
                    daily_seasonality=False,
                    weekly_seasonality=False
                )
                final_model.fit(data)
                models[group_key] = final_model
        
        logger.info(f"Trained {len(models)} Prophet models for {target_col}")
        return models
    
    def train_arima_sarima(self, df: pd.DataFrame, target_col: str = 'enrollments') -> Tuple[Dict, Dict]:
        """Train both ARIMA and SARIMA models with enhanced parameter search."""
        logger.info(f"Training ARIMA/SARIMA models for {target_col}")
        
        ts_df = self.prepare_time_series_data(df, target_col)
        arima_models = {}
        sarima_models = {}
        
        for group_key, group_df in ts_df.groupby('group_key'):
            if len(group_df) < 6:  # Need sufficient data for ARIMA/SARIMA
                continue
            
            series = group_df.sort_values('ds')['y'].values
            
            # ARIMA
            best_arima = self._tune_arima(series, group_key)
            if best_arima:
                arima_models[group_key] = best_arima
            
            # SARIMA
            best_sarima = self._tune_sarima(series, group_key)
            if best_sarima:
                sarima_models[group_key] = best_sarima
        
        logger.info(f"Trained {len(arima_models)} ARIMA and {len(sarima_models)} SARIMA models for {target_col}")
        return arima_models, sarima_models
    
    def _tune_arima(self, series: np.ndarray, group_key: str) -> Optional[Any]:
        """Tune ARIMA parameters for a single series."""
        best_aic = float('inf')
        best_model = None
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                    except Exception:
                        continue
        
        return best_model
    
    def _tune_sarima(self, series: np.ndarray, group_key: str) -> Optional[Any]:
        """Tune SARIMA parameters for a single series."""
        best_aic = float('inf')
        best_model = None
        
        for p in range(2):
            for d in range(2):
                for q in range(2):
                    for sp in range(2):
                        for sd in range(1):
                            for sq in range(2):
                                try:
                                    model = SARIMAX(
                                        series, 
                                        order=(p, d, q), 
                                        seasonal_order=(sp, sd, sq, 4)
                                    )
                                    fitted = model.fit(disp=False)
                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_model = fitted
                                except Exception:
                                    continue
        
        return best_model
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def save_model(self, model: Any, model_name: str) -> None:
        """Enhanced model saving with error handling."""
        try:
            filepath = f'models/trained_models/{model_name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Successfully saved model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
    
    def save_metrics(self) -> None:
        """Save all collected metrics to JSON."""
        try:
            with open('results/metrics/training_metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            logger.info("Successfully saved training metrics")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def train_all_models(self, df: pd.DataFrame) -> None:
        """Enhanced main training function with better error handling and progress tracking."""
        target_columns = ['enrollments', 'applications']
        
        for target_col in target_columns:
            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found in dataset")
                continue
            
            logger.info(f"Starting training pipeline for {target_col}")
            
            try:
                # Prepare data
                X, y = self.prepare_data(df, target_col)
                
                # Train traditional ML models
                rf_model = self.train_random_forest(X, y, target_col)
                self.save_model(rf_model, f'random_forest_{target_col}')
                
                xgb_model = self.train_xgboost(X, y, target_col)
                self.save_model(xgb_model, f'xgboost_{target_col}')
                
                # Train LSTM
                X_lstm, y_lstm = self.prepare_lstm_data(df, target_col=target_col)
                lstm_model = self.train_lstm(X_lstm, y_lstm, target_col)
                self.save_model(lstm_model, f'lstm_{target_col}')
                
                # Train time series models
                prophet_models = self.train_prophet(df, target_col)
                self.save_model(prophet_models, f'prophet_{target_col}')
                
                arima_models, sarima_models = self.train_arima_sarima(df, target_col)
                self.save_model(arima_models, f'arima_{target_col}')
                self.save_model(sarima_models, f'sarima_{target_col}')
                
                logger.info(f"Completed training pipeline for {target_col}")
                
            except Exception as e:
                logger.error(f"Training failed for {target_col}: {e}")
                continue
        
        # Save all metrics
        self.save_metrics()
        logger.info("Training pipeline completed successfully")

def main():
    """Main function to execute the training pipeline."""
    try:
        # Load data
        df = load_csv('data/processed/final_dataset.csv')
        if df is None:
            logger.error("Failed to load final_dataset.csv")
            return
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Initialize trainer and run training
        config = ModelConfig()
        trainer = ModelTrainer(config)
        trainer.train_all_models(df)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()