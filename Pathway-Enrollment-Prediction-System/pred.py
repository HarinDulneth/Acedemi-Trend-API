import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# Model imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Time series specific imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore 
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class EnrollmentForecaster:
    def __init__(self, data_path=None, data=None):
        """Initialize the forecaster with data"""
        if data is not None:
            self.data = data
        else:
            self.data = pd.read_csv(data_path)
        
        self.models = {}
        self.forecasts = {}
        self.model_performance = {}
        self.scalers = {}
        
    def prepare_data(self):
        """Prepare data for modeling"""
        # Create date column
        self.data['date'] = pd.to_datetime(self.data['year'], format='%Y')
        
        # Get unique pathways
        self.pathways = self.data['pathway'].unique()
        
        # Create pathway-specific datasets
        self.pathway_data = {}
        for pathway in self.pathways:
            pathway_df = self.data[self.data['pathway'] == pathway].copy()
            pathway_df = pathway_df.sort_values('year')
            pathway_df['year_numeric'] = pathway_df['year'] - pathway_df['year'].min()
            self.pathway_data[pathway] = pathway_df
            
        print(f"Data prepared for {len(self.pathways)} pathways")
        print(f"Years covered: {self.data['year'].min()} - {self.data['year'].max()}")

    def save_models(self, output_dir: str):
        """
        Save each fitted model in self.models to disk under output_dir.
        Models are saved as '<pathway_name>_model.pkl'.
        """
        os.makedirs(output_dir, exist_ok=True)
        for pathway, model in self.models.items():
            file_path = os.path.join(output_dir, f"{pathway}_model.pkl")
            joblib.dump(model, file_path)
            print(f"Saved {pathway!r} model to {file_path}")

            if pathway in self.scalers:
                joblib.dump(self.scalers[pathway], f"{output_dir}/{pathway}_LSTM_scaler.pkl")

    def load_models(self, input_dir: str):
        """
        Load all '*.pkl' files from input_dir into self.models.
        Expects files named '<pathway_name>_model.pkl'.
        """
        self.models = {}
        self.scalers = {}

        for fname in os.listdir(input_dir):
            if fname.endswith("_model.pkl"):
                pathway = fname.rsplit("_model.pkl", 1)[0]
                file_path = os.path.join(input_dir, fname)
                self.models[pathway] = joblib.load(file_path)
                print(f"Loaded {pathway!r} model from {file_path}")

                scaler_path = os.path.join(input_dir, f"{pathway}_LSTM_scaler.pkl")
                if os.path.exists(scaler_path):
                    self.scalers[pathway] = joblib.load(scaler_path)
                    print(f"Loaded LSTM scaler for '{pathway}' from {scaler_path}")
        
    def check_stationarity(self, timeseries):
        """Check stationarity of time series"""
        result = adfuller(timeseries)
        return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
    
    def prepare_time_series(self, pathway_data):
        """Prepare time series data with proper datetime index for yearly data"""
        # Create datetime index from year column
        pathway_data = pathway_data.copy()
        pathway_data['date'] = pd.to_datetime(pathway_data['year'], format='%Y')
        
        # Sort by year to ensure proper time series order
        pathway_data = pathway_data.sort_values('year')
        
        # Create time series with datetime index and explicit frequency
        ts_data = pathway_data.set_index('date')['enrollment_total']
        
        # Set explicit annual frequency
        ts_data.index = pd.DatetimeIndex(ts_data.index, freq='AS')
        
        return ts_data
    
    def create_lstm_sequences(self, data, sequence_length=5):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def train_prophet_model(self, pathway_data, pathway_name):
        """Train Prophet model with hyperparameter tuning"""
        print(f"Training Prophet model for {pathway_name}...")
        
        # Prepare data for Prophet
        prophet_data = pathway_data[['date', 'enrollment_total']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Hyperparameter tuning for Prophet
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }
        
        best_params = None
        best_score = float('inf')
        
        # Simple grid search for Prophet
        for changepoint_prior_scale in param_grid['changepoint_prior_scale']:
            for seasonality_prior_scale in param_grid['seasonality_prior_scale']:
                for holidays_prior_scale in param_grid['holidays_prior_scale']:
                    for seasonality_mode in param_grid['seasonality_mode']:
                        try:
                            model = Prophet(
                                changepoint_prior_scale=changepoint_prior_scale,
                                seasonality_prior_scale=seasonality_prior_scale,
                                holidays_prior_scale=holidays_prior_scale,
                                seasonality_mode=seasonality_mode,
                                yearly_seasonality=True,
                                daily_seasonality=False,
                                weekly_seasonality=False
                            )
                            
                            # Cross-validation
                            train_size = int(len(prophet_data) * 0.8)
                            train_data = prophet_data[:train_size]
                            test_data = prophet_data[train_size:]
                            
                            model.fit(train_data)
                            future = model.make_future_dataframe(periods=len(test_data), freq='Y')
                            forecast = model.predict(future)
                            
                            # Calculate error
                            test_predictions = forecast.iloc[train_size:]['yhat'].values
                            test_actual = test_data['y'].values
                            mse = mean_squared_error(test_actual, test_predictions)
                            
                            if mse < best_score:
                                best_score = mse
                                best_params = {
                                    'changepoint_prior_scale': changepoint_prior_scale,
                                    'seasonality_prior_scale': seasonality_prior_scale,
                                    'holidays_prior_scale': holidays_prior_scale,
                                    'seasonality_mode': seasonality_mode
                                }
                        except:
                            continue
        
        # Train final model with best parameters
        if best_params:
            final_model = Prophet(**best_params, yearly_seasonality=True, 
                                daily_seasonality=False, weekly_seasonality=False)
        else:
            final_model = Prophet(yearly_seasonality=True, daily_seasonality=False, 
                                weekly_seasonality=False)
        
        final_model.fit(prophet_data)
        
        return final_model, best_params
    
    def train_arima_model(self, pathway_data, pathway_name):
        """Train ARIMA model with robust convergence handling"""
        print(f"Training ARIMA model for {pathway_name}...")

        # Prepare time series with proper index
        ts_data = self.prepare_time_series(pathway_data)
        
        # Check stationarity
        if not self.check_stationarity(ts_data.values):
            d = 1
        else:
            d = 0
        
        # Grid search with convergence checking
        best_aic = float('inf')
        best_params = None
        converged_models = []
        
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    model = ARIMA(ts_data, order=(p, d, q))
                    # Use correct method parameter for ARIMA
                    fitted_model = model.fit(method_kwargs={'maxiter': 1000})
                    
                    # Check if model converged
                    if hasattr(fitted_model, 'mle_retvals') and fitted_model.mle_retvals is not None:
                        if fitted_model.mle_retvals.get('converged', False):
                            converged_models.append((fitted_model, (p, d, q)))
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                    else:
                        # If no mle_retvals, assume it worked if we get here
                        converged_models.append((fitted_model, (p, d, q)))
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                            
                except Exception as e:
                    continue
        
        # Train final model with best parameters
        if best_params:
            try:
                final_model = ARIMA(ts_data, order=best_params)
                fitted_model = final_model.fit(method_kwargs={'maxiter': 1000})
                print(f"Best ARIMA params for {pathway_name}: {best_params}")
            except:
                # Fallback to simple model
                final_model = ARIMA(ts_data, order=(1, d, 0))
                fitted_model = final_model.fit()
                best_params = (1, d, 0)
                print(f"Using fallback ARIMA params for {pathway_name}: {best_params}")
        else:
            # No model converged, use simple AR(1)
            final_model = ARIMA(ts_data, order=(1, d, 0))
            fitted_model = final_model.fit()
            best_params = (1, d, 0)
            print(f"No convergence, using simple ARIMA params for {pathway_name}: {best_params}")
        
        return fitted_model, best_params
    
    def train_sarima_model(self, pathway_data, pathway_name):
        """Train SARIMA model with hyperparameter tuning for yearly data"""
        print(f"Training SARIMA model for {pathway_name}...")
        
        # Prepare time series with proper index
        ts_data = self.prepare_time_series(pathway_data)
        
        # Check stationarity
        if not self.check_stationarity(ts_data.values):
            d = 1
        else:
            d = 0
        
        # Reduced grid search for better convergence
        best_aic = float('inf')
        best_params = None
        converged_models = []
        
        # Smaller parameter ranges for better convergence
        for p in range(0, 2):
            for q in range(0, 2):
                for P in range(0, 2):
                    for Q in range(0, 2):
                        try:
                            model = SARIMAX(ts_data, 
                                           order=(p, d, q), 
                                           seasonal_order=(P, 1, Q, 4))
                            # SARIMAX accepts maxiter directly
                            fitted_model = model.fit(disp=False, maxiter=1000)
                            
                            # Check convergence
                            if hasattr(fitted_model, 'mle_retvals') and fitted_model.mle_retvals is not None:
                                if fitted_model.mle_retvals.get('converged', False):
                                    converged_models.append((fitted_model, ((p, d, q), (P, 1, Q, 4))))
                                    if fitted_model.aic < best_aic:
                                        best_aic = fitted_model.aic
                                        best_params = ((p, d, q), (P, 1, Q, 4))
                            else:
                                converged_models.append((fitted_model, ((p, d, q), (P, 1, Q, 4))))
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_params = ((p, d, q), (P, 1, Q, 4))
                                    
                        except Exception as e:
                            continue
        
        # Train final model with best parameters
        if best_params:
            try:
                final_model = SARIMAX(ts_data, 
                                    order=best_params[0], 
                                    seasonal_order=best_params[1])
                fitted_model = final_model.fit(disp=False, maxiter=1000)
                print(f"Best SARIMA params for {pathway_name}: {best_params}")
            except:
                # Fallback to simple SARIMA
                final_model = SARIMAX(ts_data, 
                                    order=(1, d, 0), 
                                    seasonal_order=(0, 1, 0, 4))
                fitted_model = final_model.fit(disp=False)
                best_params = ((1, d, 0), (0, 1, 0, 4))
                print(f"Using fallback SARIMA params for {pathway_name}: {best_params}")
        else:
            # No model converged, use simple model
            final_model = SARIMAX(ts_data, 
                                order=(1, d, 0), 
                                seasonal_order=(0, 1, 0, 4))
            fitted_model = final_model.fit(disp=False)
            best_params = ((1, d, 0), (0, 1, 0, 4))
            print(f"No convergence, using simple SARIMA params for {pathway_name}: {best_params}")
        
        return fitted_model, best_params
    
    def train_lstm_model(self, pathway_data, pathway_name):
        """Train LSTM model with hyperparameter tuning"""
        print(f"Training LSTM model for {pathway_name}...")
        
        # Prepare data
        data = pathway_data['enrollment_total'].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        self.scalers[pathway_name] = scaler
        
        # Create sequences
        sequence_length = min(5, len(scaled_data) - 2)
        X, y = self.create_lstm_sequences(scaled_data.flatten(), sequence_length)
        
        if len(X) < 5:  # Not enough data for LSTM
            return None, None
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Hyperparameter tuning
        best_score = float('inf')
        best_params = None
        best_model = None
        
        param_grid = {
            'units': [32, 64, 128],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.01, 0.1]
        }
        
        for units in param_grid['units']:
            for dropout in param_grid['dropout']:
                for lr in param_grid['learning_rate']:
                    try:
                        model = Sequential([
                            LSTM(units, return_sequences=True, input_shape=(sequence_length, 1)),
                            Dropout(dropout),
                            LSTM(units//2, return_sequences=False),
                            Dropout(dropout),
                            Dense(1)
                        ])
                        
                        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
                        
                        # Early stopping
                        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                        
                        # Train model
                        history = model.fit(X_train, y_train, 
                                          epochs=100, 
                                          batch_size=8, 
                                          validation_data=(X_test, y_test),
                                          callbacks=[early_stopping],
                                          verbose=0)
                        
                        # Evaluate
                        val_loss = min(history.history['val_loss'])
                        
                        if val_loss < best_score:
                            best_score = val_loss
                            best_params = {'units': units, 'dropout': dropout, 'learning_rate': lr}
                            best_model = model
                            
                    except Exception as e:
                        continue
        
        return best_model, best_params
    
    def train_ml_models(self, pathway_data, pathway_name):
        """Train traditional ML models with hyperparameter tuning"""
        print(f"Training ML models for {pathway_name}...")
        
        # Prepare features
        X = pathway_data[['year_numeric']].values
        y = pathway_data['enrollment_total'].values
        
        # Add lag features
        if len(y) > 3:
            X_enhanced = []
            y_enhanced = []
            for i in range(2, len(y)):
                features = [X[i][0], y[i-1], y[i-2]]  # year, lag1, lag2
                X_enhanced.append(features)
                y_enhanced.append(y[i])
            X = np.array(X_enhanced)
            y = np.array(y_enhanced)
        
        if len(X) < 5:
            return {}
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # Random Forest
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), 
                              rf_param_grid, cv=3, scoring='neg_mean_squared_error')
        rf_grid.fit(X_train, y_train)
        models['RandomForest'] = rf_grid.best_estimator_
        
        # Gradient Boosting
        gb_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), 
                              gb_param_grid, cv=3, scoring='neg_mean_squared_error')
        gb_grid.fit(X_train, y_train)
        models['GradientBoosting'] = gb_grid.best_estimator_
        
        # XGBoost
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        xgb_grid = GridSearchCV(xgb.XGBRegressor(random_state=42), 
                               xgb_param_grid, cv=3, scoring='neg_mean_squared_error')
        xgb_grid.fit(X_train, y_train)
        models['XGBoost'] = xgb_grid.best_estimator_
        
        # SVR
        svr_param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        svr_grid = GridSearchCV(SVR(), svr_param_grid, cv=3, scoring='neg_mean_squared_error')
        svr_grid.fit(X_train_scaled, y_train)
        models['SVR'] = svr_grid.best_estimator_
        
        return models
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    
    def train_all_models(self):
        """Train all models for all pathways"""
        self.prepare_data()
        
        for pathway in self.pathways:
            print(f"\n{'='*60}")
            print(f"Training models for pathway: {pathway}")
            print(f"{'='*60}")
            
            pathway_data = self.pathway_data[pathway]
            
            # Initialize storage for this pathway
            self.models[pathway] = {}
            self.model_performance[pathway] = []
            
            # Train Prophet
            try:
                prophet_model, prophet_params = self.train_prophet_model(pathway_data, pathway)
                self.models[pathway]['Prophet'] = prophet_model
                print(f"Prophet best params: {prophet_params}")
            except Exception as e:
                print(f"Prophet training failed: {e}")
            
            # Train ARIMA
            try:
                arima_model, arima_params = self.train_arima_model(pathway_data, pathway)
                self.models[pathway]['ARIMA'] = arima_model
                print(f"ARIMA best params: {arima_params}")
            except Exception as e:
                print(f"ARIMA training failed: {e}")
            
            # Train SARIMA
            try:
                sarima_model, sarima_params = self.train_sarima_model(pathway_data, pathway)
                self.models[pathway]['SARIMA'] = sarima_model
                print(f"SARIMA best params: {sarima_params}")
            except Exception as e:
                print(f"SARIMA training failed: {e}")
            
            # Train LSTM
            try:
                lstm_model, lstm_params = self.train_lstm_model(pathway_data, pathway)
                if lstm_model is not None:
                    self.models[pathway]['LSTM'] = lstm_model
                    print(f"LSTM best params: {lstm_params}")
            except Exception as e:
                print(f"LSTM training failed: {e}")
            
            # Train ML models
            try:
                ml_models = self.train_ml_models(pathway_data, pathway)
                for model_name, model in ml_models.items():
                    self.models[pathway][model_name] = model
                print(f"ML models trained: {list(ml_models.keys())}")
            except Exception as e:
                print(f"ML models training failed: {e}")
    
    def generate_forecasts(self, forecast_years=5):
        """Generate forecasts for all pathways and models"""
        print(f"\nGenerating forecasts for {forecast_years} years...")
        
        for pathway in self.pathways:
            print(f"\nForecasting for pathway: {pathway}")
            
            self.forecasts[pathway] = {}
            pathway_data = self.pathway_data[pathway]
            
            # Generate future dates
            last_year = pathway_data['year'].max()
            future_years = list(range(last_year + 1, last_year + forecast_years + 1))
            
            # Prophet forecasts
            if 'Prophet' in self.models[pathway]:
                try:
                    model = self.models[pathway]['Prophet']
                    future_dates = pd.DataFrame({
                        'ds': pd.date_range(start=f'{last_year + 1}-01-01', 
                                          periods=forecast_years, freq='Y')
                    })
                    forecast = model.predict(future_dates)
                    self.forecasts[pathway]['Prophet'] = forecast['yhat'].values
                except Exception as e:
                    print(f"Prophet forecasting failed: {e}")
            
            # ARIMA forecasts
            if 'ARIMA' in self.models[pathway]:
                try:
                    model = self.models[pathway]['ARIMA']
                    forecast = model.forecast(steps=forecast_years)
                    self.forecasts[pathway]['ARIMA'] = forecast
                except Exception as e:
                    print(f"ARIMA forecasting failed: {e}")
            
            # SARIMA forecasts
            if 'SARIMA' in self.models[pathway]:
                try:
                    model = self.models[pathway]['SARIMA']
                    forecast = model.forecast(steps=forecast_years)
                    self.forecasts[pathway]['SARIMA'] = forecast
                except Exception as e:
                    print(f"SARIMA forecasting failed: {e}")
            
            # LSTM forecasts
            if 'LSTM' in self.models[pathway]:
                try:
                    model = self.models[pathway]['LSTM']
                    scaler = self.scalers[pathway]
                    
                    # Get last sequence
                    data = pathway_data['enrollment_total'].values.reshape(-1, 1)
                    scaled_data = scaler.transform(data)
                    
                    sequence_length = 5
                    last_sequence = scaled_data[-sequence_length:].flatten()
                    
                    forecasts = []
                    current_sequence = last_sequence.copy()
                    
                    for _ in range(forecast_years):
                        # Reshape for prediction
                        input_seq = current_sequence.reshape(1, sequence_length, 1)
                        
                        # Predict next value
                        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
                        
                        # Inverse transform
                        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
                        forecasts.append(pred)
                        
                        # Update sequence
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[-1] = pred_scaled
                    
                    self.forecasts[pathway]['LSTM'] = forecasts
                except Exception as e:
                    print(f"LSTM forecasting failed: {e}")
            
            # ML model forecasts
            ml_models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'SVR']
            for model_name in ml_models:
                if model_name in self.models[pathway]:
                    try:
                        model = self.models[pathway][model_name]
                        
                        # Prepare features for future predictions
                        last_year_numeric = pathway_data['year_numeric'].max()
                        last_values = pathway_data['enrollment_total'].values[-2:]
                        
                        forecasts = []
                        
                        for i, year in enumerate(future_years):
                            year_numeric = last_year_numeric + i + 1
                            
                            if len(last_values) >= 2:
                                features = [year_numeric, last_values[-1], last_values[-2]]
                            else:
                                features = [year_numeric, last_values[-1], last_values[-1]]
                            
                            if model_name == 'SVR':
                                # Need to scale features for SVR
                                scaler = StandardScaler()
                                # Fit scaler on training data (approximation)
                                train_features = pathway_data[['year_numeric']].values
                                train_y = pathway_data['enrollment_total'].values
                                
                                if len(train_y) > 3:
                                    train_X = []
                                    for j in range(2, len(train_y)):
                                        train_X.append([train_features[j][0], train_y[j-1], train_y[j-2]])
                                    train_X = np.array(train_X)
                                    scaler.fit(train_X)
                                    features_scaled = scaler.transform([features])
                                    pred = model.predict(features_scaled)[0]
                                else:
                                    pred = model.predict([[features[0]]])[0]
                            else:
                                pred = model.predict([features])[0]
                            
                            forecasts.append(max(0, pred))  # Ensure non-negative
                            
                            # Update last_values for next prediction
                            last_values = np.append(last_values, pred)[-2:]
                        
                        self.forecasts[pathway][model_name] = forecasts
                    except Exception as e:
                        print(f"{model_name} forecasting failed: {e}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Historical trends for all pathways
        plt.figure(figsize=(15, 10))
        
        for pathway in self.pathways:
            pathway_data = self.pathway_data[pathway]
            plt.plot(pathway_data['year'], pathway_data['enrollment_total'], 
                    marker='o', label=pathway, linewidth=2)
        
        plt.title('Historical Enrollment Trends by Pathway', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Enrollment Total', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. Forecasts for each pathway
        for pathway in self.pathways:
            if pathway not in self.forecasts:
                continue
                
            pathway_data = self.pathway_data[pathway]
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Forecasts for {pathway}', fontsize=16, fontweight='bold')
            
            # Historical data
            historical_years = pathway_data['year'].values
            historical_values = pathway_data['enrollment_total'].values
            
            # Future years
            last_year = historical_years[-1]
            future_years = list(range(last_year + 1, last_year + 6))
            
            # Plot each model type in different subplots
            subplot_idx = 0
            
            # Time series models (Prophet, ARIMA, SARIMA)
            ax = axes[0, 0]
            ax.plot(historical_years, historical_values, 'ko-', label='Historical', linewidth=2)
            
            ts_models = ['Prophet', 'ARIMA', 'SARIMA']
            colors = ['blue', 'red', 'green']
            
            for i, model_name in enumerate(ts_models):
                if model_name in self.forecasts[pathway]:
                    forecasts = self.forecasts[pathway][model_name]
                    ax.plot(future_years, forecasts, 
                           color=colors[i], marker='s', linestyle='--', 
                           label=f'{model_name} Forecast', linewidth=2)
            
            ax.set_title('Time Series Models')
            ax.set_xlabel('Year')
            ax.set_ylabel('Enrollment Total')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # ML models
            ax = axes[0, 1]
            ax.plot(historical_years, historical_values, 'ko-', label='Historical', linewidth=2)
            
            ml_models = ['RandomForest', 'GradientBoosting', 'XGBoost']
            colors = ['purple', 'orange', 'brown']
            
            for i, model_name in enumerate(ml_models):
                if model_name in self.forecasts[pathway]:
                    forecasts = self.forecasts[pathway][model_name]
                    ax.plot(future_years, forecasts, 
                           color=colors[i], marker='s', linestyle='--', 
                           label=f'{model_name} Forecast', linewidth=2)
            
            ax.set_title('Machine Learning Models')
            ax.set_xlabel('Year')
            ax.set_ylabel('Enrollment Total')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # LSTM and SVR
            ax = axes[1, 0]
            ax.plot(historical_years, historical_values, 'ko-', label='Historical', linewidth=2)
            
            other_models = ['LSTM', 'SVR']
            colors = ['cyan', 'magenta']
            
            for i, model_name in enumerate(other_models):
                if model_name in self.forecasts[pathway]:
                    forecasts = self.forecasts[pathway][model_name]
                    ax.plot(future_years, forecasts, 
                           color=colors[i], marker='s', linestyle='--', 
                           label=f'{model_name} Forecast', linewidth=2)
            
            ax.set_title('Deep Learning & SVR Models')
            ax.set_xlabel('Year')
            ax.set_ylabel('Enrollment Total')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # All models comparison
            ax = axes[1, 1]
            ax.plot(historical_years, historical_values, 'ko-', label='Historical', linewidth=3)
            
            model_colors = {
                'Prophet': 'blue', 'ARIMA': 'red', 'SARIMA': 'green',
                'RandomForest': 'purple', 'GradientBoosting': 'orange', 
                'XGBoost': 'brown', 'LSTM': 'cyan', 'SVR': 'magenta'
            }
            
            for model_name, color in model_colors.items():
                if model_name in self.forecasts[pathway]:
                    forecasts = self.forecasts[pathway][model_name]
                    ax.plot(future_years, forecasts, 
                           color=color, marker='s', linestyle='--', 
                           label=f'{model_name}', linewidth=2, alpha=0.7)
            
            ax.set_title('All Models Comparison')
            ax.set_xlabel('Year')
            ax.set_ylabel('Enrollment Total')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # 3. Model Performance Comparison (if validation data exists)
        self.create_performance_comparison()
        
        # 4. Forecast Summary Table
        self.create_forecast_summary()
    
    def create_performance_comparison(self):
        """Create model performance comparison charts"""
        print("\nEvaluating model performance...")
        
        # Collect performance metrics for all models
        all_performance = []
        
        for pathway in self.pathways:
            pathway_data = self.pathway_data[pathway]
            
            if len(pathway_data) < 8:  # Need enough data for train/test split
                continue
            
            # Split data for evaluation
            train_size = int(len(pathway_data) * 0.8)
            train_data = pathway_data[:train_size]
            test_data = pathway_data[train_size:]
            
            y_true = test_data['enrollment_total'].values
            
            # Evaluate each model
            for model_name in self.models.get(pathway, {}):
                try:
                    if model_name == 'Prophet':
                        model = self.models[pathway][model_name]
                        future = pd.DataFrame({
                            'ds': pd.date_range(start=f'{test_data["year"].iloc[0]}-01-01', 
                                              periods=len(test_data), freq='Y')
                        })
                        forecast = model.predict(future)
                        y_pred = forecast['yhat'].values
                        
                    elif model_name in ['ARIMA', 'SARIMA']:
                        # For ARIMA/SARIMA, we need to retrain on training data
                        if model_name == 'ARIMA':
                            temp_model = ARIMA(train_data['enrollment_total'], order=(1, 0, 1))
                            fitted = temp_model.fit()
                            y_pred = fitted.forecast(steps=len(test_data))
                        else:
                            temp_model = SARIMAX(train_data['enrollment_total'], 
                                               order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
                            fitted = temp_model.fit(disp=False)
                            y_pred = fitted.forecast(steps=len(test_data))
                    
                    elif model_name == 'LSTM':
                        # Skip LSTM evaluation due to complexity
                        continue
                    
                    else:  # ML models
                        model = self.models[pathway][model_name]
                        
                        # Prepare test features
                        X_test = []
                        for i in range(len(test_data)):
                            year_numeric = test_data['year_numeric'].iloc[i]
                            if i == 0:
                                lag1 = train_data['enrollment_total'].iloc[-1]
                                lag2 = train_data['enrollment_total'].iloc[-2] if len(train_data) > 1 else lag1
                            else:
                                lag1 = test_data['enrollment_total'].iloc[i-1]
                                lag2 = test_data['enrollment_total'].iloc[i-2] if i > 1 else train_data['enrollment_total'].iloc[-1]
                            
                            X_test.append([year_numeric, lag1, lag2])
                        
                        X_test = np.array(X_test)
                        
                        if model_name == 'SVR':
                            # Scale features for SVR
                            scaler = StandardScaler()
                            # Fit on training data
                            train_X = []
                            train_y = train_data['enrollment_total'].values
                            for j in range(2, len(train_y)):
                                train_X.append([train_data['year_numeric'].iloc[j], train_y[j-1], train_y[j-2]])
                            train_X = np.array(train_X)
                            scaler.fit(train_X)
                            X_test_scaled = scaler.transform(X_test)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    performance = self.evaluate_model(y_true, y_pred, model_name)
                    performance['Pathway'] = pathway
                    all_performance.append(performance)
                    
                except Exception as e:
                    print(f"Error evaluating {model_name} for {pathway}: {e}")
                    continue
        
        # Create performance DataFrame
        if all_performance:
            performance_df = pd.DataFrame(all_performance)
            
            # Performance visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            
            metrics = ['MAE', 'RMSE', 'R2', 'MSE']
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                
                # Create box plot for each metric
                models = performance_df['Model'].unique()
                metric_data = []
                labels = []
                
                for model in models:
                    model_data = performance_df[performance_df['Model'] == model][metric].values
                    if len(model_data) > 0:
                        metric_data.append(model_data)
                        labels.append(model)
                
                if metric_data:
                    bp = ax.boxplot(metric_data, labels=labels, patch_artist=True)
                    
                    # Color the boxes
                    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                
                ax.set_title(f'{metric} Distribution')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Print summary statistics
            print("\nModel Performance Summary:")
            print("=" * 80)
            summary_stats = performance_df.groupby('Model').agg({
                'MAE': ['mean', 'std'],
                'RMSE': ['mean', 'std'],
                'R2': ['mean', 'std']
            }).round(4)
            
            print(summary_stats)
            
            # Best model per pathway
            print("\nBest Model per Pathway (based on RMSE):")
            print("=" * 50)
            best_models = performance_df.loc[performance_df.groupby('Pathway')['RMSE'].idxmin()]
            for _, row in best_models.iterrows():
                print(f"{row['Pathway']}: {row['Model']} (RMSE: {row['RMSE']:.4f})")
    
    def create_forecast_summary(self):
        """Create summary tables and charts for forecasts"""
        print("\nCreating forecast summary...")
        
        # Create summary table
        summary_data = []
        
        for pathway in self.pathways:
            if pathway not in self.forecasts:
                continue
                
            pathway_data = self.pathway_data[pathway]
            current_enrollment = pathway_data['enrollment_total'].iloc[-1]
            
            row = {
                'Pathway': pathway,
                'Current_Enrollment': current_enrollment,
                'Degree_Program': pathway_data['degree_program'].iloc[0]
            }
            
            # Add forecasts from each model
            for model_name, forecasts in self.forecasts[pathway].items():
                if len(forecasts) >= 5:
                    row[f'{model_name}_Year1'] = round(forecasts[0], 1)
                    row[f'{model_name}_Year5'] = round(forecasts[4], 1)
                    row[f'{model_name}_Avg'] = round(np.mean(forecasts), 1)
            
            summary_data.append(row)
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Display summary table
        print("\nForecast Summary Table:")
        print("=" * 120)
        print(summary_df.to_string(index=False))
        
        # Create heatmap of average forecasts
        if len(summary_df) > 0:
            # Extract model columns
            model_columns = [col for col in summary_df.columns if col.endswith('_Avg')]
            
            if model_columns:
                heatmap_data = summary_df[['Pathway'] + model_columns].set_index('Pathway')
                heatmap_data.columns = [col.replace('_Avg', '') for col in heatmap_data.columns]
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                           cbar_kws={'label': 'Average Predicted Enrollment'})
                plt.title('Average Forecast by Model and Pathway', fontsize=16, fontweight='bold')
                plt.xlabel('Model')
                plt.ylabel('Pathway')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()
        
        # Growth rate analysis
        print("\nGrowth Rate Analysis:")
        print("=" * 60)
        
        for pathway in self.pathways:
            if pathway not in self.forecasts:
                continue
                
            pathway_data = self.pathway_data[pathway]
            current_enrollment = pathway_data['enrollment_total'].iloc[-1]
            
            print(f"\n{pathway}:")
            print(f"  Current Enrollment: {current_enrollment}")
            
            for model_name, forecasts in self.forecasts[pathway].items():
                if len(forecasts) >= 5:
                    year5_forecast = forecasts[4]
                    growth_rate = ((year5_forecast / current_enrollment) ** (1/5) - 1) * 100
                    print(f"  {model_name}: 5-year CAGR = {growth_rate:.2f}%")
    
    def save_results(self, filename='enrollment_forecasts.csv'):
        """Save forecast results to CSV"""
        print(f"\nSaving results to {filename}...")
        
        results = []
        
        for pathway in self.pathways:
            if pathway not in self.forecasts:
                continue
                
            pathway_data = self.pathway_data[pathway]
            degree_program = pathway_data['degree_program'].iloc[0]
            
            # Add historical data
            for _, row in pathway_data.iterrows():
                results.append({
                    'pathway': pathway,
                    'degree_program': degree_program,
                    'year': row['year'],
                    'enrollment_total': row['enrollment_total'],
                    'data_type': 'historical'
                })
            
            # Add forecasts
            last_year = pathway_data['year'].max()
            future_years = list(range(last_year + 1, last_year + 6))
            
            for model_name, forecasts in self.forecasts[pathway].items():
                for i, forecast in enumerate(forecasts):
                    if i < len(future_years):
                        results.append({
                            'pathway': pathway,
                            'degree_program': degree_program,
                            'year': future_years[i],
                            'enrollment_total': round(forecast, 1),
                            'data_type': f'forecast_{model_name}'
                        })
        
        # Save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        return results_df

def run_forecasting(use_saved_models=True,
                    models_dir="saved_models/",
                    forecast_years=5,
                    results_path="enrollment_forecasts_complete.csv"):
    """Run the complete forecasting pipeline, optionally loading saved models."""
    
    # 1. Load your data
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_dir, models_dir)
        csv_path = os.path.join(base_dir, 'enrollment_trend.csv')
        results_path = csv_path = os.path.join(base_dir, 'enrollment_forecasts_complete.csv')

        print(f"Reading CSV: {csv_path}")
        data = pd.read_csv(csv_path)
        print("Data loaded successfully!")
        print(f"Data shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Date range: {data['year'].min()} - {data['year'].max()}")
        print(f"Unique pathways: {data['pathway'].nunique()}")
    except FileNotFoundError:
        print("enrollment_trend.csv not found. Please ensure the file is in the same directory.")
        return
    
    # 2. Initialize forecaster and prepare data
    forecaster = EnrollmentForecaster(data=data)
    forecaster.prepare_data()
    
    # 3. Load or train models
    if use_saved_models:
        try:
            forecaster.load_models(models_path)
            print("Loaded saved models. Skipping training.")
        except Exception as e:
            print(f"Could not load saved models from '{models_path}': {e}")
            print("→ Proceeding to train all models from scratch.")
            forecaster.train_all_models()
            forecaster.save_models(models_path)
    else:
        print("Training all models from scratch.")
        forecaster.train_all_models()
        forecaster.save_models(models_path)
    
    # 4. Generate forecasts
    print(f"\nGenerating {forecast_years}-year forecasts...")
    forecaster.generate_forecasts(forecast_years=forecast_years)
    
    # 5. Create visualizations
    print("\nCreating visualizations...")
    forecaster.create_visualizations()
    
    # 6. Save results
    print(f"\nSaving results to '{results_path}'...")
    forecaster.save_results(results_path)
    print("All done!")
    
    return forecaster

if __name__ == "__main__":
    forecaster = run_forecasting()