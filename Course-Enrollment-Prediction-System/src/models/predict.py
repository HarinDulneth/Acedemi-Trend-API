import pandas as pd
import numpy as np
import pickle
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from src.utils.helper_functions import load_csv, save_dataframe
from src.models.train_model import ModelTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    """Configuration class for prediction parameters."""
    future_years: List[int]
    target_columns: List[str]
    model_names: List[str]
    models_dir: Path = Path('models/trained_models')
    data_dir: Path = Path('data/processed')
    viz_dir: Path = Path('visualizations/templates')
    pre_dir: Path = Path('visualizations/predictions')
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.future_years:
            raise ValueError("future_years cannot be empty")
        if not self.target_columns:
            raise ValueError("target_columns cannot be empty")
        if not self.model_names:
            raise ValueError("model_names cannot be empty")
        
        # Create visualization directory if it doesn't exist
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.pre_dir.mkdir(parents=True, exist_ok=True)

class ModelLoader:
    """Handles loading of trained models."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._model_cache = {}
    
    def load_model(self, model_name: str) -> Optional[object]:
        """Load a trained model from disk with caching."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        model_path = self.models_dir / f'{model_name}.pkl'
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self._model_cache[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

class DataPreparator:
    """Handles data preparation for predictions."""
    
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
    
    def validate_dataframe(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """Validate dataframe has required columns and data."""
        if df.empty:
            logger.error("Input dataframe is empty")
            return False
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        return True
    
    def prepare_future_data(self, df: pd.DataFrame, future_years: List[int], target_col: str) -> pd.DataFrame:
        """Prepare data for future predictions."""
        required_cols = ['university', 'course_name', 'year']
        
        if not self.validate_dataframe(df, required_cols):
            return pd.DataFrame()
        
        logger.info(f"Processing {len(df)} rows for future prediction")
        logger.info(f"Unique university-course combinations: {df.groupby(['university', 'course_name']).ngroups}")
        
        future_rows = []
        
        for (university, course), group_df in df.groupby(['university', 'course_name']):
            if group_df.empty:
                continue
                
            group_df = group_df.sort_values('year')
            last_year = group_df['year'].max()
            template_row = group_df.iloc[-1].copy()
            
            # Generate future rows for this university-course combination
            for year in future_years:
                if year > last_year:
                    future_row = template_row.copy()
                    future_row['year'] = year
                    future_row[target_col] = np.nan
                    future_rows.append(future_row)
        
        if not future_rows:
            logger.warning("No future data rows created")
            return pd.DataFrame()
        
        future_df = pd.DataFrame(future_rows)
        logger.info(f"Created {len(future_df)} future prediction rows")
        return future_df

class PredictionVisualizer:
    """Handles visualization of prediction results."""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.viz_dir = config.viz_dir
        
    def create_time_series_plot(self, historical_df: pd.DataFrame, predictions_df: pd.DataFrame, 
                               target_col: str, university: str = None, course: str = None) -> None:
        """Create time series plots showing historical data and predictions."""
        
        # Filter data if specific university/course specified
        if university and course:
            hist_filtered = historical_df[
                (historical_df['university'] == university) & 
                (historical_df['course_name'] == course)
            ].copy()
            pred_filtered = predictions_df[
                (predictions_df['university'] == university) & 
                (predictions_df['course_name'] == course)
            ].copy()
            title_suffix = f" - {university} - {course}"
            filename_suffix = f"_{university}_{course}".replace(" ", "_").replace("/", "_")
        else:
            # Aggregate data
            hist_filtered = historical_df.groupby('year')[target_col].sum().reset_index()
            pred_col = f'{target_col}_pred'
            pred_filtered = predictions_df.groupby(['year', 'model'])[pred_col].sum().reset_index()
            title_suffix = " - All Universities"
            filename_suffix = "_all"
        
        # Create interactive plotly figure
        fig = go.Figure()
        
        # Add historical data
        if not hist_filtered.empty:
            fig.add_trace(go.Scatter(
                x=hist_filtered['year'],
                y=hist_filtered[target_col] if university and course else hist_filtered[target_col],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
        
        # Add predictions for each model
        colors = px.colors.qualitative.Set3
        for i, model in enumerate(pred_filtered['model'].unique()):
            model_data = pred_filtered[pred_filtered['model'] == model]
            pred_col = f'{target_col}_pred'
            
            fig.add_trace(go.Scatter(
                x=model_data['year'],
                y=model_data[pred_col],
                mode='lines+markers',
                name=f'{model.title()} Predictions',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{target_col.title()} Predictions Over Time{title_suffix}',
            xaxis_title='Year',
            yaxis_title=target_col.title(),
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save plot
        filename = f"timeseries_{target_col}{filename_suffix}.html"
        fig.write_html(self.viz_dir / filename)
        logger.info(f"Saved time series plot: {filename}")
    
    def create_model_comparison_plot(self, predictions_df: pd.DataFrame, target_col: str) -> None:
        """Create comparison plots showing predictions from different models."""
        
        pred_col = f'{target_col}_pred'
        
        # Aggregate predictions by model and year
        model_comparison = predictions_df.groupby(['model', 'year'])[pred_col].sum().reset_index()
        
        # Create subplot with multiple comparison views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model Predictions by Year',
                'Model Prediction Distribution',
                'Prediction Variance by Model',
                'Year-over-Year Growth Rate'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Line plot of predictions by year for each model
        colors = px.colors.qualitative.Set3
        for i, model in enumerate(model_comparison['model'].unique()):
            model_data = model_comparison[model_comparison['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['year'],
                    y=model_data[pred_col],
                    mode='lines+markers',
                    name=model.title(),
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot 2: Box plot of prediction distributions
        for i, model in enumerate(model_comparison['model'].unique()):
            model_data = model_comparison[model_comparison['model'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data[pred_col],
                    name=model.title(),
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Variance analysis
        variance_data = model_comparison.groupby('model')[pred_col].agg(['mean', 'std']).reset_index()
        fig.add_trace(
            go.Bar(
                x=variance_data['model'],
                y=variance_data['std'],
                name='Std Deviation',
                marker_color='lightcoral',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Growth rate analysis
        growth_rates = []
        for model in model_comparison['model'].unique():
            model_data = model_comparison[model_comparison['model'] == model].sort_values('year')
            if len(model_data) > 1:
                growth_rate = model_data[pred_col].pct_change().mean() * 100
                growth_rates.append({'model': model, 'growth_rate': growth_rate})
        
        if growth_rates:
            growth_df = pd.DataFrame(growth_rates)
            fig.add_trace(
                go.Bar(
                    x=growth_df['model'],
                    y=growth_df['growth_rate'],
                    name='Avg Growth Rate (%)',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f'Model Comparison Analysis - {target_col.title()}',
            template='plotly_white',
            height=800,
            width=1200
        )
        
        # Save plot
        filename = f"model_comparison_{target_col}.html"
        fig.write_html(self.viz_dir / filename)
        logger.info(f"Saved model comparison plot: {filename}")
    
    def create_university_ranking_plot(self, predictions_df: pd.DataFrame, target_col: str, 
                                     year: int = None) -> None:
        """Create ranking plots for universities based on predictions."""
        
        if year is None:
            year = max(predictions_df['year'])
        
        pred_col = f'{target_col}_pred'
        
        # Filter for specific year and aggregate by university
        year_data = predictions_df[predictions_df['year'] == year]
        university_totals = year_data.groupby('university')[pred_col].sum().reset_index()
        university_totals = university_totals.sort_values(pred_col, ascending=True)
        
        # Create horizontal bar plot
        fig = px.bar(
            university_totals,
            x=pred_col,
            y='university',
            orientation='h',
            title=f'University Rankings by Predicted {target_col.title()} ({year})',
            labels={pred_col: f'Predicted {target_col.title()}', 'university': 'University'},
            color=pred_col,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=max(400, len(university_totals) * 30),
            width=1000,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        # Save plot
        filename = f"university_ranking_{target_col}_{year}.html"
        fig.write_html(self.viz_dir / filename)
        logger.info(f"Saved university ranking plot: {filename}")
    
    def create_course_popularity_plot(self, predictions_df: pd.DataFrame, target_col: str,
                                    top_n: int = 20) -> None:
        """Create plots showing course popularity based on predictions."""
        
        pred_col = f'{target_col}_pred'
        
        # Aggregate predictions by course across all years and models
        course_totals = predictions_df.groupby('course_name')[pred_col].sum().reset_index()
        course_totals = course_totals.sort_values(pred_col, ascending=False).head(top_n)
        
        # Create treemap for course popularity
        fig = px.treemap(
            course_totals,
            values=pred_col,
            names='course_name',
            title=f'Top {top_n} Courses by Predicted {target_col.title()}',
            color=pred_col,
            color_continuous_scale='plasma'
        )
        
        fig.update_layout(
            template='plotly_white',
            width=1200,
            height=800
        )
        
        # Save plot
        filename = f"course_popularity_{target_col}_top{top_n}.html"
        fig.write_html(self.viz_dir / filename)
        logger.info(f"Saved course popularity plot: {filename}")
    
    def create_prediction_summary_dashboard(self, historical_df: pd.DataFrame, 
                                          predictions_df: pd.DataFrame) -> None:
        """Create a comprehensive dashboard with multiple visualizations."""
        
        # Create a multi-tab dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Total Predictions by Year',
                'Model Performance Comparison',
                'University Distribution',
                'Course Category Analysis',
                'Prediction Confidence Intervals',
                'Growth Trend Analysis'
            ),
            specs=[[{"colspan": 2}, None],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Total predictions by year (spanning two columns)
        for target_col in self.config.target_columns:
            pred_col = f'{target_col}_pred'
            yearly_totals = predictions_df.groupby('year')[pred_col].sum().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=yearly_totals['year'],
                    y=yearly_totals[pred_col],
                    mode='lines+markers',
                    name=f'Total {target_col.title()}',
                    line=dict(width=3)
                ),
                row=1, col=1
            )
        
        # Plot 2: Model performance (count of predictions)
        model_counts = predictions_df['model'].value_counts()
        fig.add_trace(
            go.Bar(
                x=model_counts.index,
                y=model_counts.values,
                name='Predictions Count',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 3: University distribution
        uni_counts = predictions_df['university'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=uni_counts.values,
                y=uni_counts.index,
                orientation='h',
                name='Top Universities',
                marker_color='lightcoral',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Plot 4: Course distribution
        course_counts = predictions_df['course_name'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=course_counts.index,
                y=course_counts.values,
                name='Top Courses',
                marker_color='lightblue',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Plot 5: Prediction ranges
        for target_col in self.config.target_columns:
            pred_col = f'{target_col}_pred'
            pred_stats = predictions_df.groupby('year')[pred_col].agg(['mean', 'std']).reset_index()
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=pred_stats['year'],
                    y=pred_stats['mean'] + pred_stats['std'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    name=f'{target_col} Upper'
                ),
                row=3, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pred_stats['year'],
                    y=pred_stats['mean'] - pred_stats['std'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    name=f'{target_col.title()} Range',
                    fillcolor='rgba(0,100,80,0.2)'
                ),
                row=3, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pred_stats['year'],
                    y=pred_stats['mean'],
                    mode='lines+markers',
                    name=f'{target_col.title()} Mean',
                    line=dict(width=2)
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text='University Enrollment Predictions Dashboard',
            template='plotly_white',
            height=1200,
            width=1400,
            showlegend=True
        )
        
        # Save dashboard
        filename = "prediction_dashboard.html"
        fig.write_html(self.viz_dir / filename)
        logger.info(f"Saved prediction dashboard: {filename}")
    
    def create_static_summary_plots(self, predictions_df: pd.DataFrame) -> None:
        """Create static matplotlib plots for quick overview."""
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('University Predictions Summary', fontsize=16, fontweight='bold')
        
        for i, target_col in enumerate(self.config.target_columns):
            pred_col = f'{target_col}_pred'
            
            # Plot 1: Predictions by year
            yearly_data = predictions_df.groupby('year')[pred_col].sum()
            axes[i, 0].plot(yearly_data.index, yearly_data.values, marker='o', linewidth=2)
            axes[i, 0].set_title(f'{target_col.title()} by Year')
            axes[i, 0].set_xlabel('Year')
            axes[i, 0].set_ylabel(target_col.title())
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot 2: Model comparison
            model_data = predictions_df.groupby('model')[pred_col].sum().sort_values(ascending=True)
            axes[i, 1].barh(range(len(model_data)), model_data.values)
            axes[i, 1].set_yticks(range(len(model_data)))
            axes[i, 1].set_yticklabels(model_data.index)
            axes[i, 1].set_title(f'{target_col.title()} by Model')
            axes[i, 1].set_xlabel(f'Total {target_col.title()}')
            
            # Plot 3: Distribution
            axes[i, 2].hist(predictions_df[pred_col], bins=30, alpha=0.7, edgecolor='black')
            axes[i, 2].set_title(f'{target_col.title()} Distribution')
            axes[i, 2].set_xlabel(target_col.title())
            axes[i, 2].set_ylabel('Frequency')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = "prediction_summary_static.png"
        save_path = os.path.join("visualizations", "predictions", filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved static summary plot: {filename}")
    
    def generate_all_visualizations(self, historical_df: pd.DataFrame, 
                                  predictions_df: pd.DataFrame) -> None:
        """Generate all visualization types."""
        logger.info("Generating comprehensive prediction visualizations...")
        
        try:
            # Create dashboard
            self.create_prediction_summary_dashboard(historical_df, predictions_df)
            
            # Create visualizations for each target column
            for target_col in self.config.target_columns:
                self.create_time_series_plot(historical_df, predictions_df, target_col)
                self.create_model_comparison_plot(predictions_df, target_col)
                self.create_university_ranking_plot(predictions_df, target_col)
                self.create_course_popularity_plot(predictions_df, target_col)
            
            # Create static summary
            self.create_static_summary_plots(predictions_df)
            
            # Create specific university/course plots for top combinations
            top_combinations = (predictions_df.groupby(['university', 'course_name'])
                              .size().nlargest(5).index.tolist())
            
            for university, course in top_combinations[:3]:  # Top 3 to avoid too many plots
                for target_col in self.config.target_columns:
                    self.create_time_series_plot(
                        historical_df, predictions_df, target_col, university, course
                    )
            
            logger.info(f"All visualizations saved to: {self.viz_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise
    """Handles different types of model predictions."""
    
    @staticmethod
    def predict_with_sklearn_model(model, X: np.ndarray) -> np.ndarray:
        """Generate predictions using sklearn-compatible models."""
        if X.shape[0] == 0:
            logger.error("Empty input data for sklearn model")
            return np.array([])
        
        try:
            return model.predict(X)
        except Exception as e:
            logger.error(f"Error in sklearn prediction: {e}")
            return np.array([])
    
    @staticmethod
    def predict_with_lstm(model, X: np.ndarray) -> np.ndarray:
        """Generate predictions using LSTM model."""
        if X.shape[0] == 0:
            logger.error("Empty input data for LSTM model")
            return np.array([])
        
        try:
            return model.predict(X).flatten()
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return np.array([])
    
    @staticmethod
    def predict_with_time_series_model(models: Dict, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Generate predictions using time series models."""
        predictions = []
        
        if df.empty:
            logger.warning("Empty dataframe for time series prediction")
            return pd.DataFrame()
        
        for group_key, group_df in df.groupby('group_key'):
            model = models.get(group_key)
            if model is None:
                logger.warning(f"No model found for group {group_key}")
                continue
                
            try:
                if model_name == 'prophet':
                    future = model.make_future_dataframe(periods=3, freq='YS')
                    forecast = model.predict(future)
                    pred_values = forecast['yhat'].tail(3).values
                elif model_name in ['arima', 'sarima']:
                    pred_values = model.forecast(steps=3)
                else:
                    logger.error(f"Unknown time series model: {model_name}")
                    continue
                
                for i, year in enumerate([2024, 2025, 2026]):
                    if i < len(pred_values):
                        predictions.append({
                            'year': year,
                            'group_key': group_key,
                            'yhat': pred_values[i]
                        })
            except Exception as e:
                logger.error(f"Error predicting for group {group_key} with {model_name}: {e}")
                continue
        
        return pd.DataFrame(predictions)

class UniversityEnrollmentPredictor:
    """Main class for university enrollment and application predictions."""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model_loader = ModelLoader(config.models_dir)
        self.trainer = ModelTrainer()
        self.data_preparator = DataPreparator(self.trainer)
        
    def load_and_validate_data(self) -> Optional[pd.DataFrame]:
        """Load and validate the main dataset."""
        dataset_path = self.config.data_dir / 'final_dataset.csv'
        df = load_csv(str(dataset_path))
        
        if df is None:
            logger.error(f"Failed to load dataset from {dataset_path}")
            return None
        
        logger.info(f"Loaded dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {list(df.columns)}")
        
        if df.empty:
            logger.error("Loaded dataset is empty")
            return None
            
        return df
    
    def prepare_model_input(self, future_df: pd.DataFrame, target_col: str, model_name: str) -> Optional[Union[np.ndarray, pd.DataFrame]]:
        """Prepare input data for specific model type."""
        if future_df.empty:
            logger.warning(f"Empty future dataframe for {target_col}")
            return None
        
        # Create temporary dataframe with dummy target values for data preparation
        future_df_temp = future_df.copy()
        future_df_temp[target_col] = 0
        
        try:
            if model_name == 'lstm':
                X, _ = self.trainer.prepare_lstm_data(future_df_temp, target_col=target_col)
            elif model_name in ['prophet', 'arima', 'sarima']:
                X = self.trainer.prepare_time_series_data(future_df_temp, target_col)
            else:
                X, _ = self.trainer.prepare_data(future_df_temp, target_col)
            
            if hasattr(X, 'shape') and X.shape[0] == 0:
                logger.warning(f"No samples in prepared data for {target_col} - {model_name}")
                return None
            elif hasattr(X, 'empty') and X.empty:
                logger.warning(f"Empty prepared data for {target_col} - {model_name}")
                return None
                
            return X
        except Exception as e:
            logger.error(f"Error preparing data for {target_col} - {model_name}: {e}")
            return None
    
    def generate_predictions_for_model(self, model, model_name: str, X_input: Union[np.ndarray, pd.DataFrame], 
                                     future_df: pd.DataFrame, target_col: str) -> Optional[pd.DataFrame]:
        """Generate predictions for a single model."""
        try:
            if model_name == 'lstm':
                preds = PredictionVisualizer.predict_with_lstm(model, X_input)
            elif model_name in ['prophet', 'arima', 'sarima']:
                pred_df = PredictionVisualizer.predict_with_time_series_model(model, X_input, model_name)
                if pred_df.empty:
                    return None
                preds = pred_df['yhat'].values
                # Update future_df with time series prediction info if needed
                if len(pred_df) <= len(future_df):
                    future_df = future_df.iloc[:len(pred_df)].copy()
                    future_df['year'] = pred_df['year'].values
            else:
                preds = PredictionVisualizer.predict_with_sklearn_model(model, X_input)
            
            if len(preds) == 0:
                logger.warning(f"No predictions generated by {model_name} for {target_col}")
                return None
            
            # Ensure we don't exceed future_df length
            n_preds = min(len(preds), len(future_df))
            
            predictions_df = pd.DataFrame({
                'year': future_df['year'].iloc[:n_preds].values,
                'university': future_df['university'].iloc[:n_preds].values,
                'course_name': future_df['course_name'].iloc[:n_preds].values,
                f'{target_col}_pred': preds[:n_preds],
                'model': model_name
            })
            
            logger.info(f"Generated {len(predictions_df)} predictions for {model_name} - {target_col}")
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error generating predictions for {model_name} - {target_col}: {e}")
            return None
    
    def predict_all(self) -> bool:
        """Main function to generate all predictions."""
        # Load and validate data
        df = self.load_and_validate_data()
        if df is None:
            return False
        
        all_predictions = []
        
        for target_col in self.config.target_columns:
            logger.info(f"Processing predictions for {target_col}")
            
            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found in dataset")
                continue
            
            # Prepare future data
            future_df = self.data_preparator.prepare_future_data(df, self.config.future_years, target_col)
            if future_df.empty:
                logger.warning(f"No future data prepared for {target_col}")
                continue
            
            # Generate predictions for each model
            for model_name in self.config.model_names:
                logger.info(f"Processing {model_name} model for {target_col}")
                
                # Load model
                model = self.model_loader.load_model(f'{model_name}_{target_col}')
                if model is None:
                    continue
                
                # Prepare input data
                X_input = self.prepare_model_input(future_df, target_col, model_name)
                if X_input is None:
                    continue
                
                # Generate predictions
                predictions_df = self.generate_predictions_for_model(
                    model, model_name, X_input, future_df, target_col
                )
                
                if predictions_df is not None:
                    all_predictions.append(predictions_df)
        
        # Save results
        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            output_path = self.config.data_dir / 'predictions.csv'
            save_dataframe(final_predictions, str(output_path))
            
            logger.info(f"Saved {len(final_predictions)} predictions to {output_path}")
            logger.info("Predictions summary by model:")
            summary = final_predictions.groupby('model').size().to_dict()
            for model, count in summary.items():
                logger.info(f"  {model}: {count} predictions")
            
            return True
        else:
            logger.warning("No predictions were generated")
            return False

def main():
    """Main entry point for the prediction script."""
    try:
        # Configuration
        config = PredictionConfig(
            future_years=[2024, 2025, 2026, 2027, 2028, 2029, 2030],
            target_columns=['enrollments', 'applications'],
            model_names=['random_forest', 'xgboost', 'lstm', 'prophet', 'arima', 'sarima']
        )
        
        # Create predictor and run predictions
        predictor = UniversityEnrollmentPredictor(config)
        visualizer = PredictionVisualizer(config)
        success = predictor.predict_all()
        
        if success:
            logger.info("Prediction process completed successfully")
            visualizer.generate_all_visualizations(
                predictor.load_and_validate_data(), 
                pd.read_csv(config.data_dir / 'predictions.csv')
            )
        else:
            logger.error("Prediction process failed")
            
    except Exception as e:
        logger.error(f"Fatal error in prediction process: {e}")
        raise

if __name__ == "__main__":
    main()