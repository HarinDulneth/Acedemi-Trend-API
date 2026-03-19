import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnrollmentPredictor:
    def __init__(self, enrollment_data):
        self.enrollment_data = enrollment_data
        self.years = np.array([2019, 2020, 2021, 2022, 2023])
        self.future_years = np.array([2024, 2025, 2026])
        self.df = self._prepare_dataframe()
        
    def _prepare_dataframe(self):
        """Convert enrollment data to DataFrame format"""
        rows = []
        for course, pathways in self.enrollment_data.items():
            for pathway, enrollments in pathways.items():
                for year_idx, enrollment in enumerate(enrollments):
                    rows.append({
                        'Course': course,
                        'Pathway': pathway,
                        'Year': self.years[year_idx],
                        'Enrollment': enrollment
                    })
        return pd.DataFrame(rows)
    
    def plot_historical_trends(self):
        """Visualize historical enrollment trends"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, course in enumerate(self.enrollment_data.keys()):
            ax = axes[idx]
            course_data = self.enrollment_data[course]
            
            for pathway, enrollments in course_data.items():
                ax.plot(self.years, enrollments, marker='o', linewidth=2, 
                       label=pathway, markersize=6)
            
            ax.set_title(f'{course} - Historical Enrollments', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Enrollment', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(self.years)
        
        plt.tight_layout()
        plt.show()
    
    def linear_regression_model(self):
        """Simple linear regression for each pathway"""
        predictions = {}
        model_metrics = {}
        
        print("=== LINEAR REGRESSION PREDICTIONS ===\n")
        
        for course, pathways in self.enrollment_data.items():
            predictions[course] = {}
            model_metrics[course] = {}
            
            print(f"Course: {course}")
            print("-" * 40)
            
            for pathway, enrollments in pathways.items():
                X = self.years.reshape(-1, 1)
                y = np.array(enrollments)
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Predictions
                future_X = self.future_years.reshape(-1, 1)
                future_pred = model.predict(future_X)
                
                # Ensure non-negative predictions
                future_pred = np.maximum(future_pred, 0)
                
                predictions[course][pathway] = future_pred
                
                # Model metrics
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                model_metrics[course][pathway] = {
                    'mse': mse,
                    'r2': r2,
                    'slope': model.coef_[0],
                    'intercept': model.intercept_
                }
                
                print(f"  {pathway}:")
                print(f"    R² Score: {r2:.3f}")
                print(f"    Trend: {model.coef_[0]:+.2f} students/year")
                print(f"    2024: {future_pred[0]:.0f} students")
                print(f"    2025: {future_pred[1]:.0f} students")
                print(f"    2026: {future_pred[2]:.0f} students")
                print()
        
        return predictions, model_metrics
    
    def polynomial_regression_model(self, degree=2):
        """Polynomial regression for capturing non-linear trends"""
        predictions = {}
        model_metrics = {}
        
        print("=== POLYNOMIAL REGRESSION PREDICTIONS ===\n")
        
        for course, pathways in self.enrollment_data.items():
            predictions[course] = {}
            model_metrics[course] = {}
            
            print(f"Course: {course}")
            print("-" * 40)
            
            for pathway, enrollments in pathways.items():
                X = self.years.reshape(-1, 1)
                y = np.array(enrollments)
                
                # Create polynomial features
                poly_features = PolynomialFeatures(degree=degree)
                X_poly = poly_features.fit_transform(X)
                
                model = LinearRegression()
                model.fit(X_poly, y)
                
                # Predictions
                future_X = self.future_years.reshape(-1, 1)
                future_X_poly = poly_features.transform(future_X)
                future_pred = model.predict(future_X_poly)
                
                # Ensure non-negative predictions
                future_pred = np.maximum(future_pred, 0)
                
                predictions[course][pathway] = future_pred
                
                # Model metrics
                y_pred = model.predict(X_poly)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                model_metrics[course][pathway] = {
                    'mse': mse,
                    'r2': r2
                }
                
                print(f"  {pathway}:")
                print(f"    R² Score: {r2:.3f}")
                print(f"    2024: {future_pred[0]:.0f} students")
                print(f"    2025: {future_pred[1]:.0f} students")
                print(f"    2026: {future_pred[2]:.0f} students")
                print()
        
        return predictions, model_metrics
    
    def moving_average_model(self, window=3):
        """Simple moving average prediction"""
        predictions = {}
        
        print("=== MOVING AVERAGE PREDICTIONS ===\n")
        
        for course, pathways in self.enrollment_data.items():
            predictions[course] = {}
            
            print(f"Course: {course}")
            print("-" * 40)
            
            for pathway, enrollments in pathways.items():
                # Calculate moving average of last 'window' years
                recent_avg = np.mean(enrollments[-window:])
                
                # Predict same value for all future years
                future_pred = np.full(len(self.future_years), recent_avg)
                
                predictions[course][pathway] = future_pred
                
                print(f"  {pathway}:")
                print(f"    Based on {window}-year average: {recent_avg:.1f}")
                print(f"    2024-2026: {recent_avg:.0f} students each year")
                print()
        
        return predictions
    
    def trend_analysis(self):
        """Analyze trends using statistical methods"""
        trend_results = {}
        
        print("=== TREND ANALYSIS ===\n")
        
        for course, pathways in self.enrollment_data.items():
            trend_results[course] = {}
            
            print(f"Course: {course}")
            print("-" * 40)
            
            for pathway, enrollments in pathways.items():
                # Mann-Kendall trend test
                def mann_kendall_test(data):
                    n = len(data)
                    s = 0
                    for i in range(n-1):
                        for j in range(i+1, n):
                            if data[j] > data[i]:
                                s += 1
                            elif data[j] < data[i]:
                                s -= 1
                    
                    # Trend interpretation
                    if s > 0:
                        return "Increasing"
                    elif s < 0:
                        return "Decreasing"
                    else:
                        return "No trend"
                
                trend = mann_kendall_test(enrollments)
                
                # Calculate growth rate
                if enrollments[0] != 0:
                    total_growth = (enrollments[-1] - enrollments[0]) / enrollments[0] * 100
                    annual_growth = total_growth / 4  # 4 years of growth
                else:
                    total_growth = 0
                    annual_growth = 0
                
                # Volatility (coefficient of variation)
                volatility = np.std(enrollments) / np.mean(enrollments) * 100
                
                trend_results[course][pathway] = {
                    'trend': trend,
                    'total_growth': total_growth,
                    'annual_growth': annual_growth,
                    'volatility': volatility
                }
                
                print(f"  {pathway}:")
                print(f"    Trend: {trend}")
                print(f"    Total Growth (5 years): {total_growth:+.1f}%")
                print(f"    Annual Growth Rate: {annual_growth:+.1f}%")
                print(f"    Volatility: {volatility:.1f}%")
                print()
        
        return trend_results
    
    def ensemble_prediction(self):
        """Combine multiple models for better predictions"""
        linear_pred, _ = self.linear_regression_model()
        poly_pred, _ = self.polynomial_regression_model()
        ma_pred = self.moving_average_model()
        
        ensemble_predictions = {}
        
        print("=== ENSEMBLE PREDICTIONS ===\n")
        
        for course in self.enrollment_data.keys():
            ensemble_predictions[course] = {}
            
            print(f"Course: {course}")
            print("-" * 40)
            
            for pathway in self.enrollment_data[course].keys():
                # Weighted average of predictions
                linear_weight = 0.4
                poly_weight = 0.4
                ma_weight = 0.2
                
                ensemble_pred = (linear_weight * linear_pred[course][pathway] +
                               poly_weight * poly_pred[course][pathway] +
                               ma_weight * ma_pred[course][pathway])
                
                ensemble_predictions[course][pathway] = ensemble_pred
                
                print(f"  {pathway}:")
                print(f"    2024: {ensemble_pred[0]:.0f} students")
                print(f"    2025: {ensemble_pred[1]:.0f} students")
                print(f"    2026: {ensemble_pred[2]:.0f} students")
                print()
        
        return ensemble_predictions
    
    def plot_predictions(self, predictions, title="Enrollment Predictions"):
        """Visualize predictions with historical data"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, course in enumerate(self.enrollment_data.keys()):
            ax = axes[idx]
            course_data = self.enrollment_data[course]
            
            for pathway, enrollments in course_data.items():
                # Historical data
                ax.plot(self.years, enrollments, 'o-', linewidth=2, 
                       label=f'{pathway} (Historical)', markersize=6)
                
                # Predictions
                future_pred = predictions[course][pathway]
                ax.plot(self.future_years, future_pred, 's--', linewidth=2, 
                       label=f'{pathway} (Predicted)', markersize=6, alpha=0.7)
                
                # Connect historical and predicted
                ax.plot([self.years[-1], self.future_years[0]], 
                       [enrollments[-1], future_pred[0]], 
                       '--', color='gray', alpha=0.5)
            
            ax.set_title(f'{course} - {title}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Enrollment', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.concatenate([self.years, self.future_years]))
            ax.axvline(x=2023.5, color='red', linestyle=':', alpha=0.5, label='Prediction Start')
        
        plt.tight_layout()
        plt.show()
    
    def summary_report(self):
        """Generate comprehensive summary report"""
        print("="*60)
        print("ENROLLMENT PREDICTION SUMMARY REPORT")
        print("="*60)
        
        # Historical overview
        print("\n1. HISTORICAL OVERVIEW")
        print("-" * 30)
        total_current = 0
        for course, pathways in self.enrollment_data.items():
            course_total = sum(enrollments[-1] for enrollments in pathways.values())
            total_current += course_total
            print(f"{course}: {course_total} students (2023)")
        print(f"Total Current Enrollment: {total_current} students")
        
        # Trend analysis
        trend_results = self.trend_analysis()
        
        # Get ensemble predictions
        ensemble_pred = self.ensemble_prediction()
        
        # Future projections
        print("\n2. FUTURE PROJECTIONS (Ensemble Model)")
        print("-" * 40)
        for year_idx, year in enumerate(self.future_years):
            year_total = 0
            print(f"\n{year} Projections:")
            for course, pathways in ensemble_pred.items():
                course_total = sum(pred[year_idx] for pred in pathways.values())
                year_total += course_total
                print(f"  {course}: {course_total:.0f} students")
            print(f"  Total: {year_total:.0f} students")
        
        # Risk assessment
        print("\n3. RISK ASSESSMENT")
        print("-" * 25)
        high_volatility = []
        declining_programs = []
        
        for course, pathways in trend_results.items():
            for pathway, metrics in pathways.items():
                if metrics['volatility'] > 30:
                    high_volatility.append(f"{course} - {pathway}")
                if metrics['trend'] == 'Decreasing':
                    declining_programs.append(f"{course} - {pathway}")
        
        if high_volatility:
            print("High Volatility Programs (>30%):")
            for program in high_volatility:
                print(f"  • {program}")
        
        if declining_programs:
            print("\nDeclining Programs:")
            for program in declining_programs:
                print(f"  • {program}")
        
        if not high_volatility and not declining_programs:
            print("All programs show stable or positive trends.")

# Initialize and run the predictor
enrollment_data = {
    'ICT': {
        'Computer Network Technology': [25, 23, 29, 27, 27],
        'Games and Animation': [20, 15, 15, 12, 13],
        'Software Systems': [30, 32, 48, 43, 44]
    },
    'ET': {
        'Materials and Process Technology': [33, 31, 24, 26, 24],
        'Industrial Automation and Robotics': [31, 30, 38, 46, 47],
        'Sustainable Technology Pathway': [6, 8, 25, 28, 27]
    },
    'CS': {
        'Cyber Security': [8, 7, 7, 8, 7],
        'Data Science': [15, 24, 14, 14, 18],
        'Artificial Intelligence': [13, 7, 31, 32, 31],
        'Standard Pathway': [14, 9, 13, 11, 12]
    }
}

# Create predictor instance
predictor = EnrollmentPredictor(enrollment_data)

# Run analysis
print("Starting Enrollment Prediction Analysis...")
print("="*50)

# Show historical trends
predictor.plot_historical_trends()

# Run different models
linear_predictions, linear_metrics = predictor.linear_regression_model()
poly_predictions, poly_metrics = predictor.polynomial_regression_model()
ma_predictions = predictor.moving_average_model()

# Ensemble predictions
ensemble_predictions = predictor.ensemble_prediction()

# Plot ensemble predictions
predictor.plot_predictions(ensemble_predictions, "Ensemble Model Predictions")

# Generate summary report
predictor.summary_report()

# Additional analysis: Confidence intervals for linear model
print("\n" + "="*60)
print("CONFIDENCE INTERVALS (Linear Model)")
print("="*60)

for course, pathways in enrollment_data.items():
    print(f"\n{course}:")
    print("-" * 20)
    
    for pathway, enrollments in pathways.items():
        X = np.array([2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)
        y = np.array(enrollments)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate residuals and standard error
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        std_error = np.sqrt(mse)
        
        # Predictions with confidence intervals
        future_X = np.array([2024, 2025, 2026]).reshape(-1, 1)
        future_pred = model.predict(future_X)
        
        # 95% confidence interval (approximate)
        ci_lower = future_pred - 1.96 * std_error
        ci_upper = future_pred + 1.96 * std_error
        
        print(f"  {pathway}:")
        for i, year in enumerate([2024, 2025, 2026]):
            print(f"    {year}: {future_pred[i]:.0f} ± {1.96*std_error:.0f} students " +
                  f"[{max(0, ci_lower[i]):.0f} - {ci_upper[i]:.0f}]")
        print()