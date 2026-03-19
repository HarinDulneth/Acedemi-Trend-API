import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model performance data
json_path = 'results/model_performance.json'
df = pd.read_json(json_path)

# Optional: Clean up model names for consistency
model_name_map = {
    'RandomForest': 'Random Forest',
    'GradientBoosting': 'Gradient Boosting',
    'XGBoost': 'XGBoost',
    'SVR': 'SVR',
    'Prophet': 'Prophet',
    'ARIMA': 'ARIMA',
    'SARIMA': 'SARIMA',
}
df['Model'] = df['Model'].map(model_name_map).fillna(df['Model'])

# Set up the plot
plt.figure(figsize=(14, 7))
sns.barplot(
    data=df,
    x='Model',
    y='RMSE',
    hue='Pathway',
    ci=None
)

plt.title('Model Evaluation: RMSE by Model and Pathway')
plt.ylabel('RMSE')
plt.xlabel('Model')
plt.xticks(rotation=30)
plt.legend(title='Pathway', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show() 