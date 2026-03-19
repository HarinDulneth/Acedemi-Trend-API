# AcademiTrend API Documentation

## Overview

The AcademiTrend API provides endpoints for course enrollment prediction and path forecasting, returning results as JSON objects. The API is built with Flask and runs on port 5000 by default.

## Base URL

```
http://localhost:5000
```

## Available Endpoints

### 1. Root Endpoint

**GET** `/`

Returns a welcome message.

**Response:**

```json
{
  "message": "Welcome to AcademiTrend API"
}
```

### 2. Hello Endpoint

**GET** `/api/hello`

Returns a simple hello world message.

**Response:**

```json
{
  "data": "Hello World!"
}
```

### 3. Path Forecasting

**GET** `/api/forecast` or **GET** `/api/path-forecast`

Returns path forecasting results using the EnrollmentForecaster model.

**Response:**

```json
{
  "pathways": ["pathway1", "pathway2", ...],
  "forecasts": {
    "pathway1": {
      "2024": 150,
      "2025": 160,
      "2026": 170
    },
    "pathway2": {
      "2024": 200,
      "2025": 210,
      "2026": 220
    }
  },
  "model_performance": {
    "pathway1": {
      "rmse": 5.2,
      "mae": 3.1,
      "r2": 0.95
    }
  }
}
```

### 4. Simple Course Enrollment Prediction

**GET** `/api/simple-course-enrollment-prediction`

Returns course enrollment predictions using the simple EnrollmentPredictor model with multiple algorithms.

**Response:**

```json
{
  "status": "success",
  "message": "Course enrollment predictions generated successfully",
  "historical_data": {
    "ICT": {
      "Computer Network Technology": [25, 23, 29, 27, 27],
      "Games and Animation": [20, 15, 15, 12, 13],
      "Software Systems": [30, 32, 48, 43, 44]
    },
    "ET": {
      "Materials and Process Technology": [33, 31, 24, 26, 24],
      "Industrial Automation and Robotics": [31, 30, 38, 46, 47],
      "Sustainable Technology Pathway": [6, 8, 25, 28, 27]
    },
    "CS": {
      "Cyber Security": [8, 7, 7, 8, 7],
      "Data Science": [15, 24, 14, 14, 18],
      "Artificial Intelligence": [13, 7, 31, 32, 31],
      "Standard Pathway": [14, 9, 13, 11, 12]
    }
  },
  "predictions": {
    "linear_regression": {
      "ICT": {
        "Computer Network Technology": [28, 29, 30],
        "Games and Animation": [11, 10, 9],
        "Software Systems": [45, 46, 47]
      }
    },
    "polynomial_regression": {
      "ICT": {
        "Computer Network Technology": [27, 28, 29],
        "Games and Animation": [12, 11, 10],
        "Software Systems": [44, 45, 46]
      }
    },
    "moving_average": {
      "ICT": {
        "Computer Network Technology": [27, 27, 27],
        "Games and Animation": [13, 13, 13],
        "Software Systems": [44, 44, 44]
      }
    },
    "ensemble": {
      "ICT": {
        "Computer Network Technology": [27, 28, 29],
        "Games and Animation": [12, 11, 10],
        "Software Systems": [44, 45, 46]
      }
    }
  },
  "model_metrics": {
    "linear_regression": {
      "ICT": {
        "Computer Network Technology": {
          "mse": 2.5,
          "r2": 0.85,
          "slope": 1.2,
          "intercept": -2400.0
        }
      }
    }
  },
  "trend_analysis": {
    "ICT": {
      "Computer Network Technology": {
        "trend": "Increasing",
        "total_growth": 8.0,
        "annual_growth": 1.6,
        "volatility": 12.5
      }
    }
  },
  "future_years": [2024, 2025, 2026],
  "historical_years": [2019, 2020, 2021, 2022, 2023]
}
```

### 5. Advanced Course Enrollment Prediction

**GET** `/api/course-enrollment-prediction`

Returns course enrollment predictions using the advanced UniversityEnrollmentPredictor model (requires trained models and data files).

**Response:**

```json
{
  "status": "success",
  "message": "Course enrollment predictions generated successfully",
  "predictions": [
    {
      "year": 2024,
      "university": "University A",
      "course_name": "Computer Science",
      "enrollments_pred": 150.5,
      "applications_pred": 300.2,
      "model": "random_forest"
    }
  ],
  "total_predictions": 100,
  "models_used": ["random_forest", "xgboost", "prophet", "arima"],
  "years_predicted": [2024, 2025, 2026, 2027, 2028]
}
```

### 6. All Predictions Combined

**GET** `/api/predictions`

Returns both course enrollment predictions and path forecasting results in a single response.

**Response:**

```json
{
  "course_enrollment_prediction": {
    // Same structure as simple course enrollment prediction
  },
  "path_forecast": {
    // Same structure as path forecasting
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "status": "error",
  "message": "Description of the error"
}
```

Or for unavailable modules:

```json
{
  "error": "Module not available",
  "status": "unavailable"
}
```

## Usage Examples

### Using curl

```bash
# Test the root endpoint
curl http://localhost:5000/

# Get path forecasting
curl http://localhost:5000/api/path-forecast

# Get simple course enrollment prediction
curl http://localhost:5000/api/simple-course-enrollment-prediction

# Get all predictions
curl http://localhost:5000/api/predictions
```

### Using Python requests

```python
import requests

# Get simple course enrollment prediction
response = requests.get('http://localhost:5000/api/simple-course-enrollment-prediction')
data = response.json()

# Access predictions
ensemble_predictions = data['predictions']['ensemble']
print(f"ICT Software Systems 2024 prediction: {ensemble_predictions['ICT']['Software Systems'][0]}")
```

### Using JavaScript fetch

```javascript
// Get path forecasting
fetch("http://localhost:5000/api/path-forecast")
  .then((response) => response.json())
  .then((data) => {
    console.log("Path forecasts:", data.forecasts);
  });
```

## Running the API

1. Navigate to the academitrend-api directory:

   ```bash
   cd academitrend-api
   ```

2. Run the Flask application:

   ```bash
   python app.py
   ```

3. The API will be available at `http://localhost:5000`

## Testing the API

Use the provided test script to verify all endpoints:

```bash
python test_api.py
```

## Dependencies

The API requires the following Python packages:

- Flask
- pandas
- numpy
- scikit-learn
- xgboost
- prophet
- statsmodels
- tensorflow
- matplotlib
- seaborn
- requests (for testing)

## Notes

- The simple course enrollment prediction uses predefined sample data
- The advanced course enrollment prediction requires trained models and data files
- Path forecasting uses the EnrollmentForecaster model with saved models
- All responses are JSON-serializable
- The API runs in debug mode by default for development
