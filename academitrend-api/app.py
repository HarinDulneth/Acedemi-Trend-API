from flask import Flask, jsonify, request
from utils.helpers import get_hello_world

# Import course enrollment prediction handlers
from course.course_handlers import (
    run_course_enrollment_prediction,
    load_course_enrollment_summary,
    load_filtered_course_predictions,
    load_course_historical_data,
    run_course_enrollment_prediction_with_years,
    load_filtered_course_predictions_with_years
)

# Import pathway prediction handlers
from path.pathway_handlers import (
    run_pathway_forecasting,
    load_existing_forecasts,
    load_filtered_pathway_forecasts,
    load_pathway_data,
    check_available_models,
    run_pathway_forecasting_with_years,
    load_filtered_pathway_forecasts_with_years
)

# Import job salary prediction handler
from job_salary_prediction.handler import StudentPredictionAPI
import pickle
import os

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to AcademiTrend API"})

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"data": get_hello_world()})

@app.route('/api/forecast', methods=['GET'])
def forecast():
    """Pathway forecasting - JSON data only, no visualizations"""
    result = run_pathway_forecasting()
    return jsonify(result)

#Course enrollment prediction

@app.route('/api/course-enrollment-prediction', methods=['GET'])
def course_enrollment_prediction():
    """API endpoint for detailed course enrollment prediction"""
    result = run_course_enrollment_prediction()
    return jsonify(result)

@app.route('/api/load-course-predictions', methods=['GET'])
def load_course_predictions():
    """API endpoint to load course enrollment prediction summary"""
    result = load_course_enrollment_summary()
    return jsonify(result)

@app.route('/api/simple-course-enrollment-prediction', methods=['GET'])
def simple_course_enrollment_prediction():
    """API endpoint for filtered course enrollment prediction with query parameters"""
    # Get query parameters
    year = request.args.get('year', type=int)
    university = request.args.get('university')
    course = request.args.get('course')
    model = request.args.get('model')
    
    result = load_filtered_course_predictions(year=year, university=university, course=course, model=model)
    return jsonify(result)

@app.route('/api/course-historical-data', methods=['GET'])
def course_historical_data():
    """API endpoint to load historical enrollments and applications data from raw data files"""
    result = load_course_historical_data()
    return jsonify(result)

@app.route('/api/course-enrollment-prediction-years', methods=['POST'])
def course_enrollment_prediction_years():
    data = request.get_json()
    forecast_years = data.get('forecast_years', 7)
    result = run_course_enrollment_prediction_with_years(forecast_years)
    return jsonify(result)

@app.route('/api/filtered-course-predictions-years', methods=['POST'])
def filtered_course_predictions_years():
    data = request.get_json()
    forecast_years = data.get('forecast_years', 7)
    year = data.get('year')
    university = data.get('university')
    course = data.get('course')
    model = data.get('model')
    result = load_filtered_course_predictions_with_years(
        forecast_years,
        year=year,
        university=university,
        course=course,
        model=model
    )
    return jsonify(result)


#Pathway forecasting


@app.route('/api/path-forecast', methods=['GET'])
def path_forecast():
    """API endpoint for pathway enrollment forecasting using enrollment_trend.csv dataset"""
    result = run_pathway_forecasting()
    return jsonify(result)

@app.route('/api/load-pathway-forecasts', methods=['GET'])
def load_pathway_forecasts():
    """API endpoint to load existing pathway forecasts from CSV file"""
    result = load_existing_forecasts()
    return jsonify(result)

@app.route('/api/filtered-pathway-forecasts', methods=['GET'])
def filtered_pathway_forecasts():
    """API endpoint for filtered pathway forecasts with query parameters"""
    # Get query parameters
    degree_program = request.args.get('degree_program')
    pathway = request.args.get('pathway')
    year = request.args.get('year', type=int)
    model = request.args.get('model')
    
    result = load_filtered_pathway_forecasts(degree_program=degree_program, pathway=pathway, year=year, model=model)
    return jsonify(result)

@app.route('/api/predictions', methods=['GET'])
def all_predictions():
    """API endpoint that returns both pathway and course enrollment predictions from two separate systems"""
    course_result = run_course_enrollment_prediction()
    path_result = run_pathway_forecasting()
    available_models = check_available_models()
    
    return jsonify({
        "pathway_enrollment_prediction": {
            "system": "Pathway Enrollment Prediction System",
            "dataset": "enrollment_trend.csv",
            "models": "Saved models from path/saved_models/",
            "available_models": available_models,
            "data": path_result
        },
        "course_enrollment_prediction": {
            "system": "University Course Enrollment Prediction System", 
            "dataset": "course_enrollment_prediction/data/processed/final_dataset.csv",
            "models": "Trained models in course_enrollment_prediction/models/trained_models/",
            "data": course_result
        },
        "timestamp": "2024-01-01T00:00:00Z"
    })

@app.route('/api/pathway-data', methods=['GET'])
def pathway_data():
    """API endpoint to load pathway enrollment data from enrollment_trend.csv"""
    result = load_pathway_data()
    return jsonify(result)

@app.route('/api/check-models', methods=['GET'])
def check_models():
    """API endpoint to check what models are available in saved files"""
    result = check_available_models()
    return jsonify(result)

@app.route('/api/path-forecast-years', methods=['POST'])
def path_forecast_years():
    data = request.get_json()
    forecast_years = data.get('forecast_years', 5)
    result = run_pathway_forecasting_with_years(forecast_years)
    return jsonify(result)

@app.route('/api/filtered-pathway-forecasts-years', methods=['POST'])
def filtered_pathway_forecasts_years():
    data = request.get_json()
    forecast_years = data.get('forecast_years', 5)
    degree_program = data.get('degree_program')
    pathway = data.get('pathway')
    year = data.get('year')
    model = data.get('model')
    result = load_filtered_pathway_forecasts_with_years(
        forecast_years,
        degree_program=degree_program,
        pathway=pathway,
        year=year,
        model=model
    )
    return jsonify(result)


#Job salary prediction


# Initialize the job salary prediction API
job_salary_api = StudentPredictionAPI()

# Paths for saved model and feature engineer
FEATURE_ENGINEER_PATH = 'job_salary_prediction/saved_feature_engineer.pkl'
TRAINED_MODEL_PATH = 'job_salary_prediction/saved_trained_model.pkl'

# Load the trained model and feature engineer if they exist
if os.path.exists(FEATURE_ENGINEER_PATH) and os.path.exists(TRAINED_MODEL_PATH):
    with open(FEATURE_ENGINEER_PATH, 'rb') as f:
        feature_engineer = pickle.load(f)
    with open(TRAINED_MODEL_PATH, 'rb') as f:
        trained_model = pickle.load(f)
    job_salary_api.load_model(feature_engineer, trained_model)
else:
    feature_engineer = None
    trained_model = None

@app.route('/api/job-salary-prediction', methods=['POST'])
def job_salary_prediction():
    """
    Predict job starting salary and career outcomes for a student.
    Expects JSON input with student data.
    """
    if not job_salary_api.model_loaded:
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'}), 503
    student_data = request.get_json()
    result = job_salary_api.predict(student_data)
    return jsonify(result)

@app.route('/api/job-salary-input-schema', methods=['GET'])
def job_salary_input_schema():
    """
    Get the expected input schema for job salary prediction.
    """
    if not job_salary_api.model_loaded:
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'}), 503
    schema = job_salary_api.get_input_schema()
    return jsonify(schema)

@app.route('/api/filtered-job-salary-predictions', methods=['GET'])
def filtered_job_salary_predictions():
    """
    Filter job salary predictions by query parameters.
    Example: /api/filtered-job-salary-predictions?pathway=Data%20Science&min_gpa=3.0
    """
    if not job_salary_api.model_loaded:
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'}), 503
    filters = {
        'pathway': request.args.get('pathway'),
        'min_gpa': request.args.get('min_gpa', type=float),
        'max_gpa': request.args.get('max_gpa', type=float),
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    from job_salary_prediction.handler import filter_job_salary_predictions
    results = filter_job_salary_predictions(feature_engineer, trained_model, filters)
    return jsonify(results)

@app.route('/api/job-salary-growth-plot', methods=['GET'])
def job_salary_growth_plot():
    """
    Returns a base64-encoded PNG image of average predicted salary growth by semester.
    """
    if not job_salary_api.model_loaded:
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'}), 503
    from job_salary_prediction.data_loader import DataLoader
    from job_salary_prediction.helpers import generate_salary_growth_plot
    data_loader = DataLoader(data_directory='job_salary_prediction')
    img_base64 = generate_salary_growth_plot(feature_engineer, trained_model, data_loader)
    return jsonify({'image_base64': img_base64})


if __name__ == '__main__':
    print(app.url_map) 
    app.run(host='0.0.0.0', port=5050, debug=True)
