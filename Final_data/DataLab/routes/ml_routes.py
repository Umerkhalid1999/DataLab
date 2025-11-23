from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
import pandas as pd
import numpy as np
import os
import math
import json

# Import the MLRecommender from the routes/app.py
try:
    from routes.app import MLRecommender
    from routes.config import Config
    from routes.ml_explainer import generate_ml_explanations
except ImportError:
    try:
        from .app import MLRecommender
        from .config import Config
        from .ml_explainer import generate_ml_explanations
    except ImportError:
        # Fallback for direct execution
        from app import MLRecommender
        from config import Config
        from ml_explainer import generate_ml_explanations

# Helper function to sanitize data for JSON serialization
def sanitize_for_json(obj):
    """Convert NaN, infinity, and numpy types to JSON-compatible values"""
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj

# Create ML blueprint for DataLab integration
ml_bp = Blueprint('ml', __name__, url_prefix='/ml')

# DataLab integration functions
datasets = None

def set_datasets_reference(datasets_ref):
    global datasets
    datasets = datasets_ref

def get_user_datasets():
    user_id = session.get('user_id')
    if not user_id or not datasets:
        return []
    return datasets.get(user_id, [])

def get_dataset_by_id(dataset_id):
    user_datasets = get_user_datasets()
    return next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

# API login required (returns JSON for API calls)
def api_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Initialize ML Recommender
config = Config()
recommender = MLRecommender(openai_api_key=config.OPENAI_API_KEY)

@ml_bp.route('/')
@login_required
def ml_home():
    """ML Module home page - show dataset selection"""
    user_datasets = get_user_datasets()
    if not user_datasets:
        # If no datasets, show message or redirect
        return render_template('ml_selection.html', dataset=None)
    # If only one dataset, use it directly
    if len(user_datasets) == 1:
        return redirect(url_for('ml.index', dataset_id=user_datasets[0]['id']))
    # If multiple datasets, let user choose (for now, use first one)
    return redirect(url_for('ml.index', dataset_id=user_datasets[0]['id']))

@ml_bp.route('/<int:dataset_id>')
@login_required
def index(dataset_id):
    dataset = get_dataset_by_id(dataset_id)
    if not dataset:
        return render_template('404.html'), 404
    return render_template('ml_selection.html', dataset=dataset)

@ml_bp.route('/api/analyze/<int:dataset_id>', methods=['POST'])
@api_login_required
def analyze(dataset_id):
    try:
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404

        # Load the dataset file
        if not os.path.exists(dataset['file_path']):
            return jsonify({'error': 'Dataset file not found'}), 404

        df = pd.read_csv(dataset['file_path'])

        # OPTIMIZED: Sample large datasets for faster analysis
        original_rows = len(df)
        if len(df) > 1000:
            print(f"[INFO] Dataset has {len(df)} rows. Sampling 1000 rows for faster ML analysis...")
            df = df.sample(n=1000, random_state=42)

        print(f"[INFO] Starting ML analysis on {len(df)} rows...")

        # Analyze dataset
        analysis = recommender.analyze_dataset(df)
        analysis['sampled'] = original_rows > 1000
        analysis['original_rows'] = original_rows

        # Preprocess data
        print("[INFO] Preprocessing data...")
        X, y, scaler = recommender.preprocess_data(df.copy(), analysis)

        # Evaluate models
        results = recommender.evaluate_models(X, y, analysis['task_type'])
        
        # Calculate suitability scores and get GPT analysis
        print("[INFO] Calculating recommendations...")
        recommendations = []
        for model_name, performance in results.items():
            if 'error' not in performance:
                suitability_score, justification = recommender.calculate_suitability_score(
                    model_name, analysis, performance
                )

                # Get GPT analysis (optional, with fallback)
                try:
                    gpt_analysis = recommender.get_gpt_analysis(model_name, analysis, performance)
                except:
                    gpt_analysis = None
                
                # Check if model has tuning grids in any level
                can_tune = any(
                    model_name in recommender.hyperparameter_grids.get(level, {})
                    for level in ['normal', 'semi_deep', 'deep']
                )
                
                recommendations.append({
                    'model': model_name,
                    'performance': performance,
                    'suitability_score': float(suitability_score),
                    'justification': justification,
                    'gpt_analysis': gpt_analysis,
                    'can_tune': can_tune
                })
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)

        # Add best model details
        best_model = recommendations[0] if recommendations else None

        # Generate comprehensive explanations
        print("[INFO] Generating explainable AI insights...")
        explanations = {}
        try:
            explanations = generate_ml_explanations(
                analysis,
                recommendations,
                best_model['model'] if best_model else None
            )
            print("[SUCCESS] Explanations generated successfully")
        except Exception as exp_err:
            print(f"[WARNING] Error generating explanations: {exp_err}")
            explanations = {}

        # Prepare response
        response_data = {
            'dataset_analysis': analysis,
            'recommendations': recommendations,
            'best_model': {
                'name': best_model['model'],
                'score': best_model['suitability_score'],
                'performance': best_model['performance'],
                'why_best': f"Achieved highest suitability score of {best_model['suitability_score']:.1f}% with {best_model['performance']['mean_score']:.3f} CV score and {best_model['performance']['training_time']:.2f}s training time."
            } if best_model else None,
            'explanations': explanations
        }

        # Sanitize all data to ensure JSON compatibility (handle NaN, inf, etc.)
        sanitized_response = sanitize_for_json(response_data)

        print("[SUCCESS] ML Analysis complete!")
        print(f"[BEST MODEL] {best_model['model']} with {best_model['suitability_score']:.1f}% suitability score")

        return jsonify(sanitized_response)

    except Exception as e:
        print(f"[ERROR] ML Analysis Error: {str(e)}")  # Server-side logging
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@ml_bp.route('/api/tune/<int:dataset_id>', methods=['POST'])
@api_login_required
def tune_hyperparameters(dataset_id):
    try:
        data = request.get_json()
        model_name = data['model_name']
        tuning_level = data.get('tuning_level', 'normal')
        original_performance = data.get('original_performance', {})
        
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        df = pd.read_csv(dataset['file_path'])
        analysis = recommender.analyze_dataset(df)
        X, y, scaler = recommender.preprocess_data(df.copy(), analysis)
        
        # Perform hyperparameter tuning
        tuning_results = recommender.tune_hyperparameters(model_name, X, y, analysis['task_type'], tuning_level)
        
        if 'error' in tuning_results:
            return jsonify({'error': tuning_results['error']}), 400
        
        # Re-evaluate model with optimized parameters
        updated_performance = recommender.evaluate_tuned_model(model_name, X, y, analysis['task_type'], tuning_results['best_params'])
        
        # Compare with original performance
        original_score = original_performance.get('mean_score', 0)
        new_score = updated_performance.get('mean_score', 0)
        improvement = new_score - original_score
        improvement_pct = (improvement / original_score * 100) if original_score > 0 else 0
        
        return jsonify({
            'message': 'Hyperparameter tuning completed',
            'model_name': model_name,
            'tuning_results': tuning_results,
            'updated_performance': updated_performance,
            'original_score': original_score,
            'new_score': new_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'improved': improvement > 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@ml_bp.route('/api/export-notebook/<int:dataset_id>', methods=['POST'])
@api_login_required
def export_notebook(dataset_id):
    try:
        data = request.get_json()
        model_name = data['model_name']
        model_params = data.get('model_params', {})
        dataset_info = data.get('dataset_info', {})
        
        # Generate notebook content
        notebook_content = recommender.generate_notebook(model_name, model_params, dataset_info)
        
        return jsonify({
            'notebook_content': notebook_content,
            'filename': f'{model_name.lower().replace(" ", "_")}_model.ipynb'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@ml_bp.route('/jupyterlite')
def jupyterlite():
    return render_template('jupyterlite.html')
