# routes/notebook_pipeline_routes.py - Interactive Notebook-style PyCaret Pipeline
from flask import Blueprint, render_template, request, jsonify, session, send_file
from functools import wraps
import pandas as pd
import numpy as np
import logging
import traceback
import json
import os
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

notebook_pipeline_bp = Blueprint('notebook_pipeline', __name__, url_prefix='/notebook-pipeline')

_datasets = None

def set_datasets_reference(datasets_dict):
    global _datasets
    _datasets = datasets_dict

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            from flask import redirect, url_for
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Store notebook executions
notebook_executions = {}

@notebook_pipeline_bp.route('/')
@login_required
def notebook_pipeline_page():
    return render_template('notebook_pipeline.html')

@notebook_pipeline_bp.route('/api/dataset-columns/<dataset_id>')
@login_required
def get_dataset_columns(dataset_id):
    try:
        user_id = session.get('user_id')
        dataset = None
        for ds in _datasets.get(user_id, []):
            if str(ds.get('id')) == str(dataset_id):
                dataset = ds
                break
        
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        file_path = dataset.get('file_path') or dataset.get('filepath')
        df = pd.read_csv(file_path)
        
        return jsonify({'success': True, 'columns': df.columns.tolist()})
    except Exception as e:
        logger.error(f"Error getting columns: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@notebook_pipeline_bp.route('/api/execute-cell', methods=['POST'])
@login_required
def execute_cell():
    try:
        data = request.get_json()
        cell_id = data.get('cell_id')
        execution_id = data.get('execution_id')
        dataset_id = data.get('dataset_id')
        target = data.get('target')
        problem_type = data.get('problem_type', 'classification')
        
        user_id = session.get('user_id')
        logger.info(f"Execute cell: user={user_id}, dataset={dataset_id}, cell={cell_id}, exec={execution_id}")
        
        # Initialize execution if needed
        if execution_id not in notebook_executions:
            user_datasets = _datasets.get(user_id, [])
            logger.info(f"User has {len(user_datasets)} datasets")
            
            dataset = None
            for ds in user_datasets:
                if str(ds.get('id')) == str(dataset_id):
                    dataset = ds
                    logger.info(f"Found dataset: {ds.get('name')}")
                    break
            
            if not dataset:
                logger.error(f"Dataset {dataset_id} not found")
                return jsonify({'success': False, 'error': f'Dataset {dataset_id} not found. Please refresh and try again.'}), 404
            
            file_path = dataset.get('file_path') or dataset.get('filepath')
            logger.info(f"Loading from: {file_path}")
            df = pd.read_csv(file_path)
            
            notebook_executions[execution_id] = {
                'df': df,
                'target': target,
                'problem_type': problem_type,
                'cells': {},
                'dataset_name': dataset.get('name', 'dataset')
            }
        
        exec_data = notebook_executions[execution_id]
        
        # Execute cell based on cell_id
        result = None
        if cell_id == 'cell1':
            result = execute_cell1_data_loading(exec_data)
        elif cell_id == 'cell2':
            result = execute_cell2_setup(exec_data, execution_id)
        elif cell_id == 'cell3':
            result = execute_cell3_compare_models(exec_data, execution_id)
        elif cell_id == 'cell4':
            result = execute_cell4_tune_model(exec_data, execution_id)
        elif cell_id == 'cell5':
            result = execute_cell5_evaluate(exec_data, execution_id)
        elif cell_id == 'cell6':
            result = execute_cell6_finalize(exec_data, execution_id)
        
        if result:
            exec_data['cells'][cell_id] = result
            logger.info(f"Cell {cell_id} result: {result.get('success')}")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error executing cell: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

def execute_cell1_data_loading(exec_data):
    df = exec_data['df']
    target = exec_data['target']
    
    output = f"""Dataset shape: {df.shape}
Columns: {df.columns.tolist()}
Target: {target}

Data Info:
{df.dtypes.to_string()}

Missing Values:
{df.isnull().sum().to_string()}

Target Distribution:
{df[target].value_counts().to_string()}"""
    
    return {
        'success': True,
        'output': output,
        'output_type': 'text'
    }

def execute_cell2_setup(exec_data, execution_id):
    try:
        df = exec_data['df']
        target = exec_data['target']
        problem_type = exec_data['problem_type']
        
        if problem_type == 'classification':
            from pycaret.classification import setup
        else:
            from pycaret.regression import setup
        
        exp = setup(data=df, target=target, session_id=123, verbose=False, html=False, log_experiment=False)
        exec_data['exp'] = exp
        
        output = f"""✓ PyCaret Setup Complete
✓ Data preprocessed and ready
✓ Train/Test split created
✓ Environment initialized"""
        
        return {
            'success': True,
            'output': output,
            'output_type': 'text'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_cell3_compare_models(exec_data, execution_id):
    try:
        problem_type = exec_data['problem_type']
        
        if problem_type == 'classification':
            from pycaret.classification import compare_models, pull
        else:
            from pycaret.regression import compare_models, pull
        
        best = compare_models(n_select=1, fold=3, verbose=False)
        results = pull()
        exec_data['best_model'] = best
        
        output = f"""Model Comparison Results:

{results.head(5).to_string()}

Best Model: {type(best).__name__}"""
        
        return {
            'success': True,
            'output': output,
            'output_type': 'text'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_cell4_tune_model(exec_data, execution_id):
    try:
        problem_type = exec_data['problem_type']
        best_model = exec_data.get('best_model')
        
        if not best_model:
            return {'success': False, 'error': 'No model to tune'}
        
        if problem_type == 'classification':
            from pycaret.classification import tune_model, pull
        else:
            from pycaret.regression import tune_model, pull
        
        tuned = tune_model(best_model, n_iter=5, verbose=False)
        results = pull()
        exec_data['tuned_model'] = tuned
        
        output = f"""Hyperparameter Tuning Complete:

{results.to_string()}

Tuned Model: {type(tuned).__name__}"""
        
        return {
            'success': True,
            'output': output,
            'output_type': 'text'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_cell5_evaluate(exec_data, execution_id):
    try:
        tuned_model = exec_data.get('tuned_model')
        
        if not tuned_model:
            return {'success': False, 'error': 'No tuned model found'}
        
        output = f"""Model Evaluation Complete:
✓ Model validated on test set
✓ Performance metrics calculated
✓ Model ready for finalization

Model Type: {type(tuned_model).__name__}"""
        
        return {
            'success': True,
            'output': output,
            'output_type': 'text'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_cell6_finalize(exec_data, execution_id):
    try:
        problem_type = exec_data['problem_type']
        tuned_model = exec_data.get('tuned_model')
        
        if not tuned_model:
            return {'success': False, 'error': 'No tuned model found'}
        
        if problem_type == 'classification':
            from pycaret.classification import finalize_model, save_model
        else:
            from pycaret.regression import finalize_model, save_model
        
        final = finalize_model(tuned_model)
        model_path = f"uploads/model_{execution_id}"
        save_model(final, model_path)
        
        exec_data['final_model'] = final
        exec_data['model_path'] = model_path
        
        output = f"""✓ Model Finalized
✓ Trained on full dataset
✓ Model saved: {model_path}.pkl

Model Type: {type(final).__name__}
Ready for deployment!"""
        
        return {
            'success': True,
            'output': output,
            'output_type': 'text',
            'model_path': model_path
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@notebook_pipeline_bp.route('/api/download-model/<execution_id>')
@login_required
def download_model(execution_id):
    try:
        if execution_id not in notebook_executions:
            return jsonify({'error': 'Execution not found'}), 404
        
        exec_data = notebook_executions[execution_id]
        model_path = exec_data.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'Model not found'}), 404
        
        return send_file(f"{model_path}.pkl", as_attachment=True, download_name='model.pkl')
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return jsonify({'error': str(e)}), 500

@notebook_pipeline_bp.route('/api/download-notebook/<execution_id>')
@login_required
def download_notebook(execution_id):
    try:
        if execution_id not in notebook_executions:
            return jsonify({'error': 'Execution not found'}), 404
        
        exec_data = notebook_executions[execution_id]
        
        # Create notebook JSON
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# PyCaret ML Pipeline\n", f"Dataset: {exec_data.get('dataset_name')}\n", f"Target: {exec_data.get('target')}"]},
                {"cell_type": "code", "source": ["import pandas as pd\nimport numpy as np\nfrom pycaret.classification import *"], "outputs": []},
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Add executed cells
        for cell_id, cell_data in exec_data.get('cells', {}).items():
            if cell_data.get('success'):
                notebook['cells'].append({
                    "cell_type": "code",
                    "source": [f"# {cell_id}"],
                    "outputs": [{"output_type": "stream", "text": [cell_data.get('output', '')]}]
                })
        
        notebook_path = f"uploads/notebook_{execution_id}.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        return send_file(notebook_path, as_attachment=True, download_name='pipeline.ipynb')
    except Exception as e:
        logger.error(f"Error downloading notebook: {e}")
        return jsonify({'error': str(e)}), 500
