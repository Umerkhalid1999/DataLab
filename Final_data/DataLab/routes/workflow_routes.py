# routes/workflow_routes.py
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from functools import wraps
import tempfile
import shutil
import zipfile

logger = logging.getLogger(__name__)

# Create workflow blueprint
workflow_bp = Blueprint('workflow', __name__, url_prefix='/workflow')

# Global datasets reference (will be set from main app)
datasets = {}

def set_datasets_reference(datasets_ref):
    """Set reference to the global datasets dictionary"""
    global datasets
    datasets = datasets_ref

def login_required(f):
    """Login required decorator for workflow routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# In-memory workflow storage (in production, use a database)
workflows = {}

@workflow_bp.route('/<int:dataset_id>')
@login_required
def workflow_page(dataset_id):
    """Main workflow management page"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])
    
    # Find dataset by id
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)
    
    if not dataset:
        flash('Dataset not found')
        return redirect(url_for('dashboard'))
    
    return render_template('workflow.html', dataset=dataset)

@workflow_bp.route('/api/<int:dataset_id>/create_pipeline', methods=['POST'])
@login_required
def create_pipeline(dataset_id):
    """Create a new preprocessing pipeline"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])
    
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
    
    try:
        data = request.get_json()
        pipeline_name = data.get('name', f'Pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        pipeline_description = data.get('description', '')
        steps = data.get('steps', [])
        
        # Create pipeline object
        pipeline = {
            'id': len(workflows.get(user_id, {})) + 1,
            'name': pipeline_name,
            'description': pipeline_description,
            'dataset_id': dataset_id,
            'steps': steps,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'status': 'draft',
            'execution_history': []
        }
        
        # Store pipeline
        if user_id not in workflows:
            workflows[user_id] = {}
        
        workflows[user_id][pipeline['id']] = pipeline
        
        return jsonify({
            "success": True,
            "pipeline": pipeline,
            "message": "Pipeline created successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@workflow_bp.route('/api/<int:dataset_id>/pipelines', methods=['GET'])
@login_required
def get_pipelines(dataset_id):
    """Get all pipelines for a dataset"""
    user_id = session['user_id']
    
    user_pipelines = workflows.get(user_id, {})
    dataset_pipelines = [p for p in user_pipelines.values() if p['dataset_id'] == dataset_id]
    
    return jsonify({
        "success": True,
        "pipelines": dataset_pipelines
    })

@workflow_bp.route('/api/pipeline/<int:pipeline_id>/execute', methods=['POST'])
@login_required
def execute_pipeline(pipeline_id):
    """Execute a preprocessing pipeline"""
    user_id = session['user_id']
    
    if user_id not in workflows or pipeline_id not in workflows[user_id]:
        return jsonify({"success": False, "message": "Pipeline not found"}), 404
    
    try:
        pipeline = workflows[user_id][pipeline_id]
        dataset_id = pipeline['dataset_id']
        
        # Get dataset
        user_datasets = datasets.get(user_id, [])
        dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)
        
        if not dataset:
            return jsonify({"success": False, "message": "Dataset not found"}), 404
        
        # Load dataset
        file_path = dataset['file_path']
        file_type = dataset['file_type']
        
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        else:
            return jsonify({"success": False, "message": "Unsupported file type"}), 400
        
        execution_log = []
        
        # Execute each step
        for step in pipeline['steps']:
            step_result = execute_pipeline_step(df, step, execution_log)
            if not step_result['success']:
                return jsonify({
                    "success": False,
                    "message": f"Pipeline execution failed at step: {step['name']}",
                    "error": step_result['error'],
                    "execution_log": execution_log
                }), 500
            
            df = step_result['data']
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"pipeline_output_{pipeline_id}_{timestamp}.csv"
        output_path = os.path.join('uploads', output_filename)
        df.to_csv(output_path, index=False)
        
        # Update execution history
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'output_file': output_path,
            'execution_log': execution_log,
            'input_shape': (len(pd.read_csv(file_path)), len(pd.read_csv(file_path).columns)),
            'output_shape': (len(df), len(df.columns))
        }
        
        pipeline['execution_history'].append(execution_record)
        pipeline['updated_at'] = datetime.now().isoformat()
        
        return jsonify({
            "success": True,
            "message": "Pipeline executed successfully",
            "execution_record": execution_record,
            "output_preview": df.head(10).to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Error executing pipeline: {e}")
        
        # Log failed execution
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e),
            'execution_log': execution_log if 'execution_log' in locals() else []
        }
        
        if user_id in workflows and pipeline_id in workflows[user_id]:
            workflows[user_id][pipeline_id]['execution_history'].append(execution_record)
        
        return jsonify({"success": False, "message": str(e)}), 500

def execute_pipeline_step(df, step, execution_log):
    """Execute a single pipeline step"""
    try:
        step_type = step['type']
        parameters = step.get('parameters', {})
        
        log_entry = {
            'step_name': step['name'],
            'step_type': step_type,
            'timestamp': datetime.now().isoformat(),
            'input_shape': df.shape
        }
        
        if step_type == 'missing_value_handling':
            strategy = parameters.get('strategy', 'mean')
            columns = parameters.get('columns', [])
            
            if not columns:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in columns:
                if col in df.columns:
                    if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == 'mode':
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                    elif strategy == 'drop':
                        df = df.dropna(subset=[col])
        
        elif step_type == 'scaling':
            method = parameters.get('method', 'standard')
            columns = parameters.get('columns', [])
            
            if not columns:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if method == 'standard':
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val > 0:
                            df[col] = (df[col] - mean_val) / std_val
                    elif method == 'minmax':
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val > min_val:
                            df[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif step_type == 'encoding':
            method = parameters.get('method', 'onehot')
            columns = parameters.get('columns', [])
            
            if not columns:
                columns = df.select_dtypes(include=['object']).columns.tolist()
            
            for col in columns:
                if col in df.columns:
                    if method == 'onehot':
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                    elif method == 'label':
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
        
        elif step_type == 'outlier_removal':
            method = parameters.get('method', 'iqr')
            columns = parameters.get('columns', [])
            
            if not columns:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if method == 'iqr':
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif step_type == 'feature_selection':
            method = parameters.get('method', 'correlation')
            target = parameters.get('target')
            n_features = parameters.get('n_features', 10)
            
            if target and target in df.columns:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target in numeric_cols:
                    numeric_cols.remove(target)
                
                if method == 'correlation' and len(numeric_cols) > 0:
                    correlations = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
                    selected_features = correlations.head(min(n_features, len(correlations))).index.tolist()
                    selected_features.append(target)
                    df = df[selected_features]
        
        elif step_type == 'feature_creation':
            operation = parameters.get('operation', 'polynomial')
            columns = parameters.get('columns', [])
            
            if operation == 'polynomial' and len(columns) >= 1:
                degree = parameters.get('degree', 2)
                for col in columns:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        df[f'{col}_squared'] = df[col] ** 2
                        if degree >= 3:
                            df[f'{col}_cubed'] = df[col] ** 3
            
            elif operation == 'interaction' and len(columns) >= 2:
                for i in range(len(columns)):
                    for j in range(i + 1, len(columns)):
                        col1, col2 = columns[i], columns[j]
                        if col1 in df.columns and col2 in df.columns:
                            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        elif step_type == 'data_cleaning':
            # Remove duplicates
            if parameters.get('remove_duplicates', False):
                df = df.drop_duplicates()
            
            # Remove empty rows/columns
            if parameters.get('remove_empty_rows', False):
                df = df.dropna(how='all')
            
            if parameters.get('remove_empty_columns', False):
                df = df.dropna(axis=1, how='all')
        
        log_entry['output_shape'] = df.shape
        log_entry['status'] = 'success'
        execution_log.append(log_entry)
        
        return {'success': True, 'data': df}
        
    except Exception as e:
        log_entry['status'] = 'failed'
        log_entry['error'] = str(e)
        execution_log.append(log_entry)
        
        return {'success': False, 'error': str(e)}

@workflow_bp.route('/api/pipeline/<int:pipeline_id>/export_notebook', methods=['POST'])
@login_required
def export_notebook(pipeline_id):
    """Export pipeline as Jupyter notebook"""
    user_id = session['user_id']
    
    if user_id not in workflows or pipeline_id not in workflows[user_id]:
        return jsonify({"success": False, "message": "Pipeline not found"}), 404
    
    try:
        pipeline = workflows[user_id][pipeline_id]
        dataset_id = pipeline['dataset_id']
        
        # Get dataset info
        user_datasets = datasets.get(user_id, [])
        dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)
        
        if not dataset:
            return jsonify({"success": False, "message": "Dataset not found"}), 404
        
        # Generate notebook content
        notebook_content = generate_jupyter_notebook(pipeline, dataset)
        
        # Save notebook to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        notebook_filename = f"pipeline_{pipeline['name']}_{timestamp}.ipynb"
        temp_path = os.path.join(tempfile.gettempdir(), notebook_filename)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=2, ensure_ascii=False)
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=notebook_filename,
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error exporting notebook: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@workflow_bp.route('/api/pipeline/<int:pipeline_id>/export_documentation', methods=['POST'])
@login_required
def export_documentation(pipeline_id):
    """Export pipeline documentation"""
    user_id = session['user_id']
    
    if user_id not in workflows or pipeline_id not in workflows[user_id]:
        return jsonify({"success": False, "message": "Pipeline not found"}), 404
    
    try:
        pipeline = workflows[user_id][pipeline_id]
        dataset_id = pipeline['dataset_id']
        
        # Get dataset info
        user_datasets = datasets.get(user_id, [])
        dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)
        
        if not dataset:
            return jsonify({"success": False, "message": "Dataset not found"}), 404
        
        # Generate documentation
        documentation = generate_pipeline_documentation(pipeline, dataset)
        
        # Create a zip file with documentation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"pipeline_docs_{pipeline['name']}_{timestamp}.zip"
        temp_zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
        
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add markdown documentation
            zipf.writestr('README.md', documentation['markdown'])
            
            # Add HTML documentation
            zipf.writestr('documentation.html', documentation['html'])
            
            # Add pipeline configuration
            zipf.writestr('pipeline_config.json', json.dumps(pipeline, indent=2))
            
            # Add execution logs if any
            if pipeline['execution_history']:
                zipf.writestr('execution_history.json', 
                            json.dumps(pipeline['execution_history'], indent=2))
        
        return send_file(
            temp_zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Error exporting documentation: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@workflow_bp.route('/api/pipeline/<int:pipeline_id>/version', methods=['POST'])
@login_required
def create_version(pipeline_id):
    """Create a new version of the pipeline"""
    user_id = session['user_id']
    
    if user_id not in workflows or pipeline_id not in workflows[user_id]:
        return jsonify({"success": False, "message": "Pipeline not found"}), 404
    
    try:
        data = request.get_json()
        version_notes = data.get('notes', '')
        
        pipeline = workflows[user_id][pipeline_id]
        
        # Create version history entry
        if 'version_history' not in pipeline:
            pipeline['version_history'] = []
        
        # Get current version info
        current_version = pipeline['version']
        major, minor, patch = map(int, current_version.split('.'))
        
        # Increment version (minor version for now)
        new_version = f"{major}.{minor + 1}.{patch}"
        
        # Save current state to history
        version_entry = {
            'version': current_version,
            'timestamp': pipeline['updated_at'],
            'steps': pipeline['steps'].copy(),
            'notes': version_notes
        }
        
        pipeline['version_history'].append(version_entry)
        pipeline['version'] = new_version
        pipeline['updated_at'] = datetime.now().isoformat()
        
        return jsonify({
            "success": True,
            "new_version": new_version,
            "message": "Version created successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating version: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@workflow_bp.route('/api/pipeline/<int:pipeline_id>/share', methods=['POST'])
@login_required
def share_pipeline(pipeline_id):
    """Share pipeline with other users"""
    user_id = session['user_id']
    
    if user_id not in workflows or pipeline_id not in workflows[user_id]:
        return jsonify({"success": False, "message": "Pipeline not found"}), 404
    
    try:
        data = request.get_json()
        share_type = data.get('type', 'link')  # 'link', 'export', 'clone'
        
        pipeline = workflows[user_id][pipeline_id]
        
        if share_type == 'link':
            # Generate shareable link (in production, implement proper sharing)
            share_link = f"/workflow/shared/{pipeline_id}?token={generate_share_token()}"
            
            return jsonify({
                "success": True,
                "share_link": share_link,
                "message": "Share link generated"
            })
        
        elif share_type == 'export':
            # Export pipeline configuration
            shareable_config = {
                'name': pipeline['name'],
                'description': pipeline['description'],
                'steps': pipeline['steps'],
                'version': pipeline['version'],
                'created_at': pipeline['created_at']
            }
            
            return jsonify({
                "success": True,
                "config": shareable_config,
                "message": "Pipeline configuration exported"
            })
        
    except Exception as e:
        logger.error(f"Error sharing pipeline: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

def generate_share_token():
    """Generate a simple share token (in production, use proper token generation)"""
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def generate_jupyter_notebook(pipeline, dataset):
    """Generate Jupyter notebook from pipeline"""
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Data Preprocessing Pipeline: {pipeline['name']}\n",
            f"\n",
            f"**Description:** {pipeline['description']}\n",
            f"**Dataset:** {dataset['name']}\n",
            f"**Created:** {pipeline['created_at']}\n",
            f"**Version:** {pipeline['version']}\n"
        ]
    })
    
    # Import libraries
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import required libraries\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
            "from sklearn.feature_selection import SelectKBest, f_classif\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Set display options\n",
            "pd.set_option('display.max_columns', None)\n",
            "pd.set_option('display.max_rows', 100)\n"
        ]
    })
    
    # Load data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# Load dataset\n",
            f"df = pd.read_csv('{dataset['name']}')\n",
            f"print(f'Dataset shape: {{df.shape}}')\n",
            f"print(f'Dataset info:')\n",
            f"df.info()\n"
        ]
    })
    
    # Data exploration
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Data Exploration\n"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display first few rows\n",
            "print('First 5 rows:')\n",
            "df.head()\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check for missing values\n",
            "print('Missing values:')\n",
            "df.isnull().sum()\n"
        ]
    })
    
    # Pipeline steps
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Preprocessing Steps\n"]
    })
    
    for i, step in enumerate(pipeline['steps'], 1):
        # Step description
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"### Step {i}: {step['name']}\n\n{step.get('description', '')}\n"]
        })
        
        # Step code
        step_code = generate_step_code(step)
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": step_code
        })
        
        # Check results
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Check results after step {i}\n",
                f"print(f'Shape after step {i}: {{df.shape}}')\n",
                f"df.head()\n"
            ]
        })
    
    # Final results
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Final Results\n"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Final dataset summary\n",
            "print('Final dataset shape:', df.shape)\n",
            "print('\\nFinal dataset info:')\n",
            "df.info()\n",
            "print('\\nFinal dataset description:')\n",
            "df.describe()\n"
        ]
    })
    
    # Save results
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save processed dataset\n",
            f"df.to_csv('processed_{dataset['name']}', index=False)\n",
            "print('Processed dataset saved!')\n"
        ]
    })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def generate_step_code(step):
    """Generate Python code for a pipeline step"""
    step_type = step['type']
    parameters = step.get('parameters', {})
    
    if step_type == 'missing_value_handling':
        strategy = parameters.get('strategy', 'mean')
        columns = parameters.get('columns', [])
        
        code = [f"# {step['name']} - Missing value handling using {strategy} strategy\n"]
        
        if columns:
            code.append(f"columns = {columns}\n")
        else:
            code.append("columns = df.select_dtypes(include=[np.number]).columns.tolist()\n")
        
        code.append("for col in columns:\n")
        code.append("    if col in df.columns:\n")
        
        if strategy == 'mean':
            code.append("        if pd.api.types.is_numeric_dtype(df[col]):\n")
            code.append("            df[col] = df[col].fillna(df[col].mean())\n")
        elif strategy == 'median':
            code.append("        if pd.api.types.is_numeric_dtype(df[col]):\n")
            code.append("            df[col] = df[col].fillna(df[col].median())\n")
        elif strategy == 'mode':
            code.append("        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)\n")
        elif strategy == 'drop':
            code.append("        df = df.dropna(subset=[col])\n")
        
        return code
    
    elif step_type == 'scaling':
        method = parameters.get('method', 'standard')
        columns = parameters.get('columns', [])
        
        code = [f"# {step['name']} - Feature scaling using {method} method\n"]
        
        if columns:
            code.append(f"columns = {columns}\n")
        else:
            code.append("columns = df.select_dtypes(include=[np.number]).columns.tolist()\n")
        
        code.append("for col in columns:\n")
        code.append("    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):\n")
        
        if method == 'standard':
            code.append("        mean_val = df[col].mean()\n")
            code.append("        std_val = df[col].std()\n")
            code.append("        if std_val > 0:\n")
            code.append("            df[col] = (df[col] - mean_val) / std_val\n")
        elif method == 'minmax':
            code.append("        min_val = df[col].min()\n")
            code.append("        max_val = df[col].max()\n")
            code.append("        if max_val > min_val:\n")
            code.append("            df[col] = (df[col] - min_val) / (max_val - min_val)\n")
        
        return code
    
    elif step_type == 'encoding':
        method = parameters.get('method', 'onehot')
        columns = parameters.get('columns', [])
        
        code = [f"# {step['name']} - Categorical encoding using {method} method\n"]
        
        if columns:
            code.append(f"columns = {columns}\n")
        else:
            code.append("columns = df.select_dtypes(include=['object']).columns.tolist()\n")
        
        code.append("for col in columns:\n")
        code.append("    if col in df.columns:\n")
        
        if method == 'onehot':
            code.append("        dummies = pd.get_dummies(df[col], prefix=col)\n")
            code.append("        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)\n")
        elif method == 'label':
            code.append("        le = LabelEncoder()\n")
            code.append("        df[col] = le.fit_transform(df[col].astype(str))\n")
        
        return code
    
    # Add more step types as needed
    else:
        return [f"# {step['name']}\n", f"# Step type: {step_type}\n", f"# Parameters: {parameters}\n"]

def generate_pipeline_documentation(pipeline, dataset):
    """Generate comprehensive pipeline documentation"""
    
    # Markdown documentation
    markdown_content = f"""# Data Preprocessing Pipeline Documentation

## Pipeline Information
- **Name:** {pipeline['name']}
- **Description:** {pipeline['description']}
- **Version:** {pipeline['version']}
- **Created:** {pipeline['created_at']}
- **Last Updated:** {pipeline['updated_at']}

## Dataset Information
- **Dataset Name:** {dataset['name']}
- **File Type:** {dataset['file_type']}
- **Rows:** {dataset['rows']}
- **Columns:** {dataset['columns']}
- **Quality Score:** {dataset['quality_score']}

## Pipeline Steps

"""
    
    for i, step in enumerate(pipeline['steps'], 1):
        markdown_content += f"""### Step {i}: {step['name']}

**Type:** {step['type']}
**Description:** {step.get('description', 'No description provided')}

**Parameters:**
```json
{json.dumps(step.get('parameters', {}), indent=2)}
```

---

"""
    
    # Execution History
    if pipeline['execution_history']:
        markdown_content += "## Execution History\n\n"
        for i, execution in enumerate(pipeline['execution_history'], 1):
            status_emoji = "✅" if execution['status'] == 'success' else "❌"
            markdown_content += f"""### Execution {i} {status_emoji}
- **Timestamp:** {execution['timestamp']}
- **Status:** {execution['status']}
"""
            if execution['status'] == 'success':
                input_shape = execution.get('input_shape', 'Unknown')
                output_shape = execution.get('output_shape', 'Unknown')
                markdown_content += f"- **Input Shape:** {input_shape}\n"
                markdown_content += f"- **Output Shape:** {output_shape}\n"
            else:
                markdown_content += f"- **Error:** {execution.get('error', 'Unknown error')}\n"
            
            markdown_content += "\n"
    
    # HTML documentation
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Documentation - {pipeline['name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .step {{ background: #ffffff; border: 1px solid #dee2e6; padding: 20px; margin-bottom: 20px; border-radius: 8px; }}
        .step-header {{ background: #e9ecef; padding: 10px; margin: -20px -20px 15px -20px; border-radius: 8px 8px 0 0; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Preprocessing Pipeline Documentation</h1>
        <h2>{pipeline['name']}</h2>
        <p><strong>Description:</strong> {pipeline['description']}</p>
        <p><strong>Version:</strong> {pipeline['version']}</p>
        <p><strong>Created:</strong> {pipeline['created_at']}</p>
    </div>

    <h2>Dataset Information</h2>
    <ul>
        <li><strong>Name:</strong> {dataset['name']}</li>
        <li><strong>Type:</strong> {dataset['file_type']}</li>
        <li><strong>Dimensions:</strong> {dataset['rows']} rows × {dataset['columns']} columns</li>
        <li><strong>Quality Score:</strong> {dataset['quality_score']}</li>
    </ul>

    <h2>Pipeline Steps</h2>
"""
    
    for i, step in enumerate(pipeline['steps'], 1):
        html_content += f"""
    <div class="step">
        <div class="step-header">
            <h3>Step {i}: {step['name']}</h3>
        </div>
        <p><strong>Type:</strong> {step['type']}</p>
        <p><strong>Description:</strong> {step.get('description', 'No description provided')}</p>
        <p><strong>Parameters:</strong></p>
        <pre>{json.dumps(step.get('parameters', {}), indent=2)}</pre>
    </div>
"""
    
    if pipeline['execution_history']:
        html_content += "<h2>Execution History</h2>"
        for i, execution in enumerate(pipeline['execution_history'], 1):
            status_class = "success" if execution['status'] == 'success' else "error"
            status_text = "✅ Success" if execution['status'] == 'success' else "❌ Failed"
            
            html_content += f"""
    <div class="step">
        <div class="step-header">
            <h3>Execution {i}</h3>
        </div>
        <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
        <p><strong>Timestamp:</strong> {execution['timestamp']}</p>
"""
            if execution['status'] == 'success':
                input_shape = execution.get('input_shape', 'Unknown')
                output_shape = execution.get('output_shape', 'Unknown')
                html_content += f"        <p><strong>Input Shape:</strong> {input_shape}</p>\n"
                html_content += f"        <p><strong>Output Shape:</strong> {output_shape}</p>\n"
            else:
                html_content += f"        <p><strong>Error:</strong> {execution.get('error', 'Unknown error')}</p>\n"
            
            html_content += "    </div>\n"
    
    html_content += """
</body>
</html>"""
    
    return {
        'markdown': markdown_content,
        'html': html_content
    }