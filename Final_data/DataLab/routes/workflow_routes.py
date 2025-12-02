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

def get_dataset_by_id(dataset_id):
    """Helper function to get dataset by ID from current user's datasets"""
    user_id = session.get('user_id')
    if not user_id:
        return None

    user_datasets = datasets.get(user_id, [])
    return next((ds for ds in user_datasets if ds['id'] == dataset_id), None)

def load_dataset(dataset):
    """Helper function to load dataset into pandas DataFrame"""
    try:
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        elif file_type == 'json':
            return pd.read_json(file_path)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

# In-memory workflow storage (in production, use a database)
workflows = {}

@workflow_bp.route('/')
@login_required
def workflow_index():
    """Workflow page without dataset ID - use first dataset or create dummy"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])

    # If user has datasets, use the first one
    if user_datasets:
        dataset = user_datasets[0]
    else:
        # Create a dummy dataset for demo purposes
        dataset = {
            'id': 0,
            'name': 'No Dataset (Upload to get started)',
            'rows': 0,
            'columns': 0,
            'quality_score': 0,
            'file_path': None
        }

    return render_template('workflow_new.html', dataset=dataset)

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
    
    # Use the new simplified template
    return render_template('workflow_new.html', dataset=dataset)

@workflow_bp.route('/api/dataset/<int:dataset_id>/info', methods=['GET'])
@login_required
def get_dataset_info(dataset_id):
    """Get dataset column information"""
    user_id = session['user_id']
    user_datasets = datasets.get(user_id, [])
    
    dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
    
    try:
        # Load dataset to get column info
        file_path = dataset['file_path']
        file_type = dataset['file_type']
        
        if file_type == 'csv':
            df = pd.read_csv(file_path, nrows=5)  # Just read first few rows for column info
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file_path, nrows=5)
        else:
            return jsonify({"success": False, "message": "Unsupported file type"}), 400
        
        # Get column information
        columns = []
        for col in df.columns:
            col_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
            columns.append({
                'name': col,
                'type': col_type
            })
        
        return jsonify({
            "success": True,
            "columns": columns
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

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

    elif step_type == 'outlier_removal':
        method = parameters.get('method', 'iqr')
        columns = parameters.get('columns', [])

        code = [f"# {step['name']} - Outlier removal using {method} method\n"]

        if columns:
            code.append(f"columns = {columns}\n")
        else:
            code.append("columns = df.select_dtypes(include=[np.number]).columns.tolist()\n")

        code.append("for col in columns:\n")
        code.append("    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):\n")

        if method == 'iqr':
            code.append("        Q1 = df[col].quantile(0.25)\n")
            code.append("        Q3 = df[col].quantile(0.75)\n")
            code.append("        IQR = Q3 - Q1\n")
            code.append("        lower_bound = Q1 - 1.5 * IQR\n")
            code.append("        upper_bound = Q3 + 1.5 * IQR\n")
            code.append("        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]\n")
        elif method == 'zscore':
            code.append("        mean_val = df[col].mean()\n")
            code.append("        std_val = df[col].std()\n")
            code.append("        if std_val > 0:\n")
            code.append("            z_scores = np.abs((df[col] - mean_val) / std_val)\n")
            code.append("            df = df[z_scores < 3]\n")

        return code

    elif step_type == 'feature_selection':
        method = parameters.get('method', 'correlation')
        target = parameters.get('target')
        n_features = parameters.get('n_features', 10)

        code = [f"# {step['name']} - Feature selection using {method} method\n"]

        if target:
            code.append(f"target = '{target}'\n")
            code.append("if target in df.columns:\n")
            code.append("    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n")
            code.append("    if target in numeric_cols:\n")
            code.append("        numeric_cols.remove(target)\n")
            code.append("    \n")

            if method == 'correlation':
                code.append("    if len(numeric_cols) > 0:\n")
                code.append("        correlations = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)\n")
                code.append(f"        selected_features = correlations.head(min({n_features}, len(correlations))).index.tolist()\n")
                code.append("        selected_features.append(target)\n")
                code.append("        df = df[selected_features]\n")
            elif method == 'variance':
                code.append("    if len(numeric_cols) > 0:\n")
                code.append("        variances = df[numeric_cols].var().sort_values(ascending=False)\n")
                code.append(f"        selected_features = variances.head(min({n_features}, len(variances))).index.tolist()\n")
                code.append("        selected_features.append(target)\n")
                code.append("        df = df[selected_features]\n")

        return code

    elif step_type == 'feature_creation':
        operation = parameters.get('operation', 'polynomial')
        columns = parameters.get('columns', [])
        degree = parameters.get('degree', 2)

        code = [f"# {step['name']} - Feature creation using {operation} operation\n"]

        if operation == 'polynomial':
            code.append(f"columns = {columns if columns else 'df.select_dtypes(include=[np.number]).columns.tolist()'}\n")
            code.append("for col in columns:\n")
            code.append("    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):\n")
            code.append(f"        df[f'{{col}}_squared'] = df[col] ** 2\n")
            if degree >= 3:
                code.append(f"        df[f'{{col}}_cubed'] = df[col] ** 3\n")

        elif operation == 'interaction':
            if columns and len(columns) >= 2:
                code.append(f"columns = {columns}\n")
                code.append("for i in range(len(columns)):\n")
                code.append("    for j in range(i + 1, len(columns)):\n")
                code.append("        col1, col2 = columns[i], columns[j]\n")
                code.append("        if col1 in df.columns and col2 in df.columns:\n")
                code.append("            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):\n")
                code.append("                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]\n")

        elif operation == 'log':
            code.append(f"columns = {columns if columns else 'df.select_dtypes(include=[np.number]).columns.tolist()'}\n")
            code.append("for col in columns:\n")
            code.append("    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):\n")
            code.append("        if (df[col] > 0).all():\n")
            code.append("            df[f'{col}_log'] = np.log(df[col])\n")

        return code

    elif step_type == 'data_cleaning':
        code = [f"# {step['name']} - Data cleaning operations\n"]

        if parameters.get('remove_duplicates', False):
            code.append("# Remove duplicate rows\n")
            code.append("df = df.drop_duplicates()\n")

        if parameters.get('remove_empty_rows', False):
            code.append("# Remove completely empty rows\n")
            code.append("df = df.dropna(how='all')\n")

        if parameters.get('remove_empty_columns', False):
            code.append("# Remove completely empty columns\n")
            code.append("df = df.dropna(axis=1, how='all')\n")

        if parameters.get('strip_whitespace', False):
            code.append("# Strip whitespace from string columns\n")
            code.append("for col in df.select_dtypes(include=['object']).columns:\n")
            code.append("    df[col] = df[col].str.strip()\n")

        if parameters.get('lowercase', False):
            code.append("# Convert string columns to lowercase\n")
            code.append("for col in df.select_dtypes(include=['object']).columns:\n")
            code.append("    df[col] = df[col].str.lower()\n")

        # Default behavior if no specific parameters
        if not any([parameters.get('remove_duplicates'), parameters.get('remove_empty_rows'),
                   parameters.get('remove_empty_columns'), parameters.get('strip_whitespace'),
                   parameters.get('lowercase')]):
            code.append("# Remove duplicate rows (default)\n")
            code.append("df = df.drop_duplicates()\n")

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


# New API endpoints for workflow nodes

@workflow_bp.route('/api/statistics/<int:dataset_id>', methods=['POST'])
def generate_statistics(dataset_id):
    """Generate statistical summary for dataset"""
    try:
        import pandas as pd
        import numpy as np
        from scipy import stats

        # Get dataset info
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404

        # Load dataset
        df = load_dataset(dataset)
        if df is None:
            return jsonify({'success': False, 'error': 'Failed to load dataset'}), 500

        # Generate comprehensive statistics
        stats_result = {
            'success': True,
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            },
            'numeric_stats': {},
            'categorical_stats': {},
            'missing_data': {},
            'data_types': {}
        }

        # Numeric columns statistics (mirrors main statistics module)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if col_data.empty:
                continue

            mean_val = float(col_data.mean())
            q1 = float(col_data.quantile(0.25))
            q3 = float(col_data.quantile(0.75))
            iqr = q3 - q1

            stats_result['numeric_stats'][col] = {
                'count': int(col_data.count()),
                'mean': mean_val,
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'q1': q1,
                'q3': q3,
                'range': float(col_data.max() - col_data.min()),
                'iqr': float(iqr),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'sem': float(col_data.sem()) if col_data.count() > 0 else None,
                'cv': float((col_data.std() / mean_val) * 100) if mean_val else None,
                'sum': float(col_data.sum()),
                'var': float(col_data.var()),
                'mode': float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                'percentiles': {
                    '1%': float(col_data.quantile(0.01)),
                    '5%': float(col_data.quantile(0.05)),
                    '10%': float(col_data.quantile(0.10)),
                    '90%': float(col_data.quantile(0.90)),
                    '95%': float(col_data.quantile(0.95)),
                    '99%': float(col_data.quantile(0.99))
                }
            }

        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            col_data = df[col].dropna()
            value_counts = col_data.value_counts().head(10)
            stats_result['categorical_stats'][col] = {
                'count': int(col_data.count()),
                'unique': int(col_data.nunique()),
                'top_values': {str(k): int(v) for k, v in value_counts.items()},
                'mode': str(col_data.mode()[0]) if len(col_data.mode()) > 0 else 'N/A'
            }

        # Missing data analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            stats_result['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(df) * 100)
            }

        # Data types
        for col in df.columns:
            stats_result['data_types'][col] = str(df[col].dtype)

        return jsonify(stats_result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@workflow_bp.route('/api/visualization/<int:dataset_id>', methods=['POST'])
def generate_visualization(dataset_id):
    """Generate visualizations for dataset (aligns with main visualization module)"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import io
        import base64

        # Get dataset info
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404

        # Load dataset
        df = load_dataset(dataset)
        if df is None:
            return jsonify({'success': False, 'error': 'Failed to load dataset'}), 500

        payload = request.get_json(silent=True) or {}
        viz_type = payload.get('visualizationType', 'all')
        selected_columns = payload.get('columns') or []

        # Check for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return jsonify({'success': False, 'error': 'No numeric columns found'}), 400

        images = []

        def use_columns(df_like, fallback_limit):
            if selected_columns:
                cols = [c for c in selected_columns if c in df_like.columns]
                return df_like[cols] if cols else df_like.iloc[:, :fallback_limit]
            return df_like.iloc[:, :fallback_limit]

        def validate_numeric(frame):
            """Drop non-finite values to avoid plotting errors"""
            clean = frame.replace([np.inf, -np.inf], np.nan).dropna(how='all')
            return clean.dropna()

        # 1. Correlation Heatmap
        if viz_type in ('all', 'correlation'):
            try:
                plt.figure(figsize=(10, 8))
                numeric_subset = validate_numeric(use_columns(numeric_df, 10))
                if numeric_subset.empty or numeric_subset.shape[1] < 2:
                    raise ValueError("Not enough clean numeric columns for correlation.")
                corr = numeric_subset.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
                plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()

                images.append({'title': 'Correlation Heatmap', 'data': image_base64})
            except Exception as e:
                logger.error(f"Error generating correlation heatmap: {e}")

        # 2. Distribution Plots
        if viz_type in ('all', 'distribution'):
            try:
                numeric_cols = validate_numeric(use_columns(numeric_df, 4))
                if not numeric_cols.empty:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    axes = axes.flatten()

                    for i, col in enumerate(numeric_cols.columns):
                        if i < 4:
                            series = numeric_cols[col].replace([np.inf, -np.inf], np.nan).dropna()
                            if series.empty:
                                continue
                            sns.histplot(series, kde=True, ax=axes[i], color='#667eea')
                            axes[i].set_title(f'Distribution of {col}', fontweight='bold')
                            axes[i].set_xlabel(col)
                            axes[i].set_ylabel('Frequency')

                    for i in range(len(numeric_cols), 4):
                        axes[i].set_visible(False)

                    plt.tight_layout()

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()

                    images.append({'title': 'Distribution Plots', 'data': image_base64})
            except Exception as e:
                logger.error(f"Error generating distribution plots: {e}")

        # 3. Box Plots
        if viz_type in ('all', 'boxplot'):
            try:
                numeric_cols = validate_numeric(use_columns(numeric_df, 6))
                if not numeric_cols.empty:
                    plt.figure(figsize=(12, 6))
                    numeric_cols.boxplot()
                    plt.title('Box Plots - Outlier Detection', fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Values')
                    plt.tight_layout()

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()

                    images.append({'title': 'Box Plots (Outlier Detection)', 'data': image_base64})
            except Exception as e:
                logger.error(f"Error generating box plots: {e}")

        # 4. Pairplot / scatter matrix
        if viz_type in ('all', 'scatter'):
            try:
                cols = validate_numeric(use_columns(numeric_df, 4))
                if 2 <= len(cols.columns) <= 6:
                    plt.figure(figsize=(10, 10))
                    pd.plotting.scatter_matrix(cols, figsize=(10, 10), diagonal='kde', alpha=0.7, color='#667eea')
                    plt.suptitle('Scatter Plot Matrix', fontsize=14, fontweight='bold', y=0.995)
                    plt.tight_layout()

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()

                    images.append({'title': 'Scatter Plot Matrix', 'data': image_base64})
            except Exception as e:
                logger.error(f"Error generating scatter matrix: {e}")

        # 5. Categorical count plots
        if viz_type in ('all', 'categorical'):
            try:
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                categorical_cols = [c for c in categorical_cols if c in selected_columns] or list(categorical_cols[:3])
                if len(categorical_cols) > 0:
                    fig, axes = plt.subplots(1, min(len(categorical_cols), 3), figsize=(15, 5))
                    if len(categorical_cols) == 1:
                        axes = [axes]

                    for i, col in enumerate(categorical_cols[:3]):
                        value_counts = df[col].value_counts().head(10)
                        ax = axes[0] if len(categorical_cols) == 1 else axes[i]
                        value_counts.plot(kind='bar', ax=ax, color='#667eea')
                        ax.set_title(f'Top Values - {col}', fontweight='bold')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Count')
                        ax.tick_params(axis='x', rotation=45)

                    plt.tight_layout()

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()

                    images.append({'title': 'Categorical Features Distribution', 'data': image_base64})
            except Exception as e:
                logger.error(f"Error generating count plots: {e}")

        if not images:
            return jsonify({'success': False, 'error': 'Failed to generate any visualizations'}), 500

        return jsonify({
            'success': True,
            'images': images
        })

    except Exception as e:
        logger.error(f"Error in generate_visualization: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@workflow_bp.route('/api/preprocessing/<int:dataset_id>', methods=['POST'])
def run_preprocessing(dataset_id):
    """Run data preprocessing - handle missing values, outliers, duplicates"""
    try:
        import pandas as pd
        import numpy as np
        from scipy import stats

        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404

        df = load_dataset(dataset)
        if df is None:
            return jsonify({'success': False, 'error': 'Failed to load dataset'}), 500

        # Store initial stats
        initial_shape = df.shape
        initial_nulls = df.isnull().sum().sum()
        initial_duplicates = df.duplicated().sum()

        preprocessing_steps = []

        # 1. Handle Missing Values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                if df[col].dtype in ['float64', 'int64']:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    preprocessing_steps.append(f"Filled {col} missing values with median: {median_val:.2f}")
                else:
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    preprocessing_steps.append(f"Filled {col} missing values with mode: {mode_val}")

        # 2. Remove Duplicates
        duplicates_removed = 0
        if initial_duplicates > 0:
            df = df.drop_duplicates()
            duplicates_removed = initial_duplicates
            preprocessing_steps.append(f"Removed {duplicates_removed} duplicate rows")

        # 3. Handle Outliers (IQR method for numeric columns)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        outliers_handled = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                outliers_handled.append({
                    'column': col,
                    'outliers_count': int(outliers),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                })
                preprocessing_steps.append(f"Capped {outliers} outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")

        # 4. Data Type Optimization
        dtype_changes = []
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].nunique() < 10:
                df[col] = df[col].astype('int8')
                dtype_changes.append(f"{col}: float64 → int8")
            elif df[col].max() < 255 and df[col].min() >= 0:
                df[col] = df[col].astype('uint8')
                dtype_changes.append(f"{col}: float64 → uint8")

        # Save preprocessed data
        preprocessed_path = dataset['file_path'].replace('.csv', '_preprocessed.csv')
        df.to_csv(preprocessed_path, index=False)

        # Prepare response
        result = {
            'initial_shape': initial_shape,
            'final_shape': df.shape,
            'rows_removed': int(initial_shape[0] - df.shape[0]),
            'initial_nulls': int(initial_nulls),
            'final_nulls': int(df.isnull().sum().sum()),
            'duplicates_removed': int(duplicates_removed),
            'missing_cols_handled': missing_cols,
            'outliers_handled': outliers_handled,
            'dtype_changes': dtype_changes,
            'preprocessing_steps': preprocessing_steps,
            'preprocessed_filepath': preprocessed_path,
            'data_quality_score': calculate_data_quality_score(df)
        }

        return jsonify({
            'success': True,
            'result': result,
            'message': f'Preprocessing completed: {len(preprocessing_steps)} steps applied'
        })

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


def calculate_data_quality_score(df):
    """Calculate overall data quality score (0-100)"""
    import numpy as np

    score = 100

    # Penalize for missing values
    null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    score -= null_percentage * 2

    # Penalize for duplicates
    duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
    score -= duplicate_percentage * 1.5

    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    score -= len(constant_cols) * 5

    return max(0, min(100, score))


@workflow_bp.route('/api/feature-engineering/<int:dataset_id>', methods=['POST'])
def generate_feature_engineering(dataset_id):
    """Run Fully Automated Feature Selection - Runs ALL methods and recommends best"""
    try:
        import pandas as pd
        import numpy as np
        from routes.feature_selection import AdvancedFeatureSelector

        # Get dataset info
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404

        # Load dataset
        df = load_dataset(dataset)
        if df is None:
            return jsonify({'success': False, 'error': 'Failed to load dataset'}), 500

        # Check if dataset has enough columns
        if len(df.columns) < 2:
            return jsonify({'success': False, 'error': 'Dataset must have at least 2 columns (features + target)'}), 400

        # Assume last column is target
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True).fillna(0)

        selector = AdvancedFeatureSelector()
        all_results = {}
        detailed_logs = []

        logger.info("="*60)
        logger.info("RUNNING FULLY AUTOMATED FEATURE SELECTION (3 METHODS)")
        logger.info("="*60)
        logger.info(f"Dataset: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
        logger.info(f"Target: {target_column}")
        logger.info(f"Task Type: {'Classification' if y.dtype == 'object' or y.nunique() < 10 else 'Regression'}")

        detailed_logs.append(f"Dataset: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
        detailed_logs.append(f"Target: {target_column}")

        # 1. Forward Selection
        logger.info("\n" + "="*60)
        logger.info("[1/3] FORWARD SELECTION")
        logger.info("="*60)
        detailed_logs.append("\n[1/3] FORWARD SELECTION")
        try:
            result = selector.forward_selection(X_encoded, y, max_features=5, cv=5)
            all_results['forward_selection'] = {
                'selected_features': result['selected_features'],
                'n_features': result['n_features'],
                'score': result['final_score'],
                'cv_details': result.get('cv_details', []),
                'cv_folds': result.get('cv_folds', 5),
                'scoring_metric': result.get('scoring_metric', 'accuracy'),
                'method': 'Forward Selection',
                'description': 'Iteratively adds features that improve model performance'
            }
            detailed_logs.append(f"Selected {result['n_features']} features with CV score: {result['final_score']:.4f}")
            logger.info(f"\nFORWARD SELECTION COMPLETE: {result['n_features']} features selected")
        except Exception as e:
            all_results['forward_selection'] = {'error': str(e)}
            detailed_logs.append(f"ERROR: {str(e)}")
            logger.error(f"ERROR: {str(e)}")

        # 2. Backward Elimination
        logger.info("\n" + "="*60)
        logger.info("[2/3] BACKWARD ELIMINATION")
        logger.info("="*60)
        detailed_logs.append("\n[2/3] BACKWARD ELIMINATION")
        try:
            result = selector.backward_elimination(X_encoded, y, cv=5, threshold=0.005)
            all_results['backward_elimination'] = {
                'selected_features': result['selected_features'],
                'n_features': result['n_features'],
                'score': result['final_score'],
                'cv_details': result.get('cv_details', []),
                'cv_folds': result.get('cv_folds', 5),
                'scoring_metric': result.get('scoring_metric', 'accuracy'),
                'method': 'Backward Elimination',
                'description': 'Iteratively removes features that hurt model performance'
            }
            detailed_logs.append(f"Selected {result['n_features']} features with CV score: {result['final_score']:.4f}")
            logger.info(f"\nBACKWARD ELIMINATION COMPLETE: {result['n_features']} features selected")
        except Exception as e:
            all_results['backward_elimination'] = {'error': str(e)}
            detailed_logs.append(f"ERROR: {str(e)}")
            logger.error(f"ERROR: {str(e)}")

        # 3. Stepwise Selection
        logger.info("\n" + "="*60)
        logger.info("[3/3] STEPWISE SELECTION")
        logger.info("="*60)
        detailed_logs.append("\n[3/3] STEPWISE SELECTION")
        try:
            result = selector.stepwise_selection(X_encoded, y, max_features=5, cv=5, threshold=0.005)
            all_results['stepwise_selection'] = {
                'selected_features': result['selected_features'],
                'n_features': result['n_features'],
                'score': result['final_score'],
                'cv_details': result.get('cv_details', []),
                'cv_folds': result.get('cv_folds', 5),
                'scoring_metric': result.get('scoring_metric', 'accuracy'),
                'method': 'Stepwise Selection',
                'description': 'Combines forward and backward: adds best features, removes worst features'
            }
            detailed_logs.append(f"Selected {result['n_features']} features with CV score: {result['final_score']:.4f}")
            logger.info(f"\nSTEPWISE SELECTION COMPLETE: {result['n_features']} features selected")
        except Exception as e:
            all_results['stepwise_selection'] = {'error': str(e)}
            detailed_logs.append(f"ERROR: {str(e)}")
            logger.error(f"ERROR: {str(e)}")

        # Calculate best method
        logger.info("\n" + "="*60)
        logger.info("CALCULATING BEST METHOD")
        logger.info("="*60)
        detailed_logs.append("\n" + "="*60)
        detailed_logs.append("CALCULATING BEST METHOD")
        detailed_logs.append("="*60)

        best_method = None
        best_score = -np.inf
        scoring_details = []

        # Score each method
        for method_name, result in all_results.items():
            if 'error' in result:
                continue

            # Calculate composite score
            n_features = result['n_features']
            model_score = result.get('score', 0)

            # Scoring criteria:
            # 1. Model performance (if available) - 50%
            # 2. Feature reduction (fewer is better) - 30%
            # 3. Method reliability - 20%

            if model_score:
                performance_score = abs(model_score) * 0.5
            else:
                performance_score = 0

            # Feature reduction score (prefer 30-50% of original features)
            original_features = len(X_encoded.columns)
            ideal_ratio = 0.4
            actual_ratio = n_features / original_features if original_features > 0 else 0
            reduction_score = (1 - abs(actual_ratio - ideal_ratio)) * 0.3

            # Method reliability scores (based on literature and empirical studies)
            reliability_scores = {
                'forward_selection': 0.9,  # High: Greedy but effective
                'backward_elimination': 0.9,  # High: Comprehensive evaluation
                'stepwise_selection': 0.95  # Highest: Combines both approaches
            }
            reliability_score = reliability_scores.get(method_name, 0.5) * 0.2

            composite_score = performance_score + reduction_score + reliability_score
            result['composite_score'] = float(composite_score)
            result['score_breakdown'] = {
                'performance': float(performance_score),
                'reduction': float(reduction_score),
                'reliability': float(reliability_score)
            }

            # Log scoring details
            score_detail = {
                'method': method_name,
                'n_features': n_features,
                'cv_score': model_score if model_score else 'N/A',
                'composite_score': float(composite_score),
                'breakdown': {
                    'performance': f"{performance_score:.4f} (50% weight)",
                    'reduction': f"{reduction_score:.4f} (30% weight)",
                    'reliability': f"{reliability_score:.4f} (20% weight)"
                }
            }
            scoring_details.append(score_detail)

            logger.info(f"\n{method_name.upper().replace('_', ' ')}:")
            logger.info(f"  Features: {n_features}/{original_features} ({n_features/original_features*100 if original_features > 0 else 0:.1f}%)")
            if model_score:
                logger.info(f"  CV Score: {model_score:.4f}")
            logger.info(f"  Composite Score: {composite_score:.4f}")
            logger.info(f"    - Performance: {performance_score:.4f} (50%)")
            logger.info(f"    - Reduction: {reduction_score:.4f} (30%)")
            logger.info(f"    - Reliability: {reliability_score:.4f} (20%)")

            detailed_logs.append(f"{method_name}: {n_features} features, composite score: {composite_score:.4f}")

            if composite_score > best_score:
                best_score = composite_score
                best_method = method_name

        # Prepare recommendation
        recommendation = {
            'best_method': best_method,
            'best_method_details': all_results[best_method] if best_method else None,
            'all_results': all_results,
            'original_features': len(X_encoded.columns),
            'scoring_details': scoring_details,
            'detailed_logs': detailed_logs,
            'recommendation_reason': f"Selected based on composite score of {best_score:.3f} considering model performance (50%), feature reduction (30%), and method reliability (20%)"
        }

        logger.info("\n" + "="*60)
        logger.info("RECOMMENDATION")
        logger.info("="*60)
        if best_method:
            logger.info(f"BEST METHOD: {best_method.upper().replace('_', ' ')}")
            logger.info(f"COMPOSITE SCORE: {best_score:.4f}")
            logger.info(f"SELECTED FEATURES: {all_results[best_method]['n_features']}/{len(X_encoded.columns)}")
            logger.info(f"REASON: {recommendation['recommendation_reason']}")
        logger.info("="*60)

        if best_method:
            detailed_logs.append(f"\nRECOMMENDATION: {best_method} with {all_results[best_method]['n_features']} features")

        return jsonify({
            'success': True,
            'recommendation': recommendation,
            'message': f'Feature selection completed. Recommended method: {best_method.replace("_", " ").title()}'
        })

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@workflow_bp.route('/api/export-notebook/<int:dataset_id>', methods=['POST'])
def export_workflow_notebook(dataset_id):
    """Export workflow as Jupyter notebook with real code"""
    try:
        import json
        from datetime import datetime

        # Get workflow configuration from request
        config = request.get_json() or {}
        nodes = config.get('nodes', [])

        # Get dataset info
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404

        # Create notebook structure
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        # Add header markdown cell
        header_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Data Analysis Workflow\n",
                f"**Dataset:** {dataset['name']}\n",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                f"\n",
                f"This notebook contains the complete workflow for data analysis.\n"
            ]
        }
        notebook["cells"].append(header_cell)

        # Add imports cell
        imports_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n",
                "from sklearn.metrics import classification_report, mean_squared_error, r2_score\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set visualization style\n",
                "sns.set_style('whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 8)\n"
            ]
        }
        notebook["cells"].append(imports_cell)

        # Generate code for each node in the workflow
        for i, node in enumerate(nodes):
            node_type = node.get('type')
            node_config = node.get('config', {})

            if node_type == 'data':
                # Data loading node
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## Step {i+1}: Load Data\n"]
                }
                notebook["cells"].append(cell)

                code_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        f"# Load dataset\n",
                        f"df = pd.read_csv('{dataset['file_path']}')\n",
                        f"print(f'Dataset shape: {{df.shape}}')\n",
                        f"print(f'Columns: {{list(df.columns)}}')\n",
                        f"df.head()\n"
                    ]
                }
                notebook["cells"].append(code_cell)

            elif node_type == 'statistics':
                # Statistics node
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## Step {i+1}: Statistical Analysis\n"]
                }
                notebook["cells"].append(cell)

                code_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Generate statistical summary\n",
                        "print('Dataset Information:')\n",
                        "print(df.info())\n",
                        "print('\\nDescriptive Statistics:')\n",
                        "print(df.describe())\n",
                        "print('\\nMissing Values:')\n",
                        "print(df.isnull().sum())\n",
                        "print('\\nData Types:')\n",
                        "print(df.dtypes)\n"
                    ]
                }
                notebook["cells"].append(code_cell)

            elif node_type == 'visualization':
                # Visualization node
                viz_type = node_config.get('visualizationType', 'correlation')
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## Step {i+1}: Data Visualization ({viz_type})\n"]
                }
                notebook["cells"].append(cell)

                if viz_type == 'correlation':
                    code_cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "# Correlation heatmap\n",
                            "numeric_df = df.select_dtypes(include=[np.number])\n",
                            "plt.figure(figsize=(12, 8))\n",
                            "sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')\n",
                            "plt.title('Correlation Heatmap')\n",
                            "plt.tight_layout()\n",
                            "plt.show()\n"
                        ]
                    }
                elif viz_type == 'distribution':
                    code_cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "# Distribution plots\n",
                            "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
                            "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
                            "axes = axes.flatten()\n",
                            "for i, col in enumerate(numeric_cols[:4]):\n",
                            "    sns.histplot(df[col].dropna(), kde=True, ax=axes[i])\n",
                            "    axes[i].set_title(f'Distribution of {col}')\n",
                            "plt.tight_layout()\n",
                            "plt.show()\n"
                        ]
                    }
                else:
                    code_cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            f"# {viz_type} visualization\n",
                            "# Add your visualization code here\n"
                        ]
                    }
                notebook["cells"].append(code_cell)

            elif node_type == 'preprocessing':
                # Preprocessing node
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## Step {i+1}: Data Preprocessing\n"]
                }
                notebook["cells"].append(cell)

                code_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Data Preprocessing - Handle missing values, outliers, and duplicates\n",
                        "from scipy import stats\n",
                        "\n",
                        "print('Initial dataset shape:', df.shape)\n",
                        "print('Initial missing values:', df.isnull().sum().sum())\n",
                        "print('Initial duplicates:', df.duplicated().sum())\n",
                        "\n",
                        "# 1. Handle Missing Values\n",
                        "for col in df.columns:\n",
                        "    if df[col].isnull().any():\n",
                        "        if df[col].dtype in ['float64', 'int64']:\n",
                        "            median_val = df[col].median()\n",
                        "            df[col].fillna(median_val, inplace=True)\n",
                        "            print(f'Filled {col} missing values with median: {median_val:.2f}')\n",
                        "        else:\n",
                        "            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'\n",
                        "            df[col].fillna(mode_val, inplace=True)\n",
                        "            print(f'Filled {col} missing values with mode: {mode_val}')\n",
                        "\n",
                        "# 2. Remove Duplicates\n",
                        "duplicates_before = df.duplicated().sum()\n",
                        "if duplicates_before > 0:\n",
                        "    df = df.drop_duplicates()\n",
                        "    print(f'\\nRemoved {duplicates_before} duplicate rows')\n",
                        "\n",
                        "# 3. Handle Outliers using IQR method\n",
                        "numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns\n",
                        "for col in numeric_cols:\n",
                        "    Q1 = df[col].quantile(0.25)\n",
                        "    Q3 = df[col].quantile(0.75)\n",
                        "    IQR = Q3 - Q1\n",
                        "    lower_bound = Q1 - 1.5 * IQR\n",
                        "    upper_bound = Q3 + 1.5 * IQR\n",
                        "    \n",
                        "    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()\n",
                        "    if outliers > 0:\n",
                        "        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)\n",
                        "        print(f'Capped {outliers} outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]')\n",
                        "\n",
                        "print('\\nPreprocessing completed!')\n",
                        "print(f'Final dataset shape: {df.shape}')\n",
                        "print(f'Final missing values: {df.isnull().sum().sum()}')\n"
                    ]
                }
                notebook["cells"].append(code_cell)

            elif node_type == 'feature_engineering':
                # Feature engineering node
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## Step {i+1}: Feature Engineering\n"]
                }
                notebook["cells"].append(cell)

                code_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Feature engineering\n",
                        "# Encode categorical variables\n",
                        "label_encoders = {}\n",
                        "for col in df.select_dtypes(include=['object']).columns:\n",
                        "    le = LabelEncoder()\n",
                        "    df[col] = le.fit_transform(df[col].astype(str))\n",
                        "    label_encoders[col] = le\n",
                        "\n",
                        "print('Feature engineering completed!')\n",
                        "print(f'Final dataset shape: {df.shape}')\n"
                    ]
                }
                notebook["cells"].append(code_cell)

            elif node_type == 'ml_model':
                # ML model node
                model_type = node_config.get('modelType', 'classification')
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## Step {i+1}: Machine Learning Model ({model_type})\n"]
                }
                notebook["cells"].append(cell)

                if model_type == 'classification':
                    code_cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "# Prepare data for classification\n",
                            "# Assuming last column is target\n",
                            "X = df.iloc[:, :-1]\n",
                            "y = df.iloc[:, -1]\n",
                            "\n",
                            "# Split data\n",
                            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                            "\n",
                            "# Scale features\n",
                            "scaler = StandardScaler()\n",
                            "X_train_scaled = scaler.fit_transform(X_train)\n",
                            "X_test_scaled = scaler.transform(X_test)\n",
                            "\n",
                            "# Train Random Forest Classifier\n",
                            "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                            "model.fit(X_train_scaled, y_train)\n",
                            "\n",
                            "# Evaluate\n",
                            "y_pred = model.predict(X_test_scaled)\n",
                            "print('Classification Report:')\n",
                            "print(classification_report(y_test, y_pred))\n",
                            "\n",
                            "# Feature importance\n",
                            "feature_importance = pd.DataFrame({\n",
                            "    'feature': X.columns,\n",
                            "    'importance': model.feature_importances_\n",
                            "}).sort_values('importance', ascending=False)\n",
                            "print('\\nTop 10 Important Features:')\n",
                            "print(feature_importance.head(10))\n"
                        ]
                    }
                else:
                    code_cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "# Prepare data for regression\n",
                            "# Assuming last column is target\n",
                            "X = df.iloc[:, :-1]\n",
                            "y = df.iloc[:, -1]\n",
                            "\n",
                            "# Split data\n",
                            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                            "\n",
                            "# Scale features\n",
                            "scaler = StandardScaler()\n",
                            "X_train_scaled = scaler.fit_transform(X_train)\n",
                            "X_test_scaled = scaler.transform(X_test)\n",
                            "\n",
                            "# Train Random Forest Regressor\n",
                            "model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
                            "model.fit(X_train_scaled, y_train)\n",
                            "\n",
                            "# Evaluate\n",
                            "y_pred = model.predict(X_test_scaled)\n",
                            "print(f'R2 Score: {r2_score(y_test, y_pred):.4f}')\n",
                            "print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\n",
                            "\n",
                            "# Feature importance\n",
                            "feature_importance = pd.DataFrame({\n",
                            "    'feature': X.columns,\n",
                            "    'importance': model.feature_importances_\n",
                            "}).sort_values('importance', ascending=False)\n",
                            "print('\\nTop 10 Important Features:')\n",
                            "print(feature_importance.head(10))\n"
                        ]
                    }
                notebook["cells"].append(code_cell)

        # Add conclusion cell
        conclusion_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n",
                "\n",
                "This notebook contains the complete workflow for analyzing the dataset. ",
                "You can now modify and run each cell to perform your analysis.\n"
            ]
        }
        notebook["cells"].append(conclusion_cell)

        return jsonify({
            'success': True,
            'notebook': notebook,
            'filename': f'workflow_{dataset["name"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
