# routes/automated_workflow_routes.py - Automated Workflow Orchestration
import os
import json
import logging
import traceback
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session
from functools import wraps

logger = logging.getLogger(__name__)

automated_workflow_bp = Blueprint('automated_workflow', __name__, url_prefix='/automated-workflow')

# Global reference to datasets
_datasets = None

def set_datasets_reference(datasets_dict):
    global _datasets
    _datasets = datasets_dict

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Workflow execution state storage
workflow_executions = {}

# Define the complete pipeline with all modules
PIPELINE_NODES = [
    {
        'id': 'node1',
        'name': 'Data Upload',
        'module': 'data_upload',
        'description': 'Upload and validate dataset',
        'icon': 'upload'
    },
    {
        'id': 'node2',
        'name': 'Data Cleaning',
        'module': 'data_cleaning',
        'description': 'Handle missing values and duplicates',
        'icon': 'broom'
    },
    {
        'id': 'node3',
        'name': 'Data Profiling',
        'module': 'data_profiling',
        'description': 'Generate statistical summary',
        'icon': 'chart-bar'
    },
    {
        'id': 'node4',
        'name': 'Feature Engineering',
        'module': 'feature_engineering',
        'description': 'Create and transform features',
        'icon': 'cogs'
    },
    {
        'id': 'node5',
        'name': 'ML Training',
        'module': 'ml_training',
        'description': 'Train machine learning models',
        'icon': 'brain'
    },
    {
        'id': 'node6',
        'name': 'Results & Export',
        'module': 'results_export',
        'description': 'View results and export models',
        'icon': 'download'
    }
]

@automated_workflow_bp.route('/')
@login_required
def automated_workflow_page():
    """Render the automated workflow page"""
    return render_template('automated_workflow.html')

@automated_workflow_bp.route('/api/pipeline-structure')
@login_required
def get_pipeline_structure():
    """Get the complete pipeline structure"""
    return jsonify({
        'success': True,
        'nodes': PIPELINE_NODES,
        'edges': [
            {'from': 'node1', 'to': 'node2'},
            {'from': 'node2', 'to': 'node3'},
            {'from': 'node3', 'to': 'node4'},
            {'from': 'node4', 'to': 'node5'},
            {'from': 'node5', 'to': 'node6'}
        ]
    })

@automated_workflow_bp.route('/api/start-execution', methods=['POST'])
@login_required
def start_execution():
    """Start automated workflow execution"""
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        config = data.get('config', {})
        
        if not dataset_id:
            return jsonify({'success': False, 'error': 'Dataset ID required'}), 400
        
        user_id = session.get('user_id')
        if not user_id or user_id not in _datasets:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Find dataset
        dataset = None
        for ds in _datasets[user_id]:
            if str(ds.get('id')) == str(dataset_id):
                dataset = ds
                break
        
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        # Create execution ID
        execution_id = f"exec_{user_id}_{dataset_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize execution state
        workflow_executions[execution_id] = {
            'id': execution_id,
            'user_id': user_id,
            'dataset_id': dataset_id,
            'status': 'running',
            'current_node': 'node1',
            'completed_nodes': [],
            'failed_nodes': [],
            'results': {},
            'logs': [],
            'started_at': datetime.now().isoformat(),
            'config': config
        }
        
        return jsonify({
            'success': True,
            'execution_id': execution_id,
            'message': 'Workflow execution started'
        })
        
    except Exception as e:
        logger.error(f"Error starting execution: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@automated_workflow_bp.route('/api/execute-node/<execution_id>/<node_id>', methods=['POST'])
@login_required
def execute_node(execution_id, node_id):
    """Execute a specific node in the workflow"""
    try:
        if execution_id not in workflow_executions:
            return jsonify({'success': False, 'error': 'Execution not found'}), 404
        
        execution = workflow_executions[execution_id]
        user_id = execution['user_id']
        dataset_id = execution['dataset_id']
        
        # Find the node
        node = next((n for n in PIPELINE_NODES if n['id'] == node_id), None)
        if not node:
            return jsonify({'success': False, 'error': 'Node not found'}), 404
        
        # Execute the node based on module
        result = execute_module(node['module'], user_id, dataset_id, execution['config'])
        
        # Update execution state
        if result['success']:
            execution['completed_nodes'].append(node_id)
            execution['results'][node_id] = result['data']
            execution['logs'].append({
                'timestamp': datetime.now().isoformat(),
                'node_id': node_id,
                'status': 'success',
                'message': result.get('message', f"{node['name']} completed successfully")
            })
            
            # Move to next node
            current_index = next((i for i, n in enumerate(PIPELINE_NODES) if n['id'] == node_id), -1)
            if current_index < len(PIPELINE_NODES) - 1:
                execution['current_node'] = PIPELINE_NODES[current_index + 1]['id']
            else:
                execution['status'] = 'completed'
                execution['completed_at'] = datetime.now().isoformat()
        else:
            execution['failed_nodes'].append(node_id)
            execution['status'] = 'failed'
            execution['logs'].append({
                'timestamp': datetime.now().isoformat(),
                'node_id': node_id,
                'status': 'failed',
                'message': result.get('error', 'Unknown error')
            })
        
        return jsonify({
            'success': result['success'],
            'node_id': node_id,
            'result': result,
            'execution': execution
        })
        
    except Exception as e:
        logger.error(f"Error executing node: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

def execute_module(module_name, user_id, dataset_id, config):
    """Execute a specific module"""
    try:
        # Get dataset
        dataset = None
        for ds in _datasets[user_id]:
            if str(ds.get('id')) == str(dataset_id):
                dataset = ds
                break
        
        if not dataset:
            return {'success': False, 'error': 'Dataset not found'}
        
        # Execute based on module
        if module_name == 'data_upload':
            return execute_data_upload(dataset, config)
        elif module_name == 'data_cleaning':
            return execute_data_cleaning(dataset, config)
        elif module_name == 'data_profiling':
            return execute_data_profiling(dataset, config)
        elif module_name == 'feature_engineering':
            return execute_feature_engineering(dataset, config)
        elif module_name == 'ml_training':
            return execute_ml_training(dataset, config)
        elif module_name == 'results_export':
            return execute_results_export(dataset, config)
        else:
            return {'success': False, 'error': f'Unknown module: {module_name}'}
            
    except Exception as e:
        logger.error(f"Error in execute_module: {e}\n{traceback.format_exc()}")
        return {'success': False, 'error': str(e)}

def execute_data_upload(dataset, config):
    """Execute data upload validation"""
    try:
        import pandas as pd
        df = pd.read_csv(dataset['filepath'])
        
        return {
            'success': True,
            'message': 'Dataset loaded successfully',
            'data': {
                'rows': len(df),
                'columns': len(df.columns),
                'size': os.path.getsize(dataset['filepath']),
                'columns_list': df.columns.tolist()
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_data_cleaning(dataset, config):
    """Execute data cleaning"""
    try:
        import pandas as pd
        df = pd.read_csv(dataset['filepath'])
        
        initial_rows = len(df)
        initial_nulls = df.isnull().sum().sum()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values based on config
        strategy = config.get('missing_value_strategy', 'drop')
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        
        # Save cleaned data
        cleaned_path = dataset['filepath'].replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_path, index=False)
        dataset['cleaned_filepath'] = cleaned_path
        
        return {
            'success': True,
            'message': 'Data cleaning completed',
            'data': {
                'initial_rows': initial_rows,
                'final_rows': len(df),
                'rows_removed': initial_rows - len(df),
                'initial_nulls': int(initial_nulls),
                'final_nulls': int(df.isnull().sum().sum()),
                'cleaned_filepath': cleaned_path
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_data_profiling(dataset, config):
    """Execute data profiling"""
    try:
        import pandas as pd
        filepath = dataset.get('cleaned_filepath', dataset['filepath'])
        df = pd.read_csv(filepath)
        
        profile = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            'categorical_summary': {col: df[col].value_counts().head(5).to_dict() 
                                   for col in df.select_dtypes(include=['object']).columns}
        }
        
        return {
            'success': True,
            'message': 'Data profiling completed',
            'data': profile
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_feature_engineering(dataset, config):
    """Execute feature engineering"""
    try:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        filepath = dataset.get('cleaned_filepath', dataset['filepath'])
        df = pd.read_csv(filepath)
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        
        # Scale numeric features if requested
        if config.get('scale_features', True):
            numeric_cols = df.select_dtypes(include=['number']).columns
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Save engineered data
        engineered_path = dataset['filepath'].replace('.csv', '_engineered.csv')
        df.to_csv(engineered_path, index=False)
        dataset['engineered_filepath'] = engineered_path
        
        return {
            'success': True,
            'message': 'Feature engineering completed',
            'data': {
                'encoded_columns': list(categorical_cols),
                'scaled': config.get('scale_features', True),
                'final_shape': df.shape,
                'engineered_filepath': engineered_path
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_ml_training(dataset, config):
    """Execute ML model training"""
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        filepath = dataset.get('engineered_filepath', dataset.get('cleaned_filepath', dataset['filepath']))
        df = pd.read_csv(filepath)
        
        # Determine target column
        target_col = config.get('target_column')
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Determine problem type
        is_classification = len(y.unique()) < 20
        
        # Train model
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            metric_name = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            metric_name = 'r2_score'
        
        # Save model
        import joblib
        model_path = dataset['filepath'].replace('.csv', '_model.pkl')
        joblib.dump(model, model_path)
        dataset['model_filepath'] = model_path
        
        return {
            'success': True,
            'message': 'ML training completed',
            'data': {
                'model_type': 'RandomForestClassifier' if is_classification else 'RandomForestRegressor',
                'target_column': target_col,
                'train_size': len(X_train),
                'test_size': len(X_test),
                metric_name: float(score),
                'model_filepath': model_path
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def execute_results_export(dataset, config):
    """Execute results export"""
    try:
        results = {
            'dataset_name': dataset.get('filename'),
            'original_filepath': dataset.get('filepath'),
            'cleaned_filepath': dataset.get('cleaned_filepath'),
            'engineered_filepath': dataset.get('engineered_filepath'),
            'model_filepath': dataset.get('model_filepath'),
            'completed_at': datetime.now().isoformat()
        }
        
        # Save results summary
        results_path = dataset['filepath'].replace('.csv', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return {
            'success': True,
            'message': 'Results exported successfully',
            'data': results
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@automated_workflow_bp.route('/api/execution-status/<execution_id>')
@login_required
def get_execution_status(execution_id):
    """Get execution status"""
    if execution_id not in workflow_executions:
        return jsonify({'success': False, 'error': 'Execution not found'}), 404
    
    return jsonify({
        'success': True,
        'execution': workflow_executions[execution_id]
    })

@automated_workflow_bp.route('/api/user-datasets')
@login_required
def get_user_datasets():
    """Get user's datasets"""
    user_id = session.get('user_id')
    if not user_id or user_id not in _datasets:
        return jsonify({'success': True, 'datasets': []})
    
    datasets_list = [
        {
            'id': ds.get('id'),
            'filename': ds.get('filename'),
            'uploaded_at': ds.get('uploaded_at')
        }
        for ds in _datasets[user_id]
    ]
    
    return jsonify({'success': True, 'datasets': datasets_list})
