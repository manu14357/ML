"""
Advanced Workflow API endpoints for managing data science workflows
"""

from flask import Blueprint, request, jsonify, current_app
from app.models import Workflow, Dataset
from app import db
from app.services.workflow_service import AdvancedWorkflowService
import json
import pandas as pd
from datetime import datetime

bp = Blueprint('workflows', __name__)
workflow_service = AdvancedWorkflowService()

@bp.route('/', methods=['GET'])
def workflows_root():
    """Root workflows endpoint - provide basic info"""
    return jsonify({
        'success': True,
        'message': 'Advanced Workflow API',
        'version': '2.0.0',
        'endpoints': {
            'workflows': '/api/workflows/workflows',
            'node_types': '/api/workflows/node-types',
            'datasets': '/api/workflows/datasets'
        }
    })

@bp.route('/datasets', methods=['GET'])
def get_available_datasets():
    """Get all available datasets for workflow nodes"""
    try:
        datasets = workflow_service.get_available_datasets()
        return jsonify({
            'success': True,
            'datasets': datasets,
            'total': len(datasets)
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching datasets: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/datasets/<dataset_id>/analysis', methods=['GET'])
def get_dataset_analysis(dataset_id):
    """Get column analysis for a specific dataset"""
    try:
        # Get dataset
        dataset = Dataset.query.filter_by(id=dataset_id).first()
        if not dataset:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        # Load and analyze the dataset
        df = workflow_service.load_dataset_for_analysis(dataset_id)
        
        # Perform column analysis
        classification_analysis = workflow_service._analyze_columns_for_classification(df)
        regression_analysis = workflow_service._analyze_columns_for_regression(df)
        
        # Combine analyses
        column_info = {}
        for col in df.columns:
            column_info[col] = {
                'name': col,
                'dtype': str(df[col].dtype),
                'unique_count': int(df[col].nunique()),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                'sample_values': df[col].dropna().head(5).tolist(),
                'classification_role': classification_analysis['columns'].get(col, {}).get('recommended_role'),
                'regression_role': regression_analysis['columns'].get(col, {}).get('recommended_role'),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_categorical': df[col].dtype == 'object',
                'is_datetime': pd.api.types.is_datetime64_any_dtype(df[col])
            }
        
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'dataset_name': dataset.filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': column_info,
            'recommendations': {
                'classification': {
                    'target_candidates': classification_analysis['recommendations']['target_candidates'],
                    'feature_candidates': classification_analysis['recommendations']['feature_candidates'][:20]  # Limit for UI
                },
                'regression': {
                    'target_candidates': regression_analysis['recommendations']['target_candidates'],
                    'feature_candidates': regression_analysis['recommendations']['feature_candidates'][:20]
                }
            },
            'column_categories': {
                'numeric_columns': [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                'categorical_columns': [col for col in df.columns if df[col].dtype == 'object'],
                'datetime_columns': [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            }
        })
    except Exception as e:
        current_app.logger.error(f"Error analyzing dataset columns: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/node-types', methods=['GET'])
def get_node_types():
    """Get all available node types and their configurations"""
    try:
        node_types = workflow_service.get_node_types()
        return jsonify({
            'success': True,
            'node_types': node_types,
            'categories': list(set(nt['category'] for nt in node_types.values()))
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching node types: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/validate', methods=['POST'])
def validate_workflow():
    """Validate a workflow configuration"""
    try:
        data = request.get_json()
        nodes = data.get('nodes', [])
        connections = data.get('connections', [])
        
        validation = workflow_service.validate_workflow(nodes, connections)
        return jsonify({
            'success': True,
            'validation': validation
        })
    except Exception as e:
        current_app.logger.error(f"Error validating workflow: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows', methods=['GET', 'OPTIONS'])
def get_workflows():
    """Get all workflows"""
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
        
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        workflows = Workflow.query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        workflows_data = []
        for w in workflows.items:
            try:
                # Safely parse JSON fields
                nodes = json.loads(w.nodes) if w.nodes else []
                connections = json.loads(w.connections) if w.connections else []
            except (json.JSONDecodeError, TypeError):
                # Handle malformed JSON
                nodes = []
                connections = []
            
            workflows_data.append({
                'id': w.id,
                'name': w.name,
                'description': w.description,
                'status': w.status,
                'created_at': w.created_at.isoformat() if w.created_at else None,
                'updated_at': w.updated_at.isoformat() if w.updated_at else None,
                'nodes': nodes,
                'connections': connections
            })
        
        return jsonify({
            'success': True,
            'workflows': workflows_data,
            'total': workflows.total,
            'pages': workflows.pages,
            'current_page': page
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching workflows: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows', methods=['POST', 'OPTIONS'])
def create_workflow():
    """Create a new workflow"""
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
        
    try:
        data = request.get_json()
        
        if not data or not data.get('name'):
            return jsonify({'success': False, 'error': 'Workflow name is required'}), 400
        
        # Get nodes and connections from request data
        nodes = data.get('nodes', [])
        connections = data.get('connections', [])
        
        workflow = Workflow(
            name=data['name'],
            description=data.get('description', ''),
            status='draft',
            nodes=json.dumps(nodes),
            connections=json.dumps(connections)
        )
        
        db.session.add(workflow)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'workflow': {
                'id': workflow.id,
                'name': workflow.name,
                'description': workflow.description,
                'status': workflow.status,
                'created_at': workflow.created_at.isoformat(),
                'updated_at': workflow.updated_at.isoformat(),
                'nodes': nodes,
                'connections': connections
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error creating workflow: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows/<int:workflow_id>', methods=['GET'])
def get_workflow(workflow_id):
    """Get a specific workflow"""
    try:
        workflow = Workflow.query.get_or_404(workflow_id)
        
        return jsonify({
            'success': True,
            'workflow': {
                'id': workflow.id,
                'name': workflow.name,
                'description': workflow.description,
                'status': workflow.status,
                'created_at': workflow.created_at.isoformat() if workflow.created_at else None,
                'updated_at': workflow.updated_at.isoformat() if workflow.updated_at else None,
                'nodes': json.loads(workflow.nodes) if workflow.nodes else [],
                'connections': json.loads(workflow.connections) if workflow.connections else []
            }
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching workflow {workflow_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows/<int:workflow_id>', methods=['PUT', 'OPTIONS'])
def update_workflow(workflow_id):
    """Update a workflow"""
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
        
    try:
        workflow = Workflow.query.get_or_404(workflow_id)
        data = request.get_json()
        
        if 'name' in data:
            workflow.name = data['name']
        if 'description' in data:
            workflow.description = data['description']
        if 'nodes' in data:
            workflow.nodes = json.dumps(data['nodes'])
        if 'connections' in data:
            workflow.connections = json.dumps(data['connections'])
        if 'status' in data:
            workflow.status = data['status']
            
        workflow.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'workflow': {
                'id': workflow.id,
                'name': workflow.name,
                'description': workflow.description,
                'status': workflow.status,
                'created_at': workflow.created_at.isoformat() if workflow.created_at else None,
                'updated_at': workflow.updated_at.isoformat() if workflow.updated_at else None,
                'nodes': json.loads(workflow.nodes) if workflow.nodes else [],
                'connections': json.loads(workflow.connections) if workflow.connections else []
            }
        })
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error updating workflow {workflow_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows/<int:workflow_id>', methods=['DELETE'])
def delete_workflow(workflow_id):
    """Delete a workflow"""
    try:
        workflow = Workflow.query.get_or_404(workflow_id)
        db.session.delete(workflow)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Workflow deleted successfully'})
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error deleting workflow {workflow_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows/<int:workflow_id>/run', methods=['POST', 'OPTIONS'])
def run_workflow(workflow_id):
    """Execute a workflow"""
    if request.method == 'OPTIONS':
        # Handle CORS preflight request
        response = jsonify({'success': True})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
        
    try:
        workflow = Workflow.query.get_or_404(workflow_id)
        data = request.get_json()
        
        # Use nodes and connections from the request if provided, otherwise use saved ones
        nodes = data.get('nodes') if data and 'nodes' in data else json.loads(workflow.nodes) if workflow.nodes else []
        connections = data.get('connections') if data and 'connections' in data else json.loads(workflow.connections) if workflow.connections else []
        
        # Update workflow with latest nodes and connections if provided in request
        if data and ('nodes' in data or 'connections' in data):
            workflow.nodes = json.dumps(nodes)
            workflow.connections = json.dumps(connections)
        
        workflow.status = 'running'
        workflow.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Execute the workflow
        execution_results = workflow_service.execute_workflow(workflow_id, nodes, connections)
        
        # Update workflow status
        workflow.status = 'completed' if execution_results.get('success') else 'failed'
        workflow.execution_results = json.dumps(execution_results)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'execution_results': execution_results,
            'workflow_status': workflow.status
        })
        
    except Exception as e:
        # Update workflow status to failed
        workflow = Workflow.query.get(workflow_id)
        if workflow:
            workflow.status = 'failed'
            db.session.commit()
        
        current_app.logger.error(f"Error running workflow {workflow_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows/<int:workflow_id>/duplicate', methods=['POST'])
def duplicate_workflow(workflow_id):
    """Duplicate a workflow"""
    try:
        original_workflow = Workflow.query.get_or_404(workflow_id)
        
        duplicated_workflow = Workflow(
            name=f"{original_workflow.name} (Copy)",
            description=original_workflow.description,
            status='draft',
            nodes=original_workflow.nodes,
            connections=original_workflow.connections
        )
        
        db.session.add(duplicated_workflow)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'workflow': {
                'id': duplicated_workflow.id,
                'name': duplicated_workflow.name,
                'description': duplicated_workflow.description,
                'status': duplicated_workflow.status,
                'created_at': duplicated_workflow.created_at.isoformat(),
                'updated_at': duplicated_workflow.updated_at.isoformat(),
                'nodes': json.loads(duplicated_workflow.nodes) if duplicated_workflow.nodes else [],
                'connections': json.loads(duplicated_workflow.connections) if duplicated_workflow.connections else []
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error duplicating workflow {workflow_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows/templates', methods=['GET'])
def get_workflow_templates():
    """Get available workflow templates"""
    try:
        templates = workflow_service.get_workflow_templates()
        return jsonify({
            'success': True,
            'templates': templates
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching workflow templates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/datasets/<int:dataset_id>/columns', methods=['GET'])
def get_dataset_columns(dataset_id):
    """Get column information for a specific dataset"""
    try:
        from app.models import Dataset
        import pandas as pd
        
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Load a sample of the dataset to analyze columns
        if dataset.file_type.lower() == 'csv':
            df = pd.read_csv(dataset.file_path, nrows=100)  # Sample first 100 rows
        elif dataset.file_type.lower() in ['xlsx', 'xls']:
            df = pd.read_excel(dataset.file_path, nrows=100)
        elif dataset.file_type.lower() == 'json':
            df = pd.read_json(dataset.file_path).head(100)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file type'}), 400
        
        # Analyze columns
        columns_info = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'is_numeric': bool(df[col].dtype in ['int64', 'float64', 'int32', 'float32']),
                'is_categorical': bool(df[col].dtype == 'object'),
                'unique_count': int(df[col].nunique()),
                'null_count': int(df[col].isnull().sum()),
                'sample_values': [str(val) for val in df[col].dropna().head(5).tolist()]
            }
            columns_info.append(col_info)
        
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'columns': columns_info,
            'total_columns': len(columns_info),
            'numeric_columns': [c['name'] for c in columns_info if c['is_numeric']],
            'categorical_columns': [c['name'] for c in columns_info if c['is_categorical']]
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting columns for dataset {dataset_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/execute', methods=['POST', 'OPTIONS'])
def execute_workflow():
    """Execute a workflow directly without saving"""
    if request.method == 'OPTIONS':
        # Handle CORS preflight request
        response = jsonify({'success': True})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Get workflow data
        nodes = data.get('nodes', [])
        connections = data.get('connections', [])
        workflow_name = data.get('name', 'Temporary Workflow')
        
        if not nodes:
            return jsonify({'success': False, 'error': 'No nodes provided'}), 400
        
        current_app.logger.info(f"Executing temporary workflow: {workflow_name}")
        
        # Execute the workflow using a temporary ID
        temp_workflow_id = 999999  # Use a high number for temporary workflows
        execution_results = workflow_service.execute_workflow(temp_workflow_id, nodes, connections)
        
        return jsonify({
            'success': True,
            'execution_results': execution_results,
            'workflow_name': workflow_name
        })
        
    except Exception as e:
        current_app.logger.error(f"Error executing temporary workflow: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
