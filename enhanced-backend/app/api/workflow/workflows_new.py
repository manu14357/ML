"""
Advanced Workflow API endpoints for managing data science workflows
"""

from flask import Blueprint, request, jsonify, current_app
from app.models import Workflow, Dataset, db
from app.services.workflow_service import AdvancedWorkflowService
import json
from datetime import datetime

bp = Blueprint('workflows', __name__)
workflow_service = AdvancedWorkflowService()

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

@bp.route('/workflows', methods=['GET'])
def get_workflows():
    """Get all workflows"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        workflows = Workflow.query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'workflows': [{
                'id': w.id,
                'name': w.name,
                'description': w.description,
                'status': w.status,
                'created_at': w.created_at.isoformat() if w.created_at else None,
                'updated_at': w.updated_at.isoformat() if w.updated_at else None,
                'nodes': json.loads(w.nodes) if w.nodes else [],
                'connections': json.loads(w.connections) if w.connections else []
            } for w in workflows.items],
            'total': workflows.total,
            'pages': workflows.pages,
            'current_page': page
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching workflows: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/workflows', methods=['POST'])
def create_workflow():
    """Create a new workflow"""
    try:
        data = request.get_json()
        
        if not data or not data.get('name'):
            return jsonify({'success': False, 'error': 'Workflow name is required'}), 400
        
        workflow = Workflow(
            name=data['name'],
            description=data.get('description', ''),
            status='draft',
            nodes='[]',
            connections='[]'
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
                'nodes': [],
                'connections': []
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

@bp.route('/workflows/<int:workflow_id>', methods=['PUT'])
def update_workflow(workflow_id):
    """Update a workflow"""
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

@bp.route('/workflows/<int:workflow_id>/run', methods=['POST'])
def run_workflow(workflow_id):
    """Execute a workflow"""
    try:
        workflow = Workflow.query.get_or_404(workflow_id)
        data = request.get_json()
        
        nodes = data.get('nodes', [])
        connections = data.get('connections', [])
        
        # Update workflow with latest nodes and connections
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
