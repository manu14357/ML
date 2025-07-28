"""
Workflow API endpoints for drag-and-drop workflow builder
"""

from flask import Blueprint
from .workflows import bp as workflows_bp

# Import the old workflow_bp for compatibility
workflow_bp = Blueprint('workflow', __name__)

# Import and register the new advanced workflows blueprint
def register_workflow_routes(app):
    """Register workflow routes"""
    app.register_blueprint(workflows_bp, url_prefix='/api/workflows')
    app.register_blueprint(workflow_bp, url_prefix='/api/workflow')


@workflow_bp.route('/workflows', methods=['GET'])
def get_workflows():
    """Get list of all workflows with filtering and pagination"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        status = request.args.get('status')
        
        # Build query
        query = Workflow.query
        
        if status:
            query = query.filter(Workflow.status == status)
        
        # Apply pagination
        workflows = query.order_by(Workflow.updated_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'workflows': [workflow.to_dict() for workflow in workflows.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': workflows.total,
                'pages': workflows.pages
            }
        })
        
    except Exception as e:
        SystemLog.log_error(
            f'Failed to fetch workflows: {str(e)}',
            category='workflow',
            event_type='fetch_error'
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/workflows', methods=['POST'])
def create_workflow():
    """Create a new workflow"""
    try:
        data = request.get_json()
        
        if not data or not data.get('name'):
            return jsonify({
                'success': False,
                'error': 'Workflow name is required'
            }), 400
        
        # Create workflow
        workflow = Workflow(
            name=data['name'],
            description=data.get('description', ''),
            status='draft'
        )
        
        db.session.add(workflow)
        db.session.commit()
        
        SystemLog.log_info(
            f'Workflow created: {workflow.name}',
            category='workflow',
            event_type='create_success',
            context_data={'workflow_id': workflow.id}
        )
        
        return jsonify({
            'success': True,
            'workflow': workflow.to_dict(),
            'message': 'Workflow created successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        SystemLog.log_error(
            f'Failed to create workflow: {str(e)}',
            category='workflow',
            event_type='create_error'
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/workflows/<int:workflow_id>', methods=['GET'])
def get_workflow(workflow_id):
    """Get a specific workflow with its nodes and connections"""
    try:
        workflow = Workflow.query.get(workflow_id)
        if not workflow:
            return jsonify({
                'success': False,
                'error': 'Workflow not found'
            }), 404
        
        return jsonify({
            'success': True,
            'workflow': workflow.to_dict(include_definition=True)
        })
        
    except Exception as e:
        SystemLog.log_error(
            f'Failed to fetch workflow {workflow_id}: {str(e)}',
            category='workflow',
            event_type='fetch_error',
            context_data={'workflow_id': workflow_id}
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/workflows/<int:workflow_id>', methods=['PUT'])
def update_workflow(workflow_id):
    """Update a workflow's definition (nodes and connections)"""
    try:
        workflow = Workflow.query.get(workflow_id)
        if not workflow:
            return jsonify({
                'success': False,
                'error': 'Workflow not found'
            }), 404
        
        data = request.get_json()
        
        # Update workflow definition
        if 'nodes' in data:
            workflow.nodes = data['nodes']
        
        if 'connections' in data:
            workflow.connections = data['connections']
        
        if 'name' in data:
            workflow.name = data['name']
        
        if 'description' in data:
            workflow.description = data['description']
        
        workflow.updated_at = datetime.utcnow()
        db.session.commit()
        
        SystemLog.log_info(
            f'Workflow updated: {workflow.name}',
            category='workflow',
            event_type='update_success',
            context_data={'workflow_id': workflow.id}
        )
        
        return jsonify({
            'success': True,
            'workflow': workflow.to_dict(include_definition=True),
            'message': 'Workflow updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        SystemLog.log_error(
            f'Failed to update workflow {workflow_id}: {str(e)}',
            category='workflow',
            event_type='update_error',
            context_data={'workflow_id': workflow_id}
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/workflows/<int:workflow_id>', methods=['DELETE'])
def delete_workflow(workflow_id):
    """Delete a workflow"""
    try:
        workflow = Workflow.query.get(workflow_id)
        if not workflow:
            return jsonify({
                'success': False,
                'error': 'Workflow not found'
            }), 404
        
        workflow_name = workflow.name
        db.session.delete(workflow)
        db.session.commit()
        
        SystemLog.log_info(
            f'Workflow deleted: {workflow_name}',
            category='workflow',
            event_type='delete_success',
            context_data={'workflow_id': workflow_id}
        )
        
        return jsonify({
            'success': True,
            'message': 'Workflow deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        SystemLog.log_error(
            f'Failed to delete workflow {workflow_id}: {str(e)}',
            category='workflow',
            event_type='delete_error',
            context_data={'workflow_id': workflow_id}
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/workflows/<int:workflow_id>/run', methods=['POST'])
def run_workflow(workflow_id):
    """Execute a workflow"""
    try:
        workflow = Workflow.query.get(workflow_id)
        if not workflow:
            return jsonify({
                'success': False,
                'error': 'Workflow not found'
            }), 404
        
        data = request.get_json()
        nodes = data.get('nodes', [])
        connections = data.get('connections', [])
        
        if not nodes:
            return jsonify({
                'success': False,
                'error': 'No nodes to execute'
            }), 400
        
        # Execute workflow
        execution_result = workflow_service.execute_workflow(workflow_id, nodes, connections)
        
        if execution_result['success']:
            # Update workflow status
            workflow.status = 'completed'
            workflow.last_run_at = datetime.utcnow()
            workflow.execution_count = (workflow.execution_count or 0) + 1
            db.session.commit()
            
            SystemLog.log_info(
                f'Workflow executed successfully: {workflow.name}',
                category='workflow',
                event_type='execution_success',
                context_data={
                    'workflow_id': workflow.id,
                    'execution_time': execution_result.get('execution_time', 0),
                    'nodes_processed': len(nodes)
                }
            )
            
            return jsonify({
                'success': True,
                'execution_results': execution_result['results'],
                'execution_time': execution_result.get('execution_time', 0),
                'message': 'Workflow executed successfully'
            })
        else:
            # Update workflow status to error
            workflow.status = 'error'
            workflow.last_error = execution_result.get('error', 'Unknown error')
            db.session.commit()
            
            return jsonify({
                'success': False,
                'error': execution_result.get('error', 'Workflow execution failed')
            }), 500
            
    except Exception as e:
        # Update workflow status to error
        try:
            workflow = Workflow.query.get(workflow_id)
            if workflow:
                workflow.status = 'error'
                workflow.last_error = str(e)
                db.session.commit()
        except:
            pass
        
        SystemLog.log_error(
            f'Failed to execute workflow {workflow_id}: {str(e)}',
            category='workflow',
            event_type='execution_error',
            context_data={'workflow_id': workflow_id}
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/workflows/<int:workflow_id>/validate', methods=['POST'])
def validate_workflow(workflow_id):
    """Validate a workflow definition"""
    try:
        data = request.get_json()
        nodes = data.get('nodes', [])
        connections = data.get('connections', [])
        
        validation_result = workflow_service.validate_workflow(nodes, connections)
        
        return jsonify({
            'success': True,
            'valid': validation_result['valid'],
            'errors': validation_result.get('errors', []),
            'warnings': validation_result.get('warnings', [])
        })
        
    except Exception as e:
        SystemLog.log_error(
            f'Failed to validate workflow {workflow_id}: {str(e)}',
            category='workflow',
            event_type='validation_error',
            context_data={'workflow_id': workflow_id}
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/node-types', methods=['GET'])
def get_node_types():
    """Get available node types for the workflow builder"""
    try:
        node_types = workflow_service.get_available_node_types()
        
        return jsonify({
            'success': True,
            'node_types': node_types
        })
        
    except Exception as e:
        SystemLog.log_error(
            f'Failed to fetch node types: {str(e)}',
            category='workflow',
            event_type='node_types_error'
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/workflows/<int:workflow_id>/export', methods=['GET'])
def export_workflow(workflow_id):
    """Export workflow definition"""
    try:
        workflow = Workflow.query.get(workflow_id)
        if not workflow:
            return jsonify({
                'success': False,
                'error': 'Workflow not found'
            }), 404
        
        export_data = {
            'name': workflow.name,
            'description': workflow.description,
            'nodes': workflow.nodes,
            'connections': workflow.connections,
            'created_at': workflow.created_at.isoformat(),
            'updated_at': workflow.updated_at.isoformat()
        }
        
        return jsonify({
            'success': True,
            'workflow_definition': export_data
        })
        
    except Exception as e:
        SystemLog.log_error(
            f'Failed to export workflow {workflow_id}: {str(e)}',
            category='workflow',
            event_type='export_error',
            context_data={'workflow_id': workflow_id}
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@workflow_bp.route('/workflows/import', methods=['POST'])
def import_workflow():
    """Import workflow definition"""
    try:
        data = request.get_json()
        
        if not data or 'workflow_definition' not in data:
            return jsonify({
                'success': False,
                'error': 'Workflow definition is required'
            }), 400
        
        workflow_def = data['workflow_definition']
        
        # Create new workflow from definition
        workflow = Workflow(
            name=workflow_def.get('name', 'Imported Workflow'),
            description=workflow_def.get('description', ''),
            nodes=workflow_def.get('nodes', []),
            connections=workflow_def.get('connections', []),
            status='draft'
        )
        
        db.session.add(workflow)
        db.session.commit()
        
        SystemLog.log_info(
            f'Workflow imported: {workflow.name}',
            category='workflow',
            event_type='import_success',
            context_data={'workflow_id': workflow.id}
        )
        
        return jsonify({
            'success': True,
            'workflow': workflow.to_dict(include_definition=True),
            'message': 'Workflow imported successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        SystemLog.log_error(
            f'Failed to import workflow: {str(e)}',
            category='workflow',
            event_type='import_error'
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

