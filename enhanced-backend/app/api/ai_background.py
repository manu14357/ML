"""
API endpoints for background AI analysis
"""

from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin
import time
import threading
import logging
from app.services.ai_service_advanced import AdvancedAIInsightService

ai_background_bp = Blueprint('ai_background', __name__)

# Global storage for background tasks (in production, use Redis or database)
background_tasks = {}

# Set up logging for background tasks
logger = logging.getLogger(__name__)

@ai_background_bp.route('/start_background_analysis', methods=['POST'])
@cross_origin()
def start_background_analysis():
    """Start background AI analysis for workflow results"""
    try:
        data = request.get_json()
        
        # Generate unique task ID (consistent with AI service format)
        task_id = f"ai_task_{int(time.time() * 1000)}"
        
        # Extract comprehensive data from request
        comprehensive_data = data.get('comprehensive_data', {})
        
        if not comprehensive_data:
            return jsonify({
                'success': False,
                'error': 'No comprehensive data provided'
            }), 400
        
        # Create background task
        background_tasks[task_id] = {
            'status': 'processing',
            'created_at': time.time(),
            'progress': 0,
            'result': None,
            'error': None
        }
        
        # Start background analysis
        def run_analysis():
            try:
                logger.info(f"🚀 Starting background analysis for task: {task_id}")
                ai_service = AdvancedAIInsightService()
                
                # Log the input data for debugging
                logger.info(f"📊 Input data keys: {list(comprehensive_data.keys())}")
                logger.info(f"📊 Input data size: {len(str(comprehensive_data))} characters")
                
                # FORCE STREAMING ENABLED - Always use streaming for background analysis
                logger.info(f"🔄 Forcing streaming mode enabled for background analysis")
                result = ai_service.execute_streaming_analysis(comprehensive_data, task_id)
                
                # Debug the result
                logger.info(f"Background task {task_id} result keys: {list(result.keys())}")
                logger.info(f"Background task {task_id} result: success={result.get('success')}")
                logger.info(f"Background task {task_id} result type: {type(result.get('success'))}")
                
                if not result.get('success'):
                    logger.error(f"AI analysis failed for task {task_id}: {result.get('error')}")
                
                # Update task with result
                success_status = result.get('success', False)
                status = 'completed' if success_status else 'failed'
                logger.info(f"Setting background task {task_id} status to: {status} (success={success_status})")
                
                background_tasks[task_id]['status'] = status
                background_tasks[task_id]['result'] = result
                background_tasks[task_id]['completed_at'] = time.time()
                background_tasks[task_id]['progress'] = 100
                
                # If failed, store the error separately
                if not success_status:
                    background_tasks[task_id]['error'] = result.get('error', 'Unknown error')
                
            except Exception as e:
                logger.error(f"Background task {task_id} failed with exception: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                background_tasks[task_id]['status'] = 'failed'
                background_tasks[task_id]['error'] = str(e)
                background_tasks[task_id]['completed_at'] = time.time()
                background_tasks[task_id]['progress'] = 100
        
        # Start analysis in background thread
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'status': 'processing',
            'message': 'Background AI analysis started'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_background_bp.route('/task_status/<task_id>', methods=['GET'])
@cross_origin()
def get_task_status(task_id):
    """Get the status of a background AI analysis task"""
    try:
        # Log all task status requests to help debug UUID sources
        is_uuid_format = len(task_id.split('-')) == 5 and len(task_id) == 36
        task_format = 'UUID' if is_uuid_format else 'timestamp'
        logger.info(f"Task status requested for: {task_id} (format: {task_format})")
        
        # If this is a UUID-style request, provide helpful error
        if is_uuid_format:
            logger.warning(f"Received UUID-style task ID: {task_id}. This system uses timestamp-based IDs.")
            return jsonify({
                'success': False,
                'error': 'Invalid task ID format. This system uses timestamp-based task IDs (ai_task_<timestamp>), not UUIDs.',
                'task_id': task_id,
                'format_expected': 'ai_task_<timestamp>',
                'format_received': 'UUID',
                'available_tasks': list(background_tasks.keys())[-3:] if background_tasks else []  # Show last 3 tasks
            }), 400
        
        if task_id not in background_tasks:
            logger.warning(f"Task not found: {task_id}")
            return jsonify({
                'success': False,
                'error': 'Task not found',
                'task_id': task_id,
                'available_tasks': list(background_tasks.keys())[-5:] if background_tasks else []  # Show last 5 available tasks
            }), 404
        
        task = background_tasks[task_id]
        
        response = {
            'success': True,
            'task_id': task_id,
            'status': task['status'],
            'progress': task['progress'],
            'created_at': task['created_at']
        }
        
        if task['status'] == 'completed':
            response['result'] = task['result']
            response['completed_at'] = task.get('completed_at')
        elif task['status'] == 'failed':
            response['error'] = task['error']
            response['completed_at'] = task.get('completed_at')
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_background_bp.route('/streaming_analysis/<task_id>', methods=['GET'])
@cross_origin()
def get_streaming_analysis(task_id):
    """Get streaming analysis results for a task"""
    try:
        if task_id not in background_tasks:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
        
        task = background_tasks[task_id]
        
        if task['status'] != 'completed':
            return jsonify({
                'success': False,
                'error': 'Task not completed yet',
                'status': task['status']
            }), 400
        
        result = task.get('result', {})
        
        # Debug the result structure
        logger.info(f"Streaming analysis request for {task_id}: result keys = {list(result.keys())}")
        logger.info(f"Result has full_response: {bool(result.get('full_response'))}")
        logger.info(f"Result has analysis_result: {bool(result.get('analysis_result'))}")
        
        # Get the analysis content - try both old and new field names
        analysis_content = result.get('analysis_result', '') or result.get('full_response', '')
        
        if not analysis_content:
            logger.warning(f"No analysis content found in result for task {task_id}")
            logger.warning(f"Available result keys: {list(result.keys())}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'analysis_result': analysis_content,
            'completion_time': result.get('completion_time'),
            'model_used': result.get('model_used', result.get('metadata', {}).get('model_used', 'Unknown'))
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_background_bp.route('/cleanup_tasks', methods=['POST'])
@cross_origin()
def cleanup_tasks():
    """Clean up old background tasks"""
    try:
        current_time = time.time()
        old_tasks = []
        
        # Remove tasks older than 1 hour
        for task_id, task in background_tasks.items():
            if current_time - task['created_at'] > 3600:  # 1 hour
                old_tasks.append(task_id)
        
        for task_id in old_tasks:
            del background_tasks[task_id]
        
        return jsonify({
            'success': True,
            'cleaned_tasks': len(old_tasks),
            'remaining_tasks': len(background_tasks)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_background_bp.route('/analysis_result/<task_id>', methods=['GET'])
@cross_origin()
def get_analysis_result(task_id):
    """Get the full AI analysis result for a completed task"""
    try:
        if task_id not in background_tasks:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
        
        task = background_tasks[task_id]
        
        if task['status'] != 'completed':
            return jsonify({
                'success': False,
                'error': 'Task not completed yet',
                'status': task['status']
            }), 400
        
        result = task.get('result', {})
        
        # Return the structured AI response
        response = {
            'success': True,
            'task_id': task_id,
            'status': 'completed',
            'completed_at': task.get('completed_at'),
            'analysis': {
                'full_response': result.get('full_response', ''),
                'executive_summary': result.get('executive_summary', ''),
                'business_intelligence': result.get('business_intelligence', ''),
                'strategic_recommendations': result.get('strategic_recommendations', ''),
                'predictive_insights': result.get('predictive_insights', ''),
                'implementation_roadmap': result.get('implementation_roadmap', ''),
                'metadata': result.get('metadata', {}),
                'workflow_analysis': result.get('workflow_analysis', {})
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        current_app.logger.error(f"Error getting analysis result: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_background_bp.route('/debug_tasks', methods=['GET'])
@cross_origin()
def debug_tasks():
    """Debug endpoint to see all stored tasks"""
    try:
        return jsonify({
            'success': True,
            'stored_tasks': list(background_tasks.keys()),
            'total_tasks': len(background_tasks),
            'task_details': {
                task_id: {
                    'status': task.get('status'),
                    'created_at': task.get('created_at'),
                    'has_result': bool(task.get('result')),
                    'error': task.get('error') if task.get('status') == 'failed' else None,
                    'progress': task.get('progress', 0)
                } for task_id, task in background_tasks.items()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
