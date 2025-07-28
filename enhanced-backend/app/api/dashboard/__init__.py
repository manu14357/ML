"""
Dashboard API endpoints for real-time metrics and overview data
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from sqlalchemy import func
from app.models import Dataset, MLModel, Workflow, Visualization, SystemLog
from app import db

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/overview', methods=['GET'])
def get_overview():
    """Get dashboard overview with key metrics"""
    try:
        # Get basic counts
        total_datasets = Dataset.query.count()
        total_models = MLModel.query.count()
        total_workflows = Workflow.query.count()
        total_visualizations = Visualization.query.count()
        
        # Get status-specific counts
        trained_models = MLModel.query.filter_by(status='trained').count()
        deployed_models = MLModel.query.filter_by(is_deployed=True).count()
        running_workflows = Workflow.query.filter_by(status='running').count()
        completed_workflows = Workflow.query.filter_by(status='completed').count()
        
        # Get recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_datasets = Dataset.query.filter(Dataset.created_at >= week_ago).count()
        recent_models = MLModel.query.filter(MLModel.created_at >= week_ago).count()
        recent_workflows = Workflow.query.filter(Workflow.created_at >= week_ago).count()
        
        # Calculate success rates
        total_workflow_executions = Workflow.query.filter(Workflow.execution_count > 0).count()
        successful_workflows = Workflow.query.filter(Workflow.success_count > 0).count()
        workflow_success_rate = (successful_workflows / total_workflow_executions * 100) if total_workflow_executions > 0 else 0
        
        # Get data quality metrics
        datasets_with_eda = Dataset.query.filter_by(eda_generated=True).count()
        avg_data_quality = db.session.query(func.avg(Dataset.data_quality_score)).filter(
            Dataset.data_quality_score.isnot(None)
        ).scalar() or 0
        
        overview = {
            'totals': {
                'datasets': total_datasets,
                'models': total_models,
                'workflows': total_workflows,
                'visualizations': total_visualizations
            },
            'status': {
                'trained_models': trained_models,
                'deployed_models': deployed_models,
                'running_workflows': running_workflows,
                'completed_workflows': completed_workflows
            },
            'recent_activity': {
                'new_datasets': recent_datasets,
                'new_models': recent_models,
                'new_workflows': recent_workflows
            },
            'quality_metrics': {
                'datasets_with_eda': datasets_with_eda,
                'avg_data_quality': round(avg_data_quality, 2),
                'workflow_success_rate': round(workflow_success_rate, 2)
            }
        }
        
        return jsonify({
            'success': True,
            'overview': overview,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@dashboard_bp.route('/activity', methods=['GET'])
def get_activity():
    """Get recent activity timeline"""
    try:
        days = int(request.args.get('days', 7))
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get recent datasets
        recent_datasets = Dataset.query.filter(
            Dataset.created_at >= cutoff_date
        ).order_by(Dataset.created_at.desc()).limit(10).all()
        
        # Get recent models
        recent_models = MLModel.query.filter(
            MLModel.created_at >= cutoff_date
        ).order_by(MLModel.created_at.desc()).limit(10).all()
        
        # Get recent workflows
        recent_workflows = Workflow.query.filter(
            Workflow.created_at >= cutoff_date
        ).order_by(Workflow.created_at.desc()).limit(10).all()
        
        # Get recent visualizations
        recent_visualizations = Visualization.query.filter(
            Visualization.created_at >= cutoff_date
        ).order_by(Visualization.created_at.desc()).limit(10).all()
        
        # Combine and sort all activities
        activities = []
        
        for dataset in recent_datasets:
            activities.append({
                'type': 'dataset',
                'action': 'created',
                'name': dataset.name,
                'id': dataset.id,
                'timestamp': dataset.created_at.isoformat(),
                'details': f"{dataset.rows_count} rows, {dataset.columns_count} columns"
            })
        
        for model in recent_models:
            activities.append({
                'type': 'model',
                'action': 'created',
                'name': model.name,
                'id': model.id,
                'timestamp': model.created_at.isoformat(),
                'details': f"{model.algorithm} ({model.model_type})"
            })
        
        for workflow in recent_workflows:
            activities.append({
                'type': 'workflow',
                'action': 'created',
                'name': workflow.name,
                'id': workflow.id,
                'timestamp': workflow.created_at.isoformat(),
                'details': f"{workflow.get_node_count()} nodes"
            })
        
        for viz in recent_visualizations:
            activities.append({
                'type': 'visualization',
                'action': 'created',
                'name': viz.name,
                'id': viz.id,
                'timestamp': viz.created_at.isoformat(),
                'details': viz.chart_type
            })
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'activities': activities[:20],  # Return top 20 activities
            'total': len(activities)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@dashboard_bp.route('/performance', methods=['GET'])
def get_performance():
    """Get performance metrics and trends"""
    try:
        # Get model performance metrics
        model_performance = db.session.query(
            MLModel.algorithm,
            func.avg(MLModel.accuracy).label('avg_accuracy'),
            func.avg(MLModel.training_time).label('avg_training_time'),
            func.count(MLModel.id).label('count')
        ).filter(
            MLModel.status == 'trained'
        ).group_by(MLModel.algorithm).all()
        
        # Get workflow performance
        workflow_performance = db.session.query(
            func.avg(Workflow.execution_time).label('avg_execution_time'),
            func.avg(Workflow.success_count / Workflow.execution_count * 100).label('avg_success_rate')
        ).filter(
            Workflow.execution_count > 0
        ).first()
        
        # Get data processing metrics
        data_metrics = db.session.query(
            func.avg(Dataset.rows_count).label('avg_rows'),
            func.avg(Dataset.columns_count).label('avg_columns'),
            func.avg(Dataset.data_quality_score).label('avg_quality')
        ).filter(
            Dataset.rows_count.isnot(None)
        ).first()
        
        # Get system resource usage trends (last 24 hours)
        # This would typically come from a time-series database
        # For now, we'll provide sample data
        resource_trends = {
            'cpu_usage': [45, 52, 48, 55, 42, 38, 44, 50],
            'memory_usage': [65, 68, 72, 70, 66, 63, 67, 71],
            'disk_usage': [78, 78, 79, 79, 80, 80, 81, 81],
            'timestamps': [
                (datetime.utcnow() - timedelta(hours=i)).isoformat()
                for i in range(7, -1, -1)
            ]
        }
        
        performance = {
            'models': [
                {
                    'algorithm': row.algorithm,
                    'avg_accuracy': round(row.avg_accuracy or 0, 3),
                    'avg_training_time': round(row.avg_training_time or 0, 2),
                    'count': row.count
                }
                for row in model_performance
            ],
            'workflows': {
                'avg_execution_time': round(workflow_performance.avg_execution_time or 0, 2),
                'avg_success_rate': round(workflow_performance.avg_success_rate or 0, 2)
            },
            'data': {
                'avg_rows': int(data_metrics.avg_rows or 0),
                'avg_columns': int(data_metrics.avg_columns or 0),
                'avg_quality_score': round(data_metrics.avg_quality or 0, 2)
            },
            'system_trends': resource_trends
        }
        
        return jsonify({
            'success': True,
            'performance': performance,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@dashboard_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Get system alerts and notifications"""
    try:
        # Get recent error logs
        error_logs = SystemLog.query.filter(
            SystemLog.level.in_(['ERROR', 'CRITICAL']),
            SystemLog.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).order_by(SystemLog.timestamp.desc()).limit(10).all()
        
        # Get failed workflows
        failed_workflows = Workflow.query.filter_by(status='failed').limit(5).all()
        
        # Get models with low performance
        low_performance_models = MLModel.query.filter(
            MLModel.accuracy < 0.7,
            MLModel.status == 'trained'
        ).limit(5).all()
        
        # Get datasets with quality issues
        quality_issues = Dataset.query.filter(
            Dataset.data_quality_score < 70
        ).limit(5).all()
        
        alerts = []
        
        # Add error alerts
        for log in error_logs:
            alerts.append({
                'type': 'error',
                'severity': 'high' if log.level == 'CRITICAL' else 'medium',
                'title': f"{log.level}: {log.category}",
                'message': log.message[:100] + '...' if len(log.message) > 100 else log.message,
                'timestamp': log.timestamp.isoformat(),
                'source': log.source
            })
        
        # Add workflow alerts
        for workflow in failed_workflows:
            alerts.append({
                'type': 'workflow',
                'severity': 'medium',
                'title': f"Workflow Failed: {workflow.name}",
                'message': workflow.error_message[:100] if workflow.error_message else 'Workflow execution failed',
                'timestamp': workflow.updated_at.isoformat(),
                'source': 'workflow_engine'
            })
        
        # Add model performance alerts
        for model in low_performance_models:
            alerts.append({
                'type': 'model',
                'severity': 'low',
                'title': f"Low Performance: {model.name}",
                'message': f"Model accuracy is {model.accuracy:.2f}, consider retraining",
                'timestamp': model.trained_at.isoformat() if model.trained_at else model.created_at.isoformat(),
                'source': 'ml_manager'
            })
        
        # Add data quality alerts
        for dataset in quality_issues:
            alerts.append({
                'type': 'data',
                'severity': 'low',
                'title': f"Data Quality Issue: {dataset.name}",
                'message': f"Quality score is {dataset.data_quality_score:.1f}%",
                'timestamp': dataset.updated_at.isoformat(),
                'source': 'data_processor'
            })
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'alerts': alerts[:15],  # Return top 15 alerts
            'summary': {
                'total': len(alerts),
                'high_severity': len([a for a in alerts if a['severity'] == 'high']),
                'medium_severity': len([a for a in alerts if a['severity'] == 'medium']),
                'low_severity': len([a for a in alerts if a['severity'] == 'low'])
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

