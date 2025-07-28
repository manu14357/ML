"""
SuperHacker Enhanced Backend Application Factory
"""

import os
import logging
import json
import numpy as np
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_socketio import SocketIO
from config.config import config


class NaNSafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN, infinity, and other numpy types"""
    
    def default(self, obj):
        import numpy as np
        from datetime import datetime, date, time
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        
        # Handle regular Python float NaN/inf
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        
        return super().default(obj)

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
cors = CORS()
socketio = SocketIO()


def create_app(config_name=None):
    """Create and configure the Flask application"""
    
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Set custom JSON encoder to handle NaN values
    app.json_encoder = NaNSafeJSONEncoder
    
    # Initialize configuration
    config[config_name].init_app(app)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app, origins=app.config['CORS_ORIGINS'])
    socketio.init_app(app, 
                     cors_allowed_origins="*",
                     async_mode=app.config['SOCKETIO_ASYNC_MODE'])
    
    # Configure logging
    configure_logging(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register shell context
    register_shell_context(app)
    
    # Initialize services
    initialize_services(app)
    
    return app


def configure_logging(app):
    """Configure application logging"""
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = logging.FileHandler(app.config['LOG_FILE'])
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('SuperHacker Enhanced Backend startup')


def register_blueprints(app):
    """Register application blueprints"""
    
    # Import blueprints
    from app.api.dashboard import dashboard_bp
    from app.api.data import data_bp
    from app.api.ml import ml_bp
    # from app.api.workflow import workflow_bp  # Temporarily disabled - using new advanced workflows
    from app.api.visualization import visualization_bp
    from app.api.system import system_bp
    from app.api.ai import ai_bp
    from app.api.ai_background import ai_background_bp
    
    # Import the new advanced workflow blueprint
    from app.api.workflow.workflows import bp as workflows_bp
    
    # Register blueprints with URL prefixes
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    app.register_blueprint(data_bp, url_prefix='/api/data')
    app.register_blueprint(ml_bp, url_prefix='/api/ml')
    # app.register_blueprint(workflow_bp, url_prefix='/api/workflow')  # Temporarily disabled
    app.register_blueprint(workflows_bp, url_prefix='/api/workflows')  # New advanced workflows
    app.register_blueprint(visualization_bp, url_prefix='/api/visualization')
    app.register_blueprint(system_bp, url_prefix='/api/system')
    app.register_blueprint(ai_bp, url_prefix='/api/ai')
    app.register_blueprint(ai_background_bp, url_prefix='/api/ai_background')  # Background AI analysis


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        return {'error': 'Bad request', 'message': str(error)}, 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return {'error': 'Unauthorized', 'message': 'Authentication required'}, 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return {'error': 'Forbidden', 'message': 'Insufficient permissions'}, 403
    
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found', 'message': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return {'error': 'Internal server error', 'message': 'An unexpected error occurred'}, 500


def register_shell_context(app):
    """Register shell context for flask shell command"""
    
    @app.shell_context_processor
    def make_shell_context():
        from app.models import Dataset, MLModel, Workflow, Visualization, SystemLog
        return {
            'db': db,
            'Dataset': Dataset,
            'MLModel': MLModel,
            'Workflow': Workflow,
            'Visualization': Visualization,
            'SystemLog': SystemLog
        }


def initialize_services(app):
    """Initialize application services"""
    
    with app.app_context():
        # Initialize data service
        from app.services.data_service import DataService
        from app.services.eda_service import EDAService
        
        # Store services in app config for access
        app.config['DATA_SERVICE'] = DataService()
        app.config['EDA_SERVICE'] = EDAService()
        
        # TODO: Initialize other services when implemented
        from app.services.ml_service import MLModelService, AutoMLService
        # from app.services.workflow_service import WorkflowService
        # from app.services.visualization_service import VisualizationService
        
        app.config['ML_MODEL_SERVICE'] = MLModelService()
        app.config['AUTOML_SERVICE'] = AutoMLService()
        # app.config['WORKFLOW_SERVICE'] = WorkflowService()
        # app.config['VISUALIZATION_SERVICE'] = VisualizationService()
        
        app.logger.info('Services initialized successfully')