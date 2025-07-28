"""
Workflow model for managing data processing workflows
"""

from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON
from app import db


class Workflow(db.Model):
    """Workflow model for managing data processing workflows"""
    
    __tablename__ = 'workflows'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    
    # Workflow definition
    workflow_data = db.Column(JSON)  # Complete workflow definition with nodes and connections
    nodes = db.Column(JSON)          # List of workflow nodes
    connections = db.Column(JSON)    # Node connections/edges
    
    # Workflow metadata
    version = db.Column(db.String(20), default='1.0.0')
    category = db.Column(db.String(100))
    tags = db.Column(JSON, default=list)
    
    # Execution information
    status = db.Column(db.String(50), default='draft')  # draft, ready, running, completed, failed, paused
    execution_log = db.Column(db.Text)
    error_message = db.Column(db.Text)
    
    # Execution results
    execution_results = db.Column(JSON)  # Results from workflow execution
    output_datasets = db.Column(JSON)    # Generated datasets
    output_models = db.Column(JSON)      # Generated models
    
    # Performance metrics
    execution_time = db.Column(db.Float)  # Total execution time in seconds
    nodes_executed = db.Column(db.Integer, default=0)
    nodes_failed = db.Column(db.Integer, default=0)
    
    # Scheduling
    is_scheduled = db.Column(db.Boolean, default=False)
    schedule_config = db.Column(JSON)    # Cron expression or interval
    last_run = db.Column(db.DateTime)
    next_run = db.Column(db.DateTime)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_executed = db.Column(db.DateTime)
    
    # Usage statistics
    execution_count = db.Column(db.Integer, default=0)
    success_count = db.Column(db.Integer, default=0)
    failure_count = db.Column(db.Integer, default=0)
    
    def __init__(self, name, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_status(self, status, message=None):
        """Update workflow status"""
        self.status = status
        if message:
            self.execution_log = message
        self.updated_at = datetime.utcnow()
        db.session.commit()
    
    def set_error(self, error_message):
        """Set error status and message"""
        self.status = 'failed'
        self.error_message = error_message
        self.failure_count += 1
        self.updated_at = datetime.utcnow()
        db.session.commit()
    
    def set_completed(self, execution_time=None, results=None):
        """Mark workflow as completed"""
        self.status = 'completed'
        self.last_executed = datetime.utcnow()
        self.success_count += 1
        if execution_time:
            self.execution_time = execution_time
        if results:
            self.execution_results = results
        db.session.commit()
    
    def start_execution(self):
        """Start workflow execution"""
        self.status = 'running'
        self.execution_count += 1
        self.updated_at = datetime.utcnow()
        db.session.commit()
    
    def add_tag(self, tag):
        """Add a tag to the workflow"""
        if not self.tags:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            db.session.commit()
    
    def remove_tag(self, tag):
        """Remove a tag from the workflow"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            db.session.commit()
    
    def get_success_rate(self):
        """Calculate workflow success rate"""
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100
    
    def get_node_count(self):
        """Get total number of nodes in workflow"""
        if not self.nodes:
            return 0
        return len(self.nodes)
    
    def get_connection_count(self):
        """Get total number of connections in workflow"""
        if not self.connections:
            return 0
        return len(self.connections)
    
    def is_valid(self):
        """Check if workflow is valid for execution"""
        return (
            self.nodes and 
            len(self.nodes) > 0 and
            self.status in ['draft', 'ready', 'completed']
        )
    
    def to_dict(self, include_data=False):
        """Convert workflow to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'category': self.category,
            'tags': self.tags or [],
            'status': self.status,
            'execution_time': self.execution_time,
            'nodes_executed': self.nodes_executed,
            'nodes_failed': self.nodes_failed,
            'is_scheduled': self.is_scheduled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_executed': self.last_executed.isoformat() if self.last_executed else None,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.get_success_rate(),
            'node_count': self.get_node_count(),
            'connection_count': self.get_connection_count(),
            'is_valid': self.is_valid()
        }
        
        if include_data:
            data.update({
                'workflow_data': self.workflow_data,
                'nodes': self.nodes,
                'connections': self.connections,
                'execution_results': self.execution_results,
                'output_datasets': self.output_datasets,
                'output_models': self.output_models,
                'schedule_config': self.schedule_config,
                'execution_log': self.execution_log,
                'error_message': self.error_message
            })
        
        return data
    
    def __repr__(self):
        return f'<Workflow {self.name}>'

