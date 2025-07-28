"""
System Log model for tracking system events and monitoring
"""

from datetime import datetime, timedelta
from sqlalchemy.dialects.postgresql import JSON
from app import db


class SystemLog(db.Model):
    """System Log model for tracking system events and monitoring"""
    
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Log information
    level = db.Column(db.String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100))  # data, ml, workflow, visualization, system
    
    # Event details
    event_type = db.Column(db.String(100))  # upload, training, execution, error, etc.
    source = db.Column(db.String(100))      # Module or component that generated the log
    
    # Context information
    context_data = db.Column(JSON)  # Additional context data
    stack_trace = db.Column(db.Text)  # Stack trace for errors
    
    # Request information
    request_id = db.Column(db.String(100))  # Request ID for tracing
    ip_address = db.Column(db.String(45))   # Client IP address
    user_agent = db.Column(db.String(500))  # User agent string
    
    # Performance metrics
    execution_time = db.Column(db.Float)    # Execution time in seconds
    memory_usage = db.Column(db.BigInteger) # Memory usage in bytes
    cpu_usage = db.Column(db.Float)         # CPU usage percentage
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Status and resolution
    status = db.Column(db.String(50), default='new')  # new, acknowledged, resolved
    resolved_at = db.Column(db.DateTime)
    resolution_notes = db.Column(db.Text)
    
    def __init__(self, level, message, **kwargs):
        self.level = level.upper()
        self.message = message
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def acknowledge(self):
        """Mark log as acknowledged"""
        self.status = 'acknowledged'
        db.session.commit()
    
    def resolve(self, notes=None):
        """Mark log as resolved"""
        self.status = 'resolved'
        self.resolved_at = datetime.utcnow()
        if notes:
            self.resolution_notes = notes
        db.session.commit()
    
    def is_error(self):
        """Check if this is an error log"""
        return self.level in ['ERROR', 'CRITICAL']
    
    def is_warning(self):
        """Check if this is a warning log"""
        return self.level == 'WARNING'
    
    def get_age_seconds(self):
        """Get log age in seconds"""
        return (datetime.utcnow() - self.timestamp).total_seconds()
    
    def get_age_formatted(self):
        """Get formatted log age"""
        age_seconds = self.get_age_seconds()
        
        if age_seconds < 60:
            return f"{int(age_seconds)} seconds ago"
        elif age_seconds < 3600:
            return f"{int(age_seconds / 60)} minutes ago"
        elif age_seconds < 86400:
            return f"{int(age_seconds / 3600)} hours ago"
        else:
            return f"{int(age_seconds / 86400)} days ago"
    
    @classmethod
    def log_info(cls, message, **kwargs):
        """Create an info log entry"""
        log = cls(level='INFO', message=message, **kwargs)
        db.session.add(log)
        db.session.commit()
        return log
    
    @classmethod
    def log_warning(cls, message, **kwargs):
        """Create a warning log entry"""
        log = cls(level='WARNING', message=message, **kwargs)
        db.session.add(log)
        db.session.commit()
        return log
    
    @classmethod
    def log_error(cls, message, **kwargs):
        """Create an error log entry"""
        log = cls(level='ERROR', message=message, **kwargs)
        db.session.add(log)
        db.session.commit()
        return log
    
    @classmethod
    def log_critical(cls, message, **kwargs):
        """Create a critical log entry"""
        log = cls(level='CRITICAL', message=message, **kwargs)
        db.session.add(log)
        db.session.commit()
        return log
    
    @classmethod
    def get_recent_errors(cls, hours=24):
        """Get recent error logs"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return cls.query.filter(
            cls.level.in_(['ERROR', 'CRITICAL']),
            cls.timestamp >= cutoff_time
        ).order_by(cls.timestamp.desc()).all()
    
    @classmethod
    def get_stats(cls, hours=24):
        """Get log statistics"""
        from sqlalchemy import func
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        stats = db.session.query(
            cls.level,
            func.count(cls.id).label('count')
        ).filter(
            cls.timestamp >= cutoff_time
        ).group_by(cls.level).all()
        
        return {level: count for level, count in stats}
    
    def to_dict(self, include_context=False):
        """Convert log to dictionary"""
        data = {
            'id': self.id,
            'level': self.level,
            'message': self.message,
            'category': self.category,
            'event_type': self.event_type,
            'source': self.source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'status': self.status,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'age_formatted': self.get_age_formatted(),
            'is_error': self.is_error(),
            'is_warning': self.is_warning()
        }
        
        if include_context:
            data.update({
                'context_data': self.context_data,
                'stack_trace': self.stack_trace,
                'request_id': self.request_id,
                'ip_address': self.ip_address,
                'user_agent': self.user_agent,
                'resolution_notes': self.resolution_notes
            })
        
        return data
    
    def __repr__(self):
        return f'<SystemLog {self.level}: {self.message[:50]}>'

