"""
Dataset model for managing uploaded data files and metadata
"""

from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON
from app import db


class Dataset(db.Model):
    """Dataset model for managing data files and their metadata"""
    
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    
    # File information
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.BigInteger)  # Size in bytes
    file_type = db.Column(db.String(50))  # csv, xlsx, json, etc.
    mime_type = db.Column(db.String(100))
    
    # Data characteristics
    rows_count = db.Column(db.Integer)
    columns_count = db.Column(db.Integer)
    memory_usage = db.Column(db.BigInteger)  # Memory usage in bytes
    
    # Data quality metrics
    missing_values_count = db.Column(db.Integer, default=0)
    duplicate_rows_count = db.Column(db.Integer, default=0)
    data_quality_score = db.Column(db.Float)  # 0-100 score
    
    # Metadata and schema
    columns_info = db.Column(JSON)  # Column names, types, statistics
    data_types = db.Column(JSON)    # Inferred data types
    sample_data = db.Column(JSON)   # Sample rows for preview
    statistics = db.Column(JSON)    # Statistical summary
    
    # EDA (Exploratory Data Analysis) results
    eda_generated = db.Column(db.Boolean, default=False)
    eda_results = db.Column(JSON)   # EDA analysis results
    eda_charts = db.Column(JSON)    # Chart configurations and data
    
    # Processing status
    status = db.Column(db.String(50), default='uploaded')  # uploaded, processing, ready, error
    processing_log = db.Column(db.Text)
    error_message = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = db.Column(db.DateTime)
    
    # Tags and categories
    tags = db.Column(JSON, default=list)
    category = db.Column(db.String(100))
    
    def __init__(self, name, filename, file_path, **kwargs):
        self.name = name
        self.filename = filename
        self.original_filename = filename
        self.file_path = file_path
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def update_last_accessed(self):
        """Update last accessed timestamp"""
        self.last_accessed = datetime.utcnow()
        db.session.commit()
    
    def set_processing_status(self, status, message=None):
        """Update processing status"""
        self.status = status
        if message:
            self.processing_log = message
        self.updated_at = datetime.utcnow()
        db.session.commit()
    
    def set_error(self, error_message):
        """Set error status and message with proper session handling"""
        try:
            # Ensure we have a clean session
            if db.session.is_active:
                db.session.rollback()
                
            self.status = 'error'
            self.error_message = str(error_message)[:1000]  # Truncate to prevent overflow
            self.updated_at = datetime.utcnow()
            
            # Use a new session to ensure we can commit the error
            try:
                db.session.add(self)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                # If we still can't commit, log to stderr at least
                import sys
                print(f"Failed to save error state: {str(e)}", file=sys.stderr)
                print(f"Original error: {error_message}", file=sys.stderr)
        except Exception as e:
            # If something goes really wrong, at least log it
            import sys
            print(f"Critical error in set_error: {str(e)}", file=sys.stderr)
            print(f"Original error: {error_message}", file=sys.stderr)
    
    def add_tag(self, tag):
        """Add a tag to the dataset"""
        if not self.tags:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            db.session.commit()
    
    def remove_tag(self, tag):
        """Remove a tag from the dataset"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            db.session.commit()
    
    def get_size_formatted(self):
        """Get formatted file size"""
        if not self.file_size:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size < 1024.0:
                return f"{self.file_size:.1f} {unit}"
            self.file_size /= 1024.0
        return f"{self.file_size:.1f} TB"
    
    def to_dict(self, include_data=False):
        """Convert dataset to dictionary, ensuring all fields are JSON serializable"""
        import json
        import numpy as np
        from datetime import datetime, date
        
        def safe_serialize(obj):
            """Recursively serialize objects to JSON-serializable types"""
            if obj is None:
                return None
            if isinstance(obj, (str, int, bool)):
                return obj
            if isinstance(obj, float):
                # Handle NaN and infinity
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            if isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8', errors='replace')
                except Exception:
                    return str(obj)
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, (list, tuple)):
                return [safe_serialize(item) for item in obj]
            if isinstance(obj, dict):
                return {str(k): safe_serialize(v) for k, v in obj.items()}
            if isinstance(obj, (np.generic, np.ndarray)):
                return safe_serialize(obj.tolist() if hasattr(obj, 'tolist') else str(obj))
            if hasattr(np, 'floating') and isinstance(obj, np.floating):
                # Handle numpy floating point types
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if hasattr(np, 'integer') and isinstance(obj, np.integer):
                return int(obj)
            if hasattr(np, 'bool_') and isinstance(obj, np.bool_):
                return bool(obj)
            if hasattr(obj, 'isoformat'):  # Handle other date/time objects
                return obj.isoformat()
            if hasattr(obj, '__dict__'):  # Handle objects with __dict__
                return safe_serialize(obj.__dict__)
            return str(obj)
        # Create base data dictionary with all standard fields
        data = {}
        # Get all column names from the model
        columns = [column.name for column in self.__table__.columns]
        
        # Add all model fields to the dictionary
        for column in columns:
            try:
                value = getattr(self, column)
                data[column] = safe_serialize(value)
            except Exception as e:
                data[column] = str(e)  # In case of any serialization error, store the error message
        
        # Add computed fields
        data.update({
            'file_size_formatted': self.get_size_formatted(),
            'tags': safe_serialize(self.tags or []),
            'created_at': safe_serialize(self.created_at),
            'updated_at': safe_serialize(self.updated_at),
            'last_accessed': safe_serialize(self.last_accessed)
        })
        if include_data:
            data.update({
                'columns_info': safe_serialize(self.columns_info),
                'data_types': safe_serialize(self.data_types),
                'sample_data': safe_serialize(self.sample_data),
                'statistics': safe_serialize(self.statistics),
                'eda_results': safe_serialize(self.eda_results),
                'eda_charts': safe_serialize(self.eda_charts)
            })
            
        # Ensure all values are JSON serializable
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            # If there's still a serialization error, convert remaining problematic values to strings
            for key, value in data.items():
                try:
                    json.dumps({key: value})
                except (TypeError, ValueError):
                    data[key] = str(value)
                    
        return data
    
    def __repr__(self):
        return f'<Dataset {self.name}>'

