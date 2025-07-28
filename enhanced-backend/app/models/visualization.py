"""
Visualization model for managing charts and dashboards
"""

from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON
from app import db


class Visualization(db.Model):
    """Visualization model for managing charts and dashboards"""
    
    __tablename__ = 'visualizations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    
    # Visualization type and configuration
    chart_type = db.Column(db.String(100), nullable=False)  # bar, line, scatter, pie, heatmap, etc.
    chart_config = db.Column(JSON)  # Chart configuration (Plotly config)
    chart_data = db.Column(JSON)    # Chart data
    
    # Data source
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'))
    dataset = db.relationship('Dataset', backref='visualizations')
    data_query = db.Column(db.Text)  # SQL query or data filter
    
    # Layout and styling
    layout_config = db.Column(JSON)  # Layout configuration
    style_config = db.Column(JSON)   # Styling configuration
    theme = db.Column(db.String(50), default='default')
    
    # Dashboard information
    is_dashboard = db.Column(db.Boolean, default=False)
    dashboard_layout = db.Column(JSON)  # Dashboard grid layout
    widgets = db.Column(JSON)           # Dashboard widgets
    
    # Interactivity
    is_interactive = db.Column(db.Boolean, default=True)
    filters = db.Column(JSON)           # Available filters
    drill_down_config = db.Column(JSON) # Drill-down configuration
    
    # Export and sharing
    export_formats = db.Column(JSON, default=lambda: ['png', 'pdf', 'html'])
    is_public = db.Column(db.Boolean, default=False)
    shared_with = db.Column(JSON, default=list)
    
    # Performance
    cache_enabled = db.Column(db.Boolean, default=True)
    cache_duration = db.Column(db.Integer, default=3600)  # Cache duration in seconds
    last_cached = db.Column(db.DateTime)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_viewed = db.Column(db.DateTime)
    
    # Usage statistics
    view_count = db.Column(db.Integer, default=0)
    export_count = db.Column(db.Integer, default=0)
    
    # Tags and metadata
    tags = db.Column(JSON, default=list)
    category = db.Column(db.String(100))
    
    def __init__(self, name, chart_type, **kwargs):
        self.name = name
        self.chart_type = chart_type
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def increment_view_count(self):
        """Increment view count"""
        self.view_count += 1
        self.last_viewed = datetime.utcnow()
        db.session.commit()
    
    def increment_export_count(self):
        """Increment export count"""
        self.export_count += 1
        db.session.commit()
    
    def update_cache(self):
        """Update cache timestamp"""
        self.last_cached = datetime.utcnow()
        db.session.commit()
    
    def is_cache_valid(self):
        """Check if cache is still valid"""
        if not self.cache_enabled or not self.last_cached:
            return False
        
        cache_age = (datetime.utcnow() - self.last_cached).total_seconds()
        return cache_age < self.cache_duration
    
    def add_tag(self, tag):
        """Add a tag to the visualization"""
        if not self.tags:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            db.session.commit()
    
    def remove_tag(self, tag):
        """Remove a tag from the visualization"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            db.session.commit()
    
    def add_widget(self, widget_config):
        """Add a widget to dashboard"""
        if not self.is_dashboard:
            return False
        
        if not self.widgets:
            self.widgets = []
        
        widget_config['id'] = len(self.widgets) + 1
        self.widgets.append(widget_config)
        db.session.commit()
        return True
    
    def remove_widget(self, widget_id):
        """Remove a widget from dashboard"""
        if not self.is_dashboard or not self.widgets:
            return False
        
        self.widgets = [w for w in self.widgets if w.get('id') != widget_id]
        db.session.commit()
        return True
    
    def get_widget_count(self):
        """Get number of widgets in dashboard"""
        if not self.is_dashboard or not self.widgets:
            return 0
        return len(self.widgets)
    
    def to_dict(self, include_data=False):
        """Convert visualization to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'chart_type': self.chart_type,
            'dataset_id': self.dataset_id,
            'theme': self.theme,
            'is_dashboard': self.is_dashboard,
            'is_interactive': self.is_interactive,
            'is_public': self.is_public,
            'cache_enabled': self.cache_enabled,
            'cache_duration': self.cache_duration,
            'last_cached': self.last_cached.isoformat() if self.last_cached else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_viewed': self.last_viewed.isoformat() if self.last_viewed else None,
            'view_count': self.view_count,
            'export_count': self.export_count,
            'tags': self.tags or [],
            'category': self.category,
            'widget_count': self.get_widget_count(),
            'is_cache_valid': self.is_cache_valid()
        }
        
        if include_data:
            data.update({
                'chart_config': self.chart_config,
                'chart_data': self.chart_data,
                'data_query': self.data_query,
                'layout_config': self.layout_config,
                'style_config': self.style_config,
                'dashboard_layout': self.dashboard_layout,
                'widgets': self.widgets,
                'filters': self.filters,
                'drill_down_config': self.drill_down_config,
                'export_formats': self.export_formats,
                'shared_with': self.shared_with
            })
        
        return data
    
    def __repr__(self):
        return f'<Visualization {self.name} ({self.chart_type})>'

