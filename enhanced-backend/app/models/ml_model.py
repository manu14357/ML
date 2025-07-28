"""
ML Model model for managing machine learning models and their metadata
"""

from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON
from app import db


class MLModel(db.Model):
    """ML Model for managing machine learning models and their metadata"""
    
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    
    # Model information
    algorithm = db.Column(db.String(100), nullable=False)  # linear_regression, random_forest, etc.
    model_type = db.Column(db.String(50), nullable=False)  # regression, classification, clustering
    version = db.Column(db.String(20), default='1.0.0')
    
    # File information
    model_path = db.Column(db.String(500))  # Path to saved model file
    model_size = db.Column(db.BigInteger)   # Model file size in bytes
    
    # Training information
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'))
    dataset = db.relationship('Dataset', backref='trained_models')
    
    # Training configuration
    training_config = db.Column(JSON)  # Hyperparameters, preprocessing steps, etc.
    features = db.Column(JSON)         # List of feature columns used
    target_column = db.Column(db.String(255))  # Target variable for supervised learning
    
    # Training results
    training_metrics = db.Column(JSON)  # Training performance metrics
    validation_metrics = db.Column(JSON)  # Validation performance metrics
    test_metrics = db.Column(JSON)     # Test performance metrics
    
    # Model performance
    accuracy = db.Column(db.Float)     # Primary accuracy metric
    precision = db.Column(db.Float)    # Precision for classification
    recall = db.Column(db.Float)       # Recall for classification
    f1_score = db.Column(db.Float)     # F1 score for classification
    r2_score = db.Column(db.Float)     # R² score for regression
    mse = db.Column(db.Float)          # Mean squared error for regression
    mae = db.Column(db.Float)          # Mean absolute error for regression
    
    # Cross-validation results
    cv_scores = db.Column(JSON)        # Cross-validation scores
    cv_mean = db.Column(db.Float)      # Mean CV score
    cv_std = db.Column(db.Float)       # Standard deviation of CV scores
    
    # Feature importance
    feature_importance = db.Column(JSON)  # Feature importance scores
    
    # Model interpretability
    shap_values = db.Column(JSON)      # SHAP values for model explanation
    permutation_importance = db.Column(JSON)  # Permutation importance
    
    # Training process
    training_time = db.Column(db.Float)  # Training time in seconds
    training_log = db.Column(db.Text)    # Training process log
    
    # Model status
    status = db.Column(db.String(50), default='training')  # training, trained, deployed, failed
    is_deployed = db.Column(db.Boolean, default=False)
    deployment_url = db.Column(db.String(500))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    trained_at = db.Column(db.DateTime)
    deployed_at = db.Column(db.DateTime)
    last_used = db.Column(db.DateTime)
    
    # Relationships
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'))
    dataset = db.relationship('Dataset', backref='trained_models')
    
    # Training configuration
    training_config = db.Column(JSON)  # Hyperparameters, preprocessing steps, etc.
    features = db.Column(JSON)         # List of feature columns used
    target_column = db.Column(db.String(255))  # Target variable for supervised learning
    
    # Training results
    training_metrics = db.Column(JSON)  # Training performance metrics
    validation_metrics = db.Column(JSON)  # Validation performance metrics
    test_metrics = db.Column(JSON)     # Test performance metrics
    
    # Model performance
    accuracy = db.Column(db.Float)     # Primary accuracy metric
    precision = db.Column(db.Float)    # Precision for classification
    recall = db.Column(db.Float)       # Recall for classification
    f1_score = db.Column(db.Float)     # F1 score for classification
    r2_score = db.Column(db.Float)     # R² score for regression
    mse = db.Column(db.Float)          # Mean squared error for regression
    mae = db.Column(db.Float)          # Mean absolute error for regression
    
    # Cross-validation results
    cv_scores = db.Column(JSON)        # Cross-validation scores
    cv_mean = db.Column(db.Float)      # Mean CV score
    cv_std = db.Column(db.Float)       # Standard deviation of CV scores
    
    # Feature importance
    feature_importance = db.Column(JSON)  # Feature importance scores
    
    # Model interpretability
    shap_values = db.Column(JSON)      # SHAP values for model explanation
    permutation_importance = db.Column(JSON)  # Permutation importance
    
    # Training process
    training_time = db.Column(db.Float)  # Training time in seconds
    training_log = db.Column(db.Text)    # Training process log
    
    # Model status
    status = db.Column(db.String(50), default='training')  # training, trained, deployed, failed
    is_deployed = db.Column(db.Boolean, default=False)
    deployment_url = db.Column(db.String(500))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    trained_at = db.Column(db.DateTime)
    deployed_at = db.Column(db.DateTime)
    last_used = db.Column(db.DateTime)
    
    # Tags and metadata
    tags = db.Column(JSON, default=list)
    category = db.Column(db.String(100))
    
    # Usage statistics
    prediction_count = db.Column(db.Integer, default=0)
    download_count = db.Column(db.Integer, default=0)
    
    def __init__(self, name, algorithm, model_type, **kwargs):
        self.name = name
        self.algorithm = algorithm
        self.model_type = model_type
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_training_complete(self, metrics=None):
        """Mark model as training complete"""
        self.status = 'trained'
        self.trained_at = datetime.utcnow()
        if metrics:
            self.training_metrics = metrics
        db.session.commit()
    
    def set_deployed(self, deployment_url=None):
        """Mark model as deployed"""
        self.is_deployed = True
        self.deployed_at = datetime.utcnow()
        if deployment_url:
            self.deployment_url = deployment_url
        db.session.commit()
    
    def increment_prediction_count(self):
        """Increment prediction usage count"""
        self.prediction_count += 1
        self.last_used = datetime.utcnow()
        db.session.commit()
    
    def increment_download_count(self):
        """Increment download count"""
        self.download_count += 1
        db.session.commit()
    
    def add_tag(self, tag):
        """Add a tag to the model"""
        if not self.tags:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            db.session.commit()
    
    def remove_tag(self, tag):
        """Remove a tag from the model"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            db.session.commit()
    
    def get_primary_metric(self):
        """Get the primary performance metric based on model type"""
        if self.model_type == 'classification':
            return self.accuracy or self.f1_score
        elif self.model_type == 'regression':
            return self.r2_score
        return None
    
    def get_model_size_formatted(self):
        """Get formatted model size"""
        if not self.model_size:
            return "Unknown"
        
        size = self.model_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def to_dict(self, include_details=False):
        """Convert model to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'algorithm': self.algorithm,
            'model_type': self.model_type,
            'version': self.version,
            'model_size': self.model_size,
            'model_size_formatted': self.get_model_size_formatted(),
            'dataset_id': self.dataset_id,
            'target_column': self.target_column,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'r2_score': self.r2_score,
            'mse': self.mse,
            'mae': self.mae,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'training_time': self.training_time,
            'status': self.status,
            'is_deployed': self.is_deployed,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'tags': self.tags or [],
            'category': self.category,
            'prediction_count': self.prediction_count,
            'download_count': self.download_count,
            'primary_metric': self.get_primary_metric()
        }
        
        if include_details:
            data.update({
                'training_config': self.training_config,
                'features': self.features,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'test_metrics': self.test_metrics,
                'cv_scores': self.cv_scores,
                'feature_importance': self.feature_importance,
                'shap_values': self.shap_values,
                'permutation_importance': self.permutation_importance,
                'training_log': self.training_log
            })
        
        return data
    
    def __repr__(self):
        return f'<MLModel {self.name} ({self.algorithm})>'

