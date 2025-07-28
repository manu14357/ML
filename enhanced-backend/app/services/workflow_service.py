"""
Advanced Workflow Service for executing and managing complex data science workflows
"""

import pandas as pd
import numpy as np
import json
import time
import os
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Generator
from datetime import datetime, time as dt_time
import traceback
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, adjusted_rand_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

from app.models import Dataset, Workflow
from app import db
from app.services.ai_service_advanced import AdvancedAIInsightService
from app.services.eda_service import EDAService
from app.services.univariate_anomaly_service import UnivariateAnomalyDetectionService
from app.services.multivariate_anomaly_service import MultivariateAnomalyDetectionService
from app.services.event_detection_service import EventDetectionService

# Import background tasks storage for AI streaming
try:
    from app.api.ai_background import background_tasks
except ImportError as e:
    # Fallback if import fails
    background_tasks = {}


class AdvancedWorkflowService:
    """Advanced service for workflow execution and management with robust error handling"""
    
    def __init__(self):
        self.data_cache = {}  # Cache for storing data between nodes
        self.model_cache = {}  # Cache for storing trained models
        self.execution_logs = []  # Logs for debugging
        self.chart_cache = {}  # Cache for generated charts
        
        # NEW: Add memory management
        self.max_cache_size = 100  # Maximum number of cached items
        self.memory_threshold = 1024 * 1024 * 1024  # 1GB memory threshold
        
        # Initialize specialized services
        self.eda_service = EDAService()
        self.univariate_anomaly_service = UnivariateAnomalyDetectionService()
        self.multivariate_anomaly_service = MultivariateAnomalyDetectionService()
        self.event_detection_service = EventDetectionService()
        
        # Enhanced node processors with comprehensive data science capabilities
        self.node_processors = self._initialize_node_processors()
        
        # Background processing for AI insights
        self.background_tasks = {}
        self.ai_service = AdvancedAIInsightService()
        self.streaming_results = {}
        self.task_queue = queue.Queue()

    def _cleanup_memory_cache(self):
            """Clean up memory cache when it gets too large"""
            try:
                # Clean up data cache if it gets too large
                if len(self.data_cache) > self.max_cache_size:
                    # Remove oldest entries (simple FIFO)
                    items_to_remove = len(self.data_cache) - self.max_cache_size + 10
                    keys_to_remove = list(self.data_cache.keys())[:items_to_remove]
                    for key in keys_to_remove:
                        del self.data_cache[key]
                    self._log_info(f"Cleaned up {items_to_remove} items from data cache")
                
                # Similar cleanup for other caches
                if len(self.chart_cache) > self.max_cache_size:
                    items_to_remove = len(self.chart_cache) - self.max_cache_size + 10
                    keys_to_remove = list(self.chart_cache.keys())[:items_to_remove]
                    for key in keys_to_remove:
                        del self.chart_cache[key]
                    self._log_info(f"Cleaned up {items_to_remove} items from chart cache")
                    
            except Exception as e:
                self._log_warning(f"Error during memory cleanup: {str(e)}")
    
        # ...existing code...

    def convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types for JSON serialization"""
        # Special handling for arrays, DataFrames, and Series first
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # Handle pandas DataFrames specially - convert to records
            try:
                return self.convert_numpy_types(obj.to_dict('records'))
            except:
                # If that fails, convert to string representation
                return str(obj)
        elif isinstance(obj, pd.Series):
            # Handle pandas Series specially
            return self.convert_numpy_types(obj.to_dict())
            
        # Then handle None and NaN values for scalar types
        if obj is None:
            return None
        try:
            # Use pandas isna only on scalar values
            if pd.isna(obj) or obj is pd.NaT:
                return None
        except (TypeError, ValueError):
            # If pd.isna fails on an object type, continue processing
            pass
            
        # Handle different data types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()  # Convert Timestamp to ISO format string
        elif isinstance(obj, datetime):
            return obj.isoformat()  # Also handle Python datetime objects
        elif isinstance(obj, dt_time):
            return obj.isoformat()  # Handle time objects
        elif hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat')):
            # Handle any other datetime-like objects with isoformat method
            return obj.isoformat()
        elif hasattr(obj, 'item'):
            # Handle numpy scalars
            return self.convert_numpy_types(obj.item())
        elif isinstance(obj, (complex, np.number)):
            # Handle other numeric types
            try:
                return float(obj)
            except:
                return str(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        elif hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat')):
            # Handle any other datetime-like objects with isoformat method
            return obj.isoformat()
        elif hasattr(obj, 'item'):
            # Handle numpy scalars
            return self.convert_numpy_types(obj.item())
        elif isinstance(obj, (complex, np.number)):
            # Handle other numeric types
            try:
                return float(obj)
            except:
                return str(obj)
        else:
            # Try to return the object as is, or convert to string if necessary
            try:
                # Try json serialization as a test
                json.dumps(obj)
                return obj
            except:
                # If it can't be serialized, convert to string
                return str(obj)

    def _initialize_node_processors(self):
        """Initialize the node processors mapping"""
        return {
            # 1. INPUT - Data Sources
            'data_source': self._process_data_source,
            'api_source': self._process_api_source,
            'database_source': self._process_database_source,
            
            # 2. PREPROCESSING - Data Cleaning & Preparation
            'data_cleaning': self._process_data_cleaning,
            'feature_engineering': self._process_feature_engineering,
            'data_validation': self._process_data_validation,
            
            # 3. DATA VISUALIZATION - Charts & Plots
            'basic_plots': self._process_basic_plots,
            'advanced_plots': self._process_advanced_plots,
            'dashboard': self._process_dashboard,
            
            # 4. ANALYSIS - Statistical Analysis & EDA
            'descriptive_stats': self._process_descriptive_stats,
            'correlation_analysis': self._process_correlation_analysis,
            'hypothesis_testing': self._process_hypothesis_testing,
            'eda_analysis': self._process_eda_analysis,
            'univariate_anomaly_detection': self._process_univariate_anomaly_detection,
            'multivariate_anomaly_detection': self._process_multivariate_anomaly_detection,
            'event_detection': self._process_event_detection,
            'data_insights': self._process_data_insights,
            
            # 5. ML SUPERVISED - Supervised Learning Models
            'classification': self._process_classification,
            'regression': self._process_regression,
            'time_series': self._process_time_series,
            'neural_network': self._process_neural_network,
            'cnn': self._process_cnn,
            'rnn_lstm': self._process_rnn_lstm,
            'model_evaluation': self._process_model_evaluation,
            'feature_importance': self._process_feature_importance,
            'model_comparison': self._process_model_comparison,
            'auto_ml': self._process_auto_ml,
            
            # 6. ML UNSUPERVISED - Unsupervised Learning Models
            'clustering': self._process_clustering,
            'dimensionality_reduction': self._process_dimensionality_reduction,
            'anomaly_detection': self._process_anomaly_detection,
            
            # 7. EXPORT DATA - Output & Model Deployment
            'ai_summary': self._process_ai_summary,
            'export_data': self._process_export_data,
            'save_model': self._process_save_model,
            'deploy_model': self._process_deploy_model
        }
    
    def get_node_types(self) -> Dict[str, Any]:
        """Get comprehensive node type definitions with parameters and metadata"""
        return {
            # Input/Data Sources
            'data_source': {
                'id': 'data_source',
                'name': 'Data Source',
                'category': 'input',
                'description': 'Load data from uploaded datasets with sampling options.',
                'icon': 'Database',
                'color': '#3B82F6',
                'inputs': [],
                'outputs': ['data'],
                'parameters': [
                    {'name': 'dataset_id', 'type': 'dataset_select', 'required': True, 'description': 'Select dataset to load'},
                    {'name': 'sample_size', 'type': 'number', 'required': False, 'description': 'Number of rows to sample (optional)', 'min': 1},
                    {'name': 'random_seed', 'type': 'number', 'required': False, 'description': 'Random seed for sampling', 'default': 42},
                    {'name': 'columns', 'type': 'multiselect', 'required': False, 'description': 'Select specific columns (optional)'}
                ]
            },
            'api_source': {
                'id': 'api_source',
                'name': 'API Source',
                'category': 'input',
                'description': 'Fetch data from REST APIs with authentication.',
                'icon': 'RefreshCw',
                'color': '#6366F1',
                'inputs': [],
                'outputs': ['data'],
                'parameters': [
                    {'name': 'url', 'type': 'text', 'required': True, 'description': 'API endpoint URL'},
                    {'name': 'method', 'type': 'select', 'options': ['GET', 'POST'], 'default': 'GET', 'description': 'HTTP method'},
                    {'name': 'headers', 'type': 'textarea', 'required': False, 'description': 'Headers (JSON format)', 'placeholder': '{"Authorization": "Bearer token"}'},
                    {'name': 'params', 'type': 'textarea', 'required': False, 'description': 'Query parameters (JSON format)'},
                    {'name': 'timeout', 'type': 'number', 'default': 30, 'description': 'Request timeout in seconds'}
                ]
            },
            
            # Data Preprocessing
            'data_cleaning': {
                'id': 'data_cleaning',
                'name': 'Data Cleaning',
                'category': 'preprocessing',
                'description': 'Clean data by handling missing values and outliers.',
                'icon': 'Filter',
                'color': '#10B981',
                'inputs': ['data'],
                'outputs': ['data'],
                'parameters': [
                    {'name': 'missing_strategy', 'type': 'select', 'options': ['drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_zero', 'forward_fill', 'backward_fill'], 'default': 'drop', 'description': 'Missing value strategy'},
                    {'name': 'missing_threshold', 'type': 'number', 'default': 0.5, 'min': 0, 'max': 1, 'description': 'Drop columns with missing % above threshold'},
                    {'name': 'outlier_method', 'type': 'select', 'options': ['none', 'iqr', 'zscore', 'isolation_forest'], 'default': 'none', 'description': 'Outlier detection method'},
                    {'name': 'outlier_threshold', 'type': 'number', 'default': 3, 'description': 'Threshold for outlier detection'},
                    {'name': 'duplicate_action', 'type': 'select', 'options': ['keep_first', 'keep_last', 'remove_all'], 'default': 'keep_first', 'description': 'Duplicate handling strategy'}
                ]
            },
            'feature_engineering': {
                'id': 'feature_engineering',
                'name': 'Feature Engineering',
                'category': 'preprocessing',
                'description': 'Create and transform features for improved model performance.',
                'icon': 'Zap',
                'color': '#F59E0B',
                'inputs': ['data'],
                'outputs': ['data'],
                'parameters': [
                    {'name': 'scaling_type', 'type': 'select', 'options': ['none', 'standard', 'minmax', 'robust', 'normalizer'], 'default': 'none', 'description': 'Scaling method'},
                    {'name': 'encoding_type', 'type': 'select', 'options': ['none', 'label', 'one_hot', 'target', 'binary'], 'default': 'none', 'description': 'Categorical encoding'},
                    {'name': 'polynomial_features', 'type': 'boolean', 'default': False, 'description': 'Create polynomial features'},
                    {'name': 'interaction_features', 'type': 'boolean', 'default': False, 'description': 'Create interaction features'},
                    {'name': 'feature_selection', 'type': 'select', 'options': ['none', 'variance', 'correlation', 'mutual_info', 'chi2'], 'default': 'none', 'description': 'Feature selection method'},
                    {'name': 'n_features', 'type': 'number', 'required': False, 'description': 'Number of features to select'}
                ]
            },
            
            # Analysis
            'descriptive_stats': {
                'id': 'descriptive_stats',
                'name': 'Descriptive Statistics',
                'category': 'analysis',
                'description': 'Generates detailed statistics, distributions, and correlations for data analysis.',
                'icon': 'BarChart3',
                'color': '#8B5CF6',
                'inputs': ['data'],
                'outputs': ['statistics', 'charts'],
                'parameters': [
                    {'name': 'include_correlations', 'type': 'boolean', 'default': True, 'description': 'Include correlation analysis for feature relationship insights'},
                    {'name': 'correlation_method', 'type': 'select', 'options': ['pearson', 'spearman', 'kendall'], 'default': 'pearson', 'description': 'Correlation method'},
                    {'name': 'group_by', 'type': 'text', 'required': False, 'description': 'Group by column name for segmented analysis'},
                    {'name': 'percentiles', 'type': 'text', 'default': '25,50,75,90,95,99', 'description': 'Percentiles to calculate (comma-separated)'},
                    {'name': 'generate_plots', 'type': 'boolean', 'default': True, 'description': 'Generate distribution plots'}
                ]
            },
            'ai_summary': {
                'id': 'ai_summary',
                'name': 'AI Data Summary',
                'category': 'analysis',
                'description': 'AI-powered analysis that synthesizes insights from all connected nodes.',
                'icon': 'FileText',
                'color': '#EC4899',
                'inputs': ['data'],
                'outputs': ['insights', 'recommendations'],
                'parameters': [
                    {'name': 'analysis_depth', 'type': 'select', 'options': ['basic', 'detailed', 'comprehensive'], 'default': 'comprehensive', 'description': 'Analysis depth level - comprehensive recommended for multi-node workflows'},
                    {'name': 'include_recommendations', 'type': 'boolean', 'default': True, 'description': 'Include advanced ML model and business recommendations'},
                    {'name': 'business_context', 'type': 'textarea', 'required': False, 'description': 'Business context for domain-specific insights and recommendations'},
                    {'name': 'target_variable', 'type': 'text', 'required': False, 'description': 'Target variable for prediction-focused analysis'},
                    {'name': 'focus_areas', 'type': 'multi_select', 'options': ['data_quality', 'statistical_patterns', 'business_insights', 'technical_recommendations', 'predictive_modeling'], 'default': ['data_quality', 'business_insights'], 'description': 'Specific focus areas for AI analysis'}
                ]
            },
            
            # Machine Learning
            'classification': {
                'id': 'classification',
                'name': 'Classification',
                'category': 'ml_supervised',
                'description': 'Classification modeling with multiple algorithms for categorical predictions.',
                'icon': 'Target',
                'color': '#EF4444',
                'inputs': ['data'],
                'outputs': ['model', 'predictions', 'metrics'],
                'parameters': [
                    {'name': 'algorithm', 'type': 'select', 'options': ['random_forest', 'logistic_regression', 'svm', 'gradient_boosting', 'naive_bayes'], 'default': 'random_forest', 'description': 'Classification algorithm for predictive modeling'},
                    {'name': 'target_column', 'type': 'column_select', 'required': True, 'description': 'Target column for classification', 'filter': 'target_suitable'},
                    {'name': 'feature_columns', 'type': 'multi_column_select', 'required': False, 'description': 'Feature columns (leave empty for auto-selection)', 'filter': 'feature_suitable'},
                    {'name': 'test_size', 'type': 'number', 'default': 0.2, 'min': 0.1, 'max': 0.5, 'description': 'Test set size ratio'},
                    {'name': 'cross_validation', 'type': 'boolean', 'default': True, 'description': 'Use cross-validation for robust performance evaluation'},
                    {'name': 'cv_folds', 'type': 'number', 'default': 5, 'min': 3, 'max': 10, 'description': 'Number of CV folds'},
                    {'name': 'feature_importance', 'type': 'boolean', 'default': True, 'description': 'Calculate feature importance for AI analysis'}
                ]
            },
            'regression': {
                'id': 'regression',
                'name': 'Regression',
                'category': 'ml_supervised',
                'description': 'Regression modeling for continuous value prediction and trend analysis.',
                'icon': 'GitBranch',
                'color': '#F97316',
                'inputs': ['data'],
                'outputs': ['model', 'predictions', 'metrics'],
                'parameters': [
                    {'name': 'algorithm', 'type': 'select', 'options': ['linear', 'random_forest', 'gradient_boosting', 'polynomial', 'ridge', 'lasso'], 'default': 'random_forest', 'description': 'Regression algorithm'},
                    {'name': 'target_column', 'type': 'column_select', 'required': True, 'description': 'Target column for regression', 'filter': 'target_suitable'},
                    {'name': 'feature_columns', 'type': 'multi_column_select', 'required': False, 'description': 'Feature columns (leave empty for auto-selection)', 'filter': 'feature_suitable'},
                    {'name': 'test_size', 'type': 'number', 'default': 0.2, 'min': 0.1, 'max': 0.5, 'description': 'Test set size ratio'},
                    {'name': 'polynomial_degree', 'type': 'number', 'default': 2, 'min': 1, 'max': 5, 'description': 'Polynomial degree (for polynomial regression)'},
                    {'name': 'alpha', 'type': 'number', 'default': 1.0, 'description': 'Regularization strength (for Ridge/Lasso)'}
                ]
            },
            'clustering': {
                'id': 'clustering',
                'name': 'Clustering',
                'category': 'ml_unsupervised',
                'description': 'Unsupervised clustering for segmentation and pattern discovery.',
                'icon': 'GitBranch',
                'color': '#06B6D4',
                'inputs': ['data'],
                'outputs': ['clusters', 'model', 'metrics'],
                'parameters': [
                    {'name': 'algorithm', 'type': 'select', 'options': ['kmeans', 'dbscan', 'hierarchical', 'gaussian_mixture'], 'default': 'kmeans', 'description': 'Clustering algorithm for pattern discovery'},
                    {'name': 'feature_columns', 'type': 'multi_column_select', 'required': False, 'description': 'Feature columns for clustering (leave empty for auto-selection)', 'filter': 'feature_suitable'},
                    {'name': 'n_clusters', 'type': 'number', 'default': 3, 'min': 2, 'max': 20, 'description': 'Number of clusters (for k-means)'},
                    {'name': 'scale_features', 'type': 'boolean', 'default': True, 'description': 'Scale features before clustering for optimal results'},
                    {'name': 'eps', 'type': 'number', 'default': 0.5, 'description': 'Epsilon parameter (for DBSCAN)'},
                    {'name': 'min_samples', 'type': 'number', 'default': 5, 'description': 'Minimum samples (for DBSCAN)'}
                ]
            },
            
            # Visualization
            'basic_plots': {
                'id': 'basic_plots',
                'name': 'Basic Plots',
                'category': 'visualization',
                'description': 'Essential statistical visualizations for data pattern analysis.',
                'icon': 'BarChart3',
                'color': '#84CC16',
                'inputs': ['data'],
                'outputs': ['charts'],
                'parameters': [
                    {'name': 'plot_type', 'type': 'select', 'options': ['histogram', 'scatter', 'boxplot', 'barplot', 'lineplot'], 'default': 'histogram', 'description': 'Plot type for data exploration'},
                    {'name': 'x_column', 'type': 'column_select', 'required': True, 'description': 'X-axis column', 'filter': 'all'},
                    {'name': 'y_column', 'type': 'column_select', 'required': False, 'description': 'Y-axis column (for relationship analysis)', 'filter': 'numeric'},
                    {'name': 'color_by', 'type': 'column_select', 'required': False, 'description': 'Color by column for segmentation insights', 'filter': 'categorical'},
                    {'name': 'bins', 'type': 'number', 'default': 30, 'description': 'Number of bins (for histogram)'},
                    {'name': 'figsize_width', 'type': 'number', 'default': 10, 'description': 'Figure width'},
                    {'name': 'figsize_height', 'type': 'number', 'default': 6, 'description': 'Figure height'},
                    {'name': 'auto_detect_columns', 'type': 'boolean', 'default': True, 'description': 'Automatically detect suitable columns for visualization'}
                ]
            },
            'advanced_plots': {
                'id': 'advanced_plots',
                'name': 'Advanced Plots',
                'category': 'visualization',
                'description': 'Sophisticated visualizations for complex data relationships.',
                'icon': 'Eye',
                'color': '#A855F7',
                'inputs': ['data'],
                'outputs': ['charts'],
                'parameters': [
                    {'name': 'plot_type', 'type': 'select', 'options': ['heatmap', 'pairplot', 'parallel_coordinates', '3d_scatter', 'violin_plot'], 'default': 'heatmap', 'description': 'Advanced plot type'},
                    {'name': 'features', 'type': 'multiselect_columns', 'required': False, 'description': 'Feature columns (leave empty for automatic selection)', 'filter': 'numeric'},
                    {'name': 'interactive', 'type': 'boolean', 'default': True, 'description': 'Make plot interactive with Plotly'},
                    {'name': 'color_palette', 'type': 'select', 'options': ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm'], 'default': 'viridis', 'description': 'Color palette'},
                    {'name': 'auto_detect_features', 'type': 'boolean', 'default': True, 'description': 'Automatically detect numeric features'}
                ]
            },
            
            # Output
            'export_data': {
                'id': 'export_data',
                'name': 'Export Data',
                'category': 'output',
                'description': 'Export processed data in various file formats.',
                'icon': 'Download',
                'color': '#6B7280',
                'inputs': ['data'],
                'outputs': ['file'],
                'parameters': [
                    {'name': 'format', 'type': 'select', 'options': ['csv', 'json', 'excel', 'parquet', 'pdf'], 'default': 'csv', 'description': 'Export format'},
                    {'name': 'filename', 'type': 'text', 'required': True, 'description': 'Output filename (without extension)', 'default': 'export_data'},
                    {'name': 'include_index', 'type': 'boolean', 'default': False, 'description': 'Include row index in export (not applicable for PDF)'},
                    {'name': 'compress', 'type': 'boolean', 'default': False, 'description': 'Compress output file (CSV/JSON only)'}
                ]
            },
            
            # Enhanced Analysis Nodes
            'eda_analysis': {
                'id': 'eda_analysis',
                'name': 'EDA Analysis',
                'category': 'analysis',
                'description': 'Exploratory Data Analysis with visualizations and statistics.',
                'icon': 'Search',
                'color': '#8B5CF6',
                'inputs': ['data'],
                'outputs': ['analysis', 'charts'],
                'parameters': [
                    {'name': 'analysis_depth', 'type': 'select', 'options': ['basic', 'detailed', 'comprehensive'], 'default': 'detailed', 'description': 'Analysis depth level'},
                    {'name': 'include_distributions', 'type': 'boolean', 'default': True, 'description': 'Include distribution analysis'},
                    {'name': 'include_correlations', 'type': 'boolean', 'default': True, 'description': 'Include correlation analysis'},
                    {'name': 'max_categorical_levels', 'type': 'number', 'default': 20, 'min': 5, 'max': 50, 'description': 'Maximum categorical levels to display'},
                    {'name': 'generate_recommendations', 'type': 'boolean', 'default': True, 'description': 'Generate data quality recommendations'}
                ]
            },
            'univariate_anomaly_detection': {
                'id': 'univariate_anomaly_detection',
                'name': 'Univariate Anomaly Detection',
                'category': 'analysis',
                'description': 'Detect anomalies in individual variables using statistical methods.',
                'icon': 'AlertTriangle',
                'color': '#F59E0B',
                'inputs': ['data'],
                'outputs': ['anomalies', 'charts'],
                'parameters': [
                    {'name': 'method', 'type': 'select', 'options': ['all', 'zscore', 'iqr', 'isolation_forest', 'modified_zscore'], 'default': 'all', 'description': 'Anomaly detection method'},
                    {'name': 'contamination', 'type': 'number', 'default': 0.1, 'min': 0.01, 'max': 0.5, 'description': 'Expected contamination rate (for Isolation Forest)'},
                    {'name': 'feature_columns', 'type': 'multi_column_select', 'required': False, 'description': 'Feature columns (leave empty for all numeric)', 'filter': 'numeric'},
                    {'name': 'zscore_threshold', 'type': 'number', 'default': 3.0, 'min': 1.0, 'max': 5.0, 'description': 'Z-score threshold for outlier detection'},
                    {'name': 'iqr_multiplier', 'type': 'number', 'default': 1.5, 'min': 1.0, 'max': 3.0, 'description': 'IQR multiplier for outlier detection'}
                ]
            },
            'multivariate_anomaly_detection': {
                'id': 'multivariate_anomaly_detection',
                'name': 'Multivariate Anomaly Detection',
                'category': 'analysis',
                'description': 'Detect anomalies across multiple variables simultaneously.',
                'icon': 'Activity',
                'color': '#EF4444',
                'inputs': ['data'],
                'outputs': ['anomalies', 'charts'],
                'parameters': [
                    {'name': 'method', 'type': 'select', 'options': ['all', 'isolation_forest', 'elliptic_envelope', 'local_outlier_factor', 'mahalanobis', 'pca_reconstruction'], 'default': 'all', 'description': 'Anomaly detection method'},
                    {'name': 'contamination', 'type': 'number', 'default': 0.1, 'min': 0.01, 'max': 0.5, 'description': 'Expected contamination rate'},
                    {'name': 'feature_columns', 'type': 'multi_column_select', 'required': False, 'description': 'Feature columns (leave empty for all numeric)', 'filter': 'numeric'},
                    {'name': 'scale_features', 'type': 'boolean', 'default': True, 'description': 'Scale features before anomaly detection'},
                    {'name': 'n_components', 'type': 'number', 'default': 2, 'min': 2, 'max': 10, 'description': 'Number of PCA components (for PCA reconstruction method)'}
                ]
            },
            'event_detection': {
                'id': 'event_detection',
                'name': 'Event Detection',
                'category': 'analysis',
                'description': 'Detect time series events like spikes, drifts, and gaps.',
                'icon': 'Zap',
                'color': '#8B5CF6',
                'inputs': ['data'],
                'outputs': ['events', 'charts'],
                'parameters': [
                    {'name': 'method', 'type': 'select', 'options': ['all', 'spikes', 'drifts', 'gaps', 'flatlines'], 'default': 'all', 'description': 'Event detection method'},
                    {'name': 'feature_columns', 'type': 'multi_column_select', 'required': False, 'description': 'Feature columns (leave empty for all numeric)', 'filter': 'numeric'},
                    {'name': 'spike_threshold', 'type': 'number', 'default': 3.0, 'min': 1.0, 'max': 5.0, 'description': 'Spike detection threshold (std deviations)'},
                    {'name': 'drift_window', 'type': 'number', 'default': 50, 'min': 10, 'max': 200, 'description': 'Window size for drift detection'},
                    {'name': 'flatline_threshold', 'type': 'number', 'default': 0.001, 'min': 0.0001, 'max': 0.1, 'description': 'Variance threshold for flatline detection'},
                    {'name': 'gap_threshold', 'type': 'number', 'default': 10, 'min': 1, 'max': 100, 'description': 'Minimum gap size to detect'}
                ]
            }
        }
    
    def get_available_datasets(self) -> List[Dict]:
        """Get list of all available datasets with comprehensive metadata"""
        try:
            datasets = Dataset.query.filter_by(status='ready').all()
            return [{
                'id': d.id,
                'name': d.name,
                'filename': d.filename,
                'file_path': d.file_path,
                'rows_count': d.rows_count,
                'columns_count': d.columns_count,
                'file_type': d.file_type,
                'description': d.description,
                'data_quality_score': d.data_quality_score,
                'file_size': d.file_size,
                'columns_info': d.columns_info,
                'sample_data': d.sample_data,
                'statistics': d.statistics,
                'created_at': d.created_at.isoformat() if d.created_at else None
            } for d in datasets]
        except Exception as e:
            self._log_error(f"Error fetching datasets: {str(e)}")
            return []
    
    def execute_workflow(self, workflow_id: int, nodes: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
        """Execute a workflow with enhanced error handling and comprehensive logging"""
        try:
            start_time = time.time()
            self.execution_logs = []
            self.data_cache = {}
            self.model_cache = {}
            self.chart_cache = {}
            
            # Store nodes for helper methods to access
            self._current_nodes = nodes
            
            self._log_info(f"Starting workflow execution for workflow {workflow_id}")
            self._log_info(f"Workflow contains {len(nodes)} nodes and {len(connections)} connections")
            
            # Enhanced validation
            validation = self.validate_workflow(nodes, connections)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"Workflow validation failed: {', '.join(validation['errors'])}",
                    'logs': self.execution_logs,
                    'validation': validation
                }
            
            if validation.get('warnings'):
                for warning in validation['warnings']:
                    self._log_warning(warning)
            
            # Build execution order with dependency resolution
            execution_order = self._build_execution_order(nodes, connections)
            if not execution_order:
                return {
                    'success': False,
                    'error': 'Could not determine execution order - check for circular dependencies',
                    'logs': self.execution_logs
                }
            
            # Store current nodes for AI Summary node type detection
            self._current_nodes = nodes
            
            self._log_info(f"Execution order: {' -> '.join(execution_order)}")
            
            # Execute nodes in order with detailed progress tracking
            results = {}
            node_map = {node['id']: node for node in nodes}
            nodes_completed = 0
            nodes_failed = 0
            
            for i, node_id in enumerate(execution_order):
                node = node_map[node_id]
                progress = (i + 1) / len(execution_order) * 100
                
                self._log_info(f"[{progress:.1f}%] Executing node: {node['name']} ({node['type']})")
                
                try:
                    # Get input data from connected nodes
                    input_data = self._get_node_inputs(node_id, connections, results)
                    
                    # Process the node with timeout and error handling
                    node_start_time = time.time()
                    
                    if node['type'] in self.node_processors:
                        result = self.node_processors[node['type']](node, input_data)
                        
                        # Store result in cache for downstream nodes
                        self.data_cache[node_id] = result
                        
                        node_execution_time = time.time() - node_start_time
                        
                        # Create serializable result summary for frontend
                        result_summary = self._get_output_summary(result)
                        
                        # Prepare the result data for the frontend
                        frontend_result = {
                            'status': 'completed',
                            'result_type': result_summary.get('type'),
                            'result_summary': result_summary,
                            'execution_time': node_execution_time,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        
                        # Include chart data for visualization and analysis nodes
                        if node_id in self.chart_cache and self.chart_cache[node_id]:
                            frontend_result['charts'] = self.chart_cache[node_id]
                            frontend_result['chart_count'] = len(self.chart_cache[node_id])
                            
                        # For descriptive stats nodes, include charts generated
                        if node['type'] == 'descriptive_stats' and isinstance(result, dict) and 'charts_generated' in result:
                            if node_id in self.chart_cache:
                                frontend_result['charts'] = self.chart_cache[node_id]
                                frontend_result['chart_count'] = result['charts_generated']
                        
                        # For analysis nodes (EDA, anomaly detection, event detection), include the full result data
                        if node['type'] in ['eda_analysis', 'univariate_anomaly_detection', 'multivariate_anomaly_detection', 'event_detection']:
                            if isinstance(result, dict):
                                # Include the full analysis results for frontend rendering
                                for key, value in result.items():
                                    if key not in ['data']:  # Exclude raw data to prevent serialization issues
                                        # Convert to JSON-serializable format using EDA service
                                        if hasattr(self.eda_service, '_convert_to_serializable'):
                                            try:
                                                frontend_result[key] = self.eda_service._convert_to_serializable(value)
                                            except Exception as e:
                                                self._log_warning(f"Could not serialize key {key}: {e}")
                                                frontend_result[key] = str(value)
                                        else:
                                            frontend_result[key] = value
                        
                        results[node_id] = frontend_result
                        
                        nodes_completed += 1
                        self._log_info(f"Node {node['name']} completed successfully in {node_execution_time:.2f}s")
                        
                    else:
                        raise Exception(f"Unknown node type: {node['type']}")
                        
                except Exception as e:
                    error_msg = f"Error in node {node['name']}: {str(e)}"
                    self._log_error(error_msg)
                    
                    results[node_id] = {
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat(),
                        'traceback': traceback.format_exc()
                    }
                    
                    nodes_failed += 1
                    
                    # Decide whether to continue or stop based on error handling strategy
                    error_handling = node.get('config', {}).get('error_handling', 'stop')
                    if error_handling == 'stop':
                        self._log_error("Stopping workflow execution due to node failure")
                        break
                    else:
                        self._log_warning(f"Continuing workflow execution despite node failure: {error_msg}")
            
            execution_time = time.time() - start_time
            self._log_info(f"Workflow execution completed in {execution_time:.2f} seconds")
            
            # Generate execution summary
            summary = {
                'total_nodes': len(nodes),
                'nodes_completed': nodes_completed,
                'nodes_failed': nodes_failed,
                'success_rate': (nodes_completed / len(nodes)) * 100 if nodes else 0,
                'execution_time': execution_time,
                'data_generated': len([r for r in results.values() if r.get('status') == 'completed']),
                'charts_generated': len(self.chart_cache)
            }
            
            # Create final result object and ensure all values are properly serializable
            final_result = {
                'success': nodes_failed == 0,
                'results': results,
                'summary': summary,
                'logs': self.execution_logs,
                'charts_count': len(self.chart_cache),
                'validation': validation
            }
            
            # Ensure all data is JSON serializable by converting numpy types
            final_result = self.convert_numpy_types(final_result)
            
            return final_result
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            self._log_error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'logs': self.execution_logs,
                'traceback': traceback.format_exc()
            }
    
    def validate_workflow(self, nodes: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
        """Comprehensive workflow validation with detailed error reporting"""
        errors = []
        warnings = []
        
        try:
            # Basic validation
            if not nodes:
                errors.append("Workflow must contain at least one node")
                return {'valid': False, 'errors': errors, 'warnings': warnings}
            
            # Check for valid node types
            valid_types = set(self.node_processors.keys())
            for node in nodes:
                if node['type'] not in valid_types:
                    errors.append(f"Unknown node type: {node['type']}")
            
            # Check for circular dependencies
            if self._has_circular_dependencies(nodes, connections):
                errors.append("Workflow contains circular dependencies")
            
            # Check for disconnected components
            if len(nodes) > 1 and not self._is_connected_workflow(nodes, connections):
                warnings.append("Workflow contains disconnected components")
            
            # Check for valid connections
            node_ids = {node['id'] for node in nodes}
            for conn in connections:
                if conn['source'] not in node_ids:
                    errors.append(f"Connection references unknown source node: {conn['source']}")
                if conn['target'] not in node_ids:
                    errors.append(f"Connection references unknown target node: {conn['target']}")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'node_count': len(nodes),
                'connection_count': len(connections)
            }
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
    
    def _build_execution_order(self, nodes: List[Dict], connections: List[Dict]) -> List[str]:
        """Build topological execution order using simple algorithm"""
        try:
            # Create adjacency lists
            in_degree = {node['id']: 0 for node in nodes}
            out_edges = {node['id']: [] for node in nodes}
            
            # Build graph from connections
            for conn in connections:
                source = conn['source']
                target = conn['target']
                if source in out_edges and target in in_degree:
                    out_edges[source].append(target)
                    in_degree[target] += 1
            
            # Topological sort using Kahn's algorithm
            queue = [node_id for node_id in in_degree if in_degree[node_id] == 0]
            result = []
            
            while queue:
                current = queue.pop(0)
                result.append(current)
                
                for neighbor in out_edges[current]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            # Check if all nodes were processed (no cycles)
            if len(result) != len(nodes):
                self._log_error("Workflow contains cycles - cannot determine execution order")
                return []
            
            return result
            
        except Exception as e:
            self._log_error(f"Error building execution order: {str(e)}")
            return []
    
    def _has_circular_dependencies(self, nodes: List[Dict], connections: List[Dict]) -> bool:
        """Check for circular dependencies in the workflow"""
        try:
            # Use simple DFS to detect cycles
            visited = set()
            rec_stack = set()
            
            # Create adjacency list
            graph = {node['id']: [] for node in nodes}
            for conn in connections:
                if conn['source'] in graph:
                    graph[conn['source']].append(conn['target'])
            
            def has_cycle(node):
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph.get(node, []):
                    if has_cycle(neighbor):
                        return True
                
                rec_stack.remove(node)
                return False
            
            for node in graph:
                if node not in visited:
                    if has_cycle(node):
                        return True
            
            return False
        except Exception:
            return True  # Assume circular if we can't determine
    
    def _is_connected_workflow(self, nodes: List[Dict], connections: List[Dict]) -> bool:
        """Check if workflow is connected (ignoring direction)"""
        try:
            if len(nodes) <= 1:
                return True
            
            # Create undirected adjacency list
            graph = {node['id']: set() for node in nodes}
            for conn in connections:
                source = conn['source']
                target = conn['target']
                if source in graph and target in graph:
                    graph[source].add(target)
                    graph[target].add(source)
            
            # BFS to check connectivity
            visited = set()
            queue = [next(iter(graph.keys()))]  # Start with any node
            visited.add(queue[0])
            
            while queue:
                current = queue.pop(0)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return len(visited) == len(nodes)
        except Exception:
            return False

    def _get_node_inputs(self, node_id: str, connections: List[Dict], results: Dict) -> Dict[str, Any]:
        """Get input data for a node from its upstream connections"""
        inputs = {}
        
        self._log_info(f"Getting inputs for node {node_id}, connections: {len(connections)}")
        
        # Check if this is an AI Summary node - if so, it needs ALL executed node outputs
        node_type = self._get_node_type_by_id(node_id)
        is_ai_summary = node_type == 'ai_summary'
        
        if is_ai_summary:
            # AI Summary nodes get SUMMARIZED data from meaningful nodes only
            self._log_info(f"AI Summary node detected. Extracting meaningful insights from {len(self.data_cache)} executed nodes")
            self._log_info(f"Current data cache contains nodes: {list(self.data_cache.keys())}")
            
            # Create a filtered and summarized input structure for AI analysis
            meaningful_insights = {}
            for executed_node_id, node_output in self.data_cache.items():
                node_info = self._get_node_info_by_id(executed_node_id)
                node_type = node_info.get('type', 'unknown')
                node_name = node_info.get('name', f'Node {executed_node_id}')
                
                # Extract only meaningful insights based on node type
                summarized_data = self._extract_meaningful_insights(node_output, node_type, node_name)
                
                if summarized_data:  # Only include if there are meaningful insights
                    meaningful_insights[executed_node_id] = {
                        'node_name': node_name,
                        'node_type': node_type,
                        'insights': summarized_data
                    }
                    self._log_info(f"Extracted meaningful insights from {node_name} ({node_type})")
                else:
                    self._log_info(f"Skipped {node_name} ({node_type}) - no meaningful insights to extract")
            
            # Pass only meaningful insights for AI analysis
            inputs['default'] = meaningful_insights
            self._log_info(f"AI Summary receiving meaningful insights from {len(meaningful_insights)} nodes")
            
            # Additional debugging: log the workflow execution state
            if hasattr(self, '_current_nodes'):
                all_node_ids = [n['id'] for n in self._current_nodes]
                executed_node_ids = list(self.data_cache.keys())
                missing_nodes = [nid for nid in all_node_ids if nid not in executed_node_ids]
                if missing_nodes:
                    self._log_warning(f"Missing nodes from cache: {missing_nodes}")
                self._log_info(f"Total workflow nodes: {len(all_node_ids)}, Executed nodes: {len(executed_node_ids)}")
            
        else:
            # Regular nodes get data from their direct connections only
            for conn in connections:
                self._log_info(f"Checking connection: {conn}")
                if conn['target'] == node_id:
                    source_id = conn['source']
                    if source_id in self.data_cache:
                        # Use output port if specified, otherwise use default
                        output_port = conn.get('sourcePort', 'default')
                        input_port = conn.get('targetPort', 'input')
                        
                        # Map input port to 'default' for node processing compatibility
                        actual_input_port = 'default' if input_port == 'input' else input_port
                        
                        source_data = self.data_cache[source_id]
                        
                        # Handle different data types for regular nodes
                        if isinstance(source_data, dict):
                            # If it's a dictionary with a 'data' key (like descriptive stats), 
                            # use that as the default data flow for non-AI nodes
                            if output_port == 'default' and 'data' in source_data:
                                inputs[actual_input_port] = source_data['data']
                            elif output_port in source_data:
                                inputs[actual_input_port] = source_data[output_port]
                            else:
                                # Fallback to the entire dictionary
                                inputs[actual_input_port] = source_data
                        else:
                            # For non-dict data (like DataFrames from data_source), 
                            # pass it directly to the default input
                            inputs[actual_input_port] = source_data
                    else:
                        self._log_warning(f"Source {source_id} not found in data cache")
        
        return inputs
    
    def _get_node_type_by_id(self, node_id: str) -> str:
        """Get node type by node ID from current execution context"""
        # This is a helper to determine node type during execution
        # We'll need to store this information during workflow execution
        if hasattr(self, '_current_nodes'):
            for node in self._current_nodes:
                if node.get('id') == node_id:
                    return node.get('type', 'unknown')
        return 'unknown'
    
    def _get_node_info_by_id(self, node_id: str) -> Dict[str, Any]:
        """Get node information by node ID from current execution context"""
        if hasattr(self, '_current_nodes'):
            for node in self._current_nodes:
                if node.get('id') == node_id:
                    return {
                        'type': node.get('type', 'unknown'),
                        'name': node.get('name', f'Node {node_id}'),
                        'config': node.get('config', {})
                    }
        return {
            'type': 'unknown',
            'name': f'Node {node_id}',
            'config': {}
        }
    
    def _extract_meaningful_insights(self, node_output: Any, node_type: str, node_name: str) -> Dict[str, Any]:
        """Extract only meaningful insights from node output for AI analysis"""
        try:
            insights = {}
            
            # Skip AI-related nodes to avoid recursive data
            if node_type in ['ai_summary', 'ai_insights']:
                return None
            
            # Handle different node types and extract relevant insights
            if node_type == 'data_source':
                if isinstance(node_output, pd.DataFrame):
                    insights = {
                        'data_summary': {
                            'rows': len(node_output),
                            'columns': len(node_output.columns),
                            'column_names': list(node_output.columns)[:20],  # Limit to first 20 columns
                            'data_types': node_output.dtypes.value_counts().to_dict(),
                            'missing_values': node_output.isnull().sum().sum(),
                            'memory_usage_mb': round(node_output.memory_usage(deep=True).sum() / 1024**2, 2)
                        }
                    }
                    
            elif node_type == 'descriptive_stats':
                if isinstance(node_output, dict):
                    insights = {
                        'statistical_summary': {},
                        'data_quality': {},
                        'key_findings': []
                    }
                    
                    # Extract basic statistics
                    if 'basic_stats' in node_output:
                        basic_stats = node_output['basic_stats']
                        if isinstance(basic_stats, pd.DataFrame):
                            insights['statistical_summary'] = {
                                'numeric_columns': len(basic_stats.columns),
                                'key_metrics': basic_stats.to_dict() if len(basic_stats.columns) <= 10 else 'Too many columns to display'
                            }
                    
                    # Extract correlations summary
                    if 'correlations' in node_output:
                        corr_data = node_output['correlations']
                        if isinstance(corr_data, pd.DataFrame):
                            # Find strongest correlations
                            corr_matrix = corr_data.abs()
                            strong_correlations = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    corr_val = corr_matrix.iloc[i, j]
                                    if corr_val > 0.7:  # Strong correlation threshold
                                        strong_correlations.append({
                                            'variables': [corr_matrix.columns[i], corr_matrix.columns[j]],
                                            'correlation': round(corr_data.iloc[i, j], 3)
                                        })
                            insights['data_quality']['strong_correlations'] = strong_correlations[:10]  # Top 10
                    
                    # Extract missing values info
                    if 'missing_values' in node_output:
                        missing_info = node_output['missing_values']
                        if isinstance(missing_info, pd.Series):
                            high_missing = missing_info[missing_info > 0].to_dict()
                            insights['data_quality']['missing_values'] = high_missing
                            
            elif node_type in ['classification', 'regression']:
                if isinstance(node_output, dict):
                    insights = {
                        'model_performance': {},
                        'model_details': {},
                        'predictions_summary': {}
                    }
                    
                    # Extract model metrics
                    if 'metrics' in node_output:
                        insights['model_performance'] = node_output['metrics']
                    
                    # Extract model algorithm and parameters
                    if 'algorithm' in node_output:
                        insights['model_details']['algorithm'] = node_output['algorithm']
                    
                    if 'feature_importance' in node_output:
                        importance = node_output['feature_importance']
                        if isinstance(importance, dict):
                            # Get top 10 most important features
                            sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                            insights['model_details']['top_features'] = dict(sorted_importance)
                            
            elif node_type == 'clustering':
                if isinstance(node_output, dict):
                    insights = {
                        'clustering_results': {},
                        'cluster_analysis': {}
                    }
                    
                    if 'n_clusters' in node_output:
                        insights['clustering_results']['n_clusters'] = node_output['n_clusters']
                    
                    if 'cluster_centers' in node_output:
                        centers = node_output['cluster_centers']
                        if hasattr(centers, 'shape'):
                            insights['clustering_results']['cluster_centers_shape'] = list(centers.shape)
                    
                    if 'silhouette_score' in node_output:
                        insights['clustering_results']['silhouette_score'] = round(node_output['silhouette_score'], 4)
                        
            elif node_type in ['eda_analysis', 'univariate_anomaly_detection', 'multivariate_anomaly_detection']:
                if isinstance(node_output, dict):
                    insights = {
                        'analysis_summary': {},
                        'key_findings': [],
                        'recommendations': []
                    }
                    
                    # Extract analysis results
                    if 'results' in node_output:
                        results = node_output['results']
                        if isinstance(results, dict):
                            # Extract overview information
                            if 'overview' in results:
                                overview = results['overview']
                                if isinstance(overview, dict):
                                    insights['analysis_summary'] = {
                                        k: v for k, v in overview.items() 
                                        if not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))
                                    }
                            
                            # Extract recommendations
                            if 'recommendations' in results:
                                recommendations = results['recommendations']
                                if isinstance(recommendations, list):
                                    insights['recommendations'] = recommendations[:10]  # Top 10 recommendations
                                    
                            # Extract anomalies summary for anomaly detection nodes
                            if 'anomalies' in results:
                                anomalies = results['anomalies']
                                if isinstance(anomalies, dict):
                                    anomaly_summary = {}
                                    for method, data in anomalies.items():
                                        if isinstance(data, pd.DataFrame):
                                            anomaly_summary[method] = {
                                                'total_anomalies': len(data),
                                                'percentage': round(len(data) / len(data) * 100, 2) if len(data) > 0 else 0
                                            }
                                    insights['analysis_summary']['anomaly_detection'] = anomaly_summary
                                    
            elif node_type in ['basic_plots', 'advanced_plots']:
                if isinstance(node_output, dict):
                    insights = {
                        'visualization_summary': {
                            'charts_generated': len([k for k in node_output.keys() if 'chart' in k.lower() or 'plot' in k.lower()]),
                            'plot_type': node_output.get('plot_type', 'unknown'),
                            'features_visualized': node_output.get('features_used', [])
                        }
                    }
                    
            elif node_type == 'data_cleaning':
                if isinstance(node_output, dict):
                    insights = {
                        'cleaning_summary': {},
                        'data_quality_improvement': {}
                    }
                    
                    if 'cleaning_summary' in node_output:
                        cleaning_info = node_output['cleaning_summary']
                        insights['cleaning_summary'] = cleaning_info
                    
                    # If there's processed data, get basic info
                    if 'data' in node_output and isinstance(node_output['data'], pd.DataFrame):
                        cleaned_data = node_output['data']
                        insights['data_quality_improvement'] = {
                            'rows_after_cleaning': len(cleaned_data),
                            'columns_after_cleaning': len(cleaned_data.columns),
                            'remaining_missing_values': cleaned_data.isnull().sum().sum()
                        }
                        
            else:
                # For unknown node types, try to extract basic information
                if isinstance(node_output, pd.DataFrame):
                    insights = {
                        'data_info': {
                            'type': 'DataFrame',
                            'shape': list(node_output.shape),
                            'columns': list(node_output.columns)[:10]  # First 10 columns only
                        }
                    }
                elif isinstance(node_output, dict):
                    # Extract non-complex data from dictionary
                    simple_data = {}
                    for k, v in node_output.items():
                        if isinstance(v, (str, int, float, bool, list)) and not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
                            if isinstance(v, list) and len(v) > 20:
                                simple_data[k] = f"List with {len(v)} items"
                            else:
                                simple_data[k] = v
                    if simple_data:
                        insights['extracted_data'] = simple_data
            
            # Add node execution metadata
            if insights:
                insights['node_metadata'] = {
                    'node_type': node_type,
                    'node_name': node_name,
                    'data_type': type(node_output).__name__
                }
            
            return insights if insights else None
            
        except Exception as e:
            self._log_error(f"Error extracting insights from {node_name}: {str(e)}")
            return {
                'error': f"Could not extract insights: {str(e)}",
                'node_metadata': {
                    'node_type': node_type,
                    'node_name': node_name,
                    'data_type': type(node_output).__name__
                }
            }
    
    def _get_output_summary(self, result: Any) -> Dict[str, Any]:
        """Generate a summary of node output for logging"""
        try:
            if isinstance(result, pd.DataFrame):
                return {
                    'type': 'DataFrame',
                    'shape': list(result.shape),
                    'columns': list(result.columns),
                    'memory_usage': f"{result.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                }
            elif isinstance(result, dict):
                # Extract relevant information from dictionary results
                summary = {
                    'type': 'Dictionary',
                    'keys': list(result.keys()),
                    'size': len(result)
                }
                
                # Special handling for descriptive stats results with JSON serialization
                if 'basic_stats' in result:
                    # Convert pandas data to JSON-serializable format
                    basic_stats = result['basic_stats']
                    if isinstance(basic_stats, dict):
                        # Ensure all values are JSON serializable
                        summary['basic_stats'] = self.convert_numpy_types(basic_stats)
                        
                    # Add statistical summary for the frontend
                    summary['statistics_summary'] = {
                        'columns_analyzed': len(basic_stats),
                        'total_records': int(list(basic_stats.values())[0].get('count', 0)) if basic_stats else 0,
                        'numeric_columns': len([col for col in basic_stats.keys() if 'mean' in basic_stats.get(col, {})]),
                    }
                    
                    # Add column type information
                    numeric_cols = [col for col in basic_stats.keys() if 'mean' in basic_stats.get(col, {})]
                    categorical_cols = len(basic_stats) - len(numeric_cols)
                    summary['column_info'] = {
                        'numeric': len(numeric_cols),
                        'categorical': categorical_cols,
                        'datetime': 0  # Will be enhanced later
                    }
                
                if 'data_types' in result:
                    # Convert pandas dtypes to strings
                    data_types = result['data_types']
                    if hasattr(data_types, 'to_dict'):
                        summary['data_types'] = {k: str(v) for k, v in data_types.to_dict().items()}
                    else:
                        summary['data_types'] = {k: str(v) for k, v in data_types.items()}
                
                if 'missing_values' in result:
                    # Convert pandas Series to dict with int values
                    missing_values = result['missing_values']
                    if hasattr(missing_values, 'to_dict'):
                        summary['missing_values'] = self.convert_numpy_types(missing_values.to_dict())
                    else:
                        summary['missing_values'] = self.convert_numpy_types(missing_values)
                
                if 'correlations' in result:
                    # Convert pandas correlation matrix to JSON-serializable format
                    correlations = result['correlations']
                    if hasattr(correlations, 'to_dict'):
                        corr_dict = correlations.to_dict()
                        summary['correlations'] = {
                            outer_key: {inner_key: float(value) if pd.notna(value) else None 
                                       for inner_key, value in inner_dict.items()}
                            for outer_key, inner_dict in corr_dict.items()
                        }
                    else:
                        summary['correlations'] = correlations
                
                # Special handling for AI summary results
                if 'ai_analysis' in result:
                    summary['ai_analysis'] = self.convert_numpy_types(result['ai_analysis'])
                
                # Special handling for ML model results
                if 'algorithm' in result and ('metrics' in result or 'n_clusters' in result):
                    summary['type'] = 'ML Model'
                    summary['algorithm'] = result['algorithm']
                    
                    # Handle classification/regression metrics
                    if 'metrics' in result:
                        summary['metrics'] = self.convert_numpy_types(result['metrics'])
                    
                    if 'target_column' in result:
                        summary['target_column'] = result['target_column']
                    
                    if 'test_size' in result and 'train_size' in result:
                        summary['dataset_split'] = {
                            'train_size': result['train_size'],
                            'test_size': result['test_size'],
                            'total_size': result['train_size'] + result['test_size']
                        }
                    
                    # Clustering specific
                    if 'n_clusters' in result:
                        summary['n_clusters'] = result['n_clusters']
                        summary['n_samples'] = result.get('n_samples', 0)
                        summary['silhouette_score'] = result.get('silhouette_score', 0)
                        
                        # Include feature information
                        if 'feature_columns_used' in result:
                            summary['feature_columns_used'] = result['feature_columns_used']
                        
                        # Include cluster distribution
                        if 'cluster_distribution' in result:
                            summary['cluster_distribution'] = result['cluster_distribution']
                        
                        # Algorithm-specific metrics
                        if 'inertia' in result:  # K-means
                            summary['inertia'] = result['inertia']
                        if 'n_noise_points' in result:  # DBSCAN
                            summary['n_noise_points'] = result['n_noise_points']
                            summary['eps'] = result.get('eps')
                            summary['min_samples'] = result.get('min_samples')
                        if 'aic' in result:  # Gaussian Mixture
                            summary['aic'] = result['aic']
                        if 'bic' in result:  # Gaussian Mixture
                            summary['bic'] = result['bic']
                        
                        # Include visualization if available
                        if 'cluster_plot' in result:
                            summary['cluster_plot'] = result['cluster_plot']
                        
                        # Feature scaling info
                        if 'scaled_features' in result:
                            summary['scaled_features'] = result['scaled_features']
                        
                        # Create comprehensive clustering metrics dict
                        clustering_metrics = {}
                        if 'silhouette_score' in result:
                            clustering_metrics['silhouette_score'] = result['silhouette_score']
                        if 'inertia' in result:
                            clustering_metrics['inertia'] = result['inertia']
                        if 'aic' in result:
                            clustering_metrics['aic'] = result['aic']
                        if 'bic' in result:
                            clustering_metrics['bic'] = result['bic']
                        if 'n_noise_points' in result:
                            clustering_metrics['n_noise_points'] = result['n_noise_points']
                        
                        if clustering_metrics:
                            summary['metrics'] = clustering_metrics
                
                # Special handling for data cleaning results
                if 'cleaning_summary' in result:
                    summary['type'] = 'Cleaned Data'
                    cleaning_summary = result['cleaning_summary']
                    summary.update({
                        'original_shape': cleaning_summary['original_shape'],
                        'final_shape': cleaning_summary['final_shape'],
                        'rows_removed': cleaning_summary['rows_removed'],
                        'columns_removed': cleaning_summary['columns_removed'],
                        'operations_performed': cleaning_summary['operations_performed'],
                        'data_quality_score': cleaning_summary['data_quality_score']
                    })
                    
                    # Include data info if available
                    if 'data' in result and isinstance(result['data'], pd.DataFrame):
                        df = result['data']
                        summary.update({
                            'shape': list(df.shape),
                            'columns': list(df.columns),
                            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                        })
                
                # Special handling for feature engineering results
                if 'engineering_summary' in result:
                    summary['type'] = 'Engineered Data'
                    eng_summary = result['engineering_summary']
                    summary.update({
                        'original_shape': eng_summary['original_shape'],
                        'final_shape': eng_summary['final_shape'],
                        'features_added': eng_summary['features_added'],
                        'operations_performed': eng_summary['operations_performed'],
                        'feature_types': eng_summary['feature_types']
                    })
                    
                    # Include data info if available
                    if 'data' in result and isinstance(result['data'], pd.DataFrame):
                        df = result['data']
                        summary.update({
                            'shape': list(df.shape),
                            'columns': list(df.columns),
                            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                        })
                
                if 'dataset_summary' in result:
                    summary['dataset_summary'] = self.convert_numpy_types(result['dataset_summary'])
                
                if 'analysis_config' in result:
                    summary['analysis_config'] = self.convert_numpy_types(result['analysis_config'])
                
                if 'advanced_ai_analysis' in result:
                    summary['advanced_ai_analysis'] = self.convert_numpy_types(result['advanced_ai_analysis'])
                
                return summary
            elif isinstance(result, list):
                return {
                    'type': 'List',
                    'length': len(result),
                    'item_type': type(result[0]).__name__ if result else 'empty'
                }
            else:
                return {
                    'type': type(result).__name__,
                    'size': len(str(result))
                }
        except Exception as e:
            self._log_error(f"Error generating output summary: {str(e)}")
            return {'type': 'Unknown', 'error': f'Could not generate summary: {str(e)}'}
    
    # Node Processing Methods
    def load_dataset_for_analysis(self, dataset_id: str) -> pd.DataFrame:
        """Load a dataset for column analysis"""
        try:
            dataset_id = int(dataset_id)
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Load the actual data file
            if dataset.file_type.lower() == 'csv':
                df = pd.read_csv(dataset.file_path)
            elif dataset.file_type.lower() in ['xlsx', 'xls']:
                df = pd.read_excel(dataset.file_path)
            elif dataset.file_type.lower() == 'json':
                df = pd.read_json(dataset.file_path)
            else:
                raise ValueError(f"Unsupported file type: {dataset.file_type}")
            
            return df
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")

    def _process_data_source(self, node: Dict, input_data: Dict) -> Any:
        """Process data source node - load dataset from database"""
        try:
            # Check both config and parameters for dataset_id
            config = node.get('config', {})
            parameters = node.get('parameters', {})
            dataset_id = config.get('dataset_id') or parameters.get('dataset_id')
            
            if not dataset_id:
                raise ValueError("Dataset ID is required for data source node")
            
            # Convert to int if it's a string
            try:
                dataset_id = int(dataset_id)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid dataset ID: {dataset_id}")
            
            # Load dataset from database
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Load the actual data file
            if dataset.file_type.lower() == 'csv':
                df = pd.read_csv(dataset.file_path)
            elif dataset.file_type.lower() in ['xlsx', 'xls']:
                df = pd.read_excel(dataset.file_path)
            elif dataset.file_type.lower() == 'json':
                df = pd.read_json(dataset.file_path)
            else:
                raise ValueError(f"Unsupported file type: {dataset.file_type}")
            
            # Apply sampling if requested
            sample_size = config.get('sample_size') or parameters.get('sample_size')
            if sample_size and sample_size < len(df):
                random_seed = config.get('random_seed') or parameters.get('random_seed', 42)
                df = df.sample(n=int(sample_size), random_state=random_seed)
            
            # Select specific columns if requested
            columns = config.get('columns') or parameters.get('columns')
            if columns:
                if isinstance(columns, str):
                    columns = [col.strip() for col in columns.split(',')]
                df = df[columns]
            
            self._log_info(f"Loaded dataset: {dataset.name} with shape {df.shape}")
            
            return df
            
        except Exception as e:
            self._log_error(f"Error processing data source: {str(e)}")
            raise
    

    
    def _prepare_eda_data_for_ai(self, df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
        """Prepare comprehensive EDA data structure for AI analysis"""
        try:
            # Basic dataset information
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Calculate correlations for numeric columns
            correlations = {}
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    correlations = corr_matrix.to_dict()
                except Exception as e:
                    self._log_error(f"Error calculating correlations: {str(e)}")
            
            # Detect outliers using IQR method
            outliers = {}
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns for performance
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                    outlier_count = outlier_condition.sum()
                    if outlier_count > 0:
                        outliers[col] = int(outlier_count)
                except Exception as e:
                    self._log_error(f"Error detecting outliers for {col}: {str(e)}")
            
            # Calculate data quality metrics
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            data_density = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
            
            # Build comprehensive EDA data structure
            eda_data = {
                'dataset_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'data_quality_score': min(100, max(0, data_density))
                },
                'results': {
                    'overview': {
                        'shape': {'rows': len(df), 'columns': len(df.columns)},
                        'data_density': data_density,
                        'memory_usage': {'total_mb': df.memory_usage(deep=True).sum() / 1024**2},
                        'column_details': {}
                    },
                    'statistics': {
                        'relationships': {'correlations': correlations},
                        'outliers': outliers,
                        'statistical_tests': {},
                        'recommendations': []
                    }
                },
                'charts': {},  # Charts would be populated if available
                'advanced_analysis': {}
            }
            
            # Add detailed column information
            for col in df.columns:
                col_info = {
                    'dtype': str(df[col].dtype),
                    'non_null_count': int(df[col].count()),
                    'unique_count': int(df[col].nunique()),
                }
                
                if col in numeric_cols:
                    try:
                        col_info.update({
                            'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                            'std': float(df[col].std()) if not df[col].isna().all() else None,
                            'min': float(df[col].min()) if not df[col].isna().all() else None,
                            'max': float(df[col].max()) if not df[col].isna().all() else None,
                            'median': float(df[col].median()) if not df[col].isna().all() else None
                        })
                    except Exception as e:
                        self._log_error(f"Error calculating stats for {col}: {str(e)}")
                
                eda_data['results']['overview']['column_details'][col] = col_info
            
            return eda_data
            
        except Exception as e:
            self._log_error(f"Error preparing EDA data for AI: {str(e)}")
            # Return minimal structure on error
            return {
                'dataset_info': {'rows': len(df), 'columns': len(df.columns)},
                'results': {'overview': {'shape': {'rows': len(df), 'columns': len(df.columns)}}},
                'error': str(e)
            }
    
    def _get_basic_dataset_summary(self, data) -> Dict[str, Any]:
        """Get basic dataset summary for fallback scenarios"""
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict) and 'data' in data:
                df = data['data']
            else:
                return {'error': 'Unable to extract dataset information'}
            
            return {
                'shape': {'rows': len(df), 'columns': len(df.columns)},
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
        except Exception as e:
            return {'error': f'Error generating summary: {str(e)}'}
    
    def _process_descriptive_stats(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process descriptive statistics node"""
        try:
            if 'default' not in input_data:
                raise ValueError("Descriptive stats node requires input data")
            
            # Handle different input data formats
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                # Handle output from advanced plots or other nodes that include 'data' key
                df = input_obj['data']
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            
            # Handle datetime columns before processing statistics
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Convert datetime columns to strings to avoid serialization issues
                    df[col] = df[col].astype(str)
            
            # Basic statistics
            try:
                stats_df = df.describe(include='all')
                
                # Convert the describe() result to a serializable dictionary
                basic_stats = {}
                for col in stats_df.columns:
                    col_stats = {}
                    for stat_name in stats_df.index:
                        value = stats_df.loc[stat_name, col]
                        # Handle timestamp and other problematic types
                        if isinstance(value, pd.Timestamp):
                            col_stats[stat_name] = value.isoformat()
                        elif pd.isna(value):  # Explicitly handle NaN values
                            col_stats[stat_name] = None
                        else:
                            col_stats[stat_name] = value
                    basic_stats[col] = col_stats
            except Exception as e:
                self._log_warning(f"Error calculating basic statistics: {str(e)}")
                # Provide minimal statistics if describe() fails
                basic_stats = {
                    col: {"count": len(df[col].dropna()), "missing": df[col].isna().sum()} 
                    for col in df.columns
                }
            
            stats = {
                'basic_stats': basic_stats,
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': df.isnull().sum().to_dict(),
                'unique_values': df.nunique().to_dict()
            }
            
            # Correlation analysis if requested
            if config.get('include_correlations', True):
                method = config.get('correlation_method', 'pearson')
                try:
                    # Only use numeric columns for correlation
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Check if we have enough numeric columns for correlation
                    if len(numeric_cols) > 1:
                        # Copy the data to avoid modifying the original
                        corr_df = df[numeric_cols].copy()
                        
                        # Calculate correlations safely
                        try:
                            corr_matrix = corr_df.corr(method=method)
                            stats['correlations'] = self.convert_numpy_types(corr_matrix.to_dict())
                        except Exception as e:
                            self._log_warning(f"Error calculating correlations: {str(e)}")
                            stats['correlations'] = {}
                    else:
                        stats['correlations'] = {}
                        self._log_info("Not enough numeric columns for correlation analysis")
                except Exception as e:
                    self._log_warning(f"Error during correlation preparation: {str(e)}")
                    stats['correlations'] = {}
            
            # Group by analysis if specified
            group_by = config.get('group_by')
            if isinstance(group_by, str) and group_by in df.columns:
                # Handle complex nested results from groupby operations
                try:
                    # Instead of directly converting to dict, which can have serialization issues,
                    # process the grouped data carefully
                    grouped_df = df.groupby(group_by).describe()
                    
                    # Manually create a serialization-friendly structure
                    grouped_stats = {}
                    for col_name in grouped_df.columns.levels[0]:
                        grouped_stats[col_name] = {}
                        for group_val in grouped_df.index:
                            grouped_stats[col_name][str(group_val)] = {}
                            for stat in grouped_df[col_name].columns:
                                value = grouped_df.loc[group_val, (col_name, stat)]
                                if isinstance(value, pd.Timestamp):
                                    grouped_stats[col_name][str(group_val)][stat] = value.isoformat()
                                else:
                                    grouped_stats[col_name][str(group_val)][stat] = value
                    
                    stats['grouped_stats'] = grouped_stats
                except Exception as e:
                    self._log_warning(f"Error in group by analysis: {str(e)}")
                    # Fallback to a simpler representation
                    simple_group = df.groupby(group_by).size().to_dict()
                    stats['grouped_stats'] = {
                        'count': {str(k): v for k, v in simple_group.items()}
                    }
            
            # Generate plots if requested
            if config.get('generate_plots', True):
                charts = self._generate_distribution_plots(df)
                self.chart_cache[node['id']] = charts
                stats['charts_generated'] = len(charts)
            
            # Include original data for downstream processing if not too large
            # Adding the DataFrame directly can cause serialization issues
            # Add metadata about the DataFrame instead
            stats['data_info'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # Only include actual data if it's not too large (less than 10,000 cells)
            # Always convert to records and apply numpy conversion to avoid serialization issues
            try:
                rows = int(df.shape[0]) if hasattr(df, 'shape') and len(df.shape) > 0 else 0
                cols = int(df.shape[1]) if hasattr(df, 'shape') and len(df.shape) > 1 else 0
                data_size = rows * cols
                
                if data_size < 10000:
                    # Convert DataFrame to records and then handle numpy types to ensure it's serializable
                    try:
                        records = df.to_dict('records') if hasattr(df, 'to_dict') else []
                        stats['data'] = self.convert_numpy_types(records)
                    except Exception as e:
                        self._log_warning(f"Error converting full dataset to records: {str(e)}")
                        # Fall back to a sample if full conversion fails
                        sample_size = min(100, rows)
                        sample_records = df.head(sample_size).to_dict('records') if hasattr(df, 'head') else []
                        stats['data_sample'] = self.convert_numpy_types(sample_records)
                else:
                    # For large datasets, provide a sample instead
                    sample_size = min(100, rows)
                    sample_records = df.head(sample_size).to_dict('records') if hasattr(df, 'head') else []
                    stats['data_sample'] = self.convert_numpy_types(sample_records)
                    self._log_info(f"Large DataFrame detected ({rows}x{cols}), returning sample of {sample_size} rows")
            except Exception as e:
                self._log_warning(f"Error handling DataFrame data: {str(e)}")
                # Fall back to just returning metadata
                stats['_error'] = f"Could not process DataFrame data: {str(e)}"
            
            self._log_info(f"Generated descriptive statistics for {len(df.columns)} columns")
            
            # Ensure all stats data is properly serializable
            stats = self.convert_numpy_types(stats)
            
            return stats
            
        except Exception as e:
            self._log_error(f"Error processing descriptive stats: {str(e)}")
            raise
    
    def _process_classification(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process classification node with advanced column selection"""
        try:
            if 'default' not in input_data:
                raise ValueError("Classification node requires input data")
            
            # Handle different input data formats
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                df = input_obj['data']
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            
            # Advanced column analysis and selection
            column_analysis = self._analyze_columns_for_classification(df)
            
            # Auto-select target column if not specified
            target_column = config.get('target_column')
            if not target_column or target_column == 'None' or target_column not in df.columns:
                # Intelligent target column selection
                auto_target = self._auto_select_target_column(df, column_analysis)
                if auto_target:
                    target_column = auto_target
                    self._log_info(f"Auto-selected target column: '{target_column}' (type: {column_analysis['columns'][target_column]['recommended_role']})")
                else:
                    raise ValueError("No suitable target column found. Please specify a target column with categorical or binary data.")
            else:
                # Validate explicitly selected target column
                self._log_info(f"Using user-specified target column: '{target_column}'")
                
            # Validate target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data. Available columns: {list(df.columns)}")
            
            # Skip identifier columns (timestamps, IDs, etc.)
            if any(keyword in target_column.lower() for keyword in ['id', 'index', 'key', 'uuid', 'timestamp']):
                raise ValueError(f"Target column '{target_column}' appears to be an identifier column. "
                               f"Please select a column with categorical or numeric values suitable for prediction. "
                               f"Suggested alternatives: {[col for col in column_analysis['recommendations']['target_candidates'] if col != target_column][:3]}")
            
            # Basic validation - must have at least 2 unique values
            unique_count = df[target_column].nunique()
            if unique_count < 2:
                raise ValueError(f"Target column '{target_column}' has only {unique_count} unique value(s). Classification requires at least 2 different values.")
            
            # Log information about target column
            self._log_info(f"Target column '{target_column}' has {unique_count} unique values. Will preprocess for classification if needed.")
            
            # Auto-select feature columns if not specified
            feature_columns = config.get('feature_columns', [])
            if not feature_columns:
                feature_columns = self._auto_select_feature_columns(df, target_column, column_analysis)
                self._log_info(f"Auto-selected {len(feature_columns)} feature columns: {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")
            
            # Filter out invalid feature columns
            valid_features = [col for col in feature_columns if col in df.columns and col != target_column]
            if not valid_features:
                # Fallback: use all numeric columns except target
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                valid_features = [col for col in numeric_cols if col != target_column]
                if not valid_features:
                    raise ValueError("No valid feature columns found for classification")
            
            algorithm = config.get('algorithm', 'random_forest')
            test_size = config.get('test_size', 0.2)
            
            # Prepare features and target
            X = df[valid_features].copy()
            y = df[target_column].copy()
            
            # Advanced target preprocessing
            y_processed, target_info = self._preprocess_target_for_classification(y, target_column)
            
            # Advanced feature preprocessing
            X_processed = self._preprocess_features_for_classification(X, config)
            
            # Ensure we have enough samples per class
            value_counts = pd.Series(y_processed).value_counts()
            if len(value_counts) < 2:
                raise ValueError(f"Target column '{target_column}' must have at least 2 classes for classification. Found: {len(value_counts)}")
            
            min_class_size = value_counts.min()
            if min_class_size < 2:
                raise ValueError(f"Target column '{target_column}' has classes with too few samples. Minimum class size: {min_class_size}")
            
            # Split data with stratification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed
                )
            except ValueError as e:
                # If stratification fails due to small class sizes, use random split
                self._log_warning(f"Stratification failed, using random split: {str(e)}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42
                )
            
            # Train model
            if algorithm == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == 'logistic_regression':
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif algorithm == 'svm':
                model = SVC(random_state=42, probability=True)
            elif algorithm == 'gradient_boosting':
                model = GradientBoostingClassifier(random_state=42)
            elif algorithm == 'naive_bayes':
                model = GaussianNB()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Cross-validation if requested
            if config.get('cross_validation', True):
                cv_folds = config.get('cv_folds', 5)
                cv_scores = cross_val_score(model, X_processed, y_processed, cv=cv_folds, scoring='accuracy')
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
            
            # Feature importance if available
            feature_importance = None
            if config.get('feature_importance', True) and hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                # For linear models, use coefficient magnitudes
                feature_importance = np.abs(model.coef_[0]).tolist() if len(model.coef_.shape) > 1 else np.abs(model.coef_).tolist()
            
            # Store model for later use
            model_id = f"model_{node['id']}"
            self.model_cache[model_id] = {
                'model': model,
                'feature_columns': list(X_processed.columns),
                'target_column': target_column,
                'algorithm': algorithm
            }
            
            result = {
                'model_id': model_id,
                'metrics': metrics,
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                'feature_importance': feature_importance,
                'feature_names': list(X_processed.columns),
                'test_size': len(y_test),
                'train_size': len(y_train),
                'algorithm': algorithm,
                'target_column': target_column
            }
            
            self._log_info(f"Classification model trained with accuracy: {metrics['accuracy']:.3f}")
            
            return result
            
        except Exception as e:
            self._log_error(f"Error processing classification: {str(e)}")
            raise
    
    def _process_basic_plots(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process basic plots node with automatic column detection"""
        try:
            if 'default' not in input_data:
                raise ValueError("Basic plots node requires input data")
            
            # Handle different input data formats
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                # Handle output from descriptive stats node
                df = input_obj['data']
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            parameters = node.get('parameters', {})
            
            # Use parameters if config is empty, for backward compatibility
            if not config and parameters:
                config = parameters
            
            plot_type = config.get('plot_type', 'histogram')
            x_column = config.get('x_column') or config.get('column_select')  # Support both formats
            y_column = config.get('y_column')
            color_by = config.get('color_by')
            auto_detect = config.get('auto_detect_columns', True)
            
            # Automatic column detection
            if auto_detect and not x_column:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if plot_type == 'histogram' and numeric_cols:
                    x_column = numeric_cols[0]
                    self._log_info(f"Auto-detected x_column: {x_column}")
                elif plot_type == 'scatter' and len(numeric_cols) >= 2:
                    x_column = numeric_cols[0]
                    y_column = y_column or numeric_cols[1]
                    self._log_info(f"Auto-detected columns: x={x_column}, y={y_column}")
                elif plot_type == 'boxplot' and numeric_cols:
                    x_column = numeric_cols[0]
                    self._log_info(f"Auto-detected x_column: {x_column}")
                elif plot_type == 'barplot' and categorical_cols and numeric_cols:
                    x_column = categorical_cols[0]
                    y_column = y_column or numeric_cols[0]
                    self._log_info(f"Auto-detected columns: x={x_column}, y={y_column}")
            
            if not x_column or x_column not in df.columns:
                available_cols = list(df.columns)
                raise ValueError(f"X column '{x_column}' not found in data. Available columns: {available_cols}")
            
            # Create plot based on type
            figsize_width = config.get('figsize_width', 10)
            figsize_height = config.get('figsize_height', 6)
            fig = plt.figure(figsize=(figsize_width, figsize_height))
            
            if plot_type == 'histogram':
                plt.hist(df[x_column].dropna(), bins=config.get('bins', 30), alpha=0.7, edgecolor='black')
                plt.xlabel(x_column)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {x_column}')
                plt.grid(True, alpha=0.3)
                
            elif plot_type == 'scatter':
                if not y_column:
                    # Auto-detect y_column if not provided
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if x_column in numeric_cols:
                        numeric_cols.remove(x_column)
                    if numeric_cols:
                        y_column = numeric_cols[0]
                
                if y_column and y_column in df.columns:
                    if color_by and color_by in df.columns:
                        scatter = plt.scatter(df[x_column], df[y_column], c=df[color_by], alpha=0.6, cmap='viridis')
                        plt.colorbar(scatter, label=color_by)
                    else:
                        plt.scatter(df[x_column], df[y_column], alpha=0.6)
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
                    plt.title(f'Scatter plot: {x_column} vs {y_column}')
                    plt.grid(True, alpha=0.3)
                else:
                    raise ValueError(f"Y column '{y_column}' not found for scatter plot")
                
            elif plot_type == 'boxplot':
                if color_by and color_by in df.columns:
                    # Group by color_by column
                    df.boxplot(column=x_column, by=color_by)
                    plt.title(f'Box plot of {x_column} by {color_by}')
                else:
                    df.boxplot(column=x_column)
                    plt.title(f'Box plot of {x_column}')
                plt.suptitle('')  # Remove default title
                
            elif plot_type == 'barplot':
                if not y_column:
                    # Count plot for categorical data
                    df[x_column].value_counts().plot(kind='bar')
                    plt.xlabel(x_column)
                    plt.ylabel('Count')
                    plt.title(f'Bar plot of {x_column}')
                else:
                    # Group by x_column and aggregate y_column
                    grouped = df.groupby(x_column)[y_column].mean()
                    grouped.plot(kind='bar')
                    plt.xlabel(x_column)
                    plt.ylabel(f'Mean {y_column}')
                    plt.title(f'Bar plot: {x_column} vs {y_column}')
                plt.xticks(rotation=45)
                
            elif plot_type == 'lineplot':
                if y_column and y_column in df.columns:
                    plt.plot(df[x_column], df[y_column], marker='o', alpha=0.7)
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
                    plt.title(f'Line plot: {x_column} vs {y_column}')
                    plt.grid(True, alpha=0.3)
                else:
                    raise ValueError(f"Y column '{y_column}' is required for line plot")
            
            plt.tight_layout()
            
            # Save plot to base64
            plot_base64 = self._save_plot_as_base64(fig)
            plt.close()
            
            # Structure result similar to advanced_plots
            result = {
                'charts': {
                    f'{plot_type}_{x_column}': plot_base64
                },
                'chart_count': 1,
                'plot_type': plot_type,
                'x_column': x_column,
                'y_column': y_column,
                'color_by': color_by,
                'columns_used': [c for c in [x_column, y_column, color_by] if c],
                'available_columns': list(df.columns),
                'data': df  # Include original DataFrame for downstream nodes
            }
            
            # Cache the charts for summary counting
            self.chart_cache[node['id']] = result['charts']
            
            self._log_info(f"Generated {plot_type} plot for {x_column} (y: {y_column}, color: {color_by})")
            
            return result
            
        except Exception as e:
            self._log_error(f"Error processing basic plots: {str(e)}")
            raise
    
    def _process_export_data(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process export data node with enhanced format support including PDF"""
        try:
            if 'default' not in input_data:
                raise ValueError("Export data node requires input data")
            
            input_obj = input_data['default']
            config = node.get('config', {})
            
            format_type = config.get('format', 'csv')
            filename = config.get('filename', 'export_data')
            include_index = config.get('include_index', False)
            compress = config.get('compress', False)
            
            # Handle different data types - convert to DataFrame if needed
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
                metadata = {'data_type': 'DataFrame', 'source': 'direct'}
            elif isinstance(input_obj, dict):
                metadata = {'data_type': 'Dictionary', 'source': 'converted'}
                
                # If it's a dictionary (e.g., from descriptive stats), use the data key or convert basic_stats
                if 'data' in input_obj and isinstance(input_obj['data'], pd.DataFrame):
                    # Use the original DataFrame
                    df = input_obj['data']
                    metadata['conversion'] = 'used_data_key'
                elif 'basic_stats' in input_obj:
                    # Convert descriptive stats to DataFrame
                    df = pd.DataFrame(input_obj['basic_stats'])
                    metadata['conversion'] = 'basic_stats_to_dataframe'
                elif 'ai_analysis' in input_obj:
                    # Convert AI analysis results to DataFrame
                    ai_data = input_obj['ai_analysis']
                    if isinstance(ai_data, dict):
                        df = pd.DataFrame([ai_data])
                    else:
                        df = pd.DataFrame(ai_data)
                    metadata['conversion'] = 'ai_analysis_to_dataframe'
                else:
                    # Generic dictionary, try to convert
                    df = pd.DataFrame([input_obj])
                    metadata['conversion'] = 'generic_dict_to_dataframe'
            else:
                # Try to convert to DataFrame
                try:
                    df = pd.DataFrame(input_obj)
                    metadata = {'data_type': f'{type(input_obj).__name__}', 'source': 'converted'}
                except:
                    raise ValueError(f"Cannot convert data of type {type(input_obj)} to exportable format")
            
            # Create exports directory if it doesn't exist
            export_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Handle different file extensions based on format
            if format_type == 'pdf':
                full_filename = f"{filename}_{timestamp}.pdf"
            elif format_type == 'excel':
                full_filename = f"{filename}_{timestamp}.xlsx"
            else:
                full_filename = f"{filename}_{timestamp}.{format_type}"
            
            file_path = os.path.join(export_dir, full_filename)
            
            # Export based on format
            if format_type == 'csv':
                df.to_csv(file_path, index=include_index)
                if compress:
                    import gzip
                    import shutil
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(f"{file_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(file_path)
                    file_path = f"{file_path}.gz"
                    full_filename = f"{full_filename}.gz"
                    
            elif format_type == 'json':
                df.to_json(file_path, orient='records', indent=2)
                if compress:
                    import gzip
                    import shutil
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(f"{file_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(file_path)
                    file_path = f"{file_path}.gz"
                    full_filename = f"{full_filename}.gz"
                    
            elif format_type == 'excel':
                df.to_excel(file_path, index=include_index, engine='openpyxl')
                
            elif format_type == 'parquet':
                df.to_parquet(file_path, index=include_index)
                
            elif format_type == 'pdf':
                # Generate PDF export with enhanced formatting
                self._export_to_pdf(df, file_path, metadata, config)
                
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            # Get file size after export
            file_size = os.path.getsize(file_path)
            
            result = {
                'file_path': file_path,
                'filename': full_filename,
                'format': format_type,
                'rows_exported': len(df),
                'columns_exported': len(df.columns),
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'compressed': compress and format_type in ['csv', 'json'],
                'metadata': metadata,
                'export_timestamp': timestamp
            }
            
            self._log_info(f"Exported {len(df)} rows x {len(df.columns)} columns to {full_filename} ({result['file_size_mb']} MB)")
            
            return result
            
        except Exception as e:
            self._log_error(f"Error processing export data: {str(e)}")
            raise
    
    def _export_to_pdf(self, df: pd.DataFrame, file_path: str, metadata: Dict, config: Dict):
        """Export DataFrame to PDF with professional formatting"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from datetime import datetime
            
            # Create the PDF document
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph("Data Export Report", title_style))
            
            # Metadata section
            meta_style = ParagraphStyle(
                'MetaStyle',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6
            )
            
            story.append(Paragraph(f"<b>Export Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
            story.append(Paragraph(f"<b>Data Shape:</b> {df.shape[0]} rows × {df.shape[1]} columns", meta_style))
            story.append(Paragraph(f"<b>Data Type:</b> {metadata.get('data_type', 'Unknown')}", meta_style))
            story.append(Paragraph(f"<b>Source:</b> {metadata.get('source', 'Unknown')}", meta_style))
            
            if 'conversion' in metadata:
                story.append(Paragraph(f"<b>Conversion:</b> {metadata['conversion']}", meta_style))
            
            story.append(Spacer(1, 20))
            
            # Data summary
            story.append(Paragraph("Data Summary", styles['Heading2']))
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                story.append(Paragraph("Numeric Columns Statistics:", styles['Heading3']))
                
                summary_data = [['Column', 'Count', 'Mean', 'Std', 'Min', 'Max']]
                for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                    stats = df[col].describe()
                    summary_data.append([
                        col,
                        f"{stats['count']:.0f}",
                        f"{stats['mean']:.2f}" if not pd.isna(stats['mean']) else 'N/A',
                        f"{stats['std']:.2f}" if not pd.isna(stats['std']) else 'N/A',
                        f"{stats['min']:.2f}" if not pd.isna(stats['min']) else 'N/A',
                        f"{stats['max']:.2f}" if not pd.isna(stats['max']) else 'N/A'
                    ])
                
                summary_table = Table(summary_data)
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(summary_table)
                story.append(Spacer(1, 20))
            
            # Column information
            story.append(Paragraph("Column Information:", styles['Heading3']))
            col_info_data = [['Column Name', 'Data Type', 'Non-Null Count', 'Null Count']]
            for col in df.columns:
                non_null = df[col].count()
                null_count = len(df) - non_null
                col_info_data.append([
                    col,
                    str(df[col].dtype),
                    str(non_null),
                    str(null_count)
                ])
            
            col_info_table = Table(col_info_data)
            col_info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(col_info_table)
            story.append(PageBreak())
            
            # Data preview (first 50 rows)
            story.append(Paragraph("Data Preview (First 50 Rows)", styles['Heading2']))
            
            # Limit the number of columns and rows for PDF display
            max_cols = 8
            max_rows = 50
            
            display_df = df.iloc[:max_rows, :max_cols].copy()
            
            # Convert to string and limit cell length
            for col in display_df.columns:
                display_df[col] = display_df[col].astype(str).apply(lambda x: x[:30] + '...' if len(str(x)) > 30 else str(x))
            
            # Create table data
            table_data = [list(display_df.columns)]
            for _, row in display_df.iterrows():
                table_data.append(list(row))
            
            # Create and style the table
            data_table = Table(table_data)
            data_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(data_table)
            
            if len(df) > max_rows:
                story.append(Spacer(1, 10))
                story.append(Paragraph(f"<i>Note: Showing first {max_rows} of {len(df)} total rows</i>", styles['Normal']))
            
            if len(df.columns) > max_cols:
                story.append(Paragraph(f"<i>Note: Showing first {max_cols} of {len(df.columns)} total columns</i>", styles['Normal']))
            
            # Build the PDF
            doc.build(story)
            
        except ImportError:
            # If reportlab is not installed, create a simple text-based PDF alternative
            self._log_warning("ReportLab not available, creating simple text export instead of PDF")
            
            # Create a simple text file with .pdf extension for now
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("DATA EXPORT REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
                f.write(f"Data Type: {metadata.get('data_type', 'Unknown')}\n")
                f.write(f"Source: {metadata.get('source', 'Unknown')}\n")
                if 'conversion' in metadata:
                    f.write(f"Conversion: {metadata['conversion']}\n")
                f.write("\n" + "=" * 50 + "\n\n")
                
                # Column information
                f.write("COLUMN INFORMATION:\n")
                f.write("-" * 30 + "\n")
                for col in df.columns:
                    non_null = df[col].count()
                    null_count = len(df) - non_null
                    f.write(f"{col}: {df[col].dtype} (Non-null: {non_null}, Null: {null_count})\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
                
                # Data preview
                f.write("DATA PREVIEW (First 20 rows):\n")
                f.write("-" * 30 + "\n")
                f.write(df.head(20).to_string())
                
                if len(df) > 20:
                    f.write(f"\n\n... ({len(df) - 20} more rows)")
        
        except Exception as e:
            self._log_error(f"Error creating PDF export: {str(e)}")
            # Fallback to text export
            with open(file_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
                f.write("DATA EXPORT (PDF generation failed)\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
                f.write("Data Preview:\n")
                f.write(df.head(50).to_string())
            raise Exception(f"PDF export failed, created text file instead: {str(e)}")
    
    # Helper methods for AI insights
    def _generate_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data overview insights"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.value_counts().to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality"""
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        duplicates = df.duplicated().sum()
        
        quality_score = 100
        if duplicates > 0:
            quality_score -= min(20, (duplicates / len(df)) * 100)
        
        for col, pct in missing_pct.items():
            if pct > 50:
                quality_score -= 15
            elif pct > 20:
                quality_score -= 10
            elif pct > 5:
                quality_score -= 5
        
        return {
            'overall_score': max(0, quality_score),
            'missing_percentages': missing_pct,
            'duplicate_rows': duplicates,
            'completeness': f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%"
        }
    
    def _generate_statistical_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical insights"""
        numeric_df = df.select_dtypes(include=[np.number])
        insights = {}
        
        if not numeric_df.empty:
            insights['distributions'] = {}
            for col in numeric_df.columns:
                data = numeric_df[col].dropna()
                if len(data) > 0:
                    insights['distributions'][col] = {
                        'mean': data.mean(),
                        'median': data.median(),
                        'std': data.std(),
                        'skewness': data.skew(),
                        'kurtosis': data.kurtosis()
                    }
        
        return insights
    
    def _analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual features"""
        analysis = {}
        
        for col in df.columns:
            col_analysis = {
                'type': str(df[col].dtype),
                'unique_values': df[col].nunique(),
                'missing_count': df[col].isnull().sum()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_analysis.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median()
                })
            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts().head(5)
                col_analysis['top_values'] = value_counts.to_dict()
            
            analysis[col] = col_analysis
        
        return analysis
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect interesting patterns in the data"""
        patterns = []
        
        # High cardinality categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                patterns.append(f"Column '{col}' has very high cardinality ({unique_ratio:.1%} unique values)")
        
        # Highly correlated numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.8:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_val))
            
            for col1, col2, corr in high_corr_pairs:
                patterns.append(f"High correlation between '{col1}' and '{col2}' ({corr:.2f})")
        
        return patterns
    
    def _generate_ml_recommendations(self, df: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """Generate ML model recommendations"""
        recommendations = {}
        
        if target_variable in df.columns:
            target_type = 'classification' if df[target_variable].dtype == 'object' else 'regression'
            
            recommendations['problem_type'] = target_type
            recommendations['recommended_algorithms'] = []
            
            if target_type == 'classification':
                unique_classes = df[target_variable].nunique()
                if unique_classes == 2:
                    recommendations['recommended_algorithms'] = [
                        'Logistic Regression', 'Random Forest', 'SVM'
                    ]
                else:
                    recommendations['recommended_algorithms'] = [
                        'Random Forest', 'Gradient Boosting', 'Multi-class SVM'
                    ]
            else:
                recommendations['recommended_algorithms'] = [
                    'Linear Regression', 'Random Forest', 'Gradient Boosting'
                ]
            
            # Feature recommendations
            numeric_features = len(df.select_dtypes(include=[np.number]).columns) - 1  # Exclude target
            categorical_features = len(df.select_dtypes(include=['object']).columns)
            
            recommendations['feature_engineering'] = []
            if categorical_features > 0:
                recommendations['feature_engineering'].append('One-hot encoding for categorical variables')
            if numeric_features > 0:
                recommendations['feature_engineering'].append('Feature scaling (StandardScaler)')
        
        return recommendations
    
    def _perform_advanced_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced statistical analysis"""
        analysis = {}
        
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            # Principal Component Analysis
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_df.fillna(numeric_df.mean()))
                pca = PCA()
                pca.fit(scaled_data)
                
                analysis['pca'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_[:5].tolist(),
                    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)[:5].tolist()
                }
            except Exception:
                analysis['pca'] = {'error': 'Could not perform PCA analysis'}
        
        return analysis
    
    def _generate_distribution_plots(self, df: pd.DataFrame) -> List[Dict]:
        """Generate distribution plots for numeric columns"""
        charts = []
        
        try:
            # Get numeric columns safely
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Limit to first 5 columns to avoid generating too many charts
            for col in numeric_cols[:5]:
                try:
                    # Get data safely
                    col_data = df[col].dropna().values
                    
                    # Skip if not enough data points
                    if len(col_data) < 5:
                        continue
                        
                    fig = plt.figure(figsize=(8, 6))
                    plt.hist(col_data, bins=30, alpha=0.7, edgecolor='black')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.getvalue()).decode()
                    plt.close()
                    
                    charts.append({
                        'type': 'histogram',
                        'column': col,
                        'image': plot_data
                    })
                except Exception as e:
                    self._log_warning(f"Error generating histogram for column {col}: {str(e)}")
                    continue
        except Exception as e:
            self._log_warning(f"Error in distribution plot generation: {str(e)}")
            
        return charts
    
    def _convert_to_json_serializable(self, obj):
        """Convert pandas objects and numpy types to JSON-serializable format"""
        if obj is None:
            return None
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                elif isinstance(value, (np.integer, np.floating)):
                    result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif pd.isna(value):
                    result[key] = None
                else:
                    result[key] = value
            return result
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    # Placeholder methods for other node types
    def _process_api_source(self, node: Dict, input_data: Dict) -> Any:
        """Process API source node - placeholder"""
        raise NotImplementedError("API source not yet implemented")
    
    def _process_database_source(self, node: Dict, input_data: Dict) -> Any:
        """Process database source node - placeholder"""
        raise NotImplementedError("Database source not yet implemented")
    
    def _process_data_cleaning(self, node: Dict, input_data: Dict) -> Any:
        """Process data cleaning node with comprehensive cleaning operations"""
        try:
            if 'default' not in input_data:
                raise ValueError("Data cleaning node requires input data")
            
            # Handle different input data formats
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj.copy()
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                df = input_obj['data'].copy()
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            original_shape = df.shape
            
            # Initialize cleaning operations log
            cleaning_operations = []
            
            # 1. Remove duplicate rows
            if config.get('remove_duplicates', True):
                duplicates_before = df.duplicated().sum()
                df = df.drop_duplicates()
                duplicates_removed = duplicates_before
                if duplicates_removed > 0:
                    cleaning_operations.append(f"Removed {duplicates_removed} duplicate rows")
            
            # 2. Handle missing values
            missing_strategy = config.get('missing_strategy', 'drop')
            missing_threshold = config.get('missing_threshold', 0.5)
            
            # Remove columns with too many missing values
            missing_cols = []
            for col in df.columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > missing_threshold:
                    missing_cols.append(col)
            
            if missing_cols:
                df = df.drop(columns=missing_cols)
                cleaning_operations.append(f"Removed {len(missing_cols)} columns with >{missing_threshold*100}% missing values: {missing_cols}")
            
            # Handle remaining missing values
            if missing_strategy == 'drop':
                rows_before = len(df)
                df = df.dropna()
                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    cleaning_operations.append(f"Dropped {rows_dropped} rows with missing values")
            
            elif missing_strategy == 'fill':
                for col in df.columns:
                    if df[col].isnull().any():
                        if df[col].dtype in ['object', 'category']:
                            # Fill categorical with mode
                            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                            df[col].fillna(mode_val, inplace=True)
                            cleaning_operations.append(f"Filled missing values in '{col}' with mode: '{mode_val}'")
                        else:
                            # Fill numerical with median
                            median_val = df[col].median()
                            df[col].fillna(median_val, inplace=True)
                            cleaning_operations.append(f"Filled missing values in '{col}' with median: {median_val:.2f}")
            
            # 3. Remove outliers (for numerical columns only)
            if config.get('remove_outliers', False):
                outlier_method = config.get('outlier_method', 'iqr')
                outlier_threshold = config.get('outlier_threshold', 1.5)
                
                for col in df.select_dtypes(include=[np.number]).columns:
                    if outlier_method == 'iqr':
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - outlier_threshold * IQR
                        upper_bound = Q3 + outlier_threshold * IQR
                        
                        outliers_before = len(df)
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        outliers_removed = outliers_before - len(df)
                        
                        if outliers_removed > 0:
                            cleaning_operations.append(f"Removed {outliers_removed} outliers from '{col}' using IQR method")
                    
                    elif outlier_method == 'zscore':
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        outliers_before = len(df)
                        df = df[z_scores <= outlier_threshold]
                        outliers_removed = outliers_before - len(df)
                        
                        if outliers_removed > 0:
                            cleaning_operations.append(f"Removed {outliers_removed} outliers from '{col}' using Z-score method")
            
            # 4. Data type optimization
            if config.get('optimize_dtypes', True):
                # Convert object columns that look like numbers
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        # Try to convert to numeric
                        numeric_col = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_col.isnull().all():
                            df[col] = numeric_col
                            cleaning_operations.append(f"Converted '{col}' from object to numeric")
                    except:
                        pass
                
                # Optimize integer columns
                for col in df.select_dtypes(include=['int64']).columns:
                    if df[col].min() >= 0:
                        if df[col].max() <= 255:
                            df[col] = df[col].astype('uint8')
                        elif df[col].max() <= 65535:
                            df[col] = df[col].astype('uint16')
                        elif df[col].max() <= 4294967295:
                            df[col] = df[col].astype('uint32')
                    else:
                        if df[col].min() >= -128 and df[col].max() <= 127:
                            df[col] = df[col].astype('int8')
                        elif df[col].min() >= -32768 and df[col].max() <= 32767:
                            df[col] = df[col].astype('int16')
                        elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                            df[col] = df[col].astype('int32')
            
            # 5. Text cleaning for string columns
            if config.get('clean_text', True):
                text_columns = df.select_dtypes(include=['object']).columns
                for col in text_columns:
                    if df[col].dtype == 'object':
                        # Remove leading/trailing whitespace
                        df[col] = df[col].astype(str).str.strip()
                        # Convert to lowercase if specified
                        if config.get('lowercase_text', False):
                            df[col] = df[col].str.lower()
                        cleaning_operations.append(f"Cleaned text in column '{col}'")
            
            final_shape = df.shape
            
            # Generate cleaning summary
            cleaning_summary = {
                'original_shape': original_shape,
                'final_shape': final_shape,
                'rows_removed': original_shape[0] - final_shape[0],
                'columns_removed': original_shape[1] - final_shape[1],
                'operations_performed': cleaning_operations,
                'data_quality_score': self._calculate_data_quality_score(df)
            }
            
            return {
                'data': df,
                'cleaning_summary': cleaning_summary,
                'type': 'cleaned_data'
            }
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate a data quality score (0-100)"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            
            # Base score from completeness
            completeness_score = ((total_cells - missing_cells) / total_cells) * 50
            
            # Consistency score (no duplicates)
            duplicates = df.duplicated().sum()
            consistency_score = ((df.shape[0] - duplicates) / df.shape[0]) * 25
            
            # Validity score (proper data types)
            validity_score = 25  # Default good score for now
            
            return min(100, completeness_score + consistency_score + validity_score)
        except:
            return 50.0  # Default score if calculation fails
    
    def _process_feature_engineering(self, node: Dict, input_data: Dict) -> Any:
        """Process feature engineering node with comprehensive feature creation and transformation"""
        try:
            if 'default' not in input_data:
                raise ValueError("Feature engineering node requires input data")
            
            # Handle different input data formats
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj.copy()
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                df = input_obj['data'].copy()
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            original_shape = df.shape
            
            # Initialize feature engineering operations log
            engineering_operations = []
            
            # 1. Handle categorical variables
            categorical_encoding = config.get('categorical_encoding', 'auto')
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols and categorical_encoding != 'none':
                for col in categorical_cols:
                    unique_vals = df[col].nunique()
                    
                    if categorical_encoding == 'auto':
                        # Use one-hot for low cardinality, label encoding for high cardinality
                        if unique_vals <= 10:
                            encoding_method = 'onehot'
                        else:
                            encoding_method = 'label'
                    else:
                        encoding_method = categorical_encoding
                    
                    if encoding_method == 'onehot':
                        # One-hot encoding
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        df = pd.concat([df, dummies], axis=1)
                        df = df.drop(columns=[col])
                        engineering_operations.append(f"Applied one-hot encoding to '{col}' ({unique_vals} categories)")
                    
                    elif encoding_method == 'label':
                        # Label encoding
                        le = LabelEncoder()
                        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                        df = df.drop(columns=[col])
                        engineering_operations.append(f"Applied label encoding to '{col}' ({unique_vals} categories)")
            
            # 2. Feature scaling
            scaling_method = config.get('scaling_method', 'standard')
            
            if scaling_method != 'none':
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numerical_cols:
                    if scaling_method == 'standard':
                        scaler = StandardScaler()
                        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                        engineering_operations.append(f"Applied StandardScaler to {len(numerical_cols)} numerical columns")
                    
                    elif scaling_method == 'minmax':
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                        engineering_operations.append(f"Applied MinMaxScaler to {len(numerical_cols)} numerical columns")
                    
                    elif scaling_method == 'robust':
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                        engineering_operations.append(f"Applied RobustScaler to {len(numerical_cols)} numerical columns")
            
            # 3. Create polynomial features
            if config.get('create_polynomial_features', False):
                poly_degree = config.get('polynomial_degree', 2)
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numerical_cols) >= 2:
                    # Limit to first few columns to avoid explosion
                    cols_to_use = numerical_cols[:min(5, len(numerical_cols))]
                    
                    poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=True)
                    poly_features = poly.fit_transform(df[cols_to_use])
                    
                    # Create feature names
                    feature_names = poly.get_feature_names_out(cols_to_use)
                    
                    # Add polynomial features to dataframe
                    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
                    
                    # Remove original columns and add polynomial features
                    df = df.drop(columns=cols_to_use)
                    df = pd.concat([df, poly_df], axis=1)
                    
                    engineering_operations.append(f"Created {len(feature_names)} polynomial features (degree {poly_degree})")
            
            # 4. Create binned features
            if config.get('create_binned_features', False):
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                bins_count = config.get('bins_count', 5)
                
                for col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                    try:
                        df[f'{col}_binned'] = pd.cut(df[col], bins=bins_count, labels=False, duplicates='drop')
                        engineering_operations.append(f"Created binned feature for '{col}' with {bins_count} bins")
                    except:
                        pass  # Skip if binning fails (e.g., all values are the same)
            
            # 5. Create statistical features
            if config.get('create_statistical_features', False):
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numerical_cols) >= 2:
                    # Create row-wise statistics
                    df['row_mean'] = df[numerical_cols].mean(axis=1)
                    df['row_std'] = df[numerical_cols].std(axis=1)
                    df['row_max'] = df[numerical_cols].max(axis=1)
                    df['row_min'] = df[numerical_cols].min(axis=1)
                    df['row_range'] = df['row_max'] - df['row_min']
                    
                    engineering_operations.append("Created row-wise statistical features (mean, std, max, min, range)")
            
            # 6. Create datetime features (if datetime columns exist)
            if config.get('create_datetime_features', True):
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            # Try to convert to datetime
                            dt_col = pd.to_datetime(df[col], errors='coerce')
                            if not dt_col.isnull().all():
                                df[f'{col}_year'] = dt_col.dt.year
                                df[f'{col}_month'] = dt_col.dt.month
                                df[f'{col}_day'] = dt_col.dt.day
                                df[f'{col}_dayofweek'] = dt_col.dt.dayofweek
                                df[f'{col}_hour'] = dt_col.dt.hour
                                df = df.drop(columns=[col])
                                engineering_operations.append(f"Extracted datetime features from '{col}'")
                        except:
                            pass
            
            # 7. Create interaction features
            if config.get('create_interaction_features', False):
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numerical_cols) >= 2:
                    # Create interactions between first few columns
                    for i in range(min(3, len(numerical_cols))):
                        for j in range(i+1, min(3, len(numerical_cols))):
                            col1, col2 = numerical_cols[i], numerical_cols[j]
                            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)  # Add small epsilon to avoid division by zero
                    
                    engineering_operations.append("Created interaction features (multiplication and division)")
            
            # 8. Remove constant features
            if config.get('remove_constant_features', True):
                constant_cols = []
                for col in df.columns:
                    if df[col].nunique() <= 1:
                        constant_cols.append(col)
                
                if constant_cols:
                    df = df.drop(columns=constant_cols)
                    engineering_operations.append(f"Removed {len(constant_cols)} constant features")
            
            # 9. Feature selection based on correlation
            if config.get('remove_highly_correlated', False):
                correlation_threshold = config.get('correlation_threshold', 0.95)
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numerical_cols) > 1:
                    corr_matrix = df[numerical_cols].corr().abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    
                    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
                    
                    if to_drop:
                        df = df.drop(columns=to_drop)
                        engineering_operations.append(f"Removed {len(to_drop)} highly correlated features (threshold: {correlation_threshold})")
            
            final_shape = df.shape
            
            # Generate feature engineering summary
            engineering_summary = {
                'original_shape': original_shape,
                'final_shape': final_shape,
                'features_added': final_shape[1] - original_shape[1],
                'operations_performed': engineering_operations,
                'feature_types': {
                    'numerical': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical': len(df.select_dtypes(include=['object', 'category']).columns),
                    'boolean': len(df.select_dtypes(include=['bool']).columns)
                }
            }
            
            return {
                'data': df,
                'engineering_summary': engineering_summary,
                'type': 'engineered_data'
            }
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def _process_data_validation(self, node: Dict, input_data: Dict) -> Any:
        """Process data validation node - placeholder"""
        if 'default' not in input_data:
            raise ValueError("Data validation node requires input data")
        return input_data['default']  # Return unchanged for now
    
    def _process_correlation_analysis(self, node: Dict, input_data: Dict) -> Any:
        """Process correlation analysis node - placeholder"""
        raise NotImplementedError("Correlation analysis not yet implemented")
    
    def _process_hypothesis_testing(self, node: Dict, input_data: Dict) -> Any:
        """Process hypothesis testing node - placeholder"""
        raise NotImplementedError("Hypothesis testing not yet implemented")
    
    def _process_regression(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process regression node with advanced column selection"""
        try:
            if 'default' not in input_data:
                raise ValueError("Regression node requires input data")
            
            # Handle different input data formats
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                # Handle output from descriptive stats node
                df = input_obj['data']
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            
            # Advanced column analysis for regression
            column_analysis = self._analyze_columns_for_regression(df)
            
            # Auto-select target column if not specified
            target_column = config.get('target_column')
            if not target_column or target_column == 'None' or target_column not in df.columns:
                auto_target = self._auto_select_regression_target(df, column_analysis)
                if auto_target:
                    target_column = auto_target
                    self._log_info(f"Auto-selected regression target: '{target_column}'")
                else:
                    raise ValueError("No suitable numeric target column found for regression. Please specify a continuous numeric target column.")
            
            # Validate target column for regression
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
            
            if not pd.api.types.is_numeric_dtype(df[target_column]):
                raise ValueError(f"Target column '{target_column}' must be numeric for regression. Found type: {df[target_column].dtype}")
            
            # Auto-select feature columns if not specified
            feature_columns = config.get('feature_columns', [])
            if not feature_columns:
                feature_columns = self._auto_select_regression_features(df, target_column, column_analysis)
                self._log_info(f"Auto-selected {len(feature_columns)} feature columns for regression")
            
            # Validate and filter feature columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            valid_features = [col for col in feature_columns if col in numeric_cols and col != target_column]
            
            if not valid_features:
                # Fallback: use all numeric columns except target
                valid_features = [col for col in numeric_cols if col != target_column]
                if not valid_features:
                    raise ValueError("No valid numeric feature columns found for regression")
            
            X = df[valid_features].copy()
            y = df[target_column].copy()
            
            # Advanced preprocessing
            X_processed = self._preprocess_features_for_regression(X, config)
            y_processed = self._preprocess_target_for_regression(y, target_column)
            
            # Split data
            test_size = config.get('test_size', 0.2)
            random_state = config.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=random_state)
            
            # Select model
            algorithm = config.get('algorithm', 'linear')
            polynomial_degree = config.get('polynomial_degree', 2)
            alpha = config.get('alpha', 1.0)
            
            if algorithm == 'linear':
                model = LinearRegression()
            elif algorithm == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            elif algorithm == 'gradient_boosting':
                model = GradientBoostingRegressor(random_state=random_state)
            elif algorithm == 'polynomial':
                poly_features = PolynomialFeatures(degree=polynomial_degree)
                model = Pipeline([
                    ('poly', poly_features),
                    ('linear', LinearRegression())
                ])
            elif algorithm == 'ridge':
                model = Ridge(alpha=alpha)
            elif algorithm == 'lasso':
                model = Lasso(alpha=alpha)
            else:
                raise ValueError(f"Unsupported regression algorithm: {algorithm}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Store model for later use
            model_id = f"model_{node['id']}"
            self.model_cache[model_id] = {
                'model': model,
                'feature_columns': list(X_processed.columns),
                'target_column': target_column,
                'algorithm': algorithm
            }
            
            # Get feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_.tolist()
            elif algorithm == 'polynomial' and hasattr(model.named_steps['linear'], 'coef_'):
                # For polynomial, get coefficients from the linear part
                feature_importance = np.abs(model.named_steps['linear'].coef_).tolist()
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_).tolist()
            
            result = {
                'model_id': model_id,
                'algorithm': algorithm,
                'target_column': target_column,
                'feature_columns': list(X_processed.columns),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'metrics': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': np.sqrt(train_mse),
                    'test_rmse': np.sqrt(test_mse),
                    'train_mae': train_mae,
                    'test_mae': test_mae
                },
                'predictions': {
                    'train': y_pred_train.tolist(),
                    'test': y_pred_test.tolist(),
                    'actual_train': y_train.tolist(),
                    'actual_test': y_test.tolist()
                },
                'feature_importance': feature_importance,
                'feature_names': list(X_processed.columns)
            }
            
            # Generate prediction plots
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Actual vs Predicted
                ax1.scatter(y_test, y_pred_test, alpha=0.7)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax1.set_xlabel('Actual')
                ax1.set_ylabel('Predicted')
                ax1.set_title(f'Actual vs Predicted (R² = {test_r2:.3f})')
                
                # Residuals plot
                residuals = y_test - y_pred_test
                ax2.scatter(y_pred_test, residuals, alpha=0.7)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Residuals Plot')
                
                plt.tight_layout()
                result['regression_plots'] = self._save_plot_as_base64(fig)
                plt.close(fig)
                
            except Exception as plot_error:
                self._log_error(f"Error creating regression plots: {str(plot_error)}")
            
            self._log_info(f"Trained {algorithm} regression model with R² = {test_r2:.3f}")
            return result
            
        except Exception as e:
            self._log_error(f"Error processing regression: {str(e)}")
            raise
    
    def _process_time_series(self, node: Dict, input_data: Dict) -> Any:
        """Process time series node - placeholder"""
        raise NotImplementedError("Time series not yet implemented")
    
    def _process_clustering(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process clustering node - Perform clustering analysis"""
        try:
            if 'default' not in input_data:
                raise ValueError("Clustering node requires input data")
            
            # Handle different input data formats
            input_obj = input_data['default']
            self._log_info(f"Clustering input type: {type(input_obj)}")
            
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
                self._log_info("Using DataFrame directly")
            elif isinstance(input_obj, dict):
                self._log_info(f"Input dict keys: {list(input_obj.keys())}")
                if 'data' in input_obj:
                    df = input_obj['data']
                    self._log_info(f"Extracted data type: {type(df)}")
                else:
                    # Try to find DataFrame in any key
                    df_candidates = []
                    for key, value in input_obj.items():
                        if isinstance(value, pd.DataFrame):
                            df_candidates.append((key, value))
                    
                    if df_candidates:
                        key, df = df_candidates[0]  # Use first DataFrame found
                        self._log_info(f"Found DataFrame in key '{key}'")
                    else:
                        available_keys = list(input_obj.keys())
                        available_types = {k: str(type(v)) for k, v in input_obj.items()}
                        raise ValueError(f"Dict input must contain 'data' key with DataFrame. Available keys: {available_keys}. Types: {available_types}")
            else:
                raise ValueError(f"Expected DataFrame or dict with DataFrame, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            
            # Get feature columns with improved handling
            feature_columns = config.get('feature_columns', [])
            features = config.get('features', '')  # Legacy support for old text-based format
            
            if feature_columns:
                # New multi-column select format
                if isinstance(feature_columns, list):
                    feature_cols = feature_columns
                else:
                    feature_cols = [feature_columns] if feature_columns else []
            elif features:
                # Legacy text-based format
                if isinstance(features, str):
                    feature_cols = [col.strip() for col in features.split(',') if col.strip()]
                else:
                    feature_cols = features if isinstance(features, list) else [features]
            else:
                # Auto-select all numeric columns (excluding identifiers)
                feature_cols = []
            
            # If specific columns provided, validate they exist and are numeric
            if feature_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in feature_cols if col in df.columns and col in numeric_cols]
                
                if not feature_cols:
                    raise ValueError("No valid numeric feature columns found in selection")
            else:
                # Auto-select numeric columns (excluding identifiers like id, timestamp)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols 
                              if not any(keyword in col.lower() for keyword in ['id', 'index', 'key', 'uuid', 'timestamp'])]
                
                if not feature_cols:
                    # Fallback to all numeric columns if no suitable ones found
                    feature_cols = numeric_cols
            
            if len(feature_cols) == 0:
                raise ValueError("No numeric columns found for clustering")
            
            X = df[feature_cols].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale the data if requested
            scale_features = config.get('scale_features', True)
            if scale_features:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
                scaler = None
            
            algorithm = config.get('algorithm', 'kmeans')
            
            if algorithm == 'kmeans':
                n_clusters = config.get('n_clusters', 3)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(X_scaled)
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
                
                result = {
                    'algorithm': 'K-Means',
                    'n_clusters': n_clusters,
                    'labels': labels.tolist(),
                    'inertia': model.inertia_,
                    'silhouette_score': silhouette_avg,
                    'n_samples': len(X)
                }
                
                # Add cluster centers (inverse transformed if scaled)
                if scaler:
                    cluster_centers = scaler.inverse_transform(model.cluster_centers_)
                else:
                    cluster_centers = model.cluster_centers_
                result['cluster_centers'] = cluster_centers.tolist()
                
            elif algorithm == 'dbscan':
                eps = config.get('eps', 0.5)
                min_samples = config.get('min_samples', 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Calculate silhouette score (excluding noise points)
                silhouette_avg = 0
                if n_clusters > 1:
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette_avg = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
                
                result = {
                    'algorithm': 'DBSCAN',
                    'eps': eps,
                    'min_samples': min_samples,
                    'labels': labels.tolist(),
                    'n_clusters': n_clusters,
                    'n_noise_points': n_noise,
                    'silhouette_score': silhouette_avg,
                    'n_samples': len(X)
                }
                
            elif algorithm == 'hierarchical':
                n_clusters = config.get('n_clusters', 3)
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X_scaled)
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
                
                result = {
                    'algorithm': 'Hierarchical',
                    'n_clusters': n_clusters,
                    'labels': labels.tolist(),
                    'silhouette_score': silhouette_avg,
                    'n_samples': len(X)
                }
                
            elif algorithm == 'gaussian_mixture':
                n_clusters = config.get('n_clusters', 3)
                model = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = model.fit_predict(X_scaled)
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
                
                result = {
                    'algorithm': 'Gaussian Mixture',
                    'n_clusters': n_clusters,
                    'labels': labels.tolist(),
                    'silhouette_score': silhouette_avg,
                    'aic': model.aic(X_scaled),
                    'bic': model.bic(X_scaled),
                    'n_samples': len(X)
                }
                
            else:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
            
            # Store model for later use
            model_id = f"model_{node['id']}"
            self.model_cache[model_id] = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_cols,
                'algorithm': algorithm
            }
            
            # Add clustering results to DataFrame
            df_result = df.copy()
            df_result['cluster'] = labels
            
            # Generate cluster visualization if we have enough features
            if len(feature_cols) >= 2:
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Use first two features for visualization
                    x_col, y_col = feature_cols[0], feature_cols[1]
                    scatter = ax.scatter(df[x_col], df[y_col], c=labels, cmap='viridis', alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'{algorithm.upper()} Clustering Results')
                    
                    # Add cluster centers for kmeans
                    if algorithm == 'kmeans' and 'cluster_centers' in result:
                        centers = result['cluster_centers']
                        ax.scatter([c[0] for c in centers], [c[1] for c in centers], 
                                  c='red', marker='x', s=200, linewidth=3, label='Centroids')
                        ax.legend()
                    
                    plt.colorbar(scatter, label='Cluster')
                    plt.tight_layout()
                    
                    # Cache the plot
                    plot_b64 = self._save_plot_as_base64(fig)
                    self.chart_cache[node['id']] = {'cluster_plot': plot_b64}
                    result['cluster_plot'] = plot_b64
                    plt.close(fig)
                    
                except Exception as plot_error:
                    self._log_error(f"Error creating cluster plot: {str(plot_error)}")
            
            # Add additional result metadata
            result.update({
                'model_id': model_id,
                'data': df_result,
                'feature_columns_used': feature_cols,
                'scaled_features': scale_features,
                'cluster_distribution': {str(i): int(np.sum(labels == i)) for i in np.unique(labels)}
            })
            
            self._log_info(f"Performed {algorithm} clustering with {result.get('n_clusters', 'variable')} clusters, silhouette score: {result.get('silhouette_score', 'N/A'):.3f}")
            return result
            
        except Exception as e:
            self._log_error(f"Error processing clustering: {str(e)}")
            raise
    
    def _process_dimensionality_reduction(self, node: Dict, input_data: Dict) -> Any:
        """Process dimensionality reduction node - placeholder"""
        raise NotImplementedError("Dimensionality reduction not yet implemented")
    
    def _process_anomaly_detection(self, node: Dict, input_data: Dict) -> Any:
        """Process anomaly detection node - placeholder"""
        raise NotImplementedError("Anomaly detection not yet implemented")
    
    def _process_neural_network(self, node: Dict, input_data: Dict) -> Any:
        """Process neural network node - placeholder"""
        raise NotImplementedError("Neural network not yet implemented")
    
    def _process_cnn(self, node: Dict, input_data: Dict) -> Any:
        """Process CNN node - placeholder"""
        raise NotImplementedError("CNN not yet implemented")
    
    def _process_rnn_lstm(self, node: Dict, input_data: Dict) -> Any:
        """Process RNN/LSTM node - placeholder"""
        raise NotImplementedError("RNN/LSTM not yet implemented")
    
    def _process_model_evaluation(self, node: Dict, input_data: Dict) -> Any:
        """Process model evaluation node - placeholder"""
        raise NotImplementedError("Model evaluation not yet implemented")
    
    def _process_feature_importance(self, node: Dict, input_data: Dict) -> Any:
        """Process feature importance node - placeholder"""
        raise NotImplementedError("Feature importance not yet implemented")
    
    def _process_model_comparison(self, node: Dict, input_data: Dict) -> Any:
        """Process model comparison node - placeholder"""
        raise NotImplementedError("Model comparison not yet implemented")
    
    def _process_advanced_plots(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process advanced plots node with automatic feature detection"""
        try:
            if 'default' not in input_data:
                raise ValueError("Advanced plots node requires input data")
            
            # Handle different input data formats
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                # Handle output from descriptive stats node
                df = input_obj['data']
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            
            plot_type = config.get('plot_type', 'heatmap')
            features = config.get('features', '')
            auto_detect = config.get('auto_detect_features', True)
            interactive = config.get('interactive', True)
            color_palette = config.get('color_palette', 'viridis')
            
            charts = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Auto-detect features if not provided
            if auto_detect and not features:
                if plot_type == 'heatmap':
                    features = numeric_cols[:10]  # Limit to first 10 for readability
                elif plot_type == 'pairplot':
                    features = numeric_cols[:5]   # Limit to first 5 for performance
                elif plot_type == 'parallel_coordinates':
                    features = numeric_cols[:6]   # Limit to first 6
                elif plot_type == '3d_scatter':
                    features = numeric_cols[:3]   # Need exactly 3 for 3D
                elif plot_type == 'violin_plot':
                    features = numeric_cols[:5]   # Limit to first 5
            elif features:
                # Parse features from string or list
                if isinstance(features, str):
                    features = [col.strip() for col in features.split(',') if col.strip()]
                
                # Filter to only include existing columns
                features = [col for col in features if col in df.columns]
            
            if not features:
                features = numeric_cols[:5]
                features = numeric_cols[:5] if numeric_cols else []
            
            self._log_info(f"Using features for {plot_type}: {features}")
            
            try:
                if plot_type == 'heatmap' and len(features) > 1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    # Use only numeric features for correlation heatmap
                    numeric_features = [col for col in features if col in numeric_cols]
                    if numeric_features:
                        corr = df[numeric_features].corr()
                        sns.heatmap(corr, annot=True, cmap=color_palette, center=0, ax=ax,
                                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
                        plt.title('Correlation Heatmap')
                        plt.tight_layout()
                        charts['correlation_heatmap'] = self._save_plot_as_base64(fig)
                        plt.close(fig)
                
                elif plot_type == 'pairplot' and len(features) >= 2:
                    # Create pairwise scatter plots
                    n_features = len(features)
                    fig, axes = plt.subplots(n_features, n_features, figsize=(3*n_features, 3*n_features))
                    
                    for i, col1 in enumerate(features):
                        for j, col2 in enumerate(features):
                            ax = axes[i, j] if n_features > 1 else axes
                            if i == j:
                                # Diagonal: histogram
                                ax.hist(df[col1].dropna(), bins=20, alpha=0.7, color='skyblue')
                                ax.set_title(f'{col1}')
                            else:
                                # Off-diagonal: scatter plot
                                ax.scatter(df[col2], df[col1], alpha=0.6, s=10)
                                ax.set_xlabel(col2)
                                ax.set_ylabel(col1)
                    
                    plt.tight_layout()
                    charts['pairplot'] = self._save_plot_as_base64(fig)
                    plt.close(fig)
                
                elif plot_type == 'violin_plot' and features:
                    fig, axes = plt.subplots(1, len(features), figsize=(4*len(features), 6))
                    if len(features) == 1:
                        axes = [axes]
                    
                    for i, feature in enumerate(features):
                        if feature in numeric_cols:
                            parts = axes[i].violinplot([df[feature].dropna()], positions=[1])
                            axes[i].set_title(f'Distribution of {feature}')
                            axes[i].set_xticks([1])
                            axes[i].set_xticklabels([feature])
                    
                    plt.tight_layout()
                    charts['violin_plot'] = self._save_plot_as_base64(fig)
                    plt.close(fig)
                
                elif plot_type == '3d_scatter' and len(features) >= 3:
                    from mpl_toolkits.mplot3d import Axes3D
                    fig = plt.figure(figsize=(12, 9))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    x, y, z = features[0], features[1], features[2]
                    scatter = ax.scatter(df[x], df[y], df[z], alpha=0.6, c=df[x], cmap=color_palette)
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)
                    ax.set_title('3D Scatter Plot')
                    plt.colorbar(scatter)
                    
                    charts['3d_scatter'] = self._save_plot_as_base64(fig)
                    plt.close(fig)
                
                elif plot_type == 'parallel_coordinates' and len(features) >= 2:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Normalize the data for parallel coordinates
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[features].fillna(df[features].mean()))
                    scaled_df = pd.DataFrame(scaled_data, columns=features)
                    
                    # Create parallel coordinates plot
                    for i in range(len(scaled_df)):
                        ax.plot(range(len(features)), scaled_df.iloc[i], alpha=0.1, color='blue')
                    
                    ax.set_xticks(range(len(features)))
                    ax.set_xticklabels(features, rotation=45)
                    ax.set_ylabel('Standardized Values')
                    ax.set_title('Parallel Coordinates Plot')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    charts['parallel_coordinates'] = self._save_plot_as_base64(fig)
                    plt.close(fig)
                
                # Add a comprehensive overview plot
                if len(numeric_cols) > 0:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                    
                    # Statistics summary
                    stats_df = df[numeric_cols].describe()
                    ax1.table(cellText=stats_df.round(2).values,
                             rowLabels=stats_df.index,
                             colLabels=stats_df.columns,
                             cellLoc='center',
                             loc='center')
                    ax1.axis('off')
                    ax1.set_title('Statistical Summary')
                    
                    # Missing values
                    missing = df[numeric_cols].isnull().sum()
                    if missing.sum() > 0:
                        missing.plot(kind='bar', ax=ax2)
                        ax2.set_title('Missing Values by Column')
                        ax2.set_ylabel('Missing Count')
                        plt.setp(ax2.get_xticklabels(), rotation=45)
                    else:
                        ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title('Missing Values')
                    
                    # Data types
                    dtype_counts = df.dtypes.value_counts()
                    ax3.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
                    ax3.set_title('Data Types Distribution')
                    
                    # Sample data preview
                    sample_text = df.head().to_string()
                    ax4.text(0.05, 0.95, sample_text, transform=ax4.transAxes, fontsize=8,
                            verticalalignment='top', fontfamily='monospace')
                    ax4.set_title('Data Preview (First 5 Rows)')
                    ax4.axis('off')
                    
                    plt.tight_layout()
                    charts['overview'] = self._save_plot_as_base64(fig)
                    plt.close(fig)
                
            except Exception as plot_error:
                self._log_error(f"Error creating {plot_type}: {str(plot_error)}")
                # Create a simple fallback plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'Error creating {plot_type}:\n{str(plot_error)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Error in {plot_type}')
                charts[f'{plot_type}_error'] = self._save_plot_as_base64(fig)
                plt.close(fig)
            
            result = {
                'charts': charts,
                'chart_count': len(charts),
                'plot_type': plot_type,
                'features_used': features,
                'numeric_columns_available': numeric_cols,
                'categorical_columns_available': categorical_cols,
                'data': df  # Preserve the original DataFrame for downstream nodes
            }
            
            # Cache the charts for summary counting
            self.chart_cache[node['id']] = charts
            
            self._log_info(f"Generated {len(charts)} advanced plots using features: {features}")
            return result
            
        except Exception as e:
            self._log_error(f"Error processing advanced plots: {str(e)}")
            raise
    
    def _process_dashboard(self, node: Dict, input_data: Dict) -> Any:
        """Process dashboard node - placeholder"""
        raise NotImplementedError("Dashboard not yet implemented")
    
    def _analyze_all_connected_nodes(self, input_data: Dict) -> Dict[str, Any]:
        """Analyze all connected nodes and collect their outputs for comprehensive AI analysis"""
        try:
            analysis = {
                'total_nodes': 0,
                'node_types': [],
                'dataframes_count': 0,
                'models_count': 0,
                'statistics_count': 0,
                'charts_count': 0,
                'data_sources': [],
                'primary_data_shape': None,
                'total_memory_usage': 0,
                'has_valid_data': False,
                'node_outputs': {}
            }
            
            self._log_info(f"Analyzing connected nodes input_data: {type(input_data)}")
            self._log_info(f"Input data keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
            
            if not isinstance(input_data, dict):
                self._log_warning("Input data is not a dictionary, returning empty analysis")
                return analysis
            
            # Analyze each connected node's output
            for node_id, node_output in input_data.items():
                try:
                    self._log_info(f"Analyzing node {node_id}: type={type(node_output)}")
                    
                    analysis['total_nodes'] += 1
                    analysis['node_outputs'][node_id] = {}
                    
                    # Check if this is the new meaningful insights format
                    if isinstance(node_output, dict) and 'insights' in node_output:
                        # New meaningful insights format: {'node_name': 'X', 'node_type': 'Y', 'insights': {...}}
                        node_type = node_output.get('node_type', 'unknown')
                        node_name = node_output.get('node_name', f'Node {node_id}')
                        insights = node_output.get('insights', {})
                        
                        self._log_info(f"Processing meaningful insights from {node_name} ({node_type})")
                        
                        analysis['node_types'].append(node_type)
                        analysis['node_outputs'][node_id] = {
                            'type': node_type,
                            'node_name': node_name,
                            'insights': insights
                        }
                        
                        # Extract meaningful information from insights
                        if node_type == 'data_source':
                            data_summary = insights.get('data_summary', {})
                            rows = data_summary.get('rows', 0)
                            columns = data_summary.get('columns', 0)
                            
                            if rows > 0 and columns > 0:
                                analysis['dataframes_count'] += 1
                                analysis['has_valid_data'] = True
                                analysis['data_sources'].append(node_id)
                                
                                # Set primary data shape from first significant data source
                                if analysis['primary_data_shape'] is None:
                                    analysis['primary_data_shape'] = (rows, columns)
                                
                                # Estimate memory usage from metadata
                                memory_mb = data_summary.get('memory_usage_mb', 0)
                                if memory_mb > 0:
                                    analysis['total_memory_usage'] += memory_mb * 1024 * 1024  # Convert to bytes
                        
                        elif node_type == 'descriptive_stats':
                            stat_summary = insights.get('statistical_summary', {})
                            if stat_summary:
                                analysis['statistics_count'] += 1
                                analysis['has_valid_data'] = True
                        
                        elif node_type in ['classification', 'regression']:
                            model_performance = insights.get('model_performance', {})
                            if model_performance:
                                analysis['models_count'] += 1
                                analysis['has_valid_data'] = True
                        
                        elif node_type == 'clustering':
                            clustering_results = insights.get('clustering_results', {})
                            if clustering_results:
                                analysis['models_count'] += 1
                                analysis['has_valid_data'] = True
                        
                        elif node_type in ['basic_plots', 'advanced_plots']:
                            viz_summary = insights.get('visualization_summary', {})
                            charts_generated = viz_summary.get('charts_generated', 0)
                            if charts_generated > 0:
                                analysis['charts_count'] += charts_generated
                                analysis['has_valid_data'] = True
                        
                        elif node_type in ['univariate_anomaly_detection', 'multivariate_anomaly_detection', 'eda_analysis']:
                            analysis_summary = insights.get('analysis_summary', {})
                            if analysis_summary:
                                analysis['statistics_count'] += 1
                                analysis['has_valid_data'] = True
                        
                        else:
                            # Any other node type with insights counts as valid data
                            if insights:
                                analysis['has_valid_data'] = True
                    
                    else:
                        # Fallback to old format analysis
                        node_type = self._determine_node_type_from_output(node_output)
                        self._log_info(f"Node {node_id} determined type (old format): {node_type}")
                        analysis['node_types'].append(node_type)
                        analysis['node_outputs'][node_id]['type'] = node_type
                        
                        # Extract data from node output (old method)
                        extracted_data = self._extract_data_from_node_output(node_output)
                        analysis['node_outputs'][node_id]['data'] = extracted_data
                        
                        # Log what data was extracted
                        has_df = extracted_data.get('dataframe') is not None
                        has_model = extracted_data.get('model') is not None
                        has_stats = extracted_data.get('statistics') is not None
                        has_charts = extracted_data.get('charts') is not None
                        self._log_info(f"Node {node_id} extracted data: DataFrame={has_df}, Model={has_model}, Stats={has_stats}, Charts={has_charts}")
                        
                        # Analyze DataFrames
                        if extracted_data.get('dataframe') is not None:
                            df = extracted_data['dataframe']
                            analysis['dataframes_count'] += 1
                            analysis['has_valid_data'] = True
                            
                            # Set primary data shape from first significant DataFrame
                            if analysis['primary_data_shape'] is None:
                                analysis['primary_data_shape'] = df.shape
                            
                            # Calculate memory usage
                            try:
                                memory_usage = df.memory_usage(deep=True).sum()
                                analysis['total_memory_usage'] += memory_usage
                            except:
                                pass
                        
                        # Count other data types
                        if extracted_data.get('model') is not None:
                            analysis['models_count'] += 1
                            analysis['has_valid_data'] = True
                        
                        if extracted_data.get('statistics'):
                            analysis['statistics_count'] += 1
                            analysis['has_valid_data'] = True
                        
                        if extracted_data.get('charts'):
                            analysis['charts_count'] += len(extracted_data['charts'])
                            analysis['has_valid_data'] = True
                        
                        # Track data sources
                        if node_type in ['data_source', 'csv_upload', 'excel_upload']:
                            analysis['data_sources'].append(node_id)
                
                except Exception as e:
                    self._log_warning(f"Error analyzing node {node_id}: {str(e)}")
                    continue
            
            self._log_info(f"Connected nodes analysis: {analysis['total_nodes']} nodes, {analysis['dataframes_count']} DataFrames, valid data: {analysis['has_valid_data']}")
            return analysis
            
        except Exception as e:
            self._log_error(f"Error in _analyze_all_connected_nodes: {str(e)}")
            return {'has_valid_data': False, 'error': str(e)}
    
    def _determine_node_type_from_output(self, node_output: Any) -> str:
        """Determine the node type based on its output structure"""
        try:
            # Handle direct DataFrame
            if isinstance(node_output, pd.DataFrame):
                return 'data_source'
            elif isinstance(node_output, dict):
                # Check if there's a 'default' key (common workflow pattern)
                actual_data = node_output
                if 'default' in node_output:
                    actual_data = node_output['default']
                    # If default contains a DataFrame, treat as data source
                    if isinstance(actual_data, pd.DataFrame):
                        return 'data_source'
                    # If default contains a dict, check its contents
                    elif isinstance(actual_data, dict):
                        # Check for explicit type indicator first
                        if 'type' in actual_data:
                            node_type = actual_data['type']
                            if node_type == 'cleaned_data':
                                return 'data_cleaning'
                            elif node_type == 'engineered_data':
                                return 'feature_engineering'
                            elif node_type in ['classification', 'regression', 'clustering']:
                                return node_type
                        
                        # Check for specific node indicators
                        if 'cleaning_summary' in actual_data:
                            return 'data_cleaning'
                        elif 'engineering_summary' in actual_data:
                            return 'feature_engineering'
                        elif 'data' in actual_data and isinstance(actual_data['data'], pd.DataFrame):
                            # Check for additional node type indicators
                            if 'eda_results' in actual_data:
                                return 'eda_analysis'
                            elif 'summary_stats' in actual_data or 'descriptive_stats' in actual_data or 'basic_stats' in actual_data:
                                return 'statistical_analysis'
                            elif 'charts' in actual_data:
                                return 'visualization'
                            elif 'cleaning_summary' in actual_data:
                                return 'data_cleaning'
                            elif 'engineering_summary' in actual_data:
                                return 'feature_engineering'
                            else:
                                return 'data_source'
                
                # Check main structure for type indicators
                sources_to_check = [actual_data, node_output] if 'default' in node_output else [node_output]
                
                for source in sources_to_check:
                    if isinstance(source, dict):
                        # Check for explicit type indicator first
                        if 'type' in source:
                            node_type = source['type']
                            if node_type == 'cleaned_data':
                                return 'data_cleaning'
                            elif node_type == 'engineered_data':
                                return 'feature_engineering'
                            elif node_type in ['classification', 'regression', 'clustering']:
                                return node_type
                        
                        # Check for specific node indicators
                        if 'cleaning_summary' in source:
                            return 'data_cleaning'
                        elif 'engineering_summary' in source:
                            return 'feature_engineering'
                        elif 'data' in source and isinstance(source['data'], pd.DataFrame):
                            # Determine specific type based on additional fields
                            if 'eda_results' in source:
                                return 'eda_analysis'
                            elif 'summary_stats' in source or 'descriptive_stats' in source or 'basic_stats' in source:
                                return 'statistical_analysis'
                            elif 'charts' in source:
                                return 'visualization'
                            elif 'cleaning_summary' in source:
                                return 'data_cleaning'
                            elif 'engineering_summary' in source:
                                return 'feature_engineering'
                            else:
                                return 'data_source'
                        elif 'eda_results' in source:
                            return 'eda_analysis'
                        elif 'charts' in source:
                            return 'visualization'
                        elif 'model' in source or 'predictions' in source:
                            return 'ml_model'
                        elif 'statistics' in source or 'summary_stats' in source or 'descriptive_stats' in source or 'basic_stats' in source:
                            return 'statistical_analysis'
                        elif 'anomalies' in source:
                            return 'anomaly_detection'
                        elif 'classification_report' in source:
                            return 'classification'
                        elif any(key in source for key in ['dataframe', 'df', 'dataset']):
                            return 'data_source'
                
                return 'processing'
            else:
                return 'unknown'
        except Exception as e:
            self._log_warning(f"Error determining node type: {str(e)}")
            return 'unknown'
    
    def _extract_data_from_node_output(self, node_output: Any) -> Dict[str, Any]:
        """Extract relevant data from node output for AI analysis"""
        extracted = {
            'dataframe': None,
            'statistics': None,
            'charts': None,
            'model': None,
            'insights': None,
            'metadata': {}
        }
        
        try:
            # Handle direct DataFrame
            if isinstance(node_output, pd.DataFrame):
                extracted['dataframe'] = node_output
            elif isinstance(node_output, dict):
                # Handle nested data structures - check for 'default' key first (common workflow pattern)
                actual_data = node_output
                if 'default' in node_output:
                    actual_data = node_output['default']
                    
                    # If 'default' contains a DataFrame directly
                    if isinstance(actual_data, pd.DataFrame):
                        extracted['dataframe'] = actual_data
                    # If 'default' contains a dict with data
                    elif isinstance(actual_data, dict):
                        # Try multiple possible DataFrame locations
                        df_found = False
                        for df_key in ['data', 'dataframe', 'df', 'dataset']:
                            if df_key in actual_data:
                                if isinstance(actual_data[df_key], pd.DataFrame):
                                    extracted['dataframe'] = actual_data[df_key]
                                    df_found = True
                                    break
                                elif isinstance(actual_data[df_key], list):
                                    # Convert list of records back to DataFrame (from descriptive stats)
                                    try:
                                        extracted['dataframe'] = pd.DataFrame(actual_data[df_key])
                                        df_found = True
                                        break
                                    except Exception as e:
                                        self._log_warning(f"Failed to convert {df_key} list to DataFrame: {str(e)}")
                        
                        # If no main data found, try data_sample (for large datasets from descriptive stats)
                        if not df_found and 'data_sample' in actual_data and isinstance(actual_data['data_sample'], list):
                            try:
                                extracted['dataframe'] = pd.DataFrame(actual_data['data_sample'])
                                df_found = True
                                self._log_info("Using data_sample for DataFrame reconstruction")
                            except Exception as e:
                                self._log_warning(f"Failed to convert data_sample to DataFrame: {str(e)}")
                        
                        # Extract data_info if available (contains shape, columns, etc.)
                        if 'data_info' in actual_data:
                            extracted['metadata'].update(actual_data['data_info'])
                
                # Extract DataFrame from main structure if not found in 'default'
                if extracted['dataframe'] is None:
                    for df_key in ['data', 'dataframe', 'df', 'dataset']:
                        if df_key in node_output:
                            if isinstance(node_output[df_key], pd.DataFrame):
                                extracted['dataframe'] = node_output[df_key]
                                break
                            elif isinstance(node_output[df_key], list):
                                try:
                                    extracted['dataframe'] = pd.DataFrame(node_output[df_key])
                                    break
                                except Exception as e:
                                    self._log_warning(f"Failed to convert {df_key} to DataFrame: {str(e)}")
                
                # Extract statistics (check both nested and main structure)
                stats_sources = [actual_data, node_output] if 'default' in node_output else [node_output]
                for source in stats_sources:
                    if isinstance(source, dict):
                        # Look for various statistics formats
                        for stats_key in ['basic_stats', 'eda_results', 'summary_stats', 'statistics', 'descriptive_stats']:
                            if stats_key in source:
                                extracted['statistics'] = source[stats_key]
                                break
                        if extracted['statistics']:
                            break
                
                # Extract charts (check both nested and main structure)
                for source in stats_sources:
                    if isinstance(source, dict) and 'charts' in source:
                        extracted['charts'] = source['charts']
                        break
                    elif isinstance(source, dict) and 'charts_generated' in source:
                        # For descriptive stats that just report chart count
                        extracted['charts'] = {'count': source['charts_generated']}
                        break
                
                # Extract models (check both nested and main structure)
                for source in stats_sources:
                    if isinstance(source, dict):
                        if 'model' in source:
                            extracted['model'] = source['model']
                            break
                        elif 'predictions' in source:
                            extracted['model'] = {'predictions': source['predictions']}
                            break
                
                # Extract insights (check both nested and main structure)
                for source in stats_sources:
                    if isinstance(source, dict):
                        if 'insights' in source:
                            extracted['insights'] = source['insights']
                            break
                        elif 'ai_analysis' in source:
                            extracted['insights'] = source['ai_analysis']
                            break
                
                # Extract metadata from main structure
                for key in ['shape', 'columns', 'dtypes', 'memory_usage', 'missing_values', 'data_info']:
                    if key in node_output:
                        extracted['metadata'][key] = node_output[key]
                        
            # Log what we extracted for debugging
            has_df = extracted['dataframe'] is not None
            df_shape = extracted['dataframe'].shape if has_df else 'None'
            has_stats = extracted['statistics'] is not None
            has_charts = extracted['charts'] is not None
            self._log_info(f"Data extraction result: DataFrame={has_df} (shape={df_shape}), Stats={has_stats}, Charts={has_charts}")
                        
        except Exception as e:
            self._log_warning(f"Error extracting data from node output: {str(e)}")
        
        return extracted
        
        return extracted
    
    def _prepare_comprehensive_data_for_ai(self, connected_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive data structure for AI analysis with detailed node outputs"""
        try:
            # Find the primary DataFrame for analysis
            primary_df = None
            all_dataframes = []
            
            for node_id, node_data in connected_analysis.get('node_outputs', {}).items():
                df = node_data.get('data', {}).get('dataframe')
                if df is not None:
                    all_dataframes.append({
                        'node_id': node_id,
                        'dataframe': df,
                        'shape': df.shape
                    })
                    if primary_df is None:
                        primary_df = df
            
            # Create nodes structure that the AI service expects
            nodes = {}
            for node_id, node_data in connected_analysis.get('node_outputs', {}).items():
                nodes[node_id] = {
                    'type': node_data.get('type', 'unknown'),
                    'data': node_data.get('data', {}),
                    'connections': []  # Connections aren't preserved in this format
                }
            
            # Create comprehensive data structure for AI service
            comprehensive_data = {
                'nodes': nodes,  # This is what the AI service expects
                'workflow_context': {
                    'total_nodes': connected_analysis.get('total_nodes', 0),
                    'node_types': list(dict.fromkeys(connected_analysis.get('node_types', []))),
                    'has_data': connected_analysis.get('has_valid_data', False),
                    'dataframes_count': connected_analysis.get('dataframes_count', 0),
                    'primary_data_shape': connected_analysis.get('primary_data_shape'),
                    'workflow_type': 'comprehensive_analysis',
                    'analysis_context': 'multi_node_workflow'
                },
                # Legacy format for backward compatibility
                'workflow_summary': {
                    'total_nodes': connected_analysis.get('total_nodes', 0),
                    'node_types': list(dict.fromkeys(connected_analysis.get('node_types', []))),
                    'has_data': connected_analysis.get('has_valid_data', False),
                    'dataframes_count': connected_analysis.get('dataframes_count', 0),
                    'primary_data_shape': connected_analysis.get('primary_data_shape'),
                    'node_outputs': connected_analysis.get('node_outputs', {}),
                },
                'primary_dataset': None,
                'all_datasets': all_dataframes,
                'aggregated_statistics': {},
                'all_insights': [],
                'workflow_metadata': {
                    'memory_usage': connected_analysis.get('total_memory_usage', 0),
                    'charts_generated': connected_analysis.get('charts_count', 0),
                    'models_created': connected_analysis.get('models_count', 0)
                },
                'detailed_node_analysis': self._create_detailed_node_analysis(connected_analysis.get('node_outputs', {}))
            }
            
            # Process primary dataset with more detail
            if primary_df is not None:
                comprehensive_data['primary_dataset'] = {
                    'shape': primary_df.shape,
                    'columns': list(primary_df.columns),
                    'dtypes': primary_df.dtypes.to_dict(),
                    'missing_values': primary_df.isnull().sum().to_dict(),
                    'memory_usage': primary_df.memory_usage(deep=True).sum() if primary_df is not None and len(primary_df) > 0 else 0,
                    'sample_data': primary_df.head().to_dict() if primary_df is not None and len(primary_df) > 0 else {},
                    'basic_stats': primary_df.describe().to_dict() if primary_df is not None and len(primary_df) > 0 else {},
                    # Add more detailed analysis
                    'data_quality': {
                        'completeness': (1 - primary_df.isnull().sum().sum() / (primary_df.shape[0] * primary_df.shape[1])) * 100 if primary_df is not None and primary_df.shape[0] > 0 and primary_df.shape[1] > 0 else 0,
                        'duplicate_rows': primary_df.duplicated().sum() if primary_df is not None else 0,
                        'unique_values_per_column': {col: primary_df[col].nunique() for col in primary_df.columns} if primary_df is not None else {}
                    }
                }
            
            # Aggregate insights from all nodes with more context
            for node_id, node_data in connected_analysis.get('node_outputs', {}).items():
                node_type = node_data.get('type', 'unknown')
                data = node_data.get('data', {})
                
                # Create comprehensive node insight
                node_insight = {
                    'node_id': node_id,
                    'node_type': node_type,
                    'data_summary': self._create_node_data_summary_for_ai(data),
                    'insights': data.get('insights'),
                    'processing_results': self._extract_processing_results(data)
                }
                
                comprehensive_data['all_insights'].append(node_insight)
            
            return comprehensive_data
            
        except Exception as e:
            self._log_error(f"Error preparing comprehensive data for AI: {str(e)}")
            return {
                'workflow_summary': {'total_nodes': 0, 'has_data': False},
                'error': str(e)
            }
    
    def _create_detailed_node_analysis(self, node_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed analysis of each node for AI context"""
        detailed_analysis = {}
        
        for node_id, node_output in node_outputs.items():
            node_type = node_output.get('type', 'unknown')
            data = node_output.get('data', {})
            
            detailed_analysis[node_id] = {
                'type': node_type,
                'has_dataframe': data.get('dataframe') is not None,
                'has_model': data.get('model') is not None,
                'has_statistics': bool(data.get('statistics')),
                'has_charts': bool(data.get('charts')),
                'data_shape': data.get('dataframe').shape if data.get('dataframe') is not None else None,
                'processing_success': data.get('success', True),
                'key_outputs': self._extract_key_outputs(data)
            }
        
        return detailed_analysis
    
    def _create_node_data_summary_for_ai(self, data: Dict[str, Any]) -> str:
        """Create a detailed summary of node data for AI analysis"""
        try:
            summary_parts = []
            
            # Dataframe analysis
            if 'dataframe' in data and data['dataframe'] is not None:
                df = data['dataframe']
                summary_parts.extend([
                    f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns",
                    f"Columns: {list(df.columns)}",
                    f"Data types: {df.dtypes.to_dict()}",
                    f"Missing values: {df.isnull().sum().to_dict()}",
                    f"Memory usage: {df.memory_usage(deep=True).sum():,.0f} bytes"
                ])
                
                # Add data quality metrics
                total_cells = df.shape[0] * df.shape[1]
                missing_cells = df.isnull().sum().sum()
                completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
                summary_parts.append(f"Data completeness: {completeness:.1f}%")
            
            # Model analysis
            if 'model' in data and data['model'] is not None:
                model_info = str(type(data['model']).__name__)
                summary_parts.append(f"Model trained: {model_info}")
            
            # Statistics analysis
            if 'statistics' in data and data['statistics'] is not None:
                stats = data['statistics']
                if isinstance(stats, dict):
                    summary_parts.append(f"Statistical analysis: {len(stats)} metrics computed")
            
            # Charts analysis
            if 'charts' in data and data['charts'] is not None:
                charts = data['charts']
                if isinstance(charts, list):
                    summary_parts.append(f"Visualizations: {len(charts)} charts generated")
                elif isinstance(charts, dict):
                    chart_count = charts.get('count', len(charts))
                    summary_parts.append(f"Visualizations: {chart_count} charts generated")
                else:
                    summary_parts.append("Visualizations: charts available")
            
            # Processing results
            if 'results' in data and data['results'] is not None:
                results = data['results']
                if isinstance(results, dict):
                    summary_parts.append(f"Processing results: {list(results.keys())}")
            
            return "; ".join(summary_parts) if summary_parts else "No detailed data available"
            
        except Exception as e:
            self._log_error(f"Error creating node data summary: {str(e)}")
            return f"Error analyzing node data: {str(e)}"
    
    def _extract_processing_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key processing results from node data"""
        results = {}
        
        if 'dataframe' in data and data['dataframe'] is not None:
            df = data['dataframe']
            results['dataframe_summary'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'sample': df.head(2).to_dict() if len(df) > 0 else {}
            }
        
        if 'model' in data:
            results['model_type'] = str(type(data['model']).__name__)
        
        if 'statistics' in data:
            results['statistics'] = data['statistics']
        
        if 'charts' in data and data['charts'] is not None:
            charts = data['charts']
            if isinstance(charts, list):
                results['chart_count'] = len(charts)
            elif isinstance(charts, dict):
                results['chart_count'] = charts.get('count', 0)
            else:
                results['chart_count'] = 1
        
        return results
    
    def _extract_key_outputs(self, data: Dict[str, Any]) -> List[str]:
        """Extract key outputs from node data for summary"""
        outputs = []
        
        if data.get('dataframe') is not None:
            df = data['dataframe']
            outputs.append(f"DataFrame({df.shape[0]}x{df.shape[1]})")
        
        if data.get('model') is not None:
            outputs.append(f"Model({type(data['model']).__name__})")
        
        if data.get('statistics'):
            outputs.append("Statistics")
        
        if data.get('charts'):
            charts = data['charts']
            if isinstance(charts, list):
                outputs.append(f"Charts({len(charts)})")
            elif isinstance(charts, dict):
                chart_count = charts.get('count', 0)
                outputs.append(f"Charts({chart_count})")
            else:
                outputs.append("Charts")
        
        return outputs
    
    def _process_ai_summary(self, node: Dict, input_data: Dict) -> Any:
        """Process AI summary node - generate comprehensive AI-powered data insights from ALL connected nodes with background streaming"""
        try:
            self._log_info("Generating AI-powered insights from all connected nodes...")
            
            # DEBUG: Log the input data structure in detail
            self._log_info(f"AI Summary input data type: {type(input_data)}")
            self._log_info(f"AI Summary input data keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
            
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    self._log_info(f"Input data['{key}'] type: {type(value)}")
                    if isinstance(value, dict):
                        self._log_info(f"Input data['{key}'] keys: {list(value.keys())}")
                    elif isinstance(value, pd.DataFrame):
                        self._log_info(f"Input data['{key}'] DataFrame shape: {value.shape}")
            
            # NEW: Collect and analyze ALL connected node outputs
            # Extract the actual node data from the input structure for AI Summary nodes
            if 'default' in input_data and isinstance(input_data['default'], dict):
                # AI Summary nodes receive: {'default': {'node1': data1, 'node2': data2, ...}}
                actual_node_data = input_data['default']
                self._log_info(f"Extracted {len(actual_node_data)} node outputs for AI analysis")
                connected_analysis = self._analyze_all_connected_nodes(actual_node_data)
            else:
                # Fallback for other cases
                connected_analysis = self._analyze_all_connected_nodes(input_data)
            
            # DEBUG: Log connected analysis results
            self._log_info(f"Connected analysis result: {connected_analysis}")
            
            if not connected_analysis['has_valid_data']:
                error_msg = "No valid data found from connected nodes"
                self._log_error(f"Error processing AI summary: {error_msg}")
                
                # Return fallback analysis structure
                return {
                    'ai_analysis': {
                        'success': False,
                        'error': error_msg,
                        'insights': {
                            'key_findings': ['No connected nodes with valid data found'],
                            'data_quality_assessment': {
                                'overall_score': 'Unknown',
                                'main_issues': [error_msg],
                                'recommendations': ['Connect valid data sources or processing nodes']
                            },
                            'analysis_recommendations': ['Add data source or processing nodes'],
                            'business_insights': ['Analysis unavailable due to missing data'],
                            'next_steps': ['Connect data sources to enable AI analysis']
                        }
                    },
                    'connected_nodes_analysis': connected_analysis,
                    'dataset_summary': {
                        'shape': {'rows': 'N/A', 'columns': 'N/A'},
                        'memory_usage': 'N/A',
                        'missing_values': {}
                    },
                    'analysis_config': {
                        'ai_enabled': False,
                        'error': error_msg,
                        'connected_nodes': len(input_data) if isinstance(input_data, dict) else 0
                    },
                    'streaming_config': {
                        'enabled': False,
                        'reason': 'No valid data for streaming analysis'
                    }
                }
            
            # Initialize AI service
            ai_service = AdvancedAIInsightService()
            
            # Prepare comprehensive data from all connected nodes for AI analysis
            comprehensive_data = self._prepare_comprehensive_data_for_ai(connected_analysis)
            
            # NEW: AI Summary nodes ALWAYS run in background streaming mode
            # This ensures other nodes display immediately while AI streams in background
            enable_streaming = True  # Force enable for AI Summary
            stream_in_background = True  # Force background processing for AI Summary
            
            if enable_streaming and stream_in_background:
                # START BACKGROUND STREAMING ANALYSIS
                self._log_info("🔄 Starting background streaming AI analysis...")
                
                # Create background task with AI service
                background_task_info = ai_service.create_background_task(comprehensive_data)
                
                # Use the task ID from AI service to maintain consistency
                task_id = background_task_info.get('background_task', {}).get('id')
                if not task_id:
                    # Fallback to generating our own ID
                    task_id = f"ai_analysis_{int(time.time() * 1000)}"
                
                # Store task in global background_tasks for API access
                background_tasks[task_id] = {
                    'status': 'processing',
                    'created_at': time.time(),
                    'progress': 0,
                    'result': None,
                    'error': None
                }
                
                # Start background analysis in a separate thread
                def run_background_analysis():
                    try:
                        result = ai_service.execute_streaming_analysis(comprehensive_data, task_id)
                        
                        # Update task with result
                        background_tasks[task_id]['status'] = 'completed' if result.get('success') else 'failed'
                        background_tasks[task_id]['result'] = result
                        background_tasks[task_id]['completed_at'] = time.time()
                        background_tasks[task_id]['progress'] = 100
                        
                    except Exception as e:
                        background_tasks[task_id]['status'] = 'failed'
                        background_tasks[task_id]['error'] = str(e)
                        background_tasks[task_id]['completed_at'] = time.time()
                        background_tasks[task_id]['progress'] = 100
                
                # Start analysis in background thread
                thread = threading.Thread(target=run_background_analysis)
                thread.daemon = True
                thread.start()
                
                self._log_info(f"Stored background task: {task_id}")
                
                # Generate quick initial insights
                quick_insights = self._generate_quick_insights(connected_analysis)
                
                # Collect all charts from connected nodes
                all_charts = {}
                chart_count = 0
                
                # Get charts from all connected nodes in the workflow
                for node_id in connected_analysis.get('node_outputs', {}):
                    if node_id in self.chart_cache and self.chart_cache[node_id]:
                        all_charts[f'{node_id}_charts'] = self.chart_cache[node_id]
                        if isinstance(self.chart_cache[node_id], dict):
                            chart_count += len(self.chart_cache[node_id])
                        elif isinstance(self.chart_cache[node_id], list):
                            chart_count += len(self.chart_cache[node_id])
                        else:
                            chart_count += 1
                
                self._log_info(f"Including {chart_count} charts from {len(all_charts)} nodes in AI Summary")
                
                # Create result with background streaming configuration
                result = {
                    'ai_analysis': {
                        'success': True,
                        'insights': quick_insights,
                        'timestamp': datetime.utcnow().isoformat(),
                        'source': 'quick_analysis_with_background_streaming',
                        'background_task': background_task_info.get('background_task'),
                        'streaming_status': 'background_processing'
                    },
                    'connected_nodes_analysis': connected_analysis,
                    'workflow_summary': {
                        'total_connected_nodes': connected_analysis['total_nodes'],
                        'node_types': connected_analysis['node_types'],
                        'dataframes_found': connected_analysis['dataframes_count'],
                        'models_found': connected_analysis['models_count'],
                        'statistics_found': connected_analysis['statistics_count'],
                        'primary_data_shape': connected_analysis['primary_data_shape'],
                        'memory_usage': connected_analysis['total_memory_usage']
                    },
                    'analysis_config': {
                        'ai_enabled': True,
                        'analysis_type': 'comprehensive_workflow_with_streaming',
                        'connected_nodes_processed': connected_analysis['total_nodes'],
                        'data_sources_analyzed': connected_analysis['data_sources'],
                        'streaming_enabled': True,
                        'background_processing': True
                    },
                    'streaming_config': {
                        'enabled': True,
                        'background_processing': True,
                        'task_id': task_id,  # Use the actual task_id we created
                        'estimated_duration': background_task_info.get('estimated_duration', '30-60 seconds'),
                        'status': 'processing'
                    },
                    # Include all charts from connected nodes
                    'charts': all_charts,
                    'chart_count': chart_count
                }
                
                self._log_info(f"AI summary generated successfully from {connected_analysis['total_nodes']} connected nodes using background streaming mode. Success: {background_task_info.get('success')}")
                
            else:
                # TRADITIONAL SYNCHRONOUS ANALYSIS
                self._log_info("🔄 Generating synchronous AI insights...")
                
                # Generate AI insights based on all connected nodes
                ai_insights = ai_service.generate_comprehensive_workflow_insights(comprehensive_data)
                
                # Collect all charts from connected nodes
                all_charts = {}
                chart_count = 0
                
                # Get charts from all connected nodes in the workflow
                for node_id in connected_analysis.get('node_outputs', {}):
                    if node_id in self.chart_cache and self.chart_cache[node_id]:
                        all_charts[f'{node_id}_charts'] = self.chart_cache[node_id]
                        if isinstance(self.chart_cache[node_id], dict):
                            chart_count += len(self.chart_cache[node_id])
                        elif isinstance(self.chart_cache[node_id], list):
                            chart_count += len(self.chart_cache[node_id])
                        else:
                            chart_count += 1
                
                self._log_info(f"Including {chart_count} charts from {len(all_charts)} nodes in synchronous AI Summary")
                
                # Create comprehensive result
                result = {
                    'ai_analysis': ai_insights,
                    'connected_nodes_analysis': connected_analysis,
                    'workflow_summary': {
                        'total_connected_nodes': connected_analysis['total_nodes'],
                        'node_types': connected_analysis['node_types'],
                        'dataframes_found': connected_analysis['dataframes_count'],
                        'models_found': connected_analysis['models_count'],
                        'statistics_found': connected_analysis['statistics_count'],
                        'primary_data_shape': connected_analysis['primary_data_shape'],
                        'memory_usage': connected_analysis['total_memory_usage']
                    },
                    'analysis_config': {
                        'ai_enabled': True,
                        'analysis_type': 'comprehensive_workflow_synchronous',
                        'connected_nodes_processed': connected_analysis['total_nodes'],
                        'data_sources_analyzed': connected_analysis['data_sources']
                    },
                    'streaming_config': {
                        'enabled': False,
                        'reason': 'Synchronous processing requested'
                    },
                    # Include all charts from connected nodes
                    'charts': all_charts,
                    'chart_count': chart_count
                }
                
                # Add advanced analysis if available
                try:
                    advanced_analysis = ai_service.generate_advanced_workflow_analysis(comprehensive_data)
                    result['advanced_ai_analysis'] = {'success': True, 'analysis': advanced_analysis}
                except Exception as e:
                    self._log_warning(f"Advanced AI analysis failed: {str(e)}")
                    result['advanced_ai_analysis'] = {'success': False, 'error': str(e)}
            
            success_status = result['ai_analysis'].get('success', False)
            processing_mode = "background streaming" if enable_streaming and stream_in_background else "synchronous"
            self._log_info(f"AI summary generated successfully from {connected_analysis['total_nodes']} connected nodes using {processing_mode} mode. Success: {success_status}")
            
            return result
            
        except Exception as e:
            self._log_error(f"Error in AI summary processing: {str(e)}")
            
            # Return fallback structure
            return {
                'ai_analysis': {
                    'success': False,
                    'error': str(e),
                    'insights': {
                        'key_findings': ['AI analysis failed due to error'],
                        'data_quality_assessment': {
                            'overall_score': 'Error',
                            'main_issues': [str(e)],
                            'recommendations': ['Check connected nodes and data format']
                        },
                        'analysis_recommendations': ['Review error and retry'],
                        'business_insights': ['Analysis unavailable'],
                        'next_steps': ['Fix connected nodes and retry analysis']
                    }
                },
                'connected_nodes_analysis': {'error': str(e)},
                'workflow_summary': {
                    'total_connected_nodes': 0,
                    'error': 'Analysis failed'
                },
                'analysis_config': {
                    'ai_enabled': False,
                    'error': str(e)
                },
                'streaming_config': {
                    'enabled': False,
                    'error': str(e)
                }
            }
    
    def _generate_quick_insights(self, connected_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quick insights based on meaningful insights extracted from connected nodes"""
        try:
            node_outputs = connected_analysis.get('node_outputs', {})
            node_types = connected_analysis.get('node_types', [])
            total_nodes = connected_analysis.get('total_nodes', 0)
            primary_shape = connected_analysis.get('primary_data_shape')
            
            quick_findings = []
            data_issues = []
            recommendations = []
            business_insights = []
            technical_insights = []
            
            self._log_info(f"Generating quick insights from {total_nodes} nodes with meaningful insights")
            
            # Analyze meaningful insights from each connected node
            for node_id, node_data in node_outputs.items():
                try:
                    # Check if this is the new meaningful insights format
                    if 'insights' in node_data and isinstance(node_data['insights'], dict):
                        insights = node_data['insights']
                        node_name = node_data.get('node_name', f'Node {node_id}')
                        node_type = node_data.get('node_type', 'unknown')
                        
                        self._log_info(f"Processing meaningful insights from {node_name} ({node_type})")
                        
                        if node_type == 'data_source':
                            data_summary = insights.get('data_summary', {})
                            rows = data_summary.get('rows', 'N/A')
                            columns = data_summary.get('columns', 'N/A')
                            missing_values = data_summary.get('missing_values', 0)
                            
                            if rows != 'N/A' and columns != 'N/A':
                                quick_findings.append(f"📊 {node_name}: {rows:,} records × {columns} features loaded")
                                
                                if missing_values > 0:
                                    missing_pct = (missing_values / (rows * columns)) * 100 if rows and columns else 0
                                    if missing_pct > 10:
                                        data_issues.append(f"⚠️ {node_name}: High missing data ({missing_pct:.1f}%)")
                                    else:
                                        quick_findings.append(f"✅ {node_name}: Minor missing data ({missing_pct:.1f}%)")
                                else:
                                    quick_findings.append(f"✅ {node_name}: Complete dataset with no missing values")
                                
                                # Business context based on data size
                                if rows > 10000:
                                    business_insights.append(f"🎯 Large dataset from {node_name} enables robust statistical analysis")
                                if columns > 10:
                                    business_insights.append(f"🔍 Rich feature set from {node_name} supports comprehensive analysis")
                        
                        elif node_type == 'descriptive_stats':
                            stat_summary = insights.get('statistical_summary', {})
                            data_quality = insights.get('data_quality', {})
                            
                            if stat_summary:
                                numeric_cols = stat_summary.get('numeric_columns', 0)
                                if numeric_cols > 0:
                                    quick_findings.append(f"📈 {node_name}: Statistical analysis of {numeric_cols} numeric features completed")
                                    technical_insights.append(f"Comprehensive statistical profiling from {node_name}")
                            
                            # Check for correlation insights
                            strong_correlations = data_quality.get('strong_correlations', [])
                            if strong_correlations:
                                correlation_count = len(strong_correlations)
                                quick_findings.append(f"🔗 {node_name}: Found {correlation_count} strong feature correlations")
                                if correlation_count > 5:
                                    recommendations.append(f"Consider feature selection due to high correlations in {node_name}")
                        
                        elif node_type in ['classification', 'regression']:
                            model_performance = insights.get('model_performance', {})
                            model_details = insights.get('model_details', {})
                            
                            algorithm = model_details.get('algorithm', 'Unknown')
                            if algorithm != 'Unknown':
                                quick_findings.append(f"🤖 {node_name}: {algorithm.replace('_', ' ').title()} model trained successfully")
                                
                            # Extract performance metrics
                            if model_performance:
                                if 'accuracy' in model_performance:
                                    accuracy = model_performance['accuracy']
                                    if accuracy > 0.8:
                                        business_insights.append(f"🎯 Excellent model performance from {node_name} ({accuracy:.1%} accuracy)")
                                    elif accuracy > 0.7:
                                        business_insights.append(f"✅ Good model performance from {node_name} ({accuracy:.1%} accuracy)")
                                    else:
                                        data_issues.append(f"⚠️ Model performance from {node_name} needs improvement ({accuracy:.1%} accuracy)")
                                
                                if 'r2' in model_performance:
                                    r2 = model_performance['r2']
                                    if r2 > 0.8:
                                        business_insights.append(f"🎯 Strong predictive model from {node_name} (R² = {r2:.3f})")
                                    elif r2 > 0.6:
                                        business_insights.append(f"✅ Decent predictive model from {node_name} (R² = {r2:.3f})")
                            
                            # Feature importance insights
                            top_features = model_details.get('top_features', {})
                            if top_features:
                                feature_count = len(top_features)
                                top_feature = max(top_features.keys(), key=lambda k: abs(top_features[k])) if top_features else None
                                if top_feature:
                                    technical_insights.append(f"Most important feature in {node_name}: {top_feature}")
                        
                        elif node_type == 'clustering':
                            clustering_results = insights.get('clustering_results', {})
                            
                            n_clusters = clustering_results.get('n_clusters', 0)
                            silhouette_score = clustering_results.get('silhouette_score', 0)
                            
                            if n_clusters > 0:
                                quick_findings.append(f"🔀 {node_name}: Identified {n_clusters} distinct clusters")
                                if silhouette_score > 0.5:
                                    business_insights.append(f"🎯 Well-defined clusters from {node_name} (silhouette: {silhouette_score:.3f})")
                                elif silhouette_score > 0.3:
                                    business_insights.append(f"✅ Reasonable clusters from {node_name} (silhouette: {silhouette_score:.3f})")
                        
                        elif node_type in ['univariate_anomaly_detection', 'multivariate_anomaly_detection']:
                            analysis_summary = insights.get('analysis_summary', {})
                            recommendations_list = insights.get('recommendations', [])
                            
                            # Extract anomaly detection results
                            anomaly_detection = analysis_summary.get('anomaly_detection', {})
                            if anomaly_detection:
                                total_anomalies = sum(method_data.get('total_anomalies', 0) for method_data in anomaly_detection.values())
                                if total_anomalies > 0:
                                    quick_findings.append(f"🚨 {node_name}: Detected {total_anomalies} potential anomalies")
                                    if total_anomalies > 100:
                                        data_issues.append(f"High anomaly count from {node_name} - investigate data quality")
                                    else:
                                        business_insights.append(f"Anomaly patterns identified by {node_name} for investigation")
                                else:
                                    quick_findings.append(f"✅ {node_name}: No significant anomalies detected")
                            
                            # Include specific recommendations from anomaly detection
                            if recommendations_list:
                                recommendations.extend([f"{node_name}: {rec}" for rec in recommendations_list[:2]])  # Top 2 recommendations
                        
                        elif node_type in ['basic_plots', 'advanced_plots']:
                            viz_summary = insights.get('visualization_summary', {})
                            charts_generated = viz_summary.get('charts_generated', 0)
                            features_visualized = viz_summary.get('features_visualized', [])
                            
                            if charts_generated > 0:
                                quick_findings.append(f"📊 {node_name}: Generated {charts_generated} visualizations")
                                if features_visualized:
                                    feature_list = ', '.join(features_visualized[:3])
                                    technical_insights.append(f"Visual analysis of {feature_list} from {node_name}")
                        
                        elif node_type == 'data_cleaning':
                            cleaning_summary = insights.get('cleaning_summary', {})
                            data_quality_improvement = insights.get('data_quality_improvement', {})
                            
                            if cleaning_summary:
                                quick_findings.append(f"🧹 {node_name}: Data cleaning operations completed")
                                
                            remaining_missing = data_quality_improvement.get('remaining_missing_values', 0)
                            if remaining_missing == 0:
                                business_insights.append(f"✅ Data quality optimized by {node_name}")
                            elif remaining_missing > 0:
                                technical_insights.append(f"Data cleaning by {node_name} - {remaining_missing} missing values remain")
                        
                        else:
                            # Handle unknown node types with available insights
                            if 'extracted_data' in insights:
                                quick_findings.append(f"📋 {node_name}: Data processing completed")
                            elif 'error' in insights:
                                data_issues.append(f"⚠️ {node_name}: {insights['error']}")
                    
                    else:
                        # Fallback for old format or nodes without meaningful insights
                        node_type = node_data.get('type', 'unknown')
                        quick_findings.append(f"📋 Node {node_id} ({node_type}): Processing completed")
                
                except Exception as e:
                    self._log_warning(f"Error processing insights for node {node_id}: {str(e)}")
                    data_issues.append(f"⚠️ Could not analyze node {node_id}")
            
            # Generate workflow-level insights
            if total_nodes > 1:
                quick_findings.append(f"🔗 Multi-step analysis pipeline with {total_nodes} connected nodes")
                business_insights.append("Comprehensive data processing workflow executed successfully")
            
            # Generate recommendations based on analysis
            if not data_issues:
                recommendations.append("✅ Excellent data quality - ready for advanced analytics and business decisions")
            else:
                recommendations.append("🔧 Address identified data quality issues for optimal results")
            
            # Specific recommendations based on node types
            if any('classification' in str(t) or 'regression' in str(t) for t in node_types):
                recommendations.append("🤖 Machine learning models ready - consider deployment or further tuning")
            
            if any('anomaly' in str(t) for t in node_types):
                recommendations.append("🚨 Review anomaly detection results for business impact assessment")
            
            if any('clustering' in str(t) for t in node_types):
                recommendations.append("🔀 Analyze cluster characteristics for business segmentation opportunities")
            
            # Determine overall quality score based on findings
            score_factors = len(quick_findings) - len(data_issues)
            if score_factors >= 5 and len(data_issues) == 0:
                overall_score = "A (Excellent)"
            elif score_factors >= 3 and len(data_issues) <= 1:
                overall_score = "B (Good)"
            elif score_factors >= 1:
                overall_score = "C (Satisfactory)"
            else:
                overall_score = "D (Needs Improvement)"
            
            # Combine all insights
            all_insights = quick_findings + business_insights + technical_insights
            
            return {
                'key_findings': all_insights[:15],  # Top 15 most important findings
                'data_quality_assessment': {
                    'overall_score': overall_score,
                    'main_issues': data_issues,
                    'recommendations': recommendations[:10]  # Top 10 recommendations
                },
                'analysis_recommendations': recommendations,
                'business_insights': business_insights,
                'technical_insights': technical_insights,
                'next_steps': [
                    "Review all identified insights and recommendations",
                    "Investigate any data quality issues flagged",
                    "Consider implementing suggested improvements",
                    "Validate model performance in production environment" if any('model' in str(f) for f in all_insights) else "Consider adding predictive modeling",
                    "Monitor workflow performance for optimization opportunities"
                ]
            }
            
        except Exception as e:
            self._log_error(f"Error generating quick insights: {str(e)}")
            return {
                'key_findings': [f"Error generating insights: {str(e)}"],
                'data_quality_assessment': {
                    'overall_score': 'Unknown',
                    'main_issues': [str(e)],
                    'recommendations': ['Review error details and retry analysis']
                },
                'analysis_recommendations': ['Fix processing errors and retry'],
                'business_insights': ['Analysis unavailable due to processing error'],
                'next_steps': ['Resolve technical issues and rerun analysis']
            }

    def _prepare_eda_data(self, data: pd.DataFrame) -> Dict[str, Any]:
            
        try:
            # Initialize AI service
            ai_service = AdvancedAIInsightService()
            # Generate AI insights from the data
            ai_insights = ai_service.generate_single_node_insights(
                {"dataframe": data}, 
                "eda_analysis", 
                node_type="eda"
            )

            # Create comprehensive result
            result = {
                'ai_analysis': ai_insights,
                'dataset_summary': {
                    'shape': {'rows': len(data), 'columns': len(data.columns)},
                    'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                    'missing_values': data.isnull().sum().to_dict(),
                    'data_types': data.dtypes.astype(str).to_dict()
                },
                'analysis_config': {
                    'ai_enabled': True,
                    'rows_processed': len(data),
                    'columns_processed': len(data.columns)
                }
            }

            # Add advanced analysis if available
            try:
                advanced_analysis = ai_service.generate_advanced_streaming_analysis(data)
                result['advanced_ai_analysis'] = {'success': True, 'analysis': advanced_analysis}
            except Exception as e:
                self._log_warning(f"Advanced AI analysis failed: {str(e)}")
                result['advanced_ai_analysis'] = {'success': False, 'error': str(e)}

            success_status = ai_insights.get('success', False)
            self._log_info(f"AI summary generated successfully. Success: {success_status}")

            return result

        except Exception as e:
            self._log_error(f"Error in AI summary processing: {str(e)}")

            # Return fallback structure
            return {
                'ai_analysis': {
                    'success': False,
                    'error': str(e),
                    'insights': {
                        'key_findings': ['AI analysis failed due to error'],
                        'data_quality_assessment': {
                            'overall_score': 'Error',
                            'main_issues': [str(e)],
                            'recommendations': ['Check data format and try again']
                        },
                        'analysis_recommendations': ['Review error and retry'],
                        'business_insights': ['Analysis unavailable'],
                        'next_steps': ['Fix data issues and retry analysis']
                    }
                },
                'dataset_summary': {
                    'shape': {'rows': 'Error', 'columns': 'Error'},
                    'memory_usage': 'Error',
                    'missing_values': {}
                },
                'analysis_config': {
                    'ai_enabled': False,
                    'error': str(e)
                }
            }
    
    def _prepare_eda_data(self, data: pd.DataFrame) -> Dict:
        """Prepare exploratory data analysis data for AI insights"""
        try:
            # Basic statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            eda_data = {
                'basic_stats': data.describe().to_dict(),
                'data_types': data.dtypes.astype(str).to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'shape': {'rows': len(data), 'columns': len(data.columns)},
                'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                'column_info': {
                    'numeric': list(numeric_cols),
                    'categorical': list(categorical_cols),
                    'total': len(data.columns)
                }
            }
            
            # Add correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                correlations = data[numeric_cols].corr()
                eda_data['correlations'] = correlations.to_dict()
            
            # Add sample data
            eda_data['sample_data'] = data.head().to_dict()
            
            return eda_data
            
        except Exception as e:
            self._log_error(f"Error preparing EDA data: {str(e)}")
            return {
                'error': str(e),
                'shape': {'rows': len(data), 'columns': len(data.columns)} if data is not None else {'rows': 0, 'columns': 0}
            }

    def _process_auto_ml(self, node: Dict, input_data: Dict) -> Any:
        """Process AutoML node - placeholder"""
        raise NotImplementedError("AutoML not yet implemented")
    
    def _process_data_insights(self, node: Dict, input_data: Dict) -> Any:
        """Process data insights node - placeholder"""
        raise NotImplementedError("Data insights not yet implemented")
    
    def _process_save_model(self, node: Dict, input_data: Dict) -> Any:
        """Process save model node - placeholder"""
        raise NotImplementedError("Save model not yet implemented")
    
    def _process_deploy_model(self, node: Dict, input_data: Dict) -> Any:
        """Process deploy model node - placeholder"""
        raise NotImplementedError("Deploy model not yet implemented")
    
    # Advanced ML Helper Methods
    def _analyze_columns_for_classification(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame columns for classification suitability"""
        analysis = {'columns': {}, 'recommendations': {'target_candidates': [], 'feature_candidates': []}}
        
        for col in df.columns:
            unique_count = df[col].nunique()
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            
            col_info = {
                'dtype': str(df[col].dtype),
                'unique_count': unique_count,
                'null_percentage': null_pct,
                'recommended_role': None
            }
            
            # Check for ID-like columns first
            if any(keyword in col.lower() for keyword in ['id', 'index', 'key', 'uuid', 'timestamp']):
                col_info['recommended_role'] = 'identifier'
            # Categorical columns with reasonable cardinality
            elif df[col].dtype == 'object' and unique_count <= 20 and unique_count >= 2:
                col_info['recommended_role'] = 'target_candidate'
                analysis['recommendations']['target_candidates'].append(col)
            # Numeric columns that can be converted to binary (good unique count range)
            elif pd.api.types.is_numeric_dtype(df[col]) and 2 <= unique_count <= 50:
                if unique_count <= 10:
                    col_info['recommended_role'] = 'binary_target_candidate'
                    analysis['recommendations']['target_candidates'].append(col)
                else:
                    col_info['recommended_role'] = 'discrete_target_candidate'
                    analysis['recommendations']['target_candidates'].append(col)
            # High-cardinality numeric columns that can be binned for classification
            elif pd.api.types.is_numeric_dtype(df[col]) and unique_count > 50:
                col_info['recommended_role'] = 'continuous_target_candidate'
                analysis['recommendations']['target_candidates'].append(col)
                # Also mark as feature candidate for flexibility
                analysis['recommendations']['feature_candidates'].append(col)
            # Good feature columns (numeric or categorical with reasonable cardinality)
            elif pd.api.types.is_numeric_dtype(df[col]) or (df[col].dtype == 'object' and unique_count <= 100):
                col_info['recommended_role'] = 'feature_candidate'
                analysis['recommendations']['feature_candidates'].append(col)
            else:
                col_info['recommended_role'] = 'exclude'
            
            analysis['columns'][col] = col_info
        
        # If no target candidates found, be more aggressive and include high-cardinality numeric columns
        if not analysis['recommendations']['target_candidates']:
            for col in df.columns:
                # Skip identifier columns even in fallback mode
                if any(keyword in col.lower() for keyword in ['id', 'index', 'key', 'uuid', 'timestamp']):
                    continue
                    
                if (pd.api.types.is_numeric_dtype(df[col]) and 
                    df[col].nunique() > 10):
                    analysis['columns'][col]['recommended_role'] = 'continuous_target_candidate'
                    analysis['recommendations']['target_candidates'].append(col)
                elif (df[col].dtype == 'object' and 
                      df[col].nunique() > 2 and 
                      df[col].nunique() <= 100):
                    # Include categorical columns with reasonable cardinality
                    analysis['columns'][col]['recommended_role'] = 'categorical_target_candidate'
                    analysis['recommendations']['target_candidates'].append(col)
        
        return analysis
    
    def _auto_select_target_column(self, df: pd.DataFrame, analysis: Dict) -> str:
        """Auto-select best target column with validation and automatic conversion"""
        candidates = analysis['recommendations']['target_candidates']
        
        # First pass: Look for naturally categorical columns
        for candidate in candidates:
            if candidate not in df.columns:
                continue
                
            # Skip identifier columns (double-check)
            if any(keyword in candidate.lower() for keyword in ['id', 'index', 'key', 'uuid', 'timestamp']):
                self._log_info(f"Skipping target candidate '{candidate}': detected as identifier column")
                continue
                
            # Check if column has reasonable class distribution for classification
            unique_count = df[candidate].nunique()
            total_rows = len(df)
            
            # Skip columns with only 1 unique value
            if unique_count < 2:
                self._log_info(f"Skipping target candidate '{candidate}': only {unique_count} unique value(s)")
                continue
            
            # Accept columns with reasonable number of classes (2-50)
            if 2 <= unique_count <= 50:
                # Check minimum class size for classification
                value_counts = df[candidate].value_counts()
                min_class_size = value_counts.min()
                
                # Require at least 2 samples per class for stable classification
                if min_class_size >= 2:
                    self._log_info(f"Selected target column '{candidate}' with {unique_count} classes, min class size: {min_class_size}")
                    return candidate
                else:
                    self._log_info(f"Skipping target candidate '{candidate}': minimum class size too small ({min_class_size})")
                    continue
            else:
                self._log_info(f"Skipping target candidate '{candidate}': too many unique values ({unique_count}) for direct classification")
        
        # Second pass: Look for continuous numeric columns that can be binned
        numeric_candidates = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col]) and 
                             not any(keyword in col.lower() for keyword in ['id', 'index', 'key', 'uuid', 'timestamp'])]
        
        for candidate in numeric_candidates:
            if candidate not in df.columns:
                continue
                
            unique_count = df[candidate].nunique()
            
            # For continuous variables, we can bin them into categories
            if unique_count > 50:
                self._log_info(f"Target candidate '{candidate}' has many unique values ({unique_count}). Will use binning for classification.")
                return candidate
        
        # Third pass: If still no candidates, try any numeric column (even those marked as features)
        all_numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and 
                           not any(keyword in col.lower() for keyword in ['id', 'index', 'key', 'uuid', 'timestamp'])]
        
        if all_numeric_cols:
            # Pick the first suitable numeric column
            for col in all_numeric_cols:
                unique_count = df[col].nunique()
                if unique_count >= 2:  # At least 2 unique values
                    self._log_info(f"Fallback: Using '{col}' with {unique_count} unique values. Will apply binning for classification.")
                    return col
        
        # No suitable candidate found
        self._log_warning("No suitable target column found for classification. Consider using regression for continuous targets.")
        return None
    
    def _auto_select_feature_columns(self, df: pd.DataFrame, target_column: str, analysis: Dict) -> List[str]:
        """Auto-select feature columns"""
        candidates = analysis['recommendations']['feature_candidates']
        return [col for col in candidates if col != target_column]
    
    def _preprocess_target_for_classification(self, y: pd.Series, target_column: str) -> tuple:
        """Preprocess target variable with smart binning for continuous data"""
        target_info = {'preprocessing_applied': []}
        
        if y.isnull().any():
            y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 'Unknown')
            target_info['preprocessing_applied'].append('filled_missing_values')
        
        if pd.api.types.is_numeric_dtype(y):
            unique_vals = y.dropna().unique()
            
            if len(unique_vals) > 50:
                # For highly continuous data, use quantile-based binning
                try:
                    # Use quartiles for binning (creates 4 balanced classes)
                    y_binned = pd.qcut(y, q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'], duplicates='drop')
                    y = y_binned.astype(str)
                    target_info['preprocessing_applied'].append('quantile_binning_4_classes')
                    self._log_info(f"Applied quantile binning to '{target_column}': created 4 balanced classes")
                except Exception as e:
                    # Fallback to equal-width binning if quantile fails
                    try:
                        y_binned = pd.cut(y, bins=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                        y = y_binned.astype(str)
                        target_info['preprocessing_applied'].append('equal_width_binning_4_classes')
                        self._log_info(f"Applied equal-width binning to '{target_column}': created 4 classes")
                    except Exception as e2:
                        # Final fallback to median split
                        median_val = y.median()
                        y = (y > median_val).map({True: 'High', False: 'Low'})
                        target_info['preprocessing_applied'].append(f'binary_median_split_{median_val:.3f}')
                        self._log_info(f"Applied binary median split to '{target_column}' at {median_val:.3f}")
            elif len(unique_vals) > 10:
                # For moderately continuous data, use median split
                median_val = y.median()
                y = (y > median_val).map({True: 'High', False: 'Low'})
                target_info['preprocessing_applied'].append(f'binary_median_split_{median_val:.3f}')
                self._log_info(f"Applied binary median split to '{target_column}' at {median_val:.3f}")
            else:
                # For discrete numeric data, convert to string
                y = y.astype(str)
                target_info['preprocessing_applied'].append('converted_to_string')
        else:
            # For categorical data, just convert to string
            y = y.astype(str)
            target_info['preprocessing_applied'].append('converted_to_string')
        
        return y, target_info
    
    def _preprocess_features_for_classification(self, X: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Preprocess features"""
        X_processed = X.copy()
        
        # Fill missing values
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col].fillna(X_processed[col].median(), inplace=True)
                else:
                    X_processed[col].fillna(X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 'Unknown', inplace=True)
        
        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X_processed[col].nunique() <= 10:
                X_processed = pd.concat([X_processed.drop(col, axis=1), pd.get_dummies(X_processed[col], prefix=col, drop_first=True)], axis=1)
            else:
                # Frequency encoding for high cardinality
                freq_map = X_processed[col].value_counts().to_dict()
                X_processed[col] = X_processed[col].map(freq_map)
        
        return X_processed

    def _analyze_columns_for_regression(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame columns for regression suitability"""
        analysis = {'columns': {}, 'recommendations': {'target_candidates': [], 'feature_candidates': []}}
        
        for col in df.columns:
            unique_count = df[col].nunique()
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            
            col_info = {
                'dtype': str(df[col].dtype),
                'unique_count': unique_count,
                'null_percentage': null_pct,
                'recommended_role': None
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                if unique_count > 20:  # Continuous variable good for regression target
                    col_info['recommended_role'] = 'regression_target_candidate'
                    analysis['recommendations']['target_candidates'].append(col)
                elif unique_count > 2:  # Can be used as feature
                    col_info['recommended_role'] = 'numeric_feature'
                    analysis['recommendations']['feature_candidates'].append(col)
            elif df[col].dtype == 'object' and unique_count <= 50:
                col_info['recommended_role'] = 'categorical_feature'
                analysis['recommendations']['feature_candidates'].append(col)
            
            analysis['columns'][col] = col_info
        
        return analysis
    
    def _auto_select_regression_target(self, df: pd.DataFrame, analysis: Dict) -> str:
        """Auto-select best target column for regression"""
        candidates = analysis['recommendations']['target_candidates']
        if candidates:
            # Select column with highest variance (most interesting to predict)
            variances = {col: df[col].var() for col in candidates}
            return max(variances, key=variances.get)
        return None
    
    def _auto_select_regression_features(self, df: pd.DataFrame, target_column: str, analysis: Dict) -> List[str]:
        """Auto-select feature columns for regression"""
        candidates = analysis['recommendations']['feature_candidates']
        candidates = [col for col in candidates if col != target_column]
        
        # Add all numeric columns not already included
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col != target_column and col not in candidates:
                candidates.append(col)
        
        return candidates[:50]  # Limit to 50 features
    
    def _preprocess_target_for_regression(self, y: pd.Series, target_column: str) -> pd.Series:
        """Preprocess target variable for regression"""
        if y.isnull().any():
            y = y.fillna(y.median())
        return y
    
    def _preprocess_features_for_regression(self, X: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Preprocess features for regression"""
        X_processed = X.copy()
        
        # Fill missing values
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col].fillna(X_processed[col].median(), inplace=True)
                else:
                    X_processed[col].fillna(X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 'Unknown', inplace=True)
        
        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X_processed[col].nunique() <= 10:
                X_processed = pd.concat([X_processed.drop(col, axis=1), pd.get_dummies(X_processed[col], prefix=col, drop_first=True)], axis=1)
            else:
                # Frequency encoding for high cardinality
                freq_map = X_processed[col].value_counts().to_dict()
                X_processed[col] = X_processed[col].map(freq_map)
        
        return X_processed

    # Logging methods
    def _save_plot_as_base64(self, fig) -> str:
        """Save matplotlib figure as base64 encoded string"""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            
            # Encode as base64
            plot_base64 = base64.b64encode(plot_data).decode()
            return f"data:image/png;base64,{plot_base64}"
        except Exception as e:
            self._log_error(f"Error saving plot as base64: {str(e)}")
            return ""

    def _log_info(self, message: str):
        """Log info message"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] INFO: {message}"
        self.execution_logs.append(log_entry)
        print(log_entry)
    
    def _log_warning(self, message: str):
        """Log warning message"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] WARNING: {message}"
        self.execution_logs.append(log_entry)
        print(log_entry)
    
    def _log_error(self, message: str):
        """Log error message"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] ERROR: {message}"
        self.execution_logs.append(log_entry)
        print(log_entry)
    
    def _process_eda_analysis(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process EDA analysis node using the EDA service"""
        try:
            # Check if we have input data from a previous node
            if 'default' in input_data:
                input_obj = input_data['default']
                if isinstance(input_obj, pd.DataFrame):
                    df = input_obj
                elif isinstance(input_obj, dict) and 'data' in input_obj:
                    df = input_obj['data']
                else:
                    raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            else:
                # No input data, try to load from dataset_id in config
                config = node.get('config', {})
                parameters = node.get('parameters', {})
                dataset_id = config.get('dataset_id') or parameters.get('dataset_id')
                
                if not dataset_id:
                    raise ValueError("EDA analysis node requires either input data or a dataset_id in config")
                
                # Load dataset using the same logic as data_source
                try:
                    dataset_id = int(dataset_id)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid dataset ID: {dataset_id}")
                
                dataset = Dataset.query.get(dataset_id)
                if not dataset:
                    raise ValueError(f"Dataset with ID {dataset_id} not found")
                
                # Load the actual data file
                if dataset.file_type.lower() == 'csv':
                    df = pd.read_csv(dataset.file_path)
                elif dataset.file_type.lower() in ['xlsx', 'xls']:
                    df = pd.read_excel(dataset.file_path)
                elif dataset.file_type.lower() == 'json':
                    df = pd.read_json(dataset.file_path)
                else:
                    raise ValueError(f"Unsupported file type: {dataset.file_type}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            # Get configuration for EDA analysis
            config = node.get('config', {})
            parameters = node.get('parameters', {})
            
            # Merge config and parameters, with config taking precedence
            eda_config = {**parameters, **config}
            
            # Use the new DataFrame analysis method
            result = self.eda_service.analyze_dataframe(df, eda_config)
            
            if result['success']:
                return {
                    'data': df,  # Pass through the original data
                    'eda_results': result['results'],
                    'charts': result['results'].get('charts', {}),
                    'recommendations': result['results'].get('recommendations', []),
                    'metadata': result['metadata'],
                    'node_type': 'eda_analysis'
                }
            else:
                raise ValueError(f"EDA analysis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self._log_error(f"Error processing EDA analysis: {str(e)}")
            raise

    def _process_univariate_anomaly_detection(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process univariate anomaly detection node"""
        try:
            if 'default' not in input_data:
                raise ValueError("Univariate anomaly detection node requires input data")
            
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                df = input_obj['data']
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            
            # Call the univariate anomaly detection service
            result = self.univariate_anomaly_service.detect_anomalies(df, config)
            
            if result['success']:
                return {
                    'data': df,  # Pass through the original data
                    'anomaly_results': result['results'],
                    'charts': result['results'].get('charts', {}),
                    'recommendations': result['results'].get('recommendations', []),
                    'metadata': result['metadata'],
                    'node_type': 'univariate_anomaly_detection'
                }
            else:
                raise ValueError(f"Univariate anomaly detection failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self._log_error(f"Error processing univariate anomaly detection: {str(e)}")
            raise

    def _process_multivariate_anomaly_detection(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process multivariate anomaly detection node"""
        try:
            if 'default' not in input_data:
                raise ValueError("Multivariate anomaly detection node requires input data")
            
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                df = input_obj['data']
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            
            # Call the multivariate anomaly detection service
            result = self.multivariate_anomaly_service.detect_anomalies(df, config)
            
            if result['success']:
                return {
                    'data': df,  # Pass through the original data
                    'anomaly_results': result['results'],
                    'charts': result['results'].get('charts', {}),
                    'recommendations': result['results'].get('recommendations', []),
                    'metadata': result['metadata'],
                    'node_type': 'multivariate_anomaly_detection'
                }
            else:
                raise ValueError(f"Multivariate anomaly detection failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self._log_error(f"Error processing multivariate anomaly detection: {str(e)}")
            raise

    def _process_event_detection(self, node: Dict, input_data: Dict) -> Dict[str, Any]:
        """Process event detection node"""
        try:
            if 'default' not in input_data:
                raise ValueError("Event detection node requires input data")
            
            input_obj = input_data['default']
            if isinstance(input_obj, pd.DataFrame):
                df = input_obj
            elif isinstance(input_obj, dict) and 'data' in input_obj:
                df = input_obj['data']
            else:
                raise ValueError(f"Expected DataFrame or dict with 'data' key, got {type(input_obj)}")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for processing, got {type(df)}")
            
            config = node.get('config', {})
            
            # Call the event detection service
            result = self.event_detection_service.detect_events(df, config)
            
            if result['success']:
                return {
                    'data': df,  # Pass through the original data
                    'event_results': result['results'],
                    'charts': result['results'].get('charts', {}),
                    'recommendations': result['results'].get('recommendations', []),
                    'metadata': result['metadata'],
                    'node_type': 'event_detection'
                }
            else:
                raise ValueError(f"Event detection failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self._log_error(f"Error processing event detection: {str(e)}")
            raise
    
    def get_workflow_templates(self) -> List[Dict]:
        """Get predefined workflow templates"""
        return [
            {
                'name': 'Basic Data Analysis',
                'description': 'Load data, perform descriptive statistics, and create basic visualizations',
                'nodes': [
                    {'id': '1', 'type': 'data_source', 'name': 'Load Data', 'x': 100, 'y': 100},
                    {'id': '2', 'type': 'descriptive_stats', 'name': 'Descriptive Stats', 'x': 300, 'y': 100},
                    {'id': '3', 'type': 'basic_plots', 'name': 'Visualizations', 'x': 500, 'y': 100}
                ],
                'connections': [
                    {'source': '1', 'target': '2'},
                    {'source': '2', 'target': '3'}
                ]
            },
            {
                'name': 'AI Data Summary',
                'description': 'Load data and generate AI-powered insights and recommendations',
                'nodes': [
                    {'id': '1', 'type': 'data_source', 'name': 'Load Data', 'x': 100, 'y': 100},
                    {'id': '2', 'type': 'ai_summary', 'name': 'AI Summary', 'x': 300, 'y': 100}
                ],
                'connections': [
                    {'source': '1', 'target': '2'}
                ]
            },
            {
                'name': 'Classification Pipeline',
                'description': 'Complete classification workflow with data loading, cleaning, and model training',
                'nodes': [
                    {'id': '1', 'type': 'data_source', 'name': 'Load Data', 'x': 100, 'y': 100},
                    {'id': '2', 'type': 'data_cleaning', 'name': 'Clean Data', 'x': 300, 'y': 100},
                    {'id': '3', 'type': 'classification', 'name': 'Train Model', 'x': 500, 'y': 100},
                    {'id': '4', 'type': 'export_data', 'name': 'Export Results', 'x': 700, 'y': 100}
                ],
                'connections': [
                    {'source': '1', 'target': '2'},
                    {'source': '2', 'target': '3'},
                    {'source': '3', 'target': '4'}
                ]
            }
        ]
