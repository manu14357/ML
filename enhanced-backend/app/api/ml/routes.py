"""
Advanced ML API routes for comprehensive machine learning model management
"""

import os
import json
import pickle
import joblib
import time
import traceback
from datetime import datetime, timedelta
from io import BytesIO
import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify, send_file, current_app
from sqlalchemy import desc, and_, or_
from werkzeug.utils import secure_filename

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor, VotingClassifier, VotingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Advanced ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier, CatBoostRegressor
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False

# SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from app import db
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.services.ml_service import MLModelService, AutoMLService, AdvancedTrainingService

ml_bp = Blueprint('ml', __name__)

# Algorithm configurations
ALGORITHM_CONFIGS = {
    'classification': {
        'random_forest': {
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'random_state': [42]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'random_state': [42]
            }
        },
        'xgboost': {
            'model': xgb.XGBClassifier if ADVANCED_LIBS_AVAILABLE else None,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'random_state': [42]
            }
        },
        'lightgbm': {
            'model': lgb.LGBMClassifier if ADVANCED_LIBS_AVAILABLE else None,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'random_state': [42]
            }
        },
        'catboost': {
            'model': CatBoostClassifier if ADVANCED_LIBS_AVAILABLE else None,
            'params': {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 4, 5],
                'random_state': [42],
                'verbose': [False]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression,
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'random_state': [42]
            }
        },
        'svm': {
            'model': SVC,
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'random_state': [42]
            }
        },
        'knn': {
            'model': KNeighborsClassifier,
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'decision_tree': {
            'model': DecisionTreeClassifier,
            'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        },
        'naive_bayes': {
            'model': GaussianNB,
            'params': {}
        },
        'extra_trees': {
            'model': ExtraTreesClassifier,
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        },
        'ada_boost': {
            'model': AdaBoostClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'random_state': [42]
            }
        }
    },
    'regression': {
        'random_forest': {
            'model': RandomForestRegressor,
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'random_state': [42]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'random_state': [42]
            }
        },
        'xgboost': {
            'model': xgb.XGBRegressor if ADVANCED_LIBS_AVAILABLE else None,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'random_state': [42]
            }
        },
        'linear_regression': {
            'model': LinearRegression,
            'params': {}
        },
        'ridge': {
            'model': Ridge,
            'params': {
                'alpha': [0.1, 1, 10, 100],
                'random_state': [42]
            }
        },
        'lasso': {
            'model': Lasso,
            'params': {
                'alpha': [0.1, 1, 10, 100],
                'random_state': [42]
            }
        },
        'svr': {
            'model': SVR,
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'lightgbm': {
            'model': lgb.LGBMRegressor if ADVANCED_LIBS_AVAILABLE else None,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'random_state': [42]
            }
        },
        'catboost': {
            'model': CatBoostRegressor if ADVANCED_LIBS_AVAILABLE else None,
            'params': {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 4, 5],
                'random_state': [42],
                'verbose': [False]
            }
        },
        'elastic_net': {
            'model': ElasticNet,
            'params': {
                'alpha': [0.1, 1, 10, 100],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9],
                'random_state': [42]
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor,
            'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        },
        'extra_trees': {
            'model': ExtraTreesRegressor,
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        },
        'ada_boost': {
            'model': AdaBoostRegressor,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'random_state': [42]
            }
        }
    }
}

def get_model_directory():
    """Get the directory for storing model files"""
    model_dir = os.path.join(current_app.instance_path, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def preprocess_data(df, target_column, preprocessing_config, model_type='classification'):
    """Advanced data preprocessing with comprehensive large dataset support"""
    
    # Use the advanced preprocessing from AutoMLService
    from app.services.ml_service import AutoMLService
    
    try:
        # Get max samples from config
        max_samples = preprocessing_config.get('max_samples', 100000)
        
        # Use the advanced preprocessing
        X, y, preprocessing_info = AutoMLService.smart_preprocess_data(
            df, target_column, model_type, max_samples
        )
        
        if X is None or y is None:
            error_msg = preprocessing_info.get('error', 'Preprocessing failed')
            raise ValueError(error_msg)
        
        # For backward compatibility, return target_encoder as None
        # (the advanced service handles encoding internally)
        target_encoder = None
        if model_type == 'classification' and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        
        return X, y, target_encoder
        
    except Exception as e:
        # Re-raise with user-friendly message
        error_msg = str(e)
        if "Preprocessing failed:" in error_msg:
            raise ValueError(error_msg)
        else:
            raise ValueError(f"Data preprocessing failed: {error_msg}")
    

def calculate_feature_importance(model, feature_names):
    """Calculate feature importance for different model types"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        return None
    
    importance_dict = dict(zip(feature_names, importances.tolist()))
    # Sort by importance
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

def calculate_shap_values(model, X_test, feature_names):
    """Calculate SHAP values for model interpretability"""
    if not SHAP_AVAILABLE:
        return None
    
    try:
        # Create appropriate explainer based on model type
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') else shap.Explainer(model)
        else:
            explainer = shap.Explainer(model)
        
        # Calculate SHAP values for a sample of test data (max 100 samples for performance)
        sample_size = min(100, len(X_test))
        shap_values = explainer.shap_values(X_test.iloc[:sample_size])
        
        # For classification, take values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        shap_importance = dict(zip(feature_names, mean_shap.tolist()))
        
        return dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None

@ml_bp.route('/models', methods=['GET'])
def get_models():
    """Get list of ML models with filtering and pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        
        # Filters
        model_type = request.args.get('type')
        algorithm = request.args.get('algorithm')
        status = request.args.get('status')
        search = request.args.get('search')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Build query
        query = MLModel.query
        
        if model_type:
            query = query.filter(MLModel.model_type == model_type)
        if algorithm:
            query = query.filter(MLModel.algorithm == algorithm)
        if status:
            query = query.filter(MLModel.status == status)
        if search:
            query = query.filter(or_(
                MLModel.name.ilike(f'%{search}%'),
                MLModel.description.ilike(f'%{search}%'),
                MLModel.algorithm.ilike(f'%{search}%')
            ))
        
        # Sorting
        if hasattr(MLModel, sort_by):
            order_column = getattr(MLModel, sort_by)
            if sort_order == 'desc':
                order_column = desc(order_column)
            query = query.order_by(order_column)
        
        # Pagination
        models_paginated = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        # Get statistics
        total_models = MLModel.query.count()
        trained_models = MLModel.query.filter(MLModel.status == 'trained').count()
        deployed_models = MLModel.query.filter(MLModel.is_deployed == True).count()
        
        return jsonify({
            'success': True,
            'models': [model.to_dict() for model in models_paginated.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': models_paginated.total,
                'pages': models_paginated.pages,
                'has_prev': models_paginated.has_prev,
                'has_next': models_paginated.has_next
            },
            'statistics': {
                'total_models': total_models,
                'trained_models': trained_models,
                'deployed_models': deployed_models,
                'failed_models': MLModel.query.filter(MLModel.status == 'failed').count()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching models: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>', methods=['GET'])
def get_model_details(model_id):
    """Get detailed information about a specific model"""
    try:
        model = MLModel.query.get_or_404(model_id)
        return jsonify({
            'success': True,
            'model': model.to_dict(include_details=True)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching model details: {str(e)}'
        }), 500

@ml_bp.route('/algorithms', methods=['GET'])
def get_algorithms():
    """Get available algorithms and their configurations"""
    try:
        algorithms = {}
        for model_type, algos in ALGORITHM_CONFIGS.items():
            algorithms[model_type] = []
            for algo_name, config in algos.items():
                if config['model'] is not None:  # Check if library is available
                    algorithms[model_type].append({
                        'name': algo_name,
                        'display_name': algo_name.replace('_', ' ').title(),
                        'parameters': list(config['params'].keys())
                    })
        
        return jsonify({
            'success': True,
            'algorithms': algorithms,
            'advanced_libraries_available': ADVANCED_LIBS_AVAILABLE,
            'shap_available': SHAP_AVAILABLE
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching algorithms: {str(e)}'
        }), 500

@ml_bp.route('/train', methods=['POST'])
def train_model():
    """Train a new ML model with advanced configuration"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'dataset_id', 'algorithm', 'model_type', 'target_column']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Load dataset
        dataset = Dataset.query.get(data['dataset_id'])
        if not dataset:
            return jsonify({
                'success': False,
                'message': 'Dataset not found'
            }), 404
        
        # Check dataset file exists
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
        
        df = pd.read_csv(dataset.file_path)
        
        # Validate target column
        if data['target_column'] not in df.columns:
            return jsonify({
                'success': False,
                'message': f'Target column "{data["target_column"]}" not found in dataset'
            }), 400
        
        # Create model record
        model = MLModel(
            name=data['name'],
            description=data.get('description', ''),
            algorithm=data['algorithm'],
            model_type=data['model_type'],
            dataset_id=data['dataset_id'],
            target_column=data['target_column'],
            training_config=data.get('training_config', {}),
            status='training'
        )
        
        db.session.add(model)
        db.session.commit()
        
        # Training configuration
        training_config = data.get('training_config', {})
        preprocessing_config = training_config.get('preprocessing', {})
        hyperparameter_tuning = training_config.get('hyperparameter_tuning', True)
        cross_validation = training_config.get('cross_validation', True)
        cv_folds = training_config.get('cv_folds', 5)
        test_size = training_config.get('test_size', 0.2)
        
        start_time = time.time()
        training_log = []
        
        try:
            # Preprocess data
            training_log.append("Starting data preprocessing...")
            X, y, target_encoder = preprocess_data(df, data['target_column'], preprocessing_config, data['model_type'])
            
            # Feature selection if specified
            if training_config.get('feature_selection'):
                k_features = training_config.get('k_features', 10)
                if data['model_type'] == 'classification':
                    selector = SelectKBest(score_func=f_classif, k=min(k_features, X.shape[1]))
                else:
                    selector = SelectKBest(score_func=f_regression, k=min(k_features, X.shape[1]))
                X = pd.DataFrame(selector.fit_transform(X, y), 
                               columns=X.columns[selector.get_support()], 
                               index=X.index)
                training_log.append(f"Selected {X.shape[1]} features using SelectKBest")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if data['model_type'] == 'classification' else None
            )
            
            training_log.append(f"Data split: {len(X_train)} training, {len(X_test)} testing samples")
            
            # Get algorithm configuration
            algo_config = ALGORITHM_CONFIGS[data['model_type']].get(data['algorithm'])
            if not algo_config or algo_config['model'] is None:
                raise ValueError(f"Algorithm {data['algorithm']} not available")
            
            # Initialize model
            model_class = algo_config['model']
            
            # Hyperparameter tuning
            if hyperparameter_tuning and algo_config['params']:
                training_log.append("Starting hyperparameter tuning...")
                
                # Use RandomizedSearchCV for faster tuning
                search_cv = RandomizedSearchCV(
                    model_class(),
                    algo_config['params'],
                    cv=cv_folds,
                    n_iter=20,
                    random_state=42,
                    n_jobs=-1,
                    scoring='accuracy' if data['model_type'] == 'classification' else 'r2'
                )
                
                search_cv.fit(X_train, y_train)
                best_model = search_cv.best_estimator_
                best_params = search_cv.best_params_
                
                training_log.append(f"Best parameters: {best_params}")
                model.training_config['best_params'] = best_params
            else:
                # Use default parameters
                best_model = model_class(**{k: v[0] if isinstance(v, list) else v 
                                          for k, v in algo_config['params'].items() 
                                          if v})
            
            # Train model
            training_log.append("Training model...")
            best_model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate metrics
            training_metrics = {}
            validation_metrics = {}
            test_metrics = {}
            
            if data['model_type'] == 'classification':
                # Classification metrics
                training_metrics = {
                    'accuracy': accuracy_score(y_train, y_train_pred),
                    'precision': precision_score(y_train, y_train_pred, average='weighted'),
                    'recall': recall_score(y_train, y_train_pred, average='weighted'),
                    'f1_score': f1_score(y_train, y_train_pred, average='weighted')
                }
                
                test_metrics = {
                    'accuracy': accuracy_score(y_test, y_test_pred),
                    'precision': precision_score(y_test, y_test_pred, average='weighted'),
                    'recall': recall_score(y_test, y_test_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
                    'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
                }
                
                # ROC AUC for binary classification
                if len(np.unique(y)) == 2:
                    try:
                        y_test_proba = best_model.predict_proba(X_test)[:, 1]
                        test_metrics['roc_auc'] = roc_auc_score(y_test, y_test_proba)
                        
                        # ROC curve data
                        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                        test_metrics['roc_curve'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist()
                        }
                        
                        # Precision-recall curve
                        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
                        test_metrics['pr_curve'] = {
                            'precision': precision.tolist(),
                            'recall': recall.tolist()
                        }
                    except:
                        pass
                
                # Update model with primary metrics
                model.accuracy = test_metrics['accuracy']
                model.precision = test_metrics['precision']
                model.recall = test_metrics['recall']
                model.f1_score = test_metrics['f1_score']
                
            else:
                # Regression metrics
                training_metrics = {
                    'mse': mean_squared_error(y_train, y_train_pred),
                    'mae': mean_absolute_error(y_train, y_train_pred),
                    'r2_score': r2_score(y_train, y_train_pred)
                }
                
                test_metrics = {
                    'mse': mean_squared_error(y_test, y_test_pred),
                    'mae': mean_absolute_error(y_test, y_test_pred),
                    'r2_score': r2_score(y_test, y_test_pred)
                }
                
                # Update model with primary metrics
                model.mse = test_metrics['mse']
                model.mae = test_metrics['mae']
                model.r2_score = test_metrics['r2_score']
            
            # Cross-validation with improved error handling
            if cross_validation:
                training_log.append("Performing cross-validation...")
                
                # Check if we have enough samples for cross-validation
                min_samples_needed = cv_folds
                if data['model_type'] == 'classification':
                    # For classification, each class needs at least cv_folds samples
                    class_counts = pd.Series(y_train).value_counts()
                    min_class_count = class_counts.min()
                    
                    if min_class_count < cv_folds:
                        # Adjust cv_folds or skip CV if not enough samples
                        adjusted_folds = min(min_class_count, 2)  # At least 2 folds
                        if adjusted_folds < 2:
                            training_log.append(f"Skipping cross-validation: not enough samples per class (min: {min_class_count}, need: {cv_folds})")
                            validation_metrics['cv_warning'] = f"Cross-validation skipped: insufficient samples per class (minimum class has {min_class_count} samples, need at least {cv_folds})"
                        else:
                            training_log.append(f"Adjusting CV folds from {cv_folds} to {adjusted_folds} due to small class sizes")
                            cv_folds = adjusted_folds
                            
                            scoring = 'accuracy'
                            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring)
                            
                            model.cv_scores = cv_scores.tolist()
                            model.cv_mean = float(cv_scores.mean())
                            model.cv_std = float(cv_scores.std())
                            
                            validation_metrics['cv_scores'] = cv_scores.tolist()
                            validation_metrics['cv_mean'] = float(cv_scores.mean())
                            validation_metrics['cv_std'] = float(cv_scores.std())
                            validation_metrics['cv_folds_used'] = cv_folds
                            
                            training_log.append(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f}) with {cv_folds} folds")
                    else:
                        # Normal cross-validation
                        scoring = 'accuracy'
                        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring)
                        
                        model.cv_scores = cv_scores.tolist()
                        model.cv_mean = float(cv_scores.mean())
                        model.cv_std = float(cv_scores.std())
                        
                        validation_metrics['cv_scores'] = cv_scores.tolist()
                        validation_metrics['cv_mean'] = float(cv_scores.mean())
                        validation_metrics['cv_std'] = float(cv_scores.std())
                        
                        training_log.append(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                else:
                    # Regression cross-validation
                    if len(X_train) < cv_folds:
                        adjusted_folds = min(len(X_train), 2)
                        training_log.append(f"Adjusting CV folds from {cv_folds} to {adjusted_folds} due to small dataset size")
                        cv_folds = adjusted_folds
                    
                    scoring = 'r2'
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring)
                    
                    model.cv_scores = cv_scores.tolist()
                    model.cv_mean = float(cv_scores.mean())
                    model.cv_std = float(cv_scores.std())
                    
                    validation_metrics['cv_scores'] = cv_scores.tolist()
                    validation_metrics['cv_mean'] = float(cv_scores.mean())
                    validation_metrics['cv_std'] = float(cv_scores.std())
                    
                    training_log.append(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Feature importance
            feature_importance = calculate_feature_importance(best_model, X.columns.tolist())
            if feature_importance:
                model.feature_importance = feature_importance
                training_log.append("Calculated feature importance")
            
            # SHAP values
            shap_importance = calculate_shap_values(best_model, X_test, X.columns.tolist())
            if shap_importance:
                model.shap_values = shap_importance
                training_log.append("Calculated SHAP values")
            
            # Save model file with preprocessing objects
            model_dir = get_model_directory()
            model_filename = f"model_{model.id}_{data['algorithm']}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            
            # Create model bundle with preprocessing info
            model_bundle = {
                'model': best_model,
                'target_encoder': target_encoder,
                'feature_names': X.columns.tolist(),
                'model_type': data['model_type'],
                'preprocessing_config': preprocessing_config
            }
            
            joblib.dump(model_bundle, model_path)
            model_size = os.path.getsize(model_path)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Update model record
            model.model_path = model_path
            model.model_size = model_size
            model.features = X.columns.tolist()
            model.training_metrics = training_metrics
            model.validation_metrics = validation_metrics
            model.test_metrics = test_metrics
            model.training_time = training_time
            model.training_log = '\n'.join(training_log)
            model.status = 'trained'
            model.trained_at = datetime.utcnow()
            
            db.session.commit()
            
            training_log.append(f"Model training completed in {training_time:.2f} seconds")
            
            return jsonify({
                'success': True,
                'model': model.to_dict(include_details=True),
                'message': 'Model trained successfully'
            })
            
        except Exception as training_error:
            # Mark model as failed
            model.status = 'failed'
            
            # Provide user-friendly error messages
            error_message = str(training_error)
            user_friendly_message = error_message
            
            # Common ML error patterns and user-friendly translations
            if "The least populated class in y has only" in error_message:
                class_info = error_message.split("The least populated class in y has only ")[1].split(" member")[0]
                user_friendly_message = f"Dataset has insufficient samples for training. The smallest class has only {class_info} sample(s). Each class needs at least 2-5 samples for reliable training. Please add more data or use a simpler model."
            
            elif "Input contains NaN" in error_message or "Input contains infinity" in error_message:
                user_friendly_message = "Dataset contains missing values (NaN) or infinite values. Please clean your data by removing or filling missing values before training."
            
            elif "could not convert string to float" in error_message:
                user_friendly_message = "Dataset contains non-numeric values in columns that should be numeric. Please ensure all feature columns contain only numbers or properly encode categorical variables."
            
            elif "Unknown label type" in error_message:
                user_friendly_message = "The target column contains unsupported data types. Please ensure the target column contains valid class labels for classification or numeric values for regression."
            
            elif "multi-class format is not supported" in error_message:
                user_friendly_message = "This algorithm doesn't support multi-class classification with your data format. Try using a different algorithm or converting to binary classification."
            
            elif "Number of features" in error_message and "does not match" in error_message:
                user_friendly_message = "Feature mismatch error. This usually happens when the training data has different columns than expected. Please check your dataset structure."
            
            elif "Memory" in error_message or "memory" in error_message:
                user_friendly_message = "Not enough memory to train this model with the current dataset. Try using a smaller dataset, simpler algorithm, or reduce the number of features."
            
            elif "Convergence" in error_message or "convergence" in error_message:
                user_friendly_message = "The algorithm failed to converge. Try increasing the maximum iterations, using feature scaling, or choosing a different algorithm."
            
            elif "empty" in error_message.lower() and "array" in error_message.lower():
                user_friendly_message = "Dataset appears to be empty or has no valid data after preprocessing. Please check that your dataset contains sufficient data."
            
            elif "singular matrix" in error_message.lower() or "linearly dependent" in error_message.lower():
                user_friendly_message = "Mathematical error due to correlated features. Try removing highly correlated features or using regularization."
            
            training_log.append(f"Error: {user_friendly_message}")
            model.training_log = '\n'.join(training_log)
            db.session.commit()
            
            return jsonify({
                'success': False,
                'message': user_friendly_message,
                'technical_error': error_message,
                'model_id': model.id
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting training: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>/predict', methods=['POST'])
def predict(model_id):
    """Make predictions using a trained model"""
    try:
        print(f"Received prediction request for model_id={model_id}")
        model = MLModel.query.get_or_404(model_id)
        print(f"Found model: {model.name}, type={model.model_type}, algorithm={model.algorithm}, status={model.status}")
        
        if model.status != 'trained':
            print(f"Model {model_id} is not trained, status={model.status}")
            return jsonify({
                'success': False,
                'message': 'Model is not trained'
            }), 400
        
        if not model.model_path or not os.path.exists(model.model_path):
            print(f"Model file not found for model_id={model_id}, path={model.model_path}")
            return jsonify({
                'success': False,
                'message': 'Model file not found'
            }), 404
        
        # Load model bundle
        model_bundle = joblib.load(model.model_path)
        
        # Handle both old and new model formats
        if isinstance(model_bundle, dict) and 'model' in model_bundle:
            trained_model = model_bundle['model']
            target_encoder = model_bundle.get('target_encoder')
            feature_names = model_bundle.get('feature_names', model.features)
        else:
            # Old format - just the model
            trained_model = model_bundle
            target_encoder = None
            feature_names = model.features
        
        # Get prediction data
        data = request.get_json()
        
        # Handle different input formats
        if isinstance(data, dict) and len(data) > 0 and 'data' not in data:
            # Direct feature values (from frontend)
            pred_data = [data]
        elif 'data' in data:
            # Wrapped in data field
            pred_data = data['data'] if isinstance(data['data'], list) else [data['data']]
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid prediction data format'
            }), 400
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(pred_data)
        
        # Store original categorical columns for later processing
        original_categorical_cols = []
        for col in pred_df.columns:
            # Check if this column was one-hot encoded during training
            encoded_variants = [f for f in feature_names if f.startswith(f"{col}_")]
            if encoded_variants:
                original_categorical_cols.append((col, encoded_variants))
        
        # Handle one-hot encoding for categorical features FIRST
        for original_col, encoded_variants in original_categorical_cols:
            if original_col in pred_df.columns:
                # Get the original categorical value
                cat_value = pred_df[original_col].iloc[0]
                
                # Remove the original categorical column
                pred_df = pred_df.drop(columns=[original_col])
                
                # Create one-hot encoded columns
                for encoded_col in encoded_variants:
                    # Extract the category from the encoded column name
                    category = encoded_col.split(f"{original_col}_")[1]
                    # Set 1 if this is the selected category, 0 otherwise
                    pred_df[encoded_col] = 1 if str(cat_value) == category else 0
        
        # Now ensure all remaining features are properly typed as numeric
        for col in pred_df.columns:
            try:
                # Try to convert to numeric
                pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce')
                # Check for NaN values after conversion
                if pred_df[col].isna().any():
                    return jsonify({
                        'success': False,
                        'message': f'Invalid numeric value in feature "{col}". Please ensure all feature values are valid numbers.'
                    }), 400
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error converting feature "{col}" to numeric: {str(e)}'
                }), 400
        
        # Add any missing encoded features (set to 0)
        for feature in feature_names:
            if feature not in pred_df.columns:
                pred_df[feature] = 0
        
        # Ensure features match training features and are in correct order
        missing_features = set(feature_names) - set(pred_df.columns)
        if missing_features:
            return jsonify({
                'success': False,
                'message': f'Missing features after processing: {list(missing_features)}'
            }), 400
        
        # Select and order features as in training
        pred_df = pred_df[feature_names]
        
        # Make predictions
        print(f"Making prediction with input shape: {pred_df.shape}")
        predictions = trained_model.predict(pred_df)
        
        # Decode predictions for classification with encoded targets
        if model.model_type == 'classification' and target_encoder is not None:
            print(f"Decoding prediction using target encoder: {target_encoder}")
            predictions = target_encoder.inverse_transform(predictions)
        
        print(f"Prediction result: {predictions}")
        
        # Get prediction probabilities for classification
        prediction_probabilities = None
        if hasattr(trained_model, 'predict_proba') and model.model_type == 'classification':
            prediction_probabilities = trained_model.predict_proba(pred_df)
        
        # Update usage statistics
        model.increment_prediction_count()
        
        # Return single prediction for single input
        if len(predictions) == 1:
            result = {
                'success': True,
                'prediction': predictions[0],
                'model_info': {
                    'id': model.id,
                    'name': model.name,
                    'algorithm': model.algorithm,
                    'model_type': model.model_type
                }
            }
            
            if prediction_probabilities is not None:
                result['prediction_probabilities'] = prediction_probabilities[0].tolist()
                # Add class names if available
                if target_encoder is not None:
                    result['class_names'] = target_encoder.classes_.tolist()
        else:
            result = {
                'success': True,
                'predictions': predictions.tolist(),
                'model_info': {
                    'id': model.id,
                    'name': model.name,
                    'algorithm': model.algorithm,
                    'model_type': model.model_type
                }
            }
            
            if prediction_probabilities is not None:
                result['prediction_probabilities'] = prediction_probabilities.tolist()
                if target_encoder is not None:
                    result['class_names'] = target_encoder.classes_.tolist()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error for model_id={model_id}: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Prediction error: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>/download', methods=['GET'])
def download_model(model_id):
    """Download a trained model file"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        if not model.model_path or not os.path.exists(model.model_path):
            return jsonify({
                'success': False,
                'message': 'Model file not found'
            }), 404
        
        # Update download count
        model.increment_download_count()
        
        return send_file(
            model.model_path,
            as_attachment=True,
            download_name=f"{model.name}_{model.algorithm}.pkl"
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Download error: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model and its associated files"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        # Delete model file if exists
        if model.model_path and os.path.exists(model.model_path):
            os.remove(model.model_path)
        
        # Delete from database
        db.session.delete(model)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Model deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error deleting model: {str(e)}'
        }), 500

@ml_bp.route('/models/compare', methods=['POST'])
def compare_models():
    """Compare multiple models"""
    try:
        data = request.get_json()
        model_ids = data.get('model_ids', [])
        
        if len(model_ids) < 2:
            return jsonify({
                'success': False,
                'message': 'At least 2 models are required for comparison'
            }), 400
        
        models = MLModel.query.filter(MLModel.id.in_(model_ids)).all()
        
        if len(models) != len(model_ids):
            return jsonify({
                'success': False,
                'message': 'Some models not found'
            }), 404
        
        comparison = {
            'models': [],
            'metrics_comparison': {},
            'summary': {}
        }
        
        # Collect model data
        for model in models:
            model_data = model.to_dict(include_details=True)
            comparison['models'].append(model_data)
        
        # Compare metrics
        if models[0].model_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        else:
            metrics = ['r2_score', 'mse', 'mae']
        
        for metric in metrics:
            comparison['metrics_comparison'][metric] = []
            for model in models:
                value = getattr(model, metric)
                comparison['metrics_comparison'][metric].append({
                    'model_id': model.id,
                    'model_name': model.name,
                    'value': value
                })
        
        # Find best performing model
        primary_metric = 'accuracy' if models[0].model_type == 'classification' else 'r2_score'
        best_model = max(models, key=lambda m: getattr(m, primary_metric) or 0)
        
        comparison['summary'] = {
            'best_model': {
                'id': best_model.id,
                'name': best_model.name,
                'primary_metric': getattr(best_model, primary_metric)
            },
            'total_models': len(models),
            'model_type': models[0].model_type
        }
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Comparison error: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>/insights', methods=['GET'])
def get_model_insights(model_id):
    """Get comprehensive insights for a model"""
    try:
        insights = MLModelService.generate_model_insights(model_id)
        if not insights:
            return jsonify({
                'success': False,
                'message': 'Model not found or not trained'
            }), 404
        
        return jsonify({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating insights: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>/learning-curves', methods=['GET'])
def get_learning_curves(model_id):
    """Generate learning curves for a model"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        if not model.model_path or not os.path.exists(model.model_path):
            return jsonify({
                'success': False,
                'message': 'Model file not found'
            }), 404
        
        # Load model and dataset
        trained_model = joblib.load(model.model_path)
        dataset = Dataset.query.get(model.dataset_id)
        dataset_path = os.path.join(current_app.config['UPLOAD_FOLDER'], dataset.filename)
        df = pd.read_csv(dataset_path)
        
        # Preprocess data
        X, y = preprocess_data(df, model.target_column, {})
        X = X[model.features]
        
        # Generate learning curves
        learning_curves = MLModelService.generate_learning_curves(trained_model, X, y)
        
        return jsonify({
            'success': True,
            'learning_curves': learning_curves
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating learning curves: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>/shap-analysis', methods=['GET'])
def get_shap_analysis(model_id):
    """Get SHAP analysis for model interpretability"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        if not model.model_path or not os.path.exists(model.model_path):
            return jsonify({
                'success': False,
                'message': 'Model file not found'
            }), 404
        
        # Load model and dataset
        trained_model = joblib.load(model.model_path)
        dataset = Dataset.query.get(model.dataset_id)
        dataset_path = os.path.join(current_app.config['UPLOAD_FOLDER'], dataset.filename)
        df = pd.read_csv(dataset_path)
        
        # Preprocess data
        X, y = preprocess_data(df, model.target_column, {})
        X = X[model.features]
        
        # Split for analysis
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Generate SHAP analysis
        shap_analysis = MLModelService.generate_shap_analysis(trained_model, X_train, X_test)
        
        return jsonify({
            'success': True,
            'shap_analysis': shap_analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating SHAP analysis: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>/report', methods=['GET'])
def export_model_report(model_id):
    """Export comprehensive model report"""
    try:
        report = MLModelService.export_model_report(model_id)
        if not report:
            return jsonify({
                'success': False,
                'message': 'Model not found'
            }), 404
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating report: {str(e)}'
        }), 500

@ml_bp.route('/datasets/<int:dataset_id>/clustering', methods=['POST'])
def perform_clustering(dataset_id):
    """Perform clustering analysis on dataset"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        dataset_path = os.path.join(current_app.config['UPLOAD_FOLDER'], dataset.filename)
        
        if not os.path.exists(dataset_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
        
        df = pd.read_csv(dataset_path)
        
        # Get clustering parameters
        data = request.get_json()
        n_clusters_range = data.get('n_clusters_range', (2, 10))
        
        clustering_results = MLModelService.perform_clustering_analysis(df, n_clusters_range)
        
        return jsonify({
            'success': True,
            'clustering_results': clustering_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error performing clustering: {str(e)}'
        }), 500

@ml_bp.route('/datasets/<int:dataset_id>/anomalies', methods=['POST'])
def detect_anomalies(dataset_id):
    """Detect anomalies in dataset"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        dataset_path = os.path.join(current_app.config['UPLOAD_FOLDER'], dataset.filename)
        
        if not os.path.exists(dataset_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
        
        df = pd.read_csv(dataset_path)
        
        # Get parameters
        data = request.get_json()
        contamination = data.get('contamination', 0.1)
        
        anomaly_results = MLModelService.detect_anomalies(df, contamination)
        
        return jsonify({
            'success': True,
            'anomaly_results': anomaly_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error detecting anomalies: {str(e)}'
        }), 500

@ml_bp.route('/automl/feature-selection', methods=['POST'])
def auto_feature_selection():
    """Automated feature selection with intelligent preprocessing"""
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        target_column = data.get('target_column')
        model_type = data.get('model_type', 'classification')
        k_features = data.get('k_features', 10)
        
        if not all([dataset_id, target_column]):
            return jsonify({
                'success': False,
                'message': 'dataset_id and target_column are required'
            }), 400
        
        # Load dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
            
        df = pd.read_csv(dataset.file_path)
        
        # Use smart preprocessing with comprehensive error handling
        try:
            X, y, preprocessing_info = AutoMLService.smart_preprocess_data(df, target_column, model_type)
            
            if X is None or y is None:
                error_msg = preprocessing_info.get('error', 'Unknown preprocessing error')
                return jsonify({
                    'success': False,
                    'message': f'Preprocessing failed: {error_msg}'
                }), 400
            
            if 'error' in preprocessing_info:
                return jsonify({
                    'success': False,
                    'message': preprocessing_info['error']
                }), 400
                
        except Exception as preprocess_error:
            return jsonify({
                'success': False,
                'message': f'Data preprocessing failed: {str(preprocess_error)}'
            }), 400
        
        # Perform feature selection with error handling
        try:
            selection_results = AutoMLService.auto_feature_selection(X, y, model_type, k_features)
            
            if 'error' in selection_results:
                return jsonify({
                    'success': False,
                    'message': selection_results['error']
                }), 400
                
        except Exception as selection_error:
            return jsonify({
                'success': False,
                'message': f'Feature selection failed: {str(selection_error)}'
            }), 400
        
        return jsonify({
            'success': True,
            'feature_selection': selection_results,
            'preprocessing_info': preprocessing_info
        })
        
    except Exception as e:
        print(f"AutoML Feature Selection Error: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Error in feature selection: {str(e)}'
        }), 500

@ml_bp.route('/automl/recommend-algorithms', methods=['POST'])
def recommend_algorithms():
    """Get algorithm recommendations based on data characteristics"""
    try:
        data = request.get_json()
        print("AutoML recommend-algorithms request data:", data)
        
        dataset_id = data.get('dataset_id')
        target_column = data.get('target_column')
        model_type = data.get('model_type', 'classification')
        
        if not all([dataset_id, target_column]):
            error_msg = f"Missing required fields: dataset_id={dataset_id}, target_column={target_column}"
            print(f"AutoML recommend-algorithms error: {error_msg}")
            return jsonify({
                'success': False,
                'message': 'dataset_id and target_column are required'
            }), 400
        
        # Load dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            error_msg = f"Dataset file not found for dataset_id={dataset_id}, file_path={dataset.file_path}"
            print(f"AutoML recommend-algorithms error: {error_msg}")
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
            
        df = pd.read_csv(dataset.file_path)
        
        # Validate target column exists in the dataset
        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}"
            print(f"AutoML recommend-algorithms error: {error_msg}")
            return jsonify({
                'success': False,
                'message': f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)[:10]}" + (", ..." if len(df.columns) > 10 else "")
            }), 400
        
        # Use smart preprocessing with comprehensive error handling
        try:
            X, y, preprocessing_info = AutoMLService.smart_preprocess_data(df, target_column, model_type)
            
            if X is None or y is None:
                error_msg = preprocessing_info.get('error', 'Unknown preprocessing error')
                return jsonify({
                    'success': False,
                    'message': f'Preprocessing failed: {error_msg}'
                }), 400
            
            if 'error' in preprocessing_info:
                return jsonify({
                    'success': False,
                    'message': preprocessing_info['error']
                }), 400
                
        except Exception as preprocess_error:
            return jsonify({
                'success': False,
                'message': f'Data preprocessing failed: {str(preprocess_error)}'
            }), 400
        
        # Get recommendations with error handling
        try:
            recommendations = AutoMLService.recommend_algorithms(X, y, model_type)
            
            if 'error' in recommendations:
                return jsonify({
                    'success': False,
                    'message': recommendations['error']
                }), 400
                
        except Exception as rec_error:
            return jsonify({
                'success': False,
                'message': f'Algorithm recommendation failed: {str(rec_error)}'
            }), 400
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'preprocessing_info': preprocessing_info
        })
        
    except Exception as e:
        print(f"AutoML Recommend Algorithms Error: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Error getting recommendations: {str(e)}'
        }), 500

@ml_bp.route('/batch-train', methods=['POST', 'OPTIONS'])
def batch_train_models_alias():
    """Alias for batch training route to match frontend URL pattern"""
    if request.method == 'OPTIONS':
        # Handle preflight CORS request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # Forward to the main batch train function
    return batch_train_models()

@ml_bp.route('/models/batch-train', methods=['POST', 'OPTIONS'])
def batch_train_models():
    """Train multiple models with different algorithms - Advanced version"""
    if request.method == 'OPTIONS':
        # Handle preflight CORS request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
        
    try:
        from app.services.ml_service import AutoMLService
        
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        target_column = data.get('target_column')
        model_type = data.get('model_type')
        algorithms = data.get('algorithms', [])
        
        if not all([dataset_id, target_column, model_type, algorithms]):
            return jsonify({
                'success': False,
                'message': 'dataset_id, target_column, model_type, and algorithms are required'
            }), 400
        
        # Load and validate dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
        
        # Load dataset with error handling
        try:
            file_size = os.path.getsize(dataset.file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB
                # Read in chunks for large files
                df = pd.read_csv(dataset.file_path, chunksize=10000)
                df = pd.concat(df, ignore_index=True)
            else:
                df = pd.read_csv(dataset.file_path)
            
            # Validate dataset compatibility
            compatibility_result = AutoMLService.analyze_dataset_compatibility(
                df, target_column, model_type
            )
            
            if not compatibility_result['compatible']:
                return jsonify({
                    'success': False,
                    'message': f'Dataset not compatible for training: {"; ".join(compatibility_result["issues"])}',
                    'compatibility_analysis': compatibility_result
                }), 400
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error loading dataset: {str(e)}'
            }), 400
        
        batch_results = []
        
        # Enhanced training configuration
        enhanced_config = {
            'test_size': data.get('test_size', 0.2),
            'cv_folds': data.get('cv_folds', 5),
            'max_samples': data.get('max_samples', 100000),
            'enable_sampling': data.get('enable_sampling', True),
            'optimize_memory': data.get('optimize_memory', True),
            'use_incremental_learning': data.get('use_incremental_learning', False),
            'preprocessing': {
                'scaling': data.get('scaling', 'standard'),
                'handle_missing': data.get('handle_missing', True),
                'encoding_strategy': data.get('encoding_strategy', 'auto')
            }
        }
        
        for algorithm in algorithms:
            try:
                # Create enhanced training data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                training_data = {
                    'name': f'Batch_{algorithm}_{timestamp}',
                    'description': f'Advanced batch training model using {algorithm}',
                    'dataset_id': dataset_id,
                    'algorithm': algorithm,
                    'model_type': model_type,
                    'target_column': target_column,
                    'training_config': enhanced_config
                }
                
                # Create model record
                model = MLModel(
                    name=training_data['name'],
                    description=training_data['description'],
                    algorithm=algorithm,
                    model_type=model_type,
                    dataset_id=dataset_id,
                    target_column=target_column,
                    status='training',
                    training_config=enhanced_config
                )
                
                db.session.add(model)
                db.session.commit()
                
                # Train model using advanced service
                success, message = train_model_synchronously(model.id)
                
                if success:
                    batch_results.append({
                        'model_id': model.id,
                        'algorithm': algorithm,
                        'status': 'trained',
                        'message': 'Model training completed successfully',
                        'model_name': model.name
                    })
                else:
                    batch_results.append({
                        'model_id': model.id,
                        'algorithm': algorithm,
                        'status': 'failed',
                        'message': f'Model training failed: {message}',
                        'model_name': model.name
                    })
                
            except Exception as train_error:
                batch_results.append({
                    'algorithm': algorithm,
                    'status': 'failed',
                    'error': f'Training setup failed: {str(train_error)}'
                })
        
        # Generate summary
        success_count = len([r for r in batch_results if r.get('status') == 'trained'])
        failed_count = len([r for r in batch_results if r.get('status') == 'failed'])
        
        return jsonify({
            'success': True,
            'batch_results': batch_results,
            'summary': {
                'total_models': len(algorithms),
                'successful': success_count,
                'failed': failed_count,
                'dataset_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'target_column': target_column,
                    'model_type': model_type
                }
            },
            'message': f'Batch training completed: {success_count} successful, {failed_count} failed'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error in batch training: {str(e)}',
            'error_details': str(e)
        }), 500

@ml_bp.route('/models/<int:model_id>/deploy', methods=['POST'])
def deploy_model(model_id):
    """Deploy a trained model"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        if model.status != 'trained':
            return jsonify({
                'success': False,
                'message': 'Model must be trained before deployment'
            }), 400
        
        # In a real deployment, this would involve creating API endpoints,
        # containerization, etc. For now, we'll simulate deployment
        deployment_url = f"/api/ml/models/{model_id}/predict"
        
        model.set_deployed(deployment_url)
        
        return jsonify({
            'success': True,
            'message': 'Model deployed successfully',
            'deployment_url': deployment_url,
            'model': model.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Deployment error: {str(e)}'
        }), 500

@ml_bp.route('/models/<int:model_id>/undeploy', methods=['POST'])
def undeploy_model(model_id):
    """Undeploy a model"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        model.is_deployed = False
        model.deployment_url = None
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Model undeployed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Undeployment error: {str(e)}'
        }), 500

@ml_bp.route('/automl/suggest-target-columns', methods=['POST'])
def suggest_target_columns():
    """Suggest target columns based on dataset analysis"""
    try:
        data = request.get_json()
        print("AutoML suggest-target-columns request data:", data)
        
        dataset_id = data.get('dataset_id')
        model_type = data.get('model_type', 'classification')
        
        if not dataset_id:
            error_msg = f"Missing required field: dataset_id={dataset_id}"
            print(f"AutoML suggest-target-columns error: {error_msg}")
            return jsonify({
                'success': False,
                'message': 'dataset_id is required'
            }), 400
        
        # Load dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
            
        df = pd.read_csv(dataset.file_path)
        
        # Get target column suggestions with error handling
        try:
            suggestions_result = AutoMLService.suggest_target_columns(df, model_type)
            
            if 'error' in suggestions_result:
                return jsonify({
                    'success': False,
                    'message': suggestions_result['error']
                }), 400
                
        except Exception as suggestion_error:
            return jsonify({
                'success': False,
                'message': f'Target column suggestion failed: {str(suggestion_error)}'
            }), 400
        
        return jsonify({
            'success': True,
            'suggestions': suggestions_result.get('suggestions', []),
            'total_columns': suggestions_result.get('total_columns', 0),
            'model_type': suggestions_result.get('model_type', model_type),
            'analysis_summary': suggestions_result.get('analysis_summary', {})
        })
        
    except Exception as e:
        print(f"AutoML Suggest Target Columns Error: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Error suggesting target columns: {str(e)}'
        }), 500

def train_model_synchronously(model_id):
    """Advanced synchronous function to train a model with large dataset support"""
    try:
        from app.services.ml_service import AdvancedTrainingService, AutoMLService
        
        model = MLModel.query.get(model_id)
        if not model:
            return False, 'Model not found'
            
        # Get dataset
        dataset = Dataset.query.get(model.dataset_id)
        if not dataset or not dataset.file_path or not os.path.exists(dataset.file_path):
            model.status = 'failed'
            model.training_log = 'Dataset not found or file missing'
            db.session.commit()
            return False, 'Dataset not found or file missing'
        
        # Load dataset with memory optimization
        try:
            # For large files, use chunked reading
            file_size = os.path.getsize(dataset.file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB
                # Read in chunks for large files
                df = pd.read_csv(dataset.file_path, chunksize=10000)
                df = pd.concat(df, ignore_index=True)
            else:
                df = pd.read_csv(dataset.file_path)
            
            # Validate dataset
            if df.empty:
                model.status = 'failed'
                model.training_log = 'Dataset is empty'
                db.session.commit()
                return False, 'Dataset is empty'
            
            # Check target column exists
            if model.target_column not in df.columns:
                model.status = 'failed'
                model.training_log = f'Target column "{model.target_column}" not found in dataset'
                db.session.commit()
                return False, f'Target column "{model.target_column}" not found in dataset'
            
        except Exception as e:
            error_msg = f'Error loading dataset: {str(e)}'
            model.status = 'failed'
            model.training_log = error_msg
            db.session.commit()
            return False, error_msg
        
        # Use advanced training service
        success, message = AdvancedTrainingService.train_model_with_optimization(
            model_id=model_id,
            df=df,
            target_column=model.target_column,
            algorithm=model.algorithm,
            model_type=model.model_type,
            config=model.training_config
        )
        
        return success, message
        
    except Exception as e:
        error_msg = f'Training failed: {str(e)}'
        try:
            model = MLModel.query.get(model_id)
            if model:
                model.status = 'failed'
                model.training_log = error_msg
                db.session.commit()
        except:
            pass
        return False, error_msg
        return False, str(e)

def train_model_background(model_id, app):
    """Background function to train a model"""
    try:
        with app.app_context():
            # Ensure we have a fresh database session
            from app import db
            
            model = MLModel.query.get(model_id)
            if not model:
                print(f'Model {model_id} not found')
                return
                
            # Get dataset
            dataset = Dataset.query.get(model.dataset_id)
            if not dataset or not dataset.file_path or not os.path.exists(dataset.file_path):
                model.status = 'failed'
                db.session.commit()
                print(f'Model {model_id} failed: Dataset not found or file missing')
                return
                
            print(f'Model {model_id} starting training with dataset {dataset.file_path}')
            df = pd.read_csv(dataset.file_path)
            
            # Preprocess data
            X, y, target_encoder = preprocess_data(
                df, 
                model.target_column, 
                model.training_config.get('preprocessing', {}),
                model.model_type
            )
            
            # Split data
            test_size = model.training_config.get('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Get algorithm configuration
            algorithm_config = ALGORITHM_CONFIGS[model.model_type].get(model.algorithm)
            if not algorithm_config:
                model.status = 'failed'
                db.session.commit()
                print(f'Model {model_id} failed: Algorithm {model.algorithm} not supported for {model.model_type}')
                return
                
            # Create and train model
            ml_model = algorithm_config['class'](**algorithm_config.get('params', {}))
            
            start_time = time.time()
            ml_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = ml_model.predict(X_test)
            
            # Calculate metrics
            metrics = {}
            if model.model_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            else:
                metrics['r2_score'] = r2_score(y_test, y_pred)
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # Cross-validation
            cv_folds = model.training_config.get('cv_folds', 5)
            cv_scores = cross_val_score(ml_model, X, y, cv=cv_folds)
            metrics['cv_mean'] = np.mean(cv_scores)
            metrics['cv_std'] = np.std(cv_scores)
            
            # Save model
            model_dir = get_model_directory()
            model_filename = f'{model.id}_{model.algorithm}_{model.model_type}.pkl'
            model_path = os.path.join(model_dir, model_filename)
            
            model_data = {
                'model': ml_model,
                'preprocessing': {
                    'scaler': None,  # Would store scaler if used
                    'target_encoder': target_encoder
                },
                'features': list(X.columns),
                'target_column': model.target_column,
                'model_type': model.model_type,
                'algorithm': model.algorithm
            }
            
            joblib.dump(model_data, model_path)
            
            # Update model record
            model.status = 'trained'
            model.model_path = model_path
            model.features = list(X.columns)
            model.training_time = training_time
            model.trained_at = datetime.now()
            
            # Store metrics
            for metric_name, metric_value in metrics.items():
                setattr(model, metric_name, metric_value)
            
            db.session.commit()
            print(f'Model {model_id} training completed successfully')
            
    except Exception as e:
        print(f'Model {model_id} training failed with exception: {e}')
        try:
            with app.app_context():
                from app import db
                model = MLModel.query.get(model_id)
                if model:
                    model.status = 'failed'
                    db.session.commit()
                    print(f'Model {model_id} status updated to failed')
        except Exception as db_error:
            print(f'Failed to update model {model_id} status: {db_error}')
        print(f'Model {model_id} training failed: {e}')

# Error handlers
@ml_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': 'Resource not found'
    }), 404

@ml_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500
