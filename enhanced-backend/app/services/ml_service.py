"""
ML Services for advanced machine learning
"""

import os
import json
import pickle
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# ML Libraries
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Advanced ML
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False

from app import db
from app.models.ml_model import MLModel


class MLModelService:
    """Service class for advanced ML model operations"""
    
    @staticmethod
    def generate_learning_curves(model, X, y, cv=5):
        """Generate learning curves for model performance analysis"""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                random_state=42
            )
            
            return {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
        except Exception as e:
            print(f"Error generating learning curves: {e}")
            return None
    
    @staticmethod
    def generate_validation_curves(model, X, y, param_name, param_range, cv=5):
        """Generate validation curves for hyperparameter analysis"""
        try:
            train_scores, val_scores = validation_curve(
                model, X, y, param_name=param_name, param_range=param_range,
                cv=cv, n_jobs=-1, random_state=42
            )
            
            return {
                'param_range': [str(p) for p in param_range],
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
        except Exception as e:
            print(f"Error generating validation curves: {e}")
            return None
    
    @staticmethod
    def calculate_permutation_importance(model, X, y, n_repeats=10):
        """Calculate permutation importance for feature analysis"""
        try:
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            
            importance_dict = {}
            for i, col in enumerate(X.columns):
                importance_dict[col] = {
                    'mean': float(perm_importance.importances_mean[i]),
                    'std': float(perm_importance.importances_std[i])
                }
            
            # Sort by mean importance
            return dict(sorted(importance_dict.items(), 
                             key=lambda x: x[1]['mean'], reverse=True))
        except Exception as e:
            print(f"Error calculating permutation importance: {e}")
            return None
    
    @staticmethod
    def generate_shap_analysis(model, X_train, X_test, max_samples=100):
        """Generate comprehensive SHAP analysis"""
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Limit samples for performance
            X_test_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
            
            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') else shap.Explainer(model)
            else:
                explainer = shap.Explainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_sample)
            
            # Handle multi-class classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Feature importance
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            feature_importance_dict = dict(zip(X_test_sample.columns, feature_importance))
            
            # Global explanation
            shap_summary = {
                'feature_importance': dict(sorted(feature_importance_dict.items(), 
                                                key=lambda x: x[1], reverse=True)),
                'mean_shap_values': np.mean(shap_values, axis=0).tolist(),
                'shap_values_sample': shap_values[:10].tolist() if len(shap_values) > 10 else shap_values.tolist()
            }
            
            return shap_summary
        except Exception as e:
            print(f"Error generating SHAP analysis: {e}")
            return None
    
    @staticmethod
    def perform_clustering_analysis(df, n_clusters_range=(2, 10)):
        """Perform clustering analysis on dataset"""
        try:
            # Prepare data
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_columns].fillna(df[numeric_columns].median())
            
            if X.empty:
                return None
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            results = {}
            
            # K-Means clustering with different k values
            kmeans_results = []
            for k in range(n_clusters_range[0], min(n_clusters_range[1] + 1, len(X))):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                # Calculate silhouette score
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(X_scaled, labels)
                except:
                    silhouette = None
                
                kmeans_results.append({
                    'n_clusters': k,
                    'inertia': float(kmeans.inertia_),
                    'silhouette_score': float(silhouette) if silhouette else None,
                    'cluster_centers': kmeans.cluster_centers_.tolist()
                })
            
            results['kmeans'] = kmeans_results
            
            # DBSCAN clustering
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(X_scaled)
                n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                results['dbscan'] = {
                    'n_clusters': n_clusters_dbscan,
                    'n_noise_points': n_noise,
                    'labels': dbscan_labels.tolist()
                }
            except Exception as e:
                print(f"DBSCAN error: {e}")
            
            # PCA for dimensionality reduction
            try:
                pca = PCA(n_components=min(3, X_scaled.shape[1]))
                X_pca = pca.fit_transform(X_scaled)
                
                results['pca'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                    'components': X_pca.tolist()
                }
            except Exception as e:
                print(f"PCA error: {e}")
            
            return results
        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            return None
    
    @staticmethod
    def detect_anomalies(df, contamination=0.1):
        """Detect anomalies in dataset"""
        if not ANOMALY_DETECTION_AVAILABLE:
            return None
        
        try:
            # Prepare data
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_columns].fillna(df[numeric_columns].median())
            
            if X.empty:
                return None
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            results = {}
            
            # Isolation Forest
            try:
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                iso_anomalies = iso_forest.fit_predict(X_scaled)
                iso_scores = iso_forest.score_samples(X_scaled)
                
                results['isolation_forest'] = {
                    'anomalies': (iso_anomalies == -1).tolist(),
                    'anomaly_scores': iso_scores.tolist(),
                    'n_anomalies': int(np.sum(iso_anomalies == -1))
                }
            except Exception as e:
                print(f"Isolation Forest error: {e}")
            
            # One-Class SVM
            try:
                oc_svm = OneClassSVM(nu=contamination)
                svm_anomalies = oc_svm.fit_predict(X_scaled)
                
                results['one_class_svm'] = {
                    'anomalies': (svm_anomalies == -1).tolist(),
                    'n_anomalies': int(np.sum(svm_anomalies == -1))
                }
            except Exception as e:
                print(f"One-Class SVM error: {e}")
            
            return results
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None
    
    @staticmethod
    def generate_model_insights(model_id: int):
        """Generate comprehensive insights for a trained model"""
        try:
            model = MLModel.query.get(model_id)
            if not model or model.status != 'trained':
                return None
            
            insights = {
                'model_info': model.to_dict(),
                'performance_analysis': {},
                'feature_analysis': {},
                'recommendations': []
            }
            
            # Performance analysis
            primary_metric = model.get_primary_metric()
            insights['performance_analysis']['primary_metric'] = primary_metric
            
            # Feature analysis
            if model.feature_importance:
                top_features = list(model.feature_importance.keys())[:10]
                insights['feature_analysis']['top_features'] = top_features
                insights['feature_analysis']['feature_count'] = len(model.features or [])
            
            # Recommendations based on performance
            if model.model_type == 'classification':
                if model.accuracy and model.accuracy < 0.7:
                    insights['recommendations'].append("Consider feature engineering or trying different algorithms")
                if model.f1_score and model.f1_score < 0.6:
                    insights['recommendations'].append("Model shows low F1 score, consider addressing class imbalance")
            else:
                if model.r2_score and model.r2_score < 0.7:
                    insights['recommendations'].append("Low R² score, consider feature selection or regularization")
                if model.mse and model.mae and model.mse > (model.mae * 2):
                    insights['recommendations'].append("High MSE relative to MAE suggests outliers in predictions")
            
            # Cross-validation insights
            if model.cv_std and model.cv_std > 0.1:
                insights['recommendations'].append("High CV standard deviation suggests model instability")
            
            # Training time insights
            if model.training_time and model.training_time > 300:  # 5 minutes
                insights['recommendations'].append("Consider model optimization for faster training")
            
            return insights
        except Exception as e:
            print(f"Error generating model insights: {e}")
            return None
    
    @staticmethod
    def export_model_report(model_id: int) -> Dict[str, Any]:
        """Generate comprehensive model report for export"""
        try:
            model = MLModel.query.get(model_id)
            if not model:
                return None
            
            report = {
                'model_metadata': {
                    'name': model.name,
                    'description': model.description,
                    'algorithm': model.algorithm,
                    'model_type': model.model_type,
                    'version': model.version,
                    'created_at': model.created_at.isoformat() if model.created_at else None,
                    'trained_at': model.trained_at.isoformat() if model.trained_at else None
                },
                'training_configuration': model.training_config or {},
                'dataset_info': {
                    'dataset_id': model.dataset_id,
                    'target_column': model.target_column,
                    'features': model.features or [],
                    'feature_count': len(model.features or [])
                },
                'performance_metrics': {
                    'training_metrics': model.training_metrics or {},
                    'validation_metrics': model.validation_metrics or {},
                    'test_metrics': model.test_metrics or {}
                },
                'model_analysis': {
                    'feature_importance': model.feature_importance or {},
                    'shap_values': model.shap_values or {},
                    'permutation_importance': model.permutation_importance or {}
                },
                'cross_validation': {
                    'cv_scores': model.cv_scores or [],
                    'cv_mean': model.cv_mean,
                    'cv_std': model.cv_std
                },
                'model_statistics': {
                    'training_time': model.training_time,
                    'model_size': model.model_size,
                    'prediction_count': model.prediction_count,
                    'download_count': model.download_count
                },
                'training_log': model.training_log or "",
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return report
        except Exception as e:
            print(f"Error generating model report: {e}")
            return None


class AutoMLService:
    """Service for automated machine learning"""
    
    @staticmethod
    def smart_preprocess_data(df, target_column, model_type='classification', max_samples=None):
        """
        Advanced data preprocessing with large dataset support and comprehensive error handling
        """
        try:
            import time
            import numpy as np
            import pandas as pd
            
            preprocessing_info = {
                'categorical_columns_processed': [],
                'high_cardinality_columns': [],
                'missing_values_handled': False,
                'encoding_strategy': 'auto',
                'scaling_applied': False,
                'warnings': [],
                'dataset_sampled': False,
                'sample_size': None,
                'original_size': len(df),
                'memory_usage_mb': 0,
                'processing_time': 0,
                'transformations': []
            }
            
            start_time = time.time()
            
            # Calculate memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            preprocessing_info['memory_usage_mb'] = memory_usage / 1024 / 1024
            
            # Validate input data
            if df.empty:
                return None, None, {'error': 'Empty dataset provided'}
            
            if target_column not in df.columns:
                return None, None, {
                    'error': f'Target column "{target_column}" not found in dataset.',
                    'available_columns': list(df.columns),
                    'suggestion': 'Please select a column that exists in the dataset.'
                }
            
            # Check target column data quality
            target_data = df[target_column]
            
            # Handle completely empty target columns or all NaN values
            if target_data.isna().all():
                return None, None, {
                    'error': f'Target column "{target_column}" contains only missing values.',
                    'suggestion': 'Please select a column that contains actual data values.'
                }
            
            # Handle empty strings or whitespace in target
            if target_data.dtype == 'object':
                empty_strings = (target_data.str.strip() == '').sum() if hasattr(target_data, 'str') else 0
                if empty_strings == len(target_data):
                    return None, None, {
                        'error': f'Target column "{target_column}" contains only empty strings.',
                        'suggestion': 'Please select a column with meaningful values.'
                    }
            
            # Check for missing values in target
            missing_values = target_data.isna().sum()
            if missing_values > 0:
                missing_percent = (missing_values / len(target_data)) * 100
                if missing_percent > 30:  # More than 30% missing
                    return None, None, {
                        'error': f'Target column has too many missing values ({missing_percent:.1f}%).',
                        'suggestion': 'Select another column with fewer missing values.'
                    }
                else:
                    preprocessing_info['warnings'].append(f'Target column has {missing_percent:.1f}% missing values')
                    # We'll handle this later by removing rows with missing targets
            
            # Check for target column data type compatibility
            if model_type == 'classification':
                # For classification, check if we have enough classes
                valid_target_data = target_data.dropna()
                
                # Try to convert numeric-like strings to numbers if needed
                if valid_target_data.dtype == 'object':
                    try:
                        # Check if this column might be numeric
                        numeric_conversion = pd.to_numeric(valid_target_data, errors='coerce')
                        if not numeric_conversion.isna().all():
                            # If successful conversion and values are few, keep it as categorical
                            unique_numeric = numeric_conversion.dropna().unique()
                            if len(unique_numeric) <= 20:  # Reasonable number of classes
                                valid_target_data = numeric_conversion.dropna()
                                preprocessing_info['warnings'].append(
                                    f'Target column converted from string to numeric with {len(unique_numeric)} unique values'
                                )
                            elif model_type == 'classification':
                                # Too many unique values for classification, but might be good for regression
                                return None, None, {
                                    'error': f'Target column has too many unique numeric values ({len(unique_numeric)}) for classification.',
                                    'suggestion': 'This column might be better suited for regression. Try changing the model type to regression.'
                                }
                    except Exception as e:
                        # Keep as string if conversion fails
                        preprocessing_info['warnings'].append(f'Could not convert target to numeric: {str(e)}')
                
                # For numeric data in classification, bin continuous values if needed
                if pd.api.types.is_numeric_dtype(valid_target_data) and valid_target_data.nunique() > 20:
                    # Automatically bin the continuous variable into categories
                    try:
                        logging.info(f"Binning continuous variable '{target_column}' with {valid_target_data.nunique()} values into categories for classification")
                        # Determine optimal number of bins (between 2 and 10)
                        n_bins = min(10, max(2, int(valid_target_data.nunique() / 10)))
                        
                        # Create bins using pandas qcut (quantile-based) for even distribution
                        valid_target_data = pd.qcut(valid_target_data, n_bins, duplicates='drop')
                        y = pd.qcut(df[target_column], n_bins, duplicates='drop')
                        
                        # Convert to string categories for better interpretability
                        valid_target_data = valid_target_data.astype(str)
                        y = y.astype(str)
                        
                        preprocessing_info['warnings'].append(
                            f'Automatically binned continuous variable "{target_column}" into {n_bins} categories for classification.'
                        )
                        preprocessing_info['transformations'].append(
                            f'Binned "{target_column}" into {n_bins} quantile-based categories'
                        )
                    except Exception as e:
                        # If binning fails, return the original error
                        logging.error(f"Failed to bin continuous variable: {str(e)}")
                        return None, None, {
                            'error': f'Target column "{target_column}" has {valid_target_data.nunique()} unique numeric values, which is too many for classification.',
                            'suggestion': 'Consider using regression instead of classification for this numeric target, or create discrete bins from this continuous variable manually.'
                        }
                
                unique_classes = valid_target_data.unique()
                
                if len(unique_classes) < 2:
                    return None, None, {
                        'error': f'Target column must have at least 2 unique values for classification.',
                        'current_values': len(unique_classes),
                        'suggestion': 'Select another column with multiple distinct values.'
                    }
                elif len(unique_classes) > 100:
                    return None, None, {
                        'error': f'Target column has too many unique values ({len(unique_classes)}) for classification.',
                        'suggestion': 'Select another column with fewer unique values or use regression instead.'
                    }
                
                # Check for unsuitable identifier columns
                if target_column.lower() in ['id', 'symbol', 'code', 'identifier', 'key', 'uuid', 'guid', 'index']:
                    # Check if column seems to be an identifier
                    unique_ratio = len(unique_classes) / len(valid_target_data)
                    if unique_ratio > 0.8:  # More than 80% unique values
                        return None, None, {
                            'error': f'Target column "{target_column}" appears to be an identifier which is not suitable for classification.',
                            'suggestion': 'Select a column that represents a category or class you want to predict, not an identifier.'
                        }
                
                # Check if classification target has a uniform distribution (may be meaningless)
                if len(unique_classes) > 1:
                    value_counts = valid_target_data.value_counts(normalize=True)
                    max_freq = value_counts.max()
                    if max_freq > 0.95:  # One class has more than 95% of the data
                        preprocessing_info['warnings'].append(f'Target column is highly imbalanced ({max_freq:.1%} dominated by one class)')
                    
            elif model_type == 'regression':
                # For regression, check if target is numeric
                if not pd.api.types.is_numeric_dtype(target_data):
                    # Try to convert to numeric
                    try:
                        numeric_target = pd.to_numeric(target_data, errors='coerce')
                        # Check if conversion was successful (not all NaN)
                        if numeric_target.isna().sum() < len(numeric_target) * 0.3:  # Allow up to 30% NaN
                            # Use the converted values
                            preprocessing_info['warnings'].append(
                                f'Target column "{target_column}" was converted to numeric for regression with '
                                f'{numeric_target.isna().sum()} missing values'
                            )
                            # We'll update the dataframe with the converted target later
                            df[target_column] = numeric_target
                            target_data = numeric_target
                        else:
                            return None, None, {
                                'error': f'Target column contains too many non-numeric values that cannot be converted for regression.',
                                'current_type': str(target_data.dtype),
                                'suggestion': 'Select a numeric column or use classification instead.'
                            }
                    except:
                        return None, None, {
                            'error': f'Target column must be numeric for regression.',
                            'current_type': str(target_data.dtype),
                            'suggestion': 'Select a numeric column or use classification instead.'
                        }
                
                # Check for constant values
                if target_data.dropna().nunique() <= 1:
                    return None, None, {
                        'error': f'Target column has only one unique value, which is not suitable for regression.',
                        'suggestion': 'Select a column with varying numeric values.'
                    }
            
            # Check for minimum rows
            if len(df) < 10:
                return None, None, {'error': 'Dataset too small (minimum 10 rows required)'}
            
            # Preprocessing for date/time columns
            datetime_columns = []
            for col in df.columns:
                if col != target_column:
                    # Check if column might be date/time
                    if df[col].dtype == 'object':
                        # Try to parse as datetime
                        try:
                            pd.to_datetime(df[col], errors='raise')
                            datetime_columns.append(col)
                        except:
                            pass
                    elif 'datetime' in str(df[col].dtype) or 'timestamp' in str(df[col].dtype):
                        datetime_columns.append(col)
            
            # Extract useful features from datetime columns
            for col in datetime_columns:
                try:
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                    # Only process if successful conversion
                    if not dt_series.isna().all():
                        # Extract month, day, hour as these are often relevant
                        df[f'{col}_month'] = dt_series.dt.month
                        df[f'{col}_day'] = dt_series.dt.day
                        if 'time' in col.lower() or any(dt_series.dt.hour.unique() > 0):
                            df[f'{col}_hour'] = dt_series.dt.hour
                        # Drop original datetime column if it's not the target column
                        if col != target_column:
                            df = df.drop(columns=[col])
                        preprocessing_info['warnings'].append(f'Extracted features from datetime column: {col}')
                except Exception as e:
                    preprocessing_info['warnings'].append(f'Failed to process datetime column {col}: {str(e)}')
            
            # Handle large datasets by sampling
            if max_samples is None:
                # Adaptive sampling based on memory usage
                if memory_usage > 500 * 1024 * 1024:  # 500MB
                    max_samples = 100000
                elif memory_usage > 100 * 1024 * 1024:  # 100MB
                    max_samples = 50000
                elif len(df) > 100000:  # 100k rows
                    max_samples = 100000
                
            # Apply sampling if needed
            if max_samples and len(df) > max_samples:
                    preprocessing_info['sample_size'] = len(df)
                    preprocessing_info['warnings'].append(f'Dataset sampled from {preprocessing_info["original_size"]} to {len(df)} rows for performance')
            
            # Separate features and target
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()
            
            # Check if we have any features
            if X.empty or len(X.columns) == 0:
                return None, None, {'error': 'No feature columns found in dataset'}
            
            # Handle missing values in target
            if y.isna().any():
                missing_target_count = y.isna().sum()
                if missing_target_count > len(y) * 0.5:  # More than 50% missing
                    return None, None, {'error': f'Target column has too many missing values ({missing_target_count}/{len(y)})'}
                
                # Remove rows with missing target
                valid_mask = ~y.isna()
                X = X[valid_mask]
                y = y[valid_mask]
                preprocessing_info['warnings'].append(f'Removed {missing_target_count} rows with missing target values')
            
            # Validate target column for classification
            if model_type == 'classification':
                unique_values = y.nunique()
                if unique_values < 2:
                    return None, None, {'error': f'Target column has only {unique_values} unique value(s). Classification requires at least 2 classes.'}
                
                if unique_values > 100:
                    return None, None, {'error': f'Target column has {unique_values} unique values. This seems like a regression problem or needs different preprocessing.'}
                
                # Check class distribution
                class_counts = y.value_counts()
                min_class_count = class_counts.min()
                if min_class_count < 2:
                    return None, None, {'error': f'Some classes have only {min_class_count} sample(s). Each class needs at least 2 samples.'}
                
                # Check for class imbalance
                max_class_count = class_counts.max()
                imbalance_ratio = max_class_count / min_class_count
                if imbalance_ratio > 10:
                    preprocessing_info['warnings'].append(f'Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1). Consider using balanced algorithms.')
            
            # Validate target column for regression
            if model_type == 'regression':
                if not pd.api.types.is_numeric_dtype(y):
                    # Try to convert to numeric
                    try:
                        y = pd.to_numeric(y, errors='coerce')
                        if y.isna().any():
                            return None, None, {'error': 'Target column contains non-numeric values that cannot be converted for regression'}
                    except:
                        return None, None, {'error': 'Target column must be numeric for regression'}
                
                # Check for reasonable variance
                if y.std() == 0:
                    return None, None, {'error': 'Target column has no variance (all values are the same). Cannot train regression model.'}
            
            # Handle missing values in features efficiently
            if X.isna().any().any():
                preprocessing_info['missing_values_handled'] = True
                
                # Check for columns with too many missing values
                missing_percentages = (X.isna().sum() / len(X)) * 100
                problematic_columns = missing_percentages[missing_percentages > 70]
                if len(problematic_columns) > 0:
                    X = X.drop(columns=problematic_columns.index)
                    preprocessing_info['warnings'].append(f'Dropped {len(problematic_columns)} columns with >70% missing values')
                
                # For numeric columns, fill with median (faster than mean)
                numeric_columns = X.select_dtypes(include=['number']).columns
                if len(numeric_columns) > 0:
                    # Use vectorized operations for better performance
                    for col in numeric_columns:
                        if X[col].isna().any():
                            X[col] = X[col].fillna(X[col].median())
                
                # For categorical columns, fill with mode or 'Unknown'
                categorical_columns = X.select_dtypes(exclude=['number']).columns
                for col in categorical_columns:
                    if X[col].isna().any():
                        mode_values = X[col].mode()
                        fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                        X[col].fillna(fill_value, inplace=True)
                        
                preprocessing_info['warnings'].append(f'Filled missing values in {len(X.columns)} columns')
            
            # Handle categorical variables intelligently
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                preprocessing_info['categorical_columns_processed'] = list(categorical_columns)
                
                # Enhanced handling for categorical columns
                from sklearn.preprocessing import LabelEncoder
                
                for col in categorical_columns:
                    # Check for high cardinality
                    unique_count = X[col].nunique()
                    unique_ratio = unique_count / len(X)
                    
                    # Column appears to be an identifier or free text
                    if unique_count > 50 and unique_ratio > 0.5:
                        preprocessing_info['high_cardinality_columns'].append(f"{col} ({unique_count} unique values)")
                        X = X.drop(columns=[col])
                        preprocessing_info['warnings'].append(f'Dropped high cardinality column: {col} with {unique_count} unique values')
                    # Reasonable cardinality for ML
                    else:
                        try:
                            # Convert to string to handle mixed types
                            X[col] = X[col].astype(str)
                            # Handle empty strings
                            X[col] = X[col].replace('', 'Unknown')
                            # Encode the column
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col])
                        except Exception as e:
                            # If encoding fails, drop the column
                            X = X.drop(columns=[col])
                            preprocessing_info['warnings'].append(f'Failed to encode {col}, column dropped: {str(e)}')
                
                preprocessing_info['encoding_strategy'] = 'label_encoding'
            
            # Check for constant features
            constant_features = X.columns[X.nunique() <= 1]
            if len(constant_features) > 0:
                X = X.drop(columns=constant_features)
                preprocessing_info['warnings'].append(f'Dropped constant features: {list(constant_features)}')
                
                if len(X.columns) == 0:
                    return None, None, {'error': 'No valid features remaining after preprocessing'}
            
            # Ensure all features are numeric
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                # Try to convert to numeric
                for col in non_numeric_cols:
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        if X[col].isna().any():
                            # If conversion creates NaN, fill with median
                            X[col].fillna(X[col].median(), inplace=True)
                    except:
                        # If conversion fails, drop the column
                        X = X.drop(columns=[col])
                        preprocessing_info['warnings'].append(f'Dropped non-numeric column: {col}')
            
            # Final validation
            if X.empty or len(X.columns) == 0:
                return None, None, {'error': 'No valid features remaining after preprocessing'}
            
            if len(X) != len(y):
                return None, None, {'error': 'Feature and target lengths do not match after preprocessing'}
            
            # Handle infinite values efficiently
            if np.isinf(X.values).any():
                X = X.replace([np.inf, -np.inf], np.nan)
                # Use vectorized median fill
                numeric_cols = X.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if X[col].isna().any():
                        X[col] = X[col].fillna(X[col].median())
                preprocessing_info['warnings'].append('Replaced infinite values with median')
            
            # Handle numeric string features that should be numeric
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        if not X[col].isna().all():
                            # Fill any new NaNs with median
                            X[col] = X[col].fillna(X[col].median())
                            preprocessing_info['warnings'].append(f'Converted column {col} from string to numeric')
                    except:
                        pass
            
            # Feature scaling for all numeric columns
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                numeric_cols = X.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                    preprocessing_info['scaling_applied'] = True
                    preprocessing_info['warnings'].append('Applied standard scaling to numeric features')
            except Exception as e:
                preprocessing_info['warnings'].append(f'Scaling failed: {str(e)}')
            
            # Calculate processing time
            preprocessing_info['processing_time'] = time.time() - start_time
            
            return X, y, preprocessing_info
            
        except Exception as e:
            error_msg = f'Preprocessing failed: {str(e)}'
            print(f"AutoMLService preprocessing error: {error_msg}")
            return None, None, {'error': error_msg}
    
    @staticmethod
    def auto_feature_selection(X, y, model_type='classification', k=10):
        """Automated feature selection with intelligent preprocessing"""
        try:
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
            
            # Validate inputs
            if X is None or y is None:
                return {'error': 'Invalid input data'}
            
            if hasattr(X, 'empty') and X.empty:
                return {'error': 'Empty dataset provided'}
            
            if len(y) == 0:
                return {'error': 'Empty target variable'}
            
            if hasattr(X, 'shape') and X.shape[0] != len(y):
                return {'error': 'X and y have different lengths'}
            
            # Ensure we don't select more features than available
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X.columns)
            max_features = min(k, n_features)
            if max_features <= 0:
                return {'error': 'No features available for selection'}
            
            # Choose appropriate scoring function
            if model_type == 'classification':
                score_funcs = [
                    ('f_classif', f_classif),
                    ('mutual_info', mutual_info_classif)
                ]
            else:
                score_funcs = [
                    ('f_regression', f_regression),
                    ('mutual_info', mutual_info_regression)
                ]
            
            results = {}
            
            for func_name, score_func in score_funcs:
                try:
                    # Handle edge cases for mutual information
                    if 'mutual_info' in func_name:
                        # Mutual info requires at least 2 samples per class
                        if model_type == 'classification':
                            unique_classes, counts = np.unique(y, return_counts=True)
                            if any(counts < 2):
                                continue  # Skip this method
                    
                    selector = SelectKBest(score_func=score_func, k=max_features)
                    X_selected = selector.fit_transform(X, y)
                    
                    # Get feature names
                    if hasattr(X, 'columns'):
                        selected_features = X.columns[selector.get_support()].tolist()
                        all_features = X.columns.tolist()
                    else:
                        selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
                        all_features = [f'feature_{i}' for i in range(X.shape[1])]
                    
                    # Handle potential NaN scores
                    scores = selector.scores_
                    if np.any(np.isnan(scores)):
                        scores = np.nan_to_num(scores, nan=0.0)
                    
                    feature_scores = dict(zip(all_features, scores))
                    
                    results[func_name] = {
                        'selected_features': selected_features,
                        'feature_scores': {k: float(v) for k, v in feature_scores.items()},
                        'n_features': len(selected_features)
                    }
                except Exception as e:
                    print(f"Error in feature selection method {func_name}: {e}")
                    results[func_name] = {
                        'error': str(e),
                        'selected_features': [],
                        'feature_scores': {},
                        'n_features': 0
                    }
            
            return results
        except Exception as e:
            print(f"Error in auto feature selection: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def recommend_algorithms(X, y, model_type='classification'):
        """Recommend best algorithms based on data characteristics"""
        try:
            import numpy as np
            import pandas as pd
            
            if X is None or y is None:
                return {'error': 'Invalid input data'}
            
            # Ensure we have valid input data
            if hasattr(X, 'shape'):
                n_samples = X.shape[0]
                n_features = X.shape[1]
            else:
                n_samples = len(X)
                n_features = len(X.columns) if hasattr(X, 'columns') else 0
            
            # Handle edge cases with empty data
            if n_samples == 0 or n_features == 0:
                return {'error': 'Empty dataset provided'}
            
            # Get unique target values with better error handling
            try:
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    # For pandas types
                    unique_values = y.dropna().unique()
                else:
                    # For numpy arrays
                    unique_values = np.unique(y[~pd.isna(y)])
                
                n_classes = len(unique_values) if model_type == 'classification' else None
            except Exception as e:
                print(f"Error determining unique values: {e}")
                n_classes = None
            
            recommendations = []
            explanations = []
            
            # Data size-based recommendations
            if n_samples < 1000:
                if model_type == 'classification':
                    recommendations.extend(['naive_bayes', 'knn', 'logistic_regression', 'decision_tree'])
                    explanations.append('Small dataset: Simple algorithms work well')
                else:
                    recommendations.extend(['linear_regression', 'ridge', 'knn', 'decision_tree'])
                    explanations.append('Small dataset: Linear models and simple algorithms recommended')
            elif n_samples < 10000:
                if model_type == 'classification':
                    recommendations.extend(['random_forest', 'svm', 'logistic_regression', 'gradient_boosting'])
                    explanations.append('Medium dataset: Ensemble methods and SVMs perform well')
                else:
                    recommendations.extend(['random_forest', 'svr', 'ridge', 'gradient_boosting'])
                    explanations.append('Medium dataset: Ensemble methods and regularized models recommended')
            else:
                if model_type == 'classification':
                    recommendations.extend(['gradient_boosting', 'random_forest', 'xgboost', 'lightgbm'])
                    explanations.append('Large dataset: Advanced ensemble methods excel')
                else:
                    recommendations.extend(['gradient_boosting', 'random_forest', 'xgboost', 'lightgbm'])
                    explanations.append('Large dataset: Gradient boosting methods recommended')
            
            # Feature count-based recommendations
            if n_features > 100:
                explanations.append(f'High-dimensional data ({n_features} features): Feature selection recommended')
                if model_type == 'classification':
                    recommendations.extend(['logistic_regression', 'ridge_classifier'])
                    explanations.append('For high-dimensional data: Regularized models prevent overfitting')
                else:
                    recommendations.extend(['ridge', 'lasso', 'elastic_net'])
                    explanations.append('For high-dimensional data: Regularization essential')
            
            # Class imbalance check for classification
            if model_type == 'classification' and n_classes and n_classes > 1:
                try:
                    if isinstance(y, (pd.Series, pd.DataFrame)):
                        class_counts = y.value_counts()
                        if len(class_counts) > 1:
                            imbalance_ratio = class_counts.max() / class_counts.min()
                            if imbalance_ratio > 5:
                                recommendations.extend(['balanced_random_forest', 'balanced_bagging'])
                                explanations.append(f'Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)')
                except Exception as e:
                    print(f"Error checking class imbalance: {e}")
                    # Add some safe recommendations for imbalanced data anyway
                    recommendations.extend(['balanced_random_forest'])
                    explanations.append('Added balanced algorithms as a precaution')
            
            # Performance vs interpretability trade-offs
            interpretable_models = ['logistic_regression', 'decision_tree', 'linear_regression', 'ridge']
            high_performance_models = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
            
            # Remove duplicates while preserving order
            recommendations = list(dict.fromkeys(recommendations))
            
            # Add performance tiers
            performance_tiers = {
                'high_performance': [alg for alg in recommendations if alg in high_performance_models],
                'interpretable': [alg for alg in recommendations if alg in interpretable_models],
                'balanced': [alg for alg in recommendations if alg not in high_performance_models and alg not in interpretable_models]
            }
            
            # Data quality assessment
            data_quality = {
                'sample_size': 'small' if n_samples < 1000 else 'medium' if n_samples < 10000 else 'large',
                'feature_count': n_features,
                'samples_per_feature': n_samples / max(n_features, 1),
                'class_count': n_classes if n_classes else 'N/A'
            }
            
            return {
                'recommended_algorithms': recommendations[:8],  # Top 8 recommendations
                'performance_tiers': performance_tiers,
                'explanations': explanations,
                'data_characteristics': {
                    'n_samples': int(n_samples),
                    'n_features': int(n_features),
                    'n_classes': int(n_classes) if n_classes is not None else None,
                    'sample_to_feature_ratio': float(n_samples / max(n_features, 1)),
                    'data_quality': data_quality
                }
            }
        except Exception as e:
            import traceback
            print(f"Error in algorithm recommendation: {e}")
            print(traceback.format_exc())
            return {
                'error': str(e),
                'error_details': traceback.format_exc(),
                'recommended_algorithms': ['random_forest', 'gradient_boosting', 'logistic_regression'],
                'fallback_note': 'Error occurred during analysis, returning safe default recommendations'
            }
    
    @staticmethod
    def suggest_target_columns(df, model_type='classification'):
        """Suggest target columns based on advanced data analysis"""
        try:
            if df.empty:
                return {'error': 'Empty dataset provided'}
            
            suggestions = []
            
            # Enhanced target column name patterns
            classification_patterns = [
                'target', 'label', 'class', 'category', 'type', 'outcome', 
                'result', 'status', 'approved', 'success', 'fraud', 'churn',
                'predict', 'response', 'y', 'survived', 'diagnosis', 'grade',
                'risk', 'default', 'spam', 'positive', 'negative', 'winner'
            ]
            
            regression_patterns = [
                'price', 'cost', 'value', 'amount', 'salary', 'revenue', 'income',
                'score', 'rating', 'age', 'time', 'duration', 'length', 'weight',
                'size', 'count', 'number', 'quantity', 'total', 'sum', 'avg',
                'temperature', 'volume', 'area', 'distance', 'speed', 'rate',
                'y', 'target', 'sales', 'profit', 'loss', 'expense'
            ]
            
            patterns = classification_patterns if model_type == 'classification' else regression_patterns
            
            for column in df.columns:
                confidence = 0.0
                reasons = []
                
                # Check name patterns (enhanced scoring)
                column_lower = column.lower()
                pattern_matches = []
                for pattern in patterns:
                    if pattern in column_lower:
                        pattern_matches.append(pattern)
                
                if pattern_matches:
                    # Higher confidence for exact matches
                    exact_matches = [p for p in pattern_matches if p == column_lower]
                    if exact_matches:
                        confidence += 0.6
                        reasons.append(f"Exact match with '{exact_matches[0]}' pattern")
                    else:
                        confidence += 0.4
                        reasons.append(f"Contains '{pattern_matches[0]}' keyword")
                
                # Check position (last columns are often targets)
                column_index = list(df.columns).index(column)
                total_cols = len(df.columns)
                if column_index >= total_cols - 3:
                    position_bonus = 0.3 if column_index == total_cols - 1 else 0.2
                    confidence += position_bonus
                    reasons.append("Located at end of dataset")
                
                # Enhanced data type and distribution analysis
                try:
                    col_data = df[column].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    unique_values = col_data.nunique()
                    total_values = len(col_data)
                    unique_ratio = unique_values / total_values
                    
                    if model_type == 'classification':
                        # Optimal classification targets
                        if unique_values == 2:  # Binary classification
                            confidence += 0.5
                            reasons.append("Binary values (ideal for binary classification)")
                        elif 2 < unique_values <= 10:  # Multi-class
                            confidence += 0.4
                            reasons.append(f"Few unique values ({unique_values}) - good for multi-class")
                        elif 10 < unique_values <= 50:  # Manageable multi-class
                            confidence += 0.2
                            reasons.append(f"Moderate unique values ({unique_values}) - manageable multi-class")
                        elif unique_ratio < 0.05:  # Low cardinality
                            confidence += 0.3
                            reasons.append(f"Low cardinality ({unique_values} unique values)")
                        
                        # Check for typical classification value patterns
                        if col_data.dtype == 'object':
                            common_values = col_data.value_counts().index.tolist()[:5]
                            classification_words = ['yes', 'no', 'true', 'false', 'positive', 'negative', 
                                                  'high', 'low', 'medium', 'good', 'bad', 'pass', 'fail',
                                                  'approved', 'rejected', 'success', 'failure']
                            
                            matches = [str(val).lower() for val in common_values 
                                     if any(word in str(val).lower() for word in classification_words)]
                            if matches:
                                confidence += 0.3
                                reasons.append(f"Contains typical classification values: {matches[:2]}")
                    
                    else:  # regression
                        # Good regression targets are numeric with reasonable variance
                        if pd.api.types.is_numeric_dtype(col_data):
                            confidence += 0.4
                            reasons.append("Numeric data type")
                            
                            # Check for reasonable range and variance
                            if col_data.std() > 0:
                                cv = col_data.std() / abs(col_data.mean()) if col_data.mean() != 0 else 0
                                if 0.1 < cv < 2:  # Reasonable coefficient of variation
                                    confidence += 0.3
                                    reasons.append("Good numerical variance")
                                elif cv > 0:
                                    confidence += 0.1
                                    reasons.append("Has numerical variance")
                            
                            # Check for continuous distribution
                            if unique_ratio > 0.8:  # High uniqueness suggests continuous variable
                                confidence += 0.2
                                reasons.append("High uniqueness (continuous-like)")
                        
                        # Check for typical regression value ranges
                        if pd.api.types.is_numeric_dtype(col_data):
                            if col_data.min() >= 0:  # Non-negative values
                                confidence += 0.1
                                reasons.append("Non-negative values")
                            
                            # Check for reasonable scale
                            data_range = col_data.max() - col_data.min()
                            if data_range > 0:
                                confidence += 0.1
                                reasons.append("Has numerical range")
                
                except Exception as e:
                    # Skip columns with analysis errors
                    continue
                
                # Penalize obvious ID columns
                id_patterns = ['id', 'index', 'key', 'uuid', 'guid', 'pk', 'primary']
                if any(id_word in column_lower for id_word in id_patterns):
                    confidence *= 0.1
                    reasons.append("Appears to be an ID column")
                
                # Penalize timestamp columns for most use cases
                if any(time_word in column_lower for time_word in ['timestamp', 'datetime', 'date', 'time']):
                    confidence *= 0.3
                    reasons.append("Appears to be a timestamp column")
                
                # Only suggest columns with decent confidence
                if confidence > 0.15:
                    suggestions.append({
                        'column_name': column,
                        'confidence': min(confidence, 1.0),
                        'reasons': reasons[:3],  # Top 3 reasons
                        'data_type': str(df[column].dtype),
                        'unique_values': int(df[column].nunique()),  # Convert to native int
                        'missing_values': int(df[column].isna().sum()),  # Convert to native int
                        'recommended': confidence >= 0.5
                    })
            
            # Sort by confidence and return top suggestions
            suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Add additional metadata
            result = {
                'suggestions': suggestions[:10],  # Top 10 suggestions
                'total_columns': int(len(df.columns)),  # Convert to native int
                'model_type': model_type,
                'analysis_summary': {
                    'high_confidence': int(len([s for s in suggestions if s['confidence'] >= 0.7])),
                    'medium_confidence': int(len([s for s in suggestions if 0.4 <= s['confidence'] < 0.7])),
                    'low_confidence': int(len([s for s in suggestions if 0.15 <= s['confidence'] < 0.4]))
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error suggesting target columns: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def analyze_dataset_compatibility(df, target_column, model_type='classification'):
        """Analyze dataset compatibility and provide comprehensive feedback"""
        try:
            analysis_result = {
                'compatible': True,
                'issues': [],
                'warnings': [],
                'recommendations': [],
                'dataset_info': {},
                'target_info': {}
            }
            
            # Basic dataset info
            analysis_result['dataset_info'] = {
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'column_names': list(df.columns),
                'memory_usage': int(df.memory_usage(deep=True).sum()),
                'missing_values': int(df.isna().sum().sum())
            }
            
            # Check basic requirements
            if df.empty:
                analysis_result['compatible'] = False
                analysis_result['issues'].append('Dataset is empty')
                return analysis_result
            
            if target_column not in df.columns:
                analysis_result['compatible'] = False
                analysis_result['issues'].append(f'Target column "{target_column}" not found')
                return analysis_result
            
            if len(df) < 10:
                analysis_result['compatible'] = False
                analysis_result['issues'].append(f'Dataset too small ({len(df)} rows). Need at least 10 rows.')
                return analysis_result
            
            # Analyze target column
            target_data = df[target_column].dropna()
            analysis_result['target_info'] = {
                'name': target_column,
                'dtype': str(df[target_column].dtype),
                'unique_values': int(df[target_column].nunique()),
                'missing_values': int(df[target_column].isna().sum()),
                'missing_percentage': float((df[target_column].isna().sum() / len(df)) * 100)
            }
            
            # Check target column suitability
            if model_type == 'classification':
                if target_data.nunique() > 100:
                    analysis_result['compatible'] = False
                    analysis_result['issues'].append(f'Too many classes ({target_data.nunique()}) for classification')
                elif target_data.nunique() < 2:
                    analysis_result['compatible'] = False
                    analysis_result['issues'].append('Target column has only one unique value')
                else:
                    class_counts = target_data.value_counts()
                    min_class_count = class_counts.min()
                    if min_class_count < 2:
                        analysis_result['warnings'].append(f'Smallest class has only {min_class_count} sample(s)')
            
            # Analyze feature columns
            feature_columns = [col for col in df.columns if col != target_column]
            if len(feature_columns) == 0:
                analysis_result['compatible'] = False
                analysis_result['issues'].append('No feature columns found')
                return analysis_result
            
            # Analyze column types
            numeric_cols = df[feature_columns].select_dtypes(include=[np.number]).columns
            categorical_cols = df[feature_columns].select_dtypes(include=['object', 'category']).columns
            datetime_cols = df[feature_columns].select_dtypes(include=['datetime64']).columns
            
            analysis_result['column_analysis'] = {
                'numeric_columns': int(len(numeric_cols)),
                'categorical_columns': int(len(categorical_cols)),
                'datetime_columns': int(len(datetime_cols)),
                'high_cardinality_columns': []
            }
            
            # Check for high cardinality categorical columns
            for col in categorical_cols:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8:
                    analysis_result['column_analysis']['high_cardinality_columns'].append({
                        'column': col,
                        'unique_values': int(df[col].nunique()),
                        'unique_ratio': float(unique_ratio)
                    })
            
            # Check missing values
            missing_cols = df.columns[df.isna().any()].tolist()
            if missing_cols:
                analysis_result['warnings'].append(f'Missing values in {len(missing_cols)} columns')
            
            # Memory usage warning
            if analysis_result['dataset_info']['memory_usage'] > 100_000_000:  # 100MB
                analysis_result['warnings'].append('Large dataset detected - processing may be slow')
            
            # Provide recommendations
            if len(analysis_result['column_analysis']['high_cardinality_columns']) > 0:
                analysis_result['recommendations'].append('Consider removing or encoding high cardinality columns')
            
            if analysis_result['dataset_info']['missing_values'] > 0:
                analysis_result['recommendations'].append('Missing values will be automatically handled')
            
            return analysis_result
            
        except Exception as e:
            return {
                'compatible': False,
                'issues': [f'Analysis failed: {str(e)}'],
                'warnings': [],
                'recommendations': [],
                'dataset_info': {},
                'target_info': {}
            }


class AdvancedTrainingService:
    """Advanced training service optimized for large datasets"""
    
    @staticmethod
    def train_model_with_optimization(model_id: int, df: pd.DataFrame, 
                                    target_column: str, algorithm: str, 
                                    model_type: str, config: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Train model with advanced optimization for large datasets
        """
        try:
            from app.models.ml_model import MLModel
            
            # Get model record
            model = MLModel.query.get(model_id)
            if not model:
                return False, 'Model not found'
            
            # Update status
            model.status = 'training'
            model.training_log = f"Starting training at {datetime.now()}\n"
            db.session.commit()
            
            # Default configuration
            if config is None:
                config = {
                    'test_size': 0.2,
                    'cv_folds': 5,
                    'max_samples': 100000,
                    'enable_sampling': True,
                    'optimize_memory': True,
                    'use_incremental_learning': False
                }
            
            # Advanced preprocessing
            X, y, preprocessing_info = AutoMLService.smart_preprocess_data(
                df, target_column, model_type, 
                max_samples=config.get('max_samples', 100000)
            )
            
            if X is None or y is None:
                error_msg = preprocessing_info.get('error', 'Preprocessing failed')
                model.status = 'failed'
                model.training_log += f"Preprocessing failed: {error_msg}\n"
                db.session.commit()
                return False, error_msg
            
            # Log preprocessing info
            model.training_log += f"Preprocessing completed:\n"
            model.training_log += f"- Dataset shape: {X.shape}\n"
            model.training_log += f"- Memory usage: {preprocessing_info.get('memory_usage_mb', 0):.2f} MB\n"
            model.training_log += f"- Processing time: {preprocessing_info.get('processing_time', 0):.2f} seconds\n"
            
            for warning in preprocessing_info.get('warnings', []):
                model.training_log += f"- Warning: {warning}\n"
            
            # Split data efficiently
            test_size = config.get('test_size', 0.2)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if model_type == 'classification' and len(np.unique(y)) > 1 else None
            )
            
            # Create optimized model
            ml_model = AdvancedTrainingService._create_optimized_model(
                algorithm, model_type, X_train.shape, config
            )
            
            if ml_model is None:
                error_msg = f'Algorithm {algorithm} not supported for {model_type}'
                model.status = 'failed'
                model.training_log += f"Error: {error_msg}\n"
                db.session.commit()
                return False, error_msg
            
            # Train model with progress tracking
            start_time = time.time()
            model.training_log += f"Training {algorithm} model...\n"
            db.session.commit()
            
            # Handle large datasets with incremental learning if supported
            if (config.get('use_incremental_learning', False) and 
                hasattr(ml_model, 'partial_fit') and len(X_train) > 50000):
                
                # Incremental learning for very large datasets
                batch_size = 1000
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train.iloc[i:i+batch_size]
                    batch_y = y_train.iloc[i:i+batch_size]
                    
                    if i == 0:
                        # First batch - determine classes for classification
                        if model_type == 'classification':
                            classes = np.unique(y_train)
                            ml_model.partial_fit(batch_X, batch_y, classes=classes)
                        else:
                            ml_model.partial_fit(batch_X, batch_y)
                    else:
                        ml_model.partial_fit(batch_X, batch_y)
                
                model.training_log += f"Incremental training completed in {len(X_train) // batch_size + 1} batches\n"
            else:
                # Standard training
                ml_model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            model.training_log += f"Training completed in {training_time:.2f} seconds\n"
            
            # Make predictions
            y_pred = ml_model.predict(X_test)
            
            # Calculate metrics
            metrics = AdvancedTrainingService._calculate_metrics(
                y_test, y_pred, model_type
            )
            
            # Cross-validation with smart CV selection
            cv_results = AdvancedTrainingService._perform_cross_validation(
                ml_model, X, y, model_type, config.get('cv_folds', 5)
            )
            
            # Feature importance
            feature_importance = AdvancedTrainingService._extract_feature_importance(
                ml_model, X.columns
            )
            
            # Save model efficiently
            model_path = AdvancedTrainingService._save_model(
                ml_model, model_id, algorithm, model_type, X.columns, target_column
            )
            
            # Update model record
            model.status = 'trained'
            model.model_path = model_path
            model.features = list(X.columns)
            model.training_time = training_time
            model.trained_at = datetime.now()
            model.model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
            
            # Store metrics
            for metric_name, metric_value in metrics.items():
                setattr(model, metric_name, metric_value)
            
            # Store cross-validation results
            model.cv_scores = cv_results.get('cv_scores', [])
            model.cv_mean = cv_results.get('cv_mean')
            model.cv_std = cv_results.get('cv_std')
            
            # Store feature importance
            model.feature_importance = feature_importance
            
            # Store preprocessing info
            model.preprocessing_info = preprocessing_info
            
            model.training_log += f"Model saved to: {model_path}\n"
            model.training_log += f"Training completed successfully!\n"
            model.training_log += f"Final metrics: {metrics}\n"
            
            db.session.commit()
            return True, 'Training completed successfully'
            
        except Exception as e:
            error_msg = f'Training failed: {str(e)}'
            try:
                model.status = 'failed'
                model.training_log += f"Training failed: {error_msg}\n"
                db.session.commit()
            except:
                pass
            return False, error_msg
    
    @staticmethod
    def _create_optimized_model(algorithm: str, model_type: str, 
                               data_shape: tuple, config: Dict[str, Any]):
        """Create optimized model based on algorithm and data characteristics"""
        try:
            n_samples, n_features = data_shape
            
            # Optimization parameters based on data size
            if n_samples > 100000:  # Large dataset
                n_jobs = -1
                max_iter = 1000
                n_estimators = 100
            elif n_samples > 10000:  # Medium dataset
                n_jobs = -1
                max_iter = 2000
                n_estimators = 200
            else:  # Small dataset
                n_jobs = 1
                max_iter = 5000
                n_estimators = 300
            
            # Create model based on algorithm
            if algorithm == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(
                    random_state=42, 
                    max_iter=max_iter, 
                    n_jobs=n_jobs,
                    solver='lbfgs' if n_features < 1000 else 'saga'
                )
            
            elif algorithm == 'random_forest':
                if model_type == 'classification':
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=42,
                        n_jobs=n_jobs,
                        max_depth=None if n_samples > 10000 else 10
                    )
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    return RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=42,
                        n_jobs=n_jobs,
                        max_depth=None if n_samples > 10000 else 10
                    )
            
            elif algorithm == 'gradient_boosting':
                if model_type == 'classification':
                    from sklearn.ensemble import GradientBoostingClassifier
                    return GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        random_state=42,
                        learning_rate=0.1,
                        max_depth=6 if n_samples > 10000 else 3
                    )
                else:
                    from sklearn.ensemble import GradientBoostingRegressor
                    return GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        random_state=42,
                        learning_rate=0.1,
                        max_depth=6 if n_samples > 10000 else 3
                    )
            
            elif algorithm == 'xgboost':
                try:
                    import xgboost as xgb
                    if model_type == 'classification':
                        return xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            random_state=42,
                            n_jobs=n_jobs,
                            max_depth=6,
                            learning_rate=0.1
                        )
                    else:
                        return xgb.XGBRegressor(
                            n_estimators=n_estimators,
                            random_state=42,
                            n_jobs=n_jobs,
                            max_depth=6,
                            learning_rate=0.1
                        )
                except ImportError:
                    # Fallback to gradient boosting
                    return AdvancedTrainingService._create_optimized_model(
                        'gradient_boosting', model_type, data_shape, config
                    )
            
            elif algorithm == 'lightgbm':
                try:
                    import lightgbm as lgb
                    if model_type == 'classification':
                        return lgb.LGBMClassifier(
                            n_estimators=n_estimators,
                            random_state=42,
                            n_jobs=n_jobs,
                            max_depth=6,
                            learning_rate=0.1,
                            verbose=-1
                        )
                    else:
                        return lgb.LGBMRegressor(
                            n_estimators=n_estimators,
                            random_state=42,
                            n_jobs=n_jobs,
                            max_depth=6,
                            learning_rate=0.1,
                            verbose=-1
                        )
                except ImportError:
                    # Fallback to gradient boosting
                    return AdvancedTrainingService._create_optimized_model(
                        'gradient_boosting', model_type, data_shape, config
                    )
            
            elif algorithm == 'svm':
                if model_type == 'classification':
                    from sklearn.svm import SVC
                    # Use linear kernel for large datasets
                    kernel = 'linear' if n_samples > 10000 else 'rbf'
                    return SVC(random_state=42, kernel=kernel)
                else:
                    from sklearn.svm import SVR
                    kernel = 'linear' if n_samples > 10000 else 'rbf'
                    return SVR(kernel=kernel)
            
            elif algorithm == 'knn':
                k = min(5, max(3, int(np.sqrt(n_samples))))
                if model_type == 'classification':
                    from sklearn.neighbors import KNeighborsClassifier
                    return KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
                else:
                    from sklearn.neighbors import KNeighborsRegressor
                    return KNeighborsRegressor(n_neighbors=k, n_jobs=n_jobs)
            
            elif algorithm == 'decision_tree':
                if model_type == 'classification':
                    from sklearn.tree import DecisionTreeClassifier
                    return DecisionTreeClassifier(
                        random_state=42,
                        max_depth=None if n_samples > 10000 else 10
                    )
                else:
                    from sklearn.tree import DecisionTreeRegressor
                    return DecisionTreeRegressor(
                        random_state=42,
                        max_depth=None if n_samples > 10000 else 10
                    )
            
            elif algorithm == 'naive_bayes':
                from sklearn.naive_bayes import GaussianNB
                return GaussianNB()
            
            elif algorithm == 'linear_regression':
                from sklearn.linear_model import LinearRegression
                return LinearRegression(n_jobs=n_jobs)
            
            elif algorithm == 'ridge':
                from sklearn.linear_model import Ridge
                return Ridge(random_state=42)
            
            elif algorithm == 'lasso':
                from sklearn.linear_model import Lasso
                return Lasso(random_state=42, max_iter=max_iter)
            
            elif algorithm == 'elastic_net':
                from sklearn.linear_model import ElasticNet
                return ElasticNet(random_state=42, max_iter=max_iter)
            
            else:
                return None
                
        except Exception as e:
            print(f"Error creating model: {e}")
            return None
    
    @staticmethod
    def _calculate_metrics(y_true, y_pred, model_type: str) -> Dict[str, float]:
        """Calculate appropriate metrics based on model type"""
        try:
            metrics = {}
            
            if model_type == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                
                # Add ROC AUC for binary classification
                if len(np.unique(y_true)) == 2:
                    try:
                        from sklearn.metrics import roc_auc_score
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred))
                    except:
                        pass
            
            else:  # regression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                metrics['r2_score'] = float(r2_score(y_true, y_pred))
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    @staticmethod
    def _perform_cross_validation(model, X, y, model_type: str, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation with smart fold selection"""
        try:
            from sklearn.model_selection import cross_val_score
            
            # Adjust CV folds based on data size and type
            if model_type == 'classification':
                class_counts = pd.Series(y).value_counts()
                min_class_count = class_counts.min()
                cv_folds = min(cv_folds, min_class_count)
            
            if len(X) < cv_folds:
                cv_folds = max(2, len(X) // 2)
            
            cv_folds = max(2, cv_folds)  # Minimum 2 folds
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, n_jobs=-1)
            
            return {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'cv_folds': cv_folds
            }
            
        except Exception as e:
            print(f"Cross-validation error: {e}")
            return {
                'cv_scores': [],
                'cv_mean': None,
                'cv_std': None,
                'cv_error': str(e)
            }
    
    @staticmethod
    def _extract_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = float(importances[i])
            
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(model.coef_.shape) == 1:
                    coefficients = model.coef_
                else:
                    coefficients = model.coef_[0]
                
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = float(abs(coefficients[i]))
            
            # Sort by importance
            if importance_dict:
                importance_dict = dict(sorted(
                    importance_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
            
            return importance_dict
            
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return {}
    
    @staticmethod
    def _save_model(model, model_id: int, algorithm: str, model_type: str, 
                   features: List[str], target_column: str) -> str:
        """Save model with metadata"""
        try:
            # Create model directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            model_filename = f'{model_id}_{algorithm}_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            model_path = os.path.join(models_dir, model_filename)
            
            model_data = {
                'model': model,
                'features': features,
                'target_column': target_column,
                'model_type': model_type,
                'algorithm': algorithm,
                'created_at': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            joblib.dump(model_data, model_path)
            return model_path
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return ""
