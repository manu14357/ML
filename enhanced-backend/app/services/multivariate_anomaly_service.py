# -*- coding: utf-8 -*-
"""
Multivariate Anomaly Detection Service
"""

import pandas as pd
import numpy as np
import time
import traceback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import pearsonr
from pyod.models.ecod import ECOD
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class MultivariateAnomalyDetectionService:
    """Enhanced Multivariate Anomaly Detection Service with Plotly visualizations"""

    def __init__(self):
        self.results = {}
        self.scalers = {}
        self.mutual_info_results = {}
        self.hierarchical_clustering_results = {}
        self.ecod_results = {}
        self.cross_correlation_results = {}
        self.variance_change_results = {}

    def detect_anomalies(self, df: pd.DataFrame, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect anomalies in multivariate data using multiple methods.
        
        Parameters:
        - df: pandas DataFrame to analyze
        - parameters: dictionary containing detection parameters
        """
        start_time = time.time()
        
        if parameters is None:
            parameters = {}
            
        method = parameters.get('method', 'all')
        contamination = parameters.get('contamination', 0.1)
        feature_columns = parameters.get('feature_columns', None)
        scale_features = parameters.get('scale_features', True)
        n_components = parameters.get('n_components', 2)
        
        logger.info(f"Starting multivariate anomaly detection with method: {method}")
        
        results = {}

        try:
            # Select feature columns
            if feature_columns is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in feature_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_columns) < 2:
                return {
                    'success': False,
                    'error': 'At least 2 numeric columns required for multivariate anomaly detection'
                }
            
            # Prepare data
            data = df[numeric_columns].copy()
            data = data.dropna()  # Remove rows with any missing values
            
            if len(data) < 10:
                return {
                    'success': False,
                    'error': 'Insufficient data points after removing missing values (minimum 10 required)'
                }
            
            # Dataset information
            dataset_info = {
                'total_rows': len(df),
                'clean_rows': len(data),
                'numeric_columns': len(numeric_columns),
                'columns_analyzed': numeric_columns,
                'missing_rows_removed': len(df) - len(data)
            }
            results['dataset_info'] = dataset_info
            
            # Scale features if requested
            if scale_features:
                scaler = StandardScaler()
                scaled_data = pd.DataFrame(
                    scaler.fit_transform(data),
                    columns=data.columns,
                    index=data.index
                )
                self.scalers['main'] = scaler
            else:
                scaled_data = data
            
            # Anomaly detection results for each method
            anomaly_results = {}
            
            if method in ['all', 'isolation_forest']:
                anomaly_results['isolation_forest'] = self._detect_isolation_forest_anomalies(scaled_data, contamination)
            
            if method in ['all', 'elliptic_envelope']:
                anomaly_results['elliptic_envelope'] = self._detect_elliptic_envelope_anomalies(scaled_data, contamination)
            
            if method in ['all', 'local_outlier_factor']:
                anomaly_results['local_outlier_factor'] = self._detect_lof_anomalies(scaled_data, contamination)
            
            if method in ['all', 'mahalanobis']:
                anomaly_results['mahalanobis'] = self._detect_mahalanobis_anomalies(scaled_data)
            
            if method in ['all', 'pca_reconstruction']:
                anomaly_results['pca_reconstruction'] = self._detect_pca_anomalies(scaled_data, n_components)
            
            if method in ['all', 'ecod']:
                anomaly_results['ecod'] = self._detect_ecod_anomalies(scaled_data, contamination)
            
            results['anomaly_results'] = anomaly_results
            
            # Advanced analysis features
            mutual_info_analysis = self._analyze_mutual_information(data)
            results['mutual_information_analysis'] = mutual_info_analysis
            self.mutual_info_results = mutual_info_analysis
            
            hierarchical_clustering = self._perform_hierarchical_clustering(data)
            results['hierarchical_clustering'] = hierarchical_clustering
            self.hierarchical_clustering_results = hierarchical_clustering
            
            cross_correlation_analysis = self._analyze_cross_correlation(data)
            results['cross_correlation_analysis'] = cross_correlation_analysis
            self.cross_correlation_results = cross_correlation_analysis
            
            variance_change_analysis = self._analyze_variance_changes(data)
            results['variance_change_analysis'] = variance_change_analysis
            self.variance_change_results = variance_change_analysis
            
            # Combine results and create summary
            combined_results = self._combine_anomaly_results(data, anomaly_results)
            results['combined_analysis'] = combined_results
            
            # Generate visualizations
            charts = self._generate_anomaly_charts(data, scaled_data, anomaly_results, parameters)
            results['charts'] = charts
            
            # Statistical summary
            stats_summary = self._generate_statistical_summary(data, anomaly_results)
            results['statistical_summary'] = stats_summary
            
            # Feature importance analysis
            feature_importance = self._analyze_feature_importance(data, anomaly_results)
            results['feature_importance'] = feature_importance
            
            # Recommendations
            recommendations = self._generate_recommendations(anomaly_results, combined_results, dataset_info)
            results['recommendations'] = recommendations
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            results['analysis_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Multivariate anomaly detection completed in {execution_time:.2f} seconds")
            
            return {
                'success': True,
                'results': results,
                'metadata': {
                    'analysis_type': 'multivariate_anomaly_detection',
                    'execution_time': execution_time,
                    'parameters_used': parameters
                }
            }
            
        except Exception as e:
            logger.error(f"Error in multivariate anomaly detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _detect_isolation_forest_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        anomaly_labels = iso_forest.fit_predict(data)
        anomaly_scores = iso_forest.decision_function(data)
        
        anomaly_indices = data.index[anomaly_labels == -1].tolist()
        
        return {
            'method': 'isolation_forest',
            'contamination': contamination,
            'anomaly_count': len(anomaly_indices),
            'anomaly_percentage': (len(anomaly_indices) / len(data)) * 100,
            'anomaly_indices': anomaly_indices,
            'anomaly_scores': anomaly_scores.tolist(),
            'min_score': float(np.min(anomaly_scores)),
            'max_score': float(np.max(anomaly_scores)),
            'mean_score': float(np.mean(anomaly_scores)),
            'threshold': float(np.percentile(anomaly_scores, contamination * 100))
        }

    def _detect_elliptic_envelope_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies using Elliptic Envelope (Robust Covariance)"""
        try:
            elliptic_env = EllipticEnvelope(contamination=contamination, random_state=42)
            anomaly_labels = elliptic_env.fit_predict(data)
            
            # Get Mahalanobis distances
            mahal_distances = elliptic_env.mahalanobis(data)
            
            anomaly_indices = data.index[anomaly_labels == -1].tolist()
            
            return {
                'method': 'elliptic_envelope',
                'contamination': contamination,
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(data)) * 100,
                'anomaly_indices': anomaly_indices,
                'mahalanobis_distances': mahal_distances.tolist(),
                'min_distance': float(np.min(mahal_distances)),
                'max_distance': float(np.max(mahal_distances)),
                'mean_distance': float(np.mean(mahal_distances)),
                'threshold': float(np.percentile(mahal_distances, (1 - contamination) * 100))
            }
        except Exception as e:
            logger.warning(f"Elliptic Envelope failed: {str(e)}")
            return {
                'method': 'elliptic_envelope',
                'error': str(e),
                'anomaly_count': 0,
                'anomaly_indices': []
            }

    def _detect_lof_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies using Local Outlier Factor"""
        n_neighbors = min(20, len(data) - 1)  # Ensure n_neighbors is valid
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        anomaly_labels = lof.fit_predict(data)
        lof_scores = -lof.negative_outlier_factor_  # Convert to positive values
        
        anomaly_indices = data.index[anomaly_labels == -1].tolist()
        
        return {
            'method': 'local_outlier_factor',
            'contamination': contamination,
            'n_neighbors': n_neighbors,
            'anomaly_count': len(anomaly_indices),
            'anomaly_percentage': (len(anomaly_indices) / len(data)) * 100,
            'anomaly_indices': anomaly_indices,
            'lof_scores': lof_scores.tolist(),
            'min_score': float(np.min(lof_scores)),
            'max_score': float(np.max(lof_scores)),
            'mean_score': float(np.mean(lof_scores)),
            'threshold': float(np.percentile(lof_scores, (1 - contamination) * 100))
        }

    def _detect_mahalanobis_anomalies(self, data: pd.DataFrame, threshold_percentile: float = 97.5) -> Dict[str, Any]:
        """Detect anomalies using Mahalanobis distance"""
        try:
            # Calculate covariance matrix
            cov_matrix = np.cov(data.T)
            inv_cov_matrix = np.linalg.pinv(cov_matrix)  # Use pseudoinverse for stability
            
            # Calculate Mahalanobis distances
            mean = np.mean(data, axis=0)
            mahal_distances = []
            
            for i, row in data.iterrows():
                diff = row - mean
                mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                mahal_distances.append(mahal_dist)
            
            mahal_distances = np.array(mahal_distances)
            
            # Determine threshold
            threshold = np.percentile(mahal_distances, threshold_percentile)
            anomaly_indices = data.index[mahal_distances > threshold].tolist()
            
            return {
                'method': 'mahalanobis',
                'threshold_percentile': threshold_percentile,
                'threshold': float(threshold),
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(data)) * 100,
                'anomaly_indices': anomaly_indices,
                'mahalanobis_distances': mahal_distances.tolist(),
                'min_distance': float(np.min(mahal_distances)),
                'max_distance': float(np.max(mahal_distances)),
                'mean_distance': float(np.mean(mahal_distances))
            }
        except Exception as e:
            logger.warning(f"Mahalanobis distance calculation failed: {str(e)}")
            return {
                'method': 'mahalanobis',
                'error': str(e),
                'anomaly_count': 0,
                'anomaly_indices': []
            }

    def _detect_pca_anomalies(self, data: pd.DataFrame, n_components: int = 2, threshold_percentile: float = 95) -> Dict[str, Any]:
        """Detect anomalies using PCA reconstruction error"""
        try:
            # Apply PCA
            pca = PCA(n_components=min(n_components, data.shape[1]))
            pca_transformed = pca.fit_transform(data)
            
            # Reconstruct data
            reconstructed = pca.inverse_transform(pca_transformed)
            
            # Calculate reconstruction errors
            reconstruction_errors = np.mean((data.values - reconstructed) ** 2, axis=1)
            
            # Determine threshold
            threshold = np.percentile(reconstruction_errors, threshold_percentile)
            anomaly_indices = data.index[reconstruction_errors > threshold].tolist()
            
            return {
                'method': 'pca_reconstruction',
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': float(np.sum(pca.explained_variance_ratio_)),
                'threshold_percentile': threshold_percentile,
                'threshold': float(threshold),
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(data)) * 100,
                'anomaly_indices': anomaly_indices,
                'reconstruction_errors': reconstruction_errors.tolist(),
                'min_error': float(np.min(reconstruction_errors)),
                'max_error': float(np.max(reconstruction_errors)),
                'mean_error': float(np.mean(reconstruction_errors))
            }
        except Exception as e:
            logger.warning(f"PCA anomaly detection failed: {str(e)}")
            return {
                'method': 'pca_reconstruction',
                'error': str(e),
                'anomaly_count': 0,
                'anomaly_indices': []
            }

    def _detect_ecod_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies using ECOD (Empirical-Cumulative-distribution-based Outlier Detection)"""
        try:
            ecod = ECOD(contamination=contamination)
            anomaly_labels = ecod.fit_predict(data)
            anomaly_scores = ecod.decision_scores_
            
            anomaly_indices = data.index[anomaly_labels == 1].tolist()
            
            return {
                'method': 'ecod',
                'contamination': contamination,
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(data)) * 100,
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': anomaly_scores.tolist(),
                'min_score': float(np.min(anomaly_scores)),
                'max_score': float(np.max(anomaly_scores)),
                'mean_score': float(np.mean(anomaly_scores)),
                'threshold': float(ecod.threshold_)
            }
        except Exception as e:
            logger.warning(f"ECOD detection failed: {str(e)}")
            return {
                'method': 'ecod',
                'error': str(e),
                'anomaly_count': 0,
                'anomaly_indices': []
            }

    def _combine_anomaly_results(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from different anomaly detection methods"""
        # Collect all anomaly indices from different methods
        all_anomaly_indices = set()
        method_counts = {}
        
        for method, method_result in anomaly_results.items():
            if 'error' not in method_result:
                indices = method_result.get('anomaly_indices', [])
                all_anomaly_indices.update(indices)
                method_counts[method] = len(indices)
        
        # Create consensus analysis
        consensus_anomalies = []
        for idx in all_anomaly_indices:
            method_detections = []
            for method, method_result in anomaly_results.items():
                if 'error' not in method_result and idx in method_result.get('anomaly_indices', []):
                    method_detections.append(method)
            
            if idx in data.index:
                consensus_anomalies.append({
                    'index': idx,
                    'values': data.loc[idx].to_dict(),
                    'detected_by': method_detections,
                    'detection_count': len(method_detections)
                })
        
        # Sort by detection count (most consensus first)
        consensus_anomalies.sort(key=lambda x: x['detection_count'], reverse=True)
        
        return {
            'total_unique_anomalies': len(all_anomaly_indices),
            'method_counts': method_counts,
            'consensus_anomalies': consensus_anomalies,
            'high_confidence_anomalies': [a for a in consensus_anomalies if a['detection_count'] >= 2],
            'anomaly_percentage': (len(all_anomaly_indices) / len(data)) * 100 if len(data) > 0 else 0,
            'methods_used': list(method_counts.keys())
        }

    def _analyze_feature_importance(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which features contribute most to anomaly detection"""
        feature_importance = {}
        
        for method, method_result in anomaly_results.items():
            if 'error' in method_result or not method_result.get('anomaly_indices'):
                continue
            
            anomaly_indices = method_result['anomaly_indices']
            normal_indices = [idx for idx in data.index if idx not in anomaly_indices]
            
            if not normal_indices:
                continue
            
            # Calculate feature statistics for anomalies vs normal
            feature_stats = {}
            for col in data.columns:
                anomaly_values = data.loc[anomaly_indices, col] if anomaly_indices else pd.Series([])
                normal_values = data.loc[normal_indices, col]
                
                if len(anomaly_values) > 0 and len(normal_values) > 0:
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(anomaly_values) - 1) * anomaly_values.var() + 
                                        (len(normal_values) - 1) * normal_values.var()) / 
                                       (len(anomaly_values) + len(normal_values) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = abs(anomaly_values.mean() - normal_values.mean()) / pooled_std
                    else:
                        cohens_d = 0
                    
                    feature_stats[col] = {
                        'anomaly_mean': float(anomaly_values.mean()),
                        'normal_mean': float(normal_values.mean()),
                        'anomaly_std': float(anomaly_values.std()),
                        'normal_std': float(normal_values.std()),
                        'difference': float(anomaly_values.mean() - normal_values.mean()),
                        'cohens_d': float(cohens_d),
                        'importance_score': float(cohens_d)  # Use Cohen's d as importance score
                    }
            
            # Sort features by importance
            sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['importance_score'], reverse=True)
            
            feature_importance[method] = {
                'feature_stats': dict(sorted_features),
                'top_features': [f[0] for f in sorted_features[:5]],
                'most_important_feature': sorted_features[0][0] if sorted_features else None
            }
        
        return feature_importance

    def _generate_statistical_summary(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical summary of anomaly detection results"""
        summary = {
            'dataset_stats': {
                'n_samples': len(data),
                'n_features': len(data.columns),
                'feature_names': data.columns.tolist()
            },
            'method_summary': {}
        }
        
        for method, method_result in anomaly_results.items():
            if 'error' not in method_result:
                summary['method_summary'][method] = {
                    'anomaly_count': method_result.get('anomaly_count', 0),
                    'anomaly_percentage': method_result.get('anomaly_percentage', 0),
                    'method_specific_stats': {
                        k: v for k, v in method_result.items() 
                        if k not in ['anomaly_indices', 'method'] and isinstance(v, (int, float))
                    }
                }
            else:
                summary['method_summary'][method] = {
                    'error': method_result['error'],
                    'anomaly_count': 0,
                    'anomaly_percentage': 0
                }
        
        return summary

    def _generate_recommendations(self, anomaly_results: Dict[str, Any], combined_results: Dict[str, Any], dataset_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on anomaly detection results"""
        recommendations = []
        
        # Overall anomaly rate recommendations
        overall_anomaly_rate = combined_results.get('anomaly_percentage', 0)
        
        if overall_anomaly_rate > 15:
            recommendations.append("High multivariate anomaly rate detected (>15%). Consider investigating data quality or collection processes.")
        elif overall_anomaly_rate > 8:
            recommendations.append("Moderate multivariate anomaly rate detected (8-15%). Review unusual patterns in the data.")
        elif overall_anomaly_rate > 0:
            recommendations.append("Low multivariate anomaly rate detected (<8%). Consider investigating high-confidence anomalies.")
        else:
            recommendations.append("No anomalies detected. Data appears to follow expected multivariate patterns.")
        
        # Method-specific recommendations
        successful_methods = [m for m, r in anomaly_results.items() if 'error' not in r]
        failed_methods = [m for m, r in anomaly_results.items() if 'error' in r]
        
        if failed_methods:
            recommendations.append(f"Some methods failed: {', '.join(failed_methods)}. This may be due to data characteristics or insufficient samples.")
        
        # High confidence anomaly recommendations
        high_confidence_count = len(combined_results.get('high_confidence_anomalies', []))
        
        if high_confidence_count > 0:
            recommendations.append(f"Found {high_confidence_count} high-confidence anomalies detected by multiple methods. Prioritize investigation of these points.")
        
        # Data quality recommendations
        if dataset_info.get('missing_rows_removed', 0) > 0:
            missing_count = dataset_info['missing_rows_removed']
            total_rows = dataset_info['total_rows']
            missing_percentage = (missing_count / total_rows) * 100
            recommendations.append(f"Removed {missing_count} rows ({missing_percentage:.1f}%) due to missing values. Consider imputation strategies for better coverage.")
        
        # Feature dimension recommendations
        n_features = dataset_info.get('numeric_columns', 0)
        n_samples = dataset_info.get('clean_rows', 0)
        
        if n_features > n_samples / 10:
            recommendations.append("High-dimensional data detected. Consider dimensionality reduction techniques for more robust anomaly detection.")
        
        return recommendations

    def _generate_anomaly_charts(self, data: pd.DataFrame, scaled_data: pd.DataFrame, anomaly_results: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive multivariate anomaly detection charts using Plotly"""
        charts = {}
        
        try:
            # Anomaly summary chart
            charts['anomaly_summary'] = {
                'chart': self._create_anomaly_summary_chart(anomaly_results),
                'insight': self._generate_anomaly_summary_insight(anomaly_results)
            }
            
            # PCA visualization with anomalies highlighted
            charts['pca_visualization'] = {
                'chart': self._create_pca_anomaly_chart(scaled_data, anomaly_results),
                'insight': self._generate_pca_insight(scaled_data, anomaly_results)
            }
            
            # Correlation matrix with anomaly patterns
            charts['correlation_matrix'] = {
                'chart': self._create_correlation_matrix_chart(data),
                'insight': self._generate_correlation_matrix_insight(data)
            }
            
            # Parallel coordinates plot
            charts['parallel_coordinates'] = {
                'chart': self._create_parallel_coordinates_chart(data, anomaly_results),
                'insight': self._generate_parallel_coordinates_insight(data, anomaly_results)
            }
            
            # Feature importance chart
            charts['feature_importance'] = {
                'chart': self._create_feature_importance_chart(anomaly_results),
                'insight': self._generate_feature_importance_insight(anomaly_results)
            }
            
            # Method comparison
            charts['method_comparison'] = {
                'chart': self._create_method_comparison_chart(anomaly_results),
                'insight': self._generate_method_comparison_insight(anomaly_results)
            }
            
            # Hierarchical clustering dendrogram
            charts['hierarchical_dendrogram'] = {
                'chart': self._create_hierarchical_dendrogram_chart(),
                'insight': self._generate_hierarchical_dendrogram_insight()
            }
            
            # Mutual information heatmap
            charts['mutual_information_heatmap'] = {
                'chart': self._create_mutual_information_heatmap(),
                'insight': self._generate_mutual_information_insight()
            }
            
            # Cross-correlation network
            charts['cross_correlation_network'] = {
                'chart': self._create_cross_correlation_chart(),
                'insight': self._generate_cross_correlation_insight()
            }
            
            # Variance change analysis
            charts['variance_change_analysis'] = {
                'chart': self._create_variance_change_chart(),
                'insight': self._generate_variance_change_insight()
            }
            
            # 3D scatter plot (if enough features)
            if len(data.columns) >= 3:
                charts['3d_scatter'] = {
                    'chart': self._create_3d_scatter_chart(data, anomaly_results),
                    'insight': self._generate_3d_scatter_insight(data, anomaly_results)
                }
            
        except Exception as e:
            logger.error(f"Error generating multivariate anomaly charts: {str(e)}")
            charts['error'] = str(e)
        
        return charts

    def _create_anomaly_summary_chart(self, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create anomaly detection summary chart"""
        methods = []
        counts = []
        
        for method, result in anomaly_results.items():
            if 'error' not in result:
                methods.append(method.replace('_', ' ').title())
                counts.append(result.get('anomaly_count', 0))
        
        if not methods:
            fig = go.Figure()
            fig.add_annotation(
                text="No successful anomaly detection results",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Anomaly Detection Summary")
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
        
        fig = px.bar(
            x=methods,
            y=counts,
            title="Multivariate Anomalies Detected by Method",
            labels={'x': 'Detection Method', 'y': 'Number of Anomalies'}
        )
        
        fig.update_layout(
            title_x=0.5,
            font=dict(size=12),
            xaxis_tickangle=45
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    def _create_pca_anomaly_chart(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create PCA visualization with anomalies highlighted"""
        try:
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data)
            
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=data.index)
            
            fig = go.Figure()
            
            # Add normal points
            normal_indices = data.index.tolist()
            for method, result in anomaly_results.items():
                if 'error' not in result:
                    anomaly_indices = result.get('anomaly_indices', [])
                    normal_indices = [idx for idx in normal_indices if idx not in anomaly_indices]
            
            # Plot normal points
            if normal_indices:
                fig.add_trace(go.Scatter(
                    x=pca_df.loc[normal_indices, 'PC1'],
                    y=pca_df.loc[normal_indices, 'PC2'],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=6, opacity=0.6)
                ))
            
            # Add anomalies from different methods
            colors = ['red', 'orange', 'purple', 'green', 'pink']
            for i, (method, result) in enumerate(anomaly_results.items()):
                if 'error' not in result:
                    anomaly_indices = result.get('anomaly_indices', [])
                    if anomaly_indices:
                        valid_indices = [idx for idx in anomaly_indices if idx in pca_df.index]
                        if valid_indices:
                            fig.add_trace(go.Scatter(
                                x=pca_df.loc[valid_indices, 'PC1'],
                                y=pca_df.loc[valid_indices, 'PC2'],
                                mode='markers',
                                name=f'{method.replace("_", " ").title()}',
                                marker=dict(color=colors[i % len(colors)], size=10, symbol='x')
                            ))
            
            variance_explained = pca.explained_variance_ratio_
            fig.update_layout(
                title=f"PCA Visualization with Anomalies<br>PC1: {variance_explained[0]:.1%}, PC2: {variance_explained[1]:.1%} variance explained",
                title_x=0.5,
                xaxis_title=f"First Principal Component ({variance_explained[0]:.1%})",
                yaxis_title=f"Second Principal Component ({variance_explained[1]:.1%})",
                font=dict(size=12)
            )
            
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error creating PCA chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating PCA visualization: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12)
            )
            fig.update_layout(title="PCA Visualization Error")
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    def _create_correlation_matrix_chart(self, data: pd.DataFrame) -> str:
        """Create correlation matrix heatmap"""
        correlation_matrix = data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        fig.update_layout(
            title_x=0.5,
            font=dict(size=12)
        )
        
        return fig.to_json()

    def _create_parallel_coordinates_chart(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create parallel coordinates plot"""
        try:
            # Limit to first 6 features for readability
            features_to_plot = data.columns[:6].tolist()
            plot_data = data[features_to_plot].copy()
            
            # Sample data if too large for better performance
            if len(plot_data) > 1000:
                plot_data = plot_data.sample(n=1000, random_state=42)
            
            # Normalize data for better visualization
            normalized_data = plot_data.copy()
            for col in features_to_plot:
                min_val = plot_data[col].min()
                max_val = plot_data[col].max()
                if max_val > min_val:
                    normalized_data[col] = (plot_data[col] - min_val) / (max_val - min_val)
                else:
                    normalized_data[col] = 0.5  # If all values are the same
            
            # Create anomaly status
            anomaly_status = pd.Series(['Normal'] * len(normalized_data), index=normalized_data.index)
            
            # Mark anomalies from any method
            for method, result in anomaly_results.items():
                if 'error' not in result:
                    anomaly_indices = result.get('anomaly_indices', [])
                    for idx in anomaly_indices:
                        if idx in anomaly_status.index:
                            anomaly_status.loc[idx] = 'Anomaly'
            
            # Create color mapping
            color_map = anomaly_status.map({'Normal': 0, 'Anomaly': 1}).values
            
            # Create dimensions for parallel coordinates
            dimensions = []
            for col in features_to_plot:
                dimensions.append(dict(
                    range=[0, 1],
                    label=col,
                    values=normalized_data[col].values,
                    tickvals=[0, 0.5, 1],
                    ticktext=['Min', 'Mid', 'Max']
                ))
            
            # Create the plot
            fig = go.Figure(data=go.Parcoords(
                line=dict(
                    color=color_map,
                    colorscale=[[0, 'rgba(0,100,255,0.6)'], [1, 'rgba(255,0,0,0.8)']],
                    showscale=True,
                    colorbar=dict(
                        title="Status",
                        tickvals=[0, 1],
                        ticktext=['Normal', 'Anomaly']
                    ),
                    cmin=0,
                    cmax=1
                ),
                dimensions=dimensions
            ))
            
            fig.update_layout(
                title="Parallel Coordinates Plot: Normal vs Anomalous Data",
                title_x=0.5,
                font=dict(size=12),
                height=500,
                margin=dict(l=100, r=100, t=80, b=50)
            )
            
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error creating parallel coordinates chart: {str(e)}")
            return self._create_error_chart(f"Error creating parallel coordinates plot: {str(e)}")

    def _create_feature_importance_chart(self, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create feature importance chart based on anomaly detection results"""
        try:
            # Collect feature importance from different methods
            feature_scores = {}
            
            # Analyze feature importance across all methods
            for method, result in anomaly_results.items():
                if 'error' in result or not result.get('anomaly_indices'):
                    continue
                
                # For isolation forest, we can get feature importances
                if method == 'isolation_forest' and 'anomaly_scores' in result:
                    # Use variance of anomaly scores as a proxy for importance
                    scores = result['anomaly_scores']
                    feature_scores[f'{method}_score'] = np.var(scores)
                
                # For other methods, use detection rate as importance
                anomaly_count = result.get('anomaly_count', 0)
                total_samples = len(result.get('anomaly_scores', [])) if 'anomaly_scores' in result else 1000
                detection_rate = (anomaly_count / max(total_samples, 1)) * 100
                feature_scores[f'{method}_detection_rate'] = detection_rate
            
            if not feature_scores:
                return self._create_error_chart("No feature importance data available")
            
            # Create chart showing method performance as feature importance
            methods = list(feature_scores.keys())
            scores = list(feature_scores.values())
            
            # Clean method names for display
            clean_methods = [m.replace('_', ' ').title() for m in methods]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=clean_methods,
                    y=scores,
                    marker_color='lightblue',
                    text=[f"{score:.2f}" for score in scores],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Method Performance Analysis (Feature Detection Capability)",
                title_x=0.5,
                xaxis_title="Detection Methods",
                yaxis_title="Performance Score",
                font=dict(size=12),
                xaxis=dict(tickangle=45)
            )
            
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {str(e)}")
            return self._create_error_chart(f"Error creating feature importance chart: {str(e)}")

    def _create_method_comparison_chart(self, anomaly_results: Dict[str, Any]) -> str:
        """Create method comparison chart"""
        methods = []
        counts = []
        percentages = []
        
        for method, result in anomaly_results.items():
            if 'error' not in result:
                methods.append(method.replace('_', ' ').title())
                counts.append(result.get('anomaly_count', 0))
                percentages.append(result.get('anomaly_percentage', 0))
        
        if not methods:
            fig = go.Figure()
            fig.add_annotation(
                text="No method results available for comparison",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Method Comparison")
            return fig.to_json()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Anomaly Counts', 'Anomaly Percentages'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add count bars
        fig.add_trace(
            go.Bar(x=methods, y=counts, name='Count'),
            row=1, col=1
        )
        
        # Add percentage bars
        fig.add_trace(
            go.Bar(x=methods, y=percentages, name='Percentage'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Method Comparison: Anomaly Detection Results",
            title_x=0.5,
            font=dict(size=12),
            showlegend=False
        )
        
        return fig.to_json()

    def _create_3d_scatter_chart(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> str:
        """Create 3D scatter plot with anomalies highlighted"""
        try:
            # Use first 3 features
            features = data.columns[:3].tolist()
            
            fig = go.Figure()
            
            # Determine normal points
            normal_indices = data.index.tolist()
            for method, result in anomaly_results.items():
                if 'error' not in result:
                    anomaly_indices = result.get('anomaly_indices', [])
                    normal_indices = [idx for idx in normal_indices if idx not in anomaly_indices]
            
            # Add normal points
            if normal_indices:
                fig.add_trace(go.Scatter3d(
                    x=data.loc[normal_indices, features[0]],
                    y=data.loc[normal_indices, features[1]],
                    z=data.loc[normal_indices, features[2]],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=4, opacity=0.6)
                ))
            
            # Add anomalies
            colors = ['red', 'orange', 'purple', 'green', 'pink']
            for i, (method, result) in enumerate(anomaly_results.items()):
                if 'error' not in result:
                    anomaly_indices = result.get('anomaly_indices', [])
                    if anomaly_indices:
                        valid_indices = [idx for idx in anomaly_indices if idx in data.index]
                        if valid_indices:
                            fig.add_trace(go.Scatter3d(
                                x=data.loc[valid_indices, features[0]],
                                y=data.loc[valid_indices, features[1]],
                                z=data.loc[valid_indices, features[2]],
                                mode='markers',
                                name=f'{method.replace("_", " ").title()}',
                                marker=dict(color=colors[i % len(colors)], size=8, symbol='x')
                            ))
            
            fig.update_layout(
                title="3D Scatter Plot with Anomalies",
                title_x=0.5,
                scene=dict(
                    xaxis_title=features[0],
                    yaxis_title=features[1],
                    zaxis_title=features[2]
                ),
                font=dict(size=12)
            )
            
            return fig.to_json()
            
        except Exception as e:
            logger.error(f"Error creating 3D scatter chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text="Could not create 3D scatter plot",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12)
            )
            fig.update_layout(title="3D Scatter Plot")
            return fig.to_json()

    def _analyze_mutual_information(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze mutual information between all feature pairs"""
        try:
            features = data.columns.tolist()
            n_features = len(features)
            
            # Calculate mutual information matrix
            mi_matrix = np.zeros((n_features, n_features))
            mi_pairs = []
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    feature1, feature2 = features[i], features[j]
                    
                    # Clean data for mutual information calculation
                    clean_data = data[[feature1, feature2]].dropna()
                    if len(clean_data) > 10:
                        # Calculate mutual information
                        mi_score = mutual_info_score(
                            pd.cut(clean_data[feature1], bins=20, labels=False, duplicates='drop'),
                            pd.cut(clean_data[feature2], bins=20, labels=False, duplicates='drop')
                        )
                        mi_matrix[i, j] = mi_score
                        mi_matrix[j, i] = mi_score
                        
                        mi_pairs.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'mutual_info': float(mi_score),
                            'relationship': f"{feature1} ↔ {feature2}"
                        })
            
            # Sort pairs by mutual information score
            mi_pairs.sort(key=lambda x: x['mutual_info'], reverse=True)
            
            # Get top relationships
            top_relationships = mi_pairs[:5]
            
            return {
                'mutual_info_matrix': mi_matrix.tolist(),
                'feature_names': features,
                'all_pairs': mi_pairs,
                'top_relationships': top_relationships,
                'summary': {
                    'total_pairs': len(mi_pairs),
                    'avg_mutual_info': float(np.mean([p['mutual_info'] for p in mi_pairs])),
                    'max_mutual_info': float(max([p['mutual_info'] for p in mi_pairs])) if mi_pairs else 0,
                    'min_mutual_info': float(min([p['mutual_info'] for p in mi_pairs])) if mi_pairs else 0
                }
            }
        except Exception as e:
            logger.error(f"Mutual information analysis failed: {str(e)}")
            return {
                'error': str(e),
                'summary': {'total_pairs': 0}
            }

    def _perform_hierarchical_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform hierarchical clustering analysis on features"""
        try:
            # Calculate correlation matrix and convert to distance matrix
            corr_matrix = data.corr().abs()
            distance_matrix = 1 - corr_matrix
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix, method='ward')
            
            # Get cluster assignments (using distance threshold)
            n_clusters = min(5, len(data.columns))  # Maximum 5 clusters
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Group features by cluster
            clusters = {}
            for i, feature in enumerate(data.columns):
                cluster_id = cluster_labels[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(feature)
            
            # Format cluster results
            cluster_results = []
            for cluster_id, features in clusters.items():
                cluster_results.append({
                    'cluster_id': int(cluster_id),
                    'features': features,
                    'size': len(features)
                })
            
            return {
                'linkage_matrix': linkage_matrix.tolist(),
                'feature_names': data.columns.tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'clusters': cluster_results,
                'n_clusters': n_clusters,
                'distance_matrix': distance_matrix.values.tolist()
            }
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {str(e)}")
            return {
                'error': str(e),
                'clusters': []
            }

    def _analyze_cross_correlation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cross-correlation between feature pairs"""
        try:
            features = data.columns.tolist()
            correlations = []
            
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    feature1, feature2 = features[i], features[j]
                    
                    # Clean data for correlation calculation
                    clean_data = data[[feature1, feature2]].dropna()
                    if len(clean_data) > 10:
                        correlation, p_value = pearsonr(clean_data[feature1], clean_data[feature2])
                        
                        correlations.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'significance': 'significant' if p_value < 0.05 else 'not_significant',
                            'strength': self._interpret_correlation_strength(abs(correlation))
                        })
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'all_correlations': correlations,
                'top_correlations': correlations[:10],
                'summary': {
                    'total_pairs': len(correlations),
                    'significant_correlations': len([c for c in correlations if c['significance'] == 'significant']),
                    'strong_correlations': len([c for c in correlations if abs(c['correlation']) > 0.7]),
                    'avg_correlation': float(np.mean([abs(c['correlation']) for c in correlations])) if correlations else 0
                }
            }
        except Exception as e:
            logger.error(f"Cross-correlation analysis failed: {str(e)}")
            return {
                'error': str(e),
                'summary': {'total_pairs': 0}
            }

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation >= 0.8:
            return 'very_strong'
        elif correlation >= 0.6:
            return 'strong'
        elif correlation >= 0.4:
            return 'moderate'
        elif correlation >= 0.2:
            return 'weak'
        else:
            return 'very_weak'

    def _analyze_variance_changes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze variance changes across features to identify potentially unreliable sensors"""
        try:
            window_size = max(50, len(data) // 20)  # Adaptive window size
            variance_changes = {}
            
            for column in data.columns:
                series = data[column].dropna()
                if len(series) < window_size * 2:
                    continue
                
                # Calculate rolling variance
                rolling_var = series.rolling(window=window_size).var()
                
                # Calculate variance changes (difference between consecutive variance values)
                var_changes = rolling_var.diff().abs()
                
                # Calculate average variance change
                avg_var_change = var_changes.mean()
                max_var_change = var_changes.max()
                std_var_change = var_changes.std()
                
                variance_changes[column] = {
                    'avg_variance_change': float(avg_var_change) if not pd.isna(avg_var_change) else 0,
                    'max_variance_change': float(max_var_change) if not pd.isna(max_var_change) else 0,
                    'std_variance_change': float(std_var_change) if not pd.isna(std_var_change) else 0,
                    'variance_instability_score': float(avg_var_change * std_var_change) if not pd.isna(avg_var_change * std_var_change) else 0
                }
            
            # Calculate threshold for high variance change (mean + 2*std)
            avg_changes = [v['avg_variance_change'] for v in variance_changes.values()]
            if avg_changes:
                threshold = np.mean(avg_changes) + 2 * np.std(avg_changes)
                
                # Identify potentially unreliable sensors
                unreliable_sensors = [
                    feature for feature, stats in variance_changes.items()
                    if stats['avg_variance_change'] > threshold
                ]
            else:
                threshold = 0
                unreliable_sensors = []
            
            return {
                'variance_changes': variance_changes,
                'threshold': float(threshold),
                'unreliable_sensors': unreliable_sensors,
                'summary': {
                    'total_features': len(variance_changes),
                    'unreliable_count': len(unreliable_sensors),
                    'avg_variance_change': float(np.mean(avg_changes)) if avg_changes else 0,
                    'max_variance_change': float(np.max(avg_changes)) if avg_changes else 0
                }
            }
        except Exception as e:
            logger.error(f"Variance change analysis failed: {str(e)}")
            return {
                'error': str(e),
                'summary': {'total_features': 0}
            }

    def _create_hierarchical_dendrogram_chart(self) -> Dict[str, Any]:
        """Create hierarchical clustering dendrogram"""
        try:
            if not hasattr(self, 'hierarchical_clustering_results') or not self.hierarchical_clustering_results:
                return self._create_error_chart("Hierarchical clustering results not available")
            
            hc_results = self.hierarchical_clustering_results
            if 'error' in hc_results:
                return self._create_error_chart(f"Hierarchical clustering error: {hc_results['error']}")
            
            linkage_matrix = np.array(hc_results['linkage_matrix'])
            feature_names = hc_results['feature_names']
            
            # Create a proper dendrogram using Plotly's built-in functionality
            # First, let's use scipy's dendrogram to get the data
            from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
            
            # Get dendrogram data
            dend_data = scipy_dendrogram(linkage_matrix, labels=feature_names, no_plot=True)
            
            # Create figure using Plotly
            fig = go.Figure()
            
            # Add dendrogram lines
            for i in range(len(dend_data['icoord'])):
                x = dend_data['icoord'][i]
                y = dend_data['dcoord'][i]
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line=dict(color='rgb(0,100,200)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add feature labels at the bottom
            for i, label in enumerate(dend_data['ivl']):
                x_pos = 5 + i * 10  # Position labels evenly
                fig.add_annotation(
                    x=x_pos,
                    y=-0.05 * max(dend_data['dcoord'][0]) if dend_data['dcoord'] else 0,
                    text=label,
                    showarrow=False,
                    textangle=-45,
                    font=dict(size=10, color='black'),
                    xanchor='right'
                )
            
            # Update layout for better visibility
            fig.update_layout(
                title="Feature Hierarchical Clustering Dendrogram",
                title_x=0.5,
                xaxis=dict(
                    title="Features",
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False
                ),
                yaxis=dict(
                    title="Linkage Distance",
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                font=dict(size=12),
                showlegend=False,
                plot_bgcolor='white',
                height=500,
                margin=dict(b=100)  # Extra bottom margin for labels
            )
            
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error creating dendrogram chart: {str(e)}")
            return self._create_error_chart(f"Error creating dendrogram: {str(e)}")

    def _create_mutual_information_heatmap(self) -> Dict[str, Any]:
        """Create mutual information heatmap"""
        try:
            if not hasattr(self, 'mutual_info_results') or not self.mutual_info_results:
                return self._create_error_chart("Mutual information results not available")
            
            mi_results = self.mutual_info_results
            if 'error' in mi_results:
                return self._create_error_chart(f"Mutual information error: {mi_results['error']}")
            
            mi_matrix = np.array(mi_results['mutual_info_matrix'])
            feature_names = mi_results['feature_names']
            
            fig = go.Figure(data=go.Heatmap(
                z=mi_matrix,
                x=feature_names,
                y=feature_names,
                colorscale='Viridis',
                colorbar=dict(title="Mutual Information"),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Mutual Information Heatmap",
                title_x=0.5,
                font=dict(size=12),
                xaxis=dict(tickangle=45),
                yaxis=dict(tickangle=0)
            )
            
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error creating mutual information heatmap: {str(e)}")
            return self._create_error_chart(f"Error creating heatmap: {str(e)}")

    def _create_cross_correlation_chart(self) -> Dict[str, Any]:
        """Create cross-correlation visualization"""
        try:
            if not hasattr(self, 'cross_correlation_results') or not self.cross_correlation_results:
                return self._create_error_chart("Cross-correlation results not available")
            
            corr_results = self.cross_correlation_results
            if 'error' in corr_results:
                return self._create_error_chart(f"Cross-correlation error: {corr_results['error']}")
            
            top_correlations = corr_results.get('top_correlations', [])[:10]
            
            if not top_correlations:
                return self._create_error_chart("No correlation data available")
            
            # Create bar chart of top correlations
            features = [f"{c['feature1']} ↔ {c['feature2']}" for c in top_correlations]
            correlations = [c['correlation'] for c in top_correlations]
            colors = ['red' if c < 0 else 'blue' for c in correlations]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=features,
                    y=correlations,
                    marker_color=colors,
                    text=[f"{c:.3f}" for c in correlations],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Top Cross-Correlations Between Features",
                title_x=0.5,
                xaxis_title="Feature Pairs",
                yaxis_title="Correlation Coefficient",
                font=dict(size=12),
                xaxis=dict(tickangle=45)
            )
            
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error creating cross-correlation chart: {str(e)}")
            return self._create_error_chart(f"Error creating correlation chart: {str(e)}")

    def _create_variance_change_chart(self) -> Dict[str, Any]:
        """Create variance change analysis chart"""
        try:
            if not hasattr(self, 'variance_change_results') or not self.variance_change_results:
                return self._create_error_chart("Variance change results not available")
            
            var_results = self.variance_change_results
            if 'error' in var_results:
                return self._create_error_chart(f"Variance change error: {var_results['error']}")
            
            variance_changes = var_results.get('variance_changes', {})
            threshold = var_results.get('threshold', 0)
            
            if not variance_changes:
                return self._create_error_chart("No variance change data available")
            
            features = list(variance_changes.keys())
            avg_changes = [variance_changes[f]['avg_variance_change'] for f in features]
            colors = ['red' if change > threshold else 'blue' for change in avg_changes]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=features,
                    y=avg_changes,
                    marker_color=colors,
                    text=[f"{c:.2f}" for c in avg_changes],
                    textposition='auto'
                )
            ])
            
            # Add threshold line
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold:.2f}"
            )
            
            fig.update_layout(
                title="Average Variance Change by Feature",
                title_x=0.5,
                xaxis_title="Features",
                yaxis_title="Average Variance Change",
                font=dict(size=12),
                xaxis=dict(tickangle=45)
            )
            
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            logger.error(f"Error creating variance change chart: {str(e)}")
            return self._create_error_chart(f"Error creating variance chart: {str(e)}")

    def _create_error_chart(self, error_message: str) -> Dict[str, Any]:
        """Create a chart showing an error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color="red")
        )
        fig.update_layout(title="Chart Error")
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    def _generate_anomaly_summary_insight(self, anomaly_results: Dict[str, Any]) -> str:
        """Generate insight for anomaly summary chart"""
        try:
            method_totals = {}
            for method, method_data in anomaly_results.items():
                if isinstance(method_data, dict) and 'anomaly_count' in method_data:
                    method_totals[method] = method_data['anomaly_count']
            
            if not method_totals:
                return "No anomaly detection results available for analysis"
            
            total_anomalies = sum(method_totals.values())
            best_method = max(method_totals, key=method_totals.get)
            
            return f"""
🔍 **Multivariate Anomaly Detection Overview**

**Detection Summary:**
• {total_anomalies} total anomalies identified across all methods
• {best_method} proved most effective ({method_totals[best_method]} detections)
• {len(method_totals)} different detection approaches compared

**Method Characteristics:**
Each multivariate method captures different anomaly types: Isolation Forest detects outliers in sparse regions, Local Outlier Factor finds density-based anomalies, Elliptic Envelope assumes normal distribution, and DBSCAN identifies non-clustered points.

**Understanding the Results:**
Higher detection counts don't always indicate better performance - they reflect method sensitivity and the specific types of patterns each algorithm targets. The goal is comprehensive coverage rather than maximum detections.

**Key Insights:**
Multivariate methods excel at finding complex relationships between variables that single-variable analysis would miss. They detect points that are unusual in the combined space of all features, even if individual features appear normal.
            """.strip()
        except Exception as e:
            return f"Error generating anomaly summary insight: {str(e)}"

    def _generate_pca_insight(self, scaled_data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> str:
        """Generate insight for PCA visualization"""
        try:
            n_features = len(scaled_data.columns)
            n_samples = len(scaled_data)
            
            # Calculate PCA for insight
            pca = PCA(n_components=min(2, n_features))
            pca_data = pca.fit_transform(scaled_data)
            
            explained_variance = pca.explained_variance_ratio_
            total_variance = sum(explained_variance)
            
            return f"""
🎯 **PCA Anomaly Visualization Analysis**

**Dimensionality Reduction:**
• {n_features} original features compressed into 2D visualization
• {total_variance:.1%} of total variance preserved (PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%})
• {n_samples} data points projected into principal component space

**Anomaly Distribution Pattern:**
Anomalies typically appear as isolated points or small clusters in the PCA space, separated from the main data distribution. Distance from the center generally correlates with anomaly severity.

**Visual Interpretation Guide:**
• Main cluster: Normal data points grouped together
• Isolated points: Potential outliers in the multi-dimensional space
• Color coding: Different methods highlight different anomaly types
• Overlapping colors: High-confidence anomalies detected by multiple methods

**PCA Effectiveness:**
{"PCA representation captures most data variance - reliable for visualization" if total_variance > 0.7 else "Consider additional components for more complete representation"}

**Understanding the Space:**
PCA transforms your original features into new dimensions that capture maximum variance. Anomalies in this space represent unusual combinations of the original features, even if individual features seem normal.
            """.strip()
        except Exception as e:
            return f"Error generating PCA insight: {str(e)}"

    def _generate_correlation_matrix_insight(self, data: pd.DataFrame) -> str:
        """Generate insight for correlation matrix"""
        try:
            corr_matrix = data.corr()
            
            # Find strong correlations
            strong_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.7:
                        strong_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            strong_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            return f"""
🔗 **Correlation Matrix Analysis**

**Feature Relationships:**
• Variables analyzed: {len(data.columns)}
• Strong correlations (>0.7): {len(strong_corr_pairs)}
• Color intensity indicates correlation strength

**Key Correlations:**
{chr(10).join([f"• {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}" for pair in strong_corr_pairs[:5]])}

**Anomaly Context:**
• Highly correlated features often show similar anomaly patterns
• Anomalies may violate expected correlation relationships
• Red/blue intensity shows positive/negative correlation strength

**Interpretation:**
Strong correlations indicate redundant information or underlying relationships. Anomalies may represent points where these relationships break down, suggesting interesting or problematic data points.

**Key Insights:**
• {"Consider dimensionality reduction due to high correlations" if len(strong_corr_pairs) > 3 else "Correlation levels are reasonable for analysis"}
• Investigate anomalies that violate strong correlation patterns - these may indicate data quality issues or interesting outliers
• Use correlation insights to guide feature selection and understand variable relationships
            """.strip()
        except Exception as e:
            return f"Error generating correlation matrix insight: {str(e)}"

    def _generate_parallel_coordinates_insight(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> str:
        """Generate insight for parallel coordinates plot"""
        try:
            n_features = len(data.columns)
            n_samples = len(data)
            
            # Count anomalies across methods
            total_anomalies = 0
            for method_data in anomaly_results.values():
                if isinstance(method_data, dict) and 'anomaly_count' in method_data:
                    total_anomalies += method_data['anomaly_count']
            
            return f"""
📊 **Parallel Coordinates Analysis**

**Multi-dimensional View:**
• Features displayed: {n_features}
• Each line represents one data point
• Anomalies highlighted in red/orange
• Normal points in blue/gray

**Pattern Recognition:**
• Parallel lines indicate similar patterns
• Crossing lines show diverse relationships
• Extreme values appear at plot boundaries

**Anomaly Characteristics:**
• Anomalous lines often cross normal patterns
• Extreme values in any dimension can indicate anomalies
• Patterns that deviate from the main flow are suspicious

**Insights:**
• This view reveals multi-dimensional outliers not visible in single-variable analysis
• Anomalies often show unusual combinations of values across features
• Use for feature engineering to create better anomaly detection rules
            """.strip()
        except Exception as e:
            return f"Error generating parallel coordinates insight: {str(e)}"

    def _generate_feature_importance_insight(self, anomaly_results: Dict[str, Any]) -> str:
        """Generate insight for feature importance chart"""
        try:
            return f"""
🎯 **Feature Importance for Anomaly Detection**

**Importance Ranking:**
• Features ranked by their contribution to anomaly detection
• Higher bars indicate more important features for identifying outliers
• Based on various detection methods' internal feature weights

**Interpretation:**
• Top-ranked features are most discriminative for anomaly detection
• These features show the most variation between normal and anomalous patterns
• Focus on these features for understanding anomaly characteristics

**Business Applications:**
• Prioritize monitoring and quality control for high-importance features
• Use for feature selection in anomaly detection models
• Guide investigation efforts toward most informative variables

**Key Insights:**
• Monitor top-ranked features more closely in production systems to catch anomalies early
• Consider feature engineering to enhance discriminative power of important variables
• Use importance weights to explain anomaly detection decisions to stakeholders
            """.strip()
        except Exception as e:
            return f"Error generating feature importance insight: {str(e)}"

    def _generate_method_comparison_insight(self, anomaly_results: Dict[str, Any]) -> str:
        """Generate insight for method comparison chart"""
        try:
            method_counts = {}
            for method, method_data in anomaly_results.items():
                if isinstance(method_data, dict) and 'anomaly_count' in method_data:
                    method_counts[method] = method_data['anomaly_count']
            
            if not method_counts:
                return "No method comparison data available"
            
            most_sensitive = max(method_counts, key=method_counts.get)
            least_sensitive = min(method_counts, key=method_counts.get)
            
            return f"""
⚖️ **Method Comparison Analysis**

**Detection Sensitivity:**
• Most sensitive: {most_sensitive} ({method_counts[most_sensitive]} anomalies)
• Least sensitive: {least_sensitive} ({method_counts[least_sensitive]} anomalies)
• Methods compared: {len(method_counts)}

**Method Characteristics:**
• Isolation Forest: Detects outliers in sparse regions
• Local Outlier Factor: Finds points with unusual local density
• Elliptic Envelope: Assumes normal distribution, detects statistical outliers
• DBSCAN: Identifies points not belonging to any cluster

**Interpretation:**
Different methods capture different types of anomalies. High sensitivity might indicate either better detection or higher false positive rates. Consider the specific characteristics of your data when choosing methods.

**Key Insights:**
• Use ensemble approach combining multiple methods for robust anomaly detection
• Validate high-sensitivity methods for false positives before taking action
• Choose methods based on your specific anomaly types and data characteristics
• Consider business context when interpreting sensitivity differences between methods
            """.strip()
        except Exception as e:
            return f"Error generating method comparison insight: {str(e)}"

    def _generate_hierarchical_dendrogram_insight(self) -> str:
        """Generate insight for hierarchical clustering dendrogram"""
        try:
            hierarchical_results = getattr(self, 'hierarchical_clustering_results', {})
            
            return f"""
🌳 **Hierarchical Clustering Dendrogram**

**Cluster Structure:**
• Tree structure shows how data points group together
• Height indicates dissimilarity between clusters
• Anomalies typically appear as single-point clusters or outliers

**Interpretation:**
• Long branches indicate isolated points (potential anomalies)
• Short branches show similar data points clustering together
• Cut-off height determines number of clusters

**Anomaly Detection:**
• Points forming individual clusters at high cut-off levels are anomalies
• Singletons separated early in the hierarchy are strong outlier candidates
• Small clusters might represent systematic anomalies

**Business Applications:**
• Use for understanding data structure and natural groupings
• Identify anomalous patterns that don't fit main clusters
• Guide threshold selection for clustering-based anomaly detection

**Key Insights:**
• Investigate single-point clusters at high levels as they likely represent strong outliers
• Use cluster membership to validate other anomaly detection methods and cross-check results
• Consider domain knowledge when interpreting cluster structure and anomaly patterns
            """.strip()
        except Exception as e:
            return f"Error generating hierarchical dendrogram insight: {str(e)}"

    def _generate_mutual_information_insight(self) -> str:
        """Generate insight for mutual information heatmap"""
        try:
            mutual_info_results = getattr(self, 'mutual_info_results', {})
            
            return f"""
🔗 **Mutual Information Analysis**

**Information Relationships:**
• Shows non-linear relationships between variables
• Higher values indicate stronger information dependency
• Complements correlation analysis with non-linear relationships

**Interpretation:**
• High mutual information suggests predictive relationships
• Low values indicate independence
• Asymmetric relationships possible (unlike correlation)

**Anomaly Context:**
• Anomalies often violate expected information relationships
• Features with high mutual information should show consistent patterns
• Deviations from expected information transfer indicate anomalies

**Insights:**
• Use to identify feature combinations that are informative for anomaly detection
• Understand complex relationships beyond linear correlation
• Guide feature selection for machine learning models

**Key Insights:**
• Investigate anomalies that violate high mutual information relationships as they may indicate data quality issues
• Use mutual information patterns for feature engineering and selection in anomaly detection models
• Consider non-linear relationships in anomaly detection strategies beyond simple correlation analysis
            """.strip()
        except Exception as e:
            return f"Error generating mutual information insight: {str(e)}"

    def _generate_cross_correlation_insight(self) -> str:
        """Generate insight for cross-correlation analysis"""
        try:
            cross_corr_results = getattr(self, 'cross_correlation_results', {})
            
            return f"""
🔄 **Cross-Correlation Network Analysis**

**Network Structure:**
• Nodes represent features
• Edge thickness indicates correlation strength
• Network topology shows feature relationships

**Interpretation:**
• Highly connected nodes are central features
• Isolated nodes may be independent variables
• Clusters in the network represent related feature groups

**Anomaly Detection:**
• Anomalies may disrupt expected network patterns
• Points that violate network relationships are suspicious
• Use network structure to understand anomaly propagation

**Business Applications:**
• Identify key features that influence multiple others
• Understand how anomalies in one feature affect others
• Guide monitoring strategies for interconnected features

**Key Insights:**
• Monitor highly connected features more closely as they have greater impact on system behavior
• Investigate anomalies that affect network hub features since they can propagate through the system
• Use network topology for feature importance ranking and resource allocation
            """.strip()
        except Exception as e:
            return f"Error generating cross-correlation insight: {str(e)}"

    def _generate_variance_change_insight(self) -> str:
        """Generate insight for variance change analysis"""
        try:
            variance_results = getattr(self, 'variance_change_results', {})
            
            return f"""
📊 **Variance Change Analysis**

**Stability Assessment:**
• Measures how feature variance changes over time or conditions
• High variance changes indicate instability
• Red bars show features exceeding threshold

**Interpretation:**
• Stable features maintain consistent variance
• High variance changes suggest systematic shifts
• Threshold violations indicate potential quality issues

**Anomaly Context:**
• Features with high variance changes are prone to anomalies
• Systematic variance changes may indicate data drift
• Use for proactive anomaly detection

**Business Applications:**
• Monitor data stability over time
• Identify features requiring additional quality control
• Guide model retraining schedules

**Key Insights:**
• Investigate features with high variance changes as they may indicate data quality issues or system changes
• Implement monitoring for variance drift to catch problems early in production
• Consider feature stability in model selection for robust anomaly detection systems
            """.strip()
        except Exception as e:
            return f"Error generating variance change insight: {str(e)}"

    def _generate_3d_scatter_insight(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> str:
        """Generate insight for 3D scatter plot"""
        try:
            n_features = len(data.columns)
            n_samples = len(data)
            
            # Count anomalies
            total_anomalies = 0
            for method_data in anomaly_results.values():
                if isinstance(method_data, dict) and 'anomaly_count' in method_data:
                    total_anomalies += method_data['anomaly_count']
            
            return f"""
🌐 **3D Scatter Plot Analysis**

**Three-Dimensional View:**
• Features visualized: 3 out of {n_features} total
• Enhanced perspective on multi-dimensional relationships
• Anomalies highlighted with different colors/shapes

**Spatial Patterns:**
• Anomalies often appear as isolated points in 3D space
• Clusters of normal points form dense regions
• Distance from main cluster indicates anomaly severity

**Interpretation:**
• 3D view reveals relationships not visible in 2D
• Anomalies may form patterns when viewed from different angles
• Use rotation to understand full spatial distribution

**Insights:**
• Provides intuitive understanding of multi-dimensional outliers
• Helps validate 2D projections and PCA results
• Shows how anomalies relate to multiple features simultaneously

**Key Insights:**
• Rotate view to explore different perspectives and understand spatial relationships
• Focus on points isolated in 3D space as they represent the strongest anomalies
• Use this visualization for explaining anomaly detection results to stakeholders
            """.strip()
        except Exception as e:
            return f"Error generating 3D scatter insight: {str(e)}"
