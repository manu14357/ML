# -*- coding: utf-8 -*-
"""
Univariate Anomaly Detection Service
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
from scipy import stats
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class UnivariateAnomalyDetectionService:
    """Statistical Univariate Anomaly Detection Service with traditional methods"""

    def __init__(self):
        self.results = {}
        self.contamination_rates = {}

    def detect_anomalies(self, df: pd.DataFrame, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Statistical anomaly detection in univariate data using traditional methods.
        
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
        zscore_threshold = parameters.get('zscore_threshold', 3.0)
        iqr_multiplier = parameters.get('iqr_multiplier', 1.5)
        spike_threshold = parameters.get('spike_threshold', 3.0)
        drift_window = parameters.get('drift_window', 50)
        flatline_threshold = parameters.get('flatline_threshold', 0.001)
        gap_threshold = parameters.get('gap_threshold', 10)
        
        logger.info(f"Starting statistical univariate anomaly detection with method: {method}")
        
        results = {}

        try:
            # Select feature columns
            if feature_columns is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in feature_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_columns:
                return {
                    'success': False,
                    'error': 'No numeric columns found for anomaly detection'
                }
            
            # Dataset information
            dataset_info = {
                'total_rows': len(df),
                'numeric_columns': len(numeric_columns),
                'columns_analyzed': numeric_columns,
                'analysis_methods': ['isolation_forest', 'zscore', 'iqr', 'modified_zscore']
            }
            results['dataset_info'] = dataset_info
            
            # Statistical anomaly detection results for each method
            anomaly_results = {}
            
            # Traditional statistical methods
            if method in ['all', 'zscore']:
                anomaly_results['zscore'] = self._detect_zscore_anomalies(df[numeric_columns], zscore_threshold)
            
            if method in ['all', 'iqr']:
                anomaly_results['iqr'] = self._detect_iqr_anomalies(df[numeric_columns], iqr_multiplier)
            
            if method in ['all', 'isolation_forest']:
                anomaly_results['isolation_forest'] = self._detect_isolation_forest_anomalies(df[numeric_columns], contamination)
            
            if method in ['all', 'modified_zscore']:
                anomaly_results['modified_zscore'] = self._detect_modified_zscore_anomalies(df[numeric_columns])
            
            results['anomaly_results'] = anomaly_results
            
            # Generate comprehensive summary report
            comprehensive_summary = self._generate_comprehensive_summary(df, anomaly_results, numeric_columns)
            results['comprehensive_summary'] = comprehensive_summary
            
            # Combine results and create analysis
            combined_results = self._combine_anomaly_results(df, anomaly_results, numeric_columns)
            results['combined_analysis'] = combined_results
            
            # Generate advanced visualizations
            charts = self._generate_advanced_anomaly_charts(df, anomaly_results, numeric_columns, parameters)
            results['charts'] = charts
            
            # Statistical summary
            stats_summary = self._generate_statistical_summary(df, anomaly_results, numeric_columns)
            results['statistical_summary'] = stats_summary
            
            # Advanced recommendations
            recommendations = self._generate_advanced_recommendations(anomaly_results, combined_results, comprehensive_summary)
            results['recommendations'] = recommendations
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            results['analysis_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Statistical univariate anomaly detection completed in {execution_time:.2f} seconds")
            
            return {
                'success': True,
                'results': results,
                'metadata': {
                    'analysis_type': 'statistical_univariate_anomaly_detection',
                    'execution_time': execution_time,
                    'parameters_used': parameters,
                    'total_anomalies_detected': sum(
                        sum(col_data.get('anomaly_count', 0) for col_data in method_data.values())
                        for method_data in anomaly_results.values()
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error in statistical univariate anomaly detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _detect_zscore_anomalies(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect anomalies using Z-score method"""
        results = {}
        
        for col in df.columns:
            non_null_data = df[col].dropna()
            if len(non_null_data) == 0:
                continue
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(non_null_data))
            anomaly_indices = non_null_data.index[z_scores > threshold]
            
            results[col] = {
                'method': 'zscore',
                'threshold': threshold,
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(non_null_data)) * 100,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_values': non_null_data[anomaly_indices].tolist(),
                'z_scores': z_scores.tolist(),
                'max_z_score': float(np.max(z_scores)),
                'mean_z_score': float(np.mean(z_scores))
            }
        
        return results

    def _detect_iqr_anomalies(self, df: pd.DataFrame, multiplier: float = 1.5) -> Dict[str, Any]:
        """Detect anomalies using IQR method"""
        results = {}
        
        for col in df.columns:
            non_null_data = df[col].dropna()
            if len(non_null_data) == 0:
                continue
            
            # Calculate IQR
            Q1 = non_null_data.quantile(0.25)
            Q3 = non_null_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Find anomalies
            anomalies = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
            
            results[col] = {
                'method': 'iqr',
                'multiplier': multiplier,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'anomaly_count': len(anomalies),
                'anomaly_percentage': (len(anomalies) / len(non_null_data)) * 100,
                'anomaly_indices': anomalies.index.tolist(),
                'anomaly_values': anomalies.tolist()
            }
        
        return results

    def _detect_isolation_forest_anomalies(self, df: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        results = {}
        
        for col in df.columns:
            non_null_data = df[col].dropna()
            if len(non_null_data) < 10:  # Need minimum data points
                continue
            
            # Reshape for sklearn
            data_reshaped = non_null_data.values.reshape(-1, 1)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(data_reshaped)
            
            # Get anomaly indices and scores
            anomaly_indices = non_null_data.index[anomaly_labels == -1]
            anomaly_scores = iso_forest.decision_function(data_reshaped)
            
            results[col] = {
                'method': 'isolation_forest',
                'contamination': contamination,
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(non_null_data)) * 100,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_values': non_null_data[anomaly_indices].tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'min_score': float(np.min(anomaly_scores)),
                'max_score': float(np.max(anomaly_scores)),
                'mean_score': float(np.mean(anomaly_scores))
            }
        
        return results

    def _detect_modified_zscore_anomalies(self, df: pd.DataFrame, threshold: float = 3.5) -> Dict[str, Any]:
        """Detect anomalies using Modified Z-score (using median)"""
        results = {}
        
        for col in df.columns:
            non_null_data = df[col].dropna()
            if len(non_null_data) == 0:
                continue
            
            # Calculate modified Z-score
            median = np.median(non_null_data)
            mad = np.median(np.abs(non_null_data - median))
            
            if mad == 0:
                modified_z_scores = np.zeros(len(non_null_data))
            else:
                modified_z_scores = 0.6745 * (non_null_data - median) / mad
            
            anomaly_indices = non_null_data.index[np.abs(modified_z_scores) > threshold]
            
            results[col] = {
                'method': 'modified_zscore',
                'threshold': threshold,
                'median': float(median),
                'mad': float(mad),
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(non_null_data)) * 100,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_values': non_null_data[anomaly_indices].tolist(),
                'modified_z_scores': modified_z_scores.tolist(),
                'max_modified_z_score': float(np.max(np.abs(modified_z_scores)))
            }
        
        return results

    def _generate_comprehensive_summary(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Generate comprehensive summary report in tabular format"""
        try:
            # Calculate summary statistics for each method and column
            method_names = {
                'isolation_forest': 'Isolation Forest',
                'zscore': 'Z-Score',
                'iqr': 'IQR',
                'modified_zscore': 'Mod Z-Score'
            }
            
            # Create summary table data
            summary_data = []
            total_by_method = {}
            
            for col in numeric_columns:
                row_data = {'column': col}
                total_anomalies = 0
                
                for method, method_results in anomaly_results.items():
                    if col in method_results:
                        count = method_results[col].get('anomaly_count', 0)
                        row_data[method] = count
                        total_anomalies += count
                        
                        if method not in total_by_method:
                            total_by_method[method] = 0
                        total_by_method[method] += count
                    else:
                        row_data[method] = 0
                
                row_data['total'] = total_anomalies
                summary_data.append(row_data)
            
            # Generate formatted summary report
            report = """🔍 **STATISTICAL ANOMALY DETECTION RESULTS**
================================================================================

📊 **Summary Overview - Statistical Detection Methods:**

┌────────────────────┬────────────┬────────────┬──────────────┬──────────────┬──────────┐
│    Column Name     │  Z-Score   │    IQR     │ Isolation F. │  Mod Z-Score │  Total   │
├────────────────────┼────────────┼────────────┼──────────────┼──────────────┼──────────┤"""

            for row in summary_data:
                col_name = row['column'][:18]  # Truncate long names
                zscore = row.get('zscore', 0)
                iqr = row.get('iqr', 0)
                isolation = row.get('isolation_forest', 0)
                mod_zscore = row.get('modified_zscore', 0)
                total = row.get('total', 0)
                
                report += f"""
│{col_name:^20}│{zscore:^12}│{iqr:^12}│{isolation:^14}│{mod_zscore:^14}│{total:^10}│"""
            
            report += """
└────────────────────┴────────────┴────────────┴────────────┴────────────┴──────────┘

📈 **Summary Statistics:**
   • Total Columns Analyzed: {}
   • Total Anomalies Detected: {:,}
   • Average Anomalies per Column: {:.2f}""".format(
                len(numeric_columns),
                sum(row.get('total', 0) for row in summary_data),
                sum(row.get('total', 0) for row in summary_data) / len(numeric_columns) if numeric_columns else 0
            )
            
            # Add detailed analysis for each method
            for method, method_results in anomaly_results.items():
                if not method_results:
                    continue
                    
                method_display = method_names.get(method, method.title())
                report += f"""

🔍 **{method_display} Detection - Detailed Analysis:**

┌─────────────────────────┬──────────┬────────────┬───────────────┬───────────────┐
│       Column Name       │  Count   │ Percentage │    Status     │  Method Info  │
├─────────────────────────┼──────────┼────────────┼───────────────┼───────────────┤"""

                method_total = 0
                has_data = False
                
                for col in numeric_columns:
                    if col in method_results:
                        has_data = True
                        count = method_results[col].get('anomaly_count', 0)
                        percentage = method_results[col].get('anomaly_percentage', 0)
                        status = "🚨 Found" if count > 0 else "✅ Clean"
                        
                        # Method-specific info
                        if method == 'isolation_forest':
                            info = "Isolation For.."
                        elif method == 'zscore':
                            info = "Z-Score"
                        elif method == 'iqr':
                            info = "IQR"
                        elif method == 'modified_zscore':
                            info = "Mod Z-Score"
                        else:
                            info = method.title()
                        
                        col_display = col[:23] if len(col) > 23 else col
                        report += f"""
│{col_display:^25}│{count:^10}│{percentage:^11.2f}%│{status:^15}│{info:^15}│"""
                        method_total += count
                
                if has_data:
                    report += f"""
├─────────────────────────┼──────────┼────────────┼───────────────┼───────────────┤
│     TOTAL ANOMALIES     │{method_total:^10}│            │               │               │
└─────────────────────────┴──────────┴────────────┴───────────────┴───────────────┘"""
                else:
                    report += """

   ✅ No anomalies detected using this method."""
            
            return report
            
        except Exception as e:
            return f"Error generating comprehensive summary: {str(e)}"

    def _combine_anomaly_results(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> Dict[str, Any]:
        """Combine results from different anomaly detection methods"""
        combined = {}
        
        for col in numeric_columns:
            # Collect all anomaly indices from different methods
            all_anomaly_indices = set()
            method_counts = {}
            
            for method, method_results in anomaly_results.items():
                if col in method_results:
                    indices = method_results[col].get('anomaly_indices', [])
                    all_anomaly_indices.update(indices)
                    method_counts[method] = len(indices)
            
            # Create consensus analysis
            consensus_anomalies = []
            for idx in all_anomaly_indices:
                method_detections = []
                for method, method_results in anomaly_results.items():
                    if col in method_results and idx in method_results[col].get('anomaly_indices', []):
                        method_detections.append(method)
                
                consensus_anomalies.append({
                    'index': idx,
                    'value': float(df.loc[idx, col]) if idx in df.index else None,
                    'detected_by': method_detections,
                    'detection_count': len(method_detections)
                })
            
            # Sort by detection count (most consensus first)
            consensus_anomalies.sort(key=lambda x: x['detection_count'], reverse=True)
            
            combined[col] = {
                'total_unique_anomalies': len(all_anomaly_indices),
                'method_counts': method_counts,
                'consensus_anomalies': consensus_anomalies,
                'high_confidence_anomalies': [a for a in consensus_anomalies if a['detection_count'] >= 2],
                'anomaly_percentage': (len(all_anomaly_indices) / len(df[col].dropna())) * 100 if len(df[col].dropna()) > 0 else 0
            }
        
        return combined

    def _generate_statistical_summary(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> Dict[str, Any]:
        """Generate statistical summary of anomaly detection results"""
        summary = {}
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            col_summary = {
                'column_stats': {
                    'count': len(col_data),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data))
                },
                'anomaly_summary': {}
            }
            
            for method, method_results in anomaly_results.items():
                if col in method_results:
                    col_summary['anomaly_summary'][method] = {
                        'count': method_results[col]['anomaly_count'],
                        'percentage': method_results[col]['anomaly_percentage']
                    }
            
            summary[col] = col_summary
        
        return summary

    def _generate_advanced_recommendations(self, anomaly_results: Dict[str, Any], combined_results: Dict[str, Any], comprehensive_summary: str) -> List[str]:
        """Generate recommendations based on statistical anomaly analysis"""
        recommendations = []
        
        # Calculate overall statistics
        total_anomalies_by_method = {}
        total_columns = 0
        
        for method, method_results in anomaly_results.items():
            total_anomalies_by_method[method] = sum(
                col_data.get('anomaly_count', 0) for col_data in method_results.values()
            )
            if method_results:
                total_columns = max(total_columns, len(method_results))
        
        # Method-specific recommendations
        if 'zscore' in total_anomalies_by_method:
            zscore_count = total_anomalies_by_method['zscore']
            if zscore_count > total_columns * 10:
                recommendations.append("🔥 High Z-score anomaly rate detected. Consider reviewing data distribution and outlier treatment.")
            elif zscore_count > 0:
                recommendations.append("📊 Z-score anomalies detected. Review for statistical outliers.")
        
        if 'iqr' in total_anomalies_by_method:
            iqr_count = total_anomalies_by_method['iqr']
            if iqr_count > total_columns * 10:
                recommendations.append("📈 High IQR anomaly rate detected. Check for data distribution skewness.")
            elif iqr_count > 0:
                recommendations.append("📊 IQR anomalies found. Consider quartile-based outlier treatment.")
        
        if 'isolation_forest' in total_anomalies_by_method:
            isolation_count = total_anomalies_by_method['isolation_forest']
            if isolation_count > 0:
                recommendations.append("🌲 Isolation Forest detected anomalies. These may indicate complex multivariate outliers.")
        
        if 'modified_zscore' in total_anomalies_by_method:
            mod_zscore_count = total_anomalies_by_method['modified_zscore']
            if mod_zscore_count > 0:
                recommendations.append("📏 Modified Z-score anomalies detected. These are robust to extreme outliers.")
        
        # High confidence anomaly recommendations
        high_confidence_count = sum(
            len(results.get('high_confidence_anomalies', []))
            for results in combined_results.values()
        )
        
        if high_confidence_count > 0:
            recommendations.append(f"🎯 Found {high_confidence_count} high-confidence anomalies detected by multiple methods. Prioritize investigation of these points.")
        
        # Overall data quality recommendations
        total_anomalies = sum(total_anomalies_by_method.values())
        if total_anomalies > len(combined_results) * 50:
            recommendations.append("⚠️ Very high overall anomaly rate. Consider comprehensive data quality audit and preprocessing.")
        elif total_anomalies > len(combined_results) * 20:
            recommendations.append("🔍 Moderate anomaly rate detected. Implement enhanced data validation procedures.")
        elif total_anomalies < len(combined_results) * 5:
            recommendations.append("✅ Low anomaly rate indicates good data quality. Maintain current procedures.")
        
        # Method consensus recommendations
        methods_with_results = [method for method, count in total_anomalies_by_method.items() if count > 0]
        if len(methods_with_results) >= 3:
            recommendations.append("🔬 Multiple detection methods found anomalies. This suggests genuine statistical outliers requiring attention.")
        
        return recommendations

    def _generate_advanced_anomaly_charts(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], numeric_columns: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical anomaly detection charts using Plotly"""
        charts = {}
        
        try:
            # Statistical summary dashboard
            charts['anomaly_dashboard'] = {
                'chart': self._create_anomaly_dashboard(anomaly_results, numeric_columns),
                'insight': self._generate_dashboard_insight(anomaly_results, numeric_columns)
            }
            
            # Method comparison heatmap
            charts['method_comparison_heatmap'] = {
                'chart': self._create_method_comparison_heatmap(anomaly_results, numeric_columns),
                'insight': self._generate_heatmap_insight(anomaly_results, numeric_columns)
            }
            
            # Individual column analysis for ALL columns (for dropdown selection)
            charts['column_specific'] = {}
            for col in numeric_columns:
                charts['column_specific'][col] = {
                    'time_series_anomaly': {
                        'chart': self._create_column_time_series_chart(df, anomaly_results, col),
                        'insight': self._generate_column_analysis_insight(df, anomaly_results, col)
                    },
                    'box_plot_outliers': {
                        'chart': self._create_column_box_plot_chart(df, anomaly_results, col),
                        'insight': self._generate_column_box_plot_insight(df, anomaly_results, col)
                    },
                    'method_comparison': {
                        'chart': self._create_column_method_comparison_chart(df, anomaly_results, col),
                        'insight': self._generate_column_method_comparison_insight(df, anomaly_results, col)
                    }
                }
            
            # Statistical overview
            charts['statistical_overview'] = {
                'chart': self._create_statistical_overview_chart(df, anomaly_results, numeric_columns),
                'insight': self._generate_statistical_overview_insight(df, anomaly_results, numeric_columns)
            }
            
            # Box plots with anomalies (use first few columns for overview)
            columns_to_plot = numeric_columns[:3]  # Use first 3 columns for overview
            charts['box_plots'] = {
                'chart': self._create_box_plots_with_anomalies(df, anomaly_results, columns_to_plot),
                'insight': self._generate_box_plots_insight(df, anomaly_results, columns_to_plot)
            }
            
        except Exception as e:
            logger.error(f"Error generating statistical anomaly charts: {str(e)}")
            charts['error'] = str(e)
        
        return charts

    def _create_anomaly_dashboard(self, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Create comprehensive anomaly detection dashboard"""
        methods = list(anomaly_results.keys())
        
        # Calculate totals for each method
        method_totals = {}
        for method in methods:
            total = sum(
                col_results.get('anomaly_count', 0)
                for col_results in anomaly_results[method].values()
            )
            method_totals[method] = total
        
        # Create subplot dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Anomalies by Detection Method',
                'Anomaly Rate by Column',
                'Method Effectiveness',
                'Detection Coverage'
            ],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'pie'}, {'type': 'scatter'}]]
        )
        
        # Chart 1: Total anomalies by method
        fig.add_trace(
            go.Bar(
                x=list(method_totals.keys()),
                y=list(method_totals.values()),
                name='Total Anomalies',
                marker_color='darkred'
            ),
            row=1, col=1
        )
        
        # Chart 2: Anomaly rate by column
        column_totals = {}
        for col in numeric_columns:
            total = sum(
                method_results.get(col, {}).get('anomaly_count', 0)
                for method_results in anomaly_results.values()
            )
            column_totals[col] = total
        
        fig.add_trace(
            go.Bar(
                x=list(column_totals.keys()),
                y=list(column_totals.values()),
                name='Column Anomalies',
                marker_color='darkorange'
            ),
            row=1, col=2
        )
        
        # Chart 3: Method distribution pie chart
        fig.add_trace(
            go.Pie(
                labels=list(method_totals.keys()),
                values=list(method_totals.values()),
                name="Method Distribution"
            ),
            row=2, col=1
        )
        
        # Chart 4: Detection coverage scatter
        coverage_data = []
        for method in methods:
            columns_with_anomalies = sum(1 for col_results in anomaly_results[method].values() 
                                       if col_results.get('anomaly_count', 0) > 0)
            avg_percentage = np.mean([col_results.get('anomaly_percentage', 0) 
                                    for col_results in anomaly_results[method].values()])
            coverage_data.append({'method': method, 'coverage': columns_with_anomalies, 'avg_percentage': avg_percentage})
        
        fig.add_trace(
            go.Scatter(
                x=[d['coverage'] for d in coverage_data],
                y=[d['avg_percentage'] for d in coverage_data],
                mode='markers+text',
                text=[d['method'] for d in coverage_data],
                textposition="top center",
                marker=dict(size=15, color='darkblue'),
                name='Coverage vs Rate'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Advanced Anomaly Detection Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_method_comparison_heatmap(self, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Create heatmap comparing anomaly detection methods"""
        methods = list(anomaly_results.keys())
        
        # Create matrix data
        matrix_data = []
        for col in numeric_columns:
            row = []
            for method in methods:
                if col in anomaly_results[method]:
                    percentage = anomaly_results[method][col].get('anomaly_percentage', 0)
                    row.append(percentage)
                else:
                    row.append(0)
            matrix_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=methods,
            y=numeric_columns,
            colorscale='Reds',
            text=[[f'{val:.1f}%' for val in row] for row in matrix_data],
            texttemplate='%{text}',
            textfont={'size': 10},
            colorbar=dict(title="Anomaly Rate (%)")
        ))
        
        fig.update_layout(
            title='Anomaly Detection Method Comparison Heatmap',
            title_x=0.5,
            xaxis_title='Detection Methods',
            yaxis_title='Columns',
            height=max(400, len(numeric_columns) * 30),
            font=dict(size=12)
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_column_analysis_chart(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], column: str) -> str:
        """Create comprehensive analysis chart for a specific column"""
        col_data = df[column].dropna()
        
        # Create subplots for comprehensive analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'{column} - Time Series with Anomalies',
                f'{column} - Distribution with Outliers',
                f'{column} - Method Comparison',
                f'{column} - Anomaly Timeline'
            ],
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Chart 1: Time series with all anomalies
        fig.add_trace(
            go.Scatter(
                x=col_data.index,
                y=col_data.values,
                mode='lines',
                name=f'{column} Data',
                line=dict(color='blue', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add anomalies from different methods
        colors = ['red', 'orange', 'purple', 'green', 'brown', 'pink', 'gray', 'cyan']
        for i, (method, method_results) in enumerate(anomaly_results.items()):
            if column in method_results:
                anomaly_indices = method_results[column].get('anomaly_indices', [])
                if anomaly_indices:
                    anomaly_values = [df.loc[idx, column] for idx in anomaly_indices if idx in df.index]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_indices,
                            y=anomaly_values,
                            mode='markers',
                            name=f'{method}',
                            marker=dict(color=colors[i % len(colors)], size=6, symbol='x')
                        ),
                        row=1, col=1
                    )
        
        # Chart 2: Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=col_data,
                nbinsx=30,
                name='Distribution',
                opacity=0.7,
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # Chart 3: Method comparison for this column
        method_counts = []
        method_names = []
        for method, method_results in anomaly_results.items():
            if column in method_results:
                count = method_results[column].get('anomaly_count', 0)
                method_counts.append(count)
                method_names.append(method)
        
        fig.add_trace(
            go.Bar(
                x=method_names,
                y=method_counts,
                name='Method Comparison',
                marker_color='darkgreen'
            ),
            row=2, col=1
        )
        
        # Chart 4: Anomaly timeline (cumulative)
        all_anomaly_indices = set()
        for method_results in anomaly_results.values():
            if column in method_results:
                all_anomaly_indices.update(method_results[column].get('anomaly_indices', []))
        
        anomaly_timeline = sorted(list(all_anomaly_indices))
        cumulative_count = list(range(1, len(anomaly_timeline) + 1))
        
        if anomaly_timeline:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_timeline,
                    y=cumulative_count,
                    mode='lines+markers',
                    name='Cumulative Anomalies',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text=f"Comprehensive Analysis: {column}",
            title_x=0.5,
            showlegend=True
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_time_series_chart(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], columns: List[str]) -> str:
        """Create advanced time series chart with multiple columns and anomaly highlighting"""
        if not columns:
            fig = go.Figure()
            fig.add_annotation(text="No columns available", x=0.5, y=0.5, xref="paper", yref="paper")
            return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
        
        # Create subplots for multiple time series
        fig = make_subplots(
            rows=len(columns), cols=1,
            subplot_titles=[f'{col} - Time Series with Anomalies' for col in columns],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        colors = ['red', 'orange', 'purple', 'green', 'brown', 'pink', 'gray', 'cyan']
        
        for i, col in enumerate(columns):
            col_data = df[col].dropna()
            
            # Add main time series
            fig.add_trace(
                go.Scatter(
                    x=col_data.index,
                    y=col_data.values,
                    mode='lines',
                    name=f'{col}',
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ),
                row=i+1, col=1
            )
            
            # Add anomalies from different methods
            for j, (method, method_results) in enumerate(anomaly_results.items()):
                if col in method_results:
                    anomaly_indices = method_results[col].get('anomaly_indices', [])
                    if anomaly_indices:
                        anomaly_values = [df.loc[idx, col] for idx in anomaly_indices if idx in df.index]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=anomaly_indices,
                                y=anomaly_values,
                                mode='markers',
                                name=f'{method}' if i == 0 else None,
                                showlegend=i == 0,
                                marker=dict(color=colors[j % len(colors)], size=6, symbol='x')
                            ),
                            row=i+1, col=1
                        )
        
        fig.update_layout(
            height=300 * len(columns),
            title_text="Advanced Multi-Column Time Series Analysis",
            title_x=0.5
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_anomaly_distribution_chart(self, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Create anomaly distribution analysis chart"""
        # Calculate anomaly percentages for each method and column
        distribution_data = []
        
        for method, method_results in anomaly_results.items():
            for col in numeric_columns:
                if col in method_results:
                    percentage = method_results[col].get('anomaly_percentage', 0)
                    count = method_results[col].get('anomaly_count', 0)
                    distribution_data.append({
                        'method': method,
                        'column': col,
                        'percentage': percentage,
                        'count': count
                    })
        
        if not distribution_data:
            fig = go.Figure()
            fig.add_annotation(text="No distribution data available", x=0.5, y=0.5, xref="paper", yref="paper")
            return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
        
        # Create violin plot showing distribution of anomaly rates
        methods = list(set(d['method'] for d in distribution_data))
        
        fig = go.Figure()
        
        for method in methods:
            method_data = [d['percentage'] for d in distribution_data if d['method'] == method]
            
            fig.add_trace(go.Violin(
                y=method_data,
                name=method,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title='Anomaly Rate Distribution by Detection Method',
            title_x=0.5,
            yaxis_title='Anomaly Percentage (%)',
            xaxis_title='Detection Methods',
            height=500
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_statistical_overview_chart(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Create statistical overview chart - Anomaly Rate by Column only"""
        # Calculate anomaly rate for each column
        stats_data = []
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Total anomalies for this column
            total_anomalies = sum(
                method_results.get(col, {}).get('anomaly_count', 0)
                for method_results in anomaly_results.values()
            )
            
            stats_data.append({
                'column': col,
                'total_anomalies': total_anomalies,
                'anomaly_rate': (total_anomalies / len(col_data)) * 100
            })
        
        if not stats_data:
            fig = go.Figure()
            fig.add_annotation(text="No statistical data available", x=0.5, y=0.5, xref="paper", yref="paper")
            return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
        
        # Create single chart - Anomaly Rate by Column
        fig = go.Figure()
        
        # Add anomaly rate bar chart
        fig.add_trace(
            go.Bar(
                x=[d['column'] for d in stats_data],
                y=[d['anomaly_rate'] for d in stats_data],
                name='Anomaly Rate (%)',
                marker_color='red',
                text=[f"{d['anomaly_rate']:.1f}%" for d in stats_data],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                             'Anomaly Rate: %{y:.2f}%<br>' +
                             'Total Anomalies: %{customdata}<br>' +
                             '<extra></extra>',
                customdata=[d['total_anomalies'] for d in stats_data]
            )
        )
        
        fig.update_layout(
            title={
                'text': "Anomaly Rate by Column",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'darkred'}
            },
            xaxis_title="Columns",
            yaxis_title="Anomaly Rate (%)",
            height=500,
            showlegend=False,
            xaxis={'tickangle': -45},
            yaxis={'tickformat': '.1f'},
            margin=dict(l=60, r=60, t=80, b=100),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add grid lines for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_advanced_box_plots(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], columns: List[str]) -> str:
        """Create advanced box plots with anomaly highlighting"""
        if not columns:
            fig = go.Figure()
            fig.add_annotation(text="No columns available", x=0.5, y=0.5, xref="paper", yref="paper")
            return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
        
        fig = go.Figure()
        
        # Add box plots for each column
        for i, col in enumerate(columns):
            col_data = df[col].dropna()
            
            # Add main box plot
            fig.add_trace(go.Box(
                y=col_data,
                name=col,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8,
                marker_color=f'rgba({50 + i*50}, {100 + i*30}, {150 + i*40}, 0.7)'
            ))
            
            # Highlight specific anomaly types
            for method, method_results in anomaly_results.items():
                if col in method_results and method in ['spikes', 'isolation_forest']:
                    anomaly_indices = method_results[col].get('anomaly_indices', [])
                    if anomaly_indices:
                        anomaly_values = [df.loc[idx, col] for idx in anomaly_indices if idx in df.index]
                        
                        # Add anomaly points with different styling
                        fig.add_trace(go.Scatter(
                            x=[col] * len(anomaly_values),
                            y=anomaly_values,
                            mode='markers',
                            name=f'{col} - {method}',
                            marker=dict(
                                size=8,
                                symbol='x' if method == 'spikes' else 'diamond',
                                color='red' if method == 'spikes' else 'orange',
                                line=dict(width=2, color='black')
                            ),
                            showlegend=i == 0
                        ))
        
        fig.update_layout(
            title='Advanced Box Plot Analysis with Anomaly Highlighting',
            title_x=0.5,
            yaxis_title='Values',
            xaxis_title='Variables',
            height=600,
            boxmode='group'
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_anomaly_summary_chart(self, anomaly_results: Dict[str, Any]) -> str:
        """Create anomaly detection summary chart"""
        methods = list(anomaly_results.keys())
        
        # Calculate total anomalies per method
        method_totals = {}
        for method in methods:
            total = sum(
                col_results.get('anomaly_count', 0)
                for col_results in anomaly_results[method].values()
            )
            method_totals[method] = total
        
        fig = px.bar(
            x=list(method_totals.keys()),
            y=list(method_totals.values()),
            title="Total Anomalies Detected by Method",
            labels={'x': 'Detection Method', 'y': 'Number of Anomalies'}
        )
        
        fig.update_layout(
            title_x=0.5,
            font=dict(size=12)
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_column_anomaly_chart(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], column: str) -> str:
        """Create anomaly detection chart for a specific column"""
        col_data = df[column].dropna()
        
        fig = go.Figure()
        
        # Add normal data points
        fig.add_trace(go.Scatter(
            x=col_data.index,
            y=col_data.values,
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=4, opacity=0.6)
        ))
        
        # Add anomalies from different methods with different colors
        colors = ['red', 'orange', 'purple', 'green']
        for i, (method, method_results) in enumerate(anomaly_results.items()):
            if column in method_results:
                anomaly_indices = method_results[column].get('anomaly_indices', [])
                if anomaly_indices:
                    anomaly_values = [df.loc[idx, column] for idx in anomaly_indices if idx in df.index]
                    
                    fig.add_trace(go.Scatter(
                        x=anomaly_indices,
                        y=anomaly_values,
                        mode='markers',
                        name=f'{method} anomalies',
                        marker=dict(color=colors[i % len(colors)], size=8, symbol='x')
                    ))
        
        fig.update_layout(
            title=f"Anomaly Detection Results for {column}",
            title_x=0.5,
            xaxis_title="Index",
            yaxis_title=column,
            font=dict(size=12)
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_box_plots_with_anomalies(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], columns: List[str]) -> str:
        """Create box plots with anomalies highlighted"""
        if not columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No columns available for box plot analysis",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Box Plot Analysis")
            return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
        
        fig = go.Figure()
        
        for col in columns:
            col_data = df[col].dropna()
            
            # Add box plot
            fig.add_trace(go.Box(
                y=col_data,
                name=col,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Box Plots with Outlier Detection",
            title_x=0.5,
            xaxis_title="Variables",
            yaxis_title="Values",
            font=dict(size=12)
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _create_time_series_anomaly_chart(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], columns: List[str]) -> str:
        """Create time series chart with anomalies highlighted"""
        if not columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No columns available for time series analysis",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Time Series Analysis")
            return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
        
        # Use first column for time series analysis
        column = columns[0]
        col_data = df[column].dropna()
        
        fig = go.Figure()
        
        # Add main time series
        fig.add_trace(go.Scatter(
            x=col_data.index,
            y=col_data.values,
            mode='lines+markers',
            name=column,
            line=dict(color='blue'),
            marker=dict(size=3)
        ))
        
        # Highlight anomalies
        for method, method_results in anomaly_results.items():
            if column in method_results:
                anomaly_indices = method_results[column].get('anomaly_indices', [])
                if anomaly_indices:
                    anomaly_values = [df.loc[idx, column] for idx in anomaly_indices if idx in df.index]
                    
                    fig.add_trace(go.Scatter(
                        x=anomaly_indices,
                        y=anomaly_values,
                        mode='markers',
                        name=f'{method} anomalies',
                        marker=dict(size=10, symbol='x')
                    ))
        
        fig.update_layout(
            title=f"Time Series Analysis with Anomalies - {column}",
            title_x=0.5,
            xaxis_title="Index/Time",
            yaxis_title=column,
            font=dict(size=12)
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _generate_dashboard_insight(self, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Generate insight for anomaly dashboard"""
        try:
            total_anomalies = sum(
                sum(col_data.get('anomaly_count', 0) for col_data in method_data.values())
                for method_data in anomaly_results.values()
            )
            
            method_effectiveness = {}
            for method, method_data in anomaly_results.items():
                method_effectiveness[method] = sum(col_data.get('anomaly_count', 0) for col_data in method_data.values())
            
            most_effective_method = max(method_effectiveness, key=method_effectiveness.get) if method_effectiveness else "None"
            
            return f"""
🔍 **Anomaly Detection Dashboard Overview**

**Detection Summary:**
• {total_anomalies} total anomalies found across {len(numeric_columns)} columns
• {most_effective_method} method detected the most anomalies ({method_effectiveness.get(most_effective_method, 0)} total)
• {len([m for m in method_effectiveness.values() if m > 0])} out of {len(method_effectiveness)} methods successfully identified outliers

**What This Shows:**
This dashboard compares how different statistical methods perform on your data. Each method has unique strengths - Z-score catches statistical outliers, IQR finds quartile-based anomalies, Isolation Forest detects complex patterns, and Modified Z-score handles skewed data robustly.

**Key Takeaways:**
• Multiple detection methods provide comprehensive coverage of different anomaly types
• Higher detection counts suggest either volatile data or sensitive method parameters
• Methods showing similar patterns likely detect the same underlying data quality issues
• Combining results from multiple methods increases confidence in anomaly identification
            """.strip()
        except Exception as e:
            return f"Error generating dashboard insight: {str(e)}"

    def _generate_heatmap_insight(self, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Generate insight for method comparison heatmap"""
        try:
            method_column_matrix = {}
            for method, method_data in anomaly_results.items():
                method_column_matrix[method] = {}
                for col in numeric_columns:
                    if col in method_data:
                        method_column_matrix[method][col] = method_data[col].get('anomaly_percentage', 0)
                    else:
                        method_column_matrix[method][col] = 0
            
            # Find most problematic columns
            column_totals = {}
            for col in numeric_columns:
                column_totals[col] = sum(method_column_matrix[method].get(col, 0) for method in method_column_matrix)
            
            most_problematic_column = max(column_totals, key=column_totals.get) if column_totals else "None"
            
            return f"""
🔥 **Method Comparison Heatmap Analysis**

**Pattern Recognition:**
• {most_problematic_column} shows the highest anomaly concentration ({column_totals.get(most_problematic_column, 0):.1f}% total rate)
• Darker red cells indicate where specific methods found high anomaly rates
• Consistent patterns across methods suggest genuine data quality issues

**Understanding the Heatmap:**
Each cell represents the anomaly percentage for a specific method-column combination. Red intensity shows severity - deeper red means more anomalies detected. When multiple methods show red for the same column, it indicates high confidence that column has genuine outliers.

**What to Look For:**
• Columns with consistent red patterns across multiple methods need immediate attention
• Methods showing similar heat patterns are detecting the same underlying issues
• Isolated red cells might indicate method-specific sensitivity rather than real problems
• Use this visualization to prioritize which columns to investigate first
            """.strip()
        except Exception as e:
            return f"Error generating heatmap insight: {str(e)}"

    def _generate_column_analysis_insight(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], column: str) -> str:
        """Generate insight for individual column analysis"""
        try:
            col_data = df[column].dropna()
            
            # Collect anomaly info for this column
            total_anomalies = sum(
                method_data.get(column, {}).get('anomaly_count', 0) 
                for method_data in anomaly_results.values()
            )
            
            methods_detecting = [
                method for method, method_data in anomaly_results.items() 
                if column in method_data and method_data[column].get('anomaly_count', 0) > 0
            ]
            
            basic_stats = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
            
            anomaly_rate = (total_anomalies / len(col_data)) * 100 if len(col_data) > 0 else 0
            
            return f"""
📊 **In-Depth Analysis: {column}**

**Statistical Profile:**
• {len(col_data)} data points ranging from {basic_stats['min']:.2f} to {basic_stats['max']:.2f}
• Average value: {basic_stats['mean']:.2f} with standard deviation of {basic_stats['std']:.2f}
• Anomaly rate: {anomaly_rate:.1f}% ({total_anomalies} outliers detected)

**Detection Consensus:**
• {len(methods_detecting)} out of {len(anomaly_results)} methods found anomalies
• Methods in agreement: {', '.join(methods_detecting) if methods_detecting else 'None'}
• {"High confidence detections" if len(methods_detecting) >= 2 else "Single method detections - verify carefully"}

**Chart Interpretation:**
The four-panel view shows: (1) Time series with anomaly markers, (2) Distribution histogram, (3) Method comparison bars, and (4) Cumulative anomaly timeline. Red markers indicate outliers, while different colors distinguish between detection methods.

**Data Quality Assessment:**
{"This column shows concerning anomaly levels that warrant investigation" if anomaly_rate > 5 else "Anomaly levels are within acceptable ranges for most applications"}
            """.strip()
        except Exception as e:
            return f"Error generating column analysis insight: {str(e)}"

    def _generate_time_series_insight(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], columns: List[str]) -> str:
        """Generate insight for time series analysis"""
        try:
            if not columns:
                return "No columns available for time series analysis"
            
            # Analyze temporal patterns
            total_anomalies = 0
            temporal_distribution = {}
            
            for col in columns:
                for method_data in anomaly_results.values():
                    if col in method_data:
                        anomaly_indices = method_data[col].get('anomaly_indices', [])
                        total_anomalies += len(anomaly_indices)
                        
                        # Group by time periods (assuming index represents time)
                        for idx in anomaly_indices:
                            time_period = idx // 100  # Group by hundreds for pattern detection
                            if time_period not in temporal_distribution:
                                temporal_distribution[time_period] = 0
                            temporal_distribution[time_period] += 1
            
            peak_period = max(temporal_distribution, key=temporal_distribution.get) if temporal_distribution else "None"
            
            return f"""
📈 **Time Series Anomaly Pattern Analysis**

**Temporal Overview:**
• {total_anomalies} total anomalies detected across {len(columns)} variables
• Anomalies distributed across {len(temporal_distribution)} distinct time periods
• Peak activity period: {peak_period} (highest anomaly concentration)

**Pattern Recognition:**
The multi-layered time series reveals when outliers occur across different variables. Each line represents a variable, with colored markers showing anomalies detected by different methods. Overlapping markers indicate high-confidence anomalies found by multiple detection approaches.

**Temporal Insights:**
• Clustered anomalies suggest systematic events or data collection issues
• Isolated anomalies may represent random occurrences or measurement errors
• Cross-variable anomalies at the same time points indicate system-wide disturbances
• Method-specific patterns help understand different types of temporal outliers

**Understanding the Visualization:**
Blue lines show normal data flow, while colored markers (red, orange, purple, green) represent different detection methods. When multiple colored markers appear at the same time point, it indicates high confidence that a genuine anomaly occurred.
            """.strip()
        except Exception as e:
            return f"Error generating time series insight: {str(e)}"

    def _generate_distribution_insight(self, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Generate insight for anomaly distribution analysis"""
        try:
            method_stats = {}
            for method, method_data in anomaly_results.items():
                percentages = [col_data.get('anomaly_percentage', 0) for col_data in method_data.values()]
                method_stats[method] = {
                    'mean': np.mean(percentages),
                    'std': np.std(percentages),
                    'max': np.max(percentages)
                }
            
            most_variable_method = max(method_stats, key=lambda x: method_stats[x]['std']) if method_stats else "None"
            
            return f"""
🎯 **Anomaly Distribution Pattern Analysis**

**Method Variability:**
• {most_variable_method} shows the highest variance in detection rates across columns
• {len(method_stats)} detection methods compared across {len(numeric_columns)} columns
• Distribution shapes reveal method consistency and reliability patterns

**Understanding the Distributions:**
Violin plots show the spread of anomaly rates across columns for each method. Wide shapes indicate the method finds varying anomaly rates across different columns, while narrow shapes suggest consistent behavior regardless of column characteristics.

**Method Characteristics:**
• Wide distributions: Method is sensitive to column-specific data patterns
• Narrow distributions: Method applies consistent detection criteria across all columns
• Outliers in distributions: Specific columns where the method behaves unusually
• Median positions: Central tendency of each method's detection rates

**Interpretation Guide:**
Use distribution width to understand method reliability. Consistent methods (narrow distributions) are good for standardized monitoring, while variable methods (wide distributions) might be better for detecting column-specific anomaly patterns.
            """.strip()
        except Exception as e:
            return f"Error generating distribution insight: {str(e)}"

    def _generate_statistical_overview_insight(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], numeric_columns: List[str]) -> str:
        """Generate insight for statistical overview - focused on anomaly rates by column"""
        try:
            # Calculate anomaly rates for each column
            anomaly_data = {}
            
            for col in numeric_columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    total_anomalies = sum(
                        method_data.get(col, {}).get('anomaly_count', 0)
                        for method_data in anomaly_results.values()
                    )
                    
                    anomaly_data[col] = {
                        'total_data_points': len(col_data),
                        'anomaly_count': total_anomalies,
                        'anomaly_rate': (total_anomalies / len(col_data)) * 100
                    }
            
            # Analyze anomaly rate patterns
            if not anomaly_data:
                return "No data available for anomaly rate analysis"
            
            # Find columns with different anomaly rate ranges
            high_anomaly_columns = [col for col, data in anomaly_data.items() if data['anomaly_rate'] > 10]
            medium_anomaly_columns = [col for col, data in anomaly_data.items() if 3 <= data['anomaly_rate'] <= 10]
            low_anomaly_columns = [col for col, data in anomaly_data.items() if data['anomaly_rate'] < 3]
            
            # Calculate summary statistics
            all_rates = [data['anomaly_rate'] for data in anomaly_data.values()]
            avg_anomaly_rate = sum(all_rates) / len(all_rates)
            max_anomaly_rate = max(all_rates)
            min_anomaly_rate = min(all_rates)
            
            # Find the column with highest and lowest anomaly rates
            highest_col = max(anomaly_data.keys(), key=lambda k: anomaly_data[k]['anomaly_rate'])
            lowest_col = min(anomaly_data.keys(), key=lambda k: anomaly_data[k]['anomaly_rate'])
            
            insight = f"""📊 **Anomaly Rate Analysis by Column**

**Overview:**
• {len(numeric_columns)} columns analyzed for anomaly detection patterns
• Average anomaly rate across all columns: {avg_anomaly_rate:.2f}%
• Highest anomaly rate: {max_anomaly_rate:.2f}% (Column: {highest_col})
• Lowest anomaly rate: {min_anomaly_rate:.2f}% (Column: {lowest_col})

**Anomaly Rate Distribution:**
• High anomaly rate columns (>10%): {len(high_anomaly_columns)} columns
• Medium anomaly rate columns (3-10%): {len(medium_anomaly_columns)} columns  
• Low anomaly rate columns (<3%): {len(low_anomaly_columns)} columns

**Key Findings:**
"""
            
            if high_anomaly_columns:
                insight += f"⚠️ **High Priority:** {', '.join(high_anomaly_columns[:3])} show elevated anomaly rates requiring immediate attention\n"
            
            if medium_anomaly_columns:
                insight += f"📋 **Monitor:** {', '.join(medium_anomaly_columns[:3])} have moderate anomaly rates worth monitoring\n"
            
            if low_anomaly_columns:
                insight += f"✅ **Good Quality:** {', '.join(low_anomaly_columns[:3])} demonstrate low anomaly rates indicating good data quality\n"
            
            # Add recommendations based on patterns
            insight += "\n**Recommendations:**\n"
            if len(high_anomaly_columns) > 0:
                insight += "• Investigate data collection processes for high anomaly columns\n"
                insight += "• Consider data cleaning or preprocessing for problematic columns\n"
            
            if max_anomaly_rate > 15:
                insight += "• Extremely high anomaly rates detected - urgent data quality review needed\n"
            
            if avg_anomaly_rate < 2:
                insight += "• Overall low anomaly rates indicate good data quality across the dataset\n"
            elif avg_anomaly_rate > 8:
                insight += "• High average anomaly rate suggests systematic data quality issues\n"
            else:
                insight += "• Moderate anomaly rates are within acceptable ranges for most use cases\n"
            
            return insight
            
        except Exception as e:
            return f"Error generating statistical overview insight: {str(e)}"

    def _generate_box_plots_insight(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], columns: List[str]) -> str:
        """Generate insight for box plots with anomalies"""
        try:
            if not columns:
                return "No columns available for box plot analysis"
            
            # Analyze box plot characteristics
            box_stats = {}
            for col in columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Count natural outliers (beyond 1.5*IQR)
                    natural_outliers = len(col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)])
                    
                    # Count method-detected anomalies
                    method_anomalies = sum(
                        method_data.get(col, {}).get('anomaly_count', 0)
                        for method_data in anomaly_results.values()
                    )
                    
                    box_stats[col] = {
                        'natural_outliers': natural_outliers,
                        'method_anomalies': method_anomalies,
                        'iqr': IQR,
                        'median': col_data.median()
                    }
            
            total_natural_outliers = sum(stats['natural_outliers'] for stats in box_stats.values())
            total_method_anomalies = sum(stats['method_anomalies'] for stats in box_stats.values())
            
            overlap_assessment = abs(total_natural_outliers - total_method_anomalies) < max(total_natural_outliers, total_method_anomalies) * 0.3
            
            insight = f"""📦 **Box Plot Analysis with Anomaly Detection**

**Outlier Comparison:**
• Traditional box plot outliers: {total_natural_outliers}
• Method-detected anomalies: {total_method_anomalies}
• Detection alignment: {"High overlap - methods agree with statistical outliers" if overlap_assessment else "Different outlier types detected by different approaches"}

**Understanding Box Plots:**
Box plots show data distribution through quartiles. The box spans from 25th to 75th percentile, with the median line inside. Whiskers extend to 1.5× the interquartile range (IQR). Points beyond whiskers are traditional statistical outliers.

**Method-Specific Markers:**
• X markers: Different detection methods with unique colors
• Diamond markers: Isolation Forest detections (complex pattern-based)
• Circle markers: Traditional statistical methods
• Overlapping markers: High-confidence anomalies detected by multiple methods

**Key Insights:**
Box plots provide context for anomaly detection by showing natural data boundaries. When method-detected anomalies align with box plot outliers, it validates the detection. Discrepancies might indicate the methods are finding different types of unusual patterns.

**Validation Assessment:**
{"Method detections align well with traditional outlier analysis" if overlap_assessment else "Methods detect different anomaly types - investigate discrepancies to understand what each approach captures"}"""
            
            return insight
        except Exception as e:
            return f"Error generating box plots insight: {str(e)}"
    
    def _create_column_time_series_chart(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], column: str) -> str:
        """Create time series chart with anomalies for a specific column"""
        col_data = df[column].dropna()
        
        fig = go.Figure()
        
        # Add main time series line
        fig.add_trace(
            go.Scatter(
                x=col_data.index,
                y=col_data.values,
                mode='lines',
                name=f'{column} Data',
                line=dict(color='blue', width=2),
                opacity=0.7
            )
        )
        
        # Add anomalies from different methods
        colors = ['red', 'orange', 'purple', 'green', 'brown', 'pink', 'gray', 'cyan']
        method_names = {'zscore': 'Z-Score', 'iqr': 'IQR', 'isolation_forest': 'Isolation Forest', 'modified_zscore': 'Modified Z-Score'}
        
        for i, (method, method_results) in enumerate(anomaly_results.items()):
            if column in method_results:
                anomaly_indices = method_results[column].get('anomaly_indices', [])
                if anomaly_indices:
                    anomaly_values = [df.loc[idx, column] for idx in anomaly_indices if idx in df.index]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_indices,
                            y=anomaly_values,
                            mode='markers',
                            name=method_names.get(method, method),
                            marker=dict(color=colors[i % len(colors)], size=8, symbol='x'),
                            legendgroup=method
                        )
                    )
        
        fig.update_layout(
            title=f'{column} - Time Series with Anomaly Detection',
            xaxis_title='Index',
            yaxis_title=column,
            height=400,
            width=600,
            hovermode='x unified',
            legend=dict(x=1, y=1),
            margin=dict(l=60, r=100, t=60, b=60),
            autosize=True
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
    
    def _create_column_method_comparison_chart(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], column: str) -> str:
        """Create method comparison chart for a specific column"""
        method_counts = []
        method_names = []
        method_percentages = []
        
        method_display_names = {'zscore': 'Z-Score', 'iqr': 'IQR', 'isolation_forest': 'Isolation Forest', 'modified_zscore': 'Modified Z-Score'}
        
        for method, method_results in anomaly_results.items():
            if column in method_results:
                count = method_results[column].get('anomaly_count', 0)
                percentage = method_results[column].get('anomaly_percentage', 0)
                method_counts.append(count)
                method_percentages.append(percentage)
                method_names.append(method_display_names.get(method, method))
        
        # Create subplot with two charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'{column} - Anomaly Count by Method', f'{column} - Anomaly Percentage by Method'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Anomaly counts
        fig.add_trace(
            go.Bar(
                x=method_names,
                y=method_counts,
                name='Anomaly Count',
                marker_color='darkred',
                text=method_counts,
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Anomaly percentages
        fig.add_trace(
            go.Bar(
                x=method_names,
                y=method_percentages,
                name='Anomaly %',
                marker_color='darkorange',
                text=[f'{p:.2f}%' for p in method_percentages],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            width=600,
            showlegend=False,
            title=f'{column} - Detection Method Comparison',
            margin=dict(l=60, r=60, t=60, b=60),
            autosize=True
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
    
    def _generate_column_method_comparison_insight(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], column: str) -> str:
        """Generate insight for column method comparison"""
        method_data = {}
        
        for method, method_results in anomaly_results.items():
            if column in method_results:
                method_data[method] = {
                    'count': method_results[column].get('anomaly_count', 0),
                    'percentage': method_results[column].get('anomaly_percentage', 0)
                }
        
        if not method_data:
            return f"No anomaly detection data available for column '{column}'."
        
        # Find the method with highest detection rate
        max_method = max(method_data.items(), key=lambda x: x[1]['percentage'])
        min_method = min(method_data.items(), key=lambda x: x[1]['percentage'])
        
        total_unique_anomalies = len(set().union(*[
            method_results[column].get('anomaly_indices', [])
            for method, method_results in anomaly_results.items()
            if column in method_results
        ]))
        
        insight = f"""**Method Comparison Analysis for {column}:**

**Detection Summary:**
• Most Sensitive Method: {max_method[0].replace('_', ' ').title()} ({max_method[1]['percentage']:.2f}%, {max_method[1]['count']} anomalies)
• Least Sensitive Method: {min_method[0].replace('_', ' ').title()} ({min_method[1]['percentage']:.2f}%, {min_method[1]['count']} anomalies)
• Total Unique Anomalies: {total_unique_anomalies} across all methods

**Method Performance:**"""
        
        for method, data in sorted(method_data.items(), key=lambda x: x[1]['percentage'], reverse=True):
            method_name = method.replace('_', ' ').title()
            insight += f"\n• {method_name}: {data['count']} anomalies ({data['percentage']:.2f}%)"
        
        # Add interpretation
        variance = max(method_data.values(), key=lambda x: x['percentage'])['percentage'] - min(method_data.values(), key=lambda x: x['percentage'])['percentage']
        
        if variance > 5:
            insight += f"\n\n**High Method Variance:** {variance:.2f}% difference between methods suggests the column has complex anomaly patterns that different methods detect differently."
        elif variance > 2:
            insight += f"\n\n**Moderate Method Variance:** {variance:.2f}% difference indicates some variation in method sensitivity."
        else:
            insight += f"\n\n**Low Method Variance:** {variance:.2f}% difference shows consistent detection across methods."
        
        return insight

    def _create_column_box_plot_chart(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], column: str) -> str:
        """Create scatter plot with blue points and red outliers - EDA panel style"""
        col_data = df[column].dropna()
        
        # Get all unique anomaly indices from all methods
        all_anomaly_indices = set()
        for method, method_results in anomaly_results.items():
            if column in method_results:
                anomaly_indices = method_results[column].get('anomaly_indices', [])
                all_anomaly_indices.update(anomaly_indices)
        
        # Create figure
        fig = go.Figure()
        
        # Create index array for x-axis (sequential indices from 0 to len-1)
        data_indices = list(range(len(col_data)))
        values = col_data.values
        original_indices = col_data.index.tolist()
        
        # Separate normal and anomaly points
        normal_x = []
        normal_y = []
        anomaly_x = []
        anomaly_y = []
        
        for i, (orig_idx, val) in enumerate(zip(original_indices, values)):
            if orig_idx in all_anomaly_indices:
                anomaly_x.append(i)
                anomaly_y.append(val)
            else:
                normal_x.append(i)
                normal_y.append(val)
        
        # Add normal data points (blue dots)
        fig.add_trace(
            go.Scatter(
                x=normal_x,
                y=normal_y,
                mode='markers',
                name='Normal Data',
                marker=dict(
                    color='#1f77b4',  # Blue like in EDA panel
                    size=4,
                    opacity=0.6,
                    symbol='circle'
                ),
                hovertemplate='<b>Normal Data</b><br>' +
                            'Index: %{x}<br>' +
                            f'{column}: %{{y:.3f}}<br>' +
                            '<extra></extra>'
            )
        )
        
        # Add anomaly data points (red dots)
        if anomaly_x:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_x,
                    y=anomaly_y,
                    mode='markers',
                    name=f'Outliers ({len(anomaly_x)})',
                    marker=dict(
                        color='#d62728',  # Red like in EDA panel
                        size=6,
                        opacity=0.8,
                        symbol='circle'
                    ),
                    hovertemplate='<b>Outlier</b><br>' +
                                'Index: %{x}<br>' +
                                f'{column}: %{{y:.3f}}<br>' +
                                '<extra></extra>'
                )
            )
        
        # Configure layout to match EDA panel style
        fig.update_layout(
            title=dict(
                text=f'{column} - Data Points with Anomaly Detection',
                x=0.5,
                font=dict(size=14, color='#2C3E50')
            ),
            xaxis=dict(
                title='Data Point Index',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                gridwidth=1,
                zeroline=False,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title=f'{column} Values',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='rgba(128, 128, 128, 0.3)',
                tickfont=dict(size=10)
            ),
            height=400,
            width=600,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(128, 128, 128, 0.3)',
                borderwidth=1,
                font=dict(size=10)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=100, t=60, b=60),
            hovermode='closest',
            autosize=True
        )
        
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

    def _generate_column_box_plot_insight(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], column: str) -> str:
        """Generate insight for column anomaly detection visualization"""
        col_data = df[column].dropna()
        
        # Calculate basic statistics
        q1 = col_data.quantile(0.25)
        median = col_data.median()
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        mean_val = col_data.mean()
        std_val = col_data.std()
        
        # Compare anomaly detection methods
        method_summary = {}
        total_unique_anomalies = set()
        
        for method, method_results in anomaly_results.items():
            if column in method_results:
                anomaly_indices = method_results[column].get('anomaly_indices', [])
                count = len(anomaly_indices)
                method_summary[method] = {
                    'count': count,
                    'percentage': (count / len(col_data)) * 100,
                    'indices': set(anomaly_indices)
                }
                total_unique_anomalies.update(anomaly_indices)
        
        # Calculate method agreement
        if len(method_summary) > 1:
            # Find overlapping anomalies between methods
            all_indices = [info['indices'] for info in method_summary.values()]
            intersection = set.intersection(*all_indices) if all_indices else set()
            agreement_rate = (len(intersection) / len(total_unique_anomalies)) * 100 if total_unique_anomalies else 0
        else:
            agreement_rate = 100 if method_summary else 0
        
        # Analyze data distribution characteristics
        skewness = col_data.skew()
        kurtosis = col_data.kurtosis()
        
        insight = f"""**Anomaly Detection Analysis for {column}:**

**📊 Data Distribution Overview:**
• Total Data Points: {len(col_data):,}
• Mean: {mean_val:.3f} | Median: {median:.3f}
• Standard Deviation: {std_val:.3f}
• Range: {col_data.min():.3f} to {col_data.max():.3f}
• IQR: {iqr:.3f} (Q1: {q1:.3f}, Q3: {q3:.3f})

**🔍 Anomaly Detection Results:**
• Total Unique Anomalies Found: {len(total_unique_anomalies)}
• Overall Anomaly Rate: {(len(total_unique_anomalies)/len(col_data)*100):.2f}%
• Method Agreement: {agreement_rate:.1f}%

**📈 Method Breakdown:**"""
        
        # Sort methods by detection count for better presentation
        sorted_methods = sorted(method_summary.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for method, info in sorted_methods:
            method_name = method.replace('_', ' ').title()
            insight += f"\n• **{method_name}:** {info['count']} anomalies ({info['percentage']:.2f}%)"
        
        # Add distribution analysis
        insight += f"\n\n**📐 Distribution Characteristics:**"
        
        if abs(skewness) < 0.5:
            skew_desc = "approximately symmetric"
        elif skewness > 0.5:
            skew_desc = "right-skewed (tail extends right)"
        else:
            skew_desc = "left-skewed (tail extends left)"
            
        insight += f"\n• Skewness: {skewness:.3f} - {skew_desc}"
        
        if kurtosis > 3:
            kurt_desc = "heavy-tailed (more outliers than normal)"
        elif kurtosis < 3:
            kurt_desc = "light-tailed (fewer outliers than normal)"
        else:
            kurt_desc = "normal-tailed"
            
        insight += f"\n• Kurtosis: {kurtosis:.3f} - {kurt_desc}"
        
        # Add interpretation and recommendations
        insight += f"\n\n**💡 Key Insights:**"
        
        if len(total_unique_anomalies) == 0:
            insight += f"\n• No anomalies detected by any method - data appears clean"
        elif agreement_rate > 75:
            insight += f"\n• High method consensus ({agreement_rate:.1f}%) indicates reliable anomaly detection"
        elif agreement_rate > 50:
            insight += f"\n• Moderate method agreement ({agreement_rate:.1f}%) - review conflicting detections"
        else:
            insight += f"\n• Low method agreement ({agreement_rate:.1f}%) - methods detect different patterns"
        
        # Anomaly rate interpretation
        anomaly_rate = (len(total_unique_anomalies)/len(col_data)*100)
        if anomaly_rate > 10:
            insight += f"\n• High anomaly rate ({anomaly_rate:.1f}%) may indicate data quality issues"
        elif anomaly_rate > 5:
            insight += f"\n• Moderate anomaly rate ({anomaly_rate:.1f}%) - investigate unusual patterns"
        elif anomaly_rate > 0:
            insight += f"\n• Low anomaly rate ({anomaly_rate:.1f}%) suggests mostly clean data"
        
        # Distribution-based recommendations
        if abs(skewness) > 1:
            insight += f"\n• High skewness may affect some detection methods - consider data transformation"
        
        if len(method_summary) > 1:
            most_sensitive = max(sorted_methods, key=lambda x: x[1]['count'])
            least_sensitive = min(sorted_methods, key=lambda x: x[1]['count'])
            insight += f"\n• Most sensitive method: {most_sensitive[0].replace('_', ' ').title()} ({most_sensitive[1]['count']} detections)"
            insight += f"\n• Least sensitive method: {least_sensitive[0].replace('_', ' ').title()} ({least_sensitive[1]['count']} detections)"
        
        return insight
