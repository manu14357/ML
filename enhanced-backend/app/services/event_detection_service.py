# -*- coding: utf-8 -*-
"""
Event Detection Service for Time Series Data
Advanced detection of spikes, drifts, gaps, and flatlines
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
from scipy import stats
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class EventDetectionService:
    """Advanced Event Detection Service for Time Series Data with Plotly visualizations"""

    def __init__(self):
        self.results = {}

    def detect_events(self, df: pd.DataFrame, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Advanced event detection in time series data using comprehensive methods.
        
        Parameters:
        - df: pandas DataFrame to analyze
        - parameters: dictionary containing detection parameters
        """
        start_time = time.time()
        
        if parameters is None:
            parameters = {}
            
        method = parameters.get('method', 'all')
        feature_columns = parameters.get('feature_columns', None)
        spike_threshold = parameters.get('spike_threshold', 3.0)
        drift_window = parameters.get('drift_window', 50)
        flatline_threshold = parameters.get('flatline_threshold', 0.001)
        gap_threshold = parameters.get('gap_threshold', 10)
        
        logger.info(f"Starting advanced event detection with method: {method}")
        
        results = {}

        try:
            # Select feature columns
            if feature_columns is None or (isinstance(feature_columns, list) and len(feature_columns) == 0):
                # Auto-detect all numeric columns if feature_columns is None or empty list
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                # Use specified columns, filtering for valid numeric columns
                numeric_columns = [col for col in feature_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_columns:
                return {
                    'success': False,
                    'error': 'No numeric columns found for event detection'
                }
            
            # Dataset information
            dataset_info = {
                'total_rows': len(df),
                'numeric_columns': len(numeric_columns),
                'columns_analyzed': numeric_columns,
                'analysis_methods': ['spikes', 'drifts', 'gaps', 'flatlines']
            }
            results['dataset_info'] = dataset_info
            
            # Event detection results for each method
            event_results = {}
            
            # Advanced event detection methods
            if method in ['all', 'spikes']:
                event_results['spikes'] = self._detect_spike_events(df[numeric_columns], spike_threshold)
            
            if method in ['all', 'drifts']:
                event_results['drifts'] = self._detect_drift_events(df[numeric_columns], drift_window)
            
            if method in ['all', 'gaps']:
                event_results['gaps'] = self._detect_gap_events(df[numeric_columns], gap_threshold)
            
            if method in ['all', 'flatlines']:
                event_results['flatlines'] = self._detect_flatline_events(df[numeric_columns], flatline_threshold)
            
            results['event_results'] = event_results
            
            # Generate comprehensive summary report
            comprehensive_summary = self._generate_comprehensive_summary(df, event_results, numeric_columns)
            results['comprehensive_summary'] = comprehensive_summary
            
            # Combine results and create analysis
            combined_results = self._combine_event_results(df, event_results, numeric_columns)
            results['combined_analysis'] = combined_results
            
            # Generate advanced visualizations
            charts = self._generate_advanced_event_charts(df, event_results, numeric_columns, parameters)
            results['charts'] = charts
            
            # Statistical summary
            stats_summary = self._generate_statistical_summary(df, event_results, numeric_columns)
            results['statistical_summary'] = stats_summary
            
            # Advanced recommendations
            recommendations = self._generate_advanced_recommendations(event_results, combined_results, comprehensive_summary)
            results['recommendations'] = recommendations
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            results['analysis_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Advanced event detection completed in {execution_time:.2f} seconds")
            
            return {
                'success': True,
                'results': results,
                'metadata': {
                    'analysis_type': 'advanced_event_detection',
                    'execution_time': execution_time,
                    'parameters_used': parameters,
                    'total_events_detected': sum(
                        sum(col_data.get('event_count', 0) for col_data in method_data.values())
                        for method_data in event_results.values()
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced event detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _detect_spike_events(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect spike events using advanced statistical methods"""
        results = {}
        
        for col in df.columns:
            non_null_data = df[col].dropna()
            if len(non_null_data) < 10:
                continue
            
            # Calculate rolling statistics for spike detection
            window_size = max(5, len(non_null_data) // 20)
            rolling_mean = non_null_data.rolling(window=window_size, center=True).mean()
            rolling_std = non_null_data.rolling(window=window_size, center=True).std()
            
            # Detect spikes using adaptive threshold
            upper_bound = rolling_mean + threshold * rolling_std
            lower_bound = rolling_mean - threshold * rolling_std
            
            spike_indices = non_null_data.index[
                (non_null_data > upper_bound) | (non_null_data < lower_bound)
            ]
            
            # Calculate spike magnitudes
            spike_magnitudes = []
            for idx in spike_indices:
                if idx in rolling_mean.index and idx in rolling_std.index:
                    magnitude = abs(non_null_data[idx] - rolling_mean[idx]) / rolling_std[idx]
                    spike_magnitudes.append(float(magnitude))
                else:
                    spike_magnitudes.append(float(threshold))
            
            results[col] = {
                'method': 'spike_detection',
                'threshold': threshold,
                'window_size': window_size,
                'event_count': len(spike_indices),
                'event_percentage': (len(spike_indices) / len(non_null_data)) * 100,
                'event_indices': spike_indices.tolist(),
                'event_values': non_null_data[spike_indices].tolist(),
                'spike_magnitudes': spike_magnitudes,
                'max_spike_magnitude': float(max(spike_magnitudes)) if spike_magnitudes else 0,
                'avg_spike_magnitude': float(np.mean(spike_magnitudes)) if spike_magnitudes else 0
            }
        
        return results

    def _detect_drift_events(self, df: pd.DataFrame, window: int = 50) -> Dict[str, Any]:
        """Detect drift events using rolling window analysis"""
        results = {}
        
        for col in df.columns:
            non_null_data = df[col].dropna()
            if len(non_null_data) < window * 2:
                continue
            
            # Calculate rolling statistics
            rolling_mean = non_null_data.rolling(window=window).mean()
            rolling_std = non_null_data.rolling(window=window).std()
            
            # Detect significant changes in mean (drift)
            mean_changes = rolling_mean.diff().abs()
            std_threshold = rolling_std.rolling(window=window//2).mean() * 2
            
            # Find drift points where mean change exceeds threshold
            drift_indices = mean_changes.index[mean_changes > std_threshold]
            
            # Calculate drift magnitudes
            drift_magnitudes = []
            drift_directions = []
            for idx in drift_indices:
                if idx in mean_changes.index and idx in std_threshold.index:
                    magnitude = mean_changes[idx] / std_threshold[idx]
                    drift_magnitudes.append(float(magnitude))
                    
                    # Determine drift direction
                    if idx in rolling_mean.index:
                        prev_idx = rolling_mean.index[rolling_mean.index.get_loc(idx) - 1] if rolling_mean.index.get_loc(idx) > 0 else idx
                        direction = 'upward' if rolling_mean[idx] > rolling_mean[prev_idx] else 'downward'
                        drift_directions.append(direction)
                    else:
                        drift_directions.append('unknown')
            
            results[col] = {
                'method': 'drift_detection',
                'window_size': window,
                'event_count': len(drift_indices),
                'event_percentage': (len(drift_indices) / len(non_null_data)) * 100,
                'event_indices': drift_indices.tolist(),
                'event_values': non_null_data[drift_indices].tolist(),
                'drift_magnitudes': drift_magnitudes,
                'drift_directions': drift_directions,
                'max_drift_magnitude': float(max(drift_magnitudes)) if drift_magnitudes else 0,
                'upward_drifts': drift_directions.count('upward'),
                'downward_drifts': drift_directions.count('downward')
            }
        
        return results

    def _detect_gap_events(self, df: pd.DataFrame, threshold: float = 10) -> Dict[str, Any]:
        """Detect gap events (missing value patterns)"""
        results = {}
        
        for col in df.columns:
            series = df[col]
            
            # Find missing value groups
            missing_mask = series.isnull()
            missing_groups = []
            
            if missing_mask.any():
                # Find consecutive missing value sequences
                missing_indices = missing_mask[missing_mask].index
                
                if len(missing_indices) > 0:
                    groups = []
                    current_group = [missing_indices[0]]
                    
                    for i in range(1, len(missing_indices)):
                        # Handle different index types (integers vs timestamps)
                        try:
                            pos_diff = series.index.get_loc(missing_indices[i]) - series.index.get_loc(missing_indices[i-1])
                        except (KeyError, ValueError):
                            # Fallback for integer indices or simple numeric difference
                            try:
                                pos_diff = int(missing_indices[i] - missing_indices[i-1])
                            except (TypeError, ValueError):
                                pos_diff = 1  # Default to consecutive if can't calculate
                        
                        if pos_diff == 1:
                            current_group.append(missing_indices[i])
                        else:
                            if len(current_group) >= threshold:
                                groups.append(current_group)
                            current_group = [missing_indices[i]]
                    
                    # Add last group if it meets threshold
                    if len(current_group) >= threshold:
                        groups.append(current_group)
                    
                    missing_groups = groups
            
            # Calculate gap statistics
            gap_lengths = [len(group) for group in missing_groups]
            total_gap_points = sum(gap_lengths)
            
            results[col] = {
                'method': 'gap_detection',
                'threshold': threshold,
                'event_count': len(missing_groups),
                'event_percentage': (total_gap_points / len(series)) * 100,
                'gap_groups': missing_groups,
                'gap_lengths': gap_lengths,
                'total_gap_points': total_gap_points,
                'max_gap_length': max(gap_lengths) if gap_lengths else 0,
                'avg_gap_length': float(np.mean(gap_lengths)) if gap_lengths else 0
            }
        
        return results

    def _detect_flatline_events(self, df: pd.DataFrame, threshold: float = 0.001) -> Dict[str, Any]:
        """Detect flatline events (constant value sequences)"""
        results = {}
        
        for col in df.columns:
            non_null_data = df[col].dropna()
            if len(non_null_data) < 10:
                continue
            
            # Calculate rolling variance to detect flatlines
            window_size = max(5, len(non_null_data) // 50)
            rolling_var = non_null_data.rolling(window=window_size).var()
            
            # Find flatline regions where variance is below threshold
            flatline_indices = rolling_var.index[rolling_var < threshold]
            
            # Group consecutive flatline points
            flatline_groups = []
            if len(flatline_indices) > 0:
                current_group = [flatline_indices[0]]
                
                for i in range(1, len(flatline_indices)):
                    # Get the position difference in the index
                    try:
                        pos_diff = rolling_var.index.get_loc(flatline_indices[i]) - rolling_var.index.get_loc(flatline_indices[i-1])
                    except (KeyError, ValueError):
                        # Fallback to simple numeric difference if indices are integers
                        pos_diff = flatline_indices[i] - flatline_indices[i-1]
                    
                    if pos_diff <= window_size:
                        current_group.append(flatline_indices[i])
                    else:
                        if len(current_group) >= window_size:
                            flatline_groups.append(current_group)
                        current_group = [flatline_indices[i]]
                
                if len(current_group) >= window_size:
                    flatline_groups.append(current_group)
            
            # Calculate flatline statistics
            flatline_lengths = [len(group) for group in flatline_groups]
            total_flatline_points = sum(flatline_lengths)
            
            results[col] = {
                'method': 'flatline_detection',
                'threshold': threshold,
                'window_size': window_size,
                'event_count': len(flatline_groups),
                'event_percentage': (total_flatline_points / len(non_null_data)) * 100,
                'flatline_groups': flatline_groups,
                'flatline_lengths': flatline_lengths,
                'total_flatline_points': total_flatline_points,
                'max_flatline_length': max(flatline_lengths) if flatline_lengths else 0,
                'avg_flatline_length': float(np.mean(flatline_lengths)) if flatline_lengths else 0
            }
        
        return results

    def _generate_comprehensive_summary(self, df: pd.DataFrame, event_results: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Generate comprehensive summary of event detection results"""
        summary = {
            'dataset_overview': {
                'total_rows': len(df),
                'total_columns': len(columns),
                'analysis_methods': list(event_results.keys())
            },
            'event_overview': {},
            'severity_analysis': {},
            'column_analysis': {}
        }
        
        # Event overview
        total_events = 0
        for method, method_results in event_results.items():
            method_events = sum(col_data.get('event_count', 0) for col_data in method_results.values())
            summary['event_overview'][method] = {
                'total_events': method_events,
                'affected_columns': len([col for col in method_results.keys() if method_results[col].get('event_count', 0) > 0]),
                'avg_events_per_column': method_events / len(columns) if columns else 0
            }
            total_events += method_events
        
        summary['event_overview']['total_events_all_methods'] = total_events
        
        # Severity analysis
        severity_levels = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for method, method_results in event_results.items():
            for col, col_data in method_results.items():
                event_percentage = col_data.get('event_percentage', 0)
                if event_percentage < 1:
                    severity_levels['low'] += 1
                elif event_percentage < 5:
                    severity_levels['medium'] += 1
                elif event_percentage < 15:
                    severity_levels['high'] += 1
                else:
                    severity_levels['critical'] += 1
        
        summary['severity_analysis'] = severity_levels
        
        # Column analysis
        for col in columns:
            col_summary = {
                'total_events': 0,
                'methods_with_events': [],
                'most_problematic_method': None,
                'overall_severity': 'low'
            }
            
            max_percentage = 0
            max_method = None
            
            for method, method_results in event_results.items():
                if col in method_results:
                    col_data = method_results[col]
                    event_count = col_data.get('event_count', 0)
                    event_percentage = col_data.get('event_percentage', 0)
                    
                    if event_count > 0:
                        col_summary['total_events'] += event_count
                        col_summary['methods_with_events'].append(method)
                        
                        if event_percentage > max_percentage:
                            max_percentage = event_percentage
                            max_method = method
            
            col_summary['most_problematic_method'] = max_method
            
            # Determine overall severity
            if max_percentage < 1:
                col_summary['overall_severity'] = 'low'
            elif max_percentage < 5:
                col_summary['overall_severity'] = 'medium'
            elif max_percentage < 15:
                col_summary['overall_severity'] = 'high'
            else:
                col_summary['overall_severity'] = 'critical'
            
            summary['column_analysis'][col] = col_summary
        
        return summary

    def _combine_event_results(self, df: pd.DataFrame, event_results: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Combine event detection results across all methods"""
        combined = {
            'combined_events_by_column': {},
            'method_comparison': {},
            'temporal_analysis': {},
            'correlation_analysis': {}
        }
        
        # Combine events by column
        for col in columns:
            col_events = {
                'all_event_indices': set(),
                'method_breakdown': {},
                'total_events': 0,
                'unique_events': 0,
                'overlap_analysis': {}
            }
            
            for method, method_results in event_results.items():
                if col in method_results:
                    method_data = method_results[col]
                    event_indices = set(method_data.get('event_indices', []))
                    event_count = method_data.get('event_count', 0)
                    
                    col_events['method_breakdown'][method] = {
                        'event_count': event_count,
                        'event_indices': list(event_indices)
                    }
                    col_events['all_event_indices'].update(event_indices)
                    col_events['total_events'] += event_count
            
            col_events['unique_events'] = len(col_events['all_event_indices'])
            col_events['all_event_indices'] = list(col_events['all_event_indices'])
            
            # Calculate overlap between methods
            methods = list(col_events['method_breakdown'].keys())
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    indices1 = set(col_events['method_breakdown'][method1]['event_indices'])
                    indices2 = set(col_events['method_breakdown'][method2]['event_indices'])
                    overlap = len(indices1.intersection(indices2))
                    col_events['overlap_analysis'][f"{method1}_vs_{method2}"] = {
                        'overlap_count': overlap,
                        'overlap_percentage': (overlap / max(len(indices1), len(indices2))) * 100 if max(len(indices1), len(indices2)) > 0 else 0
                    }
            
            combined['combined_events_by_column'][col] = col_events
        
        # Method comparison
        for method in event_results.keys():
            method_stats = {
                'total_events': sum(col_data.get('event_count', 0) for col_data in event_results[method].values()),
                'affected_columns': len([col for col in event_results[method] if event_results[method][col].get('event_count', 0) > 0]),
                'avg_events_per_column': 0,
                'max_events_in_column': 0
            }
            
            if columns:
                method_stats['avg_events_per_column'] = method_stats['total_events'] / len(columns)
            
            max_events = max((col_data.get('event_count', 0) for col_data in event_results[method].values()), default=0)
            method_stats['max_events_in_column'] = max_events
            
            combined['method_comparison'][method] = method_stats
        
        return combined

    def _generate_advanced_event_charts(self, df: pd.DataFrame, event_results: Dict[str, Any], columns: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced visualizations for event detection results"""
        charts = {}
        
        try:
            # 1. Event overview chart
            charts['event_overview'] = {
                'chart': self._create_event_overview_chart(event_results, columns),
                'insight': self._generate_event_overview_insight(event_results, columns)
            }
            
            # 2. Timeline charts for each method
            for method in event_results.keys():
                charts[f'{method}_timeline'] = {
                    'chart': self._create_event_timeline_chart(df, event_results[method], method, columns),
                    'insight': self._generate_timeline_insight(df, event_results[method], method, columns)
                }
            
            # 3. Event severity heatmap
            charts['severity_heatmap'] = {
                'chart': self._create_event_severity_heatmap(event_results, columns),
                'insight': self._generate_severity_heatmap_insight(event_results, columns)
            }
            
            # 4. Event distribution charts
            charts['event_distribution'] = {
                'chart': self._create_event_distribution_chart(event_results, columns),
                'insight': self._generate_event_distribution_insight(event_results, columns)
            }
            
            # 5. Method comparison chart
            charts['method_comparison'] = {
                'chart': self._create_method_comparison_chart(event_results, columns),
                'insight': self._generate_method_comparison_insight(event_results, columns)
            }
            
        except Exception as e:
            logger.error(f"Error generating event detection charts: {str(e)}")
            charts['error'] = f"Error generating charts: {str(e)}"
        
        return charts

    def _create_event_overview_chart(self, event_results: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Create event overview bar chart"""
        methods = list(event_results.keys())
        event_counts = []
        
        for method in methods:
            total_events = sum(col_data.get('event_count', 0) for col_data in event_results[method].values())
            event_counts.append(total_events)
        
        fig = go.Figure(data=[
            go.Bar(x=methods, y=event_counts, name='Events Detected',
                   marker_color='rgba(55, 83, 109, 0.7)',
                   text=event_counts, textposition='auto')
        ])
        
        fig.update_layout(
            title='Event Detection Overview by Method',
            xaxis_title='Detection Method',
            yaxis_title='Number of Events',
            showlegend=False,
            height=400
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    def _create_event_timeline_chart(self, df: pd.DataFrame, method_results: Dict[str, Any], method: str, columns: List[str]) -> Dict[str, Any]:
        """Create timeline chart for a specific detection method"""
        fig = make_subplots(
            rows=min(len(columns), 4), cols=1,
            subplot_titles=[f'{method.title()} Events in {col}' for col in columns[:4]],
            vertical_spacing=0.08
        )
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, col in enumerate(columns[:4]):
            if col in method_results:
                col_data = method_results[col]
                event_indices = col_data.get('event_indices', [])
                
                # Plot the original data
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[col],
                            mode='lines',
                            name=f'{col} - Data',
                            line=dict(color='lightgray'),
                            showlegend=i == 0
                        ),
                        row=i+1, col=1
                    )
                    
                    # Highlight events
                    if event_indices:
                        event_values = [df.loc[idx, col] for idx in event_indices if idx in df.index]
                        fig.add_trace(
                            go.Scatter(
                                x=event_indices,
                                y=event_values,
                                mode='markers',
                                name=f'{col} - Events',
                                marker=dict(color=colors[i % len(colors)], size=8),
                                showlegend=i == 0
                            ),
                            row=i+1, col=1
                        )
        
        fig.update_layout(
            title=f'{method.title()} Event Detection Timeline',
            height=200 * min(len(columns), 4),
            showlegend=True
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    def _create_event_severity_heatmap(self, event_results: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Create severity heatmap showing event percentages"""
        methods = list(event_results.keys())
        
        # Create matrix of event percentages
        z_data = []
        for method in methods:
            method_percentages = []
            for col in columns:
                if col in event_results[method]:
                    percentage = event_results[method][col].get('event_percentage', 0)
                    method_percentages.append(percentage)
                else:
                    method_percentages.append(0)
            z_data.append(method_percentages)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=columns,
            y=methods,
            colorscale='Reds',
            colorbar=dict(title="Event Percentage")
        ))
        
        fig.update_layout(
            title='Event Detection Severity Heatmap',
            xaxis_title='Columns',
            yaxis_title='Detection Methods',
            height=400
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    def _create_event_distribution_chart(self, event_results: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Create event distribution pie chart"""
        method_totals = {}
        for method in event_results.keys():
            total_events = sum(col_data.get('event_count', 0) for col_data in event_results[method].values())
            method_totals[method] = total_events
        
        fig = go.Figure(data=[go.Pie(
            labels=list(method_totals.keys()),
            values=list(method_totals.values()),
            hole=.3
        )])
        
        fig.update_layout(
            title='Distribution of Events by Detection Method',
            height=400
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    def _create_method_comparison_chart(self, event_results: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Create method comparison chart"""
        methods = list(event_results.keys())
        
        # Calculate metrics for each method
        total_events = []
        affected_columns = []
        avg_percentage = []
        
        for method in methods:
            method_events = sum(col_data.get('event_count', 0) for col_data in event_results[method].values())
            method_affected = len([col for col in event_results[method] if event_results[method][col].get('event_count', 0) > 0])
            method_avg_pct = np.mean([col_data.get('event_percentage', 0) for col_data in event_results[method].values()])
            
            total_events.append(method_events)
            affected_columns.append(method_affected)
            avg_percentage.append(method_avg_pct)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Total Events', 'Affected Columns', 'Avg Event %'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(go.Bar(x=methods, y=total_events, name='Total Events'), row=1, col=1)
        fig.add_trace(go.Bar(x=methods, y=affected_columns, name='Affected Columns'), row=1, col=2)
        fig.add_trace(go.Bar(x=methods, y=avg_percentage, name='Avg Event %'), row=1, col=3)
        
        fig.update_layout(
            title='Event Detection Method Comparison',
            height=400,
            showlegend=False
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    def _generate_statistical_summary(self, df: pd.DataFrame, event_results: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Generate statistical summary of event detection results"""
        stats = {
            'overall_statistics': {},
            'method_statistics': {},
            'column_statistics': {}
        }
        
        # Overall statistics
        total_events = sum(
            sum(col_data.get('event_count', 0) for col_data in method_data.values())
            for method_data in event_results.values()
        )
        
        total_possible_events = len(df) * len(columns) * len(event_results)
        
        stats['overall_statistics'] = {
            'total_events_detected': total_events,
            'total_data_points': len(df) * len(columns),
            'overall_event_rate': (total_events / (len(df) * len(columns))) * 100 if len(df) * len(columns) > 0 else 0,
            'methods_used': len(event_results),
            'columns_analyzed': len(columns)
        }
        
        # Method statistics
        for method, method_results in event_results.items():
            method_events = sum(col_data.get('event_count', 0) for col_data in method_results.values())
            method_percentages = [col_data.get('event_percentage', 0) for col_data in method_results.values()]
            
            stats['method_statistics'][method] = {
                'total_events': method_events,
                'avg_event_percentage': float(np.mean(method_percentages)) if method_percentages else 0,
                'max_event_percentage': float(np.max(method_percentages)) if method_percentages else 0,
                'min_event_percentage': float(np.min(method_percentages)) if method_percentages else 0,
                'std_event_percentage': float(np.std(method_percentages)) if method_percentages else 0
            }
        
        # Column statistics
        for col in columns:
            col_events = sum(
                method_results.get(col, {}).get('event_count', 0)
                for method_results in event_results.values()
            )
            
            col_percentages = [
                method_results.get(col, {}).get('event_percentage', 0)
                for method_results in event_results.values()
            ]
            
            stats['column_statistics'][col] = {
                'total_events': col_events,
                'avg_event_percentage': float(np.mean(col_percentages)) if col_percentages else 0,
                'max_event_percentage': float(np.max(col_percentages)) if col_percentages else 0,
                'affected_by_methods': len([p for p in col_percentages if p > 0])
            }
        
        return stats

    def _generate_advanced_recommendations(self, event_results: Dict[str, Any], combined_results: Dict[str, Any], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate advanced recommendations based on event detection results"""
        recommendations = []
        
        # Check for high event rates
        for method, method_data in summary['event_overview'].items():
            if isinstance(method_data, dict) and method_data.get('total_events', 0) > 0:
                total_events = method_data['total_events']
                affected_columns = method_data['affected_columns']
                
                if total_events > 100:
                    recommendations.append({
                        'type': 'high_event_count',
                        'severity': 'high',
                        'title': f'High {method} Event Count Detected',
                        'description': f'Detected {total_events} {method} events across {affected_columns} columns.',
                        'action': f'Investigate the root cause of {method} events and consider data preprocessing or cleaning.',
                        'priority': 'high'
                    })
        
        # Check severity levels
        severity_analysis = summary.get('severity_analysis', {})
        critical_count = severity_analysis.get('critical', 0)
        high_count = severity_analysis.get('high', 0)
        
        if critical_count > 0:
            recommendations.append({
                'type': 'critical_severity',
                'severity': 'critical',
                'title': 'Critical Event Severity Detected',
                'description': f'{critical_count} column(s) have critical event severity (>15% events).',
                'action': 'Immediate investigation required for columns with critical event rates.',
                'priority': 'critical'
            })
        
        if high_count > 0:
            recommendations.append({
                'type': 'high_severity',
                'severity': 'medium',
                'title': 'High Event Severity Detected',
                'description': f'{high_count} column(s) have high event severity (5-15% events).',
                'action': 'Review data quality and consider preprocessing for high-severity columns.',
                'priority': 'medium'
            })
        
        # Method-specific recommendations
        if 'spikes' in event_results:
            spike_events = sum(col_data.get('event_count', 0) for col_data in event_results['spikes'].values())
            if spike_events > 0:
                recommendations.append({
                    'type': 'spike_events',
                    'severity': 'medium',
                    'title': 'Spike Events Detected',
                    'description': f'Found {spike_events} spike events. These may indicate measurement errors or extreme conditions.',
                    'action': 'Verify spike events with domain knowledge and consider outlier treatment.',
                    'priority': 'medium'
                })
        
        if 'drifts' in event_results:
            drift_events = sum(col_data.get('event_count', 0) for col_data in event_results['drifts'].values())
            if drift_events > 0:
                recommendations.append({
                    'type': 'drift_events',
                    'severity': 'high',
                    'title': 'Drift Events Detected',
                    'description': f'Found {drift_events} drift events. These indicate systematic changes in data distribution.',
                    'action': 'Investigate temporal patterns and consider change point analysis.',
                    'priority': 'high'
                })
        
        if 'gaps' in event_results:
            gap_events = sum(col_data.get('event_count', 0) for col_data in event_results['gaps'].values())
            if gap_events > 0:
                recommendations.append({
                    'type': 'gap_events',
                    'severity': 'medium',
                    'title': 'Gap Events Detected',
                    'description': f'Found {gap_events} significant gaps (missing data sequences).',
                    'action': 'Review data collection process and consider imputation strategies.',
                    'priority': 'medium'
                })
        
        if 'flatlines' in event_results:
            flatline_events = sum(col_data.get('event_count', 0) for col_data in event_results['flatlines'].values())
            if flatline_events > 0:
                recommendations.append({
                    'type': 'flatline_events',
                    'severity': 'medium',
                    'title': 'Flatline Events Detected',
                    'description': f'Found {flatline_events} flatline events indicating potential sensor failures or constant values.',
                    'action': 'Check for sensor malfunctions or data collection issues.',
                    'priority': 'medium'
                })
        
        return recommendations

    def _generate_event_overview_insight(self, event_results: Dict[str, Any], columns: List[str]) -> str:
        """Generate insight for event overview chart"""
        try:
            method_totals = {}
            for method, method_data in event_results.items():
                total_events = sum(col_data.get('event_count', 0) for col_data in method_data.values())
                method_totals[method] = total_events
            
            if not method_totals:
                return "No event detection results available"
            
            total_events = sum(method_totals.values())
            most_active_method = max(method_totals, key=method_totals.get)
            
            return f"""
🎯 **Event Detection Overview**

**Detection Summary:**
• {total_events} total events identified across {len(columns)} columns
• {most_active_method} detected the most events ({method_totals[most_active_method]} total)
• {len(method_totals)} different event types analyzed

**Event Type Explanations:**
• **Spikes**: Sudden value jumps or drops that deviate significantly from surrounding data
• **Drifts**: Gradual trend changes that shift data patterns over time
• **Gaps**: Missing data periods or discontinuities in the data stream
• **Flatlines**: Extended periods where values remain constant or show minimal variation

**Data Quality Insights:**
High event counts may indicate volatile data conditions, collection issues, or genuine system behavior changes. The pattern of events helps distinguish between normal variations and problematic data quality issues.

**Understanding Event Patterns:**
Different event types suggest different underlying causes. Spikes might indicate measurement errors or genuine outliers, drifts suggest systematic changes, gaps indicate collection problems, and flatlines might show sensor failures or system downtime.
            """.strip()
        except Exception as e:
            return f"Error generating event overview insight: {str(e)}"

    def _generate_timeline_insight(self, df: pd.DataFrame, method_data: Dict[str, Any], method: str, columns: List[str]) -> str:
        """Generate insight for timeline charts"""
        try:
            total_events = sum(col_data.get('event_count', 0) for col_data in method_data.values())
            
            method_descriptions = {
                'spikes': 'sudden value changes or outliers',
                'drifts': 'gradual trend changes over time',
                'gaps': 'missing data or discontinuities',
                'flatlines': 'periods of constant values'
            }
            
            description = method_descriptions.get(method, 'events')
            event_rate = (total_events / len(df)) * 100 if len(df) > 0 else 0
            
            return f"""
📈 **{method.title()} Timeline Analysis**

**Event Pattern Summary:**
• Event type: {description}
• {total_events} {method} detected across timeline
• Event rate: {event_rate:.2f}% of total data points
• {len([col for col in method_data if method_data[col].get('event_count', 0) > 0])} columns affected

**Timeline Interpretation:**
The timeline shows data progression over time with red markers indicating detected {method}. Multiple colored lines represent different variables, allowing you to see cross-variable patterns and timing relationships.

**Pattern Recognition:**
• **Clustered events**: Suggest systematic issues or genuine system-wide changes
• **Isolated events**: May represent random occurrences or measurement errors
• **Cross-variable events**: Simultaneous {method} across multiple variables indicate systemic problems
• **Temporal concentration**: High event density in specific time periods warrants investigation

**Event Frequency Assessment:**
{"High event frequency suggests significant data quality or system issues" if event_rate > 10 else "Event frequency appears within normal operational ranges"}

**Timeline Usage:**
Use this visualization to identify when problems occur, correlate events with external factors, and understand whether issues are localized or system-wide.
            """.strip()
        except Exception as e:
            return f"Error generating timeline insight: {str(e)}"

    def _generate_severity_heatmap_insight(self, event_results: Dict[str, Any], columns: List[str]) -> str:
        """Generate insight for severity heatmap"""
        try:
            # Calculate severity matrix
            severity_matrix = {}
            for method, method_data in event_results.items():
                severity_matrix[method] = {}
                for col in columns:
                    if col in method_data:
                        severity_matrix[method][col] = method_data[col].get('event_count', 0)
                    else:
                        severity_matrix[method][col] = 0
            
            # Find hotspots
            column_totals = {}
            for col in columns:
                column_totals[col] = sum(severity_matrix[method].get(col, 0) for method in severity_matrix)
            
            hotspot_column = max(column_totals, key=column_totals.get) if column_totals else "None"
            
            return f"""
🔥 **Event Severity Heatmap Analysis**

**Severity Patterns:**
• Hotspot column: {hotspot_column} ({column_totals.get(hotspot_column, 0)} total events)
• Color intensity indicates event frequency
• Red areas show high event concentration

**Interpretation:**
• Darker red cells indicate columns with high event rates for specific methods
• Patterns across methods reveal systematic vs. random events
• Consistent red patterns suggest persistent data quality issues

**Heatmap Reading:**
• Rows represent detection methods
• Columns represent data variables
• Color intensity shows event severity/frequency

**Business Impact:**
• High-severity columns require immediate attention
• Method-specific patterns guide targeted interventions
• Use for prioritizing data quality improvement efforts

**Key Insights:**
• {"Focus on " + hotspot_column + " - highest event concentration requires immediate attention" if hotspot_column != "None" else "No clear hotspots identified - events are evenly distributed"}
• Investigate columns with consistent high severity across methods as they indicate persistent issues
• Use severity patterns to allocate monitoring resources effectively
            """.strip()
        except Exception as e:
            return f"Error generating severity heatmap insight: {str(e)}"

    def _generate_event_distribution_insight(self, event_results: Dict[str, Any], columns: List[str]) -> str:
        """Generate insight for event distribution chart"""
        try:
            # Calculate distribution statistics
            distribution_stats = {}
            for method, method_data in event_results.items():
                event_counts = [col_data.get('event_count', 0) for col_data in method_data.values()]
                distribution_stats[method] = {
                    'mean': np.mean(event_counts),
                    'std': np.std(event_counts),
                    'max': np.max(event_counts),
                    'total': sum(event_counts)
                }
            
            most_variable_method = max(distribution_stats, key=lambda x: distribution_stats[x]['std']) if distribution_stats else "None"
            
            return f"""
📊 **Event Distribution Analysis**

**Distribution Characteristics:**
• Methods analyzed: {len(distribution_stats)}
• Most variable method: {most_variable_method}
• Shows event frequency distribution across columns

**Statistical Insights:**
• Box plots show median, quartiles, and outliers
• Violin plots reveal distribution shape
• Width indicates variability across columns

**Interpretation:**
• Wide distributions suggest inconsistent event rates
• Narrow distributions indicate uniform event patterns
• Outliers represent columns with unusually high event rates

**Method Comparison:**
• Consistent methods show similar event rates across columns
• Variable methods are sensitive to column characteristics
• Use distribution shape to understand method behavior

**Key Insights:**
• {"Focus on " + most_variable_method + " - shows highest variability and may need parameter tuning" if most_variable_method != "None" else "Event distributions are consistent across methods"}
• Investigate outlier columns with extreme event rates as they may indicate data quality issues
• Use distribution insights to guide method selection based on your specific data characteristics
            """.strip()
        except Exception as e:
            return f"Error generating event distribution insight: {str(e)}"

    def _generate_method_comparison_insight(self, event_results: Dict[str, Any], columns: List[str]) -> str:
        """Generate insight for method comparison chart"""
        try:
            method_effectiveness = {}
            for method, method_data in event_results.items():
                total_events = sum(col_data.get('event_count', 0) for col_data in method_data.values())
                columns_affected = len([col for col in method_data if method_data[col].get('event_count', 0) > 0])
                method_effectiveness[method] = {
                    'total_events': total_events,
                    'columns_affected': columns_affected,
                    'avg_events_per_column': total_events / len(columns) if columns else 0
                }
            
            if not method_effectiveness:
                return "No method comparison data available"
            
            most_sensitive = max(method_effectiveness, key=lambda x: method_effectiveness[x]['total_events'])
            broadest_coverage = max(method_effectiveness, key=lambda x: method_effectiveness[x]['columns_affected'])
            
            return f"""
⚖️ **Method Comparison Analysis**

**Effectiveness Metrics:**
• Most sensitive: {most_sensitive} ({method_effectiveness[most_sensitive]['total_events']} events)
• Broadest coverage: {broadest_coverage} ({method_effectiveness[broadest_coverage]['columns_affected']} columns)
• Methods compared: {len(method_effectiveness)}

**Method Characteristics:**
• Spikes: Sensitive to sudden changes and outliers
• Drifts: Detects gradual systematic changes
• Gaps: Identifies missing data patterns
• Flatlines: Finds periods of no variation

**Comparison Insights:**
• Different methods capture different event types
• High sensitivity may indicate data volatility
• Broad coverage suggests systematic issues

**Interpretation:**
Each method serves different purposes. Combine multiple methods for comprehensive event detection. High event counts don't always indicate problems - consider context and business requirements.

**Key Insights:**
• Use {most_sensitive} for sensitive change detection when early warning is critical
• Apply {broadest_coverage} for broad event coverage to catch diverse event types
• Combine methods for comprehensive monitoring rather than relying on single approaches
• Validate high-sensitivity methods to avoid false alarms while maintaining detection capability
            """.strip()
        except Exception as e:
            return f"Error generating method comparison insight: {str(e)}"
