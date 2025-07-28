"""
Advanced EDA Service for comprehensive exploratory data analysis generation
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.utils import PlotlyJSONEncoder
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, normaltest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from app.models import Dataset
from app import db


class EDAService:
    """Service for automatic exploratory data analysis generation"""
    
    def __init__(self):
        self.chart_colors = px.colors.qualitative.Set3
    
    def _convert_to_serializable(self, obj):
        """Recursively convert numpy types and other non-serializable objects to JSON-serializable types"""
        import numpy as np
        from datetime import datetime, date, time
        
        # Handle None and basic types
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
            
        # Handle datetime objects
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
            
        # Handle numpy numeric types
        if hasattr(np, 'integer') and isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                                                    np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
            
        if hasattr(np, 'floating') and isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
            
        if hasattr(np, 'bool_') and isinstance(obj, np.bool_):
            return bool(obj)
            
        # Handle numpy arrays and generic numpy types
        if hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
            return [self._convert_to_serializable(x) for x in obj.tolist()]
            
        if hasattr(np, 'generic') and isinstance(obj, np.generic):
            return self._convert_to_serializable(obj.item())
            
        # Handle numpy dtypes
        if hasattr(obj, 'dtype') and hasattr(obj, 'name'):
            return str(obj.name)
            
        # Handle dictionaries - ensure all keys are strings
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Convert key to string if it's not already
                safe_key = str(k) if not isinstance(k, (str, int, float, bool)) or k is None else k
                result[safe_key] = self._convert_to_serializable(v)
            return result
            
        # Handle sequences (list, tuple, set)
        if isinstance(obj, (list, tuple, set)):
            return [self._convert_to_serializable(item) for item in obj]
            
        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            return self._convert_to_serializable(obj.__dict__)
            
        # Handle objects with item() method (like numpy scalars)
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            return self._convert_to_serializable(obj.item())
            
        # Handle numpy dtypes
        if type(obj).__module__ == 'numpy' and hasattr(obj, 'dtype'):
            return str(obj.dtype)
            
        # As a last resort, convert to string
        return str(obj)

    def generate_eda(self, dataset_id: int) -> Dict[str, Any]:
        """Generate comprehensive advanced EDA for a dataset"""
        try:
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return {'success': False, 'error': 'Dataset not found'}
            
            if not os.path.exists(dataset.file_path):
                return {'success': False, 'error': 'Dataset file not found'}
            
            # Read data
            df = self._read_data_from_file(dataset.file_path, dataset.file_type)
            if df is None:
                return {'success': False, 'error': 'Failed to read dataset'}
            
            # Handle large datasets by sampling if necessary
            if len(df) > 50000:
                df_sample = df.sample(n=50000, random_state=42)
                sample_note = f"Analysis performed on a sample of 50,000 rows from {len(df):,} total rows"
            else:
                df_sample = df.copy()
                sample_note = None

            # Prepare data types and clean data
            df_clean = self._prepare_data(df_sample)
            
            # Generate comprehensive EDA components
            eda_results = {}
            charts = {}

            # 1. Executive Summary
            eda_results['executive_summary'] = self._generate_executive_summary(df_clean, sample_note)

            # 2. Dataset Overview with enhanced metrics
            eda_results['overview'] = self._generate_advanced_overview(df_clean)

            # 3. Data Quality Assessment
            eda_results['data_quality'] = self._generate_comprehensive_data_quality(df_clean)
            charts['data_quality_charts'] = self._create_data_quality_charts(df_clean)

            # 4. Variable Analysis
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df_clean.select_dtypes(include=['datetime64']).columns.tolist()

            # 5. Advanced Numerical Analysis
            if numeric_cols:
                eda_results['numerical_analysis'] = self._analyze_numerical_advanced(df_clean, numeric_cols)
                charts['numerical_charts'] = self._create_advanced_numerical_charts(df_clean, numeric_cols)

            # 6. Advanced Categorical Analysis
            if categorical_cols:
                eda_results['categorical_analysis'] = self._analyze_categorical_advanced(df_clean, categorical_cols)
                charts['categorical_charts'] = self._create_advanced_categorical_charts(df_clean, categorical_cols)

            # 7. Feature Relationships and Correlations
            if len(numeric_cols) > 1:
                eda_results['relationships'] = self._analyze_feature_relationships(df_clean, numeric_cols, categorical_cols)
                charts['relationship_charts'] = self._create_relationship_charts(df_clean, numeric_cols, categorical_cols)

            # 8. Outlier Detection with Multiple Methods
            if numeric_cols:
                eda_results['outliers'] = self._detect_outliers_advanced(df_clean, numeric_cols)
                charts['outlier_charts'] = self._create_advanced_outlier_charts(df_clean, numeric_cols)

            # 9. Dimensionality Reduction Analysis
            if len(numeric_cols) >= 3:
                eda_results['dimensionality'] = self._analyze_dimensionality(df_clean, numeric_cols)
                charts['dimensionality_charts'] = self._create_dimensionality_charts(df_clean, numeric_cols)

            # 10. Clustering Analysis
            if len(numeric_cols) >= 2:
                eda_results['clustering'] = self._analyze_clustering(df_clean, numeric_cols)
                charts['clustering_charts'] = self._create_clustering_charts(df_clean, numeric_cols)

            # 11. Time Series Analysis (if datetime columns exist)
            if datetime_cols and numeric_cols:
                eda_results['time_series'] = self._analyze_time_series_advanced(df_clean, datetime_cols, numeric_cols)
                charts['time_series_charts'] = self._create_time_series_charts(df_clean, datetime_cols, numeric_cols)

            # 12. Feature Importance and Selection
            if len(numeric_cols) > 1:
                eda_results['feature_importance'] = self._analyze_feature_importance(df_clean, numeric_cols, categorical_cols)
                charts['feature_importance_charts'] = self._create_feature_importance_charts(df_clean, numeric_cols, categorical_cols)

            # 13. Distribution Analysis
            eda_results['distributions'] = self._analyze_distributions_advanced(df_clean, numeric_cols, categorical_cols)
            charts['distribution_charts'] = self._create_distribution_charts_advanced(df_clean, numeric_cols, categorical_cols)

            # 14. Statistical Tests
            eda_results['statistical_tests'] = self._perform_statistical_tests(df_clean, numeric_cols, categorical_cols)

            # 15. Recommendations with Priority
            eda_results['recommendations'] = self._generate_advanced_recommendations(df_clean, eda_results)

            # Convert all numpy types to Python native types for JSON serialization
            eda_results = self._convert_to_serializable(eda_results)
            charts = self._convert_to_serializable(charts)

            return {
                'success': True,
                'eda_results': eda_results,
                'charts': charts
            }
            
        except Exception as e:
            import traceback
            print(f"EDA Generation Error: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data for analysis"""
        df_clean = df.copy()
        
        # Auto-detect and convert datetime columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Check if it might be a datetime
                if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'created', 'updated']):
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    except:
                        pass
                
                # Check if it might be a categorical with low cardinality
                elif df_clean[col].nunique() < 50 and df_clean[col].nunique() / len(df_clean) < 0.1:
                    df_clean[col] = df_clean[col].astype('category')
        
        return df_clean

    def _generate_executive_summary(self, df: pd.DataFrame, sample_note: str = None) -> Dict[str, Any]:
        """Generate executive summary of the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        duplicates = df.duplicated().sum()
        
        # Data complexity score
        complexity_score = 0
        if len(numeric_cols) > 10: complexity_score += 2
        elif len(numeric_cols) > 5: complexity_score += 1
        
        if len(categorical_cols) > 10: complexity_score += 2
        elif len(categorical_cols) > 3: complexity_score += 1
        
        if missing_percentage > 20: complexity_score += 2
        elif missing_percentage > 5: complexity_score += 1
        
        complexity_labels = {0: 'Simple', 1: 'Simple', 2: 'Moderate', 3: 'Moderate', 4: 'Complex', 5: 'Complex', 6: 'Very Complex'}
        
        key_insights = []
        if missing_percentage > 10:
            key_insights.append(f"High missing data rate: {missing_percentage:.1f}%")
        if duplicates > len(df) * 0.05:
            key_insights.append(f"Contains {duplicates:,} duplicate rows ({duplicates/len(df)*100:.1f}%)")
        if len(numeric_cols) == 0:
            key_insights.append("No numeric variables detected - primarily categorical dataset")
        if len(categorical_cols) == 0:
            key_insights.append("No categorical variables detected - primarily numeric dataset")
        if len(datetime_cols) > 0:
            key_insights.append(f"Time series data detected with {len(datetime_cols)} datetime column(s)")
        
        return {
            'dataset_size': {'rows': len(df), 'columns': len(df.columns)},
            'variable_types': {
                'numeric': len(numeric_cols),
                'categorical': len(categorical_cols),
                'datetime': len(datetime_cols)
            },
            'data_quality': {
                'missing_percentage': missing_percentage,
                'duplicate_rows': duplicates,
                'completeness_score': 100 - missing_percentage
            },
            'complexity': {
                'score': complexity_score,
                'label': complexity_labels.get(complexity_score, 'Very Complex')
            },
            'key_insights': key_insights,
            'sample_note': sample_note
        }

    def _generate_advanced_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate advanced dataset overview with memory and performance metrics"""
        memory_usage = df.memory_usage(deep=True)
        
        return {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'memory_usage': {
                'total_mb': memory_usage.sum() / (1024 * 1024),
                'by_column': {col: memory_usage[col] / (1024 * 1024) for col in df.columns},
                'average_per_row': memory_usage.sum() / len(df) if len(df) > 0 else 0
            },
            'column_details': {
                col: {
                    'dtype': str(df[col].dtype),
                    'non_null_count': df[col].count(),
                    'null_count': df[col].isnull().sum(),
                    'unique_count': df[col].nunique(),
                    'memory_mb': memory_usage[col] / (1024 * 1024)
                } for col in df.columns
            },
            'data_density': (df.count().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
        }

    def _generate_comprehensive_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality assessment"""
        quality_metrics = {}
        
        # Completeness
        missing_by_col = df.isnull().sum()
        completeness_score = ((len(df) * len(df.columns)) - missing_by_col.sum()) / (len(df) * len(df.columns)) * 100
        
        # Consistency (data type consistency)
        consistency_issues = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 50:  # High cardinality text columns
                consistency_issues.append(f"{col}: High cardinality ({df[col].nunique()} unique values)")
        
        # Validity (basic range checks for numeric data)
        validity_issues = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if (df[col] < 0).any() and col.lower() in ['age', 'price', 'cost', 'amount', 'quantity']:
                validity_issues.append(f"{col}: Contains negative values")
            if np.isinf(df[col]).any():
                validity_issues.append(f"{col}: Contains infinite values")
        
        # Uniqueness
        duplicate_rows = df.duplicated().sum()
        uniqueness_score = (len(df) - duplicate_rows) / len(df) * 100 if len(df) > 0 else 100
        
        # Overall quality score
        quality_score = (completeness_score * 0.4 + uniqueness_score * 0.3 + 
                        (100 - len(consistency_issues) * 10) * 0.2 + 
                        (100 - len(validity_issues) * 10) * 0.1)
        quality_score = max(0, min(100, quality_score))
        
        return {
            'completeness': {
                'score': completeness_score,
                'missing_by_column': missing_by_col.to_dict(),
                'total_missing_cells': missing_by_col.sum()
            },
            'consistency': {
                'score': max(0, 100 - len(consistency_issues) * 10),
                'issues': consistency_issues
            },
            'validity': {
                'score': max(0, 100 - len(validity_issues) * 10),
                'issues': validity_issues
            },
            'uniqueness': {
                'score': uniqueness_score,
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': (duplicate_rows / len(df) * 100) if len(df) > 0 else 0
            },
            'overall_quality_score': quality_score,
            'quality_grade': self._get_quality_grade(quality_score)
        }

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
        """Generate dataset overview"""
        return {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'memory_usage': {
                'total_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'by_column': {col: df[col].memory_usage(deep=True) / (1024 * 1024) 
                             for col in df.columns}
            },
            'column_names': df.columns.tolist(),
            'data_types_summary': df.dtypes.value_counts().to_dict()
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataset"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing': int(missing_counts.sum()),
            'missing_percentage': float(missing_percentages.sum() / len(df.columns)),
            'by_column': {
                col: {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentages[col])
                }
                for col in df.columns if missing_counts[col] > 0
            },
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist(),
            'complete_rows': int(len(df) - df.isnull().any(axis=1).sum())
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types in the dataset"""
        type_counts = df.dtypes.value_counts()
        
        return {
            'summary': {str(dtype): int(count) for dtype, count in type_counts.items()},
            'by_column': {col: str(df[col].dtype) for col in df.columns},
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
    
    def _analyze_numerical_variables(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze numerical variables"""
        analysis = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            analysis[col] = {
                'count': int(series.count()),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'unique_values': int(series.nunique()),
                'zero_count': int((series == 0).sum()),
                'negative_count': int((series < 0).sum())
            }
        
        return analysis
    
    def _analyze_categorical_variables(self, df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Any]:
        """Analyze categorical variables"""
        analysis = {}
        
        for col in categorical_cols:
            series = df[col].dropna()
            value_counts = series.value_counts()
            
            analysis[col] = {
                'count': int(series.count()),
                'unique_values': int(series.nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'top_values': {str(k): int(v) for k, v in value_counts.head(10).items()},
                'avg_length': float(series.astype(str).str.len().mean()) if not series.empty else 0
            }
        
        return analysis
    
    def _detect_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        outliers = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = outlier_mask.sum()
            
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': float(outlier_count / len(series) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values': series[outlier_mask].tolist()[:20]  # Limit to first 20
            }
        
        return outliers
    
    def _generate_data_quality_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality summary"""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        return {
            'completeness': {
                'score': float((total_cells - missing_cells) / total_cells * 100),
                'missing_cells': int(missing_cells),
                'total_cells': int(total_cells)
            },
            'uniqueness': {
                'duplicate_rows': int(duplicate_rows),
                'unique_rows': int(len(df) - duplicate_rows),
                'uniqueness_score': float((len(df) - duplicate_rows) / len(df) * 100)
            },
            'overall_score': float(((total_cells - missing_cells) / total_cells * 0.6 + 
                                  (len(df) - duplicate_rows) / len(df) * 0.4) * 100)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, eda_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate data improvement recommendations"""
        recommendations = []
        
        # Missing values recommendations
        missing_data = eda_results.get('missing_values', {})
        for col, info in missing_data.get('by_column', {}).items():
            if info['percentage'] > 50:
                recommendations.append({
                    'type': 'missing_values',
                    'priority': 'high',
                    'column': col,
                    'message': f"Consider dropping column '{col}' due to {info['percentage']:.1f}% missing values",
                    'action': 'drop_column'
                })
            elif info['percentage'] > 10:
                recommendations.append({
                    'type': 'missing_values',
                    'priority': 'medium',
                    'column': col,
                    'message': f"Handle missing values in '{col}' ({info['percentage']:.1f}% missing)",
                    'action': 'impute_values'
                })
        
        # Outliers recommendations
        outliers_data = eda_results.get('outliers', {})
        for col, info in outliers_data.items():
            if info['percentage'] > 5:
                recommendations.append({
                    'type': 'outliers',
                    'priority': 'medium',
                    'column': col,
                    'message': f"Column '{col}' has {info['percentage']:.1f}% outliers",
                    'action': 'investigate_outliers'
                })
        
        # Data types recommendations
        categorical_data = eda_results.get('categorical_analysis', {})
        for col, info in categorical_data.items():
            if info['unique_values'] > len(df) * 0.8:
                recommendations.append({
                    'type': 'data_types',
                    'priority': 'low',
                    'column': col,
                    'message': f"Column '{col}' has high cardinality ({info['unique_values']} unique values)",
                    'action': 'consider_grouping'
                })
        
        # Duplicate rows recommendation
        quality_data = eda_results.get('data_quality', {})
        if quality_data.get('uniqueness', {}).get('duplicate_rows', 0) > 0:
            recommendations.append({
                'type': 'duplicates',
                'priority': 'medium',
                'column': None,
                'message': f"Dataset contains {quality_data['uniqueness']['duplicate_rows']} duplicate rows",
                'action': 'remove_duplicates'
            })
        
        return recommendations
    
    def _create_data_quality_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create comprehensive data quality visualization charts"""
        charts = []
        
        # 1. Missing Values Heatmap
        missing_data = df.isnull()
        if missing_data.any().any():
            fig = go.Figure(data=go.Heatmap(
                z=missing_data.values.astype(int),
                x=df.columns,
                y=list(range(len(df))),
                colorscale=[[0, 'lightblue'], [1, 'red']],
                showscale=True,
                colorbar=dict(title="Missing", tickvals=[0, 1], ticktext=["Present", "Missing"])
            ))
            fig.update_layout(
                title='Missing Values Pattern',
                xaxis_title='Columns',
                yaxis_title='Rows',
                template='plotly_white',
                height=400
            )
            charts.append({
                'type': 'missing_values_heatmap',
                'title': 'Missing Values Pattern',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        # 2. Data Completeness by Column
        completeness = (1 - df.isnull().mean()) * 100
        fig = go.Figure(data=[
            go.Bar(
                x=completeness.values,
                y=completeness.index,
                orientation='h',
                marker_color=['red' if x < 50 else 'orange' if x < 80 else 'green' for x in completeness.values]
            )
        ])
        fig.update_layout(
            title='Data Completeness by Column',
            xaxis_title='Completeness (%)',
            yaxis_title='Columns',
            template='plotly_white'
        )
        charts.append({
            'type': 'completeness_by_column',
            'title': 'Data Completeness by Column',
            'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
        })
        
        # 3. Data Types Distribution
        type_counts = df.dtypes.value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=[str(dtype) for dtype in type_counts.index],
                values=type_counts.values,
                hole=0.3,
                marker_colors=self.chart_colors[:len(type_counts)]
            )
        ])
        fig.update_layout(
            title='Data Types Distribution',
            template='plotly_white'
        )
        charts.append({
            'type': 'data_types_distribution',
            'title': 'Data Types Distribution', 
            'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
        })
        
        return charts

    def _create_advanced_numerical_charts(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced numerical analysis charts"""
        charts = []
        
        # 1. Distribution Dashboard (subplot with histograms)
        cols_to_plot = numeric_cols[:6]  # Limit to 6 for readability
        if cols_to_plot:
            rows = 2
            cols = 3
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=cols_to_plot,
                specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
            )
            
            for i, col in enumerate(cols_to_plot):
                row = i // cols + 1
                col_pos = i % cols + 1
                
                data = df[col].dropna()
                fig.add_trace(
                    go.Histogram(x=data, name=col, showlegend=False, marker_color=self.chart_colors[i]),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title='Distribution Dashboard',
                template='plotly_white',
                height=600
            )
            
            charts.append({
                'type': 'distribution_dashboard',
                'title': 'Distribution Dashboard',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        # 2. Advanced Correlation Matrix with Clustering
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Create clustered heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={'size': 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Enhanced Correlation Matrix',
                template='plotly_white',
                width=600,
                height=600
            )
            
            charts.append({
                'type': 'correlation_matrix',
                'title': 'Enhanced Correlation Matrix',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        # 3. Box Plot Dashboard
        if len(numeric_cols) > 1:
            fig = go.Figure()
            
            for i, col in enumerate(numeric_cols[:8]):  # Limit to 8 columns
                fig.add_trace(go.Box(
                    y=df[col].dropna(),
                    name=col,
                    marker_color=self.chart_colors[i % len(self.chart_colors)]
                ))
            
            fig.update_layout(
                title='Box Plot Comparison',
                yaxis_title='Values',
                template='plotly_white',
                showlegend=False
            )
            
            charts.append({
                'type': 'box_plot_dashboard',
                'title': 'Box Plot Comparison',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        # 4. Violin Plots for Distribution Shape
        if len(numeric_cols) >= 2:
            fig = go.Figure()
            
            for i, col in enumerate(numeric_cols[:6]):
                fig.add_trace(go.Violin(
                    y=df[col].dropna(),
                    name=col,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.chart_colors[i % len(self.chart_colors)],
                    line_color='black'
                ))
            
            fig.update_layout(
                title='Distribution Shapes (Violin Plots)',
                yaxis_title='Values',
                template='plotly_white'
            )
            
            charts.append({
                'type': 'violin_plots',
                'title': 'Distribution Shapes (Violin Plots)',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        return charts

    def _create_advanced_categorical_charts(self, df: pd.DataFrame, categorical_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced categorical analysis charts"""
        charts = []
        
        # 1. Top Categories Dashboard
        for i, col in enumerate(categorical_cols[:6]):
            value_counts = df[col].value_counts().head(15)  # Top 15 values
            
            fig = go.Figure(data=[
                go.Bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    marker_color=self.chart_colors[i % len(self.chart_colors)],
                    text=value_counts.values,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title=f'Top Categories: {col}',
                xaxis_title='Count',
                yaxis_title=col,
                template='plotly_white',
                height=400
            )
            
            charts.append({
                'type': 'categorical_bar',
                'column': col,
                'title': f'Top Categories: {col}',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        # 2. Categorical Distribution Pie Charts
        for i, col in enumerate(categorical_cols[:4]):
            value_counts = df[col].value_counts()
            
            # Group small categories into "Others"
            if len(value_counts) > 10:
                top_categories = value_counts.head(9)
                others_count = value_counts.tail(len(value_counts) - 9).sum()
                
                labels = list(top_categories.index) + ['Others']
                values = list(top_categories.values) + [others_count]
            else:
                labels = list(value_counts.index)
                values = list(value_counts.values)
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    marker_colors=self.chart_colors[:len(labels)]
                )
            ])
            
            fig.update_layout(
                title=f'Distribution: {col}',
                template='plotly_white'
            )
            
            charts.append({
                'type': 'categorical_pie',
                'column': col,
                'title': f'Distribution: {col}',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        # 3. Cardinality Comparison
        cardinality_data = {col: df[col].nunique() for col in categorical_cols}
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(cardinality_data.keys()),
                y=list(cardinality_data.values()),
                marker_color=[self.chart_colors[i % len(self.chart_colors)] 
                            for i in range(len(cardinality_data))],
                text=list(cardinality_data.values()),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Cardinality by Categorical Column',
            xaxis_title='Columns',
            yaxis_title='Unique Values Count',
            template='plotly_white'
        )
        
        charts.append({
            'type': 'cardinality_comparison',
            'title': 'Cardinality by Categorical Column',
            'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
        })
        
        return charts

    def _create_relationship_charts(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced relationship analysis charts"""
        charts = []
        
        # 1. Enhanced Scatter Matrix
        if len(numeric_cols) >= 3:
            cols_to_plot = numeric_cols[:5]  # Limit for performance
            
            fig = px.scatter_matrix(
                df[cols_to_plot].dropna(),
                dimensions=cols_to_plot,
                title='Enhanced Scatter Matrix',
                color_discrete_sequence=self.chart_colors
            )
            
            fig.update_layout(
                template='plotly_white',
                height=800
            )
            
            charts.append({
                'type': 'scatter_matrix',
                'title': 'Enhanced Scatter Matrix',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        # 2. Correlation Network (if many features)
        if len(numeric_cols) > 5:
            corr_matrix = df[numeric_cols].corr()
            
            # Create network graph for strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation
                        strong_correlations.append({
                            'source': corr_matrix.columns[i],
                            'target': corr_matrix.columns[j],
                            'weight': abs(corr_val)
                        })
            
            if strong_correlations:
                # Create a simple network visualization
                nodes = list(set([c['source'] for c in strong_correlations] + 
                               [c['target'] for c in strong_correlations]))
                
                fig = go.Figure()
                
                # Add edges
                for corr in strong_correlations:
                    fig.add_trace(go.Scatter(
                        x=[nodes.index(corr['source']), nodes.index(corr['target'])],
                        y=[0, 0],
                        mode='lines',
                        line=dict(width=corr['weight'] * 5, color='gray'),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=list(range(len(nodes))),
                    y=[0] * len(nodes),
                    mode='markers+text',
                    marker=dict(size=20, color=self.chart_colors[:len(nodes)]),
                    text=nodes,
                    textposition='top center',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title='Correlation Network (Strong Correlations)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    template='plotly_white',
                    height=400
                )
                
                charts.append({
                    'type': 'correlation_network',
                    'title': 'Correlation Network',
                    'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
                })
        
        # 3. Categorical vs Numeric Relationships
        if categorical_cols and numeric_cols:
            for cat_col in categorical_cols[:3]:  # Limit to first 3
                for num_col in numeric_cols[:3]:  # Limit to first 3
                    # Create box plot by category
                    fig = go.Figure()
                    
                    categories = df[cat_col].value_counts().head(10).index  # Top 10 categories
                    for category in categories:
                        data = df[df[cat_col] == category][num_col].dropna()
                        if len(data) > 0:
                            fig.add_trace(go.Box(
                                y=data,
                                name=str(category),
                                boxpoints='outliers'
                            ))
                    
                    fig.update_layout(
                        title=f'{num_col} by {cat_col}',
                        xaxis_title=cat_col,
                        yaxis_title=num_col,
                        template='plotly_white'
                    )
                    
                    charts.append({
                        'type': 'categorical_numeric_relationship',
                        'title': f'{num_col} by {cat_col}',
                        'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
                    })
        
        return charts

    def _create_advanced_outlier_charts(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced outlier detection charts"""
        charts = []
        
        # 1. Multi-method Outlier Detection Dashboard
        for col in numeric_cols[:4]:  # Limit to first 4 columns
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            # Calculate outliers using different methods
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            iqr_lower = Q1 - 1.5 * IQR
            iqr_upper = Q3 + 1.5 * IQR
            
            z_scores = np.abs(stats.zscore(series))
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'{col} - IQR Method',
                    f'{col} - Z-Score Method',
                    f'{col} - Box Plot',
                    f'{col} - Distribution'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # IQR Method
            colors = ['blue' if iqr_lower <= val <= iqr_upper else 'red' for val in series]
            fig.add_trace(
                go.Scatter(x=list(range(len(series))), y=series, mode='markers',
                          marker=dict(color=colors), name='IQR Outliers'),
                row=1, col=1
            )
            
            # Z-Score Method
            colors = ['blue' if z_score <= 3 else 'red' for z_score in z_scores]
            fig.add_trace(
                go.Scatter(x=list(range(len(series))), y=series, mode='markers',
                          marker=dict(color=colors), name='Z-Score Outliers'),
                row=1, col=2
            )
            
            # Box Plot
            fig.add_trace(
                go.Box(y=series, name='Box Plot'),
                row=2, col=1
            )
            
            # Distribution
            fig.add_trace(
                go.Histogram(x=series, name='Distribution', nbinsx=30),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'Outlier Analysis: {col}',
                template='plotly_white',
                height=600,
                showlegend=False
            )
            
            charts.append({
                'type': 'outlier_analysis',
                'column': col,
                'title': f'Outlier Analysis: {col}',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        return charts

    def _create_dimensionality_charts(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Create dimensionality reduction visualization charts"""
        charts = []
        
        if len(numeric_cols) < 3:
            return charts
        
        try:
            # Prepare data
            data = df[numeric_cols].dropna()
            if len(data) < 10:
                return charts
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # PCA
            pca = PCA()
            pca_result = pca.fit_transform(data_scaled)
            
            # 1. Explained Variance Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=pca.explained_variance_ratio_,
                mode='lines+markers',
                name='Individual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=np.cumsum(pca.explained_variance_ratio_),
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='PCA Explained Variance',
                xaxis_title='Principal Component',
                yaxis_title='Explained Variance Ratio',
                template='plotly_white'
            )
            
            charts.append({
                'type': 'pca_explained_variance',
                'title': 'PCA Explained Variance',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
            
            # 2. PCA Biplot (first 2 components)
            if len(pca_result) > 0 and pca_result.shape[1] >= 2:
                fig = go.Figure()
                
                # Add data points
                fig.add_trace(go.Scatter(
                    x=pca_result[:, 0],
                    y=pca_result[:, 1],
                    mode='markers',
                    marker=dict(size=5, opacity=0.6, color='blue'),
                    name='Data Points'
                ))
                
                # Add feature vectors
                for i, feature in enumerate(numeric_cols[:10]):  # Limit to 10 features
                    fig.add_annotation(
                        x=pca.components_[0, i] * 3,
                        y=pca.components_[1, i] * 3,
                        ax=0, ay=0,
                        arrowhead=2,
                        arrowcolor='red',
                        text=feature,
                        font=dict(color='red')
                    )
                
                fig.update_layout(
                    title='PCA Biplot (First 2 Components)',
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                    template='plotly_white'
                )
                
                charts.append({
                    'type': 'pca_biplot',
                    'title': 'PCA Biplot',
                    'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
                })
            
        except Exception as e:
            print(f"Error creating dimensionality charts: {e}")
        
        return charts

    def _create_clustering_charts(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Create clustering analysis charts"""
        charts = []
        
        if len(numeric_cols) < 2:
            return charts
        
        try:
            # Prepare data
            data = df[numeric_cols].dropna()
            if len(data) < 10:
                return charts
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Elbow curve
            max_clusters = min(10, len(data) // 2)
            inertias = []
            silhouette_scores = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data_scaled)
                inertias.append(kmeans.inertia_)
                
                try:
                    from sklearn.metrics import silhouette_score
                    sil_score = silhouette_score(data_scaled, cluster_labels)
                    silhouette_scores.append(sil_score)
                except:
                    silhouette_scores.append(0)
            
            # 1. Elbow Curve
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=list(range(2, max_clusters + 1)), y=inertias,
                          mode='lines+markers', name='Inertia', line=dict(color='blue')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(2, max_clusters + 1)), y=silhouette_scores,
                          mode='lines+markers', name='Silhouette Score', line=dict(color='red')),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Number of Clusters")
            fig.update_yaxes(title_text="Inertia", secondary_y=False)
            fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
            fig.update_layout(title='Clustering Optimization', template='plotly_white')
            
            charts.append({
                'type': 'clustering_optimization',
                'title': 'Clustering Optimization',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
            
            # 2. Cluster Visualization (2D)
            optimal_k = self._find_optimal_clusters(inertias, silhouette_scores)
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_clusters = kmeans_final.fit_predict(data_scaled)
            
            # Use first 2 PCA components for visualization
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data_scaled)
            
            fig = go.Figure()
            
            for cluster_id in range(optimal_k):
                cluster_mask = final_clusters == cluster_id
                fig.add_trace(go.Scatter(
                    x=data_2d[cluster_mask, 0],
                    y=data_2d[cluster_mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(size=6, color=self.chart_colors[cluster_id % len(self.chart_colors)])
                ))
            
            # Add centroids
            centroids_2d = pca.transform(kmeans_final.cluster_centers_)
            fig.add_trace(go.Scatter(
                x=centroids_2d[:, 0],
                y=centroids_2d[:, 1],
                mode='markers',
                name='Centroids',
                marker=dict(size=15, color='black', symbol='x')
            ))
            
            fig.update_layout(
                title=f'Cluster Visualization ({optimal_k} Clusters)',
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                template='plotly_white'
            )
            
            charts.append({
                'type': 'cluster_visualization',
                'title': 'Cluster Visualization',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
            
        except Exception as e:
            print(f"Error creating clustering charts: {e}")
        
        return charts

    def _create_time_series_charts(self, df: pd.DataFrame, datetime_cols: List[str], numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced time series analysis charts"""
        charts = []
        
        for date_col in datetime_cols:
            for num_col in numeric_cols[:3]:  # Limit for performance
                ts_data = df[[date_col, num_col]].dropna().sort_values(date_col)
                if len(ts_data) < 10:
                    continue
                
                # 1. Time Series with Trend and Moving Average
                fig = go.Figure()
                
                # Original series
                fig.add_trace(go.Scatter(
                    x=ts_data[date_col],
                    y=ts_data[num_col],
                    mode='lines',
                    name=num_col,
                    line=dict(color='blue')
                ))
                
                # Moving average
                if len(ts_data) > 10:
                    window = min(30, len(ts_data) // 4)
                    moving_avg = ts_data[num_col].rolling(window=window, center=True).mean()
                    fig.add_trace(go.Scatter(
                        x=ts_data[date_col],
                        y=moving_avg,
                        mode='lines',
                        name=f'Moving Average ({window} periods)',
                        line=dict(color='red', dash='dash')
                    ))
                
                # Trend line
                try:
                    ts_data_copy = ts_data.copy()
                    ts_data_copy['date_numeric'] = (ts_data_copy[date_col] - ts_data_copy[date_col].min()).dt.days
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        ts_data_copy['date_numeric'], ts_data_copy[num_col]
                    )
                    
                    trend_line = slope * ts_data_copy['date_numeric'] + intercept
                    fig.add_trace(go.Scatter(
                        x=ts_data[date_col],
                        y=trend_line,
                        mode='lines',
                        name=f'Trend (R²={r_value**2:.3f})',
                        line=dict(color='green', dash='dot')
                    ))
                except:
                    pass
                
                fig.update_layout(
                    title=f'Time Series Analysis: {num_col} over {date_col}',
                    xaxis_title=date_col,
                    yaxis_title=num_col,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                charts.append({
                    'type': 'time_series_analysis',
                    'title': f'Time Series: {num_col}',
                    'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
                })
        
        return charts

    def _create_feature_importance_charts(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[Dict[str, Any]]:
        """Create feature importance visualization charts"""
        charts = []
        
        if len(numeric_cols) < 2:
            return charts
        
        # For each numeric target, show feature importance
        for target_col in numeric_cols[:3]:  # Limit to first 3 targets
            other_numeric = [col for col in numeric_cols if col != target_col]
            
            if len(other_numeric) == 0:
                continue
            
            try:
                data = df[other_numeric + [target_col]].dropna()
                if len(data) < 10:
                    continue
                
                X = data[other_numeric]
                y = data[target_col]
                
                # Mutual information scores
                mi_scores = mutual_info_regression(X, y, random_state=42)
                
                # Correlation scores
                corr_scores = [abs(data[col].corr(data[target_col])) for col in other_numeric]
                
                fig = go.Figure()
                
                # Mutual Information bars
                fig.add_trace(go.Bar(
                    x=other_numeric,
                    y=mi_scores,
                    name='Mutual Information',
                    marker_color='lightblue',
                    opacity=0.7
                ))
                
                # Correlation line
                fig.add_trace(go.Scatter(
                    x=other_numeric,
                    y=corr_scores,
                    mode='lines+markers',
                    name='Correlation',
                    line=dict(color='red'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title=f'Feature Importance for {target_col}',
                    xaxis_title='Features',
                    yaxis_title='Mutual Information',
                    yaxis2=dict(
                        title='Correlation',
                        overlaying='y',
                        side='right'
                    ),
                    template='plotly_white'
                )
                
                charts.append({
                    'type': 'feature_importance',
                    'target': target_col,
                    'title': f'Feature Importance: {target_col}',
                    'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
                })
            
            except Exception as e:
                print(f"Error creating feature importance chart for {target_col}: {e}")
                continue
        
        return charts

    def _create_distribution_charts_advanced(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced distribution analysis charts"""
        charts = []
        
        # 1. Distribution Comparison Dashboard
        if len(numeric_cols) >= 2:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram Overlay', 'Q-Q Plots', 'Density Plots', 'Box Plot Comparison'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Select top 4 numeric columns for comparison
            cols_to_compare = numeric_cols[:4]
            
            # Histogram overlay
            for i, col in enumerate(cols_to_compare):
                data = df[col].dropna()
                fig.add_trace(
                    go.Histogram(x=data, name=col, opacity=0.6, 
                               marker_color=self.chart_colors[i % len(self.chart_colors)]),
                    row=1, col=1
                )
            
            # Q-Q plots (comparing to normal distribution)
            for i, col in enumerate(cols_to_compare[:2]):  # Limit to 2 for clarity
                data = df[col].dropna()
                if len(data) > 10:
                    try:
                        qq_data = stats.probplot(data, dist="norm")
                        fig.add_trace(
                            go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                                     mode='markers', name=f'{col} Q-Q',
                                     marker_color=self.chart_colors[i]),
                            row=1, col=2
                        )
                    except:
                        pass
            
            # Density plots
            for i, col in enumerate(cols_to_compare):
                data = df[col].dropna()
                if len(data) > 10:
                    try:
                        # Create density using histogram
                        hist, bin_edges = np.histogram(data, bins=30, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        fig.add_trace(
                            go.Scatter(x=bin_centers, y=hist, mode='lines',
                                     name=f'{col} Density',
                                     line=dict(color=self.chart_colors[i % len(self.chart_colors)])),
                            row=2, col=1
                        )
                    except:
                        pass
            
            # Box plot comparison
            for i, col in enumerate(cols_to_compare):
                data = df[col].dropna()
                fig.add_trace(
                    go.Box(y=data, name=col, 
                          marker_color=self.chart_colors[i % len(self.chart_colors)]),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Distribution Analysis Dashboard',
                template='plotly_white',
                height=800,
                showlegend=True
            )
            
            charts.append({
                'type': 'distribution_dashboard',
                'title': 'Distribution Analysis Dashboard',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
    def _read_data_from_file(self, file_path: str, file_type: str) -> Optional[pd.DataFrame]:
        """Read data from file path"""
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path)
            elif file_type in ['xlsx', 'xls']:
                return pd.read_excel(file_path)
            elif file_type == 'json':
                return pd.read_json(file_path)
            elif file_type == 'parquet':
                return pd.read_parquet(file_path)
            else:
                return None
        except Exception:
            return None

    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value"""
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif skewness < -0.5:
            return "Left-skewed (negatively skewed)"
        else:
            return "Right-skewed (positively skewed)"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value"""
        if abs(kurtosis) < 0.5:
            return "Mesokurtic (normal-like)"
        elif kurtosis < -0.5:
            return "Platykurtic (flatter than normal)"
        else:
            return "Leptokurtic (more peaked than normal)"

    def _analyze_categorical_advanced(self, df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Any]:
        """Advanced categorical analysis with entropy and association measures"""
        analysis = {}
        
        for col in categorical_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            value_counts = series.value_counts()
            
            # Basic statistics
            basic_stats = {
                'count': len(series),
                'unique_values': len(value_counts),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
            }
            
            # Cardinality analysis
            cardinality_ratio = len(value_counts) / len(series)
            cardinality_type = self._classify_cardinality(cardinality_ratio, len(value_counts))
            
            # Entropy calculation
            probabilities = value_counts / len(series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Distribution analysis
            distribution_stats = {
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'concentration_ratio': float(value_counts.iloc[0] / len(series)),  # Top category ratio
                'top_10_coverage': float(value_counts.head(10).sum() / len(series)),
                'cardinality_ratio': cardinality_ratio,
                'cardinality_type': cardinality_type
            }
            
            # Mode analysis
            mode_info = {
                'mode': str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                'mode_frequency': int(value_counts.max()),
                'mode_percentage': float(value_counts.max() / len(series) * 100)
            }
            
            analysis[col] = {
                **basic_stats,
                'distribution': distribution_stats,
                'mode_analysis': mode_info,
                'top_values': {str(k): int(v) for k, v in value_counts.head(20).items()},
                'value_length_stats': {
                    'avg_length': float(series.astype(str).str.len().mean()),
                    'max_length': int(series.astype(str).str.len().max()),
                    'min_length': int(series.astype(str).str.len().min())
                } if series.dtype == 'object' else None
            }
        
        return analysis

    def _classify_cardinality(self, ratio: float, unique_count: int) -> str:
        """Classify cardinality level"""
        if ratio > 0.9:
            return "High cardinality (mostly unique)"
        elif ratio > 0.5:
            return "Medium-high cardinality"
        elif ratio > 0.1:
            return "Medium cardinality"
        elif unique_count < 10:
            return "Low cardinality"
        else:
            return "Medium-low cardinality"
    
    def _analyze_feature_relationships(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Analyze relationships between features using various correlation methods"""
        relationships = {}
        
        # Numeric-Numeric correlations
        if len(numeric_cols) > 1:
            numeric_df = df[numeric_cols].dropna()
            
            # Pearson correlation
            pearson_corr = numeric_df.corr(method='pearson')
            
            # Spearman correlation (rank-based)
            spearman_corr = numeric_df.corr(method='spearman')
            
            # Find strongest correlations
            strong_correlations = []
            for i in range(len(pearson_corr.columns)):
                for j in range(i+1, len(pearson_corr.columns)):
                    corr_val = pearson_corr.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        strong_correlations.append({
                            'feature1': pearson_corr.columns[i],
                            'feature2': pearson_corr.columns[j],
                            'pearson_correlation': float(corr_val),
                            'spearman_correlation': float(spearman_corr.iloc[i, j]),
                            'strength': self._interpret_correlation_strength(abs(corr_val))
                        })
            
            relationships['numeric_correlations'] = {
                'pearson_matrix': pearson_corr.to_dict(),
                'spearman_matrix': spearman_corr.to_dict(),
                'strong_correlations': strong_correlations[:20]  # Top 20
            }
        
        # Categorical-Categorical associations (Cramér's V)
        if len(categorical_cols) > 1:
            categorical_associations = []
            for i in range(len(categorical_cols)):
                for j in range(i+1, len(categorical_cols)):
                    try:
                        col1, col2 = categorical_cols[i], categorical_cols[j]
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        
                        categorical_associations.append({
                            'feature1': col1,
                            'feature2': col2,
                            'cramers_v': float(cramers_v),
                            'chi2_statistic': float(chi2),
                            'p_value': float(p_val),
                            'association_strength': self._interpret_cramers_v(cramers_v)
                        })
                    except:
                        continue
            
            relationships['categorical_associations'] = sorted(
                categorical_associations, key=lambda x: x['cramers_v'], reverse=True
            )[:20]
        
        # Mixed type relationships (ANOVA for categorical-numeric)
        if numeric_cols and categorical_cols:
            mixed_relationships = []
            for num_col in numeric_cols[:10]:  # Limit for performance
                for cat_col in categorical_cols[:10]:
                    try:
                        groups = [df[df[cat_col] == cat][num_col].dropna() for cat in df[cat_col].unique()]
                        groups = [g for g in groups if len(g) > 0]
                        
                        if len(groups) > 1:
                            f_stat, p_val = stats.f_oneway(*groups)
                            eta_squared = self._calculate_eta_squared(df, num_col, cat_col)
                            
                            mixed_relationships.append({
                                'numeric_feature': num_col,
                                'categorical_feature': cat_col,
                                'f_statistic': float(f_stat),
                                'p_value': float(p_val),
                                'eta_squared': float(eta_squared),
                                'effect_size': self._interpret_eta_squared(eta_squared)
                            })
                    except:
                        continue
            
            relationships['mixed_type_relationships'] = sorted(
                mixed_relationships, key=lambda x: x['eta_squared'], reverse=True
            )[:20]
        
        return relationships

    def _interpret_correlation_strength(self, corr: float) -> str:
        """Interpret correlation strength"""
        if corr >= 0.9:
            return "Very strong"
        elif corr >= 0.7:
            return "Strong"
        elif corr >= 0.5:
            return "Moderate"
        elif corr >= 0.3:
            return "Weak"
        else:
            return "Very weak"

    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramér's V strength"""
        if cramers_v >= 0.5:
            return "Strong association"
        elif cramers_v >= 0.3:
            return "Moderate association"
        elif cramers_v >= 0.1:
            return "Weak association"
        else:
            return "Very weak association"

    def _calculate_eta_squared(self, df: pd.DataFrame, numeric_col: str, categorical_col: str) -> float:
        """Calculate eta squared (effect size for ANOVA)"""
        try:
            groups = [df[df[categorical_col] == cat][numeric_col].dropna() for cat in df[categorical_col].unique()]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) <= 1:
                return 0.0
            
            overall_mean = df[numeric_col].mean()
            ss_total = ((df[numeric_col] - overall_mean) ** 2).sum()
            
            ss_between = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            return min(1.0, max(0.0, eta_squared))
        except:
            return 0.0

    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta squared effect size"""
        if eta_squared >= 0.14:
            return "Large effect"
        elif eta_squared >= 0.06:
            return "Medium effect"
        elif eta_squared >= 0.01:
            return "Small effect"
        else:
            return "Negligible effect"

    def _detect_outliers_advanced(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Advanced outlier detection using multiple methods"""
        outlier_results = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:  # Need minimum data points
                continue
            
            outlier_info = {'column': col, 'methods': {}}
            
            # Method 1: IQR Method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            iqr_lower = Q1 - 1.5 * IQR
            iqr_upper = Q3 + 1.5 * IQR
            iqr_outliers = series[(series < iqr_lower) | (series > iqr_upper)]
            
            outlier_info['methods']['iqr'] = {
                'outlier_count': len(iqr_outliers),
                'outlier_percentage': (len(iqr_outliers) / len(series)) * 100,
                'bounds': {'lower': float(iqr_lower), 'upper': float(iqr_upper)}
            }
            
            # Method 2: Z-Score Method
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]
            
            outlier_info['methods']['zscore'] = {
                'outlier_count': len(z_outliers),
                'outlier_percentage': (len(z_outliers) / len(series)) * 100,
                'threshold': 3.0
            }
            
            # Method 3: Isolation Forest (for anomaly detection)
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                iso_outliers = series[outlier_labels == -1]
                
                outlier_info['methods']['isolation_forest'] = {
                    'outlier_count': len(iso_outliers),
                    'outlier_percentage': (len(iso_outliers) / len(series)) * 100,
                    'contamination': 0.1
                }
            except:
                outlier_info['methods']['isolation_forest'] = {'error': 'Could not compute'}
            
            # Method 4: Modified Z-Score (using median)
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad if mad > 0 else np.zeros_like(series)
            modified_z_outliers = series[np.abs(modified_z_scores) > 3.5]
            
            outlier_info['methods']['modified_zscore'] = {
                'outlier_count': len(modified_z_outliers),
                'outlier_percentage': (len(modified_z_outliers) / len(series)) * 100,
                'threshold': 3.5
            }
            
            # Summary of consensus outliers (appearing in multiple methods)
            all_outlier_indices = set()
            if len(iqr_outliers) > 0:
                all_outlier_indices.update(iqr_outliers.index)
            if len(z_outliers) > 0:
                all_outlier_indices.update(z_outliers.index)
            if len(modified_z_outliers) > 0:
                all_outlier_indices.update(modified_z_outliers.index)
            
            outlier_info['summary'] = {
                'total_unique_outliers': len(all_outlier_indices),
                'consensus_recommendation': self._get_outlier_recommendation(outlier_info['methods'])
            }
            
            outlier_results[col] = outlier_info
        
        return outlier_results

    def _get_outlier_recommendation(self, methods: Dict[str, Any]) -> str:
        """Get recommendation based on outlier detection results"""
        avg_percentage = np.mean([m.get('outlier_percentage', 0) for m in methods.values() if isinstance(m, dict) and 'outlier_percentage' in m])
        
        if avg_percentage > 10:
            return "High outlier rate - investigate data quality"
        elif avg_percentage > 5:
            return "Moderate outlier rate - consider outlier treatment"
        elif avg_percentage > 1:
            return "Low outlier rate - outliers may be valid extreme values"
        else:
            return "Very low outlier rate - data appears clean"
    
    def _analyze_dimensionality(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze dimensionality and perform PCA analysis"""
        if len(numeric_cols) < 3:
            return {'error': 'Need at least 3 numeric columns for dimensionality analysis'}
        
        try:
            # Prepare data
            data = df[numeric_cols].dropna()
            if len(data) < 10:
                return {'error': 'Insufficient data for dimensionality analysis'}
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(data_scaled)
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # Find number of components for different variance thresholds
            components_80 = np.argmax(cumulative_variance >= 0.8) + 1
            components_90 = np.argmax(cumulative_variance >= 0.9) + 1
            components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            # Feature importance in first few components
            feature_importance = {}
            for i in range(min(5, len(pca.components_))):
                component_name = f'PC{i+1}'
                feature_importance[component_name] = {
                    col: float(abs(pca.components_[i][j])) 
                    for j, col in enumerate(numeric_cols)
                }
            
            return {
                'total_features': len(numeric_cols),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': cumulative_variance.tolist(),
                'components_for_variance': {
                    '80_percent': int(components_80),
                    '90_percent': int(components_90),
                    '95_percent': int(components_95)
                },
                'feature_importance_by_component': feature_importance,
                'dimensionality_recommendation': self._get_dimensionality_recommendation(
                    len(numeric_cols), components_80, components_90
                ),
                'principal_components': pca_result[:, :5].tolist() if len(pca_result) > 0 else []  # First 5 PCs
            }
        except Exception as e:
            return {'error': f'Dimensionality analysis failed: {str(e)}'}

    def _get_dimensionality_recommendation(self, total_features: int, comp_80: int, comp_90: int) -> str:
        """Get recommendation for dimensionality reduction"""
        if comp_80 < total_features * 0.3:
            return f"Strong dimensionality reduction potential: {comp_80} components explain 80% variance"
        elif comp_90 < total_features * 0.5:
            return f"Moderate dimensionality reduction potential: {comp_90} components explain 90% variance"
        else:
            return "Low dimensionality reduction potential - most features are informative"

    def _analyze_clustering(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Perform clustering analysis to identify natural groupings"""
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns for clustering analysis'}
        
        try:
            # Prepare data
            data = df[numeric_cols].dropna()
            if len(data) < 10:
                return {'error': 'Insufficient data for clustering analysis'}
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Determine optimal number of clusters using elbow method
            max_clusters = min(10, len(data) // 2)
            inertias = []
            silhouette_scores = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data_scaled)
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score
                try:
                    from sklearn.metrics import silhouette_score
                    sil_score = silhouette_score(data_scaled, cluster_labels)
                    silhouette_scores.append(sil_score)
                except:
                    silhouette_scores.append(0)
            
            # Find optimal number of clusters
            optimal_k = self._find_optimal_clusters(inertias, silhouette_scores)
            
            # Perform final clustering with optimal k
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_clusters = kmeans_final.fit_predict(data_scaled)
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in range(optimal_k):
                cluster_mask = final_clusters == cluster_id
                cluster_data = data[cluster_mask]
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': int(cluster_mask.sum()),
                    'percentage': float(cluster_mask.sum() / len(data) * 100),
                    'centroid': cluster_data.mean().to_dict(),
                    'characteristics': self._describe_cluster(cluster_data, data, numeric_cols)
                }
            
            return {
                'optimal_clusters': int(optimal_k),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'cluster_analysis': cluster_analysis,
                'cluster_labels': final_clusters.tolist(),
                'clustering_quality': float(np.mean(silhouette_scores)) if silhouette_scores else 0,
                'recommendation': self._get_clustering_recommendation(optimal_k, silhouette_scores)
            }
        except Exception as e:
            return {'error': f'Clustering analysis failed: {str(e)}'}

    def _find_optimal_clusters(self, inertias: List[float], silhouette_scores: List[float]) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        if not silhouette_scores:
            return 3  # Default
        
        # Find the k with highest silhouette score
        best_silhouette_k = np.argmax(silhouette_scores) + 2
        
        # If silhouette score is reasonable, use it
        if silhouette_scores[best_silhouette_k - 2] > 0.3:
            return best_silhouette_k
        
        # Otherwise, use elbow method
        if len(inertias) >= 3:
            # Calculate second derivative to find elbow
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow_k = np.argmax(diffs2) + 3  # +3 because we start from k=2 and lose 2 points in double diff
            return min(elbow_k, len(inertias) + 1)
        
        return 3  # Default fallback

    def _describe_cluster(self, cluster_data: pd.DataFrame, full_data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, str]:
        """Describe characteristics of a cluster"""
        characteristics = []
        
        for col in numeric_cols:
            cluster_mean = cluster_data[col].mean()
            full_mean = full_data[col].mean()
            
            if cluster_mean > full_mean * 1.2:
                characteristics.append(f"High {col}")
            elif cluster_mean < full_mean * 0.8:
                characteristics.append(f"Low {col}")
        
        return characteristics[:3]  # Top 3 characteristics

    def _get_clustering_recommendation(self, optimal_k: int, silhouette_scores: List[float]) -> str:
        """Get clustering recommendation"""
        avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0
        
        if avg_silhouette > 0.7:
            return f"Excellent clustering: {optimal_k} distinct clusters identified"
        elif avg_silhouette > 0.5:
            return f"Good clustering: {optimal_k} clusters with clear separation"
        elif avg_silhouette > 0.3:
            return f"Moderate clustering: {optimal_k} clusters with some overlap"
        else:
            return f"Weak clustering: Data may not have natural cluster structure"
    
    def _analyze_time_series_advanced(self, df: pd.DataFrame, datetime_cols: List[str], numeric_cols: List[str]) -> Dict[str, Any]:
        """Advanced time series analysis with trend and seasonality detection"""
        time_series_analysis = {}
        
        for date_col in datetime_cols:
            for num_col in numeric_cols[:5]:  # Limit for performance
                ts_data = df[[date_col, num_col]].dropna().sort_values(date_col)
                if len(ts_data) < 10:
                    continue
                
                key = f"{date_col}__{num_col}"
                
                # Basic time series statistics
                time_span = ts_data[date_col].max() - ts_data[date_col].min()
                frequency = self._detect_frequency(ts_data[date_col])
                
                # Trend analysis
                trend_analysis = self._analyze_trend(ts_data, date_col, num_col)
                
                # Seasonality detection (if enough data points)
                seasonality_analysis = {}
                if len(ts_data) > 24:  # Need sufficient data for seasonality
                    seasonality_analysis = self._detect_seasonality(ts_data, date_col, num_col)
                
                time_series_analysis[key] = {
                    'data_points': len(ts_data),
                    'time_span_days': time_span.days,
                    'frequency': frequency,
                    'trend': trend_analysis,
                    'seasonality': seasonality_analysis,
                    'start_date': str(ts_data[date_col].min()),
                    'end_date': str(ts_data[date_col].max())
                }
        
        return time_series_analysis

    def _detect_frequency(self, date_series: pd.Series) -> str:
        """Detect the frequency of time series data"""
        try:
            time_diffs = date_series.diff().dropna()
            most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            
            if most_common_diff.days == 1:
                return "Daily"
            elif most_common_diff.days == 7:
                return "Weekly"
            elif 28 <= most_common_diff.days <= 31:
                return "Monthly"
            elif 85 <= most_common_diff.days <= 95:
                return "Quarterly"
            elif 360 <= most_common_diff.days <= 370:
                return "Yearly"
            else:
                return f"Irregular ({most_common_diff.days} days avg)"
        except:
            return "Unknown"

    def _analyze_trend(self, ts_data: pd.DataFrame, date_col: str, num_col: str) -> Dict[str, Any]:
        """Analyze trend in time series data"""
        try:
            # Convert dates to numeric for regression
            ts_data = ts_data.copy()
            ts_data['date_numeric'] = (ts_data[date_col] - ts_data[date_col].min()).dt.days
            
            # Linear regression for trend
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(
                ts_data['date_numeric'], ts_data[num_col]
            )
            
            trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
            trend_strength = abs(r_value)
            
            return {
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'trend_direction': trend_direction,
                'trend_strength': self._interpret_trend_strength(trend_strength),
                'is_significant': p_value < 0.05
            }
        except:
            return {'error': 'Could not analyze trend'}

    def _interpret_trend_strength(self, r_value: float) -> str:
        """Interpret trend strength based on correlation coefficient"""
        if r_value >= 0.8:
            return "Very strong"
        elif r_value >= 0.6:
            return "Strong"
        elif r_value >= 0.4:
            return "Moderate"
        elif r_value >= 0.2:
            return "Weak"
        else:
            return "Very weak"

    def _detect_seasonality(self, ts_data: pd.DataFrame, date_col: str, num_col: str) -> Dict[str, Any]:
        """Detect seasonality patterns in time series data"""
        try:
            ts_data = ts_data.copy()
            ts_data['month'] = ts_data[date_col].dt.month
            ts_data['day_of_week'] = ts_data[date_col].dt.dayofweek
            ts_data['quarter'] = ts_data[date_col].dt.quarter
            
            # Monthly seasonality
            monthly_means = ts_data.groupby('month')[num_col].mean()
            monthly_cv = monthly_means.std() / monthly_means.mean() if monthly_means.mean() != 0 else 0
            
            # Weekly seasonality
            weekly_means = ts_data.groupby('day_of_week')[num_col].mean()
            weekly_cv = weekly_means.std() / weekly_means.mean() if weekly_means.mean() != 0 else 0
            
            # Quarterly seasonality
            quarterly_means = ts_data.groupby('quarter')[num_col].mean()
            quarterly_cv = quarterly_means.std() / quarterly_means.mean() if quarterly_means.mean() != 0 else 0
            
            return {
                'monthly_seasonality': {
                    'coefficient_of_variation': float(monthly_cv),
                    'strength': self._interpret_seasonality_strength(monthly_cv),
                    'pattern': monthly_means.to_dict()
                },
                'weekly_seasonality': {
                    'coefficient_of_variation': float(weekly_cv),
                    'strength': self._interpret_seasonality_strength(weekly_cv),
                    'pattern': weekly_means.to_dict()
                },
                'quarterly_seasonality': {
                    'coefficient_of_variation': float(quarterly_cv),
                    'strength': self._interpret_seasonality_strength(quarterly_cv),
                    'pattern': quarterly_means.to_dict()
                }
            }
        except:
            return {'error': 'Could not detect seasonality'}

    def _interpret_seasonality_strength(self, cv: float) -> str:
        """Interpret seasonality strength based on coefficient of variation"""
        if cv >= 0.3:
            return "Strong"
        elif cv >= 0.15:
            return "Moderate"
        elif cv >= 0.05:
            return "Weak"
        else:
            return "None"

    def _analyze_feature_importance(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Analyze feature importance using various methods"""
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns for feature importance analysis'}
        
        try:
            feature_importance = {}
            
            # For each numeric column, calculate its relationship with other features
            for target_col in numeric_cols[:5]:  # Limit to first 5 as targets
                other_numeric = [col for col in numeric_cols if col != target_col]
                
                if len(other_numeric) == 0:
                    continue
                
                # Prepare data
                data = df[other_numeric + [target_col]].dropna()
                if len(data) < 10:
                    continue
                
                X = data[other_numeric]
                y = data[target_col]
                
                # Mutual information for numeric features
                try:
                    mi_scores = mutual_info_regression(X, y, random_state=42)
                    mi_importance = {other_numeric[i]: float(mi_scores[i]) for i in range(len(other_numeric))}
                except:
                    mi_importance = {}
                
                # Correlation-based importance
                corr_importance = {}
                for col in other_numeric:
                    try:
                        corr = abs(data[col].corr(data[target_col]))
                        corr_importance[col] = float(corr) if not np.isnan(corr) else 0
                    except:
                        corr_importance[col] = 0
                
                feature_importance[target_col] = {
                    'mutual_information': mi_importance,
                    'correlation_based': corr_importance,
                    'top_features_mi': sorted(mi_importance.items(), key=lambda x: x[1], reverse=True)[:5],
                    'top_features_corr': sorted(corr_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                }
            
            return feature_importance
        except Exception as e:
            return {'error': f'Feature importance analysis failed: {str(e)}'}

    def _analyze_distributions_advanced(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Advanced distribution analysis"""
        distribution_analysis = {}
        
        # Numeric distributions
        if numeric_cols:
            numeric_distributions = {}
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) < 10:
                    continue
                
                # Distribution fit tests
                distribution_fits = self._test_distribution_fits(series)
                
                # Histogram analysis
                hist_analysis = self._analyze_histogram(series)
                
                numeric_distributions[col] = {
                    'distribution_fits': distribution_fits,
                    'histogram_analysis': hist_analysis
                }
            
            distribution_analysis['numeric'] = numeric_distributions
        
        # Categorical distributions
        if categorical_cols:
            categorical_distributions = {}
            for col in categorical_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                value_counts = series.value_counts()
                
                # Analyze distribution shape
                dist_analysis = {
                    'uniformity_score': self._calculate_uniformity_score(value_counts),
                    'concentration_score': float(value_counts.iloc[0] / len(series)),
                    'diversity_index': self._calculate_diversity_index(value_counts),
                    'top_category_dominance': float(value_counts.iloc[0] / value_counts.iloc[1]) if len(value_counts) > 1 else float('inf')
                }
                
                categorical_distributions[col] = dist_analysis
            
            distribution_analysis['categorical'] = categorical_distributions
        
        return distribution_analysis

    def _test_distribution_fits(self, series: pd.Series) -> Dict[str, Any]:
        """Test how well the data fits common distributions"""
        try:
            # Test normality
            normality_stat, normality_p = normaltest(series.sample(min(5000, len(series))))
            
            # Test for other distributions using simple heuristics
            skewness = series.skew()
            kurtosis = series.kurtosis()
            
            distribution_scores = {
                'normal': {
                    'p_value': float(normality_p),
                    'likely': normality_p > 0.05,
                    'score': 1 - min(1, abs(skewness) + abs(kurtosis - 3))
                },
                'uniform': {
                    'likely': abs(kurtosis + 1.2) < 0.5,  # Uniform has kurtosis ≈ -1.2
                    'score': 1 - abs(kurtosis + 1.2)
                },
                'exponential': {
                    'likely': skewness > 1.5 and kurtosis > 5,
                    'score': max(0, (skewness - 1.5) / 2) if skewness > 1.5 else 0
                }
            }
            
            # Find best fit
            best_fit = max(distribution_scores.keys(), 
                          key=lambda k: distribution_scores[k]['score'])
            
            return {
                'best_fit': best_fit,
                'distribution_scores': distribution_scores,
                'confidence': distribution_scores[best_fit]['score']
            }
        except:
            return {'error': 'Could not test distribution fits'}

    def _analyze_histogram(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze histogram characteristics"""
        try:
            # Create histogram
            hist, bin_edges = np.histogram(series, bins='auto')
            
            # Find peaks and valleys
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append(i)
            
            # Analyze shape
            is_multimodal = len(peaks) > 1
            is_bimodal = len(peaks) == 2
            
            return {
                'bins_used': len(bin_edges) - 1,
                'peaks_count': len(peaks),
                'is_multimodal': is_multimodal,
                'is_bimodal': is_bimodal,
                'peak_positions': [float(bin_edges[p]) for p in peaks],
                'shape_description': self._describe_histogram_shape(peaks, hist)
            }
        except:
            return {'error': 'Could not analyze histogram'}

    def _describe_histogram_shape(self, peaks: List[int], hist: np.ndarray) -> str:
        """Describe the shape of the histogram"""
        if len(peaks) == 0:
            return "Flat or unclear shape"
        elif len(peaks) == 1:
            return "Unimodal (single peak)"
        elif len(peaks) == 2:
            return "Bimodal (two peaks)"
        else:
            return f"Multimodal ({len(peaks)} peaks)"

    def _calculate_uniformity_score(self, value_counts: pd.Series) -> float:
        """Calculate how uniform a categorical distribution is"""
        expected_count = len(value_counts) / value_counts.nunique() if value_counts.nunique() > 0 else 0
        if expected_count == 0:
            return 0
        
        chi_square = sum((count - expected_count) ** 2 / expected_count for count in value_counts)
        # Normalize by degrees of freedom
        uniformity_score = 1 / (1 + chi_square / (value_counts.nunique() - 1)) if value_counts.nunique() > 1 else 1
        return float(uniformity_score)

    def _calculate_diversity_index(self, value_counts: pd.Series) -> float:
        """Calculate diversity index (similar to Shannon entropy)"""
        proportions = value_counts / value_counts.sum()
        entropy = -sum(p * np.log(p) for p in proportions if p > 0)
        max_entropy = np.log(len(value_counts)) if len(value_counts) > 1 else 1
        return float(entropy / max_entropy) if max_entropy > 0 else 0

    def _perform_statistical_tests(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Perform various statistical tests"""
        test_results = {}
        
        # Normality tests for numeric columns
        if numeric_cols:
            normality_tests = {}
            for col in numeric_cols[:10]:  # Limit for performance
                series = df[col].dropna()
                if len(series) < 8:  # Minimum for normality test
                    continue
                
                try:
                    stat, p_value = normaltest(series.sample(min(5000, len(series))))
                    normality_tests[col] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > 0.05,
                        'interpretation': 'Normal' if p_value > 0.05 else 'Not normal'
                    }
                except:
                    normality_tests[col] = {'error': 'Could not perform test'}
            
            test_results['normality_tests'] = normality_tests
        
        # Independence tests for categorical pairs
        if len(categorical_cols) >= 2:
            independence_tests = []
            for i in range(len(categorical_cols)):
                for j in range(i + 1, min(i + 6, len(categorical_cols))):  # Limit combinations
                    col1, col2 = categorical_cols[i], categorical_cols[j]
                    try:
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        if contingency_table.size < 4:  # Need at least 2x2 table
                            continue
                        
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        independence_tests.append({
                            'variable1': col1,
                            'variable2': col2,
                            'chi2_statistic': float(chi2),
                            'p_value': float(p_value),
                            'degrees_of_freedom': int(dof),
                            'are_independent': p_value > 0.05,
                            'interpretation': 'Independent' if p_value > 0.05 else 'Associated'
                        })
                    except:
                        continue
            
            test_results['independence_tests'] = independence_tests[:20]  # Top 20
        
        return test_results

    def _generate_advanced_recommendations(self, df: pd.DataFrame, eda_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate advanced recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Data quality recommendations
        quality_data = eda_results.get('data_quality', {})
        overall_quality = quality_data.get('overall_quality_score', 0)
        
        if overall_quality < 70:
            recommendations.append({
                'category': 'data_quality',
                'priority': 'high',
                'title': 'Poor Data Quality Detected',
                'description': f'Overall data quality score is {overall_quality:.1f}%. Immediate attention required.',
                'actions': ['Review missing values', 'Check for duplicates', 'Validate data consistency'],
                'impact': 'High impact on analysis reliability'
            })
        
        # Missing values recommendations
        completeness = quality_data.get('completeness', {})
        if completeness.get('score', 100) < 80:
            high_missing_cols = [col for col, count in completeness.get('missing_by_column', {}).items() 
                               if count / len(df) > 0.3]
            
            recommendations.append({
                'category': 'missing_values',
                'priority': 'medium',
                'title': 'High Missing Value Rate',
                'description': f'Completeness score: {completeness.get("score", 0):.1f}%',
                'actions': [f'Consider dropping columns: {", ".join(high_missing_cols[:3])}',
                           'Implement imputation strategies', 'Investigate missing data patterns'],
                'impact': 'Affects statistical power and model performance'
            })
        
        # Outlier recommendations
        outliers_data = eda_results.get('outliers', {})
        high_outlier_cols = []
        for col, info in outliers_data.items():
            if isinstance(info, dict) and info.get('summary', {}).get('total_unique_outliers', 0) > len(df) * 0.05:
                high_outlier_cols.append(col)
        
        if high_outlier_cols:
            recommendations.append({
                'category': 'outliers',
                'priority': 'medium',
                'title': 'High Outlier Rate Detected',
                'description': f'Columns with >5% outliers: {", ".join(high_outlier_cols[:3])}',
                'actions': ['Investigate outlier sources', 'Consider outlier treatment methods',
                           'Validate extreme values', 'Use robust statistical methods'],
                'impact': 'May skew statistical analysis and model training'
            })
        
        # Dimensionality recommendations
        dim_data = eda_results.get('dimensionality', {})
        if not dim_data.get('error') and dim_data.get('components_for_variance', {}).get('80_percent', 0) < len(df.select_dtypes(include=[np.number]).columns) * 0.5:
            recommendations.append({
                'category': 'dimensionality',
                'priority': 'low',
                'title': 'Dimensionality Reduction Opportunity',
                'description': dim_data.get('dimensionality_recommendation', ''),
                'actions': ['Apply PCA or other dimensionality reduction', 'Feature selection',
                           'Remove redundant features'],
                'impact': 'Can improve model performance and reduce computational cost'
            })
        
        # Clustering recommendations
        cluster_data = eda_results.get('clustering', {})
        if not cluster_data.get('error') and cluster_data.get('clustering_quality', 0) > 0.5:
            recommendations.append({
                'category': 'clustering',
                'priority': 'low',
                'title': 'Natural Groupings Detected',
                'description': cluster_data.get('recommendation', ''),
                'actions': ['Consider clustering-based analysis', 'Investigate cluster characteristics',
                           'Use clusters for stratified sampling'],
                'impact': 'Can reveal hidden patterns and improve segmentation analysis'
            })
        
        # Distribution recommendations
        distributions = eda_results.get('distributions', {}).get('numeric', {})
        non_normal_cols = []
        for col, dist_info in distributions.items():
            if isinstance(dist_info, dict):
                dist_fits = dist_info.get('distribution_fits', {})
                if not dist_fits.get('error') and dist_fits.get('best_fit') != 'normal':
                    non_normal_cols.append(col)
        
        if len(non_normal_cols) > len(df.select_dtypes(include=[np.number]).columns) * 0.5:
            recommendations.append({
                'category': 'distributions',
                'priority': 'medium',
                'title': 'Non-Normal Distributions Detected',
                'description': f'Most numeric variables are not normally distributed',
                'actions': ['Consider data transformations', 'Use non-parametric statistical tests',
                           'Apply robust statistical methods'],
                'impact': 'Affects choice of statistical methods and model assumptions'
            })
        
        # Feature relationships recommendations
        relationships = eda_results.get('relationships', {})
        strong_corr = relationships.get('numeric_correlations', {}).get('strong_correlations', [])
        if len(strong_corr) > 5:
            recommendations.append({
                'category': 'multicollinearity',
                'priority': 'medium',
                'title': 'High Feature Correlations Detected',
                'description': f'{len(strong_corr)} strong correlations found',
                'actions': ['Check for multicollinearity', 'Consider feature selection',
                           'Use regularization techniques', 'Apply PCA'],
                'impact': 'Can cause instability in linear models'
            })
        
        return sorted(recommendations, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)

