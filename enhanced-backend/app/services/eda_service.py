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
    """Service for advanced exploratory data analysis generation"""
    
    def __init__(self):
        self.chart_colors = px.colors.qualitative.Set3
    
    def _convert_to_serializable(self, obj):
        """Recursively convert numpy types and other non-serializable objects to JSON-serializable types"""
        import numpy as np
        from datetime import datetime, date, time
        
        # Handle None and basic types
        if obj is None:
            return None
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, float):
            # Handle NaN, infinity, and -infinity
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None if obj > 0 else None
            else:
                return obj
            
        # Handle datetime objects
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
            
        # Handle numpy numeric types
        if hasattr(np, 'integer') and isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                                                    np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
            
        if hasattr(np, 'floating') and isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            # Handle NaN and infinity
            float_val = float(obj)
            if np.isnan(float_val):
                return None
            elif np.isinf(float_val):
                return None
            else:
                return float_val
            
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

            # 9. Statistical Tests
            eda_results['statistical_tests'] = self._perform_statistical_tests(df_clean, numeric_cols, categorical_cols)

            # 10. Recommendations with Priority
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

    def _generate_comprehensive_executive_summary(self, df: pd.DataFrame, sample_note: str = None) -> str:
        """Generate a comprehensive executive summary with detailed insights"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            total_missing = df.isnull().sum().sum()
            missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
            
            # Determine dataset complexity
            complexity = "Simple"
            if len(df.columns) > 20:
                complexity = "Complex"
            elif len(df.columns) > 10:
                complexity = "Moderate"
            
            insights = []
            
            if len(categorical_cols) == 0:
                insights.append("No categorical variables detected - primarily numeric dataset")
            else:
                insights.append(f"{len(categorical_cols)} categorical variable(s) identified")
            
            if len(datetime_cols) > 0:
                insights.append(f"Time series data detected with {len(datetime_cols)} datetime column(s)")
            
            if missing_percentage > 10:
                insights.append(f"High missing data rate ({missing_percentage:.1f}%) - requires attention")
            elif missing_percentage > 0:
                insights.append(f"Minor missing data ({missing_percentage:.1f}%) detected")
            else:
                insights.append("No missing values detected - clean dataset")
            
            if len(numeric_cols) > 0:
                skew_analysis = df[numeric_cols].skew()
                highly_skewed = sum(abs(skew_analysis) > 2)
                if highly_skewed > 0:
                    insights.append(f"{highly_skewed} column(s) with high skewness detected")
            
            summary = f"""📊 **COMPREHENSIVE EDA RESULTS - ALL COLUMNS ANALYSIS**
============================================================

📋 **Dataset Overview:**
   • Total Columns: {len(df.columns)}
   • Numeric Columns: {len(numeric_cols)}
   • Categorical Columns: {len(categorical_cols)}
   • Datetime Columns: {len(datetime_cols)}
   • Total Observations: {len(df):,}
   • Missing Data: {missing_percentage:.1f}%
   • Complexity: {complexity}

🔍 **Key Insights:**
"""
            for i, insight in enumerate(insights, 1):
                summary += f"   • {insight}\n"
            
            if sample_note:
                summary += f"\n📝 **Note:** {sample_note}"
            
            return summary
            
        except Exception as e:
            return f"Error generating executive summary: {str(e)}"

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

    def _generate_comprehensive_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset overview with detailed metrics"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            overview = {
                'shape': {
                    'rows': len(df),
                    'columns': len(df.columns)
                },
                'rows': len(df),
                'columns': len(df.columns),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols),
                'datetime_columns': len(datetime_cols),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'total_missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
                'column_types': {
                    'numeric': numeric_cols,
                    'categorical': categorical_cols,
                    'datetime': datetime_cols
                }
            }
            
            # Add time range analysis if datetime columns exist
            if datetime_cols:
                time_col = datetime_cols[0]
                if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    overview['time_range'] = {
                        'start': df[time_col].min(),
                        'end': df[time_col].max(),
                        'duration_days': (df[time_col].max() - df[time_col].min()).days,
                        'frequency_detected': pd.infer_freq(df[time_col]) if df[time_col].is_monotonic_increasing else None
                    }
            
            return overview
            
        except Exception as e:
            return {'error': str(e)}

    def _generate_comprehensive_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality assessment"""
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

    def _analyze_numerical_advanced(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Advanced numerical analysis with statistical tests and distributions"""
        analysis = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            # Basic statistics
            basic_stats = {
                'count': len(series),
                'mean': float(series.mean()) if not np.isnan(series.mean()) else None,
                'std': float(series.std()) if not np.isnan(series.std()) else None,
                'min': float(series.min()) if not np.isnan(series.min()) else None,
                'max': float(series.max()) if not np.isnan(series.max()) else None,
                'median': float(series.median()) if not np.isnan(series.median()) else None,
                'q25': float(series.quantile(0.25)) if not np.isnan(series.quantile(0.25)) else None,
                'q75': float(series.quantile(0.75)) if not np.isnan(series.quantile(0.75)) else None,
                'range': float(series.max() - series.min()) if not (np.isnan(series.max()) or np.isnan(series.min())) else None,
                'iqr': float(series.quantile(0.75) - series.quantile(0.25)) if not (np.isnan(series.quantile(0.75)) or np.isnan(series.quantile(0.25))) else None,
                'cv': float(series.std() / series.mean()) if series.mean() != 0 and not (np.isnan(series.std()) or np.isnan(series.mean())) else None
            }
            
            # Distribution characteristics
            try:
                skewness = float(series.skew())
                kurtosis = float(series.kurtosis())
                
                # Handle NaN values in skewness and kurtosis
                if np.isnan(skewness):
                    skewness = None
                if np.isnan(kurtosis):
                    kurtosis = None
                
                # Normality test
                if len(series) > 8:  # Minimum for normality test
                    normality_stat, normality_p = normaltest(series.sample(min(5000, len(series))))
                    is_normal = normality_p > 0.05
                    
                    distribution_stats = {
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'skewness_interpretation': self._interpret_skewness(skewness) if skewness is not None else 'Unable to determine',
                        'kurtosis_interpretation': self._interpret_kurtosis(kurtosis) if kurtosis is not None else 'Unable to determine',
                        'normality_test': {
                            'statistic': float(normality_stat) if not np.isnan(normality_stat) else None,
                            'p_value': float(normality_p) if not np.isnan(normality_p) else None,
                            'is_normal': bool(is_normal) if not np.isnan(normality_p) else None
                        }
                    }
                else:
                    distribution_stats = {
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'skewness_interpretation': self._interpret_skewness(skewness) if skewness is not None else 'Unable to determine',
                        'kurtosis_interpretation': self._interpret_kurtosis(kurtosis) if kurtosis is not None else 'Unable to determine'
                    }
            except:
                distribution_stats = {}
            
            # Outlier detection using IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            outlier_info = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(series)) * 100,
                'method': 'IQR',
                'bounds': {'lower': float(lower_bound) if not np.isnan(lower_bound) else None, 'upper': float(upper_bound) if not np.isnan(upper_bound) else None}
            }
            
            analysis[col] = {
                **basic_stats,
                'distribution': distribution_stats,
                'outliers': outlier_info,
                'unique_values': len(series.unique()),
                'zeros_count': int((series == 0).sum()),
                'negative_count': int((series < 0).sum()),
                'positive_count': int((series > 0).sum())
            }
        
        return analysis

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
                'entropy': float(entropy) if not np.isnan(entropy) else None,
                'normalized_entropy': float(normalized_entropy) if not np.isnan(normalized_entropy) else None,
                'concentration_ratio': float(value_counts.iloc[0] / len(series)) if not np.isnan(value_counts.iloc[0] / len(series)) else None,  # Top category ratio
                'top_10_coverage': float(value_counts.head(10).sum() / len(series)) if not np.isnan(value_counts.head(10).sum() / len(series)) else None,
                'cardinality_ratio': cardinality_ratio,
                'cardinality_type': cardinality_type
            }
            
            # Mode analysis
            mode_info = {
                'mode': str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                'mode_frequency': int(value_counts.max()),
                'mode_percentage': float(value_counts.max() / len(series) * 100) if not np.isnan(value_counts.max() / len(series) * 100) else None
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
            
            # Summary
            outlier_info['summary'] = {
                'total_unique_outliers': len(set(iqr_outliers.index).union(set(z_outliers.index))),
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

    def _perform_statistical_tests(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Perform statistical tests"""
        tests = {}
        
        # Normality tests for numeric columns
        if numeric_cols:
            normality_tests = {}
            for col in numeric_cols[:10]:  # Limit for performance
                series = df[col].dropna()
                if len(series) > 8:
                    try:
                        stat, p_value = normaltest(series.sample(min(5000, len(series))))
                        normality_tests[col] = {
                            'statistic': float(stat) if not np.isnan(stat) else None,
                            'p_value': float(p_value) if not np.isnan(p_value) else None,
                            'is_normal': bool(p_value > 0.05) if not np.isnan(p_value) else None,
                            'interpretation': 'Normal distribution' if not np.isnan(p_value) and p_value > 0.05 else 'Non-normal distribution' if not np.isnan(p_value) else 'Unable to determine'
                        }
                    except:
                        continue
            
            tests['normality_tests'] = normality_tests
        
        return tests

    def _generate_advanced_recommendations(self, df: pd.DataFrame, eda_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate advanced data improvement recommendations"""
        recommendations = []
        
        # Data Quality Recommendations
        data_quality = eda_results.get('data_quality', {})
        overall_score = data_quality.get('overall_quality_score', 100)
        
        if overall_score < 70:
            recommendations.append({
                'priority': 'high',
                'title': 'Data Quality Issues Detected',
                'description': f'Overall data quality score is {overall_score:.1f}%. This indicates significant quality issues that should be addressed.',
                'actions': [
                    'Review and clean missing values',
                    'Check for data entry errors',
                    'Validate data consistency',
                    'Remove or fix duplicate records'
                ],
                'impact': 'Improving data quality will enhance analysis reliability and model performance'
            })
        
        # Missing Values Recommendations
        completeness = data_quality.get('completeness', {})
        if completeness.get('score', 100) < 90:
            missing_cols = [col for col, count in completeness.get('missing_by_column', {}).items() if count > 0]
            recommendations.append({
                'priority': 'medium',
                'title': 'Handle Missing Values',
                'description': f'Dataset has missing values in {len(missing_cols)} columns.',
                'actions': [
                    'Consider imputation strategies for important variables',
                    'Drop columns with excessive missing values (>50%)',
                    'Use domain knowledge to fill missing values where appropriate'
                ],
                'impact': 'Proper handling of missing values improves analysis completeness'
            })
        
        # Outlier Recommendations
        outliers = eda_results.get('outliers', {})
        high_outlier_cols = [col for col, info in outliers.items() 
                            if any(method.get('outlier_percentage', 0) > 5 
                                  for method in info.get('methods', {}).values())]
        
        if high_outlier_cols:
            recommendations.append({
                'priority': 'medium',
                'title': 'Review Outliers',
                'description': f'Found potential outliers in {len(high_outlier_cols)} columns: {", ".join(high_outlier_cols[:3])}{"..." if len(high_outlier_cols) > 3 else ""}',
                'actions': [
                    'Investigate outliers to determine if they are valid or errors',
                    'Consider robust statistical methods if outliers are valid',
                    'Remove or transform outliers if they are data errors'
                ],
                'impact': 'Proper outlier handling improves statistical analysis accuracy'
            })
        
        # Feature Engineering Recommendations
        numeric_analysis = eda_results.get('numerical_analysis', {})
        skewed_features = [col for col, analysis in numeric_analysis.items()
                          if analysis.get('distribution', {}).get('skewness_interpretation') in ['Left-skewed (negatively skewed)', 'Right-skewed (positively skewed)']]
        
        if skewed_features:
            recommendations.append({
                'priority': 'low',
                'title': 'Consider Feature Transformations',
                'description': f'Found {len(skewed_features)} skewed features that might benefit from transformation.',
                'actions': [
                    'Apply log transformation for right-skewed features',
                    'Consider square root or Box-Cox transformations',
                    'Use robust scalers for skewed distributions'
                ],
                'impact': 'Feature transformations can improve model performance and interpretability'
            })
        
        return recommendations

    def _create_data_quality_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create data quality visualization charts"""
        charts = []
        
        # Missing values chart
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            missing_data = missing_counts[missing_counts > 0]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    marker_color=self.chart_colors[0]
                )
            ])
            
            fig.update_layout(
                title='Missing Values by Column',
                xaxis_title='Columns',
                yaxis_title='Missing Count',
                template='plotly_white'
            )
            
            charts.append({
                'title': 'Missing Values Distribution',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        return charts

    def _create_advanced_numerical_charts(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced numerical analysis charts"""
        charts = []
        
        # Distribution charts with normality overlay
        for i, col in enumerate(numeric_cols[:6]):  # Limit to first 6 columns
            series = df[col].dropna()
            
            # Create subplot with histogram and normal overlay
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Distribution', 'Box Plot', 'Q-Q Plot', 'Time Series'],
                specs=[[{'type': 'xy'}, {'type': 'xy'}],
                       [{'type': 'xy'}, {'type': 'xy'}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=series, name='Distribution', nbinsx=30),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=series, name='Box Plot'),
                row=1, col=2
            )
            
            # Q-Q plot (simplified)
            try:
                from scipy.stats import probplot
                (osm, osr), (slope, intercept, r) = probplot(series, dist='norm', plot=None)
                fig.add_trace(
                    go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', name='Normal Line'),
                    row=2, col=1
                )
            except:
                pass
            
            # Time series (if data has index)
            if len(series) > 1:
                fig.add_trace(
                    go.Scatter(x=series.index, y=series.values, mode='lines', name='Time Series'),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=600,
                title=f'Advanced Analysis: {col}',
                template='plotly_white',
                showlegend=False
            )
            
            charts.append({
                'title': f'Advanced Analysis: {col}',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # Replace NaN values with 0 for visualization
            corr_matrix = corr_matrix.fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={'size': 10}
            ))
            
            fig.update_layout(
                title='Correlation Matrix',
                template='plotly_white'
            )
            
            charts.append({
                'title': 'Correlation Matrix',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        return charts

    def _create_advanced_categorical_charts(self, df: pd.DataFrame, categorical_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced categorical analysis charts"""
        charts = []
        
        for i, col in enumerate(categorical_cols[:6]):  # Limit to first 6 columns
            value_counts = df[col].value_counts().head(20)  # Top 20 values
            
            # Create pie chart for top categories
            fig = go.Figure(data=[
                go.Pie(
                    labels=value_counts.index,
                    values=value_counts.values,
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title=f'Distribution of {col}',
                template='plotly_white'
            )
            
            charts.append({
                'title': f'Distribution of {col}',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        return charts

    def _create_relationship_charts(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[Dict[str, Any]]:
        """Create relationship analysis charts"""
        charts = []
        
        # Scatter matrix for top numeric variables
        if len(numeric_cols) >= 2:
            cols_to_plot = numeric_cols[:4]  # Limit for performance
            
            fig = px.scatter_matrix(
                df[cols_to_plot].dropna(),
                dimensions=cols_to_plot,
                title='Feature Relationships Matrix'
            )
            
            fig.update_layout(template='plotly_white')
            
            charts.append({
                'title': 'Feature Relationships Matrix',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        return charts

    def _create_advanced_outlier_charts(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Create advanced outlier detection charts"""
        charts = []
        
        for i, col in enumerate(numeric_cols[:6]):  # Limit to first 6 columns
            series = df[col].dropna()
            
            # Create combined outlier detection chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Box Plot with Outliers', 'Outlier Detection Methods'],
                specs=[[{'type': 'xy'}, {'type': 'xy'}]]
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=series, name=col, boxpoints='outliers'),
                row=1, col=1
            )
            
            # Outlier detection comparison
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            outliers_iqr = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
            
            # Scatter plot with outliers highlighted
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(series))),
                    y=series,
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=4)
                ),
                row=1, col=2
            )
            
            if len(outliers_iqr) > 0:
                outlier_indices = [i for i, val in enumerate(series) if val in outliers_iqr.values]
                fig.add_trace(
                    go.Scatter(
                        x=outlier_indices,
                        y=outliers_iqr,
                        mode='markers',
                        name='Outliers',
                        marker=dict(color='red', size=6)
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                height=400,
                title=f'Outlier Analysis: {col}',
                template='plotly_white'
            )
            
            charts.append({
                'title': f'Outlier Analysis: {col}',
                'chart': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            })
        
        return charts

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

    def analyze_dataframe(self, df: pd.DataFrame, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a DataFrame directly for workflow nodes.
        
        Parameters:
        - df: pandas DataFrame to analyze
        - parameters: dictionary containing analysis parameters
        """
        try:
            if parameters is None:
                parameters = {}
            
            # Get analysis parameters
            analysis_depth = parameters.get('analysis_depth', 'detailed')
            include_distributions = parameters.get('include_distributions', True)
            include_correlations = parameters.get('include_correlations', True)
            max_categorical_levels = parameters.get('max_categorical_levels', 20)
            generate_recommendations = parameters.get('generate_recommendations', True)
            
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

            # 1. Advanced Executive Summary with comprehensive insights
            eda_results['executive_summary'] = self._generate_comprehensive_executive_summary(df_clean, sample_note)

            # 2. Enhanced Dataset Overview with detailed metrics
            eda_results['overview'] = self._generate_comprehensive_overview(df_clean)

            # 3. Comprehensive Data Quality Assessment
            eda_results['data_quality'] = self._generate_comprehensive_data_quality(df_clean)

            # 4. Detailed Statistical Analysis for all columns
            eda_results['statistical_insights'] = self._generate_basic_statistics(df_clean)

            # 5. Advanced correlation analysis if requested
            if include_correlations:
                eda_results['correlations'] = self._generate_basic_correlations(df_clean)

            # 6. Distribution analysis for all numeric columns (placeholder)
            eda_results['distribution_analysis'] = {'note': 'Distribution analysis integrated in statistical insights'}

            # 7. Time series analysis if applicable (placeholder)
            eda_results['time_analysis'] = {'note': 'Time series analysis will be added in future updates'}

            # 8. Advanced recommendations
            if generate_recommendations:
                eda_results['recommendations'] = self._generate_advanced_recommendations(df_clean, eda_results)

            # 9. Generate advanced charts with multiple visualization types
            try:
                eda_results['charts'] = self._generate_basic_charts(df_clean)
            except Exception as e:
                print(f"Warning: Could not generate charts: {e}")
                eda_results['charts'] = {'error': str(e)}

            return {
                'success': True,
                'results': eda_results,
                'metadata': {
                    'analysis_type': 'comprehensive_eda',
                    'sample_note': sample_note,
                    'parameters_used': parameters,
                    'dataset_shape': df.shape,
                    'original_shape': df.shape,
                    'charts_generated': len(eda_results.get('charts', {}))
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': None
            }

    def _generate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistical insights"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            insights = {
                'numeric_summary': {},
                'categorical_summary': {},
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            
            # Numeric columns statistics
            if numeric_cols:
                insights['numeric_summary'] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns statistics
            if categorical_cols:
                for col in categorical_cols:
                    insights['categorical_summary'][col] = {
                        'unique_values': int(df[col].nunique()),
                        'most_common': df[col].value_counts().head(5).to_dict()
                    }
            
            return insights
        except Exception as e:
            return {'error': str(e)}

    def _generate_basic_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic correlation analysis"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {'correlation_matrix': None, 'note': 'Insufficient numeric columns for correlation analysis'}
            
            correlation_matrix = df[numeric_cols].corr()
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': self._find_high_correlations(correlation_matrix),
                'numeric_columns_analyzed': numeric_cols
            }
        except Exception as e:
            return {'error': str(e)}

    def _find_high_correlations(self, corr_matrix, threshold=0.7):
        """Find pairs of variables with high correlation"""
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        return high_corr_pairs

    def _generate_basic_charts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic charts for EDA with insights"""
        try:
            charts = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Create a simple correlation heatmap
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix.values, 
                               x=corr_matrix.columns, 
                               y=corr_matrix.columns,
                               title="Correlation Heatmap",
                               color_continuous_scale='RdBu_r')
                
                # Generate insight for correlation heatmap
                strong_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.7:
                            strong_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                correlation_insight = f"""
📊 **Correlation Heatmap Analysis**

**Key Relationships:**
• Variables analyzed: {len(numeric_cols)}
• Strong correlations (>0.7): {len(strong_corr_pairs)}
• Color intensity shows correlation strength

**Interpretation:**
• Red colors indicate positive correlations
• Blue colors indicate negative correlations
• Darker colors represent stronger relationships

**Key Insights:**
{"• Strong correlations detected - consider multicollinearity when building models" if len(strong_corr_pairs) > 0 else "• No strong correlations found - variables are relatively independent"}
• Use correlation patterns to guide feature selection and understand data relationships
• Strong correlations can indicate redundant information or related business processes

**Next Steps:**
• Investigate highly correlated pairs to understand underlying business logic
• Consider dimensionality reduction techniques if many strong correlations exist
• Monitor correlated features for consistent behavior in production
                """.strip()
                
                charts['correlation_heatmap'] = {
                    'chart': json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder),
                    'insight': correlation_insight
                }
            
            # Create distribution plots for first few numeric columns
            for i, col in enumerate(numeric_cols[:3]):
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                
                # Generate insight for distribution
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    skewness = col_data.skew()
                    kurtosis = col_data.kurtosis()
                    
                    distribution_insight = f"""
📈 **Distribution Analysis: {col}**

**Statistical Properties:**
• Data points: {len(col_data)}
• Mean: {col_data.mean():.2f}
• Median: {col_data.median():.2f}
• Standard deviation: {col_data.std():.2f}
• Skewness: {skewness:.2f}
• Kurtosis: {kurtosis:.2f}

**Distribution Shape:**
• {"Right-skewed (tail extends right)" if skewness > 0.5 else "Left-skewed (tail extends left)" if skewness < -0.5 else "Approximately symmetric"}
• {"Heavy-tailed distribution" if kurtosis > 3 else "Light-tailed distribution" if kurtosis < 3 else "Normal-like tail behavior"}

**Key Insights:**
• {"Consider log transformation for right-skewed data to improve normality" if skewness > 1 else "Distribution shape is suitable for most analytical methods"}
• {"Potential outliers detected in the tails - investigate extreme values" if abs(kurtosis) > 3 else "No significant outlier concerns based on tail behavior"}
• Distribution characteristics help determine appropriate statistical methods

**Next Steps:**
• Use distribution shape to guide data preprocessing and transformation choices
• Consider normalization or standardization if distribution is heavily skewed
• Monitor for consistency in production data to detect distribution drift
                    """.strip()
                    
                    charts[f'distribution_{col}'] = {
                        'chart': json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder),
                        'insight': distribution_insight
                    }
            
            return charts
        except Exception as e:
            print(f"Error generating basic charts: {str(e)}")
            return {'error': str(e)}

# Create a global instance for import
eda_service = EDAService()
