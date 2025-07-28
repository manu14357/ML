"""
Data Service for advanced data processing and management
"""

import pandas as pd
import numpy as np
import json
import io
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from app.models import Dataset
from app import db


class DataService:
    """Service for advanced data processing and management"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls', 'json', 'parquet']
    
    def process_dataset(self, dataset_id: int, file_content: bytes, file_type: str) -> Dict[str, Any]:
        """Process uploaded dataset and extract metadata"""
        try:
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return {'success': False, 'error': 'Dataset not found'}
            
            # Read data based on file type
            df = self._read_data(file_content, file_type)
            
            if df is None:
                return {'success': False, 'error': 'Failed to read data file'}
            
            # Extract basic information
            rows_count = len(df)
            columns_count = len(df.columns)
            memory_usage = df.memory_usage(deep=True).sum()
            
            # Analyze data types
            data_types = {}
            for col in df.columns:
                dtype = str(df[col].dtype)
                data_types[col] = dtype
            
            # Get column information
            columns_info = []
            for col in df.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'non_null_count': int(df[col].count()),
                    'null_count': int(df[col].isnull().sum()),
                    'unique_count': int(df[col].nunique()),
                    'memory_usage': int(df[col].memory_usage(deep=True))
                }
                
                # Add type-specific statistics
                if df[col].dtype in ['int64', 'float64']:
                    col_info.update({
                        'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                        'median': float(df[col].median()) if not pd.isna(df[col].median()) else None
                    })
                elif df[col].dtype == 'object':
                    col_info.update({
                        'most_frequent': str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None,
                        'avg_length': float(df[col].astype(str).str.len().mean()) if not df[col].empty else 0
                    })
                elif 'datetime' in str(df[col].dtype):
                    # Handle datetime columns
                    col_info.update({
                        'min': df[col].min().isoformat() if not pd.isna(df[col].min()) else None,
                        'max': df[col].max().isoformat() if not pd.isna(df[col].max()) else None,
                        'most_frequent': df[col].mode().iloc[0].isoformat() if len(df[col].mode()) > 0 and not pd.isna(df[col].mode().iloc[0]) else None
                    })
                
                columns_info.append(col_info)
            
            # Calculate data quality metrics
            missing_values_count = int(df.isnull().sum().sum())
            duplicate_rows_count = int(df.duplicated().sum())
            
            # Calculate data quality score (0-100)
            total_cells = rows_count * columns_count
            missing_ratio = missing_values_count / total_cells if total_cells > 0 else 0
            duplicate_ratio = duplicate_rows_count / rows_count if rows_count > 0 else 0
            
            quality_score = max(0, 100 - (missing_ratio * 50) - (duplicate_ratio * 30))
            
            # Generate sample data for preview
            sample_size = min(10, rows_count)
            sample_data = df.head(sample_size).to_dict('records')
            
            # Convert numpy types to Python types for JSON serialization
            sample_data = self._convert_numpy_types(sample_data)
            
            # Generate basic statistics
            statistics = {}
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = df[numeric_cols].describe()
                    statistics = stats_df.to_dict()
                    statistics = self._convert_numpy_types(statistics)
            except Exception:
                pass
            
            # Update dataset with processed information
            dataset.rows_count = rows_count
            dataset.columns_count = columns_count
            dataset.memory_usage = memory_usage
            dataset.missing_values_count = missing_values_count
            dataset.duplicate_rows_count = duplicate_rows_count
            dataset.data_quality_score = quality_score
            dataset.columns_info = self._convert_numpy_types(columns_info)
            dataset.data_types = data_types
            dataset.sample_data = sample_data
            dataset.statistics = statistics
            dataset.status = 'ready'
            dataset.updated_at = datetime.utcnow()
            
            try:
                db.session.commit()
            except Exception as e:
                # If commit fails due to serialization, try to re-convert all JSON fields
                db.session.rollback()
                dataset.columns_info = self._convert_numpy_types(columns_info)
                dataset.sample_data = self._convert_numpy_types(sample_data)
                dataset.statistics = self._convert_numpy_types(statistics)
                try:
                    db.session.commit()
                except Exception as e2:
                    # If it still fails, set minimal data and log the error
                    db.session.rollback()
                    dataset.columns_info = None
                    dataset.sample_data = None
                    dataset.statistics = None
                    dataset.status = 'ready'
                    db.session.commit()
                    print(f"Warning: Could not save full dataset metadata due to serialization error: {str(e2)}")
                    print(f"Original error: {str(e)}")
            
            return {
                'success': True,
                'dataset_info': {
                    'rows_count': rows_count,
                    'columns_count': columns_count,
                    'memory_usage': memory_usage,
                    'data_quality_score': quality_score,
                    'missing_values_count': missing_values_count,
                    'duplicate_rows_count': duplicate_rows_count
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_dataset_preview(self, dataset_id: int, rows: int = 10, page: int = 1, limit: int = 20) -> Dict[str, Any]:
        """Get dataset preview with pagination support"""
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
            
            # Calculate pagination
            total_rows = len(df)
            start_idx = (page - 1) * limit
            end_idx = min(start_idx + limit, total_rows)
            
            # Get preview data with pagination
            if start_idx >= total_rows:
                preview_data = []
            else:
                preview_data = df.iloc[start_idx:end_idx].to_dict('records')
                
            # Ensure all types are JSON serializable, especially handling timestamps
            preview_data = self._convert_numpy_types(preview_data)
            
            # Return preview data in the format expected by the frontend (flat list of records)
            return {
                'success': True,
                'preview': preview_data,
                'total': total_rows,
                'page': page,
                'limit': limit,
                'columns': list(df.columns)
            }
            
        except Exception as e:
            import traceback
            print(f"Error getting dataset preview: {str(e)}")
            print(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def assess_data_quality(self, dataset_id: int) -> Dict[str, Any]:
        """Perform detailed data quality assessment"""
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
            
            assessment = {}
            recommendations = []
            
            # Completeness assessment
            missing_by_column = df.isnull().sum().to_dict()
            total_rows = len(df)
            
            completeness = {}
            for col, missing_count in missing_by_column.items():
                missing_ratio = missing_count / total_rows
                completeness[col] = {
                    'missing_count': int(missing_count),
                    'missing_ratio': float(missing_ratio),
                    'completeness_score': float(1 - missing_ratio) * 100
                }
                
                if missing_ratio > 0.5:
                    recommendations.append({
                        'type': 'completeness',
                        'severity': 'high',
                        'column': col,
                        'message': f'Column {col} has {missing_ratio:.1%} missing values'
                    })
                elif missing_ratio > 0.1:
                    recommendations.append({
                        'type': 'completeness',
                        'severity': 'medium',
                        'column': col,
                        'message': f'Column {col} has {missing_ratio:.1%} missing values'
                    })
            
            assessment['completeness'] = completeness
            
            # Uniqueness assessment
            uniqueness = {}
            for col in df.columns:
                unique_count = df[col].nunique()
                uniqueness_ratio = unique_count / total_rows
                uniqueness[col] = {
                    'unique_count': int(unique_count),
                    'uniqueness_ratio': float(uniqueness_ratio),
                    'uniqueness_score': float(uniqueness_ratio) * 100
                }
                
                if uniqueness_ratio < 0.1 and df[col].dtype == 'object':
                    recommendations.append({
                        'type': 'uniqueness',
                        'severity': 'low',
                        'column': col,
                        'message': f'Column {col} has low uniqueness ({uniqueness_ratio:.1%})'
                    })
            
            assessment['uniqueness'] = uniqueness
            
            # Consistency assessment
            consistency = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check for inconsistent formatting
                    sample_values = df[col].dropna().astype(str).head(100)
                    has_mixed_case = any(v != v.lower() and v != v.upper() for v in sample_values)
                    has_whitespace = any(v != v.strip() for v in sample_values)
                    
                    consistency[col] = {
                        'has_mixed_case': has_mixed_case,
                        'has_whitespace': has_whitespace,
                        'consistency_score': 100 - (has_mixed_case * 20) - (has_whitespace * 10)
                    }
                    
                    if has_mixed_case:
                        recommendations.append({
                            'type': 'consistency',
                            'severity': 'low',
                            'column': col,
                            'message': f'Column {col} has inconsistent case formatting'
                        })
                    
                    if has_whitespace:
                        recommendations.append({
                            'type': 'consistency',
                            'severity': 'low',
                            'column': col,
                            'message': f'Column {col} has leading/trailing whitespace'
                        })
            
            assessment['consistency'] = consistency
            
            # Validity assessment (basic data type checks)
            validity = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if numeric columns are stored as strings
                    sample_values = df[col].dropna().head(100)
                    numeric_like = sum(1 for v in sample_values if str(v).replace('.', '').replace('-', '').isdigit())
                    numeric_ratio = numeric_like / len(sample_values) if len(sample_values) > 0 else 0
                    
                    validity[col] = {
                        'numeric_like_ratio': float(numeric_ratio),
                        'validity_score': 100 - (numeric_ratio * 20 if numeric_ratio > 0.8 else 0)
                    }
                    
                    if numeric_ratio > 0.8:
                        recommendations.append({
                            'type': 'validity',
                            'severity': 'medium',
                            'column': col,
                            'message': f'Column {col} appears to contain numeric data but is stored as text'
                        })
            
            assessment['validity'] = validity
            
            # Overall quality score
            overall_score = dataset.data_quality_score or 0
            
            return {
                'success': True,
                'assessment': {
                    'overall_score': overall_score,
                    'completeness': assessment['completeness'],
                    'uniqueness': assessment['uniqueness'],
                    'consistency': assessment['consistency'],
                    'validity': assessment['validity']
                },
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def apply_transformations(self, dataset_id: int, transformations: List[Dict], save_as_new: bool = False) -> Dict[str, Any]:
        """Apply data transformations"""
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
            
            # Apply transformations
            for transform in transformations:
                transform_type = transform.get('type')
                
                if transform_type == 'drop_column':
                    column = transform.get('column')
                    if column in df.columns:
                        df = df.drop(columns=[column])
                
                elif transform_type == 'rename_column':
                    old_name = transform.get('old_name')
                    new_name = transform.get('new_name')
                    if old_name in df.columns:
                        df = df.rename(columns={old_name: new_name})
                
                elif transform_type == 'fill_missing':
                    column = transform.get('column')
                    method = transform.get('method', 'mean')
                    if column in df.columns:
                        if method == 'mean' and df[column].dtype in ['int64', 'float64']:
                            df[column] = df[column].fillna(df[column].mean())
                        elif method == 'median' and df[column].dtype in ['int64', 'float64']:
                            df[column] = df[column].fillna(df[column].median())
                        elif method == 'mode':
                            df[column] = df[column].fillna(df[column].mode().iloc[0] if len(df[column].mode()) > 0 else '')
                        elif method == 'forward_fill':
                            df[column] = df[column].fillna(method='ffill')
                        elif method == 'backward_fill':
                            df[column] = df[column].fillna(method='bfill')
                
                elif transform_type == 'remove_duplicates':
                    df = df.drop_duplicates()
                
                elif transform_type == 'convert_type':
                    column = transform.get('column')
                    target_type = transform.get('target_type')
                    if column in df.columns:
                        try:
                            if target_type == 'numeric':
                                df[column] = pd.to_numeric(df[column], errors='coerce')
                            elif target_type == 'datetime':
                                df[column] = pd.to_datetime(df[column], errors='coerce')
                            elif target_type == 'string':
                                df[column] = df[column].astype(str)
                        except Exception:
                            pass  # Skip if conversion fails
            
            # Save transformed data
            if save_as_new:
                # Create new dataset
                new_dataset = Dataset(
                    name=f"{dataset.name}_transformed",
                    filename=f"{dataset.filename}_transformed",
                    file_path='',
                    description=f"Transformed version of {dataset.name}",
                    file_type=dataset.file_type,
                    status='processing'
                )
                db.session.add(new_dataset)
                db.session.commit()
                
                # Save transformed data
                new_file_path = dataset.file_path.replace(f"{dataset.id}_", f"{new_dataset.id}_")
                self._save_dataframe(df, new_file_path, dataset.file_type)
                new_dataset.file_path = new_file_path
                
                # Process the new dataset
                file_content = open(new_file_path, 'rb').read()
                self.process_dataset(new_dataset.id, file_content, dataset.file_type)
                
                return {
                    'success': True,
                    'result': 'New dataset created with transformations',
                    'new_dataset_id': new_dataset.id
                }
            else:
                # Update existing dataset
                self._save_dataframe(df, dataset.file_path, dataset.file_type)
                
                # Reprocess the dataset
                file_content = open(dataset.file_path, 'rb').read()
                self.process_dataset(dataset_id, file_content, dataset.file_type)
                
                return {
                    'success': True,
                    'result': 'Dataset updated with transformations'
                }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def export_dataset(self, dataset_id: int, export_format: str, include_transformations: bool = False) -> Dict[str, Any]:
        """Export dataset in specified format"""
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
            
            # Create export filename
            export_filename = f"{dataset.name}_export.{export_format}"
            export_path = os.path.join(os.path.dirname(dataset.file_path), export_filename)
            
            # Save in requested format
            self._save_dataframe(df, export_path, export_format)
            
            return {
                'success': True,
                'file_path': export_path,
                'filename': export_filename
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _read_data(self, file_content: bytes, file_type: str) -> Optional[pd.DataFrame]:
        """Read data from file content"""
        try:
            if file_type == 'csv':
                return pd.read_csv(io.BytesIO(file_content))
            elif file_type in ['xlsx', 'xls']:
                # Try to read Excel file with better error handling
                try:
                    # First try to read with engine='openpyxl' for xlsx
                    if file_type == 'xlsx':
                        df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
                    else:
                        df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')
                    
                    # Check if dataframe is empty or has no columns
                    if df.empty or len(df.columns) == 0:
                        return None
                        
                    # Handle cases where Excel file might have leading empty rows
                    # Find the first row with non-null values
                    for i in range(min(10, len(df))):  # Check first 10 rows
                        if not df.iloc[i].isnull().all():
                            if i > 0:
                                # Re-read with proper header
                                if file_type == 'xlsx':
                                    df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl', header=i)
                                else:
                                    df = pd.read_excel(io.BytesIO(file_content), engine='xlrd', header=i)
                            break
                    
                    return df
                    
                except Exception as e:
                    print(f"Excel reading error: {str(e)}")
                    # Fallback: try with different parameters
                    try:
                        return pd.read_excel(io.BytesIO(file_content), header=0)
                    except:
                        return None
                        
            elif file_type == 'json':
                data = json.loads(file_content.decode('utf-8'))
                if isinstance(data, list):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame([data])
            elif file_type == 'parquet':
                return pd.read_parquet(io.BytesIO(file_content))
            else:
                return None
        except Exception as e:
            print(f"Error reading {file_type} file: {str(e)}")
            return None
    
    def _read_data_from_file(self, file_path: str, file_type: str) -> Optional[pd.DataFrame]:
        """Read data from file path"""
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path)
            elif file_type in ['xlsx', 'xls']:
                # Try to read Excel file with better error handling
                try:
                    # First try to read with engine='openpyxl' for xlsx
                    if file_type == 'xlsx':
                        df = pd.read_excel(file_path, engine='openpyxl')
                    else:
                        df = pd.read_excel(file_path, engine='xlrd')
                    
                    # Check if dataframe is empty or has no columns
                    if df.empty or len(df.columns) == 0:
                        return None
                        
                    # Handle cases where Excel file might have leading empty rows
                    # Find the first row with non-null values
                    for i in range(min(10, len(df))):  # Check first 10 rows
                        if not df.iloc[i].isnull().all():
                            if i > 0:
                                # Re-read with proper header
                                if file_type == 'xlsx':
                                    df = pd.read_excel(file_path, engine='openpyxl', header=i)
                                else:
                                    df = pd.read_excel(file_path, engine='xlrd', header=i)
                            break
                    
                    return df
                    
                except Exception as e:
                    print(f"Excel reading error: {str(e)}")
                    # Fallback: try with different parameters
                    try:
                        return pd.read_excel(file_path, header=0)
                    except:
                        return None
                        
            elif file_type == 'json':
                return pd.read_json(file_path)
            elif file_type == 'parquet':
                return pd.read_parquet(file_path)
            else:
                return None
        except Exception as e:
            print(f"Error reading {file_type} file: {str(e)}")
            return None
    
    def _save_dataframe(self, df: pd.DataFrame, file_path: str, file_type: str):
        """Save dataframe to file"""
        if file_type == 'csv':
            df.to_csv(file_path, index=False)
        elif file_type == 'xlsx':
            df.to_excel(file_path, index=False)
        elif file_type == 'json':
            df.to_json(file_path, orient='records')
        elif file_type == 'parquet':
            df.to_parquet(file_path, index=False)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        from datetime import datetime, date, time
        
        # Special handling for arrays, DataFrames, and Series first
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # Handle pandas DataFrames specially
            return self._convert_numpy_types(obj.to_dict('records'))
        elif isinstance(obj, pd.Series):
            # Handle pandas Series specially
            return self._convert_numpy_types(obj.to_dict())
            
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
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (complex, np.number)):
            # Handle other numeric types
            return float(obj)
        elif hasattr(obj, 'item'):
            # Handle numpy scalars
            return self._convert_numpy_types(obj.item())
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat')):
            # Handle any datetime-like objects with isoformat method
            return obj.isoformat()
            # Handle any datetime-like objects with isoformat method
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            # Handle pandas Series specially
            return self._convert_numpy_types(obj.to_dict())
        elif isinstance(obj, pd.DataFrame):
            # Handle pandas DataFrames specially
            return self._convert_numpy_types(obj.to_dict('records'))
        elif hasattr(obj, 'item'):
            # Handle numpy scalars
            return self._convert_numpy_types(obj.item())
        elif isinstance(obj, (complex, np.number)):
            # Handle other numeric types
            return float(obj)
        else:
            # For any other types, try to convert to a basic type
            try:
                return str(obj)
            except:
                return None

