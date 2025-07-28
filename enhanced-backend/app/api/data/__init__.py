"""
Data API endpoints for advanced data management, upload, and EDA generation
"""

from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import io
import json
import pandas as pd
import numpy as np
from datetime import datetime
from app.models import Dataset, SystemLog
from app import db
from app.services.data_service import DataService
from app.services.eda_service import EDAService

data_bp = Blueprint('data', __name__)

# Initialize services
data_service = DataService()
eda_service = EDAService()


@data_bp.route('/upload', methods=['POST'])
def upload_data():
    """Upload and process data file with automatic analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Get additional parameters
        name = request.form.get('name', file.filename)
        description = request.form.get('description', '')
        generate_eda = request.form.get('generate_eda', 'true').lower() == 'true'
        
        # Secure filename
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Validate file type
        allowed_extensions = {'csv', 'xlsx', 'xls', 'json', 'parquet'}
        if file_extension not in allowed_extensions:
            return jsonify({
                'success': False, 
                'error': f'File type not supported. Allowed: {", ".join(allowed_extensions)}'
            }), 400
        
        # Read and process file
        file_content = file.read()
        file_size = len(file_content)
        
        # Create dataset record
        dataset = Dataset(
            name=name,
            filename=filename,
            file_path='',  # Will be set after saving
            description=description,
            file_size=file_size,
            file_type=file_extension,
            mime_type=file.mimetype,
            status='processing'
        )
        
        db.session.add(dataset)
        db.session.commit()
        
        # Save file
        upload_folder = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        # Create unique filename with dataset ID prefix
        unique_filename = f"{dataset.id}_{filename}"
        file_path = os.path.join(upload_folder, unique_filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Update dataset with correct filename and path for ML training compatibility
        dataset.filename = unique_filename  # ML training looks for this in UPLOAD_FOLDER
        dataset.file_path = file_path
        db.session.commit()
        
        # Process data
        try:
            processing_result = data_service.process_dataset(dataset.id, file_content, file_extension)
            
            if processing_result['success']:
                # Generate EDA if requested
                if generate_eda:
                    eda_result = eda_service.generate_eda(dataset.id)
                    if eda_result['success']:
                        dataset.eda_generated = True
                        # Ensure EDA results are properly serialized
                        dataset.eda_results = data_service._convert_numpy_types(eda_result['eda_results'])
                        dataset.eda_charts = data_service._convert_numpy_types(eda_result['charts'])
                
                dataset.status = 'ready'
                db.session.commit()
                
                SystemLog.log_info(
                    f'Dataset uploaded and processed: {dataset.name}',
                    category='data',
                    event_type='upload_success',
                    context_data={'dataset_id': dataset.id, 'file_size': file_size}
                )
                
                return jsonify({
                    'success': True,
                    'dataset': dataset.to_dict(include_data=False),  # Do not include raw data
                    'dataset_id': dataset.id,  # Explicitly include dataset_id for training
                    'message': 'Dataset uploaded and processed successfully'
                })
            else:
                dataset.set_error(processing_result['error'])
                db.session.commit()
                SystemLog.log_error(
                    f'Processing error: {processing_result["error"]}',
                    category='data',
                    event_type='processing_error',
                    context_data={'dataset_id': dataset.id}
                )
                return jsonify({
                    'success': False,
                    'error': processing_result['error']
                }), 400
                
        except Exception as e:
            dataset.set_error(str(e))
            db.session.commit()
            SystemLog.log_error(
                f'Failed to process dataset: {str(e)}',
                category='data',
                event_type='processing_error',
                context_data={'dataset_id': dataset.id}
            )
            import traceback
            print('PROCESSING ERROR:', traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }), 500
            
    except Exception as e:
        import traceback
        print('UPLOAD ERROR:', traceback.format_exc())
        SystemLog.log_error(
            f'Dataset upload failed: {str(e)}',
            category='data',
            event_type='upload_error'
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/datasets', methods=['GET'])
def get_datasets():
    """Get list of all datasets with filtering and pagination"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        status = request.args.get('status')
        category = request.args.get('category')
        search = request.args.get('search')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Build query
        query = Dataset.query
        
        if status:
            query = query.filter(Dataset.status == status)
        
        if category:
            query = query.filter(Dataset.category == category)
        
        if search:
            query = query.filter(
                Dataset.name.contains(search) | 
                Dataset.description.contains(search)
            )
        
        # Apply sorting
        if hasattr(Dataset, sort_by):
            column = getattr(Dataset, sort_by)
            if sort_order == 'desc':
                query = query.order_by(column.desc())
            else:
                query = query.order_by(column.asc())
        
        # Paginate
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        # Avoid including raw data in the response
        datasets = [dataset.to_dict(include_data=False) for dataset in pagination.items]
        
        return jsonify({
            'success': True,
            'datasets': datasets,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        })
        
    except Exception as e:
        import traceback
        print('GET DATASETS ERROR:', traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset_details(dataset_id):
    """Get detailed information about a specific dataset including columns"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Update last accessed
        dataset.update_last_accessed()
        
        # Get basic dataset info
        dataset_info = dataset.to_dict(include_data=False)
        
        # Extract column information
        columns_info = dataset.columns_info or {}
        columns = []
        
        if columns_info:
            # Check if columns_info is a dict or list
            if isinstance(columns_info, dict):
                # Dictionary format: {column_name: {type: ..., dtype: ...}}
                for col_name, col_info in columns_info.items():
                    if isinstance(col_info, dict):
                        columns.append({
                            'name': col_name,
                            'type': col_info.get('type', 'unknown'),
                            'dtype': col_info.get('dtype', 'object'),
                            'non_null_count': col_info.get('non_null_count', 0),
                            'unique_count': col_info.get('unique_count', 0),
                            'is_numeric': col_info.get('type') in ['int64', 'float64', 'int32', 'float32'] or col_info.get('dtype') in ['int64', 'float64', 'int32', 'float32'],
                            'is_categorical': col_info.get('type') in ['object', 'category', 'string'] or col_info.get('dtype') in ['object', 'category', 'string'],
                            'is_datetime': col_info.get('type') in ['datetime64', 'datetime'] or 'datetime' in str(col_info.get('dtype', ''))
                        })
                    else:
                        # Simple format: just column name
                        columns.append({
                            'name': col_name,
                            'type': 'unknown',
                            'dtype': 'object',
                            'non_null_count': 0,
                            'unique_count': 0,
                            'is_numeric': False,
                            'is_categorical': True,
                            'is_datetime': False
                        })
            elif isinstance(columns_info, list):
                # List format: [{name: ..., type: ...}, ...]
                for col_info in columns_info:
                    if isinstance(col_info, dict) and 'name' in col_info:
                        columns.append({
                            'name': col_info['name'],
                            'type': col_info.get('type', 'unknown'),
                            'dtype': col_info.get('dtype', 'object'),
                            'non_null_count': col_info.get('non_null_count', 0),
                            'unique_count': col_info.get('unique_count', 0),
                            'is_numeric': col_info.get('type') in ['int64', 'float64', 'int32', 'float32'] or col_info.get('dtype') in ['int64', 'float64', 'int32', 'float32'],
                            'is_categorical': col_info.get('type') in ['object', 'category', 'string'] or col_info.get('dtype') in ['object', 'category', 'string'],
                            'is_datetime': col_info.get('type') in ['datetime64', 'datetime'] or 'datetime' in str(col_info.get('dtype', ''))
                        })
                    elif isinstance(col_info, str):
                        # Simple string: just column name
                        columns.append({
                            'name': col_info,
                            'type': 'unknown',
                            'dtype': 'object',
                            'non_null_count': 0,
                            'unique_count': 0,
                            'is_numeric': False,
                            'is_categorical': True,
                            'is_datetime': False
                        })
        else:
            # Fallback: try to load the dataset and get basic column info
            try:
                df = data_service.load_dataset(dataset_id)
                if df is not None:
                    for col in df.columns:
                        dtype_str = str(df[col].dtype)
                        columns.append({
                            'name': col,
                            'type': dtype_str,
                            'dtype': dtype_str,
                            'non_null_count': int(df[col].notna().sum()),
                            'unique_count': int(df[col].nunique()),
                            'is_numeric': df[col].dtype.kind in 'biufc',
                            'is_categorical': df[col].dtype.kind in 'OSU',
                            'is_datetime': df[col].dtype.kind in 'M'
                        })
            except Exception as load_error:
                print(f"Could not load dataset for column info: {load_error}")
        
        return jsonify({
            'success': True,
            'dataset': dataset_info,
            'columns': columns,
            'numeric_columns': [col['name'] for col in columns if col['is_numeric']],
            'categorical_columns': [col['name'] for col in columns if col['is_categorical']],
            'datetime_columns': [col['name'] for col in columns if col['is_datetime']]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/datasets/<int:dataset_id>/preview', methods=['GET'])
def preview_dataset(dataset_id):
    """Get dataset preview with sample data and pagination support"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 20)), 100)  # Limit to max 100 rows per page
        
        preview_data = data_service.get_dataset_preview(
            dataset_id, 
            rows=limit,  # Legacy parameter maintained for compatibility
            page=page, 
            limit=limit
        )
        
        if preview_data['success']:
            return jsonify({
                'success': True,
                'preview': preview_data['preview'],
                'total': preview_data['total'],
                'page': preview_data['page'],
                'limit': preview_data['limit'],
                'columns': preview_data.get('columns', []),
                'dataset_info': dataset.to_dict(include_data=False)
            })
        else:
            return jsonify({
                'success': False,
                'error': preview_data['error']
            }), 400
            
    except Exception as e:
        import traceback
        print(f"Error in preview endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/datasets/<int:dataset_id>/eda', methods=['GET'])
def get_eda(dataset_id):
    """Get existing EDA for a dataset"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.eda_generated or not dataset.eda_results:
            return jsonify({
                'success': False,
                'error': 'No EDA available for this dataset',
                'can_generate': True
            }), 404
            
        return jsonify({
            'success': True,
            'overview': dataset.eda_results.get('overview', {}),
            'statistics': {
                'numerical_analysis': dataset.eda_results.get('numerical_analysis', {}),
                'categorical_analysis': dataset.eda_results.get('categorical_analysis', {}),
                'data_quality': dataset.eda_results.get('data_quality', {}),
                'relationships': dataset.eda_results.get('relationships', {}),
                'outliers': dataset.eda_results.get('outliers', {}),
                'statistical_tests': dataset.eda_results.get('statistical_tests', {}),
                'recommendations': dataset.eda_results.get('recommendations', [])
            },
            'charts': dataset.eda_charts or {}
        })
        
    except Exception as e:
        SystemLog.log_error(
            f'Failed to get EDA for dataset {dataset_id}: {str(e)}',
            category='data',
            event_type='eda_error'
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@data_bp.route('/datasets/<int:dataset_id>/eda/generate', methods=['POST'])
def generate_eda(dataset_id):
    """Generate or regenerate EDA for a dataset"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if dataset.status != 'ready':
            return jsonify({
                'success': False,
                'error': 'Dataset is not ready for EDA generation'
            }), 400
        
        # Generate EDA
        eda_result = eda_service.generate_eda(dataset_id)
        
        if eda_result['success']:
            # Update dataset
            dataset.eda_generated = True
            # Ensure EDA results are properly serialized
            dataset.eda_results = data_service._convert_numpy_types(eda_result['eda_results'])
            dataset.eda_charts = data_service._convert_numpy_types(eda_result['charts'])
            dataset.updated_at = datetime.utcnow()
            db.session.commit()
            
            SystemLog.log_info(
                f'EDA generated for dataset: {dataset.name}',
                category='data',
                event_type='eda_generated',
                context_data={'dataset_id': dataset_id}
            )
            
            return jsonify({
                'success': True,
                'overview': eda_result['eda_results'].get('overview', {}),
                'statistics': eda_result['eda_results'].get('statistics', {}),
                'charts': eda_result['charts'],
                'message': 'EDA generated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': eda_result['error']
            }), 500
            
    except Exception as e:
        SystemLog.log_error(
            f'EDA generation failed for dataset {dataset_id}: {str(e)}',
            category='data',
            event_type='eda_error'
        )
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/datasets/<int:dataset_id>/quality', methods=['GET'])
def get_data_quality(dataset_id):
    """Get detailed data quality assessment"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        quality_result = data_service.assess_data_quality(dataset_id)
        
        if quality_result['success']:
            return jsonify({
                'success': True,
                'quality_assessment': quality_result['assessment'],
                'recommendations': quality_result['recommendations']
            })
        else:
            return jsonify({
                'success': False,
                'error': quality_result['error']
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/datasets/<int:dataset_id>/transform', methods=['POST'])
def transform_dataset(dataset_id):
    """Apply data transformations"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if dataset.status != 'ready':
            return jsonify({
                'success': False,
                'error': 'Dataset is not ready for transformation'
            }), 400
        
        # Get transformation parameters
        transformations = request.json.get('transformations', [])
        save_as_new = request.json.get('save_as_new', False)
        
        if not transformations:
            return jsonify({
                'success': False,
                'error': 'No transformations specified'
            }), 400
        
        # Apply transformations
        transform_result = data_service.apply_transformations(dataset_id, transformations, save_as_new)
        
        if transform_result['success']:
            return jsonify({
                'success': True,
                'result': transform_result['result'],
                'new_dataset_id': transform_result.get('new_dataset_id'),
                'message': 'Transformations applied successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': transform_result['error']
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/datasets/<int:dataset_id>/export', methods=['POST'])
def export_dataset(dataset_id):
    """Export dataset in various formats"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        export_format = request.json.get('format', 'csv')
        include_transformations = request.json.get('include_transformations', False)
        
        export_result = data_service.export_dataset(dataset_id, export_format, include_transformations)
        
        if export_result['success']:
            return send_file(
                export_result['file_path'],
                as_attachment=True,
                download_name=export_result['filename']
            )
        else:
            return jsonify({
                'success': False,
                'error': export_result['error']
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/datasets/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a dataset"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Delete file if it exists
        if dataset.file_path and os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # Delete from database
        db.session.delete(dataset)
        db.session.commit()
        
        SystemLog.log_info(
            f'Dataset deleted: {dataset.name}',
            category='data',
            event_type='dataset_deleted',
            context_data={'dataset_id': dataset_id}
        )
        
        return jsonify({
            'success': True,
            'message': 'Dataset deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

