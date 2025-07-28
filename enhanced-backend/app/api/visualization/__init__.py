"""
Advanced Visualization API endpoints for charts and dashboards
"""

from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from app.models.dataset import Dataset
import numpy as np

visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route('/charts', methods=['GET'])
def get_charts():
    """Get list of visualizations"""
    try:
        # For now, return a sample response
        # In a real implementation, this would fetch from database
        charts = [
            {
                'id': 1,
                'title': 'Sample Line Chart',
                'type': 'line',
                'dataset_id': 1,
                'created_at': '2024-01-01T00:00:00Z'
            }
        ]
        
        return jsonify({
            'success': True,
            'charts': charts,
            'message': 'Charts retrieved successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving charts: {str(e)}'
        }), 500

@visualization_bp.route('/create', methods=['POST'])
def create_chart():
    """Create a new visualization"""
    try:
        data = request.get_json()
        
        dataset_id = data.get('dataset_id')
        chart_type = data.get('chart_type', 'line')
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        title = data.get('title', 'Untitled Chart')
        color_column = data.get('color_column')
        size_column = data.get('size_column')
        color_scheme = data.get('color_scheme', 'viridis')
        
        if not all([dataset_id, x_column]):
            return jsonify({
                'success': False,
                'message': 'dataset_id and x_column are required'
            }), 400
            
        # Load dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
            
        df = pd.read_csv(dataset.file_path)
        
        # Validate columns exist
        if x_column not in df.columns:
            return jsonify({
                'success': False,
                'message': f'Column {x_column} not found in dataset'
            }), 400
            
        if y_column and y_column not in df.columns:
            return jsonify({
                'success': False,
                'message': f'Column {y_column} not found in dataset'
            }), 400
        
        # Generate chart based on type
        chart_config = generate_plotly_chart(
            df, chart_type, x_column, y_column, 
            color_column, size_column, title, color_scheme
        )
        
        # In a real implementation, save chart to database
        chart_id = f"chart_{dataset_id}_{chart_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        return jsonify({
            'success': True,
            'chart_id': chart_id,
            'chart_config': chart_config,
            'message': 'Chart created successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error creating chart: {str(e)}'
        }), 500

@visualization_bp.route('/datasets/<int:dataset_id>/analyze', methods=['POST'])
def analyze_dataset_for_visualization(dataset_id):
    """Analyze dataset and suggest optimal visualizations"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
            
        df = pd.read_csv(dataset.file_path)
        
        # Analyze column types and suggest visualizations
        suggestions = analyze_and_suggest_charts(df)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'column_info': get_column_analysis(df),
            'message': 'Dataset analyzed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error analyzing dataset: {str(e)}'
        }), 500

@visualization_bp.route('/generate-advanced/<int:dataset_id>', methods=['POST'])
def generate_advanced_visualizations(dataset_id):
    """Generate advanced visualization suite for a dataset"""
    try:
        data = request.get_json()
        chart_types = data.get('chart_types', ['correlation_heatmap', 'distribution_plots', 'scatter_matrix'])
        
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
            
        df = pd.read_csv(dataset.file_path)
        
        # Generate multiple advanced visualizations
        charts = {}
        
        if 'correlation_heatmap' in chart_types:
            charts['correlation_heatmap'] = generate_correlation_heatmap(df)
            
        if 'distribution_plots' in chart_types:
            charts['distribution_plots'] = generate_distribution_plots(df)
            
        if 'scatter_matrix' in chart_types:
            charts['scatter_matrix'] = generate_scatter_matrix(df)
            
        if 'time_series' in chart_types:
            charts['time_series'] = generate_time_series_plots(df)
            
        if 'box_plots' in chart_types:
            charts['box_plots'] = generate_box_plots(df)
        
        return jsonify({
            'success': True,
            'charts': charts,
            'message': 'Advanced visualizations generated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating visualizations: {str(e)}'
        }), 500

@visualization_bp.route('/analyze/<int:dataset_id>', methods=['POST'])
def analyze_dataset_for_charts(dataset_id):
    """
    Analyze dataset and generate recommended chart configurations
    """
    try:
        data = request.get_json() or {}
        chart_types = data.get('chart_types', ['correlation_heatmap', 'distribution_plots', 'scatter_matrix', 'trend_analysis'])
        
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
            
        df = pd.read_csv(dataset.file_path)
        
        # Analyze dataset structure
        analysis = analyze_dataset_structure(df)
        recommendations = generate_chart_recommendations(df, analysis, chart_types)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'recommendations': recommendations,
            'message': 'Dataset analyzed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error analyzing dataset: {str(e)}'
        }), 500


def generate_plotly_chart(df, chart_type, x_column, y_column=None, color_column=None, 
                         size_column=None, title='Chart', color_scheme='viridis'):
    """Generate Plotly chart configuration"""
    
    # Limit data for performance
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
    
    if chart_type == 'line':
        fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
    elif chart_type == 'bar':
        fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
    elif chart_type == 'scatter':
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, 
                        size=size_column, title=title)
    elif chart_type == 'histogram':
        fig = px.histogram(df, x=x_column, color=color_column, title=title)
    elif chart_type == 'box':
        fig = px.box(df, x=x_column, y=y_column, color=color_column, title=title)
    elif chart_type == 'violin':
        fig = px.violin(df, x=x_column, y=y_column, color=color_column, title=title)
    elif chart_type == 'heatmap':
        numeric_cols = df.select_dtypes(include=['number']).columns
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title=title)
    else:
        # Default to scatter plot
        fig = px.scatter(df, x=x_column, y=y_column, title=title)
    
    # Apply color scheme
    if color_scheme and color_scheme != 'viridis':
        fig.update_layout(coloraxis_colorscale=color_scheme)
    
    # Convert to JSON serializable format
    return {
        'data': fig.data,
        'layout': fig.layout
    }

def analyze_and_suggest_charts(df):
    """Analyze dataset and suggest optimal chart types"""
    suggestions = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Time series suggestions
    if datetime_cols:
        for date_col in datetime_cols:
            for num_col in numeric_cols[:3]:  # Limit suggestions
                suggestions.append({
                    'type': 'line',
                    'x_column': date_col,
                    'y_column': num_col,
                    'title': f'{num_col} over time',
                    'confidence': 0.9
                })
    
    # Correlation analysis
    if len(numeric_cols) >= 2:
        suggestions.append({
            'type': 'heatmap',
            'x_column': 'correlation_matrix',
            'title': 'Correlation Heatmap',
            'confidence': 0.8
        })
        
        # Scatter plots for highly correlated pairs
        corr_matrix = df[numeric_cols].corr()
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if abs(corr_matrix.loc[col1, col2]) > 0.5:
                    suggestions.append({
                        'type': 'scatter',
                        'x_column': col1,
                        'y_column': col2,
                        'title': f'{col1} vs {col2}',
                        'confidence': abs(corr_matrix.loc[col1, col2])
                    })
    
    # Distribution analysis
    for col in numeric_cols[:5]:  # Limit suggestions
        suggestions.append({
            'type': 'histogram',
            'x_column': col,
            'title': f'Distribution of {col}',
            'confidence': 0.7
        })
    
    # Categorical analysis
    for cat_col in categorical_cols[:3]:
        if df[cat_col].nunique() <= 20:  # Reasonable number of categories
            suggestions.append({
                'type': 'bar',
                'x_column': cat_col,
                'y_column': 'count',
                'title': f'Count by {cat_col}',
                'confidence': 0.6
            })
    
    # Sort by confidence and return top suggestions
    suggestions.sort(key=lambda x: x['confidence'], reverse=True)
    return suggestions[:10]

def get_column_analysis(df):
    """Get detailed analysis of dataset columns"""
    analysis = {}
    
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'unique_count': df[col].nunique(),
            'sample_values': df[col].dropna().head(5).tolist()
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            })
        
        analysis[col] = col_info
    
    return analysis

def generate_correlation_heatmap(df):
    """Generate correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        return None
        
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                   title="Correlation Heatmap", color_continuous_scale='RdBu')
    
    return {
        'data': fig.data,
        'layout': fig.layout
    }

def generate_distribution_plots(df):
    """Generate distribution plots for numeric columns"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    plots = []
    
    for col in numeric_cols[:6]:  # Limit to 6 plots
        fig = px.histogram(df, x=col, title=f'Distribution of {col}', 
                          marginal="box", hover_data=df.columns)
        plots.append({
            'column': col,
            'data': fig.data,
            'layout': fig.layout
        })
    
    return plots

def generate_scatter_matrix(df):
    """Generate scatter matrix for numeric columns"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        return None
    
    # Limit columns for performance
    cols_to_use = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
    
    fig = px.scatter_matrix(df[cols_to_use], title="Scatter Matrix")
    
    return {
        'data': fig.data,
        'layout': fig.layout
    }

def generate_time_series_plots(df):
    """Generate time series plots if datetime columns exist"""
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if not datetime_cols or not numeric_cols:
        return None
    
    plots = []
    date_col = datetime_cols[0]
    
    for num_col in numeric_cols[:3]:  # Limit to 3 plots
        fig = px.line(df, x=date_col, y=num_col, 
                     title=f'{num_col} over time')
        plots.append({
            'column': num_col,
            'data': fig.data,
            'layout': fig.layout
        })
    
    return plots

def generate_box_plots(df):
    """Generate box plots for numeric columns"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    plots = []
    
    # Box plots by category
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        for num_col in numeric_cols[:3]:
            if df[cat_col].nunique() <= 10:  # Reasonable number of categories
                fig = px.box(df, x=cat_col, y=num_col, 
                           title=f'{num_col} by {cat_col}')
                plots.append({
                    'columns': f'{num_col}_by_{cat_col}',
                    'data': fig.data,
                    'layout': fig.layout
                })
    
    return plots if plots else None

def analyze_dataset_structure(df):
    """Analyze dataset structure and return insights"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to detect datetime columns that might be stored as strings
    for col in categorical_cols[:]:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].dropna().head(100))
                datetime_cols.append(col)
                categorical_cols.remove(col)
            except:
                pass
    
    analysis = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'datetime_columns': len(datetime_cols),
        'numeric_cols_list': numeric_cols,
        'categorical_cols_list': categorical_cols,
        'datetime_cols_list': datetime_cols,
        'missing_data': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    return analysis


def generate_chart_recommendations(df, analysis, requested_types):
    """Generate chart recommendations based on dataset analysis"""
    recommendations = []
    
    numeric_cols = analysis['numeric_cols_list']
    categorical_cols = analysis['categorical_cols_list']
    datetime_cols = analysis['datetime_cols_list']
    
    # Correlation heatmap for numeric data
    if 'correlation_heatmap' in requested_types and len(numeric_cols) >= 2:
        corr_data = df[numeric_cols].corr().values.tolist()
        recommendations.append({
            'chart_type': 'heatmap',
            'title': 'Correlation Matrix',
            'x_column': 'columns',
            'y_column': 'columns',
            'color_column': None,
            'size_column': None,
            'color_scheme': 'viridis',
            'plot_config': {
                'data': [{
                    'z': corr_data,
                    'x': numeric_cols,
                    'y': numeric_cols,
                    'type': 'heatmap',
                    'colorscale': 'Viridis'
                }],
                'layout': {
                    'title': 'Correlation Matrix',
                    'xaxis': {'title': 'Features'},
                    'yaxis': {'title': 'Features'}
                }
            }
        })
    
    # Distribution plots for numeric columns
    if 'distribution_plots' in requested_types and len(numeric_cols) >= 1:
        for col in numeric_cols[:3]:  # Limit to first 3 to avoid too many charts
            recommendations.append({
                'chart_type': 'histogram',
                'title': f'{col} Distribution',
                'x_column': col,
                'y_column': 'frequency',
                'color_column': None,
                'size_column': None,
                'color_scheme': 'blues',
                'plot_config': {
                    'data': [{
                        'x': df[col].dropna().tolist(),
                        'type': 'histogram',
                        'name': col,
                        'opacity': 0.7
                    }],
                    'layout': {
                        'title': f'{col} Distribution',
                        'xaxis': {'title': col},
                        'yaxis': {'title': 'Frequency'}
                    }
                }
            })
    
    # Scatter plots for numeric pairs
    if 'scatter_matrix' in requested_types and len(numeric_cols) >= 2:
        # Create scatter plot for the two most correlated variables
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            recommendations.append({
                'chart_type': 'scatter',
                'title': f'{x_col} vs {y_col}',
                'x_column': x_col,
                'y_column': y_col,
                'color_column': categorical_cols[0] if categorical_cols else None,
                'size_column': None,
                'color_scheme': 'viridis',
                'plot_config': {
                    'data': [{
                        'x': df[x_col].tolist(),
                        'y': df[y_col].tolist(),
                        'mode': 'markers',
                        'type': 'scatter',
                        'name': f'{x_col} vs {y_col}'
                    }],
                    'layout': {
                        'title': f'{x_col} vs {y_col}',
                        'xaxis': {'title': x_col},
                        'yaxis': {'title': y_col}
                    }
                }
            })
    
    # Time series analysis
    if 'trend_analysis' in requested_types and len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        time_col = datetime_cols[0]
        value_col = numeric_cols[0]
        
        # Sort by time column for proper time series
        df_sorted = df.sort_values(time_col)
        
        recommendations.append({
            'chart_type': 'line',
            'title': f'{value_col} Over Time',
            'x_column': time_col,
            'y_column': value_col,
            'color_column': None,
            'size_column': None,
            'color_scheme': 'viridis',
            'plot_config': {
                'data': [{
                    'x': df_sorted[time_col].tolist(),
                    'y': df_sorted[value_col].tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': f'{value_col} Trend'
                }],
                'layout': {
                    'title': f'{value_col} Over Time',
                    'xaxis': {'title': time_col},
                    'yaxis': {'title': value_col}
                }
            }
        })
    
    # Category analysis
    if len(categorical_cols) >= 1:
        cat_col = categorical_cols[0]
        value_counts = df[cat_col].value_counts().head(10)  # Top 10 categories
        
        recommendations.append({
            'chart_type': 'bar',
            'title': f'{cat_col} Distribution',
            'x_column': cat_col,
            'y_column': 'count',
            'color_column': None,
            'size_column': None,
            'color_scheme': 'viridis',
            'plot_config': {
                'data': [{
                    'x': value_counts.index.tolist(),
                    'y': value_counts.values.tolist(),
                    'type': 'bar',
                    'name': f'{cat_col} Count'
                }],
                'layout': {
                    'title': f'{cat_col} Distribution',
                    'xaxis': {'title': cat_col},
                    'yaxis': {'title': 'Count'}
                }
            }
        })
    
    return recommendations

@visualization_bp.route('/insight', methods=['POST'])
def get_visualization_insight():
    """
    Generates AI-powered insight for a given visualization configuration.
    Enhanced with dataset context and column type analysis.
    """
    from app.services.ai_service_advanced import AdvancedAIInsightService
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    chart_type = data.get('chart_type')
    dataset_name = data.get('dataset_name')
    dataset_id = data.get('dataset_id')
    x_column = data.get('x_column')
    y_column = data.get('y_column')
    title = data.get('title')
    x_column_type = data.get('x_column_type', 'unknown')
    y_column_type = data.get('y_column_type', 'unknown')
    is_x_numeric = data.get('is_x_numeric', False)
    is_y_numeric = data.get('is_y_numeric', False)
    is_x_datetime = data.get('is_x_datetime', False)
    is_y_datetime = data.get('is_y_datetime', False)
    color_column = data.get('color_column')
    size_column = data.get('size_column')
    color_scheme = data.get('color_scheme', 'viridis')

    if not all([chart_type, dataset_name, x_column]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        ai_service = AdvancedAIInsightService()
        
        # Enhanced prompt with dataset context and column types
        context_info = []
        if is_x_numeric:
            context_info.append(f"X-axis '{x_column}' contains numeric data")
        elif is_x_datetime:
            context_info.append(f"X-axis '{x_column}' contains time-series data")
        else:
            context_info.append(f"X-axis '{x_column}' contains categorical data")
            
        if y_column:
            if is_y_numeric:
                context_info.append(f"Y-axis '{y_column}' contains numeric data")
            elif is_y_datetime:
                context_info.append(f"Y-axis '{y_column}' contains time-series data")
            else:
                context_info.append(f"Y-axis '{y_column}' contains categorical data")
        
        additional_context = ""
        if color_column:
            additional_context += f" with color mapping by '{color_column}'"
        if size_column:
            additional_context += f" and size mapping by '{size_column}'"
            
        context_str = ". ".join(context_info)
        
        # Build comprehensive prompt
        prompt = f"""
Analyze this {chart_type} chart titled '{title}' from the '{dataset_name}' dataset.

Chart Context:
- {context_str}{additional_context}
- Chart type: {chart_type}
- Visual theme: {color_scheme}

Based on this visualization configuration, provide:
1. One key business insight (2-3 sentences)
2. A specific actionable recommendation
3. What trends or patterns this chart might reveal

Focus on practical business value and data-driven decision making. Be concise but insightful.
"""
        
        # Generate AI insight using the streaming API
        insight = ai_service._call_nvidia_api_streaming(prompt)

        return jsonify({"insight": insight})
    except Exception as e:
        # Log the error for debugging
        print(f"Error generating insight: {e}")
        return jsonify({"error": "Failed to generate insight"}), 500

@visualization_bp.route('/data/<int:dataset_id>', methods=['POST'])
def get_chart_data(dataset_id):
    """
    Get actual data from dataset for chart rendering
    """
    try:
        data = request.get_json()
        chart_type = data.get('chart_type')
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        color_column = data.get('color_column')
        size_column = data.get('size_column')
        sample_size = data.get('sample_size', 1000)  # Limit data for performance
        
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if not dataset.file_path or not os.path.exists(dataset.file_path):
            return jsonify({
                'success': False,
                'message': 'Dataset file not found'
            }), 404
            
        df = pd.read_csv(dataset.file_path)
        
        # Sample data if too large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Prepare chart data based on type
        chart_data = prepare_chart_data(df, chart_type, x_column, y_column, color_column, size_column)
        
        # Ensure all NaN, infinity values are replaced with None for proper JSON serialization
        from app.utils.json_encoder import dumps
        
        return jsonify({
            'success': True,
            'data': chart_data,
            'sample_size': len(df),
            'total_rows': len(pd.read_csv(dataset.file_path)),
            'message': 'Chart data retrieved successfully'
        })
        
    except Exception as e:
        import traceback
        print(f"Error retrieving chart data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Error retrieving chart data: {str(e)}'
        }), 500


def prepare_chart_data(df, chart_type, x_column, y_column, color_column=None, size_column=None):
    """Prepare data for different chart types"""
    
    # Clean data - remove NaN values
    columns_to_use = [col for col in [x_column, y_column, color_column, size_column] if col and col in df.columns]
    df_clean = df[columns_to_use].dropna()
    
    # Replace any remaining NaN values with None for JSON compatibility
    df_clean = df_clean.replace({np.nan: None, np.inf: None, -np.inf: None})
    
    # Convert to Python native types to avoid numpy JSON serialization issues
    chart_data = {}
    
    if x_column in df_clean.columns:
        x_values = df_clean[x_column].tolist()
        # Replace any remaining NaN values with None
        chart_data['x'] = [None if (pd.isna(x) or (isinstance(x, float) and np.isinf(x))) else x for x in x_values]
    else:
        chart_data['x'] = []
    
    if y_column and y_column in df_clean.columns:
        y_values = df_clean[y_column].tolist()
        # Replace any remaining NaN values with None
        chart_data['y'] = [None if (pd.isna(y) or (isinstance(y, float) and np.isinf(y))) else y for y in y_values]
    else:
        chart_data['y'] = []
    
    # Add color data if specified
    if color_column and color_column in df_clean.columns:
        color_values = df_clean[color_column].tolist()
        # Replace any remaining NaN values with None
        chart_data['color'] = [None if (pd.isna(c) or (isinstance(c, float) and np.isinf(c))) else c for c in color_values]
        chart_data['color_column'] = color_column
    
    # Add size data if specified
    if size_column and size_column in df_clean.columns:
        size_values = df_clean[size_column].tolist()
        # Replace any remaining NaN values with None
        chart_data['size'] = [None if (pd.isna(s) or (isinstance(s, float) and np.isinf(s))) else s for s in size_values]
        chart_data['size_column'] = size_column
    
    # Chart-specific data preparation
    if chart_type == 'pie':
        # For pie charts, aggregate data
        if x_column in df_clean.columns:
            value_counts = df_clean[x_column].value_counts()
            chart_data = {
                'labels': value_counts.index.tolist(),
                'values': value_counts.values.tolist()
            }
    
    elif chart_type == 'histogram':
        # For histograms, just need x data
        if x_column in df_clean.columns:
            x_values = df_clean[x_column].tolist()
            chart_data = {
                'x': [None if (pd.isna(x) or (isinstance(x, float) and np.isinf(x))) else x for x in x_values]
            }
    
    elif chart_type == 'box':
        # For box plots, organize by categories if color column is categorical
        if color_column and color_column in df_clean.columns and y_column in df_clean.columns:
            grouped_data = {}
            for group, data in df_clean.groupby(color_column)[y_column]:
                # Clean the data to remove any remaining NaN values
                clean_data = [None if (pd.isna(v) or (isinstance(v, float) and np.isinf(v))) else v for v in data.tolist()]
                grouped_data[str(group)] = clean_data
            
            chart_data = {
                'grouped_data': grouped_data,
                'y_column': y_column,
                'group_column': color_column
            }
        elif y_column in df_clean.columns:
            y_values = df_clean[y_column].tolist()
            chart_data = {
                'y': [None if (pd.isna(y) or (isinstance(y, float) and np.isinf(y))) else y for y in y_values]
            }
    
    elif chart_type == 'heatmap':
        # For heatmaps, create correlation matrix or pivot table
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df_clean[numeric_cols].corr()
            # Clean the correlation matrix
            corr_matrix = corr_matrix.fillna(0).replace([np.inf, -np.inf], 0)
            
            chart_data = {
                'z': corr_matrix.values.tolist(),
                'x': corr_matrix.columns.tolist(),
                'y': corr_matrix.columns.tolist(),
                'type': 'correlation'
            }
    
    return chart_data

