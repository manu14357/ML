from flask import Blueprint, request, jsonify
from app.services.ai_service_advanced import AdvancedAIInsightService

visualization_bp = Blueprint('visualization_bp', __name__)

@visualization_bp.route('/insight', methods=['POST'])
def get_visualization_insight():
    """
    Generates AI-powered insight for a given visualization configuration.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    chart_type = data.get('chart_type')
    dataset_name = data.get('dataset_name')
    x_column = data.get('x_column')
    y_column = data.get('y_column')
    title = data.get('title')

    if not all([chart_type, dataset_name, x_column]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        ai_service = AdvancedAIInsightService()
        prompt = f"Given a {chart_type} titled '{title}' for the '{dataset_name}' dataset, with X-axis '{x_column}' and Y-axis '{y_column}', what is one key business insight that can be derived? The insight should be actionable."
        
        # In a real scenario, you might pass actual data summary.
        # For now, we generate a plausible insight based on the metadata.
        insight = ai_service._call_nvidia_api_streaming(prompt)

        return jsonify({"insight": insight})
    except Exception as e:
        # In a real app, you'd log this error.
        print(f"Error generating insight: {e}")
        return jsonify({"error": "Failed to generate insight"}), 500
