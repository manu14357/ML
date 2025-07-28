"""
Node Prompt Router - Dynamically routes to appropriate prompt generators
"""

from .data_source_prompt import DataSourcePrompt
from .statistical_analysis_prompt import StatisticalAnalysisPrompt
from .data_cleaning_prompt import DataCleaningPrompt
from .visualization_prompt import VisualizationPrompt
from .machine_learning_prompt import MachineLearningPrompt
from .eda_prompt import EDAPrompt
from .feature_engineering_prompt import FeatureEngineeringPrompt
from .anomaly_detection_prompt import AnomalyDetectionPrompt

class NodePromptRouter:
    """Route to appropriate prompt generator based on node type"""
    
    # Node type mappings to prompt generators
    PROMPT_GENERATORS = {
        # Data source nodes
        'data_source': DataSourcePrompt,
        'csv_upload': DataSourcePrompt,
        'database_connection': DataSourcePrompt,
        'file_upload': DataSourcePrompt,
        
        # Analysis nodes
        'statistical_analysis': StatisticalAnalysisPrompt,
        'descriptive_stats': StatisticalAnalysisPrompt,
        'processing': StatisticalAnalysisPrompt,
        
        # EDA nodes
        'eda': EDAPrompt,
        'exploratory_data_analysis': EDAPrompt,
        'data_exploration': EDAPrompt,
        
        # Cleaning nodes
        'data_cleaning': DataCleaningPrompt,
        'preprocessing': DataCleaningPrompt,
        'data_preparation': DataCleaningPrompt,
        
        # Visualization nodes
        'visualization': VisualizationPrompt,
        'chart': VisualizationPrompt,
        'plotting': VisualizationPrompt,
        'dashboard': VisualizationPrompt,
        
        # Machine learning nodes
        'classification': MachineLearningPrompt,
        'regression': MachineLearningPrompt,
        'clustering': MachineLearningPrompt,
        'machine_learning': MachineLearningPrompt,
        'ml_model': MachineLearningPrompt,
        'automl': MachineLearningPrompt,
        
        # Feature engineering nodes
        'feature_engineering': FeatureEngineeringPrompt,
        'feature_creation': FeatureEngineeringPrompt,
        'feature_transformation': FeatureEngineeringPrompt,
        
        # Anomaly detection nodes
        'anomaly_detection': AnomalyDetectionPrompt,
        'outlier_detection': AnomalyDetectionPrompt,
        'univariate_anomaly': AnomalyDetectionPrompt,
        'univariate_anomaly_detection': AnomalyDetectionPrompt,  # Add this missing mapping!
        'multivariate_anomaly': AnomalyDetectionPrompt,
        'multivariate_anomaly_detection': AnomalyDetectionPrompt,  # Add this too!
        'event_detection': AnomalyDetectionPrompt,
    }
    
    @classmethod
    def generate_prompt(cls, node_type: str, data: dict, node_id: str, context: dict = None) -> str:
        """
        Generate appropriate prompt based on node type
        
        Args:
            node_type: Type of the node (e.g., 'data_source', 'classification', etc.)
            data: Node output data
            node_id: Unique identifier for the node
            context: Additional context about the workflow
            
        Returns:
            Generated prompt string for the specific node type
        """
        # Normalize node type
        normalized_type = node_type.lower().replace(' ', '_').replace('-', '_')
        
        # Find appropriate prompt generator
        prompt_generator = cls.PROMPT_GENERATORS.get(normalized_type)
        
        if prompt_generator:
            try:
                return prompt_generator.generate_prompt(data, node_id, context)
            except Exception as e:
                # Log the specific error for debugging
                print(f"🚨 ERROR in {prompt_generator.__name__}.generate_prompt(): {str(e)}")
                import traceback
                traceback.print_exc()
                return cls._generate_fallback_prompt(node_type, data, node_id, str(e))
        else:
            return cls._generate_generic_prompt(node_type, data, node_id, context)
    
    @classmethod
    def _generate_generic_prompt(cls, node_type: str, data: dict, node_id: str, context: dict = None) -> str:
        """Generate a generic prompt for unknown node types"""
        
        data_keys = list(data.keys()) if isinstance(data, dict) else []
        data_summary = cls._analyze_generic_data(data)
        
        return f"""
🔧 **SPECIALIZED PROCESSING NODE - {node_id} | TYPE: {node_type.upper()}**

🎯 **NODE PROCESSING OVERVIEW**:
Operation Type: {node_type.replace('_', ' ').title()}
Processing Status: Completed
Data Components: {len(data_keys)} result elements
Analysis Scope: Specialized {node_type} operation

📊 **DATA PROCESSING RESULTS**:
{chr(10).join(data_summary) if data_summary else "⚠️ Processing results not available"}

💡 **SPECIALIZED NODE INTELLIGENCE REQUIREMENTS**:

1. **PROCESSING EFFECTIVENESS**: Evaluate the technical success and completeness of the {node_type} operation
2. **DATA TRANSFORMATION IMPACT**: Assess how this processing step affects data quality and statistical properties
3. **DATA VALUE ASSESSMENT**: Determine the analytical significance of this specialized processing
4. **INTEGRATION ANALYSIS**: Evaluate how this node's output integrates with the overall analytical workflow
5. **QUALITY ASSURANCE**: Assess the reliability and accuracy of the processing results
6. **TECHNICAL INSIGHTS**: Extract statistical insights specific to this processing type
7. **OPTIMIZATION OPPORTUNITIES**: Identify potential improvements for enhanced performance
8. **ANALYTICAL APPLICATIONS**: Recommend statistical applications for the processed data

🎯 **CRITICAL ANALYSIS REQUIREMENTS**:
- Provide SPECIFIC insights about the {node_type} processing results
- Assess DATA QUALITY IMPACT of this specialized operation
- Identify STATISTICAL OUTCOMES from the processing
- Evaluate ANALYTICAL INTEGRATION potential with other workflow components
- Recommend OPTIMIZATION strategies for enhanced data processing
- Establish QUALITY METRICS for statistical validation

⚡ **RESPONSE FOCUS**: Analyze the ACTUAL processing results and their statistical implications. Provide concrete, technical insights specific to this {node_type} operation and its contribution to the overall analytical workflow.
"""
    
    @classmethod
    def _analyze_generic_data(cls, data: dict) -> list:
        """Analyze generic data structure for insights"""
        insights = []
        
        if isinstance(data, dict):
            # Check for common data patterns
            if 'dataframe' in data:
                df = data['dataframe']
                if hasattr(df, 'shape'):
                    insights.append(f"📊 **Dataset Processing**: {df.shape[0]:,} records × {df.shape[1]} features processed")
                else:
                    insights.append("📊 **Dataset Processing**: Dataframe successfully processed")
            
            if 'statistics' in data:
                stats = data['statistics']
                if isinstance(stats, dict):
                    insights.append(f"📈 **Statistical Analysis**: {len(stats)} variables analyzed")
                else:
                    insights.append("📈 **Statistical Analysis**: Statistical processing completed")
            
            if 'results' in data:
                results = data['results']
                if isinstance(results, list):
                    insights.append(f"📋 **Processing Results**: {len(results)} result items generated")
                else:
                    insights.append("📋 **Processing Results**: Operation results available")
            
            if 'model' in data:
                insights.append("🤖 **Model Processing**: Machine learning model component processed")
            
            if 'charts' in data or 'visualizations' in data:
                charts = data.get('charts', data.get('visualizations', {}))
                chart_count = len(charts) if isinstance(charts, (dict, list)) else 1
                insights.append(f"📊 **Visualization Output**: {chart_count} visualizations generated")
            
            if 'predictions' in data:
                predictions = data['predictions']
                pred_count = len(predictions) if isinstance(predictions, list) else "Multiple"
                insights.append(f"🎯 **Prediction Output**: {pred_count} predictions generated")
            
            if 'anomalies' in data:
                anomalies = data['anomalies']
                anomaly_count = len(anomalies) if isinstance(anomalies, list) else "Multiple"
                insights.append(f"⚠️ **Anomaly Detection**: {anomaly_count} anomalies identified")
            
            # General data quality assessment
            if len(data) > 5:
                insights.append(f"🔧 **Comprehensive Processing**: {len(data)} processing components completed")
            elif len(data) > 2:
                insights.append(f"📊 **Standard Processing**: {len(data)} processing components completed")
            else:
                insights.append("⚡ **Basic Processing**: Essential processing completed")
        
        return insights
    
    @classmethod
    def _generate_fallback_prompt(cls, node_type: str, data: dict, node_id: str, error: str) -> str:
        """Generate fallback prompt when specific prompt generation fails"""
        
        return f"""
⚠️ **PROCESSING NODE - {node_id} | TYPE: {node_type.upper()}**

🔧 **NODE STATUS**: Processing completed with specialized {node_type} operation

📊 **AVAILABLE DATA**: {len(data) if isinstance(data, dict) else 'Processing results'} components available

💡 **ANALYSIS REQUIREMENTS**:
Please provide comprehensive analysis of this {node_type} node including:

1. **Processing Assessment**: Evaluate the technical effectiveness of the {node_type} operation
2. **Data Quality Impact**: Assess improvements or changes to data structure and properties
3. **Statistical Value**: Determine analytical significance of the processing results
4. **Integration Potential**: Evaluate connection with other analytical components
5. **Technical Insights**: Extract specific statistical findings
6. **Optimization Opportunities**: Identify potential algorithm improvements
7. **Analytical Applications**: Recommend statistical use cases

🎯 **Response Requirements**: Provide specific, technical insights based on the {node_type} processing results and their statistical implications.

Note: Advanced prompt generation temporarily unavailable - using standard analysis framework.
"""
    
    @classmethod
    def get_supported_node_types(cls) -> list:
        """Get list of all supported node types"""
        return list(cls.PROMPT_GENERATORS.keys())
    
    @classmethod
    def is_supported_node_type(cls, node_type: str) -> bool:
        """Check if a node type is supported"""
        normalized_type = node_type.lower().replace(' ', '_').replace('-', '_')
        return normalized_type in cls.PROMPT_GENERATORS
    
    @classmethod
    def add_prompt_generator(cls, node_type: str, prompt_generator_class):
        """Add a new prompt generator for a node type"""
        normalized_type = node_type.lower().replace(' ', '_').replace('-', '_')
        cls.PROMPT_GENERATORS[normalized_type] = prompt_generator_class
