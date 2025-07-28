"""
Advanced AI Prompt System and API endpoints
Provides AI-powered analysis and recommendations for datasets
"""

from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import logging

# Import advanced prompt system
from .node_prompt_router import NodePromptRouter
from .data_source_prompt import DataSourcePrompt
from .statistical_analysis_prompt import StatisticalAnalysisPrompt
from .data_cleaning_prompt import DataCleaningPrompt
from .visualization_prompt import VisualizationPrompt
from .machine_learning_prompt import MachineLearningPrompt
from .eda_prompt import EDAPrompt
from .feature_engineering_prompt import FeatureEngineeringPrompt
from .anomaly_detection_prompt import AnomalyDetectionPrompt
from app.services.eda_service import eda_service
from app.models.dataset import Dataset

# Create blueprint
ai_bp = Blueprint('ai', __name__)
api = Api(ai_bp)

logger = logging.getLogger(__name__)

class AIInsightsResource(Resource):
    """Resource for generating AI insights from dataset"""
    
    def post(self, dataset_id):
        """
        Generate AI insights for a dataset
        
        Args:
            dataset_id: ID of the dataset to analyze
            
        Returns:
            JSON response with AI-generated insights
        """
        try:
            # Get dataset
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return {
                    'success': False,
                    'error': 'Dataset not found'
                }, 404
            
            # Get existing EDA data
            if not dataset.eda_generated or not dataset.eda_results:
                return {
                    'success': False,
                    'error': 'EDA analysis not found. Please run EDA analysis first.'
                }, 400
            
            # Prepare EDA data for AI analysis
            eda_data = {
                'results': dataset.eda_results,
                'charts': dataset.eda_charts,
                'dataset_info': {
                    'name': dataset.name,
                    'rows': dataset.rows_count,
                    'columns': dataset.columns_count,
                    'data_quality_score': dataset.data_quality_score
                }
            }
            
            # Generate AI insights
            logger.info(f"Generating AI insights for dataset {dataset_id}")
            from app.services.ai_service_advanced import AdvancedAIInsightService
            ai_service = AdvancedAIInsightService()
            # Use single node analysis for dataset insights
            insights = ai_service.generate_single_node_insights(
                eda_data, 
                f"dataset_{dataset_id}",
                node_type="data_source"
            )
            
            return {
                'success': True,
                'dataset_id': dataset_id,
                'dataset_name': dataset.name,
                'insights': insights,
                'message': 'AI insights generated successfully'
            }, 200
            
        except Exception as e:
            logger.error(f"Error generating AI insights for dataset {dataset_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }, 500
    
    def get(self, dataset_id):
        """
        Get cached AI insights for a dataset
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            JSON response with cached insights or instruction to generate new ones
        """
        try:
            # For now, return instruction to generate new insights
            # In future, implement caching mechanism
            return {
                'success': False,
                'message': 'No cached insights found. Use POST to generate new insights.',
                'cache_available': False
            }, 404
            
        except Exception as e:
            logger.error(f"Error retrieving AI insights for dataset {dataset_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }, 500

class AICapabilitiesResource(Resource):
    """Resource for checking ultra-advanced AI service capabilities"""
    
    def get(self):
        """
        Get ultra-advanced AI service capabilities and status
        
        Returns:
            JSON response with comprehensive AI service information
        """
        try:
            return {
                'success': True,
                'ai_enabled': True,
                'service_level': 'Ultra-Advanced Enterprise AI',
                'model': 'nvidia/llama-3.3-nemotron-super-49b-v1',
                'provider': 'NVIDIA',
                'intelligence_framework': 'Advanced Workflow Intelligence System',
                'capabilities': [
                    'Ultra-Advanced Dataset Analysis',
                    'Node-Specific Intelligence Generation',
                    'Business-Ready Strategic Insights',
                    'Executive-Level Pattern Recognition',
                    'Cross-Node Correlation Intelligence',
                    'Predictive Business Intelligence',
                    'Risk Assessment & Opportunity Identification',
                    'ROI-Focused Recommendations',
                    'Competitive Advantage Analysis',
                    'Implementation Roadmap Generation'
                ],
                'supported_features': [
                    'Multi-Node Workflow Analysis',
                    'Contextual Business Intelligence',
                    'Advanced Statistical Pattern Recognition',
                    'Predictive Modeling Intelligence',
                    'Real-Time Anomaly Risk Assessment',
                    'Strategic Feature Engineering',
                    'Executive Dashboard Insights',
                    'Operational Excellence Optimization'
                ],
                'node_specific_intelligence': {
                    'data_source': 'Business metrics identification and data quality scoring',
                    'statistical_analysis': 'Advanced pattern recognition and correlation intelligence',
                    'eda_analysis': 'Strategic exploratory insights with business context',
                    'data_cleaning': 'Quality optimization with business impact assessment',
                    'feature_engineering': 'Strategic feature innovation with business logic',
                    'visualization': 'Executive-level visual intelligence and communication',
                    'machine_learning': 'Predictive business intelligence and strategic impact',
                    'anomaly_detection': 'Risk intelligence with business impact evaluation'
                },
                'business_intelligence_areas': [
                    'Revenue Optimization',
                    'Cost Reduction Strategies',
                    'Risk Mitigation Planning',
                    'Market Opportunity Identification',
                    'Operational Excellence',
                    'Strategic Innovation',
                    'Competitive Positioning',
                    'Investment Decision Support'
                ],
                'analysis_depth_levels': [
                    'Basic: Standard analytical processing',
                    'Intermediate: Advanced pattern recognition',
                    'Advanced: Strategic business intelligence',
                    'Expert: Executive-level insights',
                    'Enterprise: Comprehensive strategic analysis'
                ]
            }, 200
            
        except Exception as e:
            logger.error(f"Error checking AI capabilities: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ai_enabled': False
            }, 500

class AdvancedAnalysisResource(Resource):
    """Resource for ultra-advanced streaming AI analysis with node-specific intelligence"""
    
    def post(self, dataset_id):
        """
        Generate ultra-advanced streaming AI analysis for dataset with node-specific prompts
        
        Args:
            dataset_id: ID of the dataset to analyze
            
        Returns:
            JSON response with detailed streaming AI analysis based on connected nodes
        """
        try:
            # Get dataset
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return {
                    'success': False,
                    'error': 'Dataset not found'
                }, 404
            
            # Get existing EDA data
            if not dataset.eda_generated or not dataset.eda_results:
                return {
                    'success': False,
                    'error': 'EDA analysis not found. Please run EDA analysis first.'
                }, 400
            
            # Prepare comprehensive data for advanced AI analysis
            eda_data = {
                'results': dataset.eda_results,
                'charts': dataset.eda_charts,
                'dataset_info': {
                    'name': dataset.name,
                    'rows': dataset.rows_count,
                    'columns': dataset.columns_count,
                    'data_quality_score': dataset.data_quality_score
                }
            }
            
            # Generate ultra-advanced streaming analysis with node-specific intelligence
            from app.services.ai_service_advanced import AdvancedAIInsightService
            ai_service = AdvancedAIInsightService()
            result = ai_service.generate_single_node_insights(
                eda_data,
                f"eda_dataset_{dataset_id}",
                node_type="eda"
            )
            
            # The new service returns insights directly, no 'success' key needed
            return {
                'success': True,
                'analysis': result,
                'dataset_id': dataset_id,
                'dataset_name': dataset.name,
                'analysis_type': 'ultra_advanced_node_specific',
                'intelligence_level': 'expert',
                'business_ready': True
            }, 200
                
        except Exception as e:
            logger.error(f"Error generating ultra-advanced analysis for dataset {dataset_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }, 500
    
    def get(self, dataset_id):
        """
        Get information about available ultra-advanced analysis features
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            JSON response with available ultra-advanced features and node-specific capabilities
        """
        try:
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return {
                    'success': False,
                    'error': 'Dataset not found'
                }, 404
                
            return {
                'success': True,
                'dataset_id': dataset_id,
                'dataset_name': dataset.name,
                'ultra_advanced_features': [
                    'Node-Specific AI Intelligence',
                    'Business-Ready Strategic Analysis',
                    'Ultra-Advanced Pattern Recognition',
                    'Executive-Level Insights Generation',
                    'Cross-Node Correlation Intelligence',
                    'Predictive Business Intelligence',
                    'Risk Assessment & Opportunity Identification',
                    'ROI-Focused Recommendations',
                    'Competitive Advantage Analysis',
                    'Implementation Roadmap Generation'
                ],
                'node_specific_capabilities': [
                    'Data Source Intelligence with Business Context',
                    'Statistical Analysis with Advanced Pattern Recognition',
                    'EDA Intelligence with Strategic Insights',
                    'Feature Engineering with Business Logic',
                    'ML Model Intelligence with Predictive Impact',
                    'Anomaly Detection with Risk Assessment',
                    'Visualization Intelligence with Executive Communication',
                    'Data Cleaning with Quality Optimization'
                ],
                'analysis_types': [
                    'ultra_comprehensive_streaming',
                    'node_specific_intelligence',
                    'business_strategy_analysis',
                    'executive_dashboard_insights',
                    'predictive_opportunity_analysis',
                    'competitive_intelligence_extraction'
                ],
                'intelligence_levels': [
                    'basic',
                    'intermediate', 
                    'advanced',
                    'expert',
                    'executive'
                ],
                'business_impact_areas': [
                    'Revenue Optimization',
                    'Cost Reduction',
                    'Risk Mitigation',
                    'Market Opportunities',
                    'Operational Excellence',
                    'Strategic Innovation',
                    'Competitive Positioning',
                    'Investment Planning'
                ]
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting advanced analysis info: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }, 500

class NodeSpecificAnalysisResource(Resource):
    """Resource for ultra-advanced node-specific AI analysis with contextual intelligence"""
    
    def post(self):
        """
        Generate node-specific AI analysis based on connected workflow nodes
        
        Expects JSON payload with:
        {
            "nodes": [
                {
                    "node_id": "string",
                    "node_type": "data_source|statistical_analysis|visualization|ml_model|etc",
                    "node_data": {...}
                }
            ],
            "workflow_context": {
                "complexity_level": "basic|intermediate|advanced",
                "business_domain": "optional domain context"
            }
        }
        
        Returns:
            JSON response with ultra-advanced node-specific AI analysis
        """
        try:
            data = request.get_json()
            if not data or 'nodes' not in data:
                return {
                    'success': False,
                    'error': 'Invalid request format. Expected JSON with nodes array.'
                }, 400
            
            nodes = data.get('nodes', [])
            workflow_context = data.get('workflow_context', {})
            
            if not nodes:
                return {
                    'success': False,
                    'error': 'No nodes provided for analysis.'
                }, 400
            
            # Prepare comprehensive data structure for AI analysis
            comprehensive_data = {
                'workflow_summary': {
                    'total_nodes': len(nodes),
                    'node_types': [node.get('node_type', 'unknown') for node in nodes],
                    'node_outputs': {}
                },
                'workflow_context': workflow_context
            }
            
            # Process each node's data
            for node in nodes:
                node_id = node.get('node_id', f'node_{len(comprehensive_data["workflow_summary"]["node_outputs"])}')
                node_type = node.get('node_type', 'unknown')
                node_data = node.get('node_data', {})
                
                comprehensive_data['workflow_summary']['node_outputs'][node_id] = {
                    'type': node_type,
                    'data': node_data
                }
            
            # Generate ultra-advanced AI insights with node-specific intelligence
            logger.info(f"Generating node-specific AI analysis for {len(nodes)} connected nodes")
            from app.services.ai_service_advanced import AdvancedAIInsightService
            ai_service = AdvancedAIInsightService()
            insights = ai_service.generate_comprehensive_workflow_insights(comprehensive_data)
            
            return {
                'success': True,
                'analysis_type': 'ultra_advanced_node_specific',
                'nodes_analyzed': len(nodes),
                'node_types': [node.get('node_type') for node in nodes],
                'insights': insights,
                'workflow_intelligence': {
                    'complexity_assessment': workflow_context.get('complexity_level', 'intermediate'),
                    'business_readiness': True,
                    'strategic_value': 'high',
                    'implementation_ready': True
                },
                'timestamp': insights.get('timestamp'),
                'message': f'Ultra-advanced AI analysis completed for {len(nodes)} connected nodes'
            }, 200
            
        except Exception as e:
            logger.error(f"Error in node-specific AI analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'analysis_type': 'node_specific_analysis_failed'
            }, 500

    def get(self):
        """
        Get information about node-specific analysis capabilities
        
        Returns:
            JSON response with supported node types and analysis capabilities
        """
        try:
            return {
                'success': True,
                'service': 'Ultra-Advanced Node-Specific AI Analysis',
                'supported_node_types': [
                    {
                        'type': 'data_source',
                        'description': 'Primary data source with business intelligence focus',
                        'analysis_depth': 'Comprehensive data profiling, quality assessment, business metrics identification'
                    },
                    {
                        'type': 'statistical_analysis',
                        'description': 'Advanced statistical operations with pattern recognition',
                        'analysis_depth': 'Correlation intelligence, outlier detection, distribution analysis, business recommendations'
                    },
                    {
                        'type': 'eda_analysis',
                        'description': 'Exploratory data analysis with strategic insights',
                        'analysis_depth': 'Pattern discovery, relationship mapping, feature engineering opportunities'
                    },
                    {
                        'type': 'data_cleaning',
                        'description': 'Data quality optimization with impact assessment',
                        'analysis_depth': 'Quality enhancement evaluation, business readiness assessment'
                    },
                    {
                        'type': 'feature_engineering',
                        'description': 'Advanced feature creation with business logic',
                        'analysis_depth': 'Feature innovation analysis, strategic enhancement evaluation'
                    },
                    {
                        'type': 'visualization',
                        'description': 'Visual intelligence with executive communication focus',
                        'analysis_depth': 'Pattern visualization, stakeholder communication optimization'
                    },
                    {
                        'type': 'machine_learning',
                        'description': 'ML models with predictive business impact',
                        'analysis_depth': 'Predictive intelligence, model performance, business applications'
                    },
                    {
                        'type': 'anomaly_detection',
                        'description': 'Risk intelligence with business impact assessment',
                        'analysis_depth': 'Risk identification, business impact evaluation, mitigation strategies'
                    }
                ],
                'intelligence_levels': [
                    'basic', 'intermediate', 'advanced', 'expert', 'executive'
                ],
                'analysis_capabilities': [
                    'Cross-node pattern recognition',
                    'Business intelligence extraction',
                    'Strategic recommendation generation',
                    'ROI analysis and quantification',
                    'Risk assessment and mitigation',
                    'Competitive advantage identification',
                    'Implementation roadmap creation',
                    'Executive-level insight synthesis'
                ]
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting node-specific analysis info: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }, 500

# Register API resources
api.add_resource(AIInsightsResource, '/datasets/<int:dataset_id>/ai-insights')
api.add_resource(AdvancedAnalysisResource, '/datasets/<int:dataset_id>/advanced-analysis')
api.add_resource(AICapabilitiesResource, '/ai/capabilities')
api.add_resource(NodeSpecificAnalysisResource, '/ai/node-specific-analysis')
