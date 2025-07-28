"""
Advanced AI Service with Node-Specific Prompts
Completely rewritten for sophisticated multi-node workflow analysis with proper node data handling
Each node type has its own advanced prompt that analyzes the actual node output data
"""

import os
import json
import logging
import time
import threading
import pandas as pd
import numpy as np
import statistics
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment

# NVIDIA API client
try:
    from openai import OpenAI
    NVIDIA_API_AVAILABLE = True
except ImportError:
    NVIDIA_API_AVAILABLE = False

# Import the prompt router instead of individual generators
from app.api.ai.node_prompt_router import NodePromptRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAIInsightService:
    """
    Advanced AI Service that generates sophisticated insights using node-specific prompts
    for connected workflow analysis. Each node type gets its own advanced prompt based on actual data.
    """
    
    def __init__(self):
        """Initialize the advanced AI service with NVIDIA API and node-specific prompt generators"""
        self.api_key = os.getenv('NVIDIA_API_KEY')
        self.base_url = "https://integrate.api.nvidia.com/v1"
        
        # Debug: Check if API key is loaded
        if self.api_key:
            logger.info(f"NVIDIA API key loaded: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else '[short key]'}")
        else:
            logger.error("NVIDIA API key not found in environment variables!")
        
        # Initialize client if API is available
        self.client = None
        if NVIDIA_API_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info("NVIDIA API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIA API client: {str(e)}")
        else:
            if not NVIDIA_API_AVAILABLE:
                logger.error("OpenAI library not available for NVIDIA API")
            if not self.api_key:
                logger.error("NVIDIA API key not provided")
        
        # Model priority list (try these in order)
        self.models = [
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "meta/llama-3.1-405b-instruct", 
        ]
        
        # Current working model
        self.model = self.models[0]
        
        # Initialize the node prompt router
        self.prompt_router = NodePromptRouter()

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 characters per token"""
        return len(str(text)) // 4

    def _truncate_data_for_api(self, data: Dict[str, Any], max_tokens: int = 120000) -> Dict[str, Any]:
        """
        Truncate data to fit within token limits while preserving the most important information
        """
        
        def truncate_dataframe_info(df_info: Dict[str, Any], max_tokens_per_df: int = 10000) -> Dict[str, Any]:
            """Truncate DataFrame information to essential parts"""
            truncated = {}
            current_tokens = 0
            
            # Always include basic info
            for key in ['shape', 'columns', 'dtypes', 'memory_usage']:
                if key in df_info and current_tokens < max_tokens_per_df:
                    value = df_info[key]
                    if key == 'columns' and isinstance(value, list) and len(value) > 50:
                        # Limit columns to first 50
                        truncated[key] = value[:50] + ['... (truncated)']
                    else:
                        truncated[key] = value
                    current_tokens += self._estimate_tokens(str(value))
            
            # Include summary statistics if space allows
            if 'describe' in df_info and current_tokens < max_tokens_per_df * 0.7:
                desc = df_info['describe']
                if isinstance(desc, dict):
                    # Limit to first 10 columns for describe
                    truncated_desc = {}
                    col_count = 0
                    for col, stats in desc.items():
                        if col_count < 10:
                            truncated_desc[col] = stats
                            col_count += 1
                        else:
                            truncated_desc['... (more columns)'] = 'truncated'
                            break
                    truncated['describe'] = truncated_desc
                else:
                    truncated['describe'] = desc
            
            # Add sample data if space allows
            if 'head' in df_info and current_tokens < max_tokens_per_df * 0.5:
                head_data = df_info['head']
                if isinstance(head_data, list) and len(head_data) > 5:
                    truncated['head'] = head_data[:5]  # Only first 5 rows
                else:
                    truncated['head'] = head_data
            
            return truncated
        
        def truncate_node_data(node_data: Dict[str, Any], max_tokens_per_node: int = 8000) -> Dict[str, Any]:
            """Truncate individual node data"""
            truncated = {
                'node_id': node_data.get('node_id'),
                'node_type': node_data.get('node_type'),
                'node_name': node_data.get('node_name'),
                'status': node_data.get('status'),
                'execution_time': node_data.get('execution_time')
            }
            
            current_tokens = self._estimate_tokens(json.dumps(truncated))
            
            # Add result summary if available and space allows
            if 'result_summary' in node_data and current_tokens < max_tokens_per_node * 0.3:
                result_summary = node_data['result_summary']
                if isinstance(result_summary, dict):
                    truncated['result_summary'] = truncate_dataframe_info(result_summary, max_tokens_per_node // 2)
                else:
                    truncated['result_summary'] = str(result_summary)[:1000]  # Limit string length
            
            # Add config if space allows
            if 'config' in node_data and current_tokens < max_tokens_per_node * 0.6:
                config = node_data['config']
                if isinstance(config, dict):
                    # Keep only essential config parameters
                    essential_config = {}
                    for key, value in list(config.items())[:10]:  # Limit to 10 config items
                        essential_config[key] = str(value)[:100]  # Limit value length
                    truncated['config'] = essential_config
            
            return truncated
        
        # Start truncation process
        truncated_data = {
            'workflow_summary': data.get('workflow_summary', {}),
            'execution_summary': data.get('execution_summary', {}),
        }
        
        # Truncate node results
        if 'node_results' in data:
            node_results = data['node_results']
            total_nodes = len(node_results)
            max_tokens_per_node = min(8000, max_tokens // max(total_nodes, 1))
            
            truncated_nodes = {}
            processed_tokens = 0
            
            for node_id, node_data in node_results.items():
                if processed_tokens < max_tokens * 0.8:  # Reserve 20% for other data
                    truncated_node = truncate_node_data(node_data, max_tokens_per_node)
                    truncated_nodes[node_id] = truncated_node
                    processed_tokens += self._estimate_tokens(json.dumps(truncated_node))
                else:
                    # Add a summary for remaining nodes
                    truncated_nodes[f'... and {total_nodes - len(truncated_nodes)} more nodes'] = {
                        'truncated': True,
                        'reason': 'Token limit reached'
                    }
                    break
            
            truncated_data['node_results'] = truncated_nodes
        
        # Add metadata about truncation
        truncated_data['_truncation_info'] = {
            'original_nodes': len(data.get('node_results', {})),
            'truncated_nodes': len(truncated_data.get('node_results', {})),
            'estimated_tokens': self._estimate_tokens(json.dumps(truncated_data)),
            'max_token_limit': max_tokens
        }
        
        return truncated_data

    def _prepare_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """
        Prepare analysis prompt with truncated data
        """
        # Truncate data first
        truncated_data = self._truncate_data_for_api(data)
        
        prompt = f"""
## WORKFLOW DATA ANALYSIS REQUEST

You are an expert data scientist analyzing workflow execution results. Provide comprehensive insights based on the following data:

### WORKFLOW SUMMARY
{json.dumps(truncated_data.get('workflow_summary', {}), indent=2)}

### EXECUTION SUMMARY  
{json.dumps(truncated_data.get('execution_summary', {}), indent=2)}

### NODE RESULTS
{json.dumps(truncated_data.get('node_results', {}), indent=2)}

### TRUNCATION INFO
{json.dumps(truncated_data.get('_truncation_info', {}), indent=2)}

Please provide:
1. **DATA SUMMARY**: Overview of the data processed and workflow execution
2. **KEY INSIGHTS**: Important patterns, trends, or findings from the analysis
3. **DATA QUALITY ASSESSMENT**: Issues, anomalies, or recommendations
4. **STATISTICAL HIGHLIGHTS**: Significant statistical findings
5. **RECOMMENDATIONS**: Next steps or improvements for the analysis

Focus on actionable insights and be specific about the findings from each node's results.
"""
        
        return prompt

    def _create_fallback_response(self, workflow_data: Dict[str, Any], task_id: str, error: str = None) -> Dict[str, Any]:
        """
        Create a comprehensive fallback response when AI API fails
        """
        node_results = workflow_data.get('node_results', {})
        execution_summary = workflow_data.get('execution_summary', {})
        
        # Generate basic insights from the data
        insights = []
        
        # Analyze execution summary
        if execution_summary:
            total_nodes = execution_summary.get('total_nodes', 0)
            completed_nodes = execution_summary.get('nodes_completed', 0)
            execution_time = execution_summary.get('execution_time', 0)
            
            insights.append(f"Workflow executed {completed_nodes}/{total_nodes} nodes successfully")
            if execution_time > 0:
                insights.append(f"Total execution time: {execution_time:.2f} seconds")
        
        # Analyze node results
        data_nodes = []
        analysis_nodes = []
        viz_nodes = []
        
        for node_id, node_data in node_results.items():
            node_type = node_data.get('node_type', '')
            node_name = node_data.get('node_name', node_id)
            
            if 'data_source' in node_type:
                data_nodes.append(node_name)
            elif any(analysis_type in node_type for analysis_type in ['stats', 'correlation', 'analysis']):
                analysis_nodes.append(node_name)
            elif 'plot' in node_type or 'visualization' in node_type:
                viz_nodes.append(node_name)
        
        if data_nodes:
            insights.append(f"Data sources processed: {', '.join(data_nodes)}")
        if analysis_nodes:
            insights.append(f"Statistical analysis performed: {', '.join(analysis_nodes)}")
        if viz_nodes:
            insights.append(f"Visualizations generated: {', '.join(viz_nodes)}")
        
        fallback_response = f"""
## DATA SUMMARY
{"NVIDIA AI API encountered token limit issues. " if error and "token" in error.lower() else ""}Based on the workflow structure and data patterns, this analysis indicates a data processing pipeline with multiple connected nodes requiring integrated assessment.

## INTEGRATED DATA ANALYSIS  
The connected nodes represent an analytical workflow combining data processing, analysis, and visualization. Each component contributes to the overall data processing and analysis pipeline.

## ANALYSIS RESULTS
{'• ' + chr(10).join(f'{insight}' for insight in insights) if insights else '• Data processing pipeline established with proper node connections'}
• Data quality assessment needed for reliable results
• Pattern detection and trend analysis possible with available data

**Note**: This is a fallback analysis due to {"token limit exceeded in AI API call" if error and "token" in error.lower() else "AI API authentication issues"}. To enable comprehensive AI-powered insights:
1. Verify your NVIDIA API key is valid and active
2. Check that the API key has access to the required models
3. Ensure the API key is properly set in environment variables as 'NVIDIA_API_KEY'
{"4. Consider reducing data size for complex workflows to stay within token limits" if error and "token" in error.lower() else ""}

For comprehensive data analysis, please configure NVIDIA API access.
"""
        
        return {
            'insights': insights,
            'full_response': fallback_response,
            'workflow_analysis': {
                'data_sources': len(data_nodes),
                'analysis_nodes': len(analysis_nodes),
                'visualization_nodes': len(viz_nodes),
                'total_nodes_processed': len(node_results)
            },
            'node_summary': {node_id: {'name': data.get('node_name', node_id), 'type': data.get('node_type', 'unknown')} 
                            for node_id, data in node_results.items()},
            'success': False,
            'error': error,
            'metadata': {
                'model_used': 'fallback',
                'processing_mode': 'fallback_analysis',
                'fallback_reason': error or 'API unavailable'
            },
            'task_id': task_id,
            'completed_at': datetime.now().isoformat(),
            'processing_mode': 'fallback'
        }

    def _extract_insights(self, ai_response: str) -> List[str]:
        """Extract key insights from AI response"""
        insights = []
        
        # Look for bullet points or numbered lists
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                insight = line[1:].strip()
                if len(insight) > 10:  # Minimum length for meaningful insight
                    insights.append(insight)
            elif line and any(keyword in line.lower() for keyword in ['insight:', 'finding:', 'result:', 'conclusion:']):
                insights.append(line)
        
        # If no structured insights found, extract sentences containing key terms
        if not insights:
            sentences = ai_response.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in ['significant', 'important', 'notable', 'key', 'main', 'primary']):
                    insights.append(sentence + '.')
        
        return insights[:10]  # Limit to top 10 insights

    def _extract_workflow_analysis(self, ai_response: str) -> Dict[str, Any]:
        """Extract workflow analysis from AI response"""
        analysis = {
            'data_quality': 'Not assessed',
            'statistical_significance': 'Not evaluated',
            'patterns_detected': [],
            'recommendations': []
        }
        
        # Extract patterns from response
        if 'pattern' in ai_response.lower():
            pattern_sentences = [s.strip() for s in ai_response.split('.') if 'pattern' in s.lower()]
            analysis['patterns_detected'] = pattern_sentences[:5]
        
        # Extract recommendations
        if 'recommend' in ai_response.lower():
            recommendation_sentences = [s.strip() for s in ai_response.split('.') if 'recommend' in s.lower()]
            analysis['recommendations'] = recommendation_sentences[:5]
        
        # Extract data quality assessment
        if 'quality' in ai_response.lower():
            quality_sentences = [s.strip() for s in ai_response.split('.') if 'quality' in s.lower()]
            if quality_sentences:
                analysis['data_quality'] = quality_sentences[0]
        
        return analysis

    def _extract_node_summary(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract node summary from workflow data"""
        node_results = workflow_data.get('node_results', {})
        summary = {}
        
        for node_id, node_data in node_results.items():
            summary[node_id] = {
                'name': node_data.get('node_name', node_id),
                'type': node_data.get('node_type', 'unknown'),
                'status': node_data.get('status', 'completed'),
                'execution_time': node_data.get('execution_time', 0)
            }
        
        return summary
    
    def generate_comprehensive_workflow_insights(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive AI insights for connected workflow nodes
        
        Args:
            workflow_data: Dictionary containing workflow data in various formats
            
        Returns:
            Dictionary containing comprehensive AI analysis
        """
        try:
            logger.info("Starting comprehensive workflow analysis")
            logger.info(f"Received workflow data keys: {list(workflow_data.keys())}")
            
            # Debug: Log sample of workflow data structure
            for key, value in list(workflow_data.items())[:3]:  # Log first 3 items
                if isinstance(value, dict):
                    logger.info(f"  {key}: dict with keys {list(value.keys())[:5]}")
                elif isinstance(value, list):
                    logger.info(f"  {key}: list with {len(value)} items")
                else:
                    logger.info(f"  {key}: {type(value).__name__}")
            
            # Normalize workflow data to expected format
            normalized_data = self._normalize_workflow_data(workflow_data)
            
            # Extract nodes and context
            nodes = normalized_data.get('nodes', {})
            workflow_context = normalized_data.get('workflow_context', {})
            
            logger.info(f"Normalized data: {len(nodes)} nodes, context: {list(workflow_context.keys())}")
            
            if not nodes:
                logger.warning(f"No nodes found in workflow data. Available keys: {list(workflow_data.keys())}")
                return self._generate_error_response("No analyzable nodes found in workflow data")
            
            # Analyze workflow structure
            workflow_analysis = self._analyze_workflow_structure(nodes, workflow_context)
            
            # Generate node-specific prompts with actual data
            node_prompts = self._generate_node_prompts(nodes, workflow_context)
            
            if not node_prompts:
                logger.warning("No valid node prompts generated")
                return self._generate_error_response("No valid nodes available for analysis")
            
            # Create comprehensive workflow prompt
            comprehensive_prompt = self._create_comprehensive_workflow_prompt(
                nodes, workflow_context, node_prompts, workflow_analysis
            )
            
            # Generate AI insights using streaming API
            ai_response = self._call_nvidia_api_streaming(comprehensive_prompt)
            
            # Process and structure the response
            structured_insights = self._structure_ai_response(ai_response, nodes, workflow_analysis)
            
            # Add metadata
            structured_insights['success'] = True
            structured_insights['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'nodes_analyzed': len(nodes),
                'workflow_complexity': workflow_analysis.get('complexity_level', 'moderate'),
                'ai_model_used': self.model,
                'prompt_system': 'advanced_node_specific',
                'analysis_type': 'comprehensive_workflow'
            }
            
            logger.info(f"Successfully generated insights for {len(nodes)} nodes")
            return structured_insights
            
        except Exception as e:
            logger.error(f"Error in comprehensive workflow analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_error_response(str(e))
    
    def generate_single_node_insights(self, node_data: Dict[str, Any], node_id: str, 
                                    node_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate AI insights for a single node
        
        Args:
            node_data: The node's data/output
            node_id: Unique identifier for the node
            node_type: Type of the node (e.g., 'data_source', 'classification')
            context: Optional context about the broader workflow
            
        Returns:
            Dictionary containing AI analysis for the single node
        """
        try:
            logger.info(f"Generating insights for single node: {node_id} (type: {node_type})")
            
            # Generate node-specific prompt
            node_prompt = self.prompt_router.generate_prompt(node_type, node_data, node_id, context)
            
            # Create focused analysis prompt
            focused_prompt = self._create_focused_node_prompt(node_prompt, node_type, context)
            
            # Generate AI insights
            ai_response = self._call_nvidia_api_streaming(focused_prompt)
            
            # Structure the response
            structured_insights = {
                'success': True,
                'node_id': node_id,
                'node_type': node_type,
                'insights': self._parse_ai_response(ai_response),
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'ai_model_used': self.model,
                    'prompt_system': 'advanced_node_specific',
                    'analysis_type': 'single_node'
                }
            }
            
            logger.info(f"Successfully generated insights for node {node_id}")
            return structured_insights
            
        except Exception as e:
            logger.error(f"Error generating insights for node {node_id}: {str(e)}")
            return self._generate_error_response(str(e), node_id)
    
    def _analyze_workflow_structure(self, nodes: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure and complexity of the workflow"""
        analysis = {
            'total_nodes': len(nodes),
            'node_types': {},
            'connections': [],
            'data_flow': [],
            'complexity_level': 'basic',
            'analysis_categories': {
                'data_sources': [],
                'statistical_nodes': [],
                'visualization_nodes': [],
                'ml_nodes': [],
                'processing_nodes': [],
                'eda_nodes': [],
                'cleaning_nodes': [],
                'feature_engineering_nodes': [],
                'anomaly_detection_nodes': []
            }
        }
        
        # Categorize nodes by type, excluding AI nodes and nodes without valid data
        for node_id, node_info in nodes.items():
            node_type = node_info.get('type', 'unknown')
            node_data = node_info.get('data', {})
            
            # Skip AI nodes
            if self._is_ai_node(node_type, node_id):
                logger.info(f"Excluding AI node {node_id} from workflow structure analysis")
                continue
            
            # Skip nodes without valid data
            if not self._has_valid_data(node_data, node_type):
                logger.info(f"Excluding node {node_id} from workflow structure analysis - no valid data")
                continue
            
            # Count node types
            if node_type not in analysis['node_types']:
                analysis['node_types'][node_type] = 0
            analysis['node_types'][node_type] += 1
            
            # Categorize for analysis
            if node_type in ['data_source', 'csv_upload', 'file_upload']:
                analysis['analysis_categories']['data_sources'].append(node_id)
            elif node_type in ['statistical_analysis', 'descriptive_stats', 'processing']:
                analysis['analysis_categories']['statistical_nodes'].append(node_id)
            elif node_type in ['visualization', 'chart', 'plotting', 'basic_plots', 'advanced_plots']:
                analysis['analysis_categories']['visualization_nodes'].append(node_id)
            elif node_type in ['classification', 'regression', 'clustering', 'automl', 'machine_learning']:
                analysis['analysis_categories']['ml_nodes'].append(node_id)
            elif node_type in ['eda', 'exploratory_data_analysis', 'eda_analysis']:
                analysis['analysis_categories']['eda_nodes'].append(node_id)
            elif node_type in ['data_cleaning', 'preprocessing']:
                analysis['analysis_categories']['cleaning_nodes'].append(node_id)
            elif node_type in ['feature_engineering', 'feature_creation']:
                analysis['analysis_categories']['feature_engineering_nodes'].append(node_id)
            elif node_type in ['anomaly_detection', 'outlier_detection', 'univariate_anomaly_detection', 'multivariate_anomaly_detection']:
                analysis['analysis_categories']['anomaly_detection_nodes'].append(node_id)
            else:
                analysis['analysis_categories']['processing_nodes'].append(node_id)
            
            # Track connections
            connections = node_info.get('connections', [])
            for connected_node in connections:
                analysis['connections'].append((node_id, connected_node))
        
        # Update total nodes to reflect only analyzed nodes
        total_analyzed_nodes = sum(len(category_nodes) for category_nodes in analysis['analysis_categories'].values())
        analysis['total_analyzed_nodes'] = total_analyzed_nodes
        
        # Determine complexity level based on analyzed nodes
        total_categories = sum(1 for category_nodes in analysis['analysis_categories'].values() if category_nodes)
        
        if total_categories >= 6 or total_analyzed_nodes >= 10:
            analysis['complexity_level'] = 'advanced'
        elif total_categories >= 4 or total_analyzed_nodes >= 5:
            analysis['complexity_level'] = 'intermediate'
        else:
            analysis['complexity_level'] = 'basic'
        
        # Analyze data flow patterns
        data_sources = analysis['analysis_categories']['data_sources']
        ml_nodes = analysis['analysis_categories']['ml_nodes']
        viz_nodes = analysis['analysis_categories']['visualization_nodes']
        
        if data_sources and ml_nodes:
            analysis['data_flow'].append('data_to_ml_pipeline')
        if data_sources and viz_nodes:
            analysis['data_flow'].append('data_to_visualization_pipeline')
        if len(analysis['analysis_categories']['statistical_nodes']) > 0:
            analysis['data_flow'].append('statistical_analysis_pipeline')
        
        return analysis
    
    def _generate_node_prompts(self, nodes: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Generate sophisticated prompts for each node using actual node data"""
        node_prompts = {}
        
        for node_id, node_info in nodes.items():
            try:
                node_type = node_info.get('type', 'unknown')
                node_data = node_info.get('data', {})
                
                # Skip AI nodes - they should not be included in AI analysis
                if self._is_ai_node(node_type, node_id):
                    logger.info(f"Skipping AI node {node_id} (type: {node_type}) from analysis")
                    continue
                
                # Validate that node has meaningful data
                if not self._has_valid_data(node_data, node_type):
                    logger.warning(f"Skipping node {node_id} (type: {node_type}) - no valid data available")
                    logger.warning(f"  Data keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'non-dict'}")
                    if isinstance(node_data, dict):
                        logger.warning(f"  Data sample: {str(node_data)[:200]}...")
                    continue
                
                logger.info(f"Generating prompt for node {node_id} (type: {node_type}) with data keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'non-dict data'}")
                
                # Summarize data before generating prompt to control size
                summarized_data = self._summarize_node_data_for_analysis(node_data, node_type)
                logger.info(f"  Summarized data keys: {list(summarized_data.keys()) if isinstance(summarized_data, dict) else 'non-dict'}")
                
                # Use the prompt router to generate the prompt with summarized data
                try:
                    prompt = self.prompt_router.generate_prompt(
                        node_type=node_type,
                        data=summarized_data,
                        node_id=node_id,
                        context=context
                    )
                    
                    # Check if prompt indicates an error or no data
                    if "CRITICAL ERROR" in prompt or "No data available" in prompt:
                        logger.warning(f"Prompt generator indicated data issue for node {node_id}: {prompt[:100]}...")
                        continue
                    
                    # Apply additional truncation if the individual prompt is still too large
                    if len(prompt) > 30000:  # Reduced from 50000 to ~7,500 tokens per node prompt
                        prompt = self._truncate_individual_node_prompt(prompt, node_id, max_chars=30000)
                        logger.info(f"Truncated individual prompt for {node_id}")
                    
                    node_prompts[node_id] = prompt
                    logger.info(f"Generated advanced prompt for {node_id} ({len(prompt)} characters)")
                    
                except Exception as prompt_error:
                    logger.error(f"Error generating prompt for node {node_id}: {str(prompt_error)}")
                    # Use fallback prompt
                    node_prompts[node_id] = self._generate_fallback_node_prompt(node_type, summarized_data, node_id)
                
            except Exception as e:
                logger.error(f"Error generating prompt for node {node_id}: {str(e)}")
                node_prompts[node_id] = f"Error generating prompt for {node_type} node: {str(e)}"
        
        return node_prompts
    
    def _create_comprehensive_workflow_prompt(self, nodes: Dict[str, Any], context: Dict[str, Any], 
                                            node_prompts: Dict[str, str], 
                                            workflow_analysis: Dict[str, Any]) -> str:
        """Create a comprehensive prompt that combines all node-specific prompts"""
        
        # Workflow overview
        total_nodes = workflow_analysis.get('total_analyzed_nodes', len(nodes))
        complexity = workflow_analysis.get('complexity_level', 'moderate')
        node_categories = workflow_analysis.get('analysis_categories', {})
        
        # Count nodes by category
        category_counts = {k: len(v) for k, v in node_categories.items() if v}
        
        # Create workflow header
        prompt = f"""
🚀 **COMPREHENSIVE WORKFLOW INTELLIGENCE ANALYSIS**
**Analyzing {total_nodes} Connected Nodes | Complexity: {complexity.upper()}**

📊 **WORKFLOW ARCHITECTURE OVERVIEW**:
• Data Sources: {len(node_categories.get('data_sources', []))} nodes
• Statistical Analysis: {len(node_categories.get('statistical_nodes', []))} nodes  
• Machine Learning: {len(node_categories.get('ml_nodes', []))} nodes
• Visualizations: {len(node_categories.get('visualization_nodes', []))} nodes
• Data Processing: {len(node_categories.get('processing_nodes', []))} nodes
• EDA Analysis: {len(node_categories.get('eda_nodes', []))} nodes
• Data Cleaning: {len(node_categories.get('cleaning_nodes', []))} nodes
• Feature Engineering: {len(node_categories.get('feature_engineering_nodes', []))} nodes
• Anomaly Detection: {len(node_categories.get('anomaly_detection_nodes', []))} nodes

🔗 **CONNECTED NODE ANALYSIS**:
The following connected nodes require comprehensive, integrated analysis:

"""
        
        # Add individual node prompts in logical order
        node_order = self._determine_node_analysis_order(nodes, workflow_analysis)
        
        # Filter to only include nodes that actually have prompts
        valid_nodes = [node_id for node_id in node_order if node_id in node_prompts]
        
        logger.info(f"Including {len(valid_nodes)} nodes with valid prompts out of {len(node_order)} total nodes")
        
        for i, node_id in enumerate(valid_nodes, 1):
            node_info = nodes.get(node_id, {})
            node_type = node_info.get('type', 'unknown')
            node_prompt = node_prompts[node_id]  # We know this exists since we filtered
            
            prompt += f"""
═══════════════════════════════════════════════════════════════════

**NODE {i}: {node_id.upper()} | TYPE: {node_type.upper()}**

{node_prompt}

═══════════════════════════════════════════════════════════════════
"""
        
        # Add comprehensive workflow synthesis requirements
        prompt += f"""

🎯 **DATA ANALYSIS SYNTHESIS REQUIREMENTS**:

**REQUIREMENT**: This analysis must synthesize data from ALL {len(valid_nodes)} connected nodes to provide:

1. **DATA INFORMATION SYNTHESIS**: 
   - Combine data information from all nodes
   - Identify cross-node patterns and relationships
   - Summarize key data characteristics

2. **DATA QUALITY ASSESSMENT**:
   - Evaluate data quality progression through the workflow
   - Assess transformation effectiveness across connected nodes
   - Identify data issues and inconsistencies

3. **RESULTS REPORTING**:
   - Synthesize statistical findings with analysis results
   - Combine visualization insights with analytical outcomes
   - Summarize key findings from the data

**RESPONSE STRUCTURE REQUIREMENTS**:

Provide a comprehensive analysis structured as follows:

## DATA SUMMARY
- Key characteristics of the data processed in this workflow
- Data quality assessment
- Main statistical properties

## INTEGRATED DATA ANALYSIS
- Cross-node pattern analysis
- Data transformation results
- Quality progression assessment

## ANALYSIS RESULTS
- Key findings from the data
- Observed patterns and trends
- Notable statistics and metrics

⚡ **CRITICAL REQUIREMENT**: All insights must be based on the ACTUAL data, patterns, and results from the connected nodes. Focus only on data information and results reporting.

**RESPONSE COMPLETENESS REQUIREMENTS**:
- Each section MUST contain at least 3-4 meaningful sentences
- Provide specific data insights, not generic statements
- Include quantitative findings where available
- Ensure comprehensive coverage of all workflow aspects
- Use bullet points and clear formatting for readability

**WORKFLOW CONTEXT**: {json.dumps(context, indent=2) if context else 'Standard analytical workflow'}
"""
        
        return prompt.strip()
    
    def _determine_node_analysis_order(self, nodes: Dict[str, Any], workflow_analysis: Dict[str, Any]) -> List[str]:
        """Determine the logical order for analyzing nodes"""
        categories = workflow_analysis.get('analysis_categories', {})
        
        # Define logical analysis order
        order_priority = [
            'data_sources',
            'cleaning_nodes', 
            'eda_nodes',
            'statistical_nodes',
            'feature_engineering_nodes',
            'processing_nodes',
            'ml_nodes',
            'anomaly_detection_nodes',
            'visualization_nodes'
        ]
        
        ordered_nodes = []
        
        # Add nodes in priority order
        for category in order_priority:
            category_nodes = categories.get(category, [])
            ordered_nodes.extend(category_nodes)
        
        # Add any remaining nodes
        all_node_ids = set(nodes.keys())
        ordered_node_ids = set(ordered_nodes)
        remaining_nodes = all_node_ids - ordered_node_ids
        ordered_nodes.extend(list(remaining_nodes))
        
        return ordered_nodes
    
    def _create_focused_node_prompt(self, node_prompt: str, node_type: str, context: Dict[str, Any]) -> str:
        """Create a focused prompt for single node analysis"""
        
        context_str = json.dumps(context, indent=2) if context else "Single node analysis"
        
        focused_prompt = f"""
🎯 **FOCUSED NODE ANALYSIS REQUEST**

**ANALYSIS CONTEXT**: {context_str}

{node_prompt}

**FOCUSED ANALYSIS REQUIREMENTS**:

Please provide a focused analysis of this specific node that includes:

1. **DATA INFORMATION**: Key characteristics of the data in this node
2. **DATA QUALITY**: Data quality assessment and reliability
3. **RESULTS SUMMARY**: Key findings and patterns from this node's data/results
4. **STATISTICAL INSIGHTS**: Important statistical properties observed

**OUTPUT FORMAT**: Provide a well-structured analysis with clear sections and bullet points based on the actual data and results from this node. Focus only on data information and results reporting.
"""
        
        return focused_prompt
    
    def _call_nvidia_api_streaming(self, prompt: str, max_retries: int = 2) -> str:
        """Call NVIDIA API with streaming for comprehensive responses and retry logic"""
        if not self.client:
            logger.warning("NVIDIA API not available, generating fallback response")
            return self._generate_fallback_ai_response(prompt)

        # Check and truncate prompt if it's too long for the model
        # Use more aggressive truncation: 115,000 tokens max (leave 16k for completion + safety margin)
        truncated_prompt = self._truncate_prompt_if_needed(prompt, max_tokens=115000)
        if len(truncated_prompt) != len(prompt):
            logger.warning(f"Prompt truncated from {len(prompt)} to {len(truncated_prompt)} characters due to token limits")

        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # System prompt for data scientist expertise
                system_prompt = """You are a comprehensive data scientist with expertise in:
- Advanced data analysis and statistical modeling
- Data quality assessment and validation
- Pattern recognition and data insights discovery
- Technical and statistical reporting
- Multi-node workflow analysis

CRITICAL INSTRUCTIONS:
1. Provide COMPLETE and DETAILED analysis for ALL sections requested
2. Generate comprehensive content for each section (minimum 3-4 sentences per section)
3. Include specific findings, patterns, and insights from the actual data
4. Use clear section headers and bullet points for readability
5. Ensure each section provides meaningful analytical value

Focus on delivering thorough data information and results reporting with actionable insights."""

                # Try different models until one works
                for model_attempt, model_name in enumerate(self.models):
                    try:
                        logger.info(f"Attempting API call with model: {model_name} (retry {retry_count}/{max_retries})")
                        
                        # Additional validation for API key before making request
                        if not self.api_key or len(self.api_key) < 10:
                            raise Exception(f"Invalid or missing NVIDIA API key (length: {len(self.api_key) if self.api_key else 0})")
                        
                        completion = self.client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": truncated_prompt}
                            ],
                            temperature=0.7,
                            top_p=0.95,
                            max_tokens=12288,  # Increased to allow more comprehensive responses
                            frequency_penalty=0.1,
                            presence_penalty=0.1,
                            stream=True,
                            timeout=150  # Increased timeout for more comprehensive analysis
                        )
                        
                        # Update working model
                        if model_name != self.model:
                            self.model = model_name
                            logger.info(f"Updated working model to: {model_name}")
                        
                        # Collect streaming response with timeout handling
                        full_response = ""
                        chunk_count = 0
                        start_time = time.time()
                        
                        try:
                            for chunk in completion:
                                if chunk.choices[0].delta.content is not None:
                                    content = chunk.choices[0].delta.content
                                    full_response += content
                                    chunk_count += 1
                                    
                                    # Log progress every 100 chunks
                                    if chunk_count % 100 == 0:
                                        elapsed = time.time() - start_time
                                        logger.info(f"Received {chunk_count} chunks in {elapsed:.1f}s, response length: {len(full_response)}")
                                    
                                    # Safety timeout for extremely long responses
                                    if time.time() - start_time > 180:  # 3 minute absolute limit
                                        logger.warning("AI response streaming timeout reached, returning partial response")
                                        break
                        except Exception as streaming_error:
                            logger.warning(f"Streaming error: {str(streaming_error)}, returning partial response")
                            if not full_response:  # If no response collected, re-raise
                                raise streaming_error
                        
                        logger.info(f"Successfully generated AI response ({len(full_response)} characters)")
                        return full_response
                        
                    except Exception as model_error:
                        error_message = str(model_error)
                        logger.warning(f"Model {model_name} failed (retry {retry_count}): {error_message}")
                        
                        # Check if it's a token limit error
                        if any(token_error in error_message.lower() for token_error in 
                               ['maximum context length', 'token limit', 'too many tokens', 'context length exceeded']):
                            logger.warning("Token limit exceeded, returning fallback response")
                            return self._generate_fallback_ai_response(prompt)
                        
                        if model_attempt == len(self.models) - 1:
                            # If this was the last model, re-raise to trigger retry
                            raise model_error
                        continue
                
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                logger.warning(f"NVIDIA API call failed (attempt {retry_count}/{max_retries + 1}): {error_message}")
                
                # Check if it's a token limit error
                if any(token_error in error_message.lower() for token_error in 
                       ['maximum context length', 'token limit', 'too many tokens', 'context length exceeded']):
                    logger.warning("Token limit exceeded, returning fallback response")
                    return self._generate_fallback_ai_response(prompt)
                
                if retry_count <= max_retries:
                    # Wait before retrying with exponential backoff
                    wait_time = min(2 ** retry_count, 5)  # Cap at 5 seconds
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"NVIDIA API call failed after {max_retries + 1} attempts: {error_message}")
                    return self._generate_fallback_ai_response(prompt)
        
        # This should not be reached, but just in case
        return self._generate_fallback_ai_response(prompt)
    
    def _structure_ai_response(self, ai_response: str, nodes: Dict[str, Any], 
                             workflow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the AI response into organized sections"""
        
        # Try to parse structured sections from the response
        sections = self._parse_response_sections(ai_response)
        
        structured_response = {
            'insights': {
                'data_summary': sections.get('data_summary', ''),
                'integrated_analysis': sections.get('integrated_analysis', ''),
                'analysis_results': sections.get('analysis_results', ''),
                'key_insights': self._extract_key_insights(ai_response),
                'data_quality': self._extract_data_quality(ai_response),
                'statistical_properties': self._extract_statistical_properties(ai_response)
            },
            'full_response': ai_response,
            'workflow_analysis': workflow_analysis,
            'node_summary': {
                'total_nodes': len(nodes),
                'node_types': list(set(node.get('type', 'unknown') for node in nodes.values())),
                'complexity_level': workflow_analysis.get('complexity_level', 'moderate')
            }
        }
        
        return structured_response
    
    def _parse_response_sections(self, response: str) -> Dict[str, str]:
        """Parse AI response into structured sections"""
        sections = {}
        
        # Define section markers
        section_markers = {
            'data_summary': ['## DATA SUMMARY', '# DATA SUMMARY', 'DATA SUMMARY'],
            'integrated_analysis': ['## INTEGRATED DATA ANALYSIS', '# INTEGRATED DATA ANALYSIS', 'INTEGRATED DATA ANALYSIS'],
            'analysis_results': ['## ANALYSIS RESULTS', '# ANALYSIS RESULTS', 'ANALYSIS RESULTS']
        }
        
        # Split response into lines
        lines = response.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Check if this line starts a new section
            section_found = False
            for section_key, markers in section_markers.items():
                if any(marker.upper() in line_upper for marker in markers):
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = section_key
                    current_content = []
                    section_found = True
                    break
            
            # Add line to current section
            if not section_found and current_section:
                current_content.append(line)
            elif not section_found and not current_section:
                # Before any section markers, treat as data summary
                if 'data_summary' not in sections:
                    sections['data_summary'] = sections.get('data_summary', '') + line + '\n'
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response for single node analysis"""
        return {
            'analysis': ai_response,
            'key_insights': self._extract_key_insights(ai_response),
            'data_quality': self._extract_data_quality(ai_response),
            'statistical_properties': self._extract_statistical_properties(ai_response)
        }
    
    def _extract_key_insights(self, response: str) -> List[str]:
        """Extract key insights from AI response"""
        insights = []
        lines = response.split('\n')
        
        for line in lines:
            # Look for bullet points or numbered insights
            if any(marker in line for marker in ['•', '◦', '▸', '1.', '2.', '3.', '-']):
                if any(keyword in line.lower() for keyword in ['insight', 'finding', 'pattern', 'trend', 'result']):
                    insights.append(line.strip())
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_data_quality(self, response: str) -> List[str]:
        """Extract data quality information from AI response"""
        quality_info = []
        lines = response.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['quality', 'missing', 'valid', 'complete', 'reliable']):
                if any(marker in line for marker in ['•', '◦', '▸', '1.', '2.', '3.', '-']):
                    quality_info.append(line.strip())
        
        return quality_info[:5]  # Return top 5 quality points
    
    def _extract_statistical_properties(self, response: str) -> str:
        """Extract statistical properties from AI response"""
        lines = response.split('\n')
        stat_content = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['statistic', 'mean', 'median', 'mode', 'std', 'variance', 'distribution']):
                stat_content.append(line.strip())
        
        return ' '.join(stat_content[:5])  # Return first 5 statistics-related lines
    
    def _generate_fallback_ai_response(self, prompt: str) -> str:
        """Generate fallback response when AI is not available"""
        # Check if this is an authentication issue
        api_status = "API not configured"
        if self.api_key:
            if len(self.api_key) < 10:
                api_status = "Invalid API key format"
            else:
                api_status = "Authentication failed - check API key validity"
        else:
            api_status = "No API key provided"
            
        return f"""
## DATA SUMMARY
NVIDIA AI API is currently unavailable ({api_status}). Based on the workflow structure and data patterns, this analysis indicates a data processing pipeline with multiple connected nodes requiring integrated assessment.

## INTEGRATED DATA ANALYSIS  
The connected nodes represent an analytical workflow combining data processing, analysis, and visualization. Each component contributes to the overall data processing and analysis pipeline.

## ANALYSIS RESULTS
• Data processing pipeline established with proper node connections
• Statistical analysis capabilities available for data insights
• Data transformation and processing steps detected in the workflow
• Data quality assessment needed for reliable results
• Pattern detection and trend analysis possible with available data

**Note**: This is a fallback analysis due to AI API authentication issues. To enable comprehensive AI-powered insights:
1. Verify your NVIDIA API key is valid and active
2. Check that the API key has access to the required models
3. Ensure the API key is properly set in environment variables as 'NVIDIA_API_KEY'

For comprehensive data analysis, please configure NVIDIA API access.
"""
    
    def _generate_error_response(self, error_message: str, node_id: str = None) -> Dict[str, Any]:
        """Generate structured error response"""
        return {
            'success': False,
            'error': error_message,
            'node_id': node_id,
            'timestamp': datetime.now().isoformat(),
            'insights': {
                'key_findings': [f'Error occurred during analysis: {error_message}'],
                'data_quality_assessment': {
                    'overall_score': 'Error',
                    'main_issues': [error_message],
                    'recommendations': [
                        "Review input data format and structure",
                        "Check API connectivity and authentication", 
                        "Validate node configuration and types"
                    ]
                },
                'data_analysis': [
                    "Analysis unavailable due to error"
                ],
                'statistical_properties': ['Analysis unavailable due to error'],
                'next_steps': ['Resolve the error and retry analysis']
            }
        }
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test API connection and available models"""
        results = {
            'api_accessible': False,
            'working_models': [],
            'failed_models': [],
            'recommended_model': None,
            'prompt_system_status': 'active'
        }
        
        if not self.client:
            results['message'] = "NVIDIA API client not initialized"
            return results
        
        test_prompt = "Respond with: AI connection test successful"
        
        for model_name in self.models:
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                response = completion.choices[0].message.content
                if "successful" in response.lower():
                    results['working_models'].append(model_name)
                    if not results['recommended_model']:
                        results['recommended_model'] = model_name
                        
            except Exception as e:
                logger.warning(f"Model {model_name} test failed: {str(e)}")
                results['failed_models'].append(model_name)
        
        results['api_accessible'] = len(results['working_models']) > 0
        
        # Update working model if we found a better one
        if results['recommended_model'] and results['recommended_model'] != self.model:
            self.model = results['recommended_model']
            logger.info(f"Updated working model to: {self.model}")
        
        return results

    def create_background_task(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a background task for streaming analysis"""
        try:
            task_id = f"ai_task_{int(datetime.now().timestamp() * 1000)}"
            
            # Estimate duration based on data complexity
            total_nodes = comprehensive_data.get('workflow_summary', {}).get('total_nodes', 0)
            estimated_duration = min(30 + (total_nodes * 5), 120)  # 30-120 seconds
            
            return {
                'success': True,
                'task_id': task_id,
                'estimated_duration': f"{estimated_duration} seconds",
                'background_task': {
                    'id': task_id,
                    'status': 'created',
                    'created_at': datetime.now().isoformat(),
                    'estimated_completion': (datetime.now().timestamp() + estimated_duration)
                }
            }
        except Exception as e:
            logger.error(f"Error creating background task: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'background_task': None
            }
    
    def execute_streaming_analysis(self, comprehensive_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute streaming analysis in background with token management"""
        try:
            logger.info(f"Starting background streaming analysis for task: {task_id}")
            logger.info(f"Input data keys: {list(comprehensive_data.keys())}")
            
            # Use the same data processing pipeline as generate_comprehensive_workflow_insights
            result = self.generate_comprehensive_workflow_insights(comprehensive_data)
            
            # Add streaming-specific metadata
            result.update({
                'task_id': task_id,
                'completed_at': datetime.now().isoformat(),
                'processing_mode': 'background_streaming'
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Streaming analysis failed for task {task_id}: {str(e)}")
            return self._create_fallback_response(comprehensive_data, task_id, error=str(e))

    def generate_advanced_workflow_analysis(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced workflow analysis (compatibility method)"""
        try:
            # This is essentially the same as comprehensive workflow insights
            # but with additional advanced analysis markers
            result = self.generate_comprehensive_workflow_insights(comprehensive_data)
            
            # Add advanced analysis indicators
            result.update({
                'analysis_type': 'advanced_workflow',
                'advanced_features': [
                    'Multi-node synthesis',
                    'Contextual prompting',
                    'Business intelligence focus',
                    'Node-specific insights'
                ]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced workflow analysis: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _convert_workflow_summary_to_nodes_format(self, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert workflow summary format to nodes format for comprehensive analysis"""
        try:
            # Check if nodes are already in the correct format
            if 'nodes' in comprehensive_data:
                logger.info("Found nodes in direct format, using as-is")
                return {
                    'nodes': comprehensive_data['nodes'],
                    'workflow_context': comprehensive_data.get('workflow_context', {})
                }
            
            # Check for workflow_summary format
            workflow_summary = comprehensive_data.get('workflow_summary', {})
            node_outputs = workflow_summary.get('node_outputs', {})
            
            # Convert node outputs to the expected nodes format
            nodes = {}
            for node_id, node_output in node_outputs.items():
                nodes[node_id] = {
                    'type': node_output.get('type', 'unknown'),
                    'data': node_output.get('data', {}),
                    'connections': []  # Connections aren't preserved in this format
                }
            
            # If no node_outputs but we have workflow_summary, create a synthetic node
            if not nodes and workflow_summary:
                logger.info("Creating synthetic node from workflow summary")
                nodes['workflow_summary_node'] = {
                    'type': 'workflow_summary',
                    'data': workflow_summary,
                    'connections': []
                }
            elif not nodes:
                # Handle completely empty data by creating a minimal analysis node
                logger.warning("No nodes found in workflow data, creating minimal analysis structure")
                nodes['empty_workflow'] = {
                    'type': 'unknown_workflow',
                    'data': {
                        'analysis_note': 'No specific workflow data available for analysis',
                        'data_keys': list(comprehensive_data.keys()),
                        'timestamp': datetime.now().isoformat()
                    },
                    'connections': []
                }
            
            # Create workflow context from summary
            workflow_context = {
                'total_nodes': workflow_summary.get('total_nodes', len(nodes)),
                'node_types': workflow_summary.get('node_types', []),
                'has_data': workflow_summary.get('has_data', False),
                'primary_data_shape': workflow_summary.get('primary_data_shape'),
                'analysis_type': 'comprehensive_workflow_summary'
            }
            
            logger.info(f"Converted workflow summary with {len(nodes)} nodes to comprehensive format")
            
            return {
                'nodes': nodes,
                'workflow_context': workflow_context
            }
            
        except Exception as e:
            logger.error(f"Error converting workflow summary format: {str(e)}")
            # Return minimal format to avoid complete failure
            return {
                'nodes': {},
                'workflow_context': {'error': str(e)}
            }
    
    def _normalize_workflow_data(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize different workflow data formats to a consistent structure
        """
        logger.info(f"Normalizing workflow data with keys: {list(workflow_data.keys())}")
        
        # Case 1: Already in expected format with 'nodes' and 'workflow_context'
        if 'nodes' in workflow_data:
            logger.info("Data already in expected nodes format")
            return workflow_data
        
        # Case 2: Workflow summary format (from execute_streaming_analysis)
        if 'workflow_summary' in workflow_data:
            logger.info("Converting from workflow_summary format")
            return self._convert_workflow_summary_to_nodes_format(workflow_data)
        
        # Case 3: Direct node results format
        if 'node_results' in workflow_data:
            logger.info("Converting from node_results format")
            node_results = workflow_data['node_results']
            nodes = {}
            
            for node_id, node_data in node_results.items():
                if isinstance(node_data, dict) and node_data:
                    nodes[node_id] = {
                        'type': node_data.get('type', 'unknown'),
                        'data': node_data.get('data', node_data),  # Use full data if no 'data' key
                        'connections': node_data.get('connections', [])
                    }
            
            workflow_context = {
                'total_nodes': len(nodes),
                'execution_summary': workflow_data.get('execution_summary', {}),
                'analysis_type': 'node_results_direct'
            }
            
            return {
                'nodes': nodes,
                'workflow_context': workflow_context
            }
        
        # Case 4: Raw nodes data (direct from frontend)
        # Check if workflow_data itself contains node-like structures
        potential_nodes = {}
        for key, value in workflow_data.items():
            if isinstance(value, dict) and ('type' in value or 'data' in value):
                potential_nodes[key] = {
                    'type': value.get('type', 'unknown'),
                    'data': value.get('data', value),
                    'connections': value.get('connections', [])
                }
        
        if potential_nodes:
            logger.info(f"Found {len(potential_nodes)} potential nodes in raw data")
            return {
                'nodes': potential_nodes,
                'workflow_context': {
                    'total_nodes': len(potential_nodes),
                    'analysis_type': 'raw_data_extraction'
                }
            }
        
        # Case 5: Fallback - create synthetic structure
        logger.warning("No recognizable workflow structure found, creating synthetic analysis")
        synthetic_node = {
            'type': 'unknown_workflow',
            'data': workflow_data,
            'connections': []
        }
        
        return {
            'nodes': {'workflow_data': synthetic_node},
            'workflow_context': {
                'total_nodes': 1,
                'analysis_type': 'synthetic_fallback',
                'original_keys': list(workflow_data.keys())
            }
        }
    
    def _is_ai_node(self, node_type: str, node_id: str) -> bool:
        """Check if a node is an AI node that should be excluded from analysis"""
        ai_node_types = [
            'ai_summary', 'ai_analysis', 'ai_insights', 'ai_data_summary',
            'ai_recommendation', 'ai_report', 'ai_workflow_summary'
        ]
        
        ai_node_patterns = [
            'ai_', 'artificial_intelligence_', 'llm_', 'gpt_', 'chatbot_',
            'assistant_', 'recommendation_engine_'
        ]
        
        # Check if node type is explicitly an AI node
        if node_type.lower() in ai_node_types:
            return True
        
        # Check if node ID contains AI patterns
        node_id_lower = node_id.lower()
        if any(pattern in node_id_lower for pattern in ai_node_patterns):
            return True
        
        # Check if node type contains AI patterns
        node_type_lower = node_type.lower()
        if any(pattern in node_type_lower for pattern in ai_node_patterns):
            return True
        
        return False
    
    def _has_valid_data(self, node_data: Dict[str, Any], node_type: str) -> bool:
        """Check if node has valid, meaningful data for analysis"""
        # Handle DataFrame case first
        if isinstance(node_data, pd.DataFrame):
            return not node_data.empty
        
        # Handle None or empty cases
        if node_data is None:
            return False
        
        # Special case: Allow synthetic/fallback nodes to be processed
        if node_type in ['unknown_workflow', 'workflow_summary']:
            return True  # These are synthetic nodes we create for minimal data
        
        # For non-DataFrame data, check if it's truthy (but avoid ambiguous DataFrame evaluation)
        if not isinstance(node_data, (pd.DataFrame, pd.Series)) and not node_data:
            return False
        
        # Check for empty dictionaries or None values
        if isinstance(node_data, dict):
            # Filter out None, empty strings, and empty collections (handle DataFrames separately)
            meaningful_data = {}
            for k, v in node_data.items():
                if v is None:
                    continue
                elif isinstance(v, str) and v == "":
                    continue
                elif isinstance(v, list) and len(v) == 0:
                    continue
                elif isinstance(v, dict) and len(v) == 0:
                    continue
                elif isinstance(v, pd.DataFrame):
                    # Handle DataFrame separately - check if empty
                    if not v.empty:
                        meaningful_data[k] = v
                else:
                    meaningful_data[k] = v
            
            if not meaningful_data:
                return False
            
            # For statistical nodes, ensure we have actual statistical results
            if node_type in ['descriptive_stats', 'statistical_analysis', 'processing']:
                # Look for statistical indicators
                expected_keys = ['statistics', 'dataframe', 'correlations', 'results', 'summary', 'result_summary']
                
                # Check if at least one expected key exists with meaningful data
                has_expected_data = False
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            has_expected_data = True
                            break
                        elif isinstance(value, dict) and len(value) > 0:
                            # Check if dict contains statistical data
                            stat_indicators = ['mean', 'median', 'std', 'count', 'min', 'max', 'describe', 'columns', 'statistics']
                            if any(indicator in str(value).lower() for indicator in stat_indicators):
                                has_expected_data = True
                                break
                        elif isinstance(value, list) and len(value) > 0:
                            has_expected_data = True
                            break
                
                return has_expected_data
            
            # For data source nodes, ensure we have actual dataset
            elif node_type in ['data_source', 'csv_upload', 'file_upload']:
                expected_keys = ['dataframe', 'data', 'source_info', 'file_info', 'shape', 'columns', 'result_summary']
                
                has_data_source = False
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            has_data_source = True
                            break
                        elif key in ['shape', 'columns'] and value:
                            has_data_source = True
                            break
                        elif isinstance(value, dict) and len(value) > 0:
                            # Check if result_summary contains meaningful data like columns, shape, etc.
                            if key == 'result_summary':
                                if any(result_key in value for result_key in ['columns', 'shape', 'rows', 'dtypes', 'memory_usage']):
                                    has_data_source = True
                                    break
                            else:
                                has_data_source = True
                                break
                
                return has_data_source
            
            # For EDA nodes
            elif node_type in ['eda', 'exploratory_data_analysis', 'eda_analysis']:
                expected_keys = ['eda_results', 'statistics', 'charts', 'dataframe', 'insights']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For machine learning nodes
            elif node_type in ['classification', 'regression', 'clustering', 'automl', 'machine_learning']:
                expected_keys = ['model', 'predictions', 'performance', 'training_info', 'results']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For visualization nodes
            elif node_type in ['visualization', 'chart', 'plotting', 'basic_plots', 'advanced_plots']:
                expected_keys = ['charts', 'visualizations', 'chart_metadata', 'insights', 'chart_summary', 'plot_type', 'features_used']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif isinstance(value, dict) and value:  # Check for chart dictionaries
                            return True
                        elif isinstance(value, list) and value:  # Check for chart lists
                            return True
                        elif isinstance(value, str) and value:  # Check for plot type or other string values
                            return True
                        elif isinstance(value, int) and value > 0:  # Chart counts
                            return True
                return False
            
            # For data cleaning nodes
            elif node_type in ['data_cleaning', 'preprocessing']:
                expected_keys = ['data', 'dataframe', 'cleaning_summary', 'cleaning_stats', 'before_cleaning', 'after_cleaning', 'result_summary']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif key == 'result_summary' and isinstance(value, dict):
                            # Check if result_summary contains cleaning-related data
                            if any(cleaning_key in value for cleaning_key in ['columns', 'rows_removed', 'data_quality_score', 'operations_performed', 'cleaning_summary']):
                                return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For feature engineering nodes
            elif node_type in ['feature_engineering', 'feature_creation']:
                expected_keys = ['data', 'dataframe', 'engineering_summary', 'feature_info', 'transformations', 'result_summary']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif key == 'result_summary' and isinstance(value, dict):
                            # Check if result_summary contains feature engineering data
                            if any(fe_key in value for fe_key in ['features_added', 'operations_performed', 'engineering_summary', 'original_shape', 'final_shape']):
                                return True
                        elif key == 'engineering_summary' and isinstance(value, dict):
                            # Check if engineering_summary contains meaningful data
                            if any(eng_key in value for eng_key in ['features_added', 'operations_performed', 'feature_types', 'original_shape']):
                                return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For anomaly detection nodes
            elif node_type in ['anomaly_detection', 'outlier_detection']:
                expected_keys = ['anomalies', 'outliers', 'results', 'scores']
                for key in expected_keys:
                    if key in meaningful_data:
                        value = meaningful_data[key]
                        if isinstance(value, pd.DataFrame) and not value.empty:
                            return True
                        elif value:  # For non-DataFrame values
                            return True
                return False
            
            # For other nodes, check for result_summary or just meaningful content
            else:
                # Check if we have result_summary with meaningful data
                if 'result_summary' in meaningful_data:
                    result_summary = meaningful_data['result_summary']
                    if isinstance(result_summary, dict) and len(result_summary) > 0:
                        # Check for common result summary indicators
                        if any(key in result_summary for key in ['columns', 'shape', 'rows', 'data_quality_score', 'statistics', 'results', 'type']):
                            return True
                
                # Otherwise, just check if we have meaningful content
                return True
        
        # Non-dict data
        elif isinstance(node_data, (list, pd.DataFrame)):
            return len(node_data) > 0
        elif isinstance(node_data, (str, int, float)):
            return node_data != "" and node_data != 0
        
        return False

    def _generate_fallback_node_prompt(self, node_type: str, node_data: Dict[str, Any], node_id: str) -> str:
        """Generate a fallback prompt for unknown node types"""
        data_summary = self._summarize_node_data(node_data)
        
        return f"""
🔍 **NODE ANALYSIS: {node_id.upper()} | TYPE: {node_type.upper()}**

**UNKNOWN NODE TYPE DETECTED**
This node type ({node_type}) doesn't have a specialized prompt generator. 
Analyzing based on available data structure and patterns.

**NODE DATA SUMMARY**:
{data_summary}

**ANALYSIS REQUEST**:
Please analyze this node's data and provide insights based on:
1. **Data Structure**: What type of data processing or analysis appears to have occurred?
2. **Key Patterns**: What patterns or trends are evident in the data?
3. **Data Quality Assessment**: How reliable and complete is this data?
4. **Results Summary**: What are the main findings from this data?

**CONTEXT**: This node is part of a larger analytical workflow. Focus only on data information and results reporting.
"""
    
    def _summarize_node_data(self, node_data: Dict[str, Any]) -> str:
        """Create a summary of node data for prompt generation"""
        if not node_data:
            return "• No data available"
        
        summary_parts = []
        
        # Check for common data structures
        for key, value in node_data.items():
            if isinstance(value, pd.DataFrame):
                summary_parts.append(f"• {key}: DataFrame with {value.shape[0]} rows and {value.shape[1]} columns")
            elif isinstance(value, dict):
                if 'accuracy' in value or 'precision' in value or 'recall' in value:
                    summary_parts.append(f"• {key}: ML model metrics")
                elif 'mean' in value or 'std' in value or 'count' in value:
                    summary_parts.append(f"• {key}: Statistical measurements")
                else:
                    summary_parts.append(f"• {key}: Dictionary with {len(value)} keys")
            elif isinstance(value, list):
                summary_parts.append(f"• {key}: List with {len(value)} items")
            elif isinstance(value, (int, float)):
                summary_parts.append(f"• {key}: Numeric value ({value})")
            elif isinstance(value, str):
                summary_parts.append(f"• {key}: Text data")
            else:
                summary_parts.append(f"• {key}: {type(value).__name__}")
        
        return "\n".join(summary_parts) if summary_parts else "• Data structure not recognized"
    
    def _truncate_prompt_if_needed(self, prompt: str, max_tokens: int = 100000) -> str:
        """
        Truncate prompt if it exceeds token limits to prevent 400 Bad Request errors.
        Rough estimate: 1 token ≈ 4 characters for English text.
        Model limit is 131,072 tokens, we reserve some for system prompt and completion.
        Being more conservative with truncation - 100K tokens = ~400K chars.
        """
        # Rough token estimate (4 chars per token)
        estimated_tokens = len(prompt) / 4
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        logger.warning(f"Prompt too long: ~{estimated_tokens:.0f} tokens (limit: {max_tokens}). Smart truncation applied...")
        
        # Calculate how many characters we can keep
        max_chars = int(max_tokens * 4)
        
        # More intelligent truncation strategy
        lines = prompt.split('\n')
        
        # Keep essential parts with better balance
        header_lines = lines[:25]  # Increased header
        
        # Find analysis requirements section
        analysis_start = None
        for i, line in enumerate(lines):
            if "ANALYSIS REQUIREMENTS" in line.upper() or "REQUIREMENTS" in line.upper():
                analysis_start = i
                break
        
        if analysis_start:
            # Keep requirements and some data context
            footer_lines = lines[analysis_start:]
            
            # Try to include some middle content if space allows
            remaining_space = max_chars - len('\n'.join(header_lines + footer_lines))
            middle_content = []
            
            if remaining_space > 10000:  # If we have significant space left
                # Include some node data summaries
                middle_start = min(len(header_lines) + 10, len(lines) - len(footer_lines) - 10)
                middle_end = max(middle_start, analysis_start - 10)
                
                for i in range(middle_start, middle_end):
                    if i < len(lines):
                        line = lines[i]
                        # Keep lines that look like data summaries
                        if any(keyword in line.lower() for keyword in ['shape:', 'columns:', 'summary:', 'statistics:', 'results:']):
                            middle_content.append(line)
                            if len('\n'.join(middle_content)) > 5000:  # Limit middle content
                                break
            
            truncated_content = '\n'.join(
                header_lines + 
                (["\n...[DATA ANALYSIS CONTENT PRESERVED]...\n"] + middle_content if middle_content else ["\n...[NODE DATA TRUNCATED FOR EFFICIENCY]...\n"]) +
                ["\n...[CONTINUING WITH ANALYSIS REQUIREMENTS]...\n"] + 
                footer_lines
            )
        else:
            # If no requirements section, keep more of the beginning
            truncated_content = prompt[:max_chars] + '\n\n...[CONTENT TRUNCATED - PLEASE ANALYZE BASED ON AVAILABLE DATA]...'
        
        # Final size check
        if len(truncated_content) > max_chars:
            truncated_content = truncated_content[:max_chars] + "\n\n[FINAL TRUNCATION APPLIED - ANALYZE AVAILABLE CONTENT]"
        
        logger.info(f"Smart truncation applied: {len(prompt)} → {len(truncated_content)} characters")
        return truncated_content
    
    def _summarize_node_data_for_analysis(self, node_data: Dict[str, Any], node_type: str) -> Dict[str, Any]:
        """
        Summarize node data to extract meaningful insights for AI analysis.
        This preserves important analytical results while preventing token overflow.
        """
        if not node_data:
            return {}
        
        summarized = {}
        
        # Special handling for data cleaning nodes
        if node_type in ['data_cleaning', 'preprocessing']:
            return self._summarize_data_cleaning_output(node_data)
        # Special handling for feature engineering nodes
        elif node_type in ['feature_engineering', 'feature_creation']:
            return self._summarize_feature_engineering_output(node_data)
        # Special handling for advanced plots nodes
        elif node_type in ['advanced_plots', 'visualization', 'chart', 'plotting']:
            return self._summarize_advanced_plots_output(node_data)
        
        for key, value in node_data.items():
            if isinstance(value, pd.DataFrame):
                # For DataFrames, extract comprehensive information
                df_summary = {
                    'shape': value.shape,
                    'columns': list(value.columns),  # Keep ALL columns for analysis
                    'dtypes': dict(value.dtypes),  # Keep ALL dtypes
                    'memory_usage_mb': round(value.memory_usage(deep=True).sum() / 1024**2, 2),
                    'null_counts': dict(value.isnull().sum()),
                    'sample_data': value.head(5).to_dict('records') if len(value) > 0 else []  # More sample data
                }
                
                # Add comprehensive statistics for ALL numeric columns
                numeric_cols = value.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = value[numeric_cols].describe()
                    df_summary['detailed_statistics'] = stats_df.to_dict()
                    
                    # Add additional statistical measures
                    df_summary['additional_stats'] = {}
                    for col in numeric_cols:
                        col_data = value[col].dropna()
                        if len(col_data) > 0:
                            df_summary['additional_stats'][col] = {
                                'skewness': float(col_data.skew()) if hasattr(col_data, 'skew') else 'N/A',
                                'kurtosis': float(col_data.kurtosis()) if hasattr(col_data, 'kurtosis') else 'N/A',
                                'unique_values': int(col_data.nunique()),
                                'zero_count': int((col_data == 0).sum()),
                                'negative_count': int((col_data < 0).sum())
                            }
                
                # Add categorical column analysis
                categorical_cols = value.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    df_summary['categorical_analysis'] = {}
                    for col in categorical_cols:
                        value_counts = value[col].value_counts().head(10)  # Top 10 categories
                        df_summary['categorical_analysis'][col] = {
                            'unique_count': int(value[col].nunique()),
                            'top_categories': dict(value_counts),
                            'null_percentage': float(value[col].isnull().mean() * 100)
                        }
                
                summarized[key] = df_summary
                
            elif isinstance(value, dict):
                # For dictionaries, preserve ALL important analytical content
                if len(value) > 50:  # Only summarize VERY large dicts
                    # Keep ALL analytical keys - don't filter aggressively
                    analytical_keys = [k for k in value.keys() if any(
                        important_word in str(k).lower() 
                        for important_word in [
                            'accuracy', 'precision', 'recall', 'score', 'result', 'results',
                            'mean', 'std', 'count', 'summary', 'performance', 'metric',
                            'statistics', 'stat', 'analysis', 'finding', 'insight',
                            'correlation', 'coefficient', 'p_value', 'significance',
                            'anomaly', 'outlier', 'detection', 'threshold', 'boundary',
                            'model', 'prediction', 'probability', 'confidence',
                            'feature', 'importance', 'weight', 'coefficient',
                            'error', 'loss', 'rmse', 'mae', 'mse', 'r2'
                        ]
                    )]
                    
                    # If we found analytical keys, keep them ALL
                    if analytical_keys:
                        limited_dict = {k: value[k] for k in analytical_keys}
                        # Also include first 20 regular keys if space allows
                        remaining_keys = [k for k in list(value.keys())[:20] if k not in analytical_keys]
                        for k in remaining_keys:
                            limited_dict[k] = value[k]
                        summarized[key] = limited_dict
                    else:
                        # Keep first 30 keys if no analytical keys found
                        summarized[key] = {k: value[k] for k in list(value.keys())[:30]}
                else:
                    # Keep smaller dicts completely
                    summarized[key] = value
                    
            elif isinstance(value, list):
                # For lists, be more generous with size limits
                if len(value) > 1000:  # Only truncate very large lists
                    summarized[key] = {
                        'type': 'large_list',
                        'length': len(value),
                        'sample_start': value[:10],  # More samples
                        'sample_end': value[-5:] if len(value) > 10 else [],
                        'data_types': list(set(type(item).__name__ for item in value[:50])),
                        'summary_stats': self._get_list_summary_stats(value)
                    }
                else:
                    # Keep smaller lists completely
                    summarized[key] = value
                    
            elif isinstance(value, str) and len(value) > 50000:  # Increased threshold
                # For very long strings, keep more content
                summarized[key] = value[:20000] + "...[CONTENT CONTINUES]..." + value[-5000:]
                
            else:
                # For other data types, include as-is
                summarized[key] = value
        
        return summarized
    
    def _get_list_summary_stats(self, data_list: list) -> Dict[str, Any]:
        """Get summary statistics for a list of data"""
        try:
            if not data_list:
                return {}
            
            # Try to convert to numeric if possible
            numeric_values = []
            for item in data_list[:1000]:  # Sample first 1000 items
                try:
                    if isinstance(item, (int, float)):
                        numeric_values.append(float(item))
                    elif isinstance(item, str) and item.replace('.', '').replace('-', '').isdigit():
                        numeric_values.append(float(item))
                except:
                    continue
            
            if len(numeric_values) > 0:
                import statistics
                return {
                    'numeric_count': len(numeric_values),
                    'mean': round(statistics.mean(numeric_values), 4),
                    'median': round(statistics.median(numeric_values), 4),
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'std': round(statistics.stdev(numeric_values), 4) if len(numeric_values) > 1 else 0
                }
            else:
                # Non-numeric list analysis
                unique_types = list(set(type(item).__name__ for item in data_list[:100]))
                return {
                    'data_types': unique_types,
                    'total_length': len(data_list)
                }
        except Exception as e:
            return {'error': str(e)}
    
    def _truncate_individual_node_prompt(self, prompt: str, node_id: str, max_chars: int = 30000) -> str:
        """
        Truncate an individual node prompt while preserving the most important parts.
        """
        if len(prompt) <= max_chars:
            return prompt
        
        lines = prompt.split('\n')
        
        # Find important sections
        header_end = 0
        analysis_start = len(lines)
        
        for i, line in enumerate(lines):
            if "**NODE ID:" in line.upper() or "NODE ANALYSIS:" in line.upper():
                header_end = min(i + 10, len(lines))  # Keep 10 lines after header
            if "ANALYSIS REQUIREMENTS" in line.upper() or "REQUIREMENTS" in line.upper():
                analysis_start = i
                break
        
        # Keep header and requirements, truncate middle data
        header_lines = lines[:header_end]
        footer_lines = lines[analysis_start:] if analysis_start < len(lines) else []
        
        truncated_lines = header_lines + [
            "",
            "...[DATA CONTENT TRUNCATED TO PREVENT TOKEN OVERFLOW]...",
            f"[Node {node_id} data summarized for analysis efficiency]",
            ""
        ] + footer_lines
        
        truncated_prompt = '\n'.join(truncated_lines)
        
        # If still too long, do simple truncation
        if len(truncated_prompt) > max_chars:
            truncated_prompt = truncated_prompt[:max_chars] + "\n\n[PROMPT TRUNCATED DUE TO LENGTH]"
        
        return truncated_prompt
    
    def _summarize_data_cleaning_output(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize data cleaning node output with specific focus on cleaning parameters
        """
        summarized = {}
        
        # Handle the main data structure - cleaned DataFrame
        # The prompt generator expects both 'data' and 'dataframe' keys
        if 'data' in node_data:
            data = node_data['data']
            if isinstance(data, pd.DataFrame):
                # Store in both keys for compatibility with prompt generator
                df_summary = {
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': dict(data.dtypes),
                    'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2),
                    'null_counts': dict(data.isnull().sum()),
                    'total_missing_values': int(data.isnull().sum().sum()),
                    'sample_data': data.head(3).to_dict('records') if len(data) > 0 else []
                }
                
                # Store DataFrame info in both expected keys
                summarized['data'] = data  # Keep original DataFrame for prompt generator
                summarized['dataframe'] = data  # Alternative key expected by prompt generator
                summarized['cleaned_data'] = df_summary
                
                # Add basic statistics for numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_summary = {}
                    for col in numeric_cols:
                        col_data = data[col].dropna()
                        if len(col_data) > 0:
                            stats_summary[col] = {
                                'count': int(len(col_data)),
                                'mean': round(float(col_data.mean()), 4),
                                'std': round(float(col_data.std()), 4),
                                'min': float(col_data.min()),
                                'max': float(col_data.max()),
                                'unique_values': int(col_data.nunique())
                            }
                    summarized['cleaned_data']['numeric_statistics'] = stats_summary
        
        # Handle cleaning summary with all the detailed cleaning operations
        if 'cleaning_summary' in node_data:
            cleaning_summary = node_data['cleaning_summary']
            
            # Store in multiple formats for compatibility with prompt generator
            summarized['cleaning_summary'] = cleaning_summary  # Original format
            summarized['cleaning_stats'] = cleaning_summary  # Alternative key expected by prompt
            
            # Enhanced cleaning operations summary
            cleaning_ops = {
                'original_shape': cleaning_summary.get('original_shape'),
                'final_shape': cleaning_summary.get('final_shape'),
                'rows_removed': cleaning_summary.get('rows_removed', 0),
                'columns_removed': cleaning_summary.get('columns_removed', 0),
                'operations_performed': cleaning_summary.get('operations_performed', []),
                'data_quality_score': cleaning_summary.get('data_quality_score', 0),
                # Add additional stats that prompt generator might expect
                'records_removed': cleaning_summary.get('rows_removed', 0),  # Alternative naming
                'duplicates_removed': 0,  # Default if not specified
                'missing_values_handled': 0,  # Default if not specified
                'data_types_corrected': 0  # Default if not specified
            }
            
            # Try to extract more detailed stats from operations_performed
            operations = cleaning_summary.get('operations_performed', [])
            for op in operations:
                op_lower = str(op).lower()
                if 'duplicate' in op_lower:
                    # Try to extract number from operation description
                    import re
                    numbers = re.findall(r'\d+', str(op))
                    if numbers:
                        cleaning_ops['duplicates_removed'] = int(numbers[0])
                elif 'missing' in op_lower or 'null' in op_lower:
                    numbers = re.findall(r'\d+', str(op))
                    if numbers:
                        cleaning_ops['missing_values_handled'] = int(numbers[0])
                elif 'type' in op_lower or 'convert' in op_lower:
                    numbers = re.findall(r'\d+', str(op))
                    if numbers:
                        cleaning_ops['data_types_corrected'] = int(numbers[0])
            
            summarized['cleaning_operations'] = cleaning_ops
            
            # Calculate cleaning impact metrics
            if cleaning_summary.get('original_shape') and cleaning_summary.get('final_shape'):
                orig_rows, orig_cols = cleaning_summary['original_shape']
                final_rows, final_cols = cleaning_summary['final_shape']
                
                # Create before/after stats for prompt generator compatibility
                summarized['before_cleaning'] = {
                    'shape': cleaning_summary['original_shape'],
                    'total_records': orig_rows,
                    'total_columns': orig_cols,
                    'data_points': orig_rows * orig_cols
                }
                
                summarized['after_cleaning'] = {
                    'shape': cleaning_summary['final_shape'],
                    'total_records': final_rows,
                    'total_columns': final_cols,
                    'data_points': final_rows * final_cols
                }
                
                summarized['cleaning_impact'] = {
                    'data_retention_rate': round((final_rows / orig_rows) * 100, 2) if orig_rows > 0 else 0,
                    'column_retention_rate': round((final_cols / orig_cols) * 100, 2) if orig_cols > 0 else 0,
                    'total_operations_count': len(cleaning_summary.get('operations_performed', [])),
                    'quality_improvement': 'Calculated' if cleaning_summary.get('data_quality_score') else 'Not available'
                }
        
        # Handle type identifier
        if 'type' in node_data:
            summarized['node_type_identifier'] = node_data['type']
        
        return summarized
    
    def _summarize_feature_engineering_output(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize feature engineering node output with specific focus on engineering parameters
        """
        summarized = {}
        
        # Check if data is in result_summary (common structure)
        actual_data = node_data
        if 'result_summary' in node_data and isinstance(node_data['result_summary'], dict):
            actual_data = node_data['result_summary']
        
        # Handle the main data structure - engineered DataFrame
        # The prompt generator expects both 'data' and 'dataframe' keys
        if 'data' in actual_data:
            data = actual_data['data']
            if isinstance(data, pd.DataFrame):
                # Store in both keys for compatibility with prompt generator
                df_summary = {
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': dict(data.dtypes),
                    'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2),
                    'null_counts': dict(data.isnull().sum()),
                    'total_missing_values': int(data.isnull().sum().sum()),
                    'sample_data': data.head(3).to_dict('records') if len(data) > 0 else []
                }
                
                # Store DataFrame info in both expected keys
                summarized['data'] = data  # Keep original DataFrame for prompt generator
                summarized['dataframe'] = data  # Alternative key expected by prompt generator
                summarized['engineered_data'] = df_summary
                
                # Add basic statistics for numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_summary = {}
                    for col in numeric_cols:
                        col_data = data[col].dropna()
                        if len(col_data) > 0:
                            stats_summary[col] = {
                                'count': int(len(col_data)),
                                'mean': round(float(col_data.mean()), 4),
                                'std': round(float(col_data.std()), 4),
                                'min': float(col_data.min()),
                                'max': float(col_data.max()),
                                'unique_values': int(col_data.nunique())
                            }
                    summarized['engineered_data']['numeric_statistics'] = stats_summary
        
        # Handle engineering summary with all the detailed feature engineering operations
        if 'engineering_summary' in actual_data:
            engineering_summary = actual_data['engineering_summary']
            
            # Store in multiple formats for compatibility with prompt generator
            summarized['engineering_summary'] = engineering_summary  # Original format
            summarized['feature_engineering_stats'] = engineering_summary  # Alternative key
            summarized['engineering_stats'] = engineering_summary  # Another alternative key
            
            # Enhanced feature engineering operations summary
            engineering_ops = {
                'original_shape': engineering_summary.get('original_shape'),
                'final_shape': engineering_summary.get('final_shape'),
                'features_added': engineering_summary.get('features_added', 0),
                'operations_performed': engineering_summary.get('operations_performed', []),
                'feature_types': engineering_summary.get('feature_types', {}),
                # Additional fields that prompt generator might expect
                'new_features': [],  # Will be populated below
                'transformations': {},  # Will be populated from operations
                'original_features': [],  # Will be calculated
            }
            
            # Extract feature names and transformations from operations
            operations = engineering_summary.get('operations_performed', [])
            transformations = {}
            new_features = []
            
            for op in operations:
                op_str = str(op).lower()
                if 'one-hot' in op_str:
                    transformations['categorical_encoding'] = 'onehot'
                elif 'label encoding' in op_str:
                    transformations['categorical_encoding'] = 'label'
                elif 'standardscaler' in op_str:
                    transformations['scaling'] = 'standard'
                elif 'minmaxscaler' in op_str:
                    transformations['scaling'] = 'minmax'
                elif 'robustscaler' in op_str:
                    transformations['scaling'] = 'robust'
                elif 'polynomial' in op_str:
                    transformations['polynomial_features'] = True
                    # Try to extract degree and count
                    import re
                    numbers = re.findall(r'\d+', str(op))
                    if len(numbers) >= 2:
                        transformations['polynomial_degree'] = int(numbers[-1])
                        transformations['polynomial_count'] = int(numbers[0])
                elif 'binned' in op_str:
                    transformations['binning'] = True
                    numbers = re.findall(r'\d+', str(op))
                    if numbers:
                        transformations['bins_count'] = int(numbers[0])
                elif 'statistical features' in op_str:
                    transformations['statistical_features'] = True
                    new_features.extend(['row_mean', 'row_std', 'row_max', 'row_min', 'row_range'])
                elif 'interaction' in op_str:
                    transformations['interaction_features'] = True
                elif 'datetime' in op_str:
                    transformations['datetime_features'] = True
                elif 'constant' in op_str:
                    transformations['remove_constant'] = True
                elif 'correlated' in op_str:
                    transformations['remove_correlated'] = True
            
            engineering_ops['transformations'] = transformations
            engineering_ops['new_features'] = new_features
            
            # Calculate original features (approximate)
            if engineering_summary.get('original_shape') and engineering_summary.get('final_shape'):
                orig_rows, orig_cols = engineering_summary['original_shape']
                final_rows, final_cols = engineering_summary['final_shape']
                
                # Store original feature count
                engineering_ops['original_features'] = list(range(orig_cols))  # Placeholder
                
                # Create before/after stats for prompt generator compatibility
                summarized['before_engineering'] = {
                    'shape': engineering_summary['original_shape'],
                    'total_features': orig_cols,
                    'total_records': orig_rows,
                    'feature_space_size': orig_rows * orig_cols
                }
                
                summarized['after_engineering'] = {
                    'shape': engineering_summary['final_shape'],
                    'total_features': final_cols,
                    'total_records': final_rows,
                    'feature_space_size': final_rows * final_cols
                }
                
                summarized['engineering_impact'] = {
                    'feature_expansion_rate': round((final_cols / orig_cols) * 100, 2) if orig_cols > 0 else 0,
                    'features_added_count': engineering_summary.get('features_added', 0),
                    'total_operations_count': len(engineering_summary.get('operations_performed', [])),
                    'feature_types_distribution': engineering_summary.get('feature_types', {}),
                    'complexity_increase': 'High' if final_cols > orig_cols * 2 else 'Moderate' if final_cols > orig_cols * 1.5 else 'Low'
                }
            
            summarized['feature_engineering_operations'] = engineering_ops
            
            # Add feature importance placeholder (will be filled if available)
            summarized['feature_importance'] = {}
        
        # Handle type identifier - check both actual_data and original node_data
        if 'type' in actual_data:
            summarized['node_type_identifier'] = actual_data['type']
        elif 'type' in node_data:
            summarized['node_type_identifier'] = node_data['type']
        
        return summarized

    def _summarize_advanced_plots_output(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize advanced plots node output with specific focus on chart metadata and visualization parameters
        """
        summarized = {}
        
        # Handle the main data structure - charts and metadata
        if 'charts' in node_data:
            charts = node_data['charts']
            
            # Basic chart information
            chart_summary = {
                'total_charts_generated': len(charts),
                'chart_types': list(charts.keys()),
                'charts_available': True,
                'chart_metadata': {}
            }
            
            # Analyze each chart type
            for chart_name, chart_data in charts.items():
                chart_info = {
                    'name': chart_name,
                    'has_data': chart_data is not None and len(str(chart_data)) > 0,
                    'data_size': len(str(chart_data)) if chart_data else 0,
                    'format': 'base64_image' if chart_data and isinstance(chart_data, str) else 'unknown'
                }
                chart_summary['chart_metadata'][chart_name] = chart_info
            
            # Don't include actual chart data (base64 images) in AI analysis to save tokens
            # summarized['charts'] = charts  # Keep original charts for downstream usage
            # summarized['visualizations'] = charts  # Alternative key for compatibility
            
            # Instead, just keep the chart metadata for AI analysis
            summarized['chart_summary'] = chart_summary
            summarized['chart_metadata'] = chart_summary  # Alternative key
        
        # Handle plot configuration and features used
        if 'plot_type' in node_data:
            plot_config = {
                'primary_plot_type': node_data['plot_type'],
                'interactive_mode': node_data.get('interactive', False),
                'color_palette': node_data.get('color_palette', 'viridis')
            }
            summarized['plot_configuration'] = plot_config
            summarized['visualization_config'] = plot_config  # Alternative key
        
        # Handle features used for plotting
        if 'features_used' in node_data:
            features_info = {
                'selected_features': node_data['features_used'],
                'feature_count': len(node_data['features_used']) if node_data['features_used'] else 0,
                'auto_detected': True  # Most advanced plots use auto-detection
            }
            summarized['features_analysis'] = features_info
            summarized['plot_features'] = features_info  # Alternative key
        
        # Handle available columns analysis
        if 'numeric_columns_available' in node_data and 'categorical_columns_available' in node_data:
            column_analysis = {
                'numeric_columns': node_data['numeric_columns_available'],
                'categorical_columns': node_data['categorical_columns_available'],
                'total_numeric': len(node_data['numeric_columns_available']),
                'total_categorical': len(node_data['categorical_columns_available']),
                'column_distribution': {
                    'numeric_percentage': round(
                        len(node_data['numeric_columns_available']) / 
                        (len(node_data['numeric_columns_available']) + len(node_data['categorical_columns_available'])) * 100, 2
                    ) if (len(node_data['numeric_columns_available']) + len(node_data['categorical_columns_available'])) > 0 else 0,
                    'categorical_percentage': round(
                        len(node_data['categorical_columns_available']) / 
                        (len(node_data['numeric_columns_available']) + len(node_data['categorical_columns_available'])) * 100, 2
                    ) if (len(node_data['numeric_columns_available']) + len(node_data['categorical_columns_available'])) > 0 else 0
                }
            }
            summarized['column_analysis'] = column_analysis
            summarized['data_structure'] = column_analysis  # Alternative key
        
        # Handle chart count for summary statistics
        if 'chart_count' in node_data:
            chart_stats = {
                'total_visualizations': node_data['chart_count'],
                'generation_success': node_data['chart_count'] > 0,
                'complexity_level': (
                    'High' if node_data['chart_count'] >= 4 else 
                    'Medium' if node_data['chart_count'] >= 2 else 
                    'Low'
                )
            }
            summarized['visualization_statistics'] = chart_stats
            summarized['chart_statistics'] = chart_stats  # Alternative key
        
        # Handle the underlying data (preserve DataFrame info but not full data for AI analysis)
        if 'data' in node_data:
            data = node_data['data']
            if isinstance(data, pd.DataFrame):
                # For AI analysis, only keep DataFrame summary (not the full data)
                # This prevents token overflow from large datasets
                
                df_summary = {
                    'shape': data.shape,
                    'columns': list(data.columns)[:20],  # Limit columns to prevent token overflow
                    'dtypes': dict(list(data.dtypes.to_dict().items())[:20]),  # Limit dtypes
                    'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2),
                    'null_counts': dict(list(data.isnull().sum().to_dict().items())[:20]),  # Limit null counts
                    'sample_data': data.head(2).to_dict('records') if len(data) > 0 else []  # Reduce sample size
                }
                
                # Add basic statistics for numeric columns (limited)
                numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10 numeric columns
                if len(numeric_cols) > 0:
                    stats_summary = {}
                    for col in numeric_cols:
                        col_data = data[col].dropna()
                        if len(col_data) > 0:
                            stats_summary[col] = {
                                'count': int(len(col_data)),
                                'mean': round(float(col_data.mean()), 4),
                                'std': round(float(col_data.std()), 4),
                                'min': float(col_data.min()),
                                'max': float(col_data.max()),
                                'unique_values': int(col_data.nunique())
                            }
                    df_summary['numeric_statistics'] = stats_summary
                
                # Only store the summary for AI analysis (not the full DataFrame)
                summarized['data_summary'] = df_summary
                
                # Note: We deliberately don't include the full DataFrame here to save tokens
                # The original DataFrame is preserved in the node output for downstream nodes
        
        # Generate visualization insights
        visualization_insights = {
            'plot_effectiveness': 'High' if summarized.get('chart_summary', {}).get('total_charts_generated', 0) > 2 else 'Medium',
            'data_coverage': 'Complete' if summarized.get('features_analysis', {}).get('feature_count', 0) > 0 else 'Partial',
            'visualization_types': summarized.get('chart_summary', {}).get('chart_types', []),
            'analysis_depth': (
                'Comprehensive' if summarized.get('chart_summary', {}).get('total_charts_generated', 0) >= 3 else
                'Standard' if summarized.get('chart_summary', {}).get('total_charts_generated', 0) >= 2 else
                'Basic'
            )
        }
        summarized['visualization_insights'] = visualization_insights
        summarized['insights'] = visualization_insights  # Alternative key
        
        # Add node type identifier
        summarized['node_type_identifier'] = 'advanced_plots'
        
        return summarized
