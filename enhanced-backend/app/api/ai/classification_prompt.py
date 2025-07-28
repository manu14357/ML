"""
Classification Prompt Generator
Generates data-focused prompts for classification node analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json

class ClassificationPrompt:
    """Prompt generator for classification analysis nodes"""
    
    @staticmethod
    def generate_advanced_prompt(node_data: Dict[str, Any], node_id: str, context: Dict[str, Any] = None) -> str:
        """
        Generate data-focused classification analysis prompt based on model results and data
        
        Args:
            node_data: Classification node output data including model, predictions, metrics
            node_id: Unique identifier for this classification node
            context: Additional workflow context
            
        Returns:
            Data-focused prompt string for AI analysis
        """
        
        # Extract classification components
        model_info = ClassificationPrompt._extract_model_information(node_data)
        performance_metrics = ClassificationPrompt._extract_performance_metrics(node_data)
        prediction_analysis = ClassificationPrompt._analyze_predictions(node_data)
        feature_importance = ClassificationPrompt._extract_feature_importance(node_data)
        data_characteristics = ClassificationPrompt._analyze_data_characteristics(node_data)
        
        prompt = f"""
📊 **CLASSIFICATION MODEL ANALYSIS: {node_id.upper()}**

{model_info}

{performance_metrics}

{prediction_analysis}

{feature_importance}

{data_characteristics}

🎯 **DATA ANALYSIS REQUIREMENTS**:

Please analyze this classification model focusing on data and results:

1. **MODEL PERFORMANCE ANALYSIS**:
   - Analyze classification accuracy, precision, recall, and F1-score metrics
   - Identify which classes are being predicted most/least accurately
   - Examine the model's statistical performance
   - Analyze the error patterns in the predictions

2. **PREDICTION PATTERN ANALYSIS**:
   - Examine class distribution and prediction patterns
   - Identify patterns in the classification results
   - Analyze misclassification patterns and common prediction errors
   - Evaluate model behavior on different data points

3. **FEATURE IMPORTANCE ANALYSIS**:
   - Identify the most influential features for the classification
   - Analyze feature importance distribution
   - Detect feature relationships and correlations
   - Summarize the key features driving the model

4. **DATA QUALITY ASSESSMENT**:
   - Evaluate data quality and its impact on results
   - Identify data characteristics affecting model performance
   - Assess the impact of data distributions on results
   - Examine statistical properties of the input data

**FOCUS**: Provide factual information about the model performance, data characteristics, and statistical properties without business implications or recommendations.

**NODE CONTEXT**: {json.dumps(context, indent=2) if context else 'Classification node in data workflow'}
"""
        
        return prompt.strip()
    
    @staticmethod
    def _extract_model_information(node_data: Dict[str, Any]) -> str:
        """Extract and format model information"""
        model_info = []
        
        # Check for model details
        if 'model' in node_data:
            model = node_data['model']
            if hasattr(model, '__class__'):
                model_type = model.__class__.__name__
                model_info.append(f"• Model Type: {model_type}")
                
                # Extract model parameters if available
                if hasattr(model, 'get_params'):
                    try:
                        params = model.get_params()
                        key_params = {k: v for k, v in params.items() if k in ['n_estimators', 'max_depth', 'C', 'gamma', 'alpha']}
                        if key_params:
                            model_info.append(f"• Key Parameters: {key_params}")
                    except:
                        pass
        
        # Check for algorithm information
        if 'algorithm' in node_data:
            model_info.append(f"• Algorithm: {node_data['algorithm']}")
        
        # Check for training information
        if 'config' in node_data:
            config = node_data['config']
            if 'target_column' in config:
                model_info.append(f"• Target Variable: {config['target_column']}")
            if 'feature_columns' in config:
                feature_count = len(config['feature_columns']) if config['feature_columns'] else 'Auto-selected'
                model_info.append(f"• Features Used: {feature_count} features")
        
        # Check for dataset split information
        if 'train_size' in node_data:
            model_info.append(f"• Training Data: {node_data['train_size']} samples")
        if 'test_size' in node_data:
            model_info.append(f"• Test Data: {node_data['test_size']} samples")
        
        return "📋 **MODEL INFORMATION**:\n" + "\n".join(model_info) if model_info else "📋 **MODEL INFORMATION**: Model information not available"
    
    @staticmethod
    def _extract_performance_metrics(node_data: Dict[str, Any]) -> str:
        """Extract and format performance metrics"""
        metrics_info = []
        
        # Check for metrics in node_data
        if 'metrics' in node_data:
            metrics = node_data['metrics']
            
            # Accuracy
            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                metrics_info.append(f"🎯 **Accuracy**: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Precision, Recall, F1
            if 'precision' in metrics:
                precision = metrics['precision']
                metrics_info.append(f"🔍 **Precision**: {precision:.3f}")
            
            if 'recall' in metrics:
                recall = metrics['recall']
                metrics_info.append(f"📊 **Recall**: {recall:.3f}")
            
            if 'f1' in metrics:
                f1 = metrics['f1']
                metrics_info.append(f"⚖️ **F1-Score**: {f1:.3f}")
            
            # Cross-validation scores
            if 'cv_scores' in metrics:
                cv_scores = metrics['cv_scores']
                if isinstance(cv_scores, (list, np.ndarray)):
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    metrics_info.append(f"🔄 **Cross-Validation**: {cv_mean:.3f} ± {cv_std:.3f}")
            
            # Classification report
            if 'classification_report' in metrics:
                metrics_info.append(f"📑 **Detailed Classification Report**: Available for per-class analysis")
            
            # Confusion matrix insights
            if 'confusion_matrix' in metrics:
                metrics_info.append(f"🔢 **Confusion Matrix**: Available for misclassification analysis")
        
        # Performance assessment
        if metrics_info:
            # Determine performance level
            accuracy = node_data.get('metrics', {}).get('accuracy', 0)
            if accuracy >= 0.9:
                performance_level = "🟢 **EXCELLENT** - Production-ready performance"
            elif accuracy >= 0.8:
                performance_level = "🟡 **GOOD** - Strong business value potential"
            elif accuracy >= 0.7:
                performance_level = "🟠 **MODERATE** - Requires optimization"
            else:
                performance_level = "🔴 **NEEDS IMPROVEMENT** - Significant enhancement required"
            
            metrics_info.insert(0, performance_level)
        
        return "📊 **PERFORMANCE METRICS**:\n" + "\n".join(metrics_info) if metrics_info else "📊 **PERFORMANCE METRICS**: Metrics not available"
    
    @staticmethod
    def _analyze_predictions(node_data: Dict[str, Any]) -> str:
        """Analyze prediction patterns and distributions"""
        prediction_analysis = []
        
        # Check for predictions
        if 'predictions' in node_data:
            predictions = node_data['predictions']
            
            if isinstance(predictions, (list, np.ndarray, pd.Series)):
                predictions = np.array(predictions)
                
                # Prediction distribution
                unique_values, counts = np.unique(predictions, return_counts=True)
                total_predictions = len(predictions)
                
                prediction_analysis.append(f"📈 **Total Predictions**: {total_predictions:,}")
                prediction_analysis.append(f"🏷️ **Unique Classes**: {len(unique_values)}")
                
                # Class distribution
                for value, count in zip(unique_values, counts):
                    percentage = (count / total_predictions) * 100
                    prediction_analysis.append(f"   • Class '{value}': {count:,} ({percentage:.1f}%)")
                
                # Class balance assessment
                max_percentage = max(counts) / total_predictions * 100
                min_percentage = min(counts) / total_predictions * 100
                balance_ratio = max_percentage / min_percentage
                
                if balance_ratio < 2:
                    balance_status = "⚖️ **Well-balanced** class distribution"
                elif balance_ratio < 5:
                    balance_status = "⚠️ **Moderately imbalanced** class distribution"
                else:
                    balance_status = "🚨 **Highly imbalanced** class distribution - consider rebalancing techniques"
                
                prediction_analysis.append(balance_status)
        
        # Check for prediction probabilities
        if 'prediction_probabilities' in node_data:
            prediction_analysis.append("🎲 **Prediction Probabilities**: Available for confidence analysis")
        
        return "🔮 **PREDICTION ANALYSIS**:\n" + "\n".join(prediction_analysis) if prediction_analysis else "🔮 **PREDICTION ANALYSIS**: Prediction data not available"
    
    @staticmethod
    def _extract_feature_importance(node_data: Dict[str, Any]) -> str:
        """Extract and analyze feature importance"""
        importance_info = []
        
        # Check for feature importance
        if 'feature_importance' in node_data:
            feature_importance = node_data['feature_importance']
            
            if isinstance(feature_importance, dict):
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                importance_info.append(f"🔍 **Top Features** (by importance):")
                
                for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                    importance_info.append(f"   {i}. **{feature}**: {importance:.4f}")
                
                # Feature importance insights
                top_feature_importance = sorted_features[0][1] if sorted_features else 0
                if top_feature_importance > 0.5:
                    importance_info.append("⚠️ **High feature dominance** - single feature has major influence")
                elif len(sorted_features) > 5 and sorted_features[4][1] > 0.05:
                    importance_info.append("✅ **Distributed importance** - multiple features contribute meaningfully")
                
        # Check for feature selection information
        if 'feature_selection' in node_data:
            selection_info = node_data['feature_selection']
            if isinstance(selection_info, dict):
                importance_info.append(f"🎯 **Feature Selection**: {selection_info.get('method', 'Applied')}")
        
        return "🎯 **FEATURE IMPORTANCE ANALYSIS**:\n" + "\n".join(importance_info) if importance_info else "🎯 **FEATURE IMPORTANCE ANALYSIS**: Feature importance data not available"
    
    @staticmethod
    def _analyze_data_characteristics(node_data: Dict[str, Any]) -> str:
        """Analyze data characteristics and quality"""
        data_analysis = []
        
        # Check for training data information
        if 'data' in node_data and isinstance(node_data['data'], pd.DataFrame):
            df = node_data['data']
            
            data_analysis.append(f"📊 **Dataset Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            data_analysis.append(f"🔢 **Numeric Features**: {len(numeric_cols)}")
            data_analysis.append(f"🏷️ **Categorical Features**: {len(categorical_cols)}")
            
            # Missing values
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                missing_percentage = (missing_values / (df.shape[0] * df.shape[1])) * 100
                data_analysis.append(f"❗ **Missing Values**: {missing_values:,} ({missing_percentage:.1f}%)")
            else:
                data_analysis.append("✅ **Data Completeness**: No missing values")
            
            # Data quality score
            completeness = 1 - (missing_values / (df.shape[0] * df.shape[1]))
            if completeness >= 0.95:
                quality_status = "🟢 **HIGH QUALITY** - Excellent for classification"
            elif completeness >= 0.9:
                quality_status = "🟡 **GOOD QUALITY** - Suitable for classification"
            else:
                quality_status = "🟠 **MODERATE QUALITY** - Consider data cleaning"
            
            data_analysis.append(quality_status)
        
        # Check for preprocessing information
        if 'preprocessing' in node_data:
            preprocessing = node_data['preprocessing']
            if isinstance(preprocessing, dict):
                if preprocessing.get('scaled'):
                    data_analysis.append("📏 **Feature Scaling**: Applied")
                if preprocessing.get('encoded'):
                    data_analysis.append("🔤 **Categorical Encoding**: Applied")
        
        return "📋 **DATA CHARACTERISTICS**:\n" + "\n".join(data_analysis) if data_analysis else "📋 **DATA CHARACTERISTICS**: Data information not available"
    
    @staticmethod
    def _generate_technical_insights(node_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate technical insights for classification analysis"""
        technical_insights = []
        
        # Model performance insights
        accuracy = node_data.get('metrics', {}).get('accuracy', 0)
        if accuracy > 0:
            if accuracy >= 0.9:
                technical_insights.append("🎯 **High Accuracy**: Model demonstrates strong predictive capability (>90%)")
                technical_insights.append("✅ **Production Ready**: Performance suitable for automated decision-making")
            elif accuracy >= 0.8:
                technical_insights.append("� **Good Performance**: Model shows reliable classification ability (80-90%)")
                technical_insights.append("� **Optimization Potential**: Fine-tuning could improve performance")
            else:
                technical_insights.append("⚠️ **Performance Issues**: Model requires significant improvement (<80%)")
                technical_insights.append("🛠️ **Enhancement Needed**: Consider feature engineering or algorithm changes")
        
        # Feature importance insights
        if 'feature_importance' in node_data:
            technical_insights.append("📈 **Feature Analysis**: Important features identified for model interpretation")
            technical_insights.append("� **Data Science Value**: Feature rankings guide data collection priorities")
        
        # Prediction distribution insights
        if 'predictions' in node_data:
            technical_insights.append("� **Classification Patterns**: Prediction distribution reveals model behavior")
            technical_insights.append("� **Statistical Analysis**: Class probabilities enable confidence scoring")
        
        # Data quality insights
        if 'data' in node_data and isinstance(node_data['data'], pd.DataFrame):
            df = node_data['data']
            missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_percentage < 5:
                technical_insights.append("✅ **Data Quality**: High completeness supports reliable predictions")
            else:
                technical_insights.append("⚠️ **Data Issues**: Missing values may impact model performance")
        
        # Cross-validation insights
        if 'metrics' in node_data and 'cv_scores' in node_data['metrics']:
            technical_insights.append("🔄 **Validation**: Cross-validation provides robust performance estimates")
            technical_insights.append("📊 **Stability**: Multiple evaluation folds ensure reliable results")
        
        return "� **TECHNICAL INSIGHTS**:\n" + "\n".join(technical_insights) if technical_insights else "� **TECHNICAL INSIGHTS**: Standard classification model results"
