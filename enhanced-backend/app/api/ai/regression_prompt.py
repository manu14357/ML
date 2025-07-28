"""
Advanced Regression Prompt Generator
Generates sophisticated prompts for regression node analysis based on actual model outputs, metrics, and prediction data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json

class RegressionPrompt:
    """Advanced prompt generator for regression analysis nodes"""
    
    @staticmethod
    def generate_advanced_prompt(node_data: Dict[str, Any], node_id: str, context: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive regression analysis prompt based on actual model results and predictions
        
        Args:
            node_data: Regression node output data including model, predictions, metrics
            node_id: Unique identifier for this regression node
            context: Additional workflow context
            
        Returns:
            Advanced prompt string for AI analysis
        """
        
        # Extract regression components
        model_info = RegressionPrompt._extract_model_information(node_data)
        performance_metrics = RegressionPrompt._extract_performance_metrics(node_data)
        prediction_analysis = RegressionPrompt._analyze_predictions(node_data)
        feature_importance = RegressionPrompt._extract_feature_importance(node_data)
        residual_analysis = RegressionPrompt._analyze_residuals(node_data)
        technical_insights = RegressionPrompt._generate_technical_insights(node_data, context)
        
        prompt = f"""
📈 **ADVANCED REGRESSION MODEL ANALYSIS: {node_id.upper()}**

═══════════════════════════════════════════════════════════════════
**REGRESSION MODEL INTELLIGENCE REPORT**
═══════════════════════════════════════════════════════════════════

{model_info}

{performance_metrics}

{prediction_analysis}

{feature_importance}

{residual_analysis}

{technical_insights}

🚀 **TECHNICAL REGRESSION ANALYSIS REQUIREMENTS**:

**CRITICAL**: Provide detailed regression model analysis focusing on technical performance and statistical insights:

1. **MODEL PERFORMANCE EVALUATION**:
   - Analyze R², RMSE, MAE, and other regression metrics
   - Evaluate prediction accuracy across different value ranges
   - Assess model fit quality and explain variance captured
   - Compare training vs validation performance for overfitting detection

2. **PREDICTION QUALITY ASSESSMENT**:
   - Examine prediction vs actual value distributions
   - Analyze prediction residuals for patterns and bias
   - Evaluate model confidence intervals and uncertainty quantification
   - Identify outliers and their impact on model performance

3. **FEATURE ANALYSIS INSIGHTS**:
   - Identify most influential features for target variable prediction
   - Analyze feature coefficients and their statistical significance
   - Detect multicollinearity issues among predictor variables
   - Suggest feature engineering opportunities for improvement

4. **RESIDUAL AND ERROR ANALYSIS**:
   - Examine residual distribution for normality and homoscedasticity
   - Identify systematic prediction errors or bias patterns
   - Analyze residual vs fitted plots for model assumptions validation
   - Detect heteroscedasticity or non-linear patterns in errors

5. **MODEL ASSUMPTIONS VALIDATION**:
   - Evaluate linearity assumptions and transformation needs
   - Assess independence of residuals and autocorrelation issues
   - Check for outliers and influential data points
   - Validate normality of error distribution

6. **STATISTICAL SIGNIFICANCE INSIGHTS**:
   - Analyze p-values and confidence intervals for feature importance
   - Evaluate overall model significance and goodness of fit
   - Identify statistically significant predictors
   - Recommend model simplification or complexity adjustments

**NODE CONTEXT**: {json.dumps(context, indent=2) if context else 'Standard regression node in data science workflow'}

**OUTPUT REQUIREMENTS**: Provide technical, statistically-focused insights that help improve model performance and understand regression relationships.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _extract_model_information(node_data: Dict[str, Any]) -> str:
        """Extract and format regression model information"""
        model_info = []
        
        # Check for model details
        if 'model' in node_data:
            model = node_data['model']
            if hasattr(model, '__class__'):
                model_type = model.__class__.__name__
                model_info.append(f"🤖 **Model Type**: {model_type}")
                
                # Extract model parameters if available
                if hasattr(model, 'get_params'):
                    try:
                        params = model.get_params()
                        key_params = {k: v for k, v in params.items() if k in ['alpha', 'fit_intercept', 'n_estimators', 'max_depth', 'degree']}
                        if key_params:
                            model_info.append(f"⚙️ **Key Parameters**: {key_params}")
                    except:
                        pass
        
        # Check for algorithm information
        if 'algorithm' in node_data:
            model_info.append(f"🔧 **Algorithm**: {node_data['algorithm']}")
        
        # Check for training information
        if 'config' in node_data:
            config = node_data['config']
            if 'target_column' in config:
                model_info.append(f"🎯 **Target Variable**: {config['target_column']}")
            if 'feature_columns' in config:
                feature_count = len(config['feature_columns']) if config['feature_columns'] else 'Auto-selected'
                model_info.append(f"📊 **Features Used**: {feature_count} features")
        
        # Check for dataset split information
        if 'train_size' in node_data:
            model_info.append(f"📈 **Training Data**: {node_data['train_size']} samples")
        if 'test_size' in node_data:
            model_info.append(f"🧪 **Test Data**: {node_data['test_size']} samples")
        
        return "📋 **MODEL ARCHITECTURE**:\n" + "\n".join(model_info) if model_info else "📋 **MODEL ARCHITECTURE**: Model information not available"
    
    @staticmethod
    def _extract_performance_metrics(node_data: Dict[str, Any]) -> str:
        """Extract and format regression performance metrics"""
        metrics_info = []
        
        # Check for metrics in node_data
        if 'metrics' in node_data:
            metrics = node_data['metrics']
            
            # R-squared
            if 'r2_score' in metrics:
                r2 = metrics['r2_score']
                metrics_info.append(f"📊 **R² Score**: {r2:.4f} ({r2*100:.1f}% variance explained)")
                
                # R² interpretation
                if r2 >= 0.9:
                    r2_quality = "🟢 **EXCELLENT** - Highly accurate predictions"
                elif r2 >= 0.8:
                    r2_quality = "🟡 **GOOD** - Strong predictive capability"
                elif r2 >= 0.6:
                    r2_quality = "🟠 **MODERATE** - Reasonable predictive power"
                else:
                    r2_quality = "🔴 **NEEDS IMPROVEMENT** - Limited predictive accuracy"
                
                metrics_info.append(r2_quality)
            
            # Mean Squared Error
            if 'mse' in metrics:
                mse = metrics['mse']
                metrics_info.append(f"📐 **Mean Squared Error**: {mse:.4f}")
            
            # Root Mean Squared Error
            if 'rmse' in metrics:
                rmse = metrics['rmse']
                metrics_info.append(f"📏 **Root Mean Squared Error**: {rmse:.4f}")
            elif 'mse' in metrics:
                rmse = np.sqrt(metrics['mse'])
                metrics_info.append(f"📏 **Root Mean Squared Error**: {rmse:.4f}")
            
            # Mean Absolute Error
            if 'mae' in metrics:
                mae = metrics['mae']
                metrics_info.append(f"📊 **Mean Absolute Error**: {mae:.4f}")
            
            # Cross-validation scores
            if 'cv_scores' in metrics:
                cv_scores = metrics['cv_scores']
                if isinstance(cv_scores, (list, np.ndarray)):
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    metrics_info.append(f"🔄 **Cross-Validation R²**: {cv_mean:.4f} ± {cv_std:.4f}")
        
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
                
                # Prediction statistics
                prediction_analysis.append(f"📈 **Total Predictions**: {len(predictions):,}")
                prediction_analysis.append(f"📊 **Prediction Range**: {predictions.min():.3f} to {predictions.max():.3f}")
                prediction_analysis.append(f"🎯 **Mean Prediction**: {predictions.mean():.3f}")
                prediction_analysis.append(f"📐 **Std Deviation**: {predictions.std():.3f}")
                
                # Prediction distribution analysis
                q1, median, q3 = np.percentile(predictions, [25, 50, 75])
                prediction_analysis.append(f"📈 **Quartiles**: Q1={q1:.3f}, Median={median:.3f}, Q3={q3:.3f}")
                
                # Check for prediction outliers
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = np.sum((predictions < lower_bound) | (predictions > upper_bound))
                outlier_percentage = (outliers / len(predictions)) * 100
                
                if outlier_percentage > 5:
                    prediction_analysis.append(f"⚠️ **Prediction Outliers**: {outliers} ({outlier_percentage:.1f}%) - Review extreme predictions")
                else:
                    prediction_analysis.append(f"✅ **Prediction Quality**: {outlier_percentage:.1f}% outliers - Normal distribution")
        
        # Check for actual vs predicted comparison
        if 'actual_values' in node_data and 'predictions' in node_data:
            prediction_analysis.append("📊 **Actual vs Predicted**: Comparison data available for accuracy assessment")
        
        return "🔮 **PREDICTION ANALYSIS**:\n" + "\n".join(prediction_analysis) if prediction_analysis else "🔮 **PREDICTION ANALYSIS**: Prediction data not available"
    
    @staticmethod
    def _extract_feature_importance(node_data: Dict[str, Any]) -> str:
        """Extract and analyze feature importance for regression"""
        importance_info = []
        
        # Check for feature importance
        if 'feature_importance' in node_data:
            feature_importance = node_data['feature_importance']
            
            if isinstance(feature_importance, dict):
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                
                importance_info.append(f"🔍 **Top Features** (by importance):")
                
                for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                    direction = "↗️" if importance > 0 else "↘️"
                    importance_info.append(f"   {i}. **{feature}**: {importance:.4f} {direction}")
                
                # Feature importance insights
                top_feature_importance = abs(sorted_features[0][1]) if sorted_features else 0
                if top_feature_importance > 0.5:
                    importance_info.append("⚠️ **High feature dominance** - single feature drives most variance")
                elif len(sorted_features) > 5 and abs(sorted_features[4][1]) > 0.05:
                    importance_info.append("✅ **Distributed importance** - multiple features contribute meaningfully")
        
        # Check for coefficient information (for linear models)
        if 'coefficients' in node_data:
            coefficients = node_data['coefficients']
            if isinstance(coefficients, dict):
                importance_info.append("📊 **Linear Coefficients**: Available for direct interpretation")
        
        return "🎯 **FEATURE IMPORTANCE ANALYSIS**:\n" + "\n".join(importance_info) if importance_info else "🎯 **FEATURE IMPORTANCE ANALYSIS**: Feature importance data not available"
    
    @staticmethod
    def _analyze_residuals(node_data: Dict[str, Any]) -> str:
        """Analyze residuals and model fit quality"""
        residual_analysis = []
        
        # Check for residual information
        if 'residuals' in node_data:
            residuals = node_data['residuals']
            
            if isinstance(residuals, (list, np.ndarray, pd.Series)):
                residuals = np.array(residuals)
                
                # Residual statistics
                residual_mean = residuals.mean()
                residual_std = residuals.std()
                
                residual_analysis.append(f"📐 **Residual Mean**: {residual_mean:.4f}")
                residual_analysis.append(f"📊 **Residual Std**: {residual_std:.4f}")
                
                # Check for residual patterns
                if abs(residual_mean) < 0.01:
                    residual_analysis.append("✅ **Unbiased Model**: Residuals centered around zero")
                else:
                    residual_analysis.append("⚠️ **Potential Bias**: Residuals show systematic deviation")
                
                # Residual distribution
                q1, median, q3 = np.percentile(residuals, [25, 50, 75])
                if abs(median) < residual_std * 0.1:
                    residual_analysis.append("✅ **Symmetric Residuals**: Good model fit indication")
                else:
                    residual_analysis.append("⚠️ **Asymmetric Residuals**: Potential model improvement needed")
        
        # Check for actual vs predicted for residual calculation
        elif 'actual_values' in node_data and 'predictions' in node_data:
            residual_analysis.append("📊 **Residual Analysis**: Can be calculated from actual vs predicted values")
        
        return "📈 **RESIDUAL ANALYSIS**:\n" + "\n".join(residual_analysis) if residual_analysis else "📈 **RESIDUAL ANALYSIS**: Residual data not available"
    
    @staticmethod
    def _generate_technical_insights(node_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate technical insights for regression analysis"""
        technical_insights = []
        
        # Model performance technical assessment
        r2_score = node_data.get('metrics', {}).get('r2_score', 0)
        if r2_score > 0:
            if r2_score >= 0.9:
                technical_insights.append("🎯 **Excellent Fit**: Model explains >90% of variance - strong predictive capability")
                technical_insights.append("✅ **Statistical Significance**: High R² indicates robust linear relationship")
            elif r2_score >= 0.8:
                technical_insights.append("� **Good Fit**: Model captures 80-90% of variance - reliable predictions")
                technical_insights.append("🔧 **Optimization Potential**: Minor improvements possible through feature engineering")
            elif r2_score >= 0.6:
                technical_insights.append("⚠️ **Moderate Fit**: Model explains 60-80% of variance - consider complexity adjustments")
            else:
                technical_insights.append("�️ **Poor Fit**: Low R² (<60%) indicates need for model redesign or feature enhancement")
        
        # Feature importance technical insights
        if 'feature_importance' in node_data:
            technical_insights.append("📈 **Feature Analysis**: Coefficient analysis reveals variable relationships with target")
            technical_insights.append("� **Statistical Significance**: Feature importance guides feature selection decisions")
        
        # Prediction accuracy insights
        if 'predictions' in node_data:
            predictions = np.array(node_data['predictions'])
            pred_std = predictions.std()
            technical_insights.append(f"📊 **Prediction Variance**: Standard deviation of {pred_std:.3f} indicates prediction consistency")
            technical_insights.append("🎲 **Model Behavior**: Prediction distribution reveals model characteristics")
        
        # Error metrics technical interpretation
        mae = node_data.get('metrics', {}).get('mae')
        rmse = node_data.get('metrics', {}).get('rmse')
        if mae is not None and rmse is not None:
            error_ratio = rmse / mae if mae > 0 else 0
            if error_ratio > 1.5:
                technical_insights.append("⚠️ **Error Pattern**: RMSE/MAE ratio suggests outliers impacting model performance")
            else:
                technical_insights.append("✅ **Consistent Errors**: RMSE/MAE ratio indicates stable prediction accuracy")
        
        # Residual analysis insights
        if 'residuals' in node_data:
            technical_insights.append("� **Residual Analysis**: Error patterns available for model assumption validation")
            technical_insights.append("� **Statistical Validation**: Residual examination enables model improvement")
        
        # Cross-validation insights
        if 'metrics' in node_data and 'cv_scores' in node_data['metrics']:
            technical_insights.append("🔄 **Cross-Validation**: Multiple folds provide robust performance estimates")
            technical_insights.append("� **Model Stability**: CV scores indicate generalization capability")
        
        return "� **TECHNICAL INSIGHTS**:\n" + "\n".join(technical_insights) if technical_insights else "� **TECHNICAL INSIGHTS**: Standard regression model results"
