"""
Advanced Machine Learning Node Prompt Generator
"""

import numpy as np

class MachineLearningPrompt:
    """Generate sophisticated prompts for machine learning nodes"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, node_type: str, context: dict = None) -> str:
        """Generate advanced machine learning analysis prompt"""
        
        model_info = data.get('model', {})
        predictions = data.get('predictions', [])
        performance_metrics = data.get('performance', {})
        feature_importance = data.get('feature_importance', {})
        training_info = data.get('training_info', {})
        validation_results = data.get('validation_results', {})
        
        if not model_info and not predictions:
            return f"❌ **CRITICAL ERROR**: No machine learning model or prediction data available for {node_type} analysis"
        
        # Analyze ML performance and capabilities
        model_analysis = MachineLearningPrompt._analyze_model_performance(performance_metrics, node_type)
        feature_insights = MachineLearningPrompt._analyze_feature_importance(feature_importance)
        model_evaluation = MachineLearningPrompt._assess_model_evaluation(model_info, predictions, node_type)
        validation_assessment = MachineLearningPrompt._assess_validation_results(performance_metrics, validation_results)
        technical_assessment = MachineLearningPrompt._assess_technical_aspects(model_info, performance_metrics, node_type)
        
        # Model type specific analysis
        ml_type_map = {
            'classification': '🎯 CLASSIFICATION ANALYSIS',
            'regression': '📈 REGRESSION ANALYSIS',
            'clustering': '🔬 CLUSTERING ANALYSIS',
            'anomaly_detection': '🚨 ANOMALY DETECTION'
        }
        
        header = ml_type_map.get(node_type, '🤖 MACHINE LEARNING')
        
        # Performance summary
        performance_summary = MachineLearningPrompt._generate_performance_summary(performance_metrics, node_type)
        
        prompt = f"""
🤖 **{header} - Node: {node_id}**

⚡ **MODEL OVERVIEW**:
Algorithm Type: {node_type.replace('_', ' ').title()} Model
Prediction Status: {"Active" if predictions else "Trained"}
Performance Status: {performance_summary}
Technical Status: {"Validated" if validation_results else "Training Only"}

🎯 **PERFORMANCE ANALYSIS**:
{chr(10).join(model_analysis) if model_analysis else "⚠️ Performance metrics not available"}

🔍 **FEATURE ANALYSIS**:
{chr(10).join(feature_insights) if feature_insights else "⚠️ Feature importance analysis not available"}

� **MODEL EVALUATION**:
{chr(10).join(model_evaluation) if model_evaluation else "⚠️ Model evaluation analysis pending"}

🧪 **VALIDATION ASSESSMENT**:
{chr(10).join(validation_assessment) if validation_assessment else "⚠️ Validation assessment not available"}

⚙️ **TECHNICAL ASSESSMENT**:
{chr(10).join(technical_assessment) if technical_assessment else "⚠️ Technical analysis not available"}

📊 **MODEL METADATA**:
• Training Data: {"Utilized" if training_info else "Standard"}
• Validation: {"Cross-validated" if validation_results else "Basic"}
• Feature Count: {len(feature_importance) if feature_importance else "Not specified"}
• Prediction Count: {len(predictions) if isinstance(predictions, list) else "Multiple" if predictions else "None"}

💡 **MACHINE LEARNING ANALYSIS REQUIREMENTS**:

1. **MODEL PERFORMANCE ASSESSMENT**: Evaluate the model's accuracy and reliability
2. **FEATURE IMPORTANCE ANALYSIS**: Assess which data elements drive model performance
3. **STATISTICAL VALIDITY**: Evaluate the statistical soundness of the model
4. **MODEL INTERPRETATION**: Explain model behavior and predictions
5. **DATA REPRESENTATION**: Assess how well the model represents the underlying data
6. **PREDICTION QUALITY**: Evaluate the quality and reliability of predictions
7. **MODEL LIMITATIONS**: Identify limitations and constraints of the model
8. **TECHNICAL EVALUATION**: Provide technical assessment of model implementation

🎯 **ML ANALYSIS REQUIREMENTS**:
- Quantify SPECIFIC model performance metrics
- Identify TOP PERFORMING features and their statistical relevance
- Assess MODEL RELIABILITY based on validation results
- Evaluate PREDICTION QUALITY for the intended purpose
- Describe STATISTICAL PROPERTIES of the model
- Identify MODEL STRENGTHS AND WEAKNESSES
- Report DATA REPRESENTATION accuracy

⚡ **RESPONSE FOCUS**: Analyze the ACTUAL model performance, predictions, and feature patterns demonstrated in this specific ML implementation. Provide concrete, measurable assessments of model quality and technical characteristics.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _analyze_model_performance(performance_metrics: dict, model_type: str) -> list:
        """Analyze model performance metrics"""
        analysis = []
        
        if not performance_metrics:
            analysis.append("📊 **Performance Metrics**: Basic model training completed")
            return analysis
        
        # Classification metrics
        if model_type == 'classification':
            accuracy = performance_metrics.get('accuracy')
            precision = performance_metrics.get('precision') 
            recall = performance_metrics.get('recall')
            f1_score = performance_metrics.get('f1_score')
            
            if accuracy is not None:
                if accuracy > 0.95:
                    analysis.append(f"🎯 **EXCEPTIONAL ACCURACY**: {accuracy:.1%} - Production-ready performance")
                elif accuracy > 0.85:
                    analysis.append(f"✅ **EXCELLENT ACCURACY**: {accuracy:.1%} - Strong business reliability")
                elif accuracy > 0.75:
                    analysis.append(f"📊 **GOOD ACCURACY**: {accuracy:.1%} - Suitable for decision support")
                else:
                    analysis.append(f"⚠️ **MODERATE ACCURACY**: {accuracy:.1%} - Requires improvement")
            
            if precision is not None and recall is not None:
                if precision > 0.8 and recall > 0.8:
                    analysis.append("⚖️ **BALANCED PERFORMANCE**: High precision and recall - reliable predictions")
                elif precision > recall + 0.1:
                    analysis.append("🎯 **PRECISION-FOCUSED**: Low false positive rate - conservative predictions")
                elif recall > precision + 0.1:
                    analysis.append("🔍 **RECALL-FOCUSED**: High detection rate - comprehensive coverage")
            
            if f1_score is not None:
                if f1_score > 0.85:
                    analysis.append(f"🏆 **EXCELLENT F1-SCORE**: {f1_score:.3f} - Well-balanced model")
        
        # Regression metrics
        elif model_type == 'regression':
            r2_score = performance_metrics.get('r2_score') or performance_metrics.get('r2')
            mse = performance_metrics.get('mse') or performance_metrics.get('mean_squared_error')
            mae = performance_metrics.get('mae') or performance_metrics.get('mean_absolute_error')
            
            if r2_score is not None:
                if r2_score > 0.9:
                    analysis.append(f"🎯 **EXCEPTIONAL PREDICTIVE POWER**: R² = {r2_score:.3f} - Explains {r2_score:.1%} of variance")
                elif r2_score > 0.7:
                    analysis.append(f"✅ **STRONG PREDICTIVE POWER**: R² = {r2_score:.3f} - Reliable forecasting capability")
                elif r2_score > 0.5:
                    analysis.append(f"📊 **MODERATE PREDICTIVE POWER**: R² = {r2_score:.3f} - Useful for trend analysis")
                else:
                    analysis.append(f"⚠️ **LIMITED PREDICTIVE POWER**: R² = {r2_score:.3f} - Consider feature engineering")
            
            if mse is not None and mae is not None:
                ratio = np.sqrt(mse) / mae if mae > 0 else 0
                if ratio < 1.2:
                    analysis.append("📊 **CONSISTENT ERRORS**: Low variance in prediction errors")
                else:
                    analysis.append("📈 **VARIABLE ERRORS**: Some predictions have larger errors - investigate outliers")
        
        # Clustering metrics
        elif model_type == 'clustering':
            silhouette_score = performance_metrics.get('silhouette_score')
            inertia = performance_metrics.get('inertia')
            n_clusters = performance_metrics.get('n_clusters')
            
            if silhouette_score is not None:
                if silhouette_score > 0.7:
                    analysis.append(f"🎯 **EXCELLENT CLUSTERING**: Silhouette = {silhouette_score:.3f} - Well-separated clusters")
                elif silhouette_score > 0.5:
                    analysis.append(f"✅ **GOOD CLUSTERING**: Silhouette = {silhouette_score:.3f} - Clear segmentation")
                else:
                    analysis.append(f"⚠️ **MODERATE CLUSTERING**: Silhouette = {silhouette_score:.3f} - Review cluster count")
            
            if n_clusters is not None:
                analysis.append(f"🔢 **Cluster Configuration**: {n_clusters} distinct segments identified")
        
        # Cross-validation results
        cv_scores = performance_metrics.get('cv_scores')
        if cv_scores and isinstance(cv_scores, list):
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            analysis.append(f"🔄 **Cross-Validation**: {cv_mean:.3f} ± {cv_std:.3f} - Model stability validated")
        
        return analysis
    
    @staticmethod
    def _analyze_feature_importance(feature_importance: dict) -> list:
        """Analyze feature importance for business insights"""
        insights = []
        
        if not feature_importance:
            insights.append("🔍 **Feature Analysis**: Standard feature set utilized")
            return insights
        
        # Sort features by importance
        if isinstance(feature_importance, dict):
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            insights.append(f"🏆 **Top Features Analyzed**: {len(sorted_features)} features ranked by importance")
            
            # Analyze top features
            if len(sorted_features) >= 3:
                top_3 = sorted_features[:3]
                insights.append("🎯 **Most Influential Features**:")
                for i, (feature, importance) in enumerate(top_3, 1):
                    percentage = importance * 100 if importance <= 1 else importance
                    insights.append(f"   {i}. **{feature}**: {percentage:.1f}% influence")
                
                # Check for feature dominance
                top_importance = sorted_features[0][1]
                if top_importance > 0.5:
                    insights.append(f"⚠️ **Feature Dominance**: {sorted_features[0][0]} heavily influences predictions")
                
                # Check for balanced importance
                top_5_sum = sum(imp for _, imp in sorted_features[:5])
                if top_5_sum < 0.8:
                    insights.append("⚖️ **Distributed Importance**: Multiple features contribute equally")
            
            # Business category analysis
            business_features = MachineLearningPrompt._categorize_business_features(sorted_features)
            if business_features:
                insights.extend(business_features)
        
        return insights
    
    @staticmethod
    def _categorize_business_features(sorted_features: list) -> list:
        """Categorize features by business domain"""
        categories = []
        
        feature_categories = {
            'financial': ['price', 'cost', 'revenue', 'profit', 'budget', 'expense'],
            'temporal': ['date', 'time', 'month', 'year', 'day', 'hour'],
            'demographic': ['age', 'gender', 'location', 'education', 'income'],
            'behavioral': ['frequency', 'usage', 'activity', 'engagement', 'interaction'],
            'performance': ['score', 'rating', 'efficiency', 'quality', 'accuracy']
        }
        
        category_features = {cat: [] for cat in feature_categories.keys()}
        
        for feature, importance in sorted_features:
            feature_lower = feature.lower()
            for category, keywords in feature_categories.items():
                if any(keyword in feature_lower for keyword in keywords):
                    category_features[category].append((feature, importance))
                    break
        
        # Generate insights for significant categories
        for category, features in category_features.items():
            if features and len(features) >= 2:
                total_importance = sum(imp for _, imp in features)
                categories.append(f"💼 **{category.title()} Features**: {len(features)} features contribute {total_importance:.1%} to predictions")
        
        return categories
    
    @staticmethod
    def _assess_business_value(model_info: dict, predictions, model_type: str) -> list:
        """Assess business value of the ML model"""
        value_assessment = []
        
        # Model type specific business value
        business_applications = {
            'classification': [
                "🎯 **Decision Automation**: Automate categorical business decisions",
                "🔍 **Risk Assessment**: Classify risk levels for automated screening",
                "👥 **Customer Segmentation**: Automatically categorize customers for targeted strategies"
            ],
            'regression': [
                "📈 **Forecasting**: Predict continuous business metrics and KPIs",
                "💰 **Price Optimization**: Optimize pricing strategies with predictive modeling",
                "📊 **Resource Planning**: Forecast resource requirements and capacity needs"
            ],
            'clustering': [
                "🔬 **Market Segmentation**: Discover hidden customer segments",
                "🎯 **Targeted Marketing**: Develop segment-specific marketing strategies",
                "⚡ **Operational Optimization**: Group similar processes for efficiency gains"
            ],
            'anomaly_detection': [
                "🚨 **Fraud Detection**: Automatically identify suspicious transactions",
                "🔧 **Quality Control**: Detect manufacturing defects and quality issues",
                "⚠️ **System Monitoring**: Identify system failures and performance issues"
            ]
        }
        
        applications = business_applications.get(model_type, ["🤖 **AI-Powered Insights**: Automated analytical capabilities"])
        value_assessment.extend(applications)
        
        # Quantify business impact based on predictions
        if predictions:
            pred_count = len(predictions) if isinstance(predictions, list) else "Multiple"
            value_assessment.append(f"⚡ **Immediate Impact**: {pred_count} predictions ready for business application")
            
            if model_type == 'classification':
                value_assessment.append("🎯 **Decision Support**: Automated classification reduces manual review time")
            elif model_type == 'regression':
                value_assessment.append("📊 **Forecasting Value**: Quantitative predictions enable data-driven planning")
        
        # Operational efficiency gains
        value_assessment.extend([
            "⚡ **Operational Efficiency**: Reduces manual analysis time and human error",
            "📊 **Scalable Intelligence**: Handles large datasets for consistent insights",
            "🎯 **Strategic Advantage**: Data-driven decisions provide competitive edge"
        ])
        
        return value_assessment
    
    @staticmethod
    def _assess_deployment_readiness(performance_metrics: dict, validation_results: dict) -> list:
        """Assess model readiness for production deployment"""
        readiness = []
        
        # Performance threshold checks
        if performance_metrics:
            meets_thresholds = False
            
            # Classification thresholds
            accuracy = performance_metrics.get('accuracy')
            if accuracy and accuracy > 0.8:
                meets_thresholds = True
                readiness.append("✅ **Performance Threshold**: Accuracy exceeds 80% - production ready")
            
            # Regression thresholds
            r2_score = performance_metrics.get('r2_score') or performance_metrics.get('r2')
            if r2_score and r2_score > 0.7:
                meets_thresholds = True
                readiness.append("✅ **Predictive Threshold**: R² exceeds 70% - reliable forecasting")
            
            if not meets_thresholds and (accuracy or r2_score):
                readiness.append("⚠️ **Performance Review**: Model performance requires validation before production")
        
        # Validation readiness
        if validation_results:
            readiness.append("✅ **Validation Complete**: Model validated for consistent performance")
        else:
            readiness.append("🔄 **Validation Needed**: Implement cross-validation for production readiness")
        
        # Deployment considerations
        readiness.extend([
            "🔧 **Infrastructure**: Requires ML model serving infrastructure",
            "📊 **Monitoring**: Implement performance monitoring and alerting",
            "🔄 **Retraining**: Establish data pipeline for model updates",
            "📝 **Documentation**: Complete model documentation for operational team"
        ])
        
        return readiness
    
    @staticmethod
    def _assess_model_risks(model_info: dict, performance_metrics: dict, model_type: str) -> list:
        """Assess potential risks and limitations of the model"""
        risks = []
        
        # Performance-based risks
        if performance_metrics:
            accuracy = performance_metrics.get('accuracy')
            r2_score = performance_metrics.get('r2_score') or performance_metrics.get('r2')
            
            if accuracy and accuracy < 0.9:
                error_rate = 1 - accuracy
                risks.append(f"⚠️ **Prediction Errors**: {error_rate:.1%} error rate may impact business decisions")
            
            if r2_score and r2_score < 0.8:
                unexplained = 1 - r2_score
                risks.append(f"📊 **Prediction Uncertainty**: {unexplained:.1%} of variance unexplained")
        
        # Model type specific risks
        type_risks = {
            'classification': [
                "🎯 **Misclassification Risk**: Incorrect categories may lead to wrong business actions",
                "⚖️ **Class Imbalance**: Ensure balanced representation of all business categories"
            ],
            'regression': [
                "📈 **Extrapolation Risk**: Predictions outside training range may be unreliable",
                "📊 **Outlier Sensitivity**: Extreme values may skew predictions"
            ],
            'clustering': [
                "🔢 **Cluster Stability**: Cluster assignments may change with new data",
                "🎯 **Interpretability**: Business meaning of clusters requires validation"
            ],
            'anomaly_detection': [
                "🚨 **False Positives**: Normal behavior flagged as anomalous",
                "🔍 **False Negatives**: Actual anomalies not detected by model"
            ]
        }
        
        risks.extend(type_risks.get(model_type, ["🤖 **Model Limitations**: Monitor for unexpected behaviors"]))
        
        # General ML risks
        risks.extend([
            "🔄 **Data Drift**: Model performance may degrade as data patterns change",
            "🛡️ **Bias Risk**: Ensure training data represents all business scenarios",
            "📊 **Overfitting**: Model may not generalize to new business conditions"
        ])
        
        return risks
    
    @staticmethod
    def _generate_performance_summary(performance_metrics: dict, model_type: str) -> str:
        """Generate a concise performance summary"""
        if not performance_metrics:
            return "Training Complete"
        
        if model_type == 'classification':
            accuracy = performance_metrics.get('accuracy')
            if accuracy:
                if accuracy > 0.9:
                    return f"Excellent ({accuracy:.1%})"
                elif accuracy > 0.8:
                    return f"Good ({accuracy:.1%})"
                else:
                    return f"Moderate ({accuracy:.1%})"
        
        elif model_type == 'regression':
            r2_score = performance_metrics.get('r2_score') or performance_metrics.get('r2')
            if r2_score:
                if r2_score > 0.8:
                    return f"Strong (R²={r2_score:.3f})"
                elif r2_score > 0.6:
                    return f"Good (R²={r2_score:.3f})"
                else:
                    return f"Moderate (R²={r2_score:.3f})"
        
        elif model_type == 'clustering':
            silhouette = performance_metrics.get('silhouette_score')
            if silhouette:
                if silhouette > 0.6:
                    return f"Well-Separated (S={silhouette:.3f})"
                else:
                    return f"Clustered (S={silhouette:.3f})"
        
        return "Metrics Available"
