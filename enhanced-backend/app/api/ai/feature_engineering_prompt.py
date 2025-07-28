"""
Advanced Feature Engineering Node Prompt Generator
"""

import pandas as pd
import numpy as np

class FeatureEngineeringPrompt:
    """Generate sophisticated prompts for feature engineering nodes"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, context: dict = None) -> str:
        """Generate advanced feature engineering analysis prompt"""
        
        df = data.get('dataframe')
        original_features = data.get('original_features', [])
        new_features = data.get('new_features', [])
        feature_transformations = data.get('transformations', {})
        feature_importance = data.get('feature_importance', {})
        engineering_stats = data.get('engineering_stats', {})
        
        if df is None and not new_features:
            return "❌ **CRITICAL ERROR**: No dataframe or feature engineering results available"
        
        # Analyze feature engineering impact
        feature_analysis = FeatureEngineeringPrompt._analyze_feature_impact(df, original_features, new_features)
        transformation_insights = FeatureEngineeringPrompt._analyze_transformations(feature_transformations)
        data_value = FeatureEngineeringPrompt._assess_data_value(df, new_features, feature_importance)
        modeling_impact = FeatureEngineeringPrompt._assess_modeling_impact(df, new_features, feature_importance)
        optimization_opportunities = FeatureEngineeringPrompt._identify_optimization_opportunities(df, feature_transformations)
        
        # Feature engineering summary
        original_count = len(original_features) if original_features else (df.shape[1] if df is not None else 0)
        new_count = len(new_features) if new_features else 0
        total_features = original_count + new_count
        
        prompt = f"""
⚙️ **FEATURE ENGINEERING LABORATORY - Node: {node_id}**

🔧 **FEATURE ENGINEERING OVERVIEW**:
Original Features: {original_count}
Engineered Features: {new_count}
Total Feature Space: {total_features}
Enhancement Ratio: {(new_count / max(1, original_count) * 100):.0f}% expansion

🎯 **FEATURE IMPACT ANALYSIS**:
{chr(10).join(feature_analysis) if feature_analysis else "⚠️ Feature impact analysis not available"}

🔄 **TRANSFORMATION INTELLIGENCE**:
{chr(10).join(transformation_insights) if transformation_insights else "⚠️ Transformation analysis not available"}

� **DATA VALUE ASSESSMENT**:
{chr(10).join(data_value) if data_value else "⚠️ Data value analysis not available"}

🤖 **MODELING IMPACT EVALUATION**:
{chr(10).join(modeling_impact) if modeling_impact else "⚠️ Modeling impact assessment not available"}

🚀 **OPTIMIZATION OPPORTUNITIES**:
{chr(10).join(optimization_opportunities) if optimization_opportunities else "⚠️ Optimization analysis not available"}

📊 **ENGINEERING METADATA**:
• Transformation Types: {len(feature_transformations) if feature_transformations else "Standard"}
• Feature Importance Available: {"Yes" if feature_importance else "No"}
• Engineering Statistics: {"Available" if engineering_stats else "Basic"}
• Data Quality Impact: {"Assessed" if df is not None else "Pending"}

💡 **ADVANCED FEATURE ENGINEERING INTELLIGENCE REQUIREMENTS**:

1. **FEATURE VALUE QUANTIFICATION**: Measure the statistical and predictive value of each engineered feature
2. **TRANSFORMATION EFFECTIVENESS**: Assess how transformations improve data quality and model performance
3. **TECHNICAL INTERPRETATION**: Explain engineered features in statistical and mathematical terms
4. **PREDICTIVE ENHANCEMENT**: Quantify how feature engineering improves predictive modeling capability
5. **COMPUTATIONAL EFFICIENCY**: Evaluate the processing requirements of implementing engineered features
6. **STATISTICAL SIGNIFICANCE**: Assess how well features capture underlying data patterns
7. **SCALABILITY ASSESSMENT**: Evaluate computational complexity of feature transformations
8. **INFORMATION GAIN**: Identify features that provide significant statistical information

🎯 **CRITICAL FEATURE ENGINEERING ANALYSIS REQUIREMENTS**:
- Quantify SPECIFIC improvements in data representation and modeling potential
- Identify MOST VALUABLE engineered features and their technical interpretations
- Assess COMPUTATIONAL EFFICIENCY and scalability of feature transformations
- Evaluate STATISTICAL RELEVANCE of engineered features
- Recommend ADDITIONAL feature engineering opportunities for enhanced performance
- Establish FEATURE MONITORING metrics for data quality assessment
- Assess INTERPRETABILITY vs PERFORMANCE trade-offs in feature engineering decisions

⚡ **RESPONSE FOCUS**: Analyze the ACTUAL engineered features, their transformations, and demonstrated impact on data quality and modeling potential. Provide concrete assessments of statistical value and actionable recommendations for feature optimization.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _analyze_feature_impact(df, original_features: list, new_features: list) -> list:
        """Analyze the impact of feature engineering"""
        impacts = []
        
        # Feature space expansion analysis
        original_count = len(original_features) if original_features else 0
        new_count = len(new_features) if new_features else 0
        
        if new_count > 0:
            expansion_ratio = (new_count / max(1, original_count)) * 100
            
            if expansion_ratio > 100:
                impacts.append(f"🚀 **SIGNIFICANT EXPANSION**: {expansion_ratio:.0f}% feature space growth - major analytical enhancement")
            elif expansion_ratio > 50:
                impacts.append(f"📈 **SUBSTANTIAL EXPANSION**: {expansion_ratio:.0f}% feature space growth - meaningful enhancement")
            elif expansion_ratio > 20:
                impacts.append(f"📊 **MODERATE EXPANSION**: {expansion_ratio:.0f}% feature space growth - targeted improvement")
            else:
                impacts.append(f"🎯 **FOCUSED EXPANSION**: {expansion_ratio:.0f}% feature space growth - selective enhancement")
        
        # Feature type analysis
        if df is not None and new_features:
            # Analyze new feature characteristics
            numeric_new = 0
            categorical_new = 0
            
            for feature in new_features:
                if feature in df.columns:
                    if df[feature].dtype in ['int64', 'float64']:
                        numeric_new += 1
                    else:
                        categorical_new += 1
            
            if numeric_new > categorical_new:
                impacts.append(f"🔢 **QUANTITATIVE FOCUS**: {numeric_new} numeric features enhance mathematical modeling")
            elif categorical_new > numeric_new:
                impacts.append(f"🏷️ **CATEGORICAL FOCUS**: {categorical_new} categorical features enhance segmentation")
            else:
                impacts.append(f"⚖️ **BALANCED ENHANCEMENT**: Mixed feature types for comprehensive modeling")
            
            # Data quality impact
            if len(new_features) > 0:
                # Check for missing values in new features
                new_feature_completeness = []
                for feature in new_features[:5]:  # Check top 5 new features
                    if feature in df.columns:
                        missing_pct = (df[feature].isnull().sum() / len(df)) * 100
                        if missing_pct < 1:
                            new_feature_completeness.append(f"✅ {feature}: {100-missing_pct:.1f}% complete")
                        elif missing_pct < 10:
                            new_feature_completeness.append(f"📊 {feature}: {100-missing_pct:.1f}% complete")
                        else:
                            new_feature_completeness.append(f"⚠️ {feature}: {100-missing_pct:.1f}% complete")
                
                if new_feature_completeness:
                    impacts.append("📋 **NEW FEATURE QUALITY**:")
                    impacts.extend([f"   {quality}" for quality in new_feature_completeness])
        
        # Feature naming and interpretability analysis
        if new_features:
            interpretable_features = []
            complex_features = []
            
            for feature in new_features:
                feature_lower = feature.lower()
                
                # Check for interpretable feature patterns
                if any(pattern in feature_lower for pattern in ['ratio', 'rate', 'percentage', 'avg', 'mean', 'sum', 'count']):
                    interpretable_features.append(feature)
                elif any(pattern in feature_lower for pattern in ['interaction', 'poly', 'log', 'sqrt', 'encoded']):
                    complex_features.append(feature)
            
            if interpretable_features:
                impacts.append(f"💡 **INTERPRETABLE FEATURES**: {len(interpretable_features)} business-intuitive features created")
            
            if complex_features:
                impacts.append(f"🔬 **ADVANCED FEATURES**: {len(complex_features)} mathematical transformations for enhanced modeling")
        
        return impacts
    
    @staticmethod
    def _analyze_transformations(transformations: dict) -> list:
        """Analyze the types and effectiveness of transformations"""
        insights = []
        
        if not transformations:
            insights.append("🔧 **Standard Transformations**: Basic feature engineering applied")
            return insights
        
        # Transformation type analysis
        transformation_categories = {
            'Mathematical': ['log', 'sqrt', 'square', 'polynomial', 'power'],
            'Statistical': ['standardize', 'normalize', 'scale', 'zscore'],
            'Interaction': ['interaction', 'multiply', 'combine', 'cross'],
            'Aggregation': ['sum', 'mean', 'count', 'max', 'min', 'groupby'],
            'Temporal': ['date', 'time', 'lag', 'diff', 'rolling'],
            'Categorical': ['encode', 'dummy', 'onehot', 'label'],
            'Binning': ['bin', 'bucket', 'discretize', 'quantile']
        }
        
        applied_categories = {}
        for category, keywords in transformation_categories.items():
            category_count = 0
            for transform_name, transform_info in transformations.items():
                if any(keyword in transform_name.lower() for keyword in keywords):
                    category_count += 1
            if category_count > 0:
                applied_categories[category] = category_count
        
        # Report applied transformation categories
        if applied_categories:
            insights.append("🔧 **TRANSFORMATION PORTFOLIO**:")
            for category, count in applied_categories.items():
                insights.append(f"   📊 {category}: {count} transformations")
            
            # Assess transformation sophistication
            total_categories = len(applied_categories)
            if total_categories >= 5:
                insights.append("🎯 **COMPREHENSIVE ENGINEERING**: Multi-dimensional transformation approach")
            elif total_categories >= 3:
                insights.append("📊 **SUBSTANTIAL ENGINEERING**: Well-rounded transformation strategy")
            elif total_categories >= 2:
                insights.append("🔧 **FOCUSED ENGINEERING**: Targeted transformation approach")
        
        # Specific transformation analysis
        for transform_name, transform_details in list(transformations.items())[:5]:
            if isinstance(transform_details, dict):
                # Analyze transformation effectiveness
                if 'before' in transform_details and 'after' in transform_details:
                    insights.append(f"📈 **{transform_name}**: Transformation applied with before/after comparison")
                elif 'target_features' in transform_details:
                    target_count = len(transform_details['target_features'])
                    insights.append(f"🎯 **{transform_name}**: Applied to {target_count} target features")
                else:
                    insights.append(f"🔧 **{transform_name}**: Transformation successfully applied")
        
        return insights
    
    @staticmethod
    def _assess_data_value(df, new_features: list, feature_importance: dict) -> list:
        """Assess the data value of engineered features"""
        data_value = []
        
        if not new_features:
            data_value.append("� **Standard Data Value**: Basic feature set maintained")
            return data_value
        
        # Feature importance analysis for data value
        if feature_importance:
            # Find new features in importance ranking
            important_new_features = []
            for feature in new_features:
                if feature in feature_importance:
                    importance = feature_importance[feature]
                    if importance > 0.1:  # High importance threshold
                        important_new_features.append((feature, importance))
            
            if important_new_features:
                # Sort by importance
                important_new_features.sort(key=lambda x: x[1], reverse=True)
                data_value.append(f"💎 **HIGH-VALUE FEATURES**: {len(important_new_features)} engineered features show high predictive importance")
                
                # Top important new features
                for feature, importance in important_new_features[:3]:
                    data_value.append(f"   🎯 **{feature}**: {importance:.1%} importance - significant statistical impact")
        
        # Technical domain analysis of new features
        if new_features:
            data_domains = FeatureEngineeringPrompt._analyze_data_domains(new_features)
            data_value.extend(data_domains)
        
        # Data completeness and reliability
        if df is not None and new_features:
            reliable_features = 0
            for feature in new_features:
                if feature in df.columns:
                    missing_pct = (df[feature].isnull().sum() / len(df)) * 100
                    if missing_pct < 5:  # Less than 5% missing
                        reliable_features += 1
            
            reliability_ratio = reliable_features / len(new_features)
            if reliability_ratio > 0.9:
                data_value.append("✅ **HIGH RELIABILITY**: >90% of new features have excellent data quality")
            elif reliability_ratio > 0.7:
                data_value.append("📊 **GOOD RELIABILITY**: >70% of new features have acceptable data quality")
            else:
                data_value.append("⚠️ **RELIABILITY CONCERN**: Some new features have data quality issues")
        
        # Technical interpretability
        interpretable_count = 0
        for feature in new_features:
            feature_lower = feature.lower()
            if any(term in feature_lower for term in ['ratio', 'rate', 'avg', 'total', 'count', 'percentage']):
                interpretable_count += 1
        
        if interpretable_count > 0:
            interpretability_ratio = interpretable_count / len(new_features)
            data_value.append(f"💡 **STATISTICAL INTERPRETABILITY**: {interpretability_ratio:.1%} of new features have clear statistical meaning")
        
        return data_value
    
    @staticmethod
    def _analyze_data_domains(new_features: list) -> list:
        """Analyze data domains represented in new features"""
        domain_analysis = []
        
        # Data domain categorization
        domain_keywords = {
            'Numerical': ['revenue', 'cost', 'profit', 'price', 'budget', 'expense', 'roi'],
            'Entity': ['customer', 'user', 'client', 'satisfaction', 'loyalty', 'retention'],
            'Process': ['efficiency', 'productivity', 'utilization', 'capacity', 'throughput'],
            'Quality': ['quality', 'defect', 'error', 'accuracy', 'performance', 'score'],
            'Temporal': ['trend', 'seasonality', 'lag', 'growth', 'velocity', 'duration'],
            'Variability': ['risk', 'volatility', 'variance', 'deviation', 'stability'],
            'Interaction': ['conversion', 'engagement', 'reach', 'impression', 'click', 'campaign']
        }
        
        domain_features = {domain: [] for domain in domain_keywords.keys()}
        
        for feature in new_features:
            feature_lower = feature.lower()
            for domain, keywords in domain_keywords.items():
                if any(keyword in feature_lower for keyword in keywords):
                    domain_features[domain].append(feature)
                    break
        
        # Generate insights for represented domains
        for domain, features in domain_features.items():
            if features:
                domain_analysis.append(f"� **{domain} Domain**: {len(features)} features enhance {domain.lower()} analytics")
        
        return domain_analysis
    
    @staticmethod
    def _assess_modeling_impact(df, new_features: list, feature_importance: dict) -> list:
        """Assess the impact on modeling capabilities"""
        modeling_impact = []
        
        if not new_features:
            modeling_impact.append("🤖 **Standard Modeling**: Basic feature set for modeling")
            return modeling_impact
        
        # Feature space enhancement
        new_count = len(new_features)
        if new_count >= 10:
            modeling_impact.append(f"🚀 **SIGNIFICANT ENHANCEMENT**: {new_count} new features dramatically expand modeling potential")
        elif new_count >= 5:
            modeling_impact.append(f"📈 **SUBSTANTIAL ENHANCEMENT**: {new_count} new features meaningfully improve modeling")
        else:
            modeling_impact.append(f"🎯 **TARGETED ENHANCEMENT**: {new_count} new features provide focused modeling improvement")
        
        # Feature type diversity for modeling
        if df is not None and new_features:
            feature_types = []
            for feature in new_features:
                if feature in df.columns:
                    if df[feature].dtype in ['int64', 'float64']:
                        feature_types.append('numeric')
                    else:
                        feature_types.append('categorical')
            
            numeric_count = feature_types.count('numeric')
            categorical_count = feature_types.count('categorical')
            
            if numeric_count > 0 and categorical_count > 0:
                modeling_impact.append("⚖️ **MIXED FEATURES**: Both numeric and categorical features enhance model flexibility")
            elif numeric_count > categorical_count:
                modeling_impact.append("🔢 **QUANTITATIVE FOCUS**: Numeric features improve mathematical modeling precision")
            else:
                modeling_impact.append("🏷️ **CATEGORICAL FOCUS**: Categorical features enhance segmentation and classification")
        
        # Predictive power assessment
        if feature_importance:
            # Calculate aggregate importance of new features
            new_feature_importance = sum(feature_importance.get(feature, 0) for feature in new_features)
            
            if new_feature_importance > 0.3:
                modeling_impact.append(f"💎 **HIGH PREDICTIVE VALUE**: New features contribute {new_feature_importance:.1%} to model performance")
            elif new_feature_importance > 0.1:
                modeling_impact.append(f"📊 **MODERATE PREDICTIVE VALUE**: New features contribute {new_feature_importance:.1%} to model performance")
            else:
                modeling_impact.append("🔍 **SUPPLEMENTARY VALUE**: New features provide additional modeling context")
        
        # Feature interaction potential
        if new_count >= 3:
            modeling_impact.append("🔗 **INTERACTION POTENTIAL**: Multiple new features enable interaction and ensemble modeling")
        
        # Model type suitability
        modeling_applications = []
        for feature in new_features:
            feature_lower = feature.lower()
            
            if any(term in feature_lower for term in ['ratio', 'rate', 'percentage']):
                modeling_applications.append("regression")
            elif any(term in feature_lower for term in ['category', 'class', 'type', 'group']):
                modeling_applications.append("classification")
            elif any(term in feature_lower for term in ['cluster', 'segment']):
                modeling_applications.append("clustering")
        
        unique_applications = list(set(modeling_applications))
        if len(unique_applications) >= 2:
            modeling_impact.append(f"🎯 **MULTI-MODEL SUPPORT**: Features support {', '.join(unique_applications)} modeling approaches")
        
        return modeling_impact
    
    @staticmethod
    def _identify_optimization_opportunities(df, transformations: dict) -> list:
        """Identify additional optimization opportunities"""
        opportunities = []
        
        if df is not None:
            # Identify potential additional transformations
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Mathematical transformation opportunities
            if len(numeric_cols) >= 2:
                opportunities.append("🔢 **INTERACTION OPPORTUNITIES**: Create ratio and interaction features between numeric variables")
            
            if len(numeric_cols) >= 3:
                opportunities.append("📊 **AGGREGATION OPPORTUNITIES**: Develop statistical aggregations and rolling window features")
            
            # Categorical enhancement opportunities
            if len(categorical_cols) >= 2:
                opportunities.append("🏷️ **CATEGORICAL COMBINATIONS**: Combine categorical variables for multi-dimensional segmentation")
            
            # Time-based opportunities
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols:
                opportunities.append("📅 **TEMPORAL FEATURES**: Extract time-based features (day of week, seasonality, trends)")
            
            # Domain-specific opportunities
            column_names = df.columns.tolist()
            column_text = " ".join(column_names).lower()
            
            if 'price' in column_text and 'quantity' in column_text:
                opportunities.append("💰 **NUMERIC RATIOS**: Create mathematical ratios between related variables")
            
            if any(term in column_text for term in ['customer', 'user', 'client']):
                opportunities.append("👥 **ENTITY METRICS**: Develop entity-level aggregation features")
            
            # Statistical transformation opportunities
            skewed_features = []
            for col in numeric_cols[:5]:
                if len(df[col].dropna()) > 0:
                    skewness = df[col].skew()
                    if abs(skewness) > 1:
                        skewed_features.append(col)
            
            if skewed_features:
                opportunities.append(f"📈 **DISTRIBUTION NORMALIZATION**: Apply log/sqrt transformations to {len(skewed_features)} skewed features")
        
        # Transformation enhancement opportunities
        if transformations:
            applied_transforms = list(transformations.keys())
            
            # Check for missing common transformations
            common_transforms = ['scaling', 'encoding', 'binning', 'interaction']
            missing_transforms = [t for t in common_transforms if not any(t in applied.lower() for applied in applied_transforms)]
            
            if missing_transforms:
                opportunities.append(f"🔧 **ADDITIONAL TRANSFORMATIONS**: Consider {', '.join(missing_transforms)} for enhanced feature engineering")
        
        # Feature selection opportunities
        opportunities.append("🎯 **FEATURE SELECTION**: Implement feature importance analysis to optimize feature set")
        opportunities.append("📊 **VALIDATION**: Establish feature performance monitoring for production systems")
        
        return opportunities
