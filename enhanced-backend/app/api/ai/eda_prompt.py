"""
Advanced EDA (Exploratory Data Analysis) Node Prompt Generator
"""

import pandas as pd
import numpy as np

class EDAPrompt:
    """Generate sophisticated prompts for EDA nodes"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, context: dict = None) -> str:
        """Generate advanced EDA analysis prompt"""
        
        eda_results = data.get('eda_results', {})
        statistics = data.get('statistics', {})
        charts = data.get('charts', {})
        df = data.get('dataframe')
        correlations = data.get('correlations', {})
        distributions = data.get('distributions', {})
        
        if not eda_results and not statistics and df is None:
            return "❌ **CRITICAL ERROR**: No EDA results, statistics, or dataframe available for analysis"
        
        # Comprehensive EDA analysis
        data_overview = EDAPrompt._analyze_data_overview(df, eda_results)
        statistical_patterns = EDAPrompt._analyze_statistical_patterns(statistics, df)
        distribution_insights = EDAPrompt._analyze_distributions(distributions, df)
        correlation_insights = EDAPrompt._analyze_correlations(correlations, df)
        data_insights = EDAPrompt._extract_data_insights(df, eda_results, statistics)
        data_quality_assessment = EDAPrompt._assess_data_quality(df, eda_results)
        
        # Chart analysis
        chart_count = len(charts) if isinstance(charts, dict) else 0
        visualization_scope = EDAPrompt._assess_visualization_scope(charts, chart_count)
        
        prompt = f"""
🔍 **EXPLORATORY DATA ANALYSIS - Node: {node_id}**

📊 **DATA OVERVIEW**:
{chr(10).join(data_overview) if data_overview else "⚠️ Data overview not available"}

📈 **STATISTICAL PATTERNS**:
{chr(10).join(statistical_patterns) if statistical_patterns else "⚠️ Statistical pattern analysis not available"}

📉 **DISTRIBUTION ANALYSIS**:
{chr(10).join(distribution_insights) if distribution_insights else "⚠️ Distribution analysis not available"}

🔗 **CORRELATION ANALYSIS**:
{chr(10).join(correlation_insights) if correlation_insights else "⚠️ Correlation analysis not available"}

🎯 **VISUALIZATIONS**:
{chr(10).join(visualization_scope) if visualization_scope else f"📊 Generated {chart_count} exploratory visualizations"}

� **DATA INSIGHTS**:
{chr(10).join(data_insights) if data_insights else "⚠️ Data insight extraction pending"}

✅ **DATA QUALITY ASSESSMENT**:
{chr(10).join(data_quality_assessment) if data_quality_assessment else "⚠️ Data quality assessment not available"}

📋 **EDA SCOPE SUMMARY**:
• Statistical Analysis: {"Complete" if statistics else "Basic"}
• Correlation Analysis: {"Available" if correlations else "Not performed"}
• Distribution Analysis: {"Detailed" if distributions else "Basic"}
• Visualization Count: {chart_count} exploratory charts
• Data Quality Review: {"Comprehensive" if df is not None else "Limited"}

💡 **EDA ANALYSIS REQUIREMENTS**:

1. **PATTERN DISCOVERY**: What statistical patterns and relationships emerge from this exploration?
2. **DATA HYPOTHESIS GENERATION**: What statistical hypotheses can be formed from the EDA findings?
3. **DATA SUMMARY**: What key statistical properties emerge from the exploratory analysis?
4. **ANALYTICAL DIRECTION**: What specific analytical techniques should follow based on discovered patterns?
5. **FEATURE CHARACTERISTICS**: What notable statistical properties are identified in the variables?
6. **STATISTICAL PROPERTIES**: What probability distributions and moments are revealed by the analysis?
7. **DATA QUALITY ASSESSMENT**: What data quality characteristics and limitations are observed?
8. **ANOMALY IDENTIFICATION**: What statistical outliers and unusual patterns are discovered?

🎯 **EDA ANALYSIS FOCUS**:
- Identify SPECIFIC statistical patterns and anomalies discovered through exploration
- Provide technical insights on data distributions and correlations
- Generate statistical HYPOTHESES supported by exploratory evidence
- Recommend further statistical analyses based on EDA findings
- Assess DATA SUITABILITY for advanced analytics
- Identify outliers and anomalies revealed through statistical exploration
- Evaluate STATISTICAL PROPERTIES of discovered patterns and relationships

⚡ **RESPONSE FOCUS**: Analyze the ACTUAL patterns, distributions, correlations, and insights discovered in this specific EDA. Provide concrete, data-driven findings based on the exploratory analysis.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _analyze_data_overview(df, eda_results: dict) -> list:
        """Analyze overall data characteristics"""
        overview = []
        
        if df is not None:
            rows, cols = df.shape
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
            
            # Dataset size assessment
            if rows > 1000000:
                overview.append(f"🏢 **LARGE SCALE**: {rows:,} records - Advanced statistical techniques applicable")
            elif rows > 100000:
                overview.append(f"📊 **SUBSTANTIAL SCALE**: {rows:,} records - Robust statistical analysis possible")
            elif rows > 10000:
                overview.append(f"📈 **MODERATE DATASET**: {rows:,} records - Comprehensive analysis ready")
            elif rows > 1000:
                overview.append(f"📊 **SMALL DATASET**: {rows:,} records - Standard analysis applicable")
            else:
                overview.append(f"📋 **VERY SMALL DATASET**: {rows:,} records - Careful statistical analysis needed")
            
            # Feature richness
            if cols > 50:
                overview.append(f"🎯 **FEATURE-RICH**: {cols} variables - Dimensionality reduction may be beneficial")
            elif cols > 20:
                overview.append(f"📊 **WELL-FEATURED**: {cols} variables - Comprehensive feature set")
            elif cols > 10:
                overview.append(f"📈 **BALANCED FEATURES**: {cols} variables - Good analytical scope")
            else:
                overview.append(f"📋 **FOCUSED FEATURES**: {cols} variables - Targeted analysis")
            
            # Memory and computational considerations
            if memory_usage > 500:
                overview.append(f"💾 **MEMORY INTENSIVE**: {memory_usage:.1f} MB - Optimize for performance")
            elif memory_usage > 100:
                overview.append(f"💾 **SUBSTANTIAL SIZE**: {memory_usage:.1f} MB - Standard processing")
            else:
                overview.append(f"💾 **EFFICIENT SIZE**: {memory_usage:.1f} MB - Fast processing")
            
            # Data type diversity
            dtypes = df.dtypes.value_counts()
            numeric_types = sum(count for dtype, count in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype))
            text_types = sum(count for dtype, count in dtypes.items() if 'object' in str(dtype))
            
            if numeric_types > text_types * 2:
                overview.append("🔢 **QUANTITATIVE FOCUS**: Numeric-heavy dataset - ideal for mathematical modeling")
            elif text_types > numeric_types:
                overview.append("📝 **QUALITATIVE FOCUS**: Text-heavy dataset - suitable for categorical analysis")
            else:
                overview.append("⚖️ **BALANCED DATA TYPES**: Mixed numeric and categorical - comprehensive analysis possible")
        
        # EDA results analysis
        if eda_results:
            if 'summary_stats' in eda_results:
                overview.append("📊 **Summary Statistics**: Comprehensive descriptive statistics available")
            if 'missing_analysis' in eda_results:
                overview.append("🔍 **Missing Data Analysis**: Completeness assessment performed")
            if 'outlier_detection' in eda_results:
                overview.append("⚠️ **Outlier Detection**: Anomaly patterns identified")
        
        return overview
    
    @staticmethod
    def _analyze_statistical_patterns(statistics: dict, df) -> list:
        """Analyze statistical patterns from EDA"""
        patterns = []
        
        if statistics:
            # Variable-specific pattern analysis
            for var_name, stats in statistics.items():
                if isinstance(stats, dict):
                    # Numeric variable patterns
                    if 'mean' in stats and 'std' in stats:
                        mean_val = stats.get('mean', 0)
                        std_val = stats.get('std', 0)
                        
                        # Coefficient of variation analysis
                        cv = std_val / abs(mean_val) if mean_val != 0 else 0
                        
                        if cv > 2.0:
                            patterns.append(f"📊 **{var_name}**: EXTREME VARIABILITY (CV={cv:.2f}) - high statistical dispersion")
                        elif cv > 1.0:
                            patterns.append(f"📈 **{var_name}**: HIGH VARIABILITY (CV={cv:.2f}) - significant standard deviation relative to mean")
                        elif cv < 0.1:
                            patterns.append(f"🎯 **{var_name}**: HIGHLY STABLE (CV={cv:.2f}) - consistent statistical measurements")
                        
                        # Central tendency analysis
                        median_val = stats.get('50%', stats.get('median'))
                        if median_val is not None and abs(mean_val - median_val) > std_val * 0.5:
                            patterns.append(f"⚠️ **{var_name}**: SKEWED DISTRIBUTION - mean≠median suggests non-normal pattern")
                        
                        # Range analysis
                        min_val = stats.get('min', 0)
                        max_val = stats.get('max', 0)
                        if max_val > 0 and min_val >= 0:
                            range_span = max_val - min_val
                            range_ratio = range_span / mean_val if mean_val > 0 else 0
                            if range_ratio > 10:
                                patterns.append(f"🎢 **{var_name}**: WIDE RANGE ({min_val:.1f} to {max_val:.1f}) - potential tiers or outliers")
                    
                    # Categorical variable patterns
                    elif 'unique' in stats:
                        unique_count = stats.get('unique', 0)
                        count = stats.get('count', 1)
                        top_freq = stats.get('freq', 0)
                        
                        cardinality_ratio = unique_count / count if count > 0 else 0
                        dominance_ratio = top_freq / count if count > 0 else 0
                        
                        if cardinality_ratio > 0.9:
                            patterns.append(f"🔍 **{var_name}**: HIGH CARDINALITY ({unique_count} unique) - potential identifier or geographic field")
                        elif cardinality_ratio < 0.05:
                            patterns.append(f"🏷️ **{var_name}**: LOW CARDINALITY ({unique_count} categories) - excellent for grouping")
                        
                        if dominance_ratio > 0.8:
                            patterns.append(f"⚠️ **{var_name}**: SINGLE CATEGORY DOMINANCE ({dominance_ratio:.1%}) - imbalanced distribution")
                        elif 0.2 <= dominance_ratio <= 0.4:
                            patterns.append(f"⚖️ **{var_name}**: BALANCED DISTRIBUTION - good for comparative analysis")
        
        # Cross-variable patterns
        elif df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Quick statistical pattern detection
            for col in numeric_cols[:5]:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    skewness = col_data.skew()
                    kurtosis = col_data.kurtosis()
                    
                    if abs(skewness) > 2:
                        direction = "right" if skewness > 0 else "left"
                        patterns.append(f"📊 **{col}**: HIGHLY {direction.upper()}-SKEWED distribution")
                    
                    if kurtosis > 3:
                        patterns.append(f"🎯 **{col}**: HEAVY-TAILED distribution - outlier investigation needed")
                    elif kurtosis < -1:
                        patterns.append(f"📊 **{col}**: LIGHT-TAILED distribution - uniform characteristics")
        
        return patterns
    
    @staticmethod
    def _analyze_distributions(distributions: dict, df) -> list:
        """Analyze distribution characteristics"""
        dist_insights = []
        
        if distributions:
            for var_name, dist_info in distributions.items():
                if isinstance(dist_info, dict):
                    dist_type = dist_info.get('type', 'unknown')
                    if dist_type == 'normal':
                        dist_insights.append(f"📊 **{var_name}**: NORMAL DISTRIBUTION - parametric methods applicable")
                    elif dist_type == 'skewed':
                        dist_insights.append(f"📈 **{var_name}**: SKEWED DISTRIBUTION - consider transformation")
                    elif dist_type == 'bimodal':
                        dist_insights.append(f"🎭 **{var_name}**: BIMODAL DISTRIBUTION - potential subpopulations")
                    elif dist_type == 'uniform':
                        dist_insights.append(f"📏 **{var_name}**: UNIFORM DISTRIBUTION - random characteristics")
        
        # Infer distributions from dataframe
        elif df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols[:5]:
                col_data = df[col].dropna()
                if len(col_data) > 10:
                    # Basic distribution analysis
                    q25, q50, q75 = col_data.quantile([0.25, 0.5, 0.75])
                    
                    # Symmetry analysis
                    left_spread = q50 - q25
                    right_spread = q75 - q50
                    symmetry_ratio = left_spread / right_spread if right_spread > 0 else 0
                    
                    if 0.8 <= symmetry_ratio <= 1.2:
                        dist_insights.append(f"⚖️ **{col}**: SYMMETRIC DISTRIBUTION - normal characteristics")
                    elif symmetry_ratio < 0.8:
                        dist_insights.append(f"📈 **{col}**: RIGHT-SKEWED DISTRIBUTION - log transformation candidate")
                    else:
                        dist_insights.append(f"📉 **{col}**: LEFT-SKEWED DISTRIBUTION - investigate ceiling effects")
                    
                    # Spread analysis
                    iqr = q75 - q25
                    total_range = col_data.max() - col_data.min()
                    spread_ratio = iqr / total_range if total_range > 0 else 0
                    
                    if spread_ratio > 0.8:
                        dist_insights.append(f"📊 **{col}**: CONCENTRATED DISTRIBUTION - most values in central range")
                    elif spread_ratio < 0.4:
                        dist_insights.append(f"🎢 **{col}**: DISPERSED DISTRIBUTION - wide value spread")
        
        return dist_insights
    
    @staticmethod
    def _analyze_correlations(correlations: dict, df) -> list:
        """Analyze correlation patterns"""
        corr_insights = []
        
        if correlations:
            # Direct correlation analysis
            if isinstance(correlations, dict):
                strong_correlations = []
                moderate_correlations = []
                
                for pair, corr_value in correlations.items():
                    if isinstance(corr_value, (int, float)):
                        abs_corr = abs(corr_value)
                        if abs_corr > 0.8:
                            direction = "↗️ POSITIVE" if corr_value > 0 else "↘️ NEGATIVE"
                            strong_correlations.append(f"{direction} **{pair}**: {corr_value:.3f}")
                        elif abs_corr > 0.5:
                            direction = "↗️ positive" if corr_value > 0 else "↘️ negative"
                            moderate_correlations.append(f"{direction} **{pair}**: {corr_value:.3f}")
                
                if strong_correlations:
                    corr_insights.append(f"🔗 **STRONG CORRELATIONS** ({len(strong_correlations)}):")
                    corr_insights.extend([f"   {corr}" for corr in strong_correlations[:5]])
                
                if moderate_correlations:
                    corr_insights.append(f"📊 **MODERATE CORRELATIONS** ({len(moderate_correlations)}):")
                    corr_insights.extend([f"   {corr}" for corr in moderate_correlations[:3]])
                
                if not strong_correlations and not moderate_correlations:
                    corr_insights.append("🔍 **INDEPENDENT VARIABLES**: No strong correlations detected - variables operate independently")
        
        # Calculate correlations from dataframe
        elif df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                strong_pairs = []
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:
                            corr_val = corr_matrix.loc[col1, col2]
                            if abs(corr_val) > 0.7:
                                direction = "↗️" if corr_val > 0 else "↘️"
                                strong_pairs.append(f"{direction} **{col1}** ↔ **{col2}**: {corr_val:.3f}")
                
                if strong_pairs:
                    corr_insights.append(f"🔗 **STRONG RELATIONSHIPS** ({len(strong_pairs)}):")
                    corr_insights.extend([f"   {pair}" for pair in strong_pairs[:5]])
                    
                    # Business implications
                    if len(strong_pairs) > 5:
                        corr_insights.append("⚠️ **MULTICOLLINEARITY RISK**: Consider dimensionality reduction")
                else:
                    corr_insights.append("🔍 **VARIABLE INDEPENDENCE**: Low correlations - minimal multicollinearity concerns")
        
        return corr_insights
    
    @staticmethod
    def _extract_data_insights(df, eda_results: dict, statistics: dict) -> list:
        """Extract data-relevant insights from EDA"""
        data_insights = []
        
        if df is not None:
            # Data domain detection
            column_names = df.columns.tolist()
            column_text = " ".join(column_names).lower()
            
            # Numerical data insights
            if any(keyword in column_text for keyword in ['price', 'revenue', 'cost', 'profit', 'sales']):
                data_insights.append("💰 **NUMERICAL VARIABLES**: Quantitative variables available for statistical analysis")
                
                # Metric analysis
                metric_cols = [col for col in column_names if any(term in col.lower() for term in ['price', 'revenue', 'cost', 'profit'])]
                if metric_cols:
                    data_insights.append(f"📊 **Quantitative Metrics**: {len(metric_cols)} numerical variables available for statistical analysis")
            
            # Entity-related data
            if any(keyword in column_text for keyword in ['customer', 'user', 'client', 'purchase', 'order']):
                data_insights.append("👥 **ENTITY DATA**: Entity-related variables suitable for grouping and aggregation")
                
                # Behavioral variables
                behavioral_cols = [col for col in column_names if any(term in col.lower() for term in ['frequency', 'amount', 'quantity', 'rating'])]
                if behavioral_cols:
                    data_insights.append(f"📈 **Behavioral Variables**: {len(behavioral_cols)} quantifiable behavioral indicators")
            
            # Process data
            if any(keyword in column_text for keyword in ['process', 'operation', 'efficiency', 'quality', 'time']):
                data_insights.append("⚙️ **PROCESS DATA**: Process-related variables suitable for temporal and sequential analysis")
            
            # Time series data
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            potential_date_cols = [col for col in column_names if any(term in col.lower() for term in ['date', 'time', 'timestamp'])]
            
            if date_cols or potential_date_cols:
                data_insights.append("📅 **TEMPORAL DATA**: Time-series variables enable temporal pattern analysis")
                data_insights.append("🔄 **CYCLICAL PATTERNS**: Data suitable for seasonality and periodicity analysis")
            
            # Data scale implications
            rows = df.shape[0]
            if rows > 100000:
                data_insights.append("🏢 **LARGE DATASET**: Sufficient sample size for complex statistical modeling")
            elif rows > 10000:
                data_insights.append("📊 **ADEQUATE SAMPLE**: Sufficient data for reliable statistical analysis")
            
            # Feature engineering potential
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                data_insights.append("🔧 **TRANSFORMATION POTENTIAL**: Multiple numeric variables suitable for mathematical transformations")
            
            # Categorical analysis potential
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(categorical_cols) >= 2:
                data_insights.append("🎯 **GROUPING POTENTIAL**: Multiple categorical variables enable advanced grouping and stratification")
        
        # EDA results data implications
        if eda_results:
            if 'outliers' in eda_results:
                data_insights.append("⚠️ **OUTLIER PATTERNS**: Statistical anomalies detected requiring further investigation")
            
            if 'patterns' in eda_results:
                data_insights.append("🔍 **STATISTICAL PATTERNS**: EDA reveals underlying data structures and relationships")
        
        return data_insights
    
    @staticmethod
    def _assess_data_quality(df, eda_results: dict) -> list:
        """Assess data quality based on EDA findings"""
        quality_assessment = []
        
        if df is not None:
            rows, cols = df.shape
            
            # Completeness assessment
            missing_percentage = (df.isnull().sum().sum() / (rows * cols)) * 100
            if missing_percentage < 1:
                quality_assessment.append("✅ **EXCELLENT COMPLETENESS**: <1% missing data - analysis ready")
            elif missing_percentage < 5:
                quality_assessment.append(f"✅ **GOOD COMPLETENESS**: {missing_percentage:.1f}% missing - manageable")
            elif missing_percentage < 15:
                quality_assessment.append(f"⚠️ **MODERATE COMPLETENESS**: {missing_percentage:.1f}% missing - imputation needed")
            else:
                quality_assessment.append(f"❌ **POOR COMPLETENESS**: {missing_percentage:.1f}% missing - data collection review required")
            
            # Duplicate assessment
            duplicate_count = df.duplicated().sum()
            duplicate_percentage = (duplicate_count / rows) * 100
            if duplicate_percentage == 0:
                quality_assessment.append("✅ **NO DUPLICATES**: Unique records maintained")
            elif duplicate_percentage < 1:
                quality_assessment.append(f"✅ **MINIMAL DUPLICATES**: {duplicate_percentage:.1f}% - acceptable level")
            else:
                quality_assessment.append(f"⚠️ **DUPLICATE CONCERN**: {duplicate_percentage:.1f}% duplicates need review")
            
            # Consistency assessment
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            consistency_issues = 0
            
            for col in numeric_cols[:5]:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Check for extreme outliers
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    extreme_outliers = ((col_data < (q1 - 3 * iqr)) | (col_data > (q3 + 3 * iqr))).sum()
                    
                    if extreme_outliers > len(col_data) * 0.05:  # More than 5% extreme outliers
                        consistency_issues += 1
            
            if consistency_issues == 0:
                quality_assessment.append("✅ **CONSISTENT DATA**: No extreme outlier patterns detected")
            elif consistency_issues <= len(numeric_cols) * 0.3:
                quality_assessment.append(f"⚠️ **MINOR INCONSISTENCIES**: {consistency_issues} variables have outlier patterns")
            else:
                quality_assessment.append(f"❌ **CONSISTENCY ISSUES**: {consistency_issues} variables need outlier investigation")
            
            # Overall quality score
            completeness_score = max(0, 100 - missing_percentage)
            duplicate_score = max(0, 100 - duplicate_percentage)
            consistency_score = max(0, 100 - (consistency_issues / max(1, len(numeric_cols)) * 100))
            
            overall_quality = (completeness_score + duplicate_score + consistency_score) / 3
            
            if overall_quality >= 90:
                quality_assessment.append(f"🏆 **OVERALL QUALITY**: {overall_quality:.0f}/100 - EXCELLENT - Analysis ready")
            elif overall_quality >= 75:
                quality_assessment.append(f"✅ **OVERALL QUALITY**: {overall_quality:.0f}/100 - GOOD - Minor preprocessing needed")
            elif overall_quality >= 60:
                quality_assessment.append(f"⚠️ **OVERALL QUALITY**: {overall_quality:.0f}/100 - MODERATE - Cleaning recommended")
            else:
                quality_assessment.append(f"❌ **OVERALL QUALITY**: {overall_quality:.0f}/100 - POOR - Comprehensive cleaning required")
        
        return quality_assessment
    
    @staticmethod
    def _assess_visualization_scope(charts: dict, chart_count: int) -> list:
        """Assess the scope and effectiveness of generated visualizations"""
        scope_assessment = []
        
        if chart_count == 0:
            scope_assessment.append("📊 **No Visualizations**: Standard EDA without custom charts")
            return scope_assessment
        
        # Portfolio size assessment
        if chart_count >= 10:
            scope_assessment.append(f"📊 **COMPREHENSIVE PORTFOLIO**: {chart_count} charts - Complete visual exploration")
        elif chart_count >= 5:
            scope_assessment.append(f"📊 **SUBSTANTIAL PORTFOLIO**: {chart_count} charts - Thorough visual analysis")
        elif chart_count >= 3:
            scope_assessment.append(f"📊 **FOCUSED PORTFOLIO**: {chart_count} charts - Key visual insights")
        else:
            scope_assessment.append(f"📊 **BASIC PORTFOLIO**: {chart_count} charts - Essential visualization")
        
        # Chart type analysis
        if isinstance(charts, dict):
            chart_names = list(charts.keys())
            
            # Analytical coverage
            analysis_types = {
                'Distribution': ['histogram', 'density', 'distribution', 'box'],
                'Relationship': ['scatter', 'correlation', 'regression'],
                'Comparison': ['bar', 'column', 'compare'],
                'Trend': ['line', 'time', 'trend'],
                'Composition': ['pie', 'stacked', 'area']
            }
            
            covered_types = []
            for analysis_type, keywords in analysis_types.items():
                if any(keyword in chart_name.lower() for chart_name in chart_names for keyword in keywords):
                    covered_types.append(analysis_type)
            
            if len(covered_types) >= 4:
                scope_assessment.append(f"🎨 **MULTI-DIMENSIONAL**: Covers {', '.join(covered_types)} analysis")
            elif len(covered_types) >= 2:
                scope_assessment.append(f"🎨 **FOCUSED ANALYSIS**: Emphasizes {' and '.join(covered_types)}")
            
            # Business readiness
            if chart_count >= 5:
                scope_assessment.append("� **COMPREHENSIVE VISUALIZATION**: Sufficient charts for thorough data interpretation")
            
        return scope_assessment
