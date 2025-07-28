"""
Advanced Statistical Analysis Node Prompt Generator
"""

import numpy as np
import pandas as pd

class StatisticalAnalysisPrompt:
    """Generate sophisticated prompts for statistical analysis nodes"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, context: dict = None) -> str:
        """Generate advanced statistical analysis prompt"""
        
        # Support both 'statistics' and 'basic_stats' keys for flexibility
        statistics = data.get('statistics', {}) or data.get('basic_stats', {})
        df = data.get('dataframe')
        metadata = data.get('metadata', {})
        correlations = data.get('correlations', {})
        
        if not statistics and df is None:
            return "❌ **CRITICAL ERROR**: No statistical data or dataframe available for analysis"
        
        # Analyze statistical patterns
        statistical_insights = StatisticalAnalysisPrompt._analyze_statistical_patterns(statistics, df)
        correlation_insights = StatisticalAnalysisPrompt._analyze_correlations(correlations, df)
        distribution_insights = StatisticalAnalysisPrompt._analyze_distributions(statistics, df)
        outlier_insights = StatisticalAnalysisPrompt._analyze_outliers(statistics, df)
        data_metrics = StatisticalAnalysisPrompt._extract_data_metrics(statistics, df)
        
        # Generate comprehensive statistical prompt focused on data information and results reporting
        prompt = f"""
📊 **STATISTICAL ANALYSIS - Node: {node_id}**

🔬 **COMPREHENSIVE STATISTICAL ANALYSIS**:

� **RAW STATISTICAL DATA**:
{StatisticalAnalysisPrompt._format_raw_statistics(statistics)}

�📈 **DISTRIBUTION ANALYSIS**:
{chr(10).join(distribution_insights) if distribution_insights else "⚠️ Limited distribution analysis available"}

🔗 **CORRELATION ANALYSIS**:
{chr(10).join(correlation_insights) if correlation_insights else "⚠️ No correlation analysis available"}

⚡ **VARIABILITY PATTERNS**:
{chr(10).join(statistical_insights) if statistical_insights else "⚠️ Limited variability analysis available"}

🎯 **OUTLIER ANALYSIS**:
{chr(10).join(outlier_insights) if outlier_insights else "⚠️ No outlier analysis available"}

📈 **DATA METRICS**:
{chr(10).join(data_metrics) if data_metrics else "⚠️ No specific data metrics detected"}

📊 **STATISTICAL FOUNDATION DATA**:
Total Variables Analyzed: {len(statistics)}
Statistical Tests Available: {"Yes" if statistics else "No"}
Correlation Matrix: {"Available" if correlations else "Not Available"}

💡 **STATISTICAL ANALYSIS REQUIREMENTS**:

1. **PATTERN RECOGNITION**: Identify statistical patterns in the data
2. **DATA INSIGHTS**: Extract patterns from the data
3. **DATA QUALITY ASSESSMENT**: Comprehensive data quality evaluation from statistical perspective
4. **STATISTICAL PROPERTIES**: Describe the statistical properties of the data
5. **CORRELATION ASSESSMENT**: Analyze relationships between variables
6. **DISTRIBUTION ANALYSIS**: Describe the distributions present in the data
7. **OUTLIER IDENTIFICATION**: Identify statistical outliers and unusual patterns
8. **SUMMARY STATISTICS**: Provide statistical summaries of the data

🎯 **ANALYSIS REQUIREMENTS**:
- Provide SPECIFIC insights about the statistical patterns observed
- Include QUANTITATIVE evidence for all claims
- Focus on DATA and STATISTICAL findings
- Identify CAUSAL vs CORRELATIONAL relationships
- Highlight STATISTICAL SIGNIFICANCE of findings
- Report DATA QUALITY issues
- Assess DATA PROPERTIES

⚡ **RESPONSE MUST BE SPECIFIC**: Base analysis on the ACTUAL statistical values and patterns shown above. Provide concrete, data-driven insights rather than generic statistical advice.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _analyze_statistical_patterns(statistics: dict, df) -> list:
        """Analyze statistical patterns for business insights"""
        insights = []
        
        for column, stats in statistics.items():
            if isinstance(stats, dict):
                # Numeric column analysis
                if 'mean' in stats and 'std' in stats:
                    mean_val = stats.get('mean', 0)
                    std_val = stats.get('std', 0)
                    min_val = stats.get('min', 0)
                    max_val = stats.get('max', 0)
                    
                    # Coefficient of variation
                    cv = std_val / abs(mean_val) if mean_val != 0 else 0
                    
                    if cv > 2.0:
                        insights.append(f"🚨 **{column}**: EXTREME VARIABILITY (CV={cv:.2f}) - investigate data source reliability")
                    elif cv > 1.0:
                        insights.append(f"📊 **{column}**: HIGH VARIABILITY (CV={cv:.2f}) - segmentation opportunity")
                    elif cv < 0.1:
                        insights.append(f"🎯 **{column}**: HIGHLY STABLE (CV={cv:.2f}) - reliable baseline metric")
                    
                    # Range analysis
                    if max_val > 0 and min_val >= 0:
                        range_ratio = (max_val - min_val) / mean_val if mean_val > 0 else 0
                        if range_ratio > 10:
                            insights.append(f"🎢 **{column}**: EXTREME RANGE detected - potential business tiers or outliers")
                
                # Categorical analysis
                elif 'unique' in stats:
                    unique_count = stats.get('unique', 0)
                    count = stats.get('count', 1)
                    top_freq = stats.get('freq', 0)
                    
                    dominance = top_freq / count if count > 0 else 0
                    cardinality_ratio = unique_count / count if count > 0 else 0
                    
                    if dominance > 0.8:
                        insights.append(f"⚠️ **{column}**: HIGHLY IMBALANCED - top category dominates {dominance:.1%}")
                    elif cardinality_ratio > 0.9:
                        insights.append(f"🔢 **{column}**: HIGH CARDINALITY ({unique_count} unique) - potential identifier")
                    elif unique_count < 10:
                        insights.append(f"🏷️ **{column}**: LOW CARDINALITY ({unique_count} categories) - grouping ready")
        
        return insights
    
    @staticmethod
    def _analyze_correlations(correlations: dict, df) -> list:
        """Analyze correlation patterns"""
        insights = []
        
        if isinstance(correlations, dict) and correlations:
            strong_correlations = []
            for pair, corr_value in correlations.items():
                if isinstance(corr_value, (int, float)) and abs(corr_value) > 0.7:
                    strength = "VERY STRONG" if abs(corr_value) > 0.9 else "STRONG"
                    direction = "positive" if corr_value > 0 else "negative"
                    strong_correlations.append(f"🔗 {pair}: {strength} {direction} relationship ({corr_value:.3f})")
            
            if strong_correlations:
                insights.extend(strong_correlations[:5])  # Top 5 correlations
                insights.append(f"📊 Total strong correlations: {len(strong_correlations)}")
            else:
                insights.append("🔍 No strong correlations detected - variables operate independently")
        
        elif df is not None:
            # Calculate correlations from dataframe
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
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
                    insights.extend(strong_pairs[:5])
                else:
                    insights.append("🔍 Variables show independent behavior - no multicollinearity concerns")
        
        return insights
    
    @staticmethod
    def _analyze_distributions(statistics: dict, df) -> list:
        """Analyze distribution characteristics"""
        insights = []
        
        for column, stats in statistics.items():
            if isinstance(stats, dict) and 'mean' in stats:
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                min_val = stats.get('min', 0)
                max_val = stats.get('max', 0)
                
                # Check for normal distribution indicators
                q25 = stats.get('25%', 0)
                q50 = stats.get('50%', 0)  # median
                q75 = stats.get('75%', 0)
                
                if q25 and q50 and q75:
                    # Symmetry check
                    left_spread = q50 - q25
                    right_spread = q75 - q50
                    symmetry_ratio = left_spread / right_spread if right_spread > 0 else 0
                    
                    if 0.8 <= symmetry_ratio <= 1.2:
                        insights.append(f"⚖️ **{column}**: SYMMETRIC distribution - suitable for parametric methods")
                    elif symmetry_ratio < 0.8:
                        insights.append(f"📈 **{column}**: RIGHT-SKEWED distribution - consider log transformation")
                    else:
                        insights.append(f"📉 **{column}**: LEFT-SKEWED distribution - investigate data collection")
                
                # Check for potential bi-modal or multi-modal
                if mean_val != q50 and abs(mean_val - q50) > std_val * 0.5:
                    insights.append(f"🎭 **{column}**: Mean≠Median suggests NON-NORMAL distribution")
        
        return insights
    
    @staticmethod
    def _analyze_outliers(statistics: dict, df) -> list:
        """Analyze outlier patterns"""
        insights = []
        
        for column, stats in statistics.items():
            if isinstance(stats, dict) and 'mean' in stats:
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                min_val = stats.get('min', 0)
                max_val = stats.get('max', 0)
                q25 = stats.get('25%', 0)
                q75 = stats.get('75%', 0)
                
                if q25 and q75 and std_val > 0:
                    # IQR method for outlier detection
                    iqr = q75 - q25
                    outlier_threshold_low = q25 - 1.5 * iqr
                    outlier_threshold_high = q75 + 1.5 * iqr
                    
                    has_low_outliers = min_val < outlier_threshold_low
                    has_high_outliers = max_val > outlier_threshold_high
                    
                    if has_low_outliers and has_high_outliers:
                        insights.append(f"⚠️ **{column}**: BILATERAL OUTLIERS detected - requires investigation")
                    elif has_high_outliers:
                        insights.append(f"📈 **{column}**: HIGH-VALUE OUTLIERS - potential premium segments")
                    elif has_low_outliers:
                        insights.append(f"📉 **{column}**: LOW-VALUE OUTLIERS - potential quality issues")
                    
                    # Extreme outlier detection (3 * IQR)
                    extreme_low = q25 - 3 * iqr
                    extreme_high = q75 + 3 * iqr
                    
                    if min_val < extreme_low or max_val > extreme_high:
                        insights.append(f"🚨 **{column}**: EXTREME OUTLIERS - data integrity concern")
        
        return insights
    
    @staticmethod
    def _extract_data_metrics(statistics: dict, df) -> list:
        """Extract data-relevant metrics from statistical analysis"""
        metrics = []
        
        # Look for relevant patterns in column names and values
        for column, stats in statistics.items():
            column_lower = column.lower()
            
            # Financial or numeric metrics
            if any(keyword in column_lower for keyword in ['price', 'revenue', 'cost', 'profit', 'sales']):
                if isinstance(stats, dict) and 'mean' in stats:
                    mean_val = stats.get('mean', 0)
                    std_val = stats.get('std', 0)
                    cv = std_val / abs(mean_val) if mean_val != 0 else 0
                    
                    metrics.append(f"� **{column}**: Numeric metric with {cv:.1%} variability")
                    
                    if cv > 1.0:
                        metrics.append(f"📊 **{column}**: HIGH VARIABILITY - potential data quality issue")
            
            # Performance or rating metrics
            elif any(keyword in column_lower for keyword in ['score', 'rating', 'performance', 'efficiency']):
                if isinstance(stats, dict) and 'mean' in stats:
                    mean_val = stats.get('mean', 0)
                    max_val = stats.get('max', 0)
                    utilization = mean_val / max_val if max_val > 0 else 0
                    
                    metrics.append(f"⭐ **{column}**: Rating metric with {utilization:.1%} of maximum value")
            
            # Volume/Count metrics
            elif any(keyword in column_lower for keyword in ['count', 'volume', 'quantity', 'amount']):
                if isinstance(stats, dict) and 'sum' in stats:
                    total_val = stats.get('sum', 0)
                    metrics.append(f"📊 **{column}**: Volume metric with total of {total_val:,.0f}")
        
        return metrics
    
    @staticmethod
    def _format_raw_statistics(statistics: dict) -> str:
        """Format raw statistical data for inclusion in the prompt"""
        if not statistics:
            return "⚠️ No statistical data available"
        
        formatted_stats = []
        
        for column, stats in statistics.items():
            if isinstance(stats, dict):
                stat_lines = [f"📊 **{column}**:"]
                
                # Format numeric statistics
                if 'mean' in stats:
                    stat_lines.append(f"   • Mean: {stats.get('mean', 'N/A')}")
                if 'std' in stats:
                    stat_lines.append(f"   • Std Dev: {stats.get('std', 'N/A')}")
                if 'min' in stats:
                    stat_lines.append(f"   • Min: {stats.get('min', 'N/A')}")
                if 'max' in stats:
                    stat_lines.append(f"   • Max: {stats.get('max', 'N/A')}")
                if 'count' in stats:
                    stat_lines.append(f"   • Count: {stats.get('count', 'N/A')}")
                if 'median' in stats:
                    stat_lines.append(f"   • Median: {stats.get('median', 'N/A')}")
                if '25%' in stats:
                    stat_lines.append(f"   • Q1 (25%): {stats.get('25%', 'N/A')}")
                if '75%' in stats:
                    stat_lines.append(f"   • Q3 (75%): {stats.get('75%', 'N/A')}")
                
                # Format categorical statistics
                if 'unique' in stats:
                    stat_lines.append(f"   • Unique Values: {stats.get('unique', 'N/A')}")
                if 'top' in stats:
                    stat_lines.append(f"   • Most Frequent: {stats.get('top', 'N/A')}")
                if 'freq' in stats:
                    stat_lines.append(f"   • Top Frequency: {stats.get('freq', 'N/A')}")
                
                formatted_stats.append('\n'.join(stat_lines))
        
        return '\n\n'.join(formatted_stats) if formatted_stats else "⚠️ No detailed statistics available"
