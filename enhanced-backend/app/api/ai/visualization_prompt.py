"""
Advanced Visualization Node Prompt Generator
"""

class VisualizationPrompt:
    """Generate sophisticated prompts for visualization nodes"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, context: dict = None) -> str:
        """Generate advanced visualization analysis prompt"""
        
        charts = data.get('charts', {})
        chart_metadata = data.get('chart_metadata', {})
        df = data.get('dataframe')
        visualization_insights = data.get('insights', {})
        
        if not charts and df is None:
            return "❌ **CRITICAL ERROR**: No visualization data or source dataframe available"
        
        # Analyze visualization portfolio
        chart_analysis = VisualizationPrompt._analyze_chart_portfolio(charts, chart_metadata)
        pattern_insights = VisualizationPrompt._extract_visual_patterns(charts, df)
        data_representation = VisualizationPrompt._assess_data_representation(charts, df)
        visualization_quality = VisualizationPrompt._assess_visualization_quality(charts, df)
        
        # Chart portfolio summary
        chart_count = len(charts) if isinstance(charts, (dict, list)) else 0
        chart_types = VisualizationPrompt._identify_chart_types(charts)
        
        prompt = f"""
📊 **VISUALIZATION ANALYSIS - Node: {node_id}**

🎨 **VISUALIZATION PORTFOLIO OVERVIEW**:
Total Visualizations: {chart_count} charts
Chart Types: {', '.join(chart_types) if chart_types else 'Standard charts'}
Visual Analysis Scope: Data visualization portfolio

📈 **CHART ANALYSIS**:
{chr(10).join(chart_analysis) if chart_analysis else "⚠️ Chart analysis details not available"}

🔍 **VISUAL PATTERN RECOGNITION**:
{chr(10).join(pattern_insights) if pattern_insights else "⚠️ Pattern recognition analysis pending"}

📖 **DATA REPRESENTATION ASSESSMENT**:
{chr(10).join(data_representation) if data_representation else "⚠️ Data representation evaluation not available"}

⚡ **VISUALIZATION QUALITY ASSESSMENT**:
{chr(10).join(visualization_quality) if visualization_quality else "⚠️ Visualization quality assessment pending"}

🎯 **VISUALIZATION METADATA**:
• Interactive Elements: {"Available" if (isinstance(charts, dict) and any("interactive" in str(chart).lower() for chart in charts.values())) else "Standard"}
• Color Schemes: Standard visualization color schemes
• Chart Complexity: {"Multi-dimensional" if chart_count > 5 else "Focused" if chart_count > 0 else "Basic"}
• Export Formats: {"Multiple" if chart_count > 0 else "Standard"}

💡 **VISUALIZATION ANALYSIS REQUIREMENTS**:

1. **PATTERN IDENTIFICATION**: What data patterns do these visualizations reveal?
2. **DATA REPRESENTATION**: How effectively do these charts represent the underlying data?
3. **INFORMATION CLARITY**: How clearly do the visualizations present the information?
4. **STATISTICAL VISUALIZATION**: How well are statistical properties visualized?
5. **TREND IDENTIFICATION**: What temporal or categorical trends are visually evident?
6. **COMPARATIVE VISUALIZATION**: What comparative insights emerge from the charts?
7. **ANOMALY VISUALIZATION**: Do any charts reveal data anomalies or outliers?
8. **DATA DISTRIBUTION VISUALIZATION**: How well do the charts show distributions and variability?

🎯 **VISUALIZATION ANALYSIS REQUIREMENTS**:
- Interpret SPECIFIC visual patterns in the data
- Evaluate how well visualizations represent the underlying data
- Identify DATA PATTERNS highlighted by visual analysis
- Assess INFORMATION CLARITY of visualizations
- Evaluate STATISTICAL REPRESENTATION in the charts
- Identify TRENDS and RELATIONSHIPS shown in visualizations
- Assess COMPLETENESS of the visualization set
- Recommend improvements for data visualization if needed

⚡ **RESPONSE FOCUS**: Analyze the ACTUAL visual patterns, trends, and insights present in these specific charts. Provide concrete interpretations of what the visualizations reveal about the underlying data.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _analyze_chart_portfolio(charts, chart_metadata) -> list:
        """Analyze the portfolio of charts created"""
        analysis = []
        
        if isinstance(charts, dict):
            chart_count = len(charts)
            
            # Portfolio size assessment
            if chart_count >= 10:
                analysis.append(f"📊 **Comprehensive Portfolio**: {chart_count} charts provide exhaustive visual analysis")
            elif chart_count >= 5:
                analysis.append(f"📊 **Substantial Portfolio**: {chart_count} charts offer thorough visual exploration")
            elif chart_count >= 3:
                analysis.append(f"📊 **Focused Portfolio**: {chart_count} charts provide targeted visual insights")
            elif chart_count > 0:
                analysis.append(f"📊 **Essential Portfolio**: {chart_count} charts cover key visual requirements")
            
            # Chart type diversity
            chart_names = list(charts.keys())
            chart_categories = {
                'distribution': ['histogram', 'boxplot', 'violin', 'density'],
                'relationship': ['scatter', 'correlation', 'regression'],
                'comparison': ['bar', 'column', 'comparison'],
                'temporal': ['line', 'time', 'trend', 'series'],
                'composition': ['pie', 'donut', 'stacked', 'area']
            }
            
            identified_categories = []
            for category, keywords in chart_categories.items():
                if any(keyword in chart_name.lower() for chart_name in chart_names for keyword in keywords):
                    identified_categories.append(category)
            
            if len(identified_categories) >= 4:
                analysis.append("🎨 **Multi-Dimensional Analysis**: Charts cover distribution, relationships, comparisons, and trends")
            elif len(identified_categories) >= 2:
                analysis.append(f"🎨 **Focused Analysis**: Charts emphasize {' and '.join(identified_categories)} patterns")
            
            # Specific chart analysis
            for chart_name, chart_data in list(charts.items())[:5]:  # Analyze top 5 charts
                if 'correlation' in chart_name.lower():
                    analysis.append("🔗 **Correlation Matrix**: Relationship patterns visualized for strategic insights")
                elif 'distribution' in chart_name.lower():
                    analysis.append("📈 **Distribution Analysis**: Data spread and frequency patterns revealed")
                elif 'scatter' in chart_name.lower():
                    analysis.append("🎯 **Scatter Analysis**: Variable relationships and clustering patterns identified")
                elif 'time' in chart_name.lower() or 'trend' in chart_name.lower():
                    analysis.append("📅 **Temporal Analysis**: Time-based patterns and trends visualized")
                elif 'box' in chart_name.lower():
                    analysis.append("📦 **Outlier Analysis**: Statistical distribution and anomaly patterns shown")
        
        return analysis
    
    @staticmethod
    def _extract_visual_patterns(charts, df) -> list:
        """Extract business patterns from visual analysis"""
        patterns = []
        
        if df is not None:
            # Infer patterns from data that would be visualized
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Distribution patterns
            if numeric_cols:
                for col in numeric_cols[:3]:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        skewness = col_data.skew()
                        if abs(skewness) > 1:
                            direction = "right" if skewness > 0 else "left"
                            patterns.append(f"📊 **{col}**: {direction.upper()}-skewed distribution pattern - investigate data source")
                        else:
                            patterns.append(f"📊 **{col}**: Normal distribution pattern - suitable for parametric analysis")
                        
                        # Range patterns
                        cv = col_data.std() / abs(col_data.mean()) if col_data.mean() != 0 else 0
                        if cv > 1.5:
                            patterns.append(f"🌊 **{col}**: High variability pattern - segmentation opportunity")
                        elif cv < 0.2:
                            patterns.append(f"🎯 **{col}**: Stable pattern - reliable metric for baseline")
            
            # Categorical patterns
            if categorical_cols:
                for col in categorical_cols[:3]:
                    unique_count = df[col].nunique()
                    total_count = df[col].count()
                    
                    if total_count > 0:
                        cardinality_ratio = unique_count / total_count
                        
                        if cardinality_ratio > 0.9:
                            patterns.append(f"🔍 **{col}**: High cardinality pattern - potential identifier or geographic data")
                        elif cardinality_ratio < 0.1:
                            patterns.append(f"🏷️ **{col}**: Low cardinality pattern - clear segmentation structure")
                        
                        # Dominance patterns
                        top_freq = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                        dominance = top_freq / total_count
                        
                        if dominance > 0.8:
                            patterns.append(f"⚠️ **{col}**: Single category dominance - imbalanced distribution")
                        elif dominance < 0.3:
                            patterns.append(f"⚖️ **{col}**: Balanced distribution - good for comparative analysis")
            
            # Correlation patterns (if charts suggest correlation analysis)
            if isinstance(charts, dict) and any('correlation' in name.lower() for name in charts.keys()):
                if len(numeric_cols) >= 2:
                    patterns.append("🔗 **Multi-variate Relationships**: Correlation patterns reveal variable interdependencies")
                    
                    # Sample correlation analysis
                    sample_corr = df[numeric_cols[:5]].corr() if len(numeric_cols) >= 2 else None
                    if sample_corr is not None:
                        strong_correlations = []
                        for i in range(len(sample_corr.columns)):
                            for j in range(i+1, len(sample_corr.columns)):
                                corr_val = sample_corr.iloc[i, j]
                                if abs(corr_val) > 0.7:
                                    strong_correlations.append(f"{sample_corr.columns[i]}↔{sample_corr.columns[j]}")
                        
                        if strong_correlations:
                            patterns.append(f"🔗 **Strong Correlations**: {len(strong_correlations)} variable pairs show strong relationships")
        
        # Chart-specific pattern analysis
        if isinstance(charts, dict):
            chart_names = list(charts.keys())
            
            if any('trend' in name.lower() for name in chart_names):
                patterns.append("📈 **Temporal Trends**: Time-based patterns indicate business cycles or seasonal effects")
            
            if any('scatter' in name.lower() for name in chart_names):
                patterns.append("🎯 **Clustering Patterns**: Scatter plots reveal natural data groupings and outliers")
            
            if any('box' in name.lower() for name in chart_names):
                patterns.append("📦 **Statistical Patterns**: Box plots highlight distribution characteristics and anomalies")
        
        return patterns
    
    @staticmethod
    def _assess_data_representation(charts, df) -> list:
        """Assess how well visualizations represent the underlying data"""
        representation = []
        
        if isinstance(charts, dict):
            chart_count = len(charts)
            chart_names = list(charts.keys())
            
            # Data coverage assessment
            has_overview = any('overview' in name.lower() or 'summary' in name.lower() for name in chart_names)
            has_details = any('detail' in name.lower() or 'breakdown' in name.lower() for name in chart_names)
            has_trends = any('trend' in name.lower() or 'time' in name.lower() for name in chart_names)
            has_comparisons = any('compare' in name.lower() or 'vs' in name.lower() for name in chart_names)
            
            representation_elements = sum([has_overview, has_details, has_trends, has_comparisons])
            
            if representation_elements >= 3:
                representation.append("📖 **Complete Data Representation**: Overview → Details → Trends → Comparisons flow")
            elif representation_elements >= 2:
                representation.append("📖 **Structured Representation**: Multiple data views present")
            elif chart_count > 1:
                representation.append("📖 **Basic Representation**: Multiple views support data understanding")
            
            # Comprehensive representation
            if chart_count >= 5:
                representation.append("� **Comprehensive Visualization**: Extensive visual representation of the data")
            elif chart_count >= 3:
                representation.append("� **Standard Visualization**: Sufficient detail for data understanding")
            
            # Technical vs summary charts
            technical_charts = sum(1 for name in chart_names if any(term in name.lower() 
                                 for term in ['correlation', 'distribution', 'statistical', 'regression']))
            summary_charts = chart_count - technical_charts
            
            if summary_charts >= technical_charts:
                representation.append("� **Summary-Focused Representation**: Charts emphasize general patterns over technical details")
            else:
                representation.append("🔬 **Technical-Focused Representation**: Charts provide detailed statistical visualization")
        
        # Data representation elements
        if df is not None:
            # Numerical data visualization
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            potential_metrics = [col for col in numeric_cols if any(term in col.lower() 
                            for term in ['total', 'sum', 'mean', 'average', 'median', 'count'])]
            
            if potential_metrics:
                representation.append(f"📊 **Metric Visualization**: {len(potential_metrics)} numeric metrics visualized")
            
            # Categorical visualization
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            segmentation_cols = [col for col in categorical_cols if any(term in col.lower() 
                               for term in ['category', 'segment', 'type', 'class', 'group'])]
            
            if segmentation_cols:
                representation.append(f"🎯 **Categorical Visualization**: {len(segmentation_cols)} grouping variables visualized")
        
        return representation
    
    @staticmethod
    def _assess_visualization_quality(charts, df) -> list:
        """Assess the quality and effectiveness of the visualizations"""
        quality = []
        
        if isinstance(charts, dict):
            chart_names = list(charts.keys())
            
            # Data clarity charts
            clarity_charts = [name for name in chart_names if any(term in name.lower() 
                             for term in ['summary', 'overview', 'main', 'primary', 'key'])]
            
            if clarity_charts:
                quality.append(f"⚡ **Data Clarity**: {len(clarity_charts)} charts focus on clear data presentation")
            
            # Data monitoring
            monitoring_charts = [name for name in chart_names if any(term in name.lower() 
                                for term in ['monitor', 'track', 'trend', 'time', 'series'])]
            
            if monitoring_charts:
                quality.append(f"📈 **Trend Visualization**: {len(monitoring_charts)} charts show data over time")
            
            # Anomaly visualization
            diagnostic_charts = [name for name in chart_names if any(term in name.lower() 
                               for term in ['outlier', 'anomaly', 'issue', 'problem', 'box', 'scatter'])]
            
            if diagnostic_charts:
                quality.append(f"� **Anomaly Visualization**: {len(diagnostic_charts)} charts highlight unusual data points")
            
            # Relationship visualization
            relationship_charts = [name for name in chart_names if any(term in name.lower() 
                                for term in ['correlation', 'relationship', 'scatter', 'regression'])]
            
            if relationship_charts:
                quality.append(f"� **Relationship Visualization**: {len(relationship_charts)} charts show variable relationships")
        
        # Data-driven quality assessment
        if df is not None:
            # Statistical visualization potential
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Distribution visualization
            if len(numeric_cols) > 0:
                quality.append(f"📊 **Distribution Visualization**: {len(numeric_cols)} numeric columns suitable for distribution charts")
            
            # Correlation visualization
            if len(numeric_cols) >= 2:
                quality.append(f"� **Relationship Visualization Potential**: {len(numeric_cols)} variables available for correlation analysis")
            
            # Categorical visualization
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(categorical_cols) > 0:
                quality.append(f"📊 **Categorical Visualization**: {len(categorical_cols)} categorical columns suitable for grouping charts")
        
        # Overall quality assessment
        total_quality_elements = len(quality)
        if total_quality_elements >= 4:
            quality.append("🎯 **HIGH VISUALIZATION QUALITY**: Multiple data aspects effectively visualized")
        elif total_quality_elements >= 2:
            quality.append("🎯 **MODERATE VISUALIZATION QUALITY**: Several data aspects visualized")
        elif total_quality_elements >= 1:
            quality.append("🎯 **BASIC VISUALIZATION QUALITY**: Limited data visualization")
        
        return quality
    
    @staticmethod
    def _identify_chart_types(charts) -> list:
        """Identify types of charts in the portfolio"""
        chart_types = []
        
        if isinstance(charts, dict):
            chart_names = list(charts.keys())
            
            type_mapping = {
                'Distribution Analysis': ['histogram', 'density', 'distribution'],
                'Relationship Analysis': ['scatter', 'correlation', 'regression'],
                'Comparison Analysis': ['bar', 'column', 'comparison'],
                'Trend Analysis': ['line', 'trend', 'time', 'series'],
                'Statistical Analysis': ['box', 'violin', 'statistical'],
                'Composition Analysis': ['pie', 'donut', 'stacked', 'area']
            }
            
            for chart_type, keywords in type_mapping.items():
                if any(keyword in chart_name.lower() for chart_name in chart_names for keyword in keywords):
                    chart_types.append(chart_type)
        
        return chart_types if chart_types else ['Standard Business Charts']
