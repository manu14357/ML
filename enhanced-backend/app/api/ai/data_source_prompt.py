"""
Advanced Data Source Node Prompt Generator
Focused on data information and results reporting
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

class DataSourcePrompt:
    """Generate prompts for data source nodes with focus on data information and results reporting"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, context: dict = None) -> str:
        """Generate data-focused analysis prompt for data source nodes"""
        
        df = data.get('dataframe')
        source_info = data.get('source_info', {})
        upload_timestamp = data.get('upload_timestamp')
        file_info = data.get('file_info', {})
        
        if df is None:
            # Metadata analysis when dataframe not available
            shape = data.get('shape', 'Unknown')
            columns = data.get('columns', [])
            data_types = data.get('dtypes', {})
            
            return f"""
� **DATA SOURCE ANALYSIS**
**Node ID: {node_id} | Analysis Mode: Metadata**

📊 **DATA STRUCTURE OVERVIEW**:
• Dataset Dimensions: {shape}
• Columns: {len(columns)} columns
• Column Names: {', '.join(columns[:8])}{'...' if len(columns) > 8 else ''}
• Data Types: {len(set(data_types.values())) if data_types else 'Unknown'} distinct data types

🎯 **DATA ANALYSIS REQUIREMENTS**:

Please analyze this data source based on:
1. **Column Structure**: Analyze the column names and their potential meanings
2. **Data Architecture**: Evaluate the dataset structure and organization
3. **Data Types**: Assess the data types and their implications for analysis
4. **Quality Assessment**: Identify potential data quality issues based on metadata
5. **Analysis Potential**: Recommend appropriate analysis techniques for this data

**FOCUS**: Provide insights about the data structure and potential analysis approaches.
"""
        
        # Data-focused analysis
        data_profile = DataSourcePrompt._extract_data_profile(df)
        quality_assessment = DataSourcePrompt._assess_data_quality(df)
        statistical_summary = DataSourcePrompt._generate_statistical_summary(df)
        
        prompt = f"""
� **DATA SOURCE ANALYSIS**
**Node ID: {node_id} | Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**

� **DATA STRUCTURE**:
• Records: {df.shape[0]:,} rows × {df.shape[1]} columns
• Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
• Completeness: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%
• Source: {source_info.get('name', 'Unknown')}

� **DATA PROFILE**:
{chr(10).join(data_profile) if data_profile else "• Data profile information not available"}

🔍 **DATA QUALITY ASSESSMENT**:
{chr(10).join(quality_assessment) if quality_assessment else "• Quality assessment not available"}

📝 **STATISTICAL SUMMARY**:
{chr(10).join(statistical_summary) if statistical_summary else "• Statistical summary not available"}

🎯 **DATA ANALYSIS REQUIREMENTS**:

Please analyze this data source focusing on:

1. **DATA STRUCTURE ANALYSIS**:
   - Examine dataset dimensions, column types, and data distributions
   - Identify patterns in the data organization and structure
   - Assess relationships between different columns
   - Evaluate data completeness across columns

2. **DATA QUALITY ASSESSMENT**:
   - Identify missing values, outliers, and inconsistencies
   - Assess data integrity and reliability
   - Evaluate data consistency and validity
   - Recommend data cleaning approaches if needed

3. **STATISTICAL PROPERTIES**:
   - Summarize key statistical measures for numeric columns
   - Identify distribution patterns and characteristics
   - Highlight significant statistical findings
   - Note any unusual statistical properties

4. **DATA INSIGHTS**:
   - Identify key patterns and trends in the data
   - Highlight notable characteristics of the dataset
   - Summarize important findings from the data
   - Suggest potential analysis directions

**FOCUS**: Provide factual information about the data structure, quality, and statistical properties without business implications or recommendations.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _extract_data_profile(df) -> list:
        """Extract basic profile of the data structure and content"""
        profile = []
        
        # Column type analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        # Column distribution
        column_distribution = {
            'numeric': len(numeric_columns),
            'categorical': len(categorical_columns), 
            'datetime': len(datetime_columns),
            'total': df.shape[1]
        }
        
        profile.append(f"• Column Types: {column_distribution['numeric']} numeric, {column_distribution['categorical']} categorical, {column_distribution['datetime']} datetime")
        
        # Add distribution of values
        if len(numeric_columns) > 0:
            numeric_stats = df[numeric_columns].describe().transpose()
            profile.append(f"• Numeric Ranges: min={numeric_stats['min'].min():.2f}, max={numeric_stats['max'].max():.2f}")
        
        if len(categorical_columns) > 0:
            avg_categories = np.mean([df[col].nunique() for col in categorical_columns])
            profile.append(f"• Categorical Values: {avg_categories:.1f} unique values on average")
        
        return profile
    
    @staticmethod
    def _assess_data_quality(df) -> list:
        """Assess the quality of the data"""
        quality = []
        
        # Missing values assessment
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            cols_with_missing = missing_values[missing_values > 0]
            quality.append(f"• Missing Values: {missing_values.sum()} missing values in {len(cols_with_missing)} columns")
        else:
            quality.append("• Missing Values: No missing values detected")
        
        # Duplicate records assessment
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        quality.append(f"• Duplicates: {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)")
        
        # Data type consistency
        mixed_type_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if this might be a numeric column with some non-numeric values
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    numeric_count = sum(pd.to_numeric(df[col], errors='coerce').notnull())
                    if numeric_count > 0 and numeric_count < len(df):
                        mixed_type_cols.append(col)
        
        if mixed_type_cols:
            quality.append(f"• Mixed Types: {len(mixed_type_cols)} columns with mixed data types")
        
        return quality
    
    @staticmethod
    def _analyze_business_architecture(df, source_info: dict) -> list:
        """Analyze business architecture and domain intelligence"""
        architecture = []
        
        # Column name intelligence
        columns = df.columns.tolist()
        
        # Detect business domains from column names
        business_indicators = {
            'financial': ['revenue', 'cost', 'price', 'amount', 'value', 'profit', 'loss', 'budget', 'sales'],
            'customer': ['customer', 'client', 'user', 'account', 'contact', 'name', 'email', 'phone'],
            'temporal': ['date', 'time', 'timestamp', 'created', 'updated', 'year', 'month', 'day'],
            'geographic': ['location', 'city', 'state', 'country', 'region', 'address', 'zip', 'postal'],
            'operational': ['status', 'type', 'category', 'department', 'process', 'stage', 'priority'],
            'performance': ['score', 'rating', 'rank', 'metric', 'kpi', 'performance', 'efficiency']
        }
        
        detected_domains = {}
        for domain, keywords in business_indicators.items():
            matching_columns = [col for col in columns if any(keyword.lower() in col.lower() for keyword in keywords)]
            if matching_columns:
                detected_domains[domain] = matching_columns
        
        # Report business domain intelligence
        if detected_domains:
            for domain, cols in detected_domains.items():
                architecture.append(f"🏢 **{domain.title()} Domain**: {len(cols)} features detected - {', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}")
        else:
            architecture.append("🔍 **Generic Architecture**: Column names suggest general-purpose dataset - domain analysis needed")
        
        # Source intelligence
        if source_info:
            file_name = source_info.get('filename', '')
            file_size = source_info.get('size', 0)
            
            if file_name:
                architecture.append(f"📁 **Source Intelligence**: {file_name} ({file_size/1024/1024:.1f} MB) - {DataSourcePrompt._infer_source_type(file_name)}")
        
        # Data scale intelligence
        record_count = df.shape[0]
        if record_count > 1000000:
            architecture.append(f"🏗️ **Enterprise Scale**: {record_count:,} records - big data analytics capabilities required")
        elif record_count > 100000:
            architecture.append(f"🏢 **Corporate Scale**: {record_count:,} records - standard business intelligence tools suitable")
        elif record_count > 10000:
            architecture.append(f"🏬 **Departmental Scale**: {record_count:,} records - ideal for focused business analysis")
        else:
            architecture.append(f"🏪 **Team Scale**: {record_count:,} records - suitable for tactical analysis and prototyping")
        
        return architecture
    
    @staticmethod
    def _evaluate_analytical_readiness(df) -> list:
        """Evaluate readiness for various analytical approaches"""
        readiness = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Statistical analysis readiness
        if len(numeric_cols) >= 2:
            correlations = df[numeric_cols].corr().abs()
            high_correlations = (correlations > 0.7).sum().sum() - len(numeric_cols)  # Exclude diagonal
            readiness.append(f"📊 **Statistical Analysis Ready**: {len(numeric_cols)} numeric features with {high_correlations} strong correlations detected")
        
        # Machine learning readiness
        if len(numeric_cols) >= 3 and len(categorical_cols) >= 1:
            readiness.append(f"🤖 **ML Ready**: Mixed feature types ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical) support comprehensive modeling")
        elif len(numeric_cols) >= 5:
            readiness.append(f"📈 **Regression Ready**: {len(numeric_cols)} numeric features ideal for predictive modeling")
        elif len(categorical_cols) >= 3:
            readiness.append(f"🏷️ **Classification Ready**: {len(categorical_cols)} categorical features support classification tasks")
        
        # Visualization readiness
        if len(numeric_cols) >= 2:
            readiness.append(f"📈 **Visualization Ready**: Multiple numeric features enable rich statistical visualizations")
        
        if len(categorical_cols) >= 1:
            readiness.append(f"📊 **Segmentation Ready**: Categorical features enable business segmentation and grouping analysis")
        
        # Business intelligence readiness
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        if completeness > 0.9:
            readiness.append(f"💼 **BI Ready**: {completeness:.1%} completeness meets enterprise business intelligence standards")
        
        return readiness
    
    @staticmethod
    def _build_value_creation_matrix(df, context: dict) -> list:
        """Build comprehensive value creation intelligence matrix"""
        value_matrix = []
        
        # Immediate value opportunities
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        categorical_features = len(df.select_dtypes(include=['object', 'category']).columns)
        
        # Revenue intelligence opportunities
        if numeric_features >= 3:
            value_matrix.append(f"💰 **Revenue Analytics**: {numeric_features} quantitative features enable KPI tracking, performance analysis, and ROI optimization")
        
        # Operational intelligence opportunities
        if categorical_features >= 2:
            value_matrix.append(f"⚡ **Operational Intelligence**: {categorical_features} categorical features support process optimization and efficiency analysis")
        
        # Strategic intelligence opportunities
        total_data_points = df.shape[0] * df.shape[1]
        if total_data_points > 100000:
            value_matrix.append(f"🎯 **Strategic Intelligence**: {total_data_points:,} data points provide enterprise-scale insights for strategic planning")
        
        # Customer intelligence opportunities
        customer_indicators = [col for col in df.columns if any(term in col.lower() for term in ['customer', 'client', 'user', 'account'])]
        if customer_indicators:
            value_matrix.append(f"👥 **Customer Intelligence**: {len(customer_indicators)} customer-related features enable segmentation and personalization strategies")
        
        # Market intelligence opportunities
        if df.shape[0] > 10000:
            value_matrix.append(f"📈 **Market Intelligence**: Large dataset ({df.shape[0]:,} records) enables market trend analysis and competitive insights")
        
        # Innovation opportunities from context
        if context and context.get('workflow_type') == 'advanced':
            value_matrix.append("🚀 **Innovation Pipeline**: Advanced workflow context indicates high-value analytical initiatives")
        
        return value_matrix
    
    @staticmethod
    def _infer_source_type(filename: str) -> str:
        """Infer source type from filename"""
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ['sales', 'revenue', 'financial']):
            return "Financial/Sales data source"
        elif any(term in filename_lower for term in ['customer', 'client', 'user']):
            return "Customer/CRM data source"
        elif any(term in filename_lower for term in ['inventory', 'product', 'stock']):
            return "Inventory/Product data source"
        elif any(term in filename_lower for term in ['transaction', 'order', 'purchase']):
            return "Transaction data source"
        elif any(term in filename_lower for term in ['employee', 'hr', 'staff']):
            return "Human Resources data source"
        elif any(term in filename_lower for term in ['sensor', 'iot', 'device']):
            return "IoT/Sensor data source"
        else:
            return "General business data source"
