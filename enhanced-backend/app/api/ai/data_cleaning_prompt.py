"""
Advanced Data Cleaning Node Prompt Generator
"""

import pandas as pd
import numpy as np

class DataCleaningPrompt:
    """Generate sophisticated prompts for data cleaning nodes"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, context: dict = None) -> str:
        """Generate advanced data cleaning analysis prompt"""
        
        # Check for DataFrame in multiple possible keys - handle DataFrame ambiguity
        df = None
        if 'dataframe' in data and data['dataframe'] is not None:
            df = data['dataframe']
        elif 'data' in data and data['data'] is not None:
            df = data['data']
        
        cleaning_stats = data.get('cleaning_stats', {}) or data.get('cleaning_summary', {})
        before_stats = data.get('before_cleaning', {})
        after_stats = data.get('after_cleaning', {})
        
        if df is None:
            return "❌ **CRITICAL ERROR**: No dataframe available for cleaning analysis"
        
        # Analyze cleaning impact
        cleaning_impact = DataCleaningPrompt._analyze_cleaning_impact(df, cleaning_stats, before_stats, after_stats)
        quality_improvements = DataCleaningPrompt._assess_quality_improvements(df, cleaning_stats)
        data_integrity_check = DataCleaningPrompt._check_data_integrity(df)
        data_usability = DataCleaningPrompt._assess_data_usability(df, cleaning_stats)
        
        prompt = f"""
🧹 **DATA CLEANING ANALYSIS - Node: {node_id}**

📊 **CLEANING TRANSFORMATION SUMMARY**:
Dataset Processed: {df.shape[0]:,} records × {df.shape[1]} features
Memory Footprint: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
Data Completeness: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%

🔧 **CLEANING OPERATIONS SUMMARY**:
{chr(10).join(cleaning_impact) if cleaning_impact else "⚠️ Limited cleaning operation details available"}

✅ **QUALITY IMPROVEMENTS ACHIEVED**:
{chr(10).join(quality_improvements) if quality_improvements else "⚠️ Quality improvement metrics not available"}

🛡️ **DATA INTEGRITY ASSESSMENT**:
{chr(10).join(data_integrity_check) if data_integrity_check else "⚠️ Data integrity check incomplete"}

� **DATA USABILITY ASSESSMENT**:
{chr(10).join(data_usability) if data_usability else "⚠️ Data usability assessment pending"}

📋 **POST-CLEANING DATA PROFILE**:
• Duplicate Records: {df.duplicated().sum():,}
• Missing Values: {df.isnull().sum().sum():,} ({(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%)
• Data Types: {len(df.dtypes.unique())} distinct types
• Unique Records: {df.drop_duplicates().shape[0]:,} ({(df.drop_duplicates().shape[0] / df.shape[0] * 100):.1f}%)

💡 **DATA QUALITY ANALYSIS REQUIREMENTS**:

1. **CLEANING EFFECTIVENESS ASSESSMENT**: Evaluate the success and completeness of data cleaning operations
2. **DATA READINESS VALIDATION**: Confirm data is ready for analytical use
3. **QUALITY ASSESSMENT**: Identify remaining data quality issues
4. **ANALYTICAL IMPACT**: Assess how cleaning improvements affect data reliability
5. **DATA STRUCTURE EVALUATION**: Evaluate the structure and organization of the cleaned data
6. **DATA VALIDITY**: Ensure cleaned data maintains internal consistency
7. **QUALITY METRICS**: Provide metrics for data quality and completeness
8. **ANOMALY IDENTIFICATION**: Identify unusual data patterns or potential issues

🎯 **ANALYSIS REQUIREMENTS**:
- Quantify SPECIFIC quality improvements achieved
- Assess impact of cleaning operations on data quality
- Identify REMAINING quality issues requiring attention
- Provide data quality metrics
- Evaluate structure and usability of cleaned dataset
- Assess data COMPLETENESS and VALIDITY
- Report statistical properties of cleaned data

⚡ **RESPONSE FOCUS**: Analyze the ACTUAL cleaning results and quality improvements demonstrated in this specific dataset. Provide concrete, measurable assessments of cleaning effectiveness and data quality.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _analyze_cleaning_impact(df, cleaning_stats: dict, before_stats: dict, after_stats: dict) -> list:
        """Analyze the impact of cleaning operations"""
        impacts = []
        
        if cleaning_stats:
            # Record count changes
            if 'records_removed' in cleaning_stats:
                removed = cleaning_stats['records_removed']
                impacts.append(f"📉 **Records Removed**: {removed:,} rows eliminated for quality improvement")
            
            if 'duplicates_removed' in cleaning_stats:
                duplicates = cleaning_stats['duplicates_removed']
                impacts.append(f"🔄 **Duplicates Eliminated**: {duplicates:,} duplicate records removed")
            
            if 'missing_values_handled' in cleaning_stats:
                missing_handled = cleaning_stats['missing_values_handled']
                impacts.append(f"🔧 **Missing Values Addressed**: {missing_handled:,} missing values handled")
            
            # Data type improvements
            if 'data_types_corrected' in cleaning_stats:
                type_corrections = cleaning_stats['data_types_corrected']
                impacts.append(f"🎯 **Data Type Corrections**: {type_corrections} columns optimized")
        
        # Current state analysis
        current_duplicates = df.duplicated().sum()
        current_missing = df.isnull().sum().sum()
        
        if current_duplicates == 0:
            impacts.append("✅ **Zero Duplicates**: Complete duplicate elimination achieved")
        elif current_duplicates < df.shape[0] * 0.01:
            impacts.append(f"✅ **Minimal Duplicates**: Only {current_duplicates} duplicates remain (<1%)")
        else:
            impacts.append(f"⚠️ **Duplicate Alert**: {current_duplicates} duplicates still present - review needed")
        
        missing_percentage = (current_missing / (df.shape[0] * df.shape[1])) * 100
        if missing_percentage < 1:
            impacts.append("✅ **Excellent Completeness**: <1% missing data achieved")
        elif missing_percentage < 5:
            impacts.append(f"✅ **Good Completeness**: {missing_percentage:.1f}% missing data - acceptable level")
        else:
            impacts.append(f"⚠️ **Completeness Alert**: {missing_percentage:.1f}% missing data - further cleaning needed")
        
        return impacts
    
    @staticmethod
    def _assess_quality_improvements(df, cleaning_stats: dict) -> list:
        """Assess specific quality improvements"""
        improvements = []
        
        # Data consistency improvements
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check for reasonable ranges
            for col in numeric_cols[:5]:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outlier_bound_low = q1 - 3 * iqr
                    outlier_bound_high = q3 + 3 * iqr
                    
                    extreme_outliers = ((col_data < outlier_bound_low) | (col_data > outlier_bound_high)).sum()
                    outlier_rate = (extreme_outliers / len(col_data)) * 100
                    
                    if outlier_rate < 1:
                        improvements.append(f"✅ **{col}**: Excellent outlier control (<1% extreme values)")
                    elif outlier_rate < 5:
                        improvements.append(f"✅ **{col}**: Good outlier management ({outlier_rate:.1f}% extreme values)")
                    else:
                        improvements.append(f"⚠️ **{col}**: High outlier rate ({outlier_rate:.1f}%) - review needed")
        
        # Categorical data quality
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:5]:
            unique_count = df[col].nunique()
            total_count = df[col].count()
            
            if total_count > 0:
                cardinality_ratio = unique_count / total_count
                
                if cardinality_ratio > 0.95:
                    improvements.append(f"🔍 **{col}**: High cardinality ({unique_count} unique) - potential ID field")
                elif cardinality_ratio < 0.1:
                    improvements.append(f"🏷️ **{col}**: Low cardinality ({unique_count} categories) - well-structured")
                else:
                    improvements.append(f"📊 **{col}**: Balanced cardinality ({unique_count} categories)")
        
        # Overall data structure improvements
        if df.shape[1] > 0:
            non_null_ratio = df.count().sum() / (df.shape[0] * df.shape[1])
            improvements.append(f"📊 **Overall Completeness**: {non_null_ratio:.1%} of data points are valid")
        
        return improvements
    
    @staticmethod
    def _check_data_integrity(df) -> list:
        """Check data integrity after cleaning"""
        integrity_checks = []
        
        # Basic integrity checks
        if df.shape[0] > 0:
            integrity_checks.append(f"✅ **Dataset Preserved**: {df.shape[0]:,} records maintained")
        else:
            integrity_checks.append("❌ **CRITICAL**: No records remaining after cleaning")
            return integrity_checks
        
        # Check for reasonable data ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                min_val, max_val = col_data.min(), col_data.max()
                
                # Check for suspicious ranges
                if min_val == max_val:
                    integrity_checks.append(f"⚠️ **{col}**: Constant values detected - verify data source")
                elif max_val > 0 and min_val >= 0:
                    range_ratio = max_val / min_val if min_val > 0 else float('inf')
                    if range_ratio > 1000:
                        integrity_checks.append(f"🔍 **{col}**: Wide value range - validate business logic")
                    else:
                        integrity_checks.append(f"✅ **{col}**: Reasonable value range maintained")
        
        # Check for data type consistency
        mixed_type_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric values are stored as strings
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    try:
                        pd.to_numeric(sample_values)
                        mixed_type_cols.append(col)
                    except:
                        pass
        
        if mixed_type_cols:
            integrity_checks.append(f"🔧 **Type Optimization Opportunity**: {len(mixed_type_cols)} columns could be numeric")
        else:
            integrity_checks.append("✅ **Data Type Consistency**: All columns have appropriate data types")
        
        # Check for logical relationships
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 1:
            integrity_checks.append(f"📅 **Temporal Integrity**: {len(date_cols)} date columns available for sequence validation")
        
        return integrity_checks
    
    @staticmethod
    def _assess_data_usability(df, cleaning_stats: dict) -> list:
        """Assess usability of the cleaned data for analysis"""
        usability_assessments = []
        
        # Data reliability improvements
        completeness_ratio = df.count().sum() / (df.shape[0] * df.shape[1])
        if completeness_ratio > 0.95:
            usability_assessments.append("� **High Reliability**: >95% data completeness enables thorough analysis")
        elif completeness_ratio > 0.90:
            usability_assessments.append("� **Good Reliability**: >90% data completeness supports most analyses")
        else:
            usability_assessments.append("⚠️ **Reliability Concern**: Data completeness may impact analysis quality")
        
        # Analytical readiness
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            usability_assessments.append("📊 **Analysis Ready**: Mixed data types support comprehensive analysis")
        elif len(numeric_cols) > len(categorical_cols):
            usability_assessments.append("📈 **Quantitative Focus**: Numeric-heavy dataset ideal for statistical analysis")
        else:
            usability_assessments.append("🏷️ **Categorical Focus**: Text-heavy dataset suitable for categorization and frequency analysis")
        
        # Data efficiency
        duplicate_ratio = df.duplicated().sum() / df.shape[0]
        if duplicate_ratio < 0.01:
            usability_assessments.append("⚡ **Processing Efficiency**: Minimal duplicates reduce computation overhead")
        
        # Data quality for analytics
        if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) < 0.05:
            usability_assessments.append("✅ **Analysis Ready**: Low missing data rate meets quality standards for analysis")
        
        return usability_assessments
