"""
Advanced Anomaly Detection Node Prompt Generator
"""

import numpy as np
import pandas as pd

class AnomalyDetectionPrompt:
    """Generate sophisticated prompts for anomaly detection nodes"""
    
    @staticmethod
    def generate_prompt(data: dict, node_id: str, context: dict = None) -> str:
        """Generate advanced anomaly detection analysis prompt"""
        
        # Handle both old and new data formats
        anomalies = data.get('anomalies', [])
        anomaly_results = data.get('anomaly_results', {})
        detection_method = data.get('method', data.get('detection_method', 'statistical'))
        anomaly_scores = data.get('anomaly_scores', [])
        df = data.get('dataframe')
        detection_stats = data.get('detection_stats', {})
        thresholds = data.get('thresholds', {})
        
        # New format: extract detailed anomaly information from anomaly_results
        if anomaly_results and isinstance(anomaly_results, dict):
            # Extract comprehensive anomaly data
            all_anomalies = []
            all_thresholds = {}
            total_anomalies = data.get('total_anomalies', 0)
            
            for column, column_data in anomaly_results.items():
                if isinstance(column_data, dict):
                    # Handle both univariate and multivariate formats
                    anomaly_count = column_data.get('anomaly_count', column_data.get('anomalies_detected', 0))
                    anomaly_indices = column_data.get('anomaly_indices', [])
                    
                    # Handle different threshold formats
                    threshold_lower = column_data.get('threshold_lower')
                    threshold_upper = column_data.get('threshold_upper')
                    threshold = column_data.get('threshold')
                    
                    method = column_data.get('method', detection_method)
                    
                    # Store anomaly information
                    all_anomalies.extend(anomaly_indices)
                    all_thresholds[column] = {
                        'lower': threshold_lower,
                        'upper': threshold_upper,
                        'threshold': threshold,  # Add single threshold for multivariate
                        'count': anomaly_count,
                        'method': method,
                        'percentage': column_data.get('anomaly_percentage', 0)
                    }
            
            # Use the extracted data
            anomalies = all_anomalies if all_anomalies else anomalies
            thresholds = all_thresholds if all_thresholds else thresholds
            
        # Enhanced validation - check for meaningful data
        if not anomalies and not anomaly_results and df is None:
            return "❌ **CRITICAL ERROR**: No anomaly detection results or dataframe available"
        
        # Analyze anomaly detection results with enhanced data
        anomaly_analysis = AnomalyDetectionPrompt._analyze_anomalies(anomalies, anomaly_scores, df, anomaly_results)
        method_assessment = AnomalyDetectionPrompt._assess_detection_method(detection_method, detection_stats)
        data_impact = AnomalyDetectionPrompt._assess_data_impact(anomalies, df, detection_method, anomaly_results)
        risk_assessment = AnomalyDetectionPrompt._assess_risk_levels(anomalies, anomaly_scores, thresholds, anomaly_results)
        action_recommendations = AnomalyDetectionPrompt._generate_action_recommendations(anomalies, detection_method, anomaly_results)
        
        # Enhanced anomaly summary with detailed results
        if anomaly_results:
            # Calculate total anomalies correctly for both formats
            anomaly_count = data.get('total_anomalies', 0)
            if anomaly_count == 0:  # If not provided, calculate from results
                anomaly_count = sum(
                    col_data.get('anomaly_count', col_data.get('anomalies_detected', 0)) 
                    for col_data in anomaly_results.values() 
                    if isinstance(col_data, dict)
                )
            
            analyzed_columns = len(anomaly_results) if isinstance(anomaly_results, dict) else 1
            detection_method = data.get('detection_method', detection_method)
        else:
            anomaly_count = len(anomalies) if isinstance(anomalies, list) else "Multiple" if anomalies else 0
            analyzed_columns = 1
        
        # Handle dataframe - could be a pandas DataFrame or dict summary
        if df is not None and hasattr(df, 'shape'):
            total_records = df.shape[0]
        elif isinstance(df, dict) and 'shape' in df:
            total_records = df['shape'][0] if isinstance(df['shape'], (list, tuple)) else df.get('total_records', "Unknown")
        else:
            # Try to get from dataset_info or other sources
            dataset_info = data.get('dataset_info', {})
            if isinstance(dataset_info, dict):
                total_records = dataset_info.get('total_rows', dataset_info.get('rows', "Unknown"))
            else:
                total_records = data.get('total_data_points', data.get('summary_statistics', {}).get('total_data_points', "Unknown"))
        
        # Calculate anomaly rate safely
        if data.get('overall_anomaly_rate') is not None:
            anomaly_rate = data.get('overall_anomaly_rate')
        elif isinstance(total_records, int) and isinstance(anomaly_count, int) and total_records > 0:
            anomaly_rate = (anomaly_count / total_records) * 100
        else:
            anomaly_rate = 0
        
        # Build comprehensive anomaly details section
        anomaly_details_section = ""
        if anomaly_results and isinstance(anomaly_results, dict):
            anomaly_details_section = "\n### **📊 DETAILED ANOMALY BREAKDOWN BY COLUMN**\n"
            for column, col_data in anomaly_results.items():
                if isinstance(col_data, dict):
                    # Handle both univariate and multivariate formats
                    anomaly_count_col = col_data.get('anomaly_count', col_data.get('anomalies_detected', 0))
                    anomaly_percentage_col = col_data.get('anomaly_percentage', 0)
                    
                    # Handle different threshold formats
                    threshold_lower = col_data.get('threshold_lower', 'N/A')
                    threshold_upper = col_data.get('threshold_upper', 'N/A')
                    threshold = col_data.get('threshold')
                    
                    if threshold is not None:
                        threshold_display = f"Threshold: {threshold}"
                    else:
                        threshold_display = f"Range: {threshold_lower} to {threshold_upper}"
                    
                    anomaly_details_section += f"""
• **{column}**:
  - Anomalies Detected: {anomaly_count_col}
  - Anomaly Rate: {anomaly_percentage_col:.3f}%
  - {threshold_display}
  - Detection Method: {col_data.get('method', 'Standard')}
  - Sample Anomaly Indices: {str(col_data.get('anomaly_indices', [])[:5])}
"""
        
        # Generate priority levels and categories for comprehensive analysis
        high_priority_columns = []
        medium_priority_columns = []
        standard_columns = []
        
        if anomaly_results and isinstance(anomaly_results, dict):
            for column, col_data in anomaly_results.items():
                if isinstance(col_data, dict):
                    rate = col_data.get('anomaly_percentage', 0)
                    count = col_data.get('anomaly_count', col_data.get('anomalies_detected', 0))
                    if rate > 15:
                        high_priority_columns.append((column, count, rate))
                    elif rate > 8:
                        medium_priority_columns.append((column, count, rate))
                    else:
                        standard_columns.append((column, count, rate))
        
        # Sort by anomaly rate
        high_priority_columns.sort(key=lambda x: x[2], reverse=True)
        medium_priority_columns.sort(key=lambda x: x[2], reverse=True)
        standard_columns.sort(key=lambda x: x[2], reverse=True)
        
        # Determine analysis type and build sections
        node_type = "MULTIVARIATE" if analyzed_columns > 1 and detection_method != "statistical" else "UNIVARIATE"
        
        prompt = f"""
## 🔍 **COMPREHENSIVE {node_type} ANOMALY DETECTION ANALYSIS**

### **📋 ANALYSIS INSTRUCTIONS**
**CRITICAL**: This analysis must report EXACT FINDINGS with specific numbers. Do not generalize or paraphrase the data. Report the precise anomaly counts, percentages, and column names as provided in the detailed breakdown below.

### **🎯 DETECTION OVERVIEW**
- **Node ID:** {node_id}
- **Detection Method:** {detection_method.replace('_', ' ').title()}
- **Total Anomalies:** {anomaly_count} out of {total_records} records
- **Overall Anomaly Rate:** {anomaly_rate:.2f}%
- **Columns Analyzed:** {analyzed_columns}
- **Analysis Duration:** Standard Processing
- **Detection Confidence:** {"High" if anomaly_scores else "Standard"}

{anomaly_details_section}

### **🚨 CRITICAL FINDINGS & PRIORITY ASSESSMENT**

#### **🔥 HIGHEST PRIORITY - EXTREME ANOMALY RATES:**
{chr(10).join([f"{i+1}. **{col}**: {count} anomalies ({rate:.2f}% rate) - 🚨 **CRITICAL ISSUE**" for i, (col, count, rate) in enumerate(high_priority_columns[:3])]) if high_priority_columns else "✅ No critical anomaly rates detected"}

#### **⚠️ HIGH PRIORITY - ELEVATED ANOMALY RATES:**
{chr(10).join([f"{i+1}. **{col}**: {count} anomalies ({rate:.2f}% rate) - ⚠️ **REQUIRES ATTENTION**" for i, (col, count, rate) in enumerate(medium_priority_columns[:5])]) if medium_priority_columns else "✅ No elevated anomaly rates detected"}

#### **📊 STANDARD MONITORING - NORMAL VARIANCE:**
{chr(10).join([f"• **{col}**: {count} anomalies ({rate:.2f}%)" for col, count, rate in standard_columns[:5]]) if standard_columns else "• No standard variance anomalies detected"}

### **🎯 REQUIRED RESPONSE FORMAT**
Your response must include:
1. EXACT count of anomalies found in each column (e.g., "CO(GT): 2063 anomalies")
2. EXACT percentage rates as provided (e.g., "22.03% anomaly rate")
3. Total multivariate anomaly count: {anomaly_count}
4. Relationships between columns and their anomaly patterns

**Example format**: "Analysis found {anomaly_count} total anomalies across {analyzed_columns} variables."

### **🔍 DETAILED PATTERN ANALYSIS**
{chr(10).join(anomaly_analysis) if anomaly_analysis else "⚠️ Pattern analysis not available"}

### **🎯 DETECTION METHOD ASSESSMENT**
{chr(10).join(method_assessment) if method_assessment else "⚠️ Method assessment not available"}

### **📊 DATA IMPACT ANALYSIS**
{chr(10).join(data_impact) if data_impact else "⚠️ Data impact assessment not available"}

### **⚡ RISK LEVEL ASSESSMENT**
{chr(10).join(risk_assessment) if risk_assessment else "⚠️ Risk assessment not available"}

### **🎯 PRIORITIZED ACTION PLAN**

#### **🚨 IMMEDIATE ACTIONS (Critical Priority):**
{chr(10).join([f"1. **Investigate {col} immediately** - {rate:.1f}% anomaly rate requires urgent attention" for col, count, rate in high_priority_columns[:2]]) if high_priority_columns else "✅ No immediate critical actions required"}

#### **⚠️ SHORT-TERM ACTIONS (High Priority):**
{chr(10).join([f"• Review {col} data quality ({rate:.1f}% anomalies)" for col, count, rate in medium_priority_columns[:3]]) if medium_priority_columns else "✅ No short-term actions required"}

#### **📊 LONG-TERM MONITORING:**
• Implement continuous monitoring for all variables
• Establish automated quality alerts for rates >5%
• Create correlation-based validation rules

### **🔬 ROOT CAUSE ANALYSIS**

#### **Primary Issues Identified:**
• **Data Quality Concerns:** {"High anomaly rates suggest systematic issues" if anomaly_rate > 15 else "Anomaly rates within acceptable ranges"}
• **Sensor/Collection Issues:** {"Multiple variables affected - check data collection processes" if len(high_priority_columns + medium_priority_columns) > 3 else "Isolated anomalies suggest specific variable issues"}
• **Temporal Patterns:** {"Time-based analysis recommended for clustered anomalies" if anomaly_count > 100 else "Standard temporal monitoring sufficient"}

#### **Business Impact Assessment:**
• **Risk Level:** {"🚨 CRITICAL" if anomaly_rate > 20 else "⚠️ HIGH" if anomaly_rate > 10 else "📊 MODERATE"}
• **Data Reliability:** {"Compromised - immediate action required" if anomaly_rate > 20 else "Acceptable with monitoring" if anomaly_rate > 5 else "Good"}
• **Operational Impact:** {"System-wide issues detected" if len(high_priority_columns) > 2 else "Localized issues identified" if medium_priority_columns else "Normal operations"}

### **📈 SUCCESS METRICS & MONITORING**
• **Target:** Reduce anomaly rates to <5% for all critical variables
• **Timeline:** {"30-day improvement plan" if high_priority_columns else "90-day monitoring cycle"}
• **Validation:** Cross-reference with historical baselines and external benchmarks
• **Review Frequency:** {"Weekly progress reviews" if anomaly_rate > 15 else "Monthly monitoring"}

### **💡 TECHNICAL RECOMMENDATIONS**

#### **Detection Enhancement:**
{chr(10).join(action_recommendations) if action_recommendations else "⚠️ Technical recommendations not available"}

#### **Quality Assurance:**
• Implement real-time anomaly detection dashboards
• Create automated alerts for anomaly rate thresholds
• Establish data validation pipelines
• Monitor correlation patterns between variables

### **📊 DETECTION METADATA**
• **Total Anomalies Found:** {anomaly_count}
• **Data Points Analyzed:** {total_records}
• **High-Confidence Detections:** {"Available" if anomaly_scores else "Standard Detection"}
• **Threshold Configuration:** {"Custom" if thresholds else "Default"}
• **Statistical Validation:** {"Available" if detection_stats else "Basic"}
• **Multi-column Analysis:** {"Yes" if analyzed_columns > 1 else "Single Column"}

---

**🎯 EXECUTIVE SUMMARY:** This {node_type.lower()} anomaly detection analysis {"reveals critical data quality issues requiring immediate attention" if high_priority_columns else "shows acceptable data quality with standard monitoring recommended"}. {"Focus on " + ", ".join([col for col, _, _ in high_priority_columns[:2]]) + " for immediate investigation." if high_priority_columns else "Continue current monitoring protocols with periodic reviews."}
"""
        
        return prompt.strip()
    
    @staticmethod
    def _analyze_anomalies(anomalies, anomaly_scores, df, anomaly_results=None) -> list:
        """Analyze detected anomalies for patterns and insights"""
        analysis = []
        
        # Handle enhanced anomaly results format
        if anomaly_results and isinstance(anomaly_results, dict):
            total_anomalies = sum(
                col_data.get('anomaly_count', col_data.get('anomalies_detected', 0)) 
                for col_data in anomaly_results.values() 
                if isinstance(col_data, dict)
            )
            
            if total_anomalies == 0:
                analysis.append("✅ **COMPREHENSIVE ANALYSIS RESULT**: All variables within normal statistical parameters")
                analysis.append("📊 **DATA QUALITY STATUS**: Excellent - no significant deviations detected")
                return analysis
            
            # Column-wise anomaly analysis with detailed statistics
            analysis.append(f"📊 **COMPREHENSIVE MULTIVARIATE ANALYSIS**: {len(anomaly_results)} variables analyzed with {total_anomalies} total anomalies")
            
            # Categorize columns by severity
            critical_columns = []
            high_priority_columns = []
            standard_columns = []
            
            for column, col_data in anomaly_results.items():
                if isinstance(col_data, dict):
                    rate = col_data.get('anomaly_percentage', 0)
                    count = col_data.get('anomaly_count', col_data.get('anomalies_detected', 0))
                    if rate > 20:
                        critical_columns.append((column, count, rate))
                    elif rate > 10:
                        high_priority_columns.append((column, count, rate))
                    else:
                        standard_columns.append((column, count, rate))
            
            # Report by severity
            if critical_columns:
                analysis.append("🚨 **CRITICAL VARIABLES** (>20% anomaly rate):")
                for column, count, rate in sorted(critical_columns, key=lambda x: x[2], reverse=True):
                    analysis.append(f"   • **{column}**: {count} anomalies ({rate:.2f}%) - **IMMEDIATE INVESTIGATION REQUIRED**")
            
            if high_priority_columns:
                analysis.append("⚠️ **HIGH PRIORITY VARIABLES** (10-20% anomaly rate):")
                for column, count, rate in sorted(high_priority_columns, key=lambda x: x[2], reverse=True):
                    analysis.append(f"   • **{column}**: {count} anomalies ({rate:.2f}%) - **REVIEW RECOMMENDED**")
            
            if standard_columns and len(standard_columns) > 0:
                analysis.append("📊 **STANDARD MONITORING VARIABLES** (<10% anomaly rate):")
                for column, count, rate in sorted(standard_columns, key=lambda x: x[2], reverse=True)[:5]:
                    analysis.append(f"   • **{column}**: {count} anomalies ({rate:.2f}%) - Normal variance")
            
            # Statistical significance assessment
            if critical_columns or high_priority_columns:
                analysis.append("⚡ **STATISTICAL SIGNIFICANCE**: Multiple variables show elevated anomaly rates indicating potential systemic issues")
                analysis.append("🔍 **CORRELATION ANALYSIS**: Investigate relationships between high-anomaly variables")
            
            return analysis
        
        # Fallback to original analysis
        if not anomalies:
            analysis.append("✅ **NO ANOMALIES DETECTED**: System operating within normal parameters")
            return analysis
        
        anomaly_count = len(anomalies) if isinstance(anomalies, list) else 1
        
        # Anomaly frequency analysis
        if df is not None:
            # Handle both DataFrame and dict representations
            if hasattr(df, 'shape'):
                total_records = df.shape[0]
            elif isinstance(df, dict) and 'shape' in df:
                total_records = df['shape'][0] if isinstance(df['shape'], (list, tuple)) else None
            else:
                total_records = None
                
            if total_records and total_records > 0:
                anomaly_rate = (anomaly_count / total_records) * 100
                
                if anomaly_rate > 10:
                    analysis.append(f"🚨 **HIGH ANOMALY RATE**: {anomaly_rate:.1f}% - Systemic issues require investigation")
                elif anomaly_rate > 5:
                    analysis.append(f"⚠️ **ELEVATED ANOMALY RATE**: {anomaly_rate:.1f}% - Process review recommended")
                elif anomaly_rate > 1:
                    analysis.append(f"📊 **MODERATE ANOMALY RATE**: {anomaly_rate:.1f}% - Normal operational variance")
                else:
                    analysis.append(f"✅ **LOW ANOMALY RATE**: {anomaly_rate:.1f}% - Excellent system stability")
        
        return analysis
    
    @staticmethod
    def _assess_detection_method(method: str, detection_stats: dict) -> list:
        """Assess the effectiveness of the detection method"""
        assessment = []
        
        # Method-specific analysis
        method_insights = {
            'statistical': [
                "📊 **STATISTICAL METHODS**: Reliable for normally distributed data",
                "✅ **STRENGTHS**: Interpretable results with clear statistical thresholds",
                "⚠️ **CONSIDERATIONS**: Assumes normal distribution and linear relationships"
            ],
            'isolation_forest': [
                "🌲 **ISOLATION FOREST**: Effective for high-dimensional data and complex anomaly patterns",
                "🎯 **STRENGTHS**: Handles non-linear patterns and doesn't require labeled data",
                "⚠️ **CONSIDERATIONS**: May struggle with very sparse anomalies"
            ]
        }
        
        method_key = method.lower().replace(' ', '_')
        if method_key in method_insights:
            assessment.extend(method_insights[method_key])
        else:
            assessment.append(f"🔧 **{method.upper()}**: Advanced anomaly detection method applied")
        
        return assessment
    
    @staticmethod
    def _assess_data_impact(anomalies, df, method: str, anomaly_results=None) -> list:
        """Assess the data impact of detected anomalies"""
        impact_assessment = []
        
        # Check for anomalies in enhanced results first, then fallback to original
        total_anomalies = 0
        if anomaly_results and isinstance(anomaly_results, dict):
            total_anomalies = sum(
                col_data.get('anomaly_count', col_data.get('anomalies_detected', 0)) 
                for col_data in anomaly_results.values() 
                if isinstance(col_data, dict)
            )
        elif anomalies:
            total_anomalies = len(anomalies) if isinstance(anomalies, list) else 1
            
        if total_anomalies == 0:
            impact_assessment.append("✅ **POSITIVE DATA QUALITY**: No anomalies detected - data consistent with expected patterns")
            return impact_assessment
        
        impact_assessment.append(f"📊 **DATA IMPACT SUMMARY**: {total_anomalies} anomalies detected affecting data quality")
        
        return impact_assessment
    
    @staticmethod
    def _assess_risk_levels(anomalies, anomaly_scores, thresholds: dict, anomaly_results=None) -> list:
        """Assess risk levels of detected anomalies"""
        risk_assessment = []
        
        # Check for anomalies in anomaly_results first, then thresholds, then fallback
        total_anomalies = 0
        if anomaly_results and isinstance(anomaly_results, dict):
            total_anomalies = sum(
                col_data.get('anomaly_count', col_data.get('anomalies_detected', 0)) 
                for col_data in anomaly_results.values() 
                if isinstance(col_data, dict)
            )
        elif anomalies:
            total_anomalies = len(anomalies) if isinstance(anomalies, list) else 1
            
        if total_anomalies == 0:
            risk_assessment.append("✅ **MINIMAL RISK**: No anomalies detected - risk levels within acceptable parameters")
            return risk_assessment
        
        risk_assessment.append(f"📊 **RISK ASSESSMENT**: {total_anomalies} anomalies require risk evaluation")
        
        return risk_assessment
    
    @staticmethod
    def _generate_action_recommendations(anomalies, method: str, anomaly_results=None) -> list:
        """Generate specific action recommendations based on anomalies"""
        recommendations = []
        
        # Check for anomalies in enhanced results first, then fallback to original
        total_anomalies = 0
        if anomaly_results and isinstance(anomaly_results, dict):
            total_anomalies = sum(
                col_data.get('anomaly_count', col_data.get('anomalies_detected', 0)) 
                for col_data in anomaly_results.values() 
                if isinstance(col_data, dict)
            )
        elif anomalies:
            total_anomalies = len(anomalies) if isinstance(anomalies, list) else 1
            
        if total_anomalies == 0:
            recommendations.extend([
                "✅ **EXCELLENCE MAINTENANCE**: Current data quality standards are optimal",
                "📊 **CONTINUOUS IMPROVEMENT**: Monitor for emerging patterns and trends"
            ])
            return recommendations
        
        recommendations.append(f"🔧 **ACTION REQUIRED**: {total_anomalies} anomalies need investigation")
        
        return recommendations
