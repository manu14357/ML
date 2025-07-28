"""
Advanced Anomaly Detection Node Prompt Generator
"""

import numpy as np
import pandas as pd

class AnomalyDetectionPrompt:
    """Generate sophisticated pro"""        
    prompt = f"""
## 🔍 **COMPREHENSIVE {node_type} ANOMALY DETECTION ANALYSIS**

### **📋 ANALYSIS INSTRUCTIONS**
**CRITICAL**: This analysis must report EXACT FINDINGS with specific numbers. Do not generalize or paraphrase the data. Report the precise anomaly counts, percentages, and column names as provided in the detailed breakdown below.

### **🎯 DETECTION OVERVIEW**   prompt = #### **🔥 HIGHEST PRIORITY - EXTREME ANOMALY RATES:**"""
## 🔍 **COMPREHENSIVE {node_type} ANOMALY DETECTION ANALYSIS**ts for anomaly detection nodes"""
    
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
            anomaly_details_section = "\n🔍 **DETAILED ANOMALY BREAKDOWN BY COLUMN**:\n"
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
## � **COMPREHENSIVE {node_type} ANOMALY DETECTION ANALYSIS**

### **🎯 DETECTION OVERVIEW**
- **Node ID:** {node_id}
- **Detection Method:** {detection_method.replace('_', ' ').title()}
- **Total Anomalies:** {anomaly_count:,} out of {total_records:,} records
- **Overall Anomaly Rate:** {anomaly_rate:.2f}%
- **Columns Analyzed:** {analyzed_columns}
- **Analysis Duration:** Standard Processing
- **Detection Confidence:** {"High" if anomaly_scores else "Standard"}

{anomaly_details_section}

### **🚨 CRITICAL FINDINGS & PRIORITY ASSESSMENT**

#### **� HIGHEST PRIORITY - EXTREME ANOMALY RATES:**
{chr(10).join([f"{i+1}. **{col}**: {count:,} anomalies ({rate:.2f}% rate) - 🚨 **CRITICAL ISSUE**" for i, (col, count, rate) in enumerate(high_priority_columns[:3])]) if high_priority_columns else "✅ No critical anomaly rates detected"}

#### **⚠️ HIGH PRIORITY - ELEVATED ANOMALY RATES:**
{chr(10).join([f"{i+1}. **{col}**: {count:,} anomalies ({rate:.2f}% rate) - ⚠️ **REQUIRES ATTENTION**" for i, (col, count, rate) in enumerate(medium_priority_columns[:5])]) if medium_priority_columns else "✅ No elevated anomaly rates detected"}

#### **📊 STANDARD MONITORING - NORMAL VARIANCE:**
{chr(10).join([f"• **{col}**: {count:,} anomalies ({rate:.2f}%)" for col, count, rate in standard_columns[:5]]) if standard_columns else "• No standard variance anomalies detected"}

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
• **Total Anomalies Found:** {anomaly_count:,}
• **Data Points Analyzed:** {total_records:,}
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
            analysis.append(f"📊 **COMPREHENSIVE MULTIVARIATE ANALYSIS**: {len(anomaly_results)} variables analyzed with {total_anomalies:,} total anomalies")
            
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
                    analysis.append(f"   • **{column}**: {count:,} anomalies ({rate:.2f}%) - **IMMEDIATE INVESTIGATION REQUIRED**")
            
            if high_priority_columns:
                analysis.append("⚠️ **HIGH PRIORITY VARIABLES** (10-20% anomaly rate):")
                for column, count, rate in sorted(high_priority_columns, key=lambda x: x[2], reverse=True):
                    analysis.append(f"   • **{column}**: {count:,} anomalies ({rate:.2f}%) - **REVIEW RECOMMENDED**")
            
            if standard_columns and len(standard_columns) > 0:
                analysis.append("📊 **STANDARD MONITORING VARIABLES** (<10% anomaly rate):")
                for column, count, rate in sorted(standard_columns, key=lambda x: x[2], reverse=True)[:5]:
                    analysis.append(f"   • **{column}**: {count:,} anomalies ({rate:.2f}%) - Normal variance")
            
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
        
        # Severity analysis using anomaly scores
        if anomaly_scores and isinstance(anomaly_scores, list):
            scores_array = np.array(anomaly_scores)
            
            if len(scores_array) > 0:
                high_severity = np.sum(scores_array > np.percentile(scores_array, 80))
                medium_severity = np.sum((scores_array > np.percentile(scores_array, 60)) & 
                                       (scores_array <= np.percentile(scores_array, 80)))
                low_severity = len(scores_array) - high_severity - medium_severity
                
                analysis.append(f"📊 **SEVERITY DISTRIBUTION**: {high_severity} high, {medium_severity} medium, {low_severity} low severity")
                
                if high_severity > 0:
                    analysis.append(f"🚨 **CRITICAL ANOMALIES**: {high_severity} high-severity anomalies require immediate attention")
        
        # Pattern analysis for anomaly types
        if isinstance(anomalies, list) and df is not None:
            # Analyze anomaly distribution across features
            anomaly_features = AnomalyDetectionPrompt._analyze_anomaly_features(anomalies, df)
            if anomaly_features:
                analysis.extend(anomaly_features)
        
        # Temporal pattern analysis (if datetime columns exist)
        if df is not None and isinstance(anomalies, list):
            # Handle DataFrame vs dict for datetime column detection
            has_datetime_cols = False
            if hasattr(df, 'select_dtypes'):
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                has_datetime_cols = bool(date_cols)
            elif isinstance(df, dict) and 'dtypes' in df:
                # Check if any columns have datetime types in the summary
                has_datetime_cols = any('datetime' in str(dtype) for dtype in df.get('dtypes', {}).values())
                
            if has_datetime_cols and len(anomalies) > 0:
                analysis.append("📅 **TEMPORAL ANALYSIS**: Time-based anomaly patterns available for trend analysis")
        
        # Clustering analysis of anomalies
        if anomaly_count >= 5:
            analysis.append("🔍 **PATTERN CLUSTERING**: Multiple anomalies enable pattern clustering analysis")
        elif anomaly_count >= 2:
            analysis.append("🔍 **PATTERN COMPARISON**: Anomaly comparison reveals common characteristics")
        else:
            analysis.append("🎯 **ISOLATED ANOMALY**: Single anomaly requires individual investigation")
        
        return analysis
    
    @staticmethod
    def _analyze_anomaly_features(anomalies, df) -> list:
        """Analyze which features contribute most to anomalies"""
        feature_analysis = []
        
        try:
            # Assume anomalies contains indices or records
            if len(anomalies) > 0 and hasattr(df, 'iloc'):
                # Get anomalous records
                anomaly_indices = anomalies if isinstance(anomalies[0], int) else range(len(anomalies))
                
                if len(anomaly_indices) > 0:
                    anomaly_data = df.iloc[list(anomaly_indices)[:100]]  # Limit to first 100 for analysis
                    normal_data = df.drop(anomaly_indices).sample(min(1000, len(df) - len(anomaly_indices)))
                    
                    # Analyze numeric features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    for col in numeric_cols[:5]:  # Analyze top 5 numeric columns
                        if col in anomaly_data.columns and col in normal_data.columns:
                            anom_mean = anomaly_data[col].mean()
                            normal_mean = normal_data[col].mean()
                            
                            if not np.isnan(anom_mean) and not np.isnan(normal_mean) and normal_mean != 0:
                                deviation_ratio = abs(anom_mean - normal_mean) / abs(normal_mean)
                                
                                if deviation_ratio > 1.0:
                                    feature_analysis.append(f"📊 **{col}**: EXTREME DEVIATION - anomalies differ by {deviation_ratio:.1f}x from normal")
                                elif deviation_ratio > 0.5:
                                    feature_analysis.append(f"📈 **{col}**: SIGNIFICANT DEVIATION - anomalies differ by {deviation_ratio:.1%} from normal")
                                elif deviation_ratio > 0.2:
                                    feature_analysis.append(f"📉 **{col}**: MODERATE DEVIATION - anomalies show {deviation_ratio:.1%} difference")
        
        except Exception:
            feature_analysis.append("🔍 **FEATURE ANALYSIS**: Advanced feature contribution analysis available")
        
        return feature_analysis
    
    @staticmethod
    def _assess_detection_method(method: str, detection_stats: dict) -> list:
        """Assess the effectiveness of the detection method"""
        assessment = []
        
        # Method-specific analysis
        method_insights = {
            'isolation_forest': [
                "🌲 **ISOLATION FOREST**: Effective for high-dimensional data and complex anomaly patterns",
                "🎯 **STRENGTHS**: Handles non-linear patterns and doesn't require labeled data",
                "⚠️ **CONSIDERATIONS**: May struggle with very sparse anomalies"
            ],
            'local_outlier_factor': [
                "🎯 **LOCAL OUTLIER FACTOR**: Excellent for density-based anomaly detection",
                "📊 **STRENGTHS**: Identifies local outliers in varying density regions",
                "⚠️ **CONSIDERATIONS**: Sensitive to parameter tuning"
            ],
            'one_class_svm': [
                "🤖 **ONE-CLASS SVM**: Robust method for novelty detection",
                "💪 **STRENGTHS**: Effective with high-dimensional data and non-linear boundaries",
                "⚠️ **CONSIDERATIONS**: Requires careful kernel and parameter selection"
            ],
            'statistical': [
                "📊 **STATISTICAL METHODS**: Reliable for normally distributed data",
                "✅ **STRENGTHS**: Interpretable results with clear statistical thresholds",
                "⚠️ **CONSIDERATIONS**: Assumes normal distribution and linear relationships"
            ],
            'dbscan': [
                "🔍 **DBSCAN CLUSTERING**: Effective for density-based anomaly detection",
                "🎯 **STRENGTHS**: Identifies clusters and noise points automatically",
                "⚠️ **CONSIDERATIONS**: Sensitive to epsilon and minimum points parameters"
            ]
        }
        
        method_key = method.lower().replace(' ', '_')
        if method_key in method_insights:
            assessment.extend(method_insights[method_key])
        else:
            assessment.append(f"🔧 **{method.upper()}**: Advanced anomaly detection method applied")
        
        # Performance assessment
        if detection_stats:
            if 'precision' in detection_stats:
                precision = detection_stats['precision']
                if precision > 0.8:
                    assessment.append(f"✅ **HIGH PRECISION**: {precision:.1%} - Low false positive rate")
                elif precision > 0.6:
                    assessment.append(f"📊 **MODERATE PRECISION**: {precision:.1%} - Acceptable false positive rate")
                else:
                    assessment.append(f"⚠️ **LOW PRECISION**: {precision:.1%} - High false positive rate needs adjustment")
            
            if 'recall' in detection_stats:
                recall = detection_stats['recall']
                if recall > 0.8:
                    assessment.append(f"✅ **HIGH RECALL**: {recall:.1%} - Excellent anomaly detection rate")
                elif recall > 0.6:
                    assessment.append(f"📊 **MODERATE RECALL**: {recall:.1%} - Good anomaly detection rate")
                else:
                    assessment.append(f"⚠️ **LOW RECALL**: {recall:.1%} - Missing many anomalies, tune sensitivity")
            
            if 'contamination' in detection_stats:
                contamination = detection_stats['contamination']
                assessment.append(f"🎯 **CONTAMINATION LEVEL**: {contamination:.1%} expected anomaly rate configured")
        
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
        
        anomaly_count = total_anomalies
        
        # Scale-based impact assessment
        if df is not None:
            # Handle both DataFrame and dict representations
            if hasattr(df, 'shape'):
                total_records = df.shape[0]
            elif isinstance(df, dict) and 'shape' in df:
                total_records = df['shape'][0] if isinstance(df['shape'], (list, tuple)) else None
            else:
                total_records = None
                
            if total_records and total_records > 0:
                impact_scale = (anomaly_count / total_records) * 100
                
                if impact_scale > 20:
                    impact_assessment.append("🚨 **CRITICAL DATA CRISIS**: Over 20% anomaly rate indicates fundamental data quality breakdown")
                    impact_assessment.append("⚡ **IMMEDIATE RESPONSE REQUIRED**: Business operations at risk - halt automated decisions")
                    impact_assessment.append("🛠️ **EMERGENCY ACTIONS**: Initiate data validation protocols and system diagnostics")
                elif impact_scale > 10:
                    impact_assessment.append("🚨 **SEVERE DATA COMPROMISE**: 10-20% anomaly rate suggests systematic collection issues")
                    impact_assessment.append("⚠️ **HIGH BUSINESS RISK**: Decision accuracy significantly compromised")
                    impact_assessment.append("📋 **CORRECTIVE MEASURES**: Implement enhanced quality controls and validation")
                elif impact_scale > 5:
                    impact_assessment.append("⚠️ **MODERATE DATA QUALITY CONCERNS**: 5-10% anomaly rate requires investigation")
                    impact_assessment.append("📊 **BUSINESS IMPACT**: Potential accuracy degradation in analytics and reporting")
                    impact_assessment.append("� **RECOMMENDED ACTIONS**: Detailed analysis and targeted corrections")
                elif impact_scale > 1:
                    impact_assessment.append("📊 **MINOR DATA QUALITY ISSUES**: 1-5% anomaly rate within acceptable variance")
                    impact_assessment.append("✅ **LIMITED BUSINESS IMPACT**: Normal operational tolerance levels")
                    impact_assessment.append("🔄 **STANDARD MONITORING**: Continue routine quality assurance procedures")
                else:
                    impact_assessment.append("📉 **MINIMAL DATA IMPACT**: <1% anomaly rate indicates excellent data quality")
                    impact_assessment.append("✅ **BUSINESS CONFIDENCE**: Data suitable for all operational and analytical purposes")
                    impact_assessment.append("🔍 **ISOLATED INVESTIGATION**: Review individual anomalies for process improvement")
        
        # Data domain-specific impact analysis
        if df is not None:
            # Handle both DataFrame and dict representations for column analysis
            if hasattr(df, 'columns'):
                column_names = df.columns.tolist()
            elif isinstance(df, dict) and 'columns' in df:
                column_names = df['columns'] if isinstance(df['columns'], list) else []
            else:
                column_names = []
                
            column_text = " ".join(column_names).lower() if column_names else ""
            
            # Financial/numeric data
            if any(keyword in column_text for keyword in ['revenue', 'cost', 'price', 'profit', 'sales']):
                impact_assessment.append("💰 **NUMERICAL DATA IMPACT**: Anomalies affect critical quantitative variables")
                impact_assessment.append("🎯 **PRIORITY**: Numerical anomalies require statistical validation")
            
            # Entity data
            if any(keyword in column_text for keyword in ['customer', 'user', 'satisfaction', 'service']):
                impact_assessment.append("👥 **ENTITY DATA IMPACT**: Anomalies affect entity-related variables")
                impact_assessment.append("� **SEGMENTATION**: Consider isolating affected data segments")
            
            # Process data
            if any(keyword in column_text for keyword in ['process', 'operation', 'efficiency', 'quality']):
                impact_assessment.append("⚙️ **PROCESS DATA IMPACT**: Data quality in process metrics compromised")
                impact_assessment.append("🔧 **DATA REVIEW**: Evaluate data collection and processing procedures")
            
            # Compliance/integrity impact
            if any(keyword in column_text for keyword in ['compliance', 'regulation', 'audit', 'risk']):
                impact_assessment.append("⚖️ **DATA INTEGRITY IMPACT**: Statistical reliability and validity implications")
                impact_assessment.append("📋 **DOCUMENTATION**: Record anomaly patterns for statistical analysis")
        
        # Method-specific data implications
        if 'fraud' in method.lower():
            impact_assessment.append("🚨 **FRAUD PATTERNS**: Statistical signatures of potentially fraudulent activity")
        elif 'quality' in method.lower():
            impact_assessment.append("✅ **QUALITY METRICS**: Data quality deviations identified")
        elif 'security' in method.lower():
            impact_assessment.append("🛡️ **SECURITY ANOMALIES**: Statistical patterns indicate potential security concerns")
        
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
        elif thresholds and isinstance(thresholds, dict):
            total_anomalies = sum(
                threshold_data.get('count', 0) 
                for threshold_data in thresholds.values() 
                if isinstance(threshold_data, dict)
            )
        elif anomalies:
            total_anomalies = len(anomalies) if isinstance(anomalies, list) else 1
            
        if total_anomalies == 0:
            risk_assessment.append("✅ **MINIMAL RISK**: No anomalies detected - risk levels within acceptable parameters")
            return risk_assessment
        
        # Score-based risk assessment
        if anomaly_scores and isinstance(anomaly_scores, list):
            scores_array = np.array(anomaly_scores)
            
            if len(scores_array) > 0:
                max_score = np.max(scores_array)
                mean_score = np.mean(scores_array)
                
                # Risk categorization based on scores
                if max_score > 0.8:
                    risk_assessment.append("🚨 **CRITICAL RISK**: Maximum anomaly score indicates severe deviation")
                elif max_score > 0.6:
                    risk_assessment.append("⚠️ **HIGH RISK**: Significant anomaly scores require attention")
                elif max_score > 0.4:
                    risk_assessment.append("📊 **MODERATE RISK**: Noticeable anomaly patterns detected")
                else:
                    risk_assessment.append("📉 **LOW RISK**: Minor anomaly scores indicate minor deviations")
                
                # Distribution analysis
                high_risk_count = np.sum(scores_array > 0.7)
                medium_risk_count = np.sum((scores_array > 0.4) & (scores_array <= 0.7))
                low_risk_count = len(scores_array) - high_risk_count - medium_risk_count
                
                risk_assessment.append(f"📊 **RISK DISTRIBUTION**: {high_risk_count} critical, {medium_risk_count} moderate, {low_risk_count} low risk")
        
        # Threshold-based assessment
        if thresholds:
            for threshold_name, threshold_value in thresholds.items():
                risk_assessment.append(f"🎯 **{threshold_name.upper()} THRESHOLD**: {threshold_value} - configured for risk detection")
        
        # Frequency-based risk
        anomaly_count = total_anomalies
        
        if anomaly_count > 100:
            risk_assessment.append("🚨 **SYSTEMIC RISK**: Large number of anomalies indicates system-wide issues")
        elif anomaly_count > 20:
            risk_assessment.append("⚠️ **PATTERN RISK**: Multiple anomalies suggest recurring issues")
        elif anomaly_count > 5:
            risk_assessment.append("📊 **CLUSTER RISK**: Several anomalies indicate localized problems")
        else:
            risk_assessment.append("🎯 **ISOLATED RISK**: Few anomalies suggest isolated incidents")
        
        # Business continuity risk
        risk_assessment.append("📋 **BUSINESS CONTINUITY**: Assess impact on ongoing operations and customer service")
        risk_assessment.append("🔄 **RECOVERY PLANNING**: Develop contingency plans for anomaly response")
        
        return risk_assessment
    
    @staticmethod
    def _generate_action_recommendations(anomalies, method: str, anomaly_results=None) -> list:
        """Generate specific action recommendations based on anomalies"""
        recommendations = []
        
        # Check for anomalies in enhanced results first, then fallback to original
        total_anomalies = 0
        critical_variables = []
        high_priority_variables = []
        
        if anomaly_results and isinstance(anomaly_results, dict):
            total_anomalies = sum(
                col_data.get('anomaly_count', col_data.get('anomalies_detected', 0)) 
                for col_data in anomaly_results.values() 
                if isinstance(col_data, dict)
            )
            
            # Categorize variables by priority for comprehensive action planning
            for variable, col_data in anomaly_results.items():
                if isinstance(col_data, dict):
                    rate = col_data.get('anomaly_percentage', 0)
                    count = col_data.get('anomaly_count', col_data.get('anomalies_detected', 0))
                    if rate > 20:
                        critical_variables.append((variable, count, rate))
                    elif rate > 10:
                        high_priority_variables.append((variable, count, rate))
                        
        elif anomalies:
            total_anomalies = len(anomalies) if isinstance(anomalies, list) else 1
            
        if total_anomalies == 0:
            recommendations.extend([
                "✅ **EXCELLENCE MAINTENANCE**: Current data quality standards are optimal",
                "📊 **CONTINUOUS IMPROVEMENT**: Monitor for emerging patterns and trends",
                "🔄 **PREVENTIVE MEASURES**: Maintain robust data validation processes",
                "🎯 **BENCHMARK ESTABLISHMENT**: Use current performance as quality baseline"
            ])
            return recommendations
        
        # Priority-based detailed recommendations
        if critical_variables:
            recommendations.append("� **CRITICAL PRIORITY ACTIONS (IMMEDIATE - 24-48 HOURS):**")
            for var, count, rate in critical_variables[:3]:
                recommendations.append(f"   1. **{var} EMERGENCY REVIEW**: {rate:.1f}% anomaly rate requires immediate investigation")
                recommendations.append(f"      - Halt automated decisions using {var}")
                recommendations.append(f"      - Validate data collection sensors/sources")
                recommendations.append(f"      - Implement manual quality checks")
                recommendations.append("")
            
        if high_priority_variables:
            recommendations.append("⚠️ **HIGH PRIORITY ACTIONS (1-2 WEEKS):**")
            for var, count, rate in high_priority_variables[:3]:
                recommendations.append(f"   • **{var} QUALITY REVIEW**: {rate:.1f}% anomaly rate needs attention")
                recommendations.append(f"     - Schedule detailed statistical analysis")
                recommendations.append(f"     - Review data processing algorithms")
                recommendations.append(f"     - Check calibration and configuration")
        
        anomaly_count = total_anomalies
        
        # Immediate actions
        recommendations.append("⚡ **IMMEDIATE ACTIONS**:")
        
        if anomaly_count > 50:
            recommendations.extend([
                "   🚨 Implement comprehensive statistical analysis",
                "   � Conduct multivariate outlier validation",
                "   � Consider isolation of affected data points"
            ])
        elif anomaly_count > 10:
            recommendations.extend([
                "   ⚠️ Perform detailed statistical analysis",
                "   📊 Conduct root cause analysis using statistical methods",
                "   🔍 Investigate correlated variables and patterns"
            ])
        else:
            recommendations.extend([
                "   🔍 Examine individual anomalies statistically",
                "   📋 Document statistical properties of anomalies",
                "   🎯 Validate with appropriate statistical tests"
            ])
        
        # Investigation actions
        recommendations.append("🔍 **INVESTIGATION ACTIONS**:")
        recommendations.extend([
            "   📊 Analyze anomaly distributions and statistical properties",
            "   🕒 Review temporal trends and autocorrelation patterns",
            "   🔗 Check variable relationships and covariance structures",
            "   � Apply multivariate analysis techniques"
        ])
        
        # Prevention actions
        recommendations.append("🛡️ **PREVENTION ACTIONS**:")
        recommendations.extend([
            "   🔧 Optimize statistical detection thresholds",
            "   📈 Enhance monitoring with additional statistical metrics",
            "   📋 Update data validation procedures",
            "   📊 Implement improved outlier detection algorithms"
        ])
        
        # Method-specific recommendations
        if 'statistical' in method.lower():
            recommendations.extend([
                "📊 **STATISTICAL ACTIONS**:",
                "   📈 Validate statistical assumptions and parameters",
                "   🔄 Consider robust statistical methods for outliers"
            ])
        elif 'machine_learning' in method.lower() or 'isolation' in method.lower():
            recommendations.extend([
                "🤖 **ML MODEL ACTIONS**:",
                "   🔄 Retrain model with recent data",
                "   🎯 Tune hyperparameters for better precision"
            ])
        
        # Communication actions
        recommendations.append("📢 **TECHNICAL DOCUMENTATION**:")
        recommendations.extend([
            "   📋 Create statistical summary of anomaly findings",
            "   � Generate visualization of anomaly distributions",
            "   � Document methodological approach to anomaly detection",
            "   📝 Record technical limitations and statistical assumptions"
        ])
        
        # Follow-up actions
        recommendations.append("🔄 **FOLLOW-UP ACTIONS**:")
        recommendations.extend([
            "   📅 Implement periodic statistical quality checks",
            "   📊 Monitor statistical properties of variables over time",
            "   🎯 Measure effectiveness of anomaly detection methods",
            "   📈 Update detection models with new data distributions"
        ])
        
        return recommendations
