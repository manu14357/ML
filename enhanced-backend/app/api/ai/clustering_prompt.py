"""
Advanced Clustering Prompt Generator
Generates sophisticated prompts for clustering node analysis based on actual cluster results, metrics, and data characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json

class ClusteringPrompt:
    """Advanced prompt generator for clustering analysis nodes"""
    
    @staticmethod
    def generate_advanced_prompt(node_data: Dict[str, Any], node_id: str, context: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive clustering analysis prompt based on actual cluster results and data
        
        Args:
            node_data: Clustering node output data including clusters, metrics, centroids
            node_id: Unique identifier for this clustering node
            context: Additional workflow context
            
        Returns:
            Advanced prompt string for AI analysis
        """
        
        # Extract clustering components
        cluster_info = ClusteringPrompt._extract_cluster_information(node_data)
        cluster_metrics = ClusteringPrompt._extract_cluster_metrics(node_data)
        cluster_analysis = ClusteringPrompt._analyze_cluster_distributions(node_data)
        centroid_analysis = ClusteringPrompt._analyze_centroids(node_data)
        data_characteristics = ClusteringPrompt._analyze_data_characteristics(node_data)
        technical_insights = ClusteringPrompt._generate_technical_insights(node_data, context)
        
        prompt = f"""
🎯 **ADVANCED CLUSTERING ANALYSIS: {node_id.upper()}**

═══════════════════════════════════════════════════════════════════
**UNSUPERVISED CLUSTERING INTELLIGENCE REPORT**
═══════════════════════════════════════════════════════════════════

{cluster_info}

{cluster_metrics}

{cluster_analysis}

{centroid_analysis}

{data_characteristics}

{technical_insights}

🚀 **TECHNICAL CLUSTERING ANALYSIS REQUIREMENTS**:

**CRITICAL**: Provide detailed clustering analysis focusing on technical performance and data patterns:

1. **CLUSTER QUALITY ASSESSMENT**:
   - Evaluate clustering metrics like silhouette score, inertia, and Davies-Bouldin index
   - Analyze within-cluster sum of squares (WCSS) and between-cluster separation
   - Assess cluster cohesion and separation quality
   - Identify optimal number of clusters using elbow method or gap statistic

2. **CLUSTER CHARACTERISTICS ANALYSIS**:
   - Examine cluster size distribution and balance
   - Analyze centroid positions and feature importance per cluster
   - Identify distinctive features that define each cluster
   - Evaluate cluster stability and robustness

3. **DATA PATTERN INSIGHTS**:
   - Analyze feature distributions within and across clusters
   - Identify natural data groupings and outlier patterns
   - Examine cluster overlap and boundary regions
   - Assess dimensionality reduction impact on clustering

4. **ALGORITHM PERFORMANCE EVALUATION**:
   - Compare different clustering algorithms (K-means, hierarchical, DBSCAN)
   - Evaluate convergence behavior and computational efficiency
   - Assess sensitivity to initialization and hyperparameters
   - Analyze clustering stability across multiple runs

5. **FEATURE SPACE ANALYSIS**:
   - Identify most discriminative features for cluster separation
   - Analyze feature scaling and normalization effects
   - Evaluate curse of dimensionality impact on clustering
   - Recommend feature selection or engineering improvements

6. **CLUSTER VALIDATION INSIGHTS**:
   - Assess clustering validity using internal and external measures
   - Evaluate cluster interpretability and actionability
   - Identify potential cluster merging or splitting opportunities
   - Recommend clustering parameter optimization strategies

**NODE CONTEXT**: {json.dumps(context, indent=2) if context else 'Standard clustering node in unsupervised learning workflow'}

**OUTPUT REQUIREMENTS**: Provide technical, data-focused insights that help optimize clustering performance and interpret data patterns.
"""
        
        return prompt.strip()
    
    @staticmethod
    def _extract_cluster_information(node_data: Dict[str, Any]) -> str:
        """Extract and format clustering algorithm and configuration information"""
        cluster_info = []
        
        # Check for algorithm information
        if 'algorithm' in node_data:
            algorithm = node_data['algorithm']
            cluster_info.append(f"🤖 **Algorithm**: {algorithm}")
        
        # Check for model details
        if 'model' in node_data:
            model = node_data['model']
            if hasattr(model, '__class__'):
                model_type = model.__class__.__name__
                cluster_info.append(f"🔧 **Model Type**: {model_type}")
                
                # Extract key parameters
                if hasattr(model, 'get_params'):
                    try:
                        params = model.get_params()
                        key_params = {k: v for k, v in params.items() if k in ['n_clusters', 'eps', 'min_samples', 'linkage', 'metric']}
                        if key_params:
                            cluster_info.append(f"⚙️ **Parameters**: {key_params}")
                    except:
                        pass
        
        # Check for configuration
        if 'config' in node_data:
            config = node_data['config']
            if 'n_clusters' in config:
                cluster_info.append(f"🎯 **Target Clusters**: {config['n_clusters']}")
            if 'feature_columns' in config:
                feature_count = len(config['feature_columns']) if config['feature_columns'] else 'Auto-selected'
                cluster_info.append(f"📊 **Features Used**: {feature_count} features")
            if 'scale_features' in config:
                scaling = "Applied" if config['scale_features'] else "Not applied"
                cluster_info.append(f"📏 **Feature Scaling**: {scaling}")
        
        # Check for number of clusters found
        if 'clusters' in node_data:
            clusters = node_data['clusters']
            if isinstance(clusters, (list, np.ndarray, pd.Series)):
                unique_clusters = len(np.unique(clusters))
                cluster_info.append(f"🏷️ **Clusters Found**: {unique_clusters}")
        
        return "📋 **CLUSTERING CONFIGURATION**:\n" + "\n".join(cluster_info) if cluster_info else "📋 **CLUSTERING CONFIGURATION**: Configuration information not available"
    
    @staticmethod
    def _extract_cluster_metrics(node_data: Dict[str, Any]) -> str:
        """Extract and format clustering quality metrics"""
        metrics_info = []
        
        # Check for metrics in node_data
        if 'metrics' in node_data:
            metrics = node_data['metrics']
            
            # Silhouette Score
            if 'silhouette_score' in metrics:
                silhouette = metrics['silhouette_score']
                metrics_info.append(f"📊 **Silhouette Score**: {silhouette:.4f}")
                
                # Silhouette interpretation
                if silhouette >= 0.7:
                    sil_quality = "🟢 **EXCELLENT** - Very well-defined clusters"
                elif silhouette >= 0.5:
                    sil_quality = "🟡 **GOOD** - Reasonable cluster structure"
                elif silhouette >= 0.25:
                    sil_quality = "🟠 **MODERATE** - Some overlap between clusters"
                else:
                    sil_quality = "🔴 **POOR** - Clusters not well-separated"
                
                metrics_info.append(sil_quality)
            
            # Inertia (for K-means)
            if 'inertia' in metrics:
                inertia = metrics['inertia']
                metrics_info.append(f"📐 **Inertia**: {inertia:.2f}")
            
            # Adjusted Rand Index
            if 'adjusted_rand_score' in metrics:
                ari = metrics['adjusted_rand_score']
                metrics_info.append(f"🎯 **Adjusted Rand Index**: {ari:.4f}")
            
            # Calinski-Harabasz Index
            if 'calinski_harabasz_score' in metrics:
                ch_score = metrics['calinski_harabasz_score']
                metrics_info.append(f"📈 **Calinski-Harabasz Score**: {ch_score:.2f}")
            
            # Davies-Bouldin Index
            if 'davies_bouldin_score' in metrics:
                db_score = metrics['davies_bouldin_score']
                metrics_info.append(f"📊 **Davies-Bouldin Score**: {db_score:.4f} (lower is better)")
        
        return "📊 **CLUSTERING QUALITY METRICS**:\n" + "\n".join(metrics_info) if metrics_info else "📊 **CLUSTERING QUALITY METRICS**: Metrics not available"
    
    @staticmethod
    def _analyze_cluster_distributions(node_data: Dict[str, Any]) -> str:
        """Analyze cluster size distributions and balance"""
        distribution_analysis = []
        
        # Check for cluster assignments
        if 'clusters' in node_data:
            clusters = node_data['clusters']
            
            if isinstance(clusters, (list, np.ndarray, pd.Series)):
                clusters = np.array(clusters)
                
                # Cluster distribution
                unique_clusters, counts = np.unique(clusters, return_counts=True)
                total_points = len(clusters)
                
                distribution_analysis.append(f"📈 **Total Data Points**: {total_points:,}")
                distribution_analysis.append(f"🏷️ **Number of Clusters**: {len(unique_clusters)}")
                
                # Cluster sizes
                distribution_analysis.append("📊 **Cluster Sizes**:")
                for cluster_id, count in zip(unique_clusters, counts):
                    percentage = (count / total_points) * 100
                    # Handle noise cluster for DBSCAN (usually -1)
                    if cluster_id == -1:
                        distribution_analysis.append(f"   • Noise Points: {count:,} ({percentage:.1f}%)")
                    else:
                        distribution_analysis.append(f"   • Cluster {cluster_id}: {count:,} ({percentage:.1f}%)")
                
                # Cluster balance assessment
                if len(unique_clusters) > 1:
                    # Exclude noise points for balance calculation
                    valid_counts = counts[unique_clusters != -1] if -1 in unique_clusters else counts
                    
                    if len(valid_counts) > 0:
                        max_percentage = max(valid_counts) / total_points * 100
                        min_percentage = min(valid_counts) / total_points * 100
                        balance_ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')
                        
                        if balance_ratio < 2:
                            balance_status = "⚖️ **Well-balanced** cluster distribution"
                        elif balance_ratio < 5:
                            balance_status = "⚠️ **Moderately imbalanced** cluster distribution"
                        else:
                            balance_status = "🚨 **Highly imbalanced** cluster distribution - consider re-clustering"
                        
                        distribution_analysis.append(balance_status)
        
        return "📊 **CLUSTER DISTRIBUTION ANALYSIS**:\n" + "\n".join(distribution_analysis) if distribution_analysis else "📊 **CLUSTER DISTRIBUTION ANALYSIS**: Cluster data not available"
    
    @staticmethod
    def _analyze_centroids(node_data: Dict[str, Any]) -> str:
        """Analyze cluster centroids and characteristics"""
        centroid_analysis = []
        
        # Check for cluster centers/centroids
        if 'cluster_centers' in node_data:
            centroids = node_data['cluster_centers']
            
            if isinstance(centroids, np.ndarray):
                n_clusters, n_features = centroids.shape
                centroid_analysis.append(f"🎯 **Centroids Available**: {n_clusters} clusters × {n_features} features")
                
                # Analyze centroid spread
                feature_ranges = []
                for i in range(n_features):
                    feature_values = centroids[:, i]
                    feature_range = feature_values.max() - feature_values.min()
                    feature_ranges.append(feature_range)
                
                avg_range = np.mean(feature_ranges)
                centroid_analysis.append(f"📐 **Centroid Separation**: Average feature range {avg_range:.3f}")
                
                # Identify most discriminating features
                max_range_idx = np.argmax(feature_ranges)
                centroid_analysis.append(f"🔍 **Most Discriminating Feature**: Feature {max_range_idx} (range: {feature_ranges[max_range_idx]:.3f})")
        
        # Check for feature names to provide better centroid interpretation
        if 'feature_names' in node_data and 'cluster_centers' in node_data:
            feature_names = node_data['feature_names']
            centroids = node_data['cluster_centers']
            
            if len(feature_names) == centroids.shape[1]:
                centroid_analysis.append("📋 **Centroid Interpretation**: Feature-level cluster characteristics available")
        
        # Check for labeled centroids or cluster profiles
        if 'cluster_profiles' in node_data:
            centroid_analysis.append("👥 **Cluster Profiles**: Detailed cluster characteristics available")
        
        return "🎯 **CENTROID ANALYSIS**:\n" + "\n".join(centroid_analysis) if centroid_analysis else "🎯 **CENTROID ANALYSIS**: Centroid data not available"
    
    @staticmethod
    def _analyze_data_characteristics(node_data: Dict[str, Any]) -> str:
        """Analyze data characteristics used for clustering"""
        data_analysis = []
        
        # Check for input data information
        if 'data' in node_data and isinstance(node_data['data'], pd.DataFrame):
            df = node_data['data']
            
            data_analysis.append(f"📊 **Dataset Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            data_analysis.append(f"🔢 **Numeric Features**: {len(numeric_cols)}")
            data_analysis.append(f"🏷️ **Categorical Features**: {len(categorical_cols)}")
            
            # Missing values
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                missing_percentage = (missing_values / (df.shape[0] * df.shape[1])) * 100
                data_analysis.append(f"❗ **Missing Values**: {missing_values:,} ({missing_percentage:.1f}%)")
            else:
                data_analysis.append("✅ **Data Completeness**: No missing values")
            
            # Feature scaling indication
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                feature_scales = numeric_df.std()
                scale_variation = feature_scales.max() / feature_scales.min() if feature_scales.min() > 0 else float('inf')
                
                if scale_variation > 10:
                    data_analysis.append("⚠️ **Feature Scaling**: Recommended - large scale differences detected")
                else:
                    data_analysis.append("✅ **Feature Scaling**: Features appear to be on similar scales")
        
        # Check for preprocessing information
        if 'preprocessing' in node_data:
            preprocessing = node_data['preprocessing']
            if isinstance(preprocessing, dict):
                if preprocessing.get('scaled'):
                    data_analysis.append("📏 **Feature Scaling**: Applied for optimal clustering")
                if preprocessing.get('encoded'):
                    data_analysis.append("🔤 **Categorical Encoding**: Applied")
        
        return "📋 **DATA CHARACTERISTICS**:\n" + "\n".join(data_analysis) if data_analysis else "📋 **DATA CHARACTERISTICS**: Data information not available"
    
    @staticmethod
    def _generate_technical_insights(node_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate technical insights for clustering analysis"""
        technical_insights = []
        
        # Clustering quality technical assessment
        silhouette_score = node_data.get('metrics', {}).get('silhouette_score', 0)
        if silhouette_score > 0:
            if silhouette_score >= 0.7:
                technical_insights.append("🎯 **Excellent Separation**: Silhouette score >0.7 indicates well-separated clusters")
                technical_insights.append("✅ **Strong Structure**: Clear data patterns with minimal cluster overlap")
            elif silhouette_score >= 0.5:
                technical_insights.append("� **Good Clustering**: Silhouette score 0.5-0.7 shows reasonable cluster separation")
                technical_insights.append("� **Optimization Potential**: Minor improvements possible through parameter tuning")
            elif silhouette_score >= 0.25:
                technical_insights.append("⚠️ **Moderate Quality**: Silhouette score 0.25-0.5 indicates overlapping clusters")
                technical_insights.append("🛠️ **Enhancement Needed**: Consider different algorithms or feature engineering")
            else:
                technical_insights.append("🚨 **Poor Clustering**: Low silhouette score suggests weak cluster structure")
                technical_insights.append("� **Algorithm Review**: Consider alternative clustering methods or data preprocessing")
        
        # Inertia/WCSS analysis
        inertia = node_data.get('metrics', {}).get('inertia')
        if inertia is not None:
            technical_insights.append(f"📊 **Within-cluster Sum of Squares**: {inertia:.2f} - measures cluster compactness")
            technical_insights.append("📈 **Convergence Analysis**: WCSS helps evaluate optimal cluster number")
        
        # Cluster distribution analysis
        if 'clusters' in node_data:
            clusters = np.array(node_data['clusters'])
            unique_clusters, counts = np.unique(clusters[clusters != -1], return_counts=True)
            
            # Cluster balance assessment
            cluster_balance = counts.std() / counts.mean() if counts.mean() > 0 else 0
            if cluster_balance < 0.5:
                technical_insights.append("⚖️ **Balanced Clusters**: Even distribution of data points across clusters")
            elif cluster_balance < 1.0:
                technical_insights.append("📊 **Moderate Imbalance**: Some clusters larger than others")
            else:
                technical_insights.append("⚠️ **Imbalanced Clusters**: Significant size differences between clusters")
            
            # Noise detection
            noise_points = np.sum(clusters == -1)
            if noise_points > 0:
                noise_percentage = (noise_points / len(clusters)) * 100
                technical_insights.append(f"� **Outlier Detection**: {noise_points} noise points ({noise_percentage:.1f}%)")
        
        # Feature space analysis
        if 'cluster_centers' in node_data or 'centroids' in node_data:
            technical_insights.append("� **Centroid Analysis**: Cluster centers define feature space partitions")
            technical_insights.append("� **Feature Importance**: Centroid differences highlight discriminative features")
        
        # Algorithm-specific insights
        algorithm = node_data.get('algorithm', '').lower()
        if 'kmeans' in algorithm:
            technical_insights.append("🎯 **K-means Properties**: Spherical clusters with equal variance assumption")
        elif 'dbscan' in algorithm:
            technical_insights.append("� **DBSCAN Properties**: Density-based clustering with automatic outlier detection")
        elif 'hierarchical' in algorithm:
            technical_insights.append("🌳 **Hierarchical Structure**: Tree-based clustering reveals data hierarchy")
        
        # Dimensionality considerations
        if 'data' in node_data and isinstance(node_data['data'], pd.DataFrame):
            n_features = node_data['data'].shape[1]
            if n_features > 10:
                technical_insights.append("📐 **High Dimensionality**: Consider dimensionality reduction for visualization")
            technical_insights.append(f"� **Feature Space**: {n_features} dimensions used for clustering")
        
        return "� **TECHNICAL INSIGHTS**:\n" + "\n".join(technical_insights) if technical_insights else "� **TECHNICAL INSIGHTS**: Standard clustering analysis results"
