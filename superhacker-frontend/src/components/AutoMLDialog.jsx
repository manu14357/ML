import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { 
  Sparkles, Brain, Zap, Target, BarChart3, Loader2, 
  CheckCircle, Database, TrendingUp, Layers, FlaskConical,
  Info, AlertTriangle, Lightbulb, Star, RefreshCw, ChevronDown
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'

// Use environment variable or fallback to relative path
const API_BASE = import.meta.env.VITE_API_URL || '/api'

// Add better error handling for API calls
const fetchWithErrorHandling = async (url, options = {}) => {
  try {
    const response = await fetch(url, options)
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.error || errorData.message || `HTTP ${response.status}: ${response.statusText}`)
    }
    
    return await response.json()
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

// Helper function to provide suggestions based on error message
const getSuggestionForError = (errorMsg, modelType, targetColumn) => {
  const errorLower = errorMsg.toLowerCase();
  
  if (errorLower.includes('too many unique') && errorLower.includes('classification')) {
    return `"${targetColumn}" has too many unique values for classification. Try using regression instead, or enable automatic binning in advanced options.`;
  }
  
  if (errorLower.includes('identifier') || errorLower.includes('unique ratio')) {
    return `"${targetColumn}" appears to be an identifier column that's not suitable for ${modelType}. Choose another target column that represents the outcome you want to predict.`;
  }
  
  if (errorLower.includes('missing value')) {
    return `"${targetColumn}" has too many missing values. Select another column with fewer missing values.`;
  }
  
  if (errorLower.includes('numeric') && modelType === 'regression') {
    return `"${targetColumn}" needs to contain numeric values for regression. Choose a numeric column or switch to classification.`;
  }
  
  if (errorLower.includes('at least 2 unique values')) {
    return `"${targetColumn}" needs at least 2 different values for classification. Choose another column with multiple categories.`;
  }
  
  if (errorLower.includes('only 1 sample') || errorLower.includes('each class needs at least 2 samples')) {
    return `Some classes in "${targetColumn}" have only 1 sample, but each class needs at least 2 samples for training. Try selecting a different column or dataset with better class distribution.`;
  }
  
  // Default suggestion
  return `Try selecting a different target column or changing the model type from ${modelType} to ${modelType === 'classification' ? 'regression' : 'classification'}.`;
}

export function AutoMLDialog({ onAutoTrain }) {
  const [isOpen, setIsOpen] = useState(false)
  const [datasets, setDatasets] = useState([])
  const [datasetsLoading, setDatasetsLoading] = useState(false)
  const [datasetsError, setDatasetsError] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState('')
  const [targetColumn, setTargetColumn] = useState('')
  const [modelType, setModelType] = useState('classification')
  const [recommendations, setRecommendations] = useState(null)
  const [featureSelection, setFeatureSelection] = useState(null)
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState(1) // 1: Setup, 2: Analysis, 3: Training
  const [datasetInfo, setDatasetInfo] = useState(null)
  const [batchTraining, setBatchTraining] = useState(false)
  const [trainedModels, setTrainedModels] = useState([])
  const [suggestedTargetColumns, setSuggestedTargetColumns] = useState([])
  const [analyzingColumns, setAnalyzingColumns] = useState(false)
  const [error, setError] = useState(null) // Add general error state
  const [showErrorDetails, setShowErrorDetails] = useState(false) // For expanding error details
  const [processingInfo, setProcessingInfo] = useState(null) // Store preprocessing info
  const [advancedConfig, setAdvancedConfig] = useState({
    max_samples: 100000,
    enable_sampling: true,
    optimize_memory: true,
    use_incremental_learning: false,
    test_size: 0.2,
    cv_folds: 5,
    scaling: 'standard',
    handle_missing: true,
    encoding_strategy: 'auto',
    auto_binning: true // Add automatic binning for continuous variables
  })
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false)

  // Fetch datasets when dialog opens
  const fetchDatasets = async () => {
    setDatasetsLoading(true)
    setDatasetsError(null)
    try {
      const data = await fetchWithErrorHandling(`${API_BASE}/data/datasets`)
      if (data.success) {
        setDatasets(data.datasets)
      } else {
        setDatasetsError(data.error || 'Failed to load datasets')
      }
    } catch (error) {
      console.error('Error fetching datasets:', error)
      setDatasetsError(error.message || 'Error loading datasets. Please ensure the backend server is running.')
    } finally {
      setDatasetsLoading(false)
    }
  }

  // Effect to fetch datasets when dialog opens
  useEffect(() => {
    if (isOpen && datasets.length === 0) {
      fetchDatasets()
    }
  }, [isOpen, datasets.length])

  const handleDatasetChange = (datasetId) => {
    setSelectedDataset(datasetId)
    const dataset = datasets.find(d => d.id.toString() === datasetId)
    
    console.log('AutoMLDialog - Selected dataset:', dataset)
    
    // Process dataset to extract column names and fix structure
    if (dataset) {
      // Extract column names from columns_info structure
      let columns = []
      
      console.log('AutoMLDialog - Dataset columns_info:', dataset.columns_info)
      console.log('AutoMLDialog - Dataset data_types:', dataset.data_types)
      
      if (dataset.columns_info) {
        if (typeof dataset.columns_info === 'object' && !Array.isArray(dataset.columns_info)) {
          // Dictionary format: {column_name: {...}}
          columns = Object.keys(dataset.columns_info)
          console.log('AutoMLDialog - Extracted columns from object:', columns)
        } else if (Array.isArray(dataset.columns_info)) {
          // Array format: [{name: ..., ...}, ...] or string representations
          columns = dataset.columns_info.map(col => {
            if (typeof col === 'string') {
              // Handle PowerShell hashtable strings like "@{...name=column_name...}"
              const nameMatch = col.match(/name=([^;]+)/)
              if (nameMatch) {
                return nameMatch[1].trim()
              }
              return col
            } else if (col && typeof col === 'object') {
              return col.name || col.column_name || String(col)
            }
            return String(col)
          }).filter(Boolean)
          console.log('AutoMLDialog - Extracted columns from array:', columns)
        } else if (typeof dataset.columns_info === 'string') {
          // Handle string representation of PowerShell hashtable
          try {
            // Try to parse as JSON-like structure
            const jsonStr = dataset.columns_info.replace(/@\{/g, '{').replace(/;/g, ',').replace(/=/g, ':')
            const parsed = JSON.parse(jsonStr)
            columns = Object.keys(parsed)
            console.log('AutoMLDialog - Extracted columns from string:', columns)
          } catch {
            // Fallback: extract column names from PowerShell hashtable string
            const matches = dataset.columns_info.match(/name=([^;]+)/g)
            if (matches) {
              columns = matches.map(match => match.replace('name=', '').trim())
              console.log('AutoMLDialog - Extracted columns from regex:', columns)
            }
          }
        }
      }
      
      // Fallback: try to extract from data_types if columns_info failed
      if (columns.length === 0 && dataset.data_types && typeof dataset.data_types === 'object') {
        columns = Object.keys(dataset.data_types)
        console.log('AutoMLDialog - Extracted columns from data_types:', columns)
      }
      
      // Last resort: if we still don't have columns, create dummy ones based on column count
      if (columns.length === 0 && dataset.columns_count && dataset.columns_count > 0) {
        columns = Array.from({ length: dataset.columns_count }, (_, i) => `column_${i + 1}`)
        console.log('AutoMLDialog - Created dummy columns:', columns)
      }
      
      // Create enhanced dataset object with processed columns
      const enhancedDataset = {
        ...dataset,
        columns: columns,
        row_count: dataset.rows_count, // Map backend field to frontend expectation
        column_count: dataset.columns_count // Map backend field to frontend expectation
      }
      
      console.log('AutoMLDialog - Enhanced dataset:', enhancedDataset)
      setDatasetInfo(enhancedDataset)
    } else {
      setDatasetInfo(null)
    }
    
    setTargetColumn('')
    setRecommendations(null)
    setFeatureSelection(null)
    setSuggestedTargetColumns([])
    
    // Analyze and suggest target columns
    if (dataset) {
      suggestTargetColumns(dataset)
    }
  }

  const suggestTargetColumns = async (dataset) => {
    setAnalyzingColumns(true)
    try {
      const data = await fetchWithErrorHandling(`${API_BASE}/ml/automl/suggest-target-columns`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: dataset.id,
          model_type: modelType
        })
      })
      
      console.log('AutoMLDialog - Suggest target columns API response:', data)
      
      if (data.success && data.suggestions && Array.isArray(data.suggestions)) {
        console.log('AutoMLDialog - Setting suggestions:', data.suggestions)
        setSuggestedTargetColumns(data.suggestions)
        // Auto-select the top suggestion if available
        if (data.suggestions.length > 0 && !targetColumn) {
          setTargetColumn(data.suggestions[0].column_name)
        }
      } else {
        console.log('AutoMLDialog - suggest target columns failed or no suggestions:', data)
        // Fallback to simple heuristic-based suggestions
        const suggestions = getHeuristicTargetSuggestions(dataset)
        setSuggestedTargetColumns(suggestions)
        if (suggestions.length > 0 && !targetColumn) {
          setTargetColumn(suggestions[0].column_name)
        }
      }
    } catch (error) {
      console.error('Error suggesting target columns:', error)
      // Fallback to heuristic suggestions
      const suggestions = getHeuristicTargetSuggestions(dataset)
      setSuggestedTargetColumns(suggestions)
      if (suggestions.length > 0 && !targetColumn) {
        setTargetColumn(suggestions[0].column_name)
      }
    } finally {
      setAnalyzingColumns(false)
    }
  }

  const getHeuristicTargetSuggestions = (dataset) => {
    if (!dataset.columns) return []
    
    const suggestions = []
    const commonTargetPatterns = {
      classification: [
        /target/i, /label/i, /class/i, /category/i, /type/i, /outcome/i, 
        /result/i, /status/i, /approved/i, /success/i, /fraud/i, /churn/i,
        /predict/i, /response/i, /y$/i
      ],
      regression: [
        /price/i, /cost/i, /value/i, /amount/i, /salary/i, /revenue/i,
        /score/i, /rating/i, /age/i, /time/i, /duration/i, /length/i,
        /size/i, /count/i, /number/i, /quantity/i, /total/i, /sum/i,
        /y$/i, /target/i
      ]
    }
    
    const patterns = commonTargetPatterns[modelType] || commonTargetPatterns.classification
    
    dataset.columns.forEach(column => {
      let score = 0
      let reason = 'Column name pattern'
      
      // Check name patterns
      for (const pattern of patterns) {
        if (pattern.test(column)) {
          score += 10
          break
        }
      }
      
      // Prefer columns at the end (common convention)
      const columnIndex = dataset.columns.indexOf(column)
      if (columnIndex >= dataset.columns.length - 3) {
        score += 5
        reason = 'Position in dataset (common target location)'
      }
      
      // Prefer shorter names for targets
      if (column.length <= 10) {
        score += 2
      }
      
      if (score > 0) {
        suggestions.push({
          column_name: column,
          confidence: Math.min(score / 15, 1), // Normalize to 0-1
          reason: reason,
          recommended: score >= 10
        })
      }
    })
    
    // Sort by confidence and return top 3
    return suggestions
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3)
  }

  const analyzeDataset = async () => {
    if (!selectedDataset || !targetColumn) {
      console.log('AutoMLDialog - Cannot analyze: missing dataset or target column')
      return
    }

    console.log('AutoMLDialog - Starting analysis for:', {
      dataset: selectedDataset,
      target: targetColumn,
      modelType: modelType
    })

    setLoading(true)
    setError(null) // Clear any previous errors
    setProcessingInfo(null)
    
    try {
      // Get algorithm recommendations
      console.log('AutoMLDialog - Getting algorithm recommendations...')
      const recData = await fetchWithErrorHandling(`${API_BASE}/ml/automl/recommend-algorithms`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: parseInt(selectedDataset),
          target_column: targetColumn,
          model_type: modelType,
          config: advancedConfig
        })
      })
      console.log('AutoMLDialog - Algorithm recommendations response:', recData)
      
      if (recData.success) {
        setRecommendations(recData.recommendations)
        // Store preprocessing info for detailed feedback
        if (recData.preprocessing_info) {
          setProcessingInfo(recData.preprocessing_info)
        }
        console.log('AutoMLDialog - Set recommendations:', recData.recommendations)
      } else {
        const errorMsg = recData.message || 'Failed to get algorithm recommendations';
        console.error('AutoMLDialog - Algorithm recommendations failed:', errorMsg);
        
        // Handle specific error cases with friendly messages
        let suggestion = '';
        if (errorMsg.includes('classes have only 1 sample')) {
          suggestion = `The target column "${targetColumn}" has some classes with only 1 sample, but each class needs at least 2 samples for training. Try selecting a different target column with better distribution.`;
        } else {
          suggestion = recData.suggestion || getSuggestionForError(errorMsg, modelType, targetColumn);
        }
        
        setError({
          title: 'Analysis Failed',
          message: errorMsg,
          suggestion: suggestion,
          technical: recData.technical_error || errorMsg
        });
        
        // Move back to setup step if we have an error
        setStep(1);
        setLoading(false);
        return; // Stop further processing
      }

      // Get feature selection recommendations
      console.log('AutoMLDialog - Getting feature selection...')
      const featData = await fetchWithErrorHandling(`${API_BASE}/ml/automl/feature-selection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: parseInt(selectedDataset),
          target_column: targetColumn,
          model_type: modelType,
          k_features: 10
        })
      })
      console.log('AutoMLDialog - Feature selection response:', featData)
      
      if (featData.success) {
        setFeatureSelection(featData.feature_selection)
        console.log('AutoMLDialog - Set feature selection:', featData.feature_selection)
      } else {
        console.error('AutoMLDialog - Feature selection failed:', featData.message)
      }

      setStep(2)
      console.log('AutoMLDialog - Analysis complete, moved to step 2')
    } catch (error) {
      console.error('AutoMLDialog - Analysis error:', error)
      
      // Extract error message safely
      const errorMessage = error && error.message ? error.message : 'An unexpected error occurred during analysis';
      
      // Handle specific error cases
      let suggestion = '';
      if (errorMessage.includes('classes have only 1 sample')) {
        suggestion = `The target column "${targetColumn}" has some classes with only 1 sample, but each class needs at least 2 samples for training. Try selecting a different target column with better distribution.`;
      } else {
        suggestion = getSuggestionForError(errorMessage, modelType, targetColumn);
      }
      
      setError({
        title: 'Analysis Failed',
        message: errorMessage,
        suggestion: suggestion,
        technical: error && error.stack ? error.stack : errorMessage
      })
      setStep(1) // Return to setup on error
    } finally {
      setLoading(false)
    }
  }

  const startBatchTraining = async () => {
    if (!recommendations?.recommended_algorithms) return

    setBatchTraining(true)
    setStep(3)

    try {
      const data = await fetchWithErrorHandling(`${API_BASE}/ml/batch-train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: parseInt(selectedDataset),
          target_column: targetColumn,
          model_type: modelType,
          algorithms: recommendations.recommended_algorithms.slice(0, 6), // Train top 6 algorithms
          // Enhanced configuration for large datasets
          max_samples: advancedConfig.max_samples,
          enable_sampling: advancedConfig.enable_sampling,
          optimize_memory: advancedConfig.optimize_memory,
          use_incremental_learning: advancedConfig.use_incremental_learning,
          test_size: advancedConfig.test_size,
          cv_folds: advancedConfig.cv_folds,
          scaling: advancedConfig.scaling,
          handle_missing: advancedConfig.handle_missing,
          encoding_strategy: advancedConfig.encoding_strategy
        })
      })
      
      if (data.success) {
        setTrainedModels(data.batch_results)
        if (onAutoTrain) {
          onAutoTrain(data.batch_results)
        }
        
        // Show success message with summary
        const summary = data.summary || {}
        const message = `🎉 Batch Training Complete!\n\n📊 Results:\n• Total Models: ${summary.total_models || 0}\n• ✅ Successful: ${summary.successful || 0}\n• ❌ Failed: ${summary.failed || 0}\n\n📋 Dataset Info:\n• Rows: ${summary.dataset_info?.rows || 0}\n• Columns: ${summary.dataset_info?.columns || 0}\n• Target: ${summary.dataset_info?.target_column || targetColumn}`
        alert(message)
      } else {
        throw new Error(data.message || 'Batch training failed')
      }
    } catch (error) {
      console.error('Batch training error:', error)
      
      // Provide helpful error guidance
      let helpfulTips = "\n\n💡 Troubleshooting Tips:\n"
      if (error.message.includes('insufficient')) {
        helpfulTips += "• Your dataset might be too small for reliable training\n• Try adding more data or reducing the number of classes\n"
      }
      if (error.message.includes('memory') || error.message.includes('Memory')) {
        helpfulTips += "• Dataset might be too large for available memory\n• Try enabling sampling or using a smaller dataset\n"
      }
      if (error.message.includes('column') || error.message.includes('feature')) {
        helpfulTips += "• Check that your target column exists and has valid values\n• Ensure feature columns contain appropriate data types\n"
      }
      if (error.message.includes('NaN') || error.message.includes('missing')) {
        helpfulTips += "• Your dataset contains missing values\n• Try cleaning the data or enabling missing value handling\n"
      }
      
      alert(`❌ Batch Training Failed\n\n${error.message}${helpfulTips}`)
    } finally {
      setBatchTraining(false)
    }
  }

  const resetDialog = () => {
    setSelectedDataset('')
    setTargetColumn('')
    setModelType('classification')
    setRecommendations(null)
    setFeatureSelection(null)
    setStep(1)
    setDatasetInfo(null)
    setTrainedModels([])
    setSuggestedTargetColumns([])
    setAnalyzingColumns(false)
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => {
      setIsOpen(open)
      if (!open) resetDialog()
    }}>
      <DialogTrigger asChild>
        <Button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 
                          h-9 sm:h-10 px-3 sm:px-4 text-sm sm:text-base">
          <Sparkles className="h-3 w-3 sm:h-4 sm:w-4 mr-2" />
          <span className="hidden sm:inline">AutoML Assistant</span>
          <span className="sm:hidden">AutoML</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[98vw] sm:max-w-[95vw] md:max-w-4xl lg:max-w-5xl xl:max-w-6xl 
                                        max-h-[92vh] sm:max-h-[90vh] md:max-h-[88vh] lg:max-h-[90vh] overflow-y-auto
                                        macbook-13:max-w-[92vw] macbook-13:max-h-[88vh]
                                        w-full mx-auto p-2 sm:p-3 md:p-4 lg:p-6">
        <DialogHeader className="pb-2 sm:pb-4 md:pb-6">
          <DialogTitle className="flex items-center gap-2 text-base sm:text-lg md:text-xl lg:text-2xl">
            <Sparkles className="h-4 w-4 sm:h-5 sm:w-5 md:h-6 md:w-6 text-purple-600 flex-shrink-0" />
            <span className="truncate">AutoML Assistant</span>
          </DialogTitle>
          <DialogDescription className="text-xs sm:text-sm md:text-base">
            Automated machine learning to find the best model for your data
          </DialogDescription>
        </DialogHeader>

        {/* Enhanced Progress Steps */}
        <div className="flex items-center justify-center space-x-1 sm:space-x-2 md:space-x-4 mb-3 sm:mb-4 md:mb-6">
          {[
            { number: 1, label: 'Setup', icon: Database },
            { number: 2, label: 'Analysis', icon: Brain },
            { number: 3, label: 'Training', icon: Zap }
          ].map((stepInfo, index) => (
            <div key={stepInfo.number} className="flex items-center">
              <div className={`relative w-6 h-6 sm:w-8 sm:h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center border-2 text-xs sm:text-sm md:text-base font-medium transition-all duration-300 ${
                step >= stepInfo.number 
                  ? 'bg-purple-600 border-purple-600 text-white shadow-lg' 
                  : 'border-gray-300 text-gray-400 hover:border-purple-300'
              }`}>
                {step > stepInfo.number ? (
                  <CheckCircle className="h-3 w-3 sm:h-4 sm:w-4 md:h-5 md:w-5" />
                ) : (
                  <stepInfo.icon className="h-3 w-3 sm:h-4 sm:w-4 md:h-5 md:w-5" />
                )}
              </div>
              <div className="hidden sm:block ml-2 mr-4 text-xs text-muted-foreground">
                {stepInfo.label}
              </div>
              {index < 2 && (
                <div className={`w-4 sm:w-8 md:w-16 h-0.5 transition-all duration-300 ${
                  step > stepInfo.number ? 'bg-purple-600' : 'bg-gray-300'
                }`} />
              )}
            </div>
          ))}
        </div>

        {/* Step 1: Dataset Setup */}
        {step === 1 && (
          <div className="space-y-4 sm:space-y-6">
            {/* Error Display */}
            {error && (
              <Card className="border-l-4 border-red-500 bg-red-50 dark:bg-red-950/20">
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div className="space-y-2 flex-1">
                      <h4 className="font-medium text-base text-red-700 dark:text-red-400">{error.title || 'Analysis Error'}</h4>
                      <p className="text-sm text-red-600 dark:text-red-300">{error.message || 'An error occurred during analysis'}</p>
                      
                      {error.suggestion && (
                        <div className="pt-1 border-t border-red-200 dark:border-red-800">
                          <div className="flex items-center gap-1.5 text-xs text-red-600 dark:text-red-300 mt-2">
                            <Lightbulb className="h-3.5 w-3.5" />
                            <span className="font-medium">Suggestion:</span>
                          </div>
                          <p className="text-sm mt-1 text-red-600 dark:text-red-300">{error.suggestion}</p>
                        </div>
                      )}
                      
                      {error.technical && (
                        <div className="flex flex-col pt-1">
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            className="h-7 px-2 text-xs text-red-600 hover:text-red-700 hover:bg-red-100 self-start"
                            onClick={() => setShowErrorDetails(!showErrorDetails)}
                          >
                            {showErrorDetails ? 'Hide Details' : 'Show Technical Details'}
                            <ChevronDown className={`h-3 w-3 ml-1 transition-transform ${showErrorDetails ? 'rotate-180' : ''}`} />
                          </Button>
                          
                          {showErrorDetails && (
                            <pre className="text-xs bg-red-100 dark:bg-red-950/50 p-2 rounded overflow-x-auto whitespace-pre-wrap break-all mt-2">
                              {error.technical}
                            </pre>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
            
            <Card>
              <CardHeader className="pb-3 sm:pb-4">
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <Database className="h-4 w-4 sm:h-5 sm:w-5" />
                  Dataset Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 sm:space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm sm:text-base">Select Dataset</Label>
                  <Select value={selectedDataset} onValueChange={handleDatasetChange} disabled={datasetsLoading}>
                    <SelectTrigger className="h-9 sm:h-10 md:h-11 text-xs sm:text-sm">
                      <SelectValue placeholder={
                        datasetsLoading 
                          ? "Loading datasets..." 
                          : datasetsError 
                            ? "Error loading datasets" 
                            : "Choose a dataset"
                      } />
                    </SelectTrigger>
                    <SelectContent className="max-h-[200px] sm:max-h-[300px] md:max-h-[400px]">
                      {datasetsLoading ? (
                        <div className="flex items-center justify-center p-3 sm:p-4">
                          <Loader2 className="h-4 w-4 animate-spin mr-2" />
                          <span className="text-xs sm:text-sm">Loading datasets...</span>
                        </div>
                      ) : datasetsError ? (
                        <div className="flex flex-col items-center p-3 sm:p-4 space-y-2">
                          <AlertTriangle className="h-4 w-4 text-red-500" />
                          <span className="text-xs sm:text-sm text-red-500 text-center max-w-xs">{datasetsError}</span>
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={fetchDatasets}
                            className="mt-2 text-xs"
                          >
                            <RefreshCw className="h-3 w-3 mr-1" />
                            Retry
                          </Button>
                        </div>
                      ) : datasets.length === 0 ? (
                        <div className="text-center p-3 sm:p-4 text-muted-foreground">
                          <span className="text-xs sm:text-sm">No datasets available</span>
                        </div>
                      ) : (
                        datasets.map(dataset => (
                          <SelectItem key={dataset.id} value={dataset.id.toString()}>
                            <div className="flex flex-col items-start">
                              <span className="truncate font-medium">{dataset.name}</span>
                              {dataset.description && (
                                <span className="text-xs text-muted-foreground truncate max-w-xs">
                                  {dataset.description}
                                </span>
                              )}
                            </div>
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                </div>

                {datasetInfo && (
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 sm:gap-3 md:gap-4 p-2 sm:p-3 md:p-4 bg-muted rounded-lg">
                    <div className="text-center lg:text-left">
                      <span className="text-xs sm:text-sm text-muted-foreground block">Rows</span>
                      <div className="font-medium text-sm sm:text-base md:text-lg truncate">
                        {datasetInfo.row_count?.toLocaleString() || 'N/A'}
                      </div>
                    </div>
                    <div className="text-center lg:text-left">
                      <span className="text-xs sm:text-sm text-muted-foreground block">Columns</span>
                      <div className="font-medium text-sm sm:text-base md:text-lg">
                        {datasetInfo.column_count || 'N/A'}
                      </div>
                    </div>
                    <div className="text-center lg:text-left">
                      <span className="text-xs sm:text-sm text-muted-foreground block">Size</span>
                      <div className="font-medium text-sm sm:text-base md:text-lg truncate">
                        {datasetInfo.file_size_formatted || 'N/A'}
                      </div>
                    </div>
                    <div className="text-center lg:text-left">
                      <span className="text-xs sm:text-sm text-muted-foreground block">Type</span>
                      <div className="font-medium text-sm sm:text-base md:text-lg truncate">
                        {datasetInfo.file_type || 'N/A'}
                      </div>
                    </div>
                  </div>
                )}

                {datasetInfo && (
                  <div className="space-y-3">
                    <Label className="text-sm sm:text-base">Target Column</Label>
                    {Array.isArray(suggestedTargetColumns) && suggestedTargetColumns.length > 0 && (
                      <div className="mb-3">
                        <div className="text-xs sm:text-sm text-muted-foreground mb-2 flex items-center gap-2">
                          {analyzingColumns ? (
                            <>
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Analyzing columns...
                            </>
                          ) : (
                            <>
                              <Lightbulb className="h-3 w-3 text-yellow-600" />
                              Suggested target columns:
                            </>
                          )}
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2 sm:gap-3">
                          {suggestedTargetColumns.map((suggestion) => (
                            <Button
                              key={suggestion.column_name}
                              variant={targetColumn === suggestion.column_name ? "default" : "outline"}
                              size="sm"
                              onClick={() => setTargetColumn(suggestion.column_name)}
                              className="h-auto flex flex-col items-start p-2 sm:p-3 text-left w-full hover:scale-105 transition-transform"
                            >
                              <div className="flex items-center gap-2 w-full">
                                {suggestion.recommended && (
                                  <Star className="h-3 w-3 text-yellow-500 fill-current flex-shrink-0" />
                                )}
                                <span className="font-medium text-xs sm:text-sm truncate flex-1">
                                  {suggestion.column_name}
                                </span>
                                <Badge variant="secondary" className="text-xs flex-shrink-0">
                                  {Math.round(suggestion.confidence * 100)}%
                                </Badge>
                              </div>
                              <span className="text-xs text-muted-foreground mt-1 line-clamp-2 text-left">
                                {suggestion.reason}
                              </span>
                            </Button>
                          ))}
                        </div>
                      </div>
                    )}
                    <Select value={targetColumn} onValueChange={setTargetColumn}>
                      <SelectTrigger className="h-10 sm:h-11">
                        <SelectValue placeholder="Select target column" />
                      </SelectTrigger>
                      <SelectContent className="max-h-60 sm:max-h-80 overflow-y-auto">
                        {datasetInfo && datasetInfo.columns && datasetInfo.columns.length > 0 ? (
                          datasetInfo.columns.map(column => (
                            <SelectItem key={column} value={column}>
                              <div className="flex items-center gap-2 w-full">
                                {Array.isArray(suggestedTargetColumns) && suggestedTargetColumns.find(s => s.column_name === column)?.recommended && (
                                  <Star className="h-3 w-3 text-yellow-500 fill-current flex-shrink-0" />
                                )}
                                <span className="truncate flex-1">{column}</span>
                                {Array.isArray(suggestedTargetColumns) && suggestedTargetColumns.find(s => s.column_name === column) && (
                                  <Badge variant="outline" className="text-xs ml-auto flex-shrink-0">
                                    {Math.round(suggestedTargetColumns.find(s => s.column_name === column).confidence * 100)}%
                                  </Badge>
                                )}
                              </div>
                            </SelectItem>
                          ))
                        ) : datasetInfo && (!datasetInfo.columns || datasetInfo.columns.length === 0) ? (
                          <div className="text-center p-4 text-muted-foreground">
                            <AlertTriangle className="h-4 w-4 mx-auto mb-2" />
                            <div className="text-sm">No columns found in dataset</div>
                            <div className="text-xs">Please check the dataset format</div>
                          </div>
                        ) : null}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="space-y-2">
                  <Label className="text-sm sm:text-base">Model Type</Label>
                  <Select value={modelType} onValueChange={(value) => {
                    console.log('AutoMLDialog - Model type changed to:', value)
                    setModelType(value)
                    // Clear previous analysis results
                    setRecommendations(null)
                    setFeatureSelection(null)
                    setStep(1)
                    // Re-suggest target columns for the new model type
                    if (datasetInfo) {
                      console.log('AutoMLDialog - Re-suggesting target columns for new model type')
                      suggestTargetColumns(datasetInfo)
                    }
                  }}>
                    <SelectTrigger className="h-10 sm:h-11">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="classification">Classification</SelectItem>
                      <SelectItem value="regression">Regression</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button 
                  onClick={analyzeDataset} 
                  disabled={!selectedDataset || !targetColumn || loading}
                  className="w-full h-10 sm:h-11"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      <span className="text-sm sm:text-base">Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      <span className="text-sm sm:text-base">Analyze Dataset</span>
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Step 2: Analysis Results */}
        {step === 2 && (
          <div className="space-y-4 sm:space-y-6">
            {/* Preprocessing Information */}
            {processingInfo && processingInfo.warnings && processingInfo.warnings.length > 0 && (
              <Card className="border-l-4 border-l-blue-500">
                <CardHeader className="pb-3 sm:pb-4">
                  <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                    <Info className="h-4 w-4 sm:h-5 sm:w-5 text-blue-500" />
                    Preprocessing Details
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-1.5">
                      <Brain className="h-4 w-4 text-blue-600" />
                      Data Transformations
                    </h4>
                    <ul className="space-y-1.5 text-sm">
                      {processingInfo.warnings.map((warning, index) => (
                        <li key={index} className="flex items-start gap-1.5">
                          <div className="rounded-full bg-blue-100 dark:bg-blue-800 w-5 h-5 flex items-center justify-center flex-shrink-0 mt-0.5">
                            <span className="text-xs font-medium text-blue-600 dark:text-blue-300">{index + 1}</span>
                          </div>
                          <span className="text-sm text-blue-700 dark:text-blue-300">{warning}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  {processingInfo.categorical_columns_processed && processingInfo.categorical_columns_processed.length > 0 && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                      <div>
                        <h4 className="text-sm font-medium mb-2">Categorical Columns Processed</h4>
                        <div className="flex flex-wrap gap-1.5">
                          {processingInfo.categorical_columns_processed.map((col, index) => (
                            <Badge key={index} variant="outline" className="bg-green-50 text-green-700 border-green-200">
                              {col}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      {processingInfo.high_cardinality_columns && processingInfo.high_cardinality_columns.length > 0 && (
                        <div>
                          <h4 className="text-sm font-medium mb-2">High Cardinality Columns Dropped</h4>
                          <div className="flex flex-wrap gap-1.5">
                            {processingInfo.high_cardinality_columns.map((col, index) => (
                              <Badge key={index} variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">
                                {col}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
            
            {/* Data Characteristics */}
            {recommendations?.data_characteristics && (
              <Card>
                <CardHeader className="pb-3 sm:pb-4">
                  <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                    <BarChart3 className="h-4 w-4 sm:h-5 sm:w-5" />
                    Data Characteristics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
                    <div className="text-center p-3 bg-blue-50 rounded-lg">
                      <div className="text-lg sm:text-2xl font-bold text-blue-600">
                        {recommendations.data_characteristics.n_samples.toLocaleString()}
                      </div>
                      <div className="text-xs sm:text-sm text-muted-foreground">Samples</div>
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded-lg">
                      <div className="text-lg sm:text-2xl font-bold text-green-600">
                        {recommendations.data_characteristics.n_features}
                      </div>
                      <div className="text-xs sm:text-sm text-muted-foreground">Features</div>
                    </div>
                    {recommendations.data_characteristics.n_classes && (
                      <div className="text-center p-3 bg-purple-50 rounded-lg">
                        <div className="text-lg sm:text-2xl font-bold text-purple-600">
                          {recommendations.data_characteristics.n_classes}
                        </div>
                        <div className="text-xs sm:text-sm text-muted-foreground">Classes</div>
                      </div>
                    )}
                    <div className="text-center p-3 bg-orange-50 rounded-lg">
                      <div className="text-lg sm:text-2xl font-bold text-orange-600">
                        {(recommendations.data_characteristics.sample_to_feature_ratio || 0).toFixed(1)}
                      </div>
                      <div className="text-xs sm:text-sm text-muted-foreground">Sample/Feature Ratio</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Algorithm Recommendations */}
            {recommendations?.recommended_algorithms && (
              <Card>
                <CardHeader className="pb-3 sm:pb-4">
                  <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                    <Lightbulb className="h-4 w-4 sm:h-5 sm:w-5 text-yellow-600" />
                    <span className="truncate">Recommended Algorithms</span>
                  </CardTitle>
                  <CardDescription className="text-xs sm:text-sm">
                    Based on your data characteristics, these algorithms are recommended
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 sm:gap-3">
                    {recommendations.recommended_algorithms.slice(0, 6).map((algorithm, index) => (
                      <div key={index} className="flex items-center gap-2 sm:gap-3 p-2 sm:p-3 border rounded-lg hover:bg-muted/50 transition-colors">
                        <div className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center text-white text-xs sm:text-sm font-bold flex-shrink-0">
                          {index + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-xs sm:text-sm md:text-base truncate">
                            {algorithm.replace('_', ' ')}
                          </div>
                          <Badge variant="outline" className="text-xs mt-1">
                            {modelType}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Feature Selection Results */}
            {featureSelection && (
              <Card>
                <CardHeader className="pb-3 sm:pb-4">
                  <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                    <Target className="h-4 w-4 sm:h-5 sm:w-5" />
                    Feature Selection Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(featureSelection).map(([method, results]) => (
                      <div key={method} className="space-y-2">
                        <h4 className="font-medium text-sm sm:text-base">Method {method.replace('method_', '')}</h4>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
                          {results.selected_features?.slice(0, 10).map((feature, i) => (
                            <Badge key={i} variant="secondary" className="text-xs truncate">
                              {feature.length > 12 ? feature.substring(0, 12) + '...' : feature}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Data Quality Indicator */}
            {recommendations?.data_characteristics && (
              <Card>
                <CardHeader className="pb-3 sm:pb-4">
                  <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                    <TrendingUp className="h-4 w-4 sm:h-5 sm:w-5" />
                    Dataset Quality Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm sm:text-base">Dataset Size</span>
                      <Badge variant={
                        recommendations.data_characteristics.n_samples > 10000 ? 'default' :
                        recommendations.data_characteristics.n_samples > 1000 ? 'secondary' : 'destructive'
                      }>
                        {recommendations.data_characteristics.n_samples > 10000 ? 'Large' :
                         recommendations.data_characteristics.n_samples > 1000 ? 'Medium' : 'Small'}
                      </Badge>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm sm:text-base">Sample-to-Feature Ratio</span>
                      <Badge variant={
                        recommendations.data_characteristics.sample_to_feature_ratio > 10 ? 'default' :
                        recommendations.data_characteristics.sample_to_feature_ratio > 5 ? 'secondary' : 'destructive'
                      }>
                        {recommendations.data_characteristics.sample_to_feature_ratio.toFixed(1)}:1
                      </Badge>
                    </div>
                    
                    {recommendations.data_characteristics.n_classes && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm sm:text-base">Number of Classes</span>
                        <Badge variant={
                          recommendations.data_characteristics.n_classes <= 10 ? 'default' :
                          recommendations.data_characteristics.n_classes <= 50 ? 'secondary' : 'destructive'
                        }>
                          {recommendations.data_characteristics.n_classes}
                        </Badge>
                      </div>
                    )}
                    
                    <div className="mt-4 p-3 bg-gray-50 rounded">
                      <div className="text-xs sm:text-sm text-gray-600">
                        {recommendations.data_characteristics.n_samples > 10000 ? 
                          "✅ Large dataset - Advanced algorithms recommended" :
                          recommendations.data_characteristics.n_samples > 1000 ?
                          "⚠️ Medium dataset - Good for most algorithms" :
                          "⚠️ Small dataset - Simple algorithms recommended"
                        }
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-3">
              <Button variant="outline" onClick={() => setStep(1)} className="w-full sm:w-auto">
                Back
              </Button>
              <Button 
                onClick={startBatchTraining}
                disabled={!recommendations?.recommended_algorithms}
                className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 h-10 sm:h-11"
              >
                <Zap className="h-4 w-4 mr-2" />
                <span className="text-sm sm:text-base">Start Automated Training</span>
              </Button>
            </div>
          </div>
        )}

        {/* Step 3: Training Progress */}
        {step === 3 && (
          <div className="space-y-4 sm:space-y-6">
            <Card>
              <CardHeader className="pb-3 sm:pb-4">
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <Layers className="h-4 w-4 sm:h-5 sm:w-5" />
                  Batch Training Progress
                </CardTitle>
                <CardDescription className="text-xs sm:text-sm">
                  Training multiple models automatically to find the best performer
                </CardDescription>
              </CardHeader>
              <CardContent>
                {batchTraining ? (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 sm:h-5 sm:w-5 animate-spin text-purple-600" />
                      <span className="text-sm sm:text-base">Training models in progress...</span>
                    </div>
                    <Progress value={33} className="w-full" />
                    <div className="text-xs sm:text-sm text-muted-foreground">
                      This may take several minutes depending on your data size and selected algorithms.
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Alert>
                      <CheckCircle className="h-4 w-4" />
                      <AlertDescription className="text-sm">
                        Batch training completed! {trainedModels.filter(m => m.status === 'queued').length} models were queued for training.
                      </AlertDescription>
                    </Alert>

                    {trainedModels.length > 0 && (
                      <div className="space-y-3">
                        <h4 className="font-medium text-sm sm:text-base">Training Results:</h4>
                        <div className="space-y-2">
                          {trainedModels.map((result, index) => (
                            <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                              <div className="flex items-center gap-3 flex-1 min-w-0">
                                <Badge variant={result.status === 'queued' ? 'default' : 'destructive'} className="text-xs">
                                  {result.status}
                                </Badge>
                                <span className="font-medium text-sm sm:text-base truncate">{result.algorithm?.replace('_', ' ')}</span>
                              </div>
                              {result.model_id && (
                                <Badge variant="outline" className="text-xs ml-2 flex-shrink-0">ID: {result.model_id}</Badge>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex flex-col sm:flex-row gap-3">
                      <Button variant="outline" onClick={resetDialog} className="w-full sm:w-auto">
                        Start New AutoML
                      </Button>
                      <Button onClick={() => setIsOpen(false)} className="w-full sm:w-auto">
                        Close
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Tips for AutoML */}
            <Card>
              <CardHeader className="pb-3 sm:pb-4">
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <Info className="h-4 w-4 sm:h-5 sm:w-5 text-blue-600" />
                  AutoML Tips
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-xs sm:text-sm">
                  <p>• Models are trained with automatic hyperparameter tuning</p>
                  <p>• Cross-validation is used to ensure robust performance estimates</p>
                  <p>• You can compare models after training to find the best performer</p>
                  <p>• Consider data preprocessing and feature engineering for better results</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Advanced Configuration */}
        <Card>
          <CardHeader className="pb-3 sm:pb-4">
            <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
              <FlaskConical className="h-4 w-4 sm:h-5 sm:w-5" />
              <span className="flex-1 min-w-0 truncate">Advanced Configuration</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                className="text-xs sm:text-sm"
              >
                {showAdvancedOptions ? 'Hide' : 'Show'} Options
              </Button>
            </CardTitle>
          </CardHeader>
          {showAdvancedOptions && (
            <CardContent className="space-y-3 sm:space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
                <div className="space-y-2">
                  <Label htmlFor="max_samples" className="text-xs sm:text-sm">Max Samples (for large datasets)</Label>
                  <Select 
                    value={advancedConfig.max_samples.toString()}
                    onValueChange={(value) => setAdvancedConfig({...advancedConfig, max_samples: parseInt(value)})}
                  >
                    <SelectTrigger className="h-9 sm:h-10 md:h-11">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="10000">10,000 (Fast)</SelectItem>
                      <SelectItem value="50000">50,000 (Balanced)</SelectItem>
                      <SelectItem value="100000">100,000 (Comprehensive)</SelectItem>
                      <SelectItem value="500000">500,000 (Full)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="test_size" className="text-xs sm:text-sm">Test Size</Label>
                  <Select 
                    value={advancedConfig.test_size.toString()}
                    onValueChange={(value) => setAdvancedConfig({...advancedConfig, test_size: parseFloat(value)})}
                  >
                    <SelectTrigger className="h-9 sm:h-10 md:h-11">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.1">10%</SelectItem>
                      <SelectItem value="0.2">20%</SelectItem>
                      <SelectItem value="0.3">30%</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="cv_folds" className="text-xs sm:text-sm">Cross-Validation Folds</Label>
                  <Select 
                    value={advancedConfig.cv_folds.toString()}
                    onValueChange={(value) => setAdvancedConfig({...advancedConfig, cv_folds: parseInt(value)})}
                  >
                    <SelectTrigger className="h-9 sm:h-10 md:h-11">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="3">3-Fold</SelectItem>
                      <SelectItem value="5">5-Fold</SelectItem>
                      <SelectItem value="10">10-Fold</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="scaling" className="text-xs sm:text-sm">Feature Scaling</Label>
                  <Select 
                    value={advancedConfig.scaling}
                    onValueChange={(value) => setAdvancedConfig({...advancedConfig, scaling: value})}
                  >
                    <SelectTrigger className="h-9 sm:h-10 md:h-11">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="standard">Standard (Z-score)</SelectItem>
                      <SelectItem value="minmax">MinMax (0-1)</SelectItem>
                      <SelectItem value="none">None</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              {/* Add Auto-Binning Option - Will help with the error in test_automl.py */}
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-3">
                <div className="flex justify-between items-center">
                  <div className="space-y-1">
                    <Label htmlFor="auto_binning" className="text-sm flex items-center gap-1.5">
                      <span>Auto-Binning for Continuous Features</span>
                      <div className="relative group">
                        <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                        <div className="absolute bottom-full left-0 transform -translate-y-2 w-64 bg-popover text-popover-foreground p-2 rounded shadow-lg text-xs z-50 hidden group-hover:block">
                          Automatically converts numeric columns with many unique values into categorical bins for classification tasks
                        </div>
                      </div>
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Solves <span className="font-mono text-blue-600 dark:text-blue-400">"too many unique values"</span> errors
                    </p>
                  </div>
                  <Switch
                    id="auto_binning"
                    checked={advancedConfig.auto_binning}
                    onCheckedChange={(checked) => setAdvancedConfig({...advancedConfig, auto_binning: checked})}
                  />
                </div>
              </div>
              
              <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 lg:gap-6">
                <div className="flex items-center space-x-2 sm:space-x-3">
                  <input 
                    type="checkbox" 
                    id="enable_sampling"
                    checked={advancedConfig.enable_sampling}
                    onChange={(e) => setAdvancedConfig({...advancedConfig, enable_sampling: e.target.checked})}
                    className="h-4 w-4 sm:h-5 sm:w-5 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                  />
                  <Label htmlFor="enable_sampling" className="text-xs sm:text-sm cursor-pointer">
                    Enable Smart Sampling
                  </Label>
                </div>
                
                <div className="flex items-center space-x-2 sm:space-x-3">
                  <input 
                    type="checkbox" 
                    id="optimize_memory"
                    checked={advancedConfig.optimize_memory}
                    onChange={(e) => setAdvancedConfig({...advancedConfig, optimize_memory: e.target.checked})}
                    className="h-4 w-4 sm:h-5 sm:w-5 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                  />
                  <Label htmlFor="optimize_memory" className="text-xs sm:text-sm cursor-pointer">
                    Optimize Memory Usage
                  </Label>
                </div>
                
                <div className="flex items-center space-x-2 sm:space-x-3">
                  <input 
                    type="checkbox" 
                    id="handle_missing"
                    checked={advancedConfig.handle_missing}
                    onChange={(e) => setAdvancedConfig({...advancedConfig, handle_missing: e.target.checked})}
                    className="h-4 w-4 sm:h-5 sm:w-5 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                  />
                  <Label htmlFor="handle_missing" className="text-xs sm:text-sm cursor-pointer">
                    Handle Missing Values
                  </Label>
                </div>
              </div>
              
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription className="text-xs sm:text-sm">
                  <strong>Smart Sampling:</strong> Automatically reduces dataset size for faster training while maintaining data quality.
                  <br />
                  <strong>Memory Optimization:</strong> Uses efficient algorithms and data structures for large datasets.
                </AlertDescription>
              </Alert>
            </CardContent>
          )}
        </Card>
      </DialogContent>
    </Dialog>
  )
}
