import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Checkbox } from '@/components/ui/checkbox'
import { 
  Brain, Plus, Play, TrendingUp, Settings, Eye, Download, Trash2, 
  BarChart3, Target, Clock, Database, Cpu, CheckCircle, XCircle, 
  Loader2, RefreshCw, Zap, Search, Filter, SortAsc, SortDesc,
  LineChart, PieChart, Activity, Layers, GitBranch,
  AlertTriangle, Info, ChevronRight, ChevronDown, Sparkles,
  Gauge, TrendingDown, FlaskConical, Microscope, 
  Network, Shuffle, Split, Merge, FileBarChart, Square,
  CheckSquare, Trash, ExternalLink,  ArrowUpDown, Columns3,
  ArrowUp, ArrowDown, Star, Archive, MoreHorizontal, ChevronLeft
} from 'lucide-react'
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, PieChart as RechartsPieChart, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'
import { AutoMLDialog } from './AutoMLDialog'

const API_BASE = '/api'

const STATUS_COLORS = {
  'trained': 'bg-green-500', 
  'deployed': 'bg-purple-500',
  'failed': 'bg-red-500',
  'training': 'bg-blue-500',
  'pending': 'bg-yellow-500',
  'draft': 'bg-gray-500'
}

const STATUS_ICONS = {
  'trained': CheckCircle,
  'failed': XCircle,
  'training': Loader2,
  'pending': Clock,
  'deployed': CheckCircle,
  'draft': Settings
}

const ALGORITHM_TYPES = {
  classification: [
    { value: 'random_forest', label: 'Random Forest', icon: Layers },
    { value: 'gradient_boosting', label: 'Gradient Boosting', icon: TrendingUp },
    { value: 'xgboost', label: 'XGBoost', icon: Zap },
    { value: 'lightgbm', label: 'LightGBM', icon: FlaskConical },
    { value: 'catboost', label: 'CatBoost', icon: Microscope },
    { value: 'logistic_regression', label: 'Logistic Regression', icon: LineChart },
    { value: 'svm', label: 'Support Vector Machine', icon: Network },
    { value: 'knn', label: 'K-Nearest Neighbors', icon: Target },
    { value: 'naive_bayes', label: 'Naive Bayes', icon: Brain },
    { value: 'decision_tree', label: 'Decision Tree', icon: GitBranch },
    { value: 'extra_trees', label: 'Extra Trees', icon: Layers },
    { value: 'adaboost', label: 'AdaBoost', icon: TrendingUp }
  ],
  regression: [
    { value: 'random_forest', label: 'Random Forest', icon: Layers },
    { value: 'gradient_boosting', label: 'Gradient Boosting', icon: TrendingUp },
    { value: 'xgboost', label: 'XGBoost', icon: Zap },
    { value: 'lightgbm', label: 'LightGBM', icon: FlaskConical },
    { value: 'catboost', label: 'CatBoost', icon: Microscope },
    { value: 'linear_regression', label: 'Linear Regression', icon: LineChart },
    { value: 'ridge', label: 'Ridge Regression', icon: BarChart3 },
    { value: 'lasso', label: 'Lasso Regression', icon: Activity },
    { value: 'svr', label: 'Support Vector Regression', icon: Network },
    { value: 'knn', label: 'K-Nearest Neighbors', icon: Target },
    { value: 'decision_tree', label: 'Decision Tree', icon: GitBranch },
    { value: 'extra_trees', label: 'Extra Trees', icon: Layers },
    { value: 'elastic_net', label: 'Elastic Net', icon: Network },
    { value: 'sgd', label: 'SGD Regressor', icon: Zap }
  ]
}

// Advanced Model Card Component with Selection Support
function ModelCard({ model, onDelete, onNavigate, isSelected, onSelect, showCheckbox = false }) {
  const StatusIcon = STATUS_ICONS[model.status] || Activity // Fallback icon
  const primaryMetric = model.model_type === 'classification' ? model.accuracy : model.r2_score
  const algorithmInfo = ALGORITHM_TYPES[model.model_type]?.find(alg => alg.value === model.algorithm)
  const AlgorithmIcon = algorithmInfo?.icon || Brain
  
  // Format model name for display
  const formatModelName = (name) => {
    // Check if it follows a pattern like algorithm_type_date_time
    const dateTimePattern = /(\d{8})_(\d{6})/;
    const underscoreSegments = name.split('_');
    
    // If the name has date-time pattern and multiple underscores
    if (dateTimePattern.test(name) && underscoreSegments.length >= 3) {
      // Extract algorithm part (possibly multiple segments)
      const algorithmPart = underscoreSegments
        .slice(0, -2)
        .map(segment => segment.charAt(0).toUpperCase() + segment.slice(1))
        .join(' ');
      
      // Extract and format date if present
      const dateMatch = name.match(dateTimePattern);
      if (dateMatch) {
        const dateStr = dateMatch[1];
        const timeStr = dateMatch[2];
        
        // Format date as YYYY-MM-DD
        const formattedDate = `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`;
        
        // Format time as HH:MM
        const formattedTime = `${timeStr.slice(0, 2)}:${timeStr.slice(2, 4)}`;
        
        return {
          displayName: algorithmPart,
          shortName: algorithmPart,
          formattedDateTime: `${formattedDate} ${formattedTime}`
        };
      }
    }
    
    // Default handling for other name patterns
    return {
      displayName: name,
      shortName: name.length > 24 ? 
        (name.includes('_') ? 
          name.split('_').slice(0, 2).join(' ') + 
          (name.split('_').length > 2 ? '...' : '') : 
          name.substring(0, 20) + '...') : 
        name,
      formattedDateTime: null
    };
  };
  
  const formattedName = formatModelName(model.name);

  return (
    <Card 
      className={`hover:shadow-xl transition-all duration-300 border-l-4 cursor-pointer group relative 
                 h-full max-h-[800px] flex flex-col overflow-hidden backdrop-blur-sm
                 ${isSelected ? 'ring-2 ring-blue-500 bg-blue-50/50' : 'hover:bg-accent/30'}`}
      style={{
        borderLeftColor: (STATUS_COLORS[model.status] || 'bg-gray-500').replace('bg-', '#'),
        background: 'linear-gradient(to bottom right, rgba(255,255,255,0.95), rgba(255,255,255,0.85))'
      }}
      onClick={() => onNavigate(model.id)}
    >
      {/* Selection Checkbox */}
      {showCheckbox && (
        <div className="absolute top-2 right-2 sm:top-3 sm:right-3 z-10" onClick={(e) => e.stopPropagation()}>
          <Checkbox
            checked={isSelected}
            onCheckedChange={onSelect}
            className="h-4 w-4 sm:h-5 sm:w-5"
          />
        </div>
      )}

      <CardHeader className="pb-2 sm:pb-3 flex-shrink-0 px-3 sm:px-4 pt-3 sm:pt-4 
                           border-b border-gray-100 bg-gradient-to-r from-white to-gray-50/80">
        <div className="space-y-2.5">
          {/* Model Name with Icon - Enhanced for long names */}
          <CardTitle className="text-sm sm:text-base md:text-lg flex items-start gap-2 
                               group-hover:text-blue-600 transition-colors leading-tight">
            <AlgorithmIcon className="h-4 w-4 sm:h-5 sm:w-5 text-purple-600 flex-shrink-0 mt-1" />
            <div className="flex flex-col">
              <div className="font-semibold break-words line-clamp-1 hover:line-clamp-none 
                            transition-all duration-300">
                {formattedName.displayName}
              </div>
              {formattedName.formattedDateTime && (
                <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-1">
                  <Clock className="h-3 w-3" /> {formattedName.formattedDateTime}
                </div>
              )}
              {!formattedName.formattedDateTime && formattedName.displayName !== formattedName.shortName && (
                <div className="text-xs text-muted-foreground mt-0.5">
                  {formattedName.shortName}
                </div>
              )}
            </div>
          </CardTitle>
          
          {/* Status Badges - Enhanced with animation */}
          <div className="flex flex-wrap items-center gap-2">
            <Badge className={`${STATUS_COLORS[model.status] || 'bg-gray-500'} text-white shadow-sm 
                              text-xs font-medium px-2 py-1 flex items-center gap-1 
                              ${model.status === 'training' ? 'animate-pulse' : ''}`}>
              <StatusIcon className={`h-3 w-3 ${model.status === 'training' ? 'animate-spin' : ''}`} />
              {model.status}
            </Badge>
            {model.status === 'trained' && (
              <Badge variant="outline" className="text-green-600 border-green-600 bg-green-50 
                                               text-xs font-medium px-2 py-1 flex items-center gap-1
                                               shadow-sm">
                <Star className="h-3 w-3 fill-current" />
                Ready
              </Badge>
            )}
          </div>
          
          {/* Algorithm and Type Badges */}
          <div className="flex flex-wrap items-center gap-1 sm:gap-2">
            <Badge variant="secondary" className="text-xs bg-gradient-to-r from-purple-100 to-blue-100 
                                                px-2 py-1 rounded-full font-medium">
              {algorithmInfo?.label || model.algorithm.replace('_', ' ')}
            </Badge>
            <Badge variant="outline" className="text-xs text-muted-foreground">
              {model.model_type}
            </Badge>
          </div>
          
          {/* Description with hover expand */}
          <CardDescription className="text-xs sm:text-sm leading-relaxed max-w-full overflow-hidden">
            <div className="line-clamp-2 group-hover:line-clamp-none transition-all duration-300">
              {model.description || `${model.model_type} model targeting ${model.target_column}`}
            </div>
          </CardDescription>
        </div>
      </CardHeader>
      
      <CardContent className="flex-grow flex flex-col px-3 sm:px-4 pb-4 sm:pb-5 space-y-4 overflow-hidden">
        {/* Enhanced Performance Metrics Grid with Hover Effects */}
        <div className="grid grid-cols-2 gap-2 sm:gap-3">
          <div className="text-center p-2 sm:p-3 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg 
                          hover:from-blue-100 hover:to-blue-200 transition-colors group cursor-help">
            <div className="text-base sm:text-lg font-bold text-blue-700 mb-1 leading-tight 
                           group-hover:scale-110 transition-transform">
              {primaryMetric ? (primaryMetric * 100).toFixed(1) + '%' : 'N/A'}
            </div>
            <div className="text-xs text-blue-600 font-medium leading-tight">
              {model.model_type === 'classification' ? 'Accuracy' : 'R² Score'}
            </div>
          </div>
          
          <div className="text-center p-2 sm:p-3 bg-gradient-to-br from-green-50 to-green-100 rounded-lg 
                          hover:from-green-100 hover:to-green-200 transition-colors group cursor-help">
            <div className="text-base sm:text-lg font-bold text-green-700 mb-1 leading-tight 
                           group-hover:scale-110 transition-transform">
              {model.cv_mean ? (model.cv_mean * 100).toFixed(1) + '%' : 'N/A'}
            </div>
            <div className="text-xs text-green-600 font-medium leading-tight">CV Score</div>
          </div>
          
          <div className="text-center p-2 sm:p-3 bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg 
                          hover:from-orange-100 hover:to-orange-200 transition-colors group cursor-help">
            <div className="text-base sm:text-lg font-bold text-orange-700 mb-1 leading-tight 
                           group-hover:scale-110 transition-transform">
              {model.training_time ? model.training_time.toFixed(1) + 's' : 'N/A'}
            </div>
            <div className="text-xs text-orange-600 font-medium leading-tight">Train Time</div>
          </div>
          
          <div className="text-center p-2 sm:p-3 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg 
                          hover:from-purple-100 hover:to-purple-200 transition-colors group cursor-help">
            <div className="text-base sm:text-lg font-bold text-purple-700 mb-1 leading-tight 
                           group-hover:scale-110 transition-transform">
              {model.prediction_count || 0}
            </div>
            <div className="text-xs text-purple-600 font-medium leading-tight">Predictions</div>
          </div>
        </div>

        {/* Enhanced Model Info */}
        <div className="flex flex-col gap-2 text-xs sm:text-sm text-muted-foreground">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1 min-w-0">
              <Database className="h-3 w-3 sm:h-4 sm:w-4 flex-shrink-0" />
              <span className="font-medium">{model.features?.length || 0}</span>
              <span>features</span>
            </div>
            <div className="flex items-center gap-1 min-w-0 group relative">
              <Target className="h-3 w-3 sm:h-4 sm:w-4 flex-shrink-0" />
              <span className="font-medium truncate max-w-[100px] hover:underline cursor-help">
                {model.target_column?.length > 15 ? 
                  model.target_column.substring(0, 12) + '...' : 
                  model.target_column}
              </span>
              {model.target_column?.length > 15 && (
                <div className="absolute bottom-full left-0 mb-1 hidden group-hover:block bg-gray-900 text-white 
                              text-xs rounded px-2 py-1 max-w-[200px] z-10 whitespace-normal break-words">
                  {model.target_column}
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1 justify-center">
            <Clock className="h-3 w-3 sm:h-4 sm:w-4 flex-shrink-0" />
            <span className="text-center text-xs">
              {model.trained_at ? new Date(model.trained_at).toLocaleDateString() : 'Not trained'}
            </span>
          </div>
        </div>

        {/* Advanced Action Buttons */}
        <div className="flex gap-2 mt-auto pt-3 border-t border-gray-100" onClick={(e) => e.stopPropagation()}>
          <Button 
            size="sm" 
            variant="outline"
            className="bg-gradient-to-r from-blue-600 to-purple-600 text-white border-0 
                      hover:from-blue-700 hover:to-purple-700 flex-1 text-xs sm:text-sm
                      h-9 sm:h-10 shadow-sm hover:shadow-md transition-all"
            onClick={(e) => {
              e.stopPropagation()
              onNavigate(model.id)
            }}
          >
            <Eye className="h-3 w-3 sm:h-4 sm:w-4 mr-1 sm:mr-2" />
            <span>View Details</span>
          </Button>
          
          <Button 
            size="sm" 
            variant="outline"
            className="border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300 
                      px-3 sm:px-4 text-xs sm:text-sm h-9 sm:h-10 hover:shadow-sm transition-all"
            onClick={(e) => {
              e.stopPropagation()
              onDelete(model)
            }}
          >
            <Trash2 className="h-3 w-3 sm:h-4 sm:w-4" />
            <span className="hidden sm:inline ml-1">Delete</span>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// Training Dialog Component
function TrainingDialog({ onTrain }) {
  const [isOpen, setIsOpen] = useState(false)
  const [datasets, setDatasets] = useState([])
  const [datasetsLoading, setDatasetsLoading] = useState(false)
  const [datasetsError, setDatasetsError] = useState(null)
  const [suggestedTargetColumns, setSuggestedTargetColumns] = useState([])
  const [analyzingColumns, setAnalyzingColumns] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [trainingError, setTrainingError] = useState(null)
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    dataset_id: '',
    algorithm: '',
    model_type: 'classification',
    target_column: '',
    training_config: {
      test_size: 0.2,
      cv_folds: 5,
      hyperparameter_tuning: true,
      cross_validation: true,
      preprocessing: {
        scaling: 'standard',
        handle_missing: true,
        encoding_strategy: 'onehot'
      }
    }
  })
  const [selectedDataset, setSelectedDataset] = useState(null)

  // When dataset changes, suggest target columns
  const handleDatasetChange = (value) => {
    setFormData({ ...formData, dataset_id: value, target_column: '' })
    const dataset = datasets.find(d => d.id.toString() === value)
    
    // Process dataset to extract column names
    if (dataset) {
      // Extract column names from columns_info structure
      let columns = []
      if (dataset.columns_info) {
        if (typeof dataset.columns_info === 'object' && !Array.isArray(dataset.columns_info)) {
          // Dictionary format: {column_name: {...}}
          columns = Object.keys(dataset.columns_info)
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
        }
      }
      
      // Fallback: try to extract from data_types if columns_info failed
      if (columns.length === 0 && dataset.data_types && typeof dataset.data_types === 'object') {
        columns = Object.keys(dataset.data_types)
      }
      
      // Create enhanced dataset object with processed columns
      const enhancedDataset = {
        ...dataset,
        columns: columns,
        row_count: dataset.rows_count, // Map backend field to frontend expectation
        column_count: dataset.columns_count // Map backend field to frontend expectation
      }
      
      setSelectedDataset(enhancedDataset)
      setSuggestedTargetColumns([])
      suggestTargetColumns(enhancedDataset)
    } else {
      setSelectedDataset(null)
      setSuggestedTargetColumns([])
    }
  }
  
  // Fetch datasets when dialog opens
  const fetchDatasets = async () => {
    setDatasetsLoading(true)
    setDatasetsError(null)
    try {
      const response = await fetch(`${API_BASE}/data/datasets`)
      const data = await response.json()
      if (data.success) {
        setDatasets(data.datasets)
      } else {
        setDatasetsError('Failed to load datasets')
      }
    } catch (error) {
      console.error('Error fetching datasets:', error)
      setDatasetsError('Error loading datasets')
    } finally {
      setDatasetsLoading(false)
    }
  }

  // Effect to fetch datasets when dialog opens
  useEffect(() => {
    if (isOpen && datasets.length === 0) {
      fetchDatasets()
    }
    
    // Reset error state when dialog opens/closes
    if (!isOpen) {
      setTrainingError(null)
      setShowTechnicalDetails(false)
    }
  }, [isOpen, datasets.length])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsTraining(true)
    setTrainingError(null)
    
    try {
      await onTrain(formData)
      setIsOpen(false)
      setFormData({
        name: '',
        description: '',
        dataset_id: '',
        algorithm: '',
        model_type: 'classification',
        target_column: '',
        training_config: {
          test_size: 0.2,
          cv_folds: 5,
          hyperparameter_tuning: true,
          cross_validation: true,
          preprocessing: {
            scaling: 'standard',
            handle_missing: true,
            encoding_strategy: 'onehot'
          }
        }
      })
    } catch (error) {
      console.error('Training failed:', error)
      
      // Handle error and set it for display
      if (error.message) {
        setTrainingError({
          message: 'Training failed: ' + error.message,
          technicalDetails: error.stack || error.toString(),
          suggestions: ['Check your dataset and parameters', 'Try a different algorithm']
        })
      } else {
        setTrainingError({
          message: 'Training failed due to an unknown error',
          technicalDetails: 'No detailed information available',
          suggestions: ['Check your dataset and parameters', 'Try a different algorithm']
        })
      }
    } finally {
      setIsTraining(false)
    }
  }

  const availableAlgorithms = ALGORITHM_TYPES[formData.model_type] || []

  // Suggest target columns based on dataset and model type
  const suggestTargetColumns = async (dataset) => {
    setAnalyzingColumns(true)
    try {
      const response = await fetch(`${API_BASE}/ml/automl/suggest-target-columns`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: dataset.id,
          model_type: formData.model_type
        })
      })
      const data = await response.json()
      
      if (data.success) {
        setSuggestedTargetColumns(data.suggestions)
        // Auto-select the top suggestion if available and no target is selected
        if (data.suggestions.length > 0 && !formData.target_column) {
          setFormData(prev => ({...prev, target_column: data.suggestions[0].column_name}))
        }
      } else {
        // Fallback to simple heuristic-based suggestions
        const suggestions = getHeuristicTargetSuggestions(dataset)
        setSuggestedTargetColumns(suggestions)
        if (suggestions.length > 0 && !formData.target_column) {
          setFormData(prev => ({...prev, target_column: suggestions[0].column_name}))
        }
      }
    } catch (error) {
      console.error('Error suggesting target columns:', error)
      // Fallback to heuristic suggestions
      const suggestions = getHeuristicTargetSuggestions(dataset)
      setSuggestedTargetColumns(suggestions)
      if (suggestions.length > 0 && !formData.target_column) {
        setFormData(prev => ({...prev, target_column: suggestions[0].column_name}))
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
    
    const patterns = commonTargetPatterns[formData.model_type] || commonTargetPatterns.classification
    
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

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button className="w-full sm:w-auto">
          <Plus className="h-4 w-4 mr-2" />
          <span className="hidden sm:inline">Train New Model</span>
          <span className="sm:hidden">Train</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[95vw] sm:max-w-[90vw] md:max-w-4xl lg:max-w-5xl 
                               max-h-[90vh] overflow-y-auto mx-auto w-full
                               macbook-13:max-w-[85vw] macbook-13:max-h-[85vh]">
        <DialogHeader className="pb-4">
          <DialogTitle className="text-lg sm:text-xl md:text-2xl">Train New ML Model</DialogTitle>
          <DialogDescription className="text-sm sm:text-base">
            Configure and train a new machine learning model with advanced options
          </DialogDescription>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
            {/* Basic Information */}
            <Card className="h-fit">
              <CardHeader className="pb-3 sm:pb-4">
                <CardTitle className="text-base sm:text-lg">Basic Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 sm:space-y-4">
                <div>
                  <Label htmlFor="name" className="text-sm sm:text-base">Model Name</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    placeholder="My Awesome Model"
                    className="mt-1 sm:mt-2"
                    required
                  />
                </div>
                
                <div>
                  <Label htmlFor="description" className="text-sm sm:text-base">Description</Label>
                  <Textarea
                    id="description"
                    value={formData.description}
                    onChange={(e) => setFormData({...formData, description: e.target.value})}
                    placeholder="Describe your model..."
                    rows={3}
                    className="mt-1 sm:mt-2"
                  />
                </div>
                
                <div>
                  <Label htmlFor="model_type" className="text-sm sm:text-base">Model Type</Label>
                  <Select 
                    value={formData.model_type} 
                    onValueChange={(value) => setFormData({...formData, model_type: value, algorithm: ''})}
                  >
                    <SelectTrigger className="mt-1 sm:mt-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="classification">Classification</SelectItem>
                      <SelectItem value="regression">Regression</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Dataset and Algorithm */}
            <Card className="h-fit">
              <CardHeader className="pb-3 sm:pb-4">
                <CardTitle className="text-base sm:text-lg">Dataset & Algorithm</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 sm:space-y-4">
                <div>
                  <Label htmlFor="dataset" className="text-sm sm:text-base">Dataset</Label>
                  <Select 
                    value={formData.dataset_id} 
                    onValueChange={handleDatasetChange}
                    disabled={datasetsLoading}
                  >
                    <SelectTrigger className="mt-1 sm:mt-2">
                      <SelectValue placeholder={
                        datasetsLoading 
                          ? "Loading datasets..." 
                          : datasetsError 
                            ? "Error loading datasets" 
                            : "Select a dataset"
                      } />
                    </SelectTrigger>
                    <SelectContent className="max-h-[200px] sm:max-h-[300px] overflow-y-auto">
                      {datasetsLoading ? (
                        <div className="flex items-center justify-center p-4">
                          <Loader2 className="h-4 w-4 animate-spin mr-2" />
                          <span className="text-sm">Loading datasets...</span>
                        </div>
                      ) : datasetsError ? (
                        <div className="flex flex-col items-center p-4 space-y-2">
                          <AlertTriangle className="h-4 w-4 text-red-500" />
                          <span className="text-sm text-red-500">{datasetsError}</span>
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={fetchDatasets}
                            className="mt-2"
                          >
                            <RefreshCw className="h-3 w-3 mr-1" />
                            Retry
                          </Button>
                        </div>
                      ) : datasets.length === 0 ? (
                        <div className="text-center p-4 text-muted-foreground">
                          <span className="text-sm">No datasets available</span>
                        </div>
                      ) : (
                        datasets.map(dataset => (
                          <SelectItem key={dataset.id} value={dataset.id.toString()}>
                            <span className="truncate">{dataset.name}</span>
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                </div>

                {selectedDataset && (
                  <div>
                    <Label htmlFor="target_column" className="text-sm sm:text-base">Target Column</Label>
                    <Select 
                      value={formData.target_column} 
                      onValueChange={(value) => setFormData({...formData, target_column: value})}
                      disabled={analyzingColumns}
                    >
                      <SelectTrigger className="mt-1 sm:mt-2">
                        <SelectValue placeholder={
                          analyzingColumns
                            ? "Analyzing columns..."
                            : suggestedTargetColumns.length > 0
                              ? `Suggested: ${suggestedTargetColumns[0].column_name}`
                              : "Select target column"
                        } />
                      </SelectTrigger>
                      <SelectContent className="max-h-[200px] sm:max-h-[300px] overflow-y-auto">
                        {analyzingColumns ? (
                          <div className="flex items-center justify-center p-4">
                            <Loader2 className="h-4 w-4 animate-spin mr-2" />
                            <span className="text-sm">Analyzing columns...</span>
                          </div>
                        ) : (
                          <>
                            {suggestedTargetColumns.length > 0 && (
                              <>
                                <div className="px-3 py-1 text-xs text-muted-foreground">Suggested</div>
                                {suggestedTargetColumns.map((col, index) => (
                                  <SelectItem key={`suggested-${col.column_name}-${index}`} value={col.column_name}>
                                    <div className="flex flex-col">
                                      <span className="font-semibold text-blue-700">{col.column_name}</span>
                                      {col.reason && (
                                        <span className="text-xs text-muted-foreground">({col.reason})</span>
                                      )}
                                    </div>
                                  </SelectItem>
                                ))}
                                <Separator className="my-1" />
                              </>
                            )}
                            {selectedDataset.columns && selectedDataset.columns.length > 0 ? (
                              selectedDataset.columns
                                .filter(column => !suggestedTargetColumns.some(suggested => suggested.column_name === column))
                                .map((column, index) => (
                                  <SelectItem key={`column-${column}-${index}`} value={column}>
                                    <span className="truncate">{column}</span>
                                  </SelectItem>
                                ))
                            ) : (
                              <div className="text-center p-4 text-muted-foreground">
                                <AlertTriangle className="h-4 w-4 mx-auto mb-2" />
                                <div className="text-sm">No columns found in dataset</div>
                                <div className="text-xs">Please check the dataset format</div>
                              </div>
                            )}
                          </>
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div>
                  <Label htmlFor="algorithm" className="text-sm sm:text-base">Algorithm</Label>
                  <Select 
                    value={formData.algorithm} 
                    onValueChange={(value) => setFormData({...formData, algorithm: value})}
                  >
                    <SelectTrigger className="mt-1 sm:mt-2">
                      <SelectValue placeholder="Select algorithm" />
                    </SelectTrigger>
                    <SelectContent className="max-h-[200px] sm:max-h-[300px] overflow-y-auto">
                      {availableAlgorithms.map(algo => {
                        const Icon = algo.icon
                        return (
                          <SelectItem key={algo.value} value={algo.value}>
                            <div className="flex items-center gap-2">
                              <Icon className="h-4 w-4" />
                              <span className="truncate">{algo.label}</span>
                            </div>
                          </SelectItem>
                        )
                      })}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Advanced Configuration */}
          <Card>
            <CardHeader className="pb-3 sm:pb-4">
              <CardTitle className="text-base sm:text-lg">Advanced Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="training" className="w-full">
                <TabsList className="grid w-full grid-cols-3 h-auto">
                  <TabsTrigger value="training" className="text-xs sm:text-sm px-2 sm:px-4">Training</TabsTrigger>
                  <TabsTrigger value="preprocessing" className="text-xs sm:text-sm px-2 sm:px-4">Preprocessing</TabsTrigger>
                  <TabsTrigger value="validation" className="text-xs sm:text-sm px-2 sm:px-4">Validation</TabsTrigger>
                </TabsList>
                
                <TabsContent value="training" className="space-y-3 sm:space-y-4 mt-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <Label htmlFor="hyperparameter_tuning" className="text-sm">Hyperparameter Tuning</Label>
                      <Switch
                        id="hyperparameter_tuning"
                        checked={formData.training_config.hyperparameter_tuning}
                        onCheckedChange={(checked) => 
                          setFormData({
                            ...formData,
                            training_config: {
                              ...formData.training_config,
                              hyperparameter_tuning: checked
                            }
                          })
                        }
                      />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <Label htmlFor="cross_validation" className="text-sm">Cross Validation</Label>
                      <Switch
                        id="cross_validation"
                        checked={formData.training_config.cross_validation}
                        onCheckedChange={(checked) => 
                          setFormData({
                            ...formData,
                            training_config: {
                              ...formData.training_config,
                              cross_validation: checked
                            }
                          })
                        }
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div>
                      <Label className="text-sm">Test Size: {(formData.training_config.test_size * 100).toFixed(0)}%</Label>
                      <Slider
                        value={[formData.training_config.test_size]}
                        onValueChange={([value]) => 
                          setFormData({
                            ...formData,
                            training_config: {
                              ...formData.training_config,
                              test_size: value
                            }
                          })
                        }
                        max={0.5}
                        min={0.1}
                        step={0.05}
                        className="mt-2"
                      />
                    </div>
                    
                    <div>
                      <Label className="text-sm">CV Folds: {formData.training_config.cv_folds}</Label>
                      <Slider
                        value={[formData.training_config.cv_folds]}
                        onValueChange={([value]) => 
                          setFormData({
                            ...formData,
                            training_config: {
                              ...formData.training_config,
                              cv_folds: value
                            }
                          })
                        }
                        max={10}
                        min={3}
                        step={1}
                        className="mt-2"
                      />
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="preprocessing" className="space-y-3 sm:space-y-4 mt-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="scaling" className="text-sm">Feature Scaling</Label>
                      <Select 
                        value={formData.training_config.preprocessing.scaling} 
                        onValueChange={(value) => 
                          setFormData({
                            ...formData,
                            training_config: {
                              ...formData.training_config,
                              preprocessing: {
                                ...formData.training_config.preprocessing,
                                scaling: value
                              }
                            }
                          })
                        }
                      >
                        <SelectTrigger className="mt-1 sm:mt-2">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          <SelectItem value="standard">Standard Scaling</SelectItem>
                          <SelectItem value="minmax">Min-Max Scaling</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div>
                      <Label htmlFor="encoding" className="text-sm">Encoding Strategy</Label>
                      <Select 
                        value={formData.training_config.preprocessing.encoding_strategy} 
                        onValueChange={(value) => 
                          setFormData({
                            ...formData,
                            training_config: {
                              ...formData.training_config,
                              preprocessing: {
                                ...formData.training_config.preprocessing,
                                encoding_strategy: value
                              }
                            }
                          })
                        }
                      >
                        <SelectTrigger className="mt-1 sm:mt-2">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="label">Label Encoding</SelectItem>
                          <SelectItem value="onehot">One-Hot Encoding</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <Label htmlFor="handle_missing" className="text-sm">Handle Missing Values</Label>
                    <Switch
                      id="handle_missing"
                      checked={formData.training_config.preprocessing.handle_missing}
                      onCheckedChange={(checked) => 
                        setFormData({
                          ...formData,
                          training_config: {
                            ...formData.training_config,
                            preprocessing: {
                              ...formData.training_config.preprocessing,
                              handle_missing: checked
                            }
                          }
                        })
                      }
                    />
                  </div>
                </TabsContent>
                
                <TabsContent value="validation" className="space-y-3 sm:space-y-4 mt-4">
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription className="text-sm">
                      Validation settings help ensure your model generalizes well to new data.
                    </AlertDescription>
                  </Alert>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-sm">Validation Method</Label>
                      <Select defaultValue="holdout">
                        <SelectTrigger className="mt-1 sm:mt-2">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="holdout">Hold-out Validation</SelectItem>
                          <SelectItem value="kfold">K-Fold Cross Validation</SelectItem>
                          <SelectItem value="stratified">Stratified K-Fold</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <Label className="text-sm">Early Stopping</Label>
                      <Switch defaultChecked />
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Error Card with Technical Details Toggle */}
          {trainingError && (
            <Card className="border-red-300 bg-red-50 shadow-md mt-4">
              <CardContent className="pt-4 pb-4">
                <div className="space-y-3">
                  {/* Error Title and Icon */}
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div className="space-y-1 flex-1">
                      <h4 className="font-medium text-red-700">Training Failed</h4>
                      <p className="text-sm text-red-700">{trainingError.message}</p>
                    </div>
                  </div>
                  
                  {/* Suggestions */}
                  {trainingError.suggestions && trainingError.suggestions.length > 0 && (
                    <div className="pl-7">
                      <h5 className="text-sm font-medium text-red-700 mb-1">Suggestions:</h5>
                      <ul className="list-disc pl-5 text-sm text-red-600 space-y-1">
                        {trainingError.suggestions.map((suggestion, idx) => (
                          <li key={idx}>{suggestion}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {/* Technical Details Toggle */}
                  {trainingError.technicalDetails && (
                    <div className="pl-7 pt-1">
                      <Button 
                        variant="outline" 
                        size="sm"
                        className="text-xs h-7 border-red-300 text-red-700 hover:bg-red-100"
                        onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                      >
                        {showTechnicalDetails ? (
                          <>
                            <ChevronDown className="h-3.5 w-3.5 mr-1" />
                            Hide Technical Details
                          </>
                        ) : (
                          <>
                            <ChevronRight className="h-3.5 w-3.5 mr-1" />
                            Show Technical Details
                          </>
                        )}
                      </Button>
                      
                      {showTechnicalDetails && (
                        <div className="mt-2 p-2 bg-red-100 border border-red-200 rounded text-xs font-mono whitespace-pre-wrap text-red-800 max-h-32 overflow-y-auto">
                          {trainingError.technicalDetails}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          <div className="flex flex-col sm:flex-row justify-end gap-2 sm:gap-3 pt-4">
            <Button 
              type="button" 
              variant="outline" 
              onClick={() => setIsOpen(false)}
              className="w-full sm:w-auto order-2 sm:order-1"
            >
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={isTraining || !formData.name || !formData.dataset_id || !formData.algorithm}
              className="w-full sm:w-auto order-1 sm:order-2"
            >
              {isTraining ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start Training
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

// Main MLModels Component
export function MLModels() {
  const navigate = useNavigate()
  const [models, setModels] = useState([])
  const [statistics, setStatistics] = useState({})
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterType, setFilterType] = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')
  const [filterAlgorithm, setFilterAlgorithm] = useState('all')
  const [sortBy, setSortBy] = useState('created_at')
  const [sortOrder, setSortOrder] = useState('desc')
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  
  // Bulk selection state
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedModels, setSelectedModels] = useState(new Set())
  const [bulkDeleteLoading, setBulkDeleteLoading] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Fetch data
  const fetchModels = async () => {
    if (isRefreshing) return // Prevent multiple concurrent requests
    
    try {
      setIsRefreshing(true)
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: '12',
        sort_by: sortBy,
        sort_order: sortOrder
      })

      if (searchTerm) params.append('search', searchTerm)
      if (filterType !== 'all') params.append('type', filterType)
      if (filterStatus !== 'all') params.append('status', filterStatus)
      if (filterAlgorithm !== 'all') params.append('algorithm', filterAlgorithm)

      const response = await fetch(`${API_BASE}/ml/models?${params}`)
      const data = await response.json()
      
      if (data.success) {
        // Ensure models have unique IDs and remove duplicates
        const uniqueModels = data.models.filter((model, index, self) => 
          index === self.findIndex(m => m.id === model.id)
        )
        setModels(uniqueModels)
        setStatistics(data.statistics)
        setTotalPages(data.pagination.pages)
      }
    } catch (error) {
      console.error('Error fetching models:', error)
    } finally {
      setIsRefreshing(false)
    }
  }

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await fetchModels()
      setLoading(false)
    }
    loadData()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, sortBy, sortOrder, filterType, filterStatus, filterAlgorithm, searchTerm])

  // Bulk selection handlers
  const toggleSelectionMode = () => {
    setSelectionMode(!selectionMode)
    setSelectedModels(new Set())
  }

  const selectAllModels = () => {
    if (selectedModels.size === models.length) {
      setSelectedModels(new Set())
    } else {
      setSelectedModels(new Set(models.map(m => m.id)))
    }
  }

  const toggleModelSelection = (modelId) => {
    const newSelection = new Set(selectedModels)
    if (newSelection.has(modelId)) {
      newSelection.delete(modelId)
    } else {
      newSelection.add(modelId)
    }
    setSelectedModels(newSelection)
  }

  const bulkDeleteModels = async () => {
    if (selectedModels.size === 0) return
    
    const confirmed = confirm(`Are you sure you want to delete ${selectedModels.size} selected model(s)?`)
    if (!confirmed) return

    setBulkDeleteLoading(true)
    try {
      const deletePromises = Array.from(selectedModels).map(modelId =>
        fetch(`${API_BASE}/ml/models/${modelId}`, { method: 'DELETE' })
      )
      
      await Promise.all(deletePromises)
      await fetchModels()
      setSelectedModels(new Set())
      setSelectionMode(false)
    } catch (error) {
      console.error('Bulk delete error:', error)
      alert('Some deletions failed')
    } finally {
      setBulkDeleteLoading(false)
    }
  }

  // Individual handlers
  const handleTrain = async (formData) => {
    try {
      const response = await fetch(`${API_BASE}/ml/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const data = await response.json()
      
      if (data.success) {
        // Refresh models list after a small delay to ensure backend state is updated
        setTimeout(() => {
          fetchModels()
        }, 500)
        // Show success message
        alert('Model training completed successfully!')
        return true
      } else {
        // Show user-friendly error message
        const errorMessage = data.message || 'Training failed'
        const technicalError = data.technical_error
        
        if (technicalError && technicalError !== errorMessage) {
          // Show detailed error in console for debugging
          console.error('Technical error:', technicalError)
        }
        
        // Don't show alert here, let the dialog component handle it
        throw new Error(errorMessage)
      }
    } catch (error) {
      console.error('Training error:', error)
      throw error // Propagate error to be handled by dialog component
    }
  }

  const handleDelete = async (model) => {
    if (confirm(`Are you sure you want to delete "${model.name}"?`)) {
      try {
        const response = await fetch(`${API_BASE}/ml/models/${model.id}`, {
          method: 'DELETE'
        })
        const data = await response.json()
        
        if (data.success) {
          fetchModels()
        } else {
          alert(data.message)
        }
      } catch (error) {
        console.error('Delete error:', error)
        alert('Delete failed')
      }
    }
  }

  const handleNavigate = (modelId) => {
    if (!selectionMode) {
      navigate(`/models/${modelId}`)
    }
  }

  // Get unique algorithms for filter
  const availableAlgorithms = [...new Set(models.map(m => m.algorithm))]

  if (loading) {
    return (
      <div className="p-3 sm:p-4 md:p-6 space-y-4 sm:space-y-6 min-h-screen bg-background">
        <div className="flex items-center justify-center py-8 sm:py-12 md:py-16">
          <div className="text-center space-y-3 sm:space-y-4">
            <Loader2 className="h-8 w-8 sm:h-10 sm:w-10 md:h-12 md:w-12 animate-spin text-purple-600 mx-auto" />
            <div className="space-y-1 sm:space-y-2">
              <span className="text-base sm:text-lg md:text-xl font-medium">Loading ML Models...</span>
              <p className="text-xs sm:text-sm text-muted-foreground">
                Fetching your machine learning models and statistics
              </p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-2 sm:p-3 md:p-4 lg:p-6 space-y-3 sm:space-y-4 md:space-y-6 
                   min-h-screen bg-gradient-to-b from-background via-background to-background/80">
      {/* Enhanced Modern Header with Bulk Actions */}
      <div className="flex flex-col gap-3 sm:gap-4">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 sm:gap-4">
          <div className="min-w-0 flex-1">
            <h1 className="text-xl sm:text-2xl md:text-3xl font-bold tracking-tight 
                          flex items-center gap-2 mb-1 sm:mb-2 bg-gradient-to-r from-purple-700 to-blue-600 
                          bg-clip-text text-transparent">
              <Brain className="h-5 w-5 sm:h-6 sm:w-6 md:h-8 md:w-8 text-purple-600" />
              <span className="truncate">ML Models</span>
              {selectionMode && selectedModels.size > 0 && (
                <Badge variant="secondary" className="ml-2 text-xs sm:text-sm flex-shrink-0">
                  {selectedModels.size} selected
                </Badge>
              )}
            </h1>
            <p className="text-xs sm:text-sm md:text-base text-muted-foreground leading-tight">
              Train, deploy, and manage advanced machine learning models
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 w-full sm:w-auto">
            {selectionMode ? (
              // Bulk Action Mode
              <div className="flex flex-col sm:flex-row gap-2">
                <Button
                  variant="outline"
                  onClick={selectAllModels}
                  disabled={models.length === 0}
                  className="w-full sm:w-auto text-xs sm:text-sm"
                >
                  <CheckSquare className="h-3 w-3 sm:h-4 sm:w-4 mr-2" />
                  {selectedModels.size === models.length ? 'Deselect All' : 'Select All'}
                </Button>
                <Button
                  variant="destructive"
                  onClick={bulkDeleteModels}
                  disabled={selectedModels.size === 0 || bulkDeleteLoading}
                  className="w-full sm:w-auto text-xs sm:text-sm"
                >
                  {bulkDeleteLoading ? (
                    <Loader2 className="h-3 w-3 sm:h-4 sm:w-4 mr-2 animate-spin" />
                  ) : (
                    <Trash className="h-3 w-3 sm:h-4 sm:w-4 mr-2" />
                  )}
                  Delete Selected ({selectedModels.size})
                </Button>
                <Button 
                  variant="outline" 
                  onClick={toggleSelectionMode} 
                  className="w-full sm:w-auto text-xs sm:text-sm"
                >
                  Cancel
                </Button>
              </div>
            ) : (
              // Normal Mode
              <div className="flex flex-col sm:flex-row gap-2">
                <Button
                  variant="outline"
                  onClick={toggleSelectionMode}
                  disabled={models.length === 0}
                  className="w-full sm:w-auto text-xs sm:text-sm"
                >
                  <CheckSquare className="h-3 w-3 sm:h-4 sm:w-4 mr-2" />
                  <span className="hidden sm:inline">Select</span>
                  <span className="sm:hidden">Select Models</span>
                </Button>
                <TrainingDialog onTrain={handleTrain} />
                <AutoMLDialog onAutoTrain={fetchModels} />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced Statistics Dashboard */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 md:gap-6">
        <Card className="border-l-4 border-l-blue-500 h-full overflow-hidden relative group">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <CardContent className="pt-4 sm:pt-6 pb-4 sm:pb-6">
            <div className="text-center">
              <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-blue-600 mb-2 group-hover:scale-110 transition-transform">
                {statistics.total_models || 0}
              </div>
              <p className="text-sm sm:text-base text-muted-foreground flex items-center justify-center gap-2">
                <Database className="h-4 w-4 sm:h-5 sm:w-5" />
                Total Models
              </p>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-l-4 border-l-green-500 h-full overflow-hidden relative group">
          <div className="absolute inset-0 bg-gradient-to-r from-green-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <CardContent className="pt-4 sm:pt-6 pb-4 sm:pb-6">
            <div className="text-center">
              <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-green-600 mb-2 group-hover:scale-110 transition-transform">
                {statistics.trained_models || 0}
              </div>
              <p className="text-sm sm:text-base text-muted-foreground flex items-center justify-center gap-2">
                <CheckCircle className="h-4 w-4 sm:h-5 sm:w-5" />
                Trained Models
              </p>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-l-4 border-l-red-500 h-full overflow-hidden relative group">
          <div className="absolute inset-0 bg-gradient-to-r from-red-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <CardContent className="pt-4 sm:pt-6 pb-4 sm:pb-6">
            <div className="text-center">
              <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-red-600 mb-2 group-hover:scale-110 transition-transform">
                {statistics.failed_models || 0}
              </div>
              <p className="text-sm sm:text-base text-muted-foreground flex items-center justify-center gap-2">
                <XCircle className="h-4 w-4 sm:h-5 sm:w-5" />
                Failed Models
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Advanced Filters and Search */}
      <Card className="overflow-hidden border-t-4 border-t-purple-500">
        <CardContent className="pt-3 sm:pt-4 md:pt-6">
          <div className="flex flex-col gap-3 sm:gap-4">
            {/* Search Bar */}
            <div className="w-full">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 
                               h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
                <Input
                  placeholder="Search models by name, algorithm, or description..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8 sm:pl-10 text-sm sm:text-base h-9 sm:h-10 bg-background/80 border-muted transition-all focus:bg-background focus:border-purple-500"
                />
              </div>
            </div>
            
            {/* Filters Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-2 sm:gap-3">
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="w-full h-9 sm:h-10 text-xs sm:text-sm border-muted transition-all hover:border-purple-500">
                  <Filter className="h-3 w-3 sm:h-4 sm:w-4 mr-1 sm:mr-2 text-purple-500" />
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="classification">Classification</SelectItem>
                  <SelectItem value="regression">Regression</SelectItem>
                </SelectContent>
              </Select>

              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger className="w-full h-9 sm:h-10 text-xs sm:text-sm">
                  <Activity className="h-3 w-3 sm:h-4 sm:w-4 mr-1 sm:mr-2" />
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="trained">Trained</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                  <SelectItem value="training">Training</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                </SelectContent>
              </Select>

              <Select value={filterAlgorithm} onValueChange={setFilterAlgorithm}>
                <SelectTrigger className="w-full h-9 sm:h-10 text-xs sm:text-sm">
                  <Brain className="h-3 w-3 sm:h-4 sm:w-4 mr-1 sm:mr-2" />
                  <SelectValue placeholder="Algorithm" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Algorithms</SelectItem>
                  {availableAlgorithms.map((algorithm, index) => (
                    <SelectItem key={`algorithm-${algorithm}-${index}`} value={algorithm}>
                      <span className="truncate">{algorithm.replace('_', ' ')}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-full h-9 sm:h-10 text-xs sm:text-sm">
                  <ArrowUpDown className="h-3 w-3 sm:h-4 sm:w-4 mr-1 sm:mr-2" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="created_at">Created Date</SelectItem>
                  <SelectItem value="trained_at">Trained Date</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="accuracy">Accuracy</SelectItem>
                  <SelectItem value="training_time">Training Time</SelectItem>
                </SelectContent>
              </Select>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                title={`Sort ${sortOrder === 'asc' ? 'Descending' : 'Ascending'}`}
                className="h-9 sm:h-10 px-2 sm:px-3"
              >
                {sortOrder === 'asc' ? 
                  <ArrowUp className="h-3 w-3 sm:h-4 sm:w-4" /> : 
                  <ArrowDown className="h-3 w-3 sm:h-4 sm:w-4" />
                }
                <span className="hidden sm:inline ml-1">{sortOrder === 'asc' ? 'Asc' : 'Desc'}</span>
              </Button>

              <Button 
                variant="outline" 
                onClick={fetchModels} 
                title="Refresh"
                size="sm"
                className="h-9 sm:h-10 px-2 sm:px-3"
              >
                <RefreshCw className="h-3 w-3 sm:h-4 sm:w-4" />
                <span className="hidden sm:inline ml-1 sm:ml-2">Refresh</span>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Models Grid - Fixed 2 cards per row with better spacing */}
      {models.length > 0 ? (
        <div className="grid grid-cols-1 
                       sm:grid-cols-2 
                       gap-5 sm:gap-7 md:gap-8
                       auto-rows-fr">
          {models.map((model, index) => (
            <div key={`model-${model.id || index}-${model.name || 'unknown'}`} className="h-full">
              <ModelCard
                model={model}
                onDelete={handleDelete}
                onNavigate={handleNavigate}
                isSelected={selectedModels.has(model.id)}
                onSelect={() => toggleModelSelection(model.id)}
                showCheckbox={selectionMode}
              />
            </div>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="pt-8 pb-8 sm:pt-12 sm:pb-12">
            <div className="text-center space-y-3 sm:space-y-4 max-w-md mx-auto">
              <div className="mx-auto w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 
                             bg-muted rounded-full flex items-center justify-center">
                <Brain className="h-8 w-8 sm:h-10 sm:w-10 md:h-12 md:w-12 text-muted-foreground" />
              </div>
              <div className="space-y-2">
                <h3 className="text-base sm:text-lg font-medium">No ML Models Found</h3>
                <p className="text-sm sm:text-base text-muted-foreground px-4">
                  {searchTerm || filterType !== 'all' || filterStatus !== 'all' || filterAlgorithm !== 'all'
                    ? 'No models match your current filters. Try adjusting your search criteria.'
                    : 'Get started by training your first machine learning model.'}
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 justify-center pt-2">
                <TrainingDialog onTrain={handleTrain} />
                <AutoMLDialog onAutoTrain={fetchModels} />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex flex-col sm:flex-row justify-center items-center gap-3 sm:gap-4 pt-4">
          <div className="flex items-center gap-2 sm:gap-3">
            <Button
              variant="outline"
              onClick={() => setPage(page - 1)}
              disabled={page === 1}
              size="sm"
              className="h-8 sm:h-9 px-2 sm:px-3"
            >
              <ChevronLeft className="h-3 w-3 sm:h-4 sm:w-4" />
              <span className="hidden sm:inline ml-1">Previous</span>
            </Button>
            
            <div className="flex items-center gap-1 sm:gap-2">
              <span className="text-xs sm:text-sm text-muted-foreground px-2 sm:px-4 whitespace-nowrap">
                Page {page} of {totalPages}
              </span>
            </div>
            
            <Button
              variant="outline"
              onClick={() => setPage(page + 1)}
              disabled={page === totalPages}
              size="sm"
              className="h-8 sm:h-9 px-2 sm:px-3"
            >
              <span className="hidden sm:inline mr-1">Next</span>
              <ChevronRight className="h-3 w-3 sm:h-4 sm:w-4" />
            </Button>
          </div>
          
          {/* Page Numbers for larger screens */}
          <div className="hidden md:flex items-center gap-1">
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              const pageNum = i + 1
              return (
                <Button
                  key={pageNum}
                  variant={page === pageNum ? "default" : "outline"}
                  size="sm"
                  onClick={() => setPage(pageNum)}
                  className="h-8 w-8 p-0"
                >
                  {pageNum}
                </Button>
              )
            })}
            {totalPages > 5 && (
              <>
                <span className="text-muted-foreground px-1">...</span>
                <Button
                  variant={page === totalPages ? "default" : "outline"}
                  size="sm"
                  onClick={() => setPage(totalPages)}
                  className="h-8 w-8 p-0"
                >
                  {totalPages}
                </Button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

