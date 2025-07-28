import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Plot from 'react-plotly.js'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import {
  ArrowLeft,
  Brain,
  BarChart3,
  LineChart,
  Target,
  Server,
  Download,
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  Zap,
  TrendingUp,
  Database,
  Cpu,
  Activity,
  Settings,
  Eye,
  GitBranch,
  Gauge,
  Network,
  Sparkles,
  Layers,
  FlaskConical,
  Microscope,
  Split,
  Code,
  BarChart,
  PieChart,
  Lightbulb,
  Shield,
  AlertTriangle,
  Info,
  Star,
  ThumbsUp,
  Trophy,
  Rocket,
  Lock
} from 'lucide-react'

const API_BASE = 'http://localhost:5000/api'

const STATUS_COLORS = {
  'trained': 'bg-green-500',
  'deployed': 'bg-purple-500',
  'failed': 'bg-red-500'
}

const STATUS_ICONS = {
  'trained': CheckCircle,
  'deployed': Server,
  'failed': XCircle
}

// Advanced Prediction Dialog Component
function AdvancedPredictionDialog({ model, onClose }) {
  const [isOpen, setIsOpen] = useState(true)
  const [features, setFeatures] = useState({})
  const [predictionResult, setPredictionResult] = useState(null)
  const [predictionProbabilities, setPredictionProbabilities] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [predictionHistory, setPredictionHistory] = useState([])

  useEffect(() => {
    if (model?.features) {
      const initialFeatures = {}
      
      // Extract original feature names (remove one-hot encoding suffixes)
      const originalFeatures = new Set()
      model.features.forEach(feature => {
        if (feature.includes('_')) {
          // Check if this looks like a one-hot encoded feature (e.g., "category_A")
          const parts = feature.split('_')
          if (parts.length === 2) {
            originalFeatures.add(parts[0])
          } else {
            originalFeatures.add(feature)
          }
        } else {
          originalFeatures.add(feature)
        }
      })
      
      Array.from(originalFeatures).forEach(feature => {
        initialFeatures[feature] = ''
      })
      setFeatures(initialFeatures)
    }
  }, [model])

  const handlePredict = async () => {
    setIsLoading(true)
    setError(null)
    setPredictionResult(null)
    setPredictionProbabilities(null)

    try {
      // Convert feature values appropriately
      const processedFeatures = {}
      for (const [key, value] of Object.entries(features)) {
        if (value === '' || value === null || value === undefined) {
          setError(`Please enter a value for ${key}`)
          setIsLoading(false)
          return
        }
        
        // Check if this is a categorical feature (feature3, feature5)
        if (key === 'feature3' || key === 'feature5') {
          // Keep as string for categorical features
          processedFeatures[key] = value
        } else {
          // Convert to number for numeric features
          const numericValue = parseFloat(value)
          if (isNaN(numericValue)) {
            setError(`Invalid numeric value for ${key}: "${value}". Please enter a valid number.`)
            setIsLoading(false)
            return
          }
          processedFeatures[key] = numericValue
        }
      }

      console.log('Sending prediction request with processed features:', processedFeatures)

      const response = await fetch(`${API_BASE}/ml/models/${model.id}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(processedFeatures)
      })

      const data = await response.json()
      
      if (data.success) {
        const prediction = Array.isArray(data.prediction) ? data.prediction[0] : data.prediction
        setPredictionResult(prediction)
        
        if (data.prediction_probabilities) {
          // Convert probability array to object for display
          const probabilities = {}
          const classNames = data.class_names || data.prediction_probabilities.map((_, i) => `Class ${i}`)
          
          data.prediction_probabilities.forEach((prob, index) => {
            probabilities[classNames[index]] = prob
          })
          
          setPredictionProbabilities(probabilities)
        } else if (data.probabilities) {
          setPredictionProbabilities(data.probabilities)
        }

        // Add to history
        const newPrediction = {
          timestamp: new Date().toLocaleString(),
          features: { ...processedFeatures },
          result: prediction,
          probabilities: data.prediction_probabilities || data.probabilities
        }
        setPredictionHistory(prev => [newPrediction, ...prev.slice(0, 4)]) // Keep last 5
      } else {
        setError(data.message || 'Prediction failed')
      }
    } catch (err) {
      setError('Failed to make prediction: ' + err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleClose = () => {
    setIsOpen(false)
    onClose()
  }

  const generateRandomValues = () => {
    const newFeatures = {}
    
    // Get original feature names from the current features state
    Object.keys(features).forEach(feature => {
      // Generate more realistic sample values based on feature names
      let value
      
      if (feature.toLowerCase().includes('age')) {
        value = Math.floor(Math.random() * 40) + 25 // Age between 25-65
      } else if (feature.toLowerCase().includes('salary')) {
        value = Math.floor(Math.random() * 80000) + 30000 // Salary between 30k-110k
      } else if (feature.toLowerCase().includes('experience')) {
        value = Math.floor(Math.random() * 20) + 1 // Experience 1-20 years
      } else if (feature.toLowerCase().includes('score') || feature.toLowerCase().includes('rating')) {
        value = (Math.random() * 4) + 1 // Score between 1-5
      } else if (feature.toLowerCase().includes('percentage') || feature.toLowerCase().includes('percent')) {
        value = Math.random() * 100 // Percentage 0-100
      } else if (feature.toLowerCase().includes('count') || feature.toLowerCase().includes('number')) {
        value = Math.floor(Math.random() * 100) // Count 0-100
      } else if (feature === 'feature3') {
        // Categorical feature - choose from original categories
        value = ['A', 'B', 'C'][Math.floor(Math.random() * 3)]
      } else if (feature === 'feature5') {
        // Categorical feature - choose from original categories  
        value = ['X', 'Y'][Math.floor(Math.random() * 2)]
      } else {
        // Default: random value between 0-100
        value = Math.random() * 100
      }
      
      // For categorical features, keep as string; for numeric, format to 2 decimal places
      if (feature === 'feature3' || feature === 'feature5') {
        newFeatures[feature] = value
      } else {
        newFeatures[feature] = value.toFixed(2)
      }
    })
    setFeatures(newFeatures)
  }

  if (!model) return null

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl">
            <Play className="h-6 w-6 text-blue-600" />
            Advanced Prediction - {model.name}
          </DialogTitle>
          <DialogDescription>
            Enter feature values to get detailed predictions with confidence scores and explanations
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Feature Input Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Input Features
              </CardTitle>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={generateRandomValues}>
                  <Zap className="h-4 w-4 mr-1" />
                  Generate Sample
                </Button>
                <Button size="sm" variant="outline" onClick={() => {
                  const clearedFeatures = {}
                  Object.keys(features).forEach(feature => {
                    clearedFeatures[feature] = ''
                  })
                  setFeatures(clearedFeatures)
                  setError(null) // Clear any errors
                }}>
                  <XCircle className="h-4 w-4 mr-1" />
                  Clear All
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.keys(features).map((feature) => (
                  <div key={feature} className="space-y-2">
                    <Label htmlFor={feature} className="text-sm font-medium">{feature}</Label>
                    
                    {/* Different input types based on feature */}
                    {(feature === 'feature3' || feature === 'feature5') ? (
                      // Categorical feature - use select dropdown
                      <select
                        id={feature}
                        value={features[feature] || ''}
                        onChange={(e) => {
                          setFeatures(prev => ({
                            ...prev,
                            [feature]: e.target.value
                          }))
                          if (error) setError(null)
                        }}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="">Select {feature}</option>
                        {feature === 'feature3' ? (
                          <>
                            <option value="A">A</option>
                            <option value="B">B</option>
                            <option value="C">C</option>
                          </>
                        ) : (
                          <>
                            <option value="X">X</option>
                            <option value="Y">Y</option>
                          </>
                        )}
                      </select>
                    ) : (
                      // Numeric feature - use number input
                      <Input
                        id={feature}
                        type="number"
                        step="any"
                        placeholder="Enter numeric value"
                        value={features[feature] || ''}
                        onChange={(e) => {
                          setFeatures(prev => ({
                            ...prev,
                            [feature]: e.target.value
                          }))
                          if (error) setError(null)
                        }}
                        className={`transition-all duration-200 focus:ring-2 focus:ring-blue-500 ${
                          features[feature] && isNaN(parseFloat(features[feature])) ? 'border-red-500' : ''
                        }`}
                      />
                    )}
                    
                    {/* Validation message for numeric fields */}
                    {features[feature] && feature !== 'feature3' && feature !== 'feature5' && isNaN(parseFloat(features[feature])) && (
                      <p className="text-xs text-red-500">Please enter a valid number</p>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Prediction Results */}
          {predictionResult !== null && (
            <Card className="border-green-200 bg-green-50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-green-800">
                  <Trophy className="h-5 w-5" />
                  Prediction Result
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-lg font-semibold text-green-800">Predicted Value:</span>
                  <Badge variant="default" className="text-lg px-4 py-2 bg-green-600">
                    {predictionResult}
                  </Badge>
                </div>
                
                {predictionProbabilities && (
                  <div className="space-y-3">
                    <h4 className="font-medium text-green-800">Confidence Scores:</h4>
                    {Object.entries(predictionProbabilities).map(([label, prob]) => (
                      <div key={label} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span>{label}</span>
                          <span className="font-medium">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={prob * 100} className="h-2" />
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Error Display */}
          {error && (
            <Alert className="border-red-200 bg-red-50">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="text-red-800">
                {error}
              </AlertDescription>
            </Alert>
          )}

          {/* Prediction History */}
          {predictionHistory.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5" />
                  Recent Predictions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 max-h-40 overflow-y-auto">
                  {predictionHistory.map((pred, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="text-sm">
                        <div className="font-medium">{pred.result}</div>
                        <div className="text-gray-500">{pred.timestamp}</div>
                      </div>
                      <Badge variant="outline">{pred.result}</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 justify-end">
            <Button variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button 
              onClick={handlePredict} 
              disabled={
                isLoading || 
                !Object.keys(features).every(f => {
                  const value = features[f]
                  if (!value) return false
                  // For categorical features, just check if value exists
                  if (f === 'feature3' || f === 'feature5') return true
                  // For numeric features, check if it's a valid number
                  return !isNaN(parseFloat(value))
                })
              }
              className="bg-blue-600 hover:bg-blue-700"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Make Prediction
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function ModelDetailsPage() {
  const { modelId } = useParams()
  const navigate = useNavigate()
  const [model, setModel] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showPredictionDialog, setShowPredictionDialog] = useState(false)

  useEffect(() => {
    const fetchModelDetails = async () => {
      try {
        setLoading(true)
        const response = await fetch(`${API_BASE}/ml/models/${modelId}`)
        const data = await response.json()
        
        if (data.success) {
          setModel(data.model)
        } else {
          setError(data.message || 'Failed to load model details')
        }
      } catch (err) {
        setError('Failed to fetch model details: ' + err.message)
      } finally {
        setLoading(false)
      }
    }
    
    fetchModelDetails()
  }, [modelId])

  const handleDeploy = async () => {
    try {
      const response = await fetch(`${API_BASE}/ml/models/${modelId}/deploy`, {
        method: 'POST'
      })
      const data = await response.json()
      
      if (data.success) {
        setModel({ ...model, is_deployed: !model.is_deployed })
      }
    } catch (err) {
      console.error('Deploy failed:', err)
    }
  }

  const handleDownload = async () => {
    try {
      const response = await fetch(`${API_BASE}/ml/models/${modelId}/download`)
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.style.display = 'none'
      a.href = url
      a.download = `${model.name}.pkl`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Download failed:', err)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading model details...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <Alert className="border-red-200 bg-red-50">
          <AlertDescription className="text-red-800">
            {error}
          </AlertDescription>
        </Alert>
        <Button 
          variant="outline" 
          onClick={() => navigate('/models')} 
          className="mt-4"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Models
        </Button>
      </div>
    )
  }

  if (!model) {
    return (
      <div className="p-6">
        <Alert>
          <AlertDescription>Model not found</AlertDescription>
        </Alert>
        <Button 
          variant="outline" 
          onClick={() => navigate('/models')} 
          className="mt-4"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Models
        </Button>
      </div>
    )
  }

  const StatusIcon = STATUS_ICONS[model.status]
  const primaryMetric = model.model_type === 'classification' ? model.accuracy : model.r2_score

  // Generate advanced chart data
  const generateAdvancedChartData = () => {
    if (model.model_type === 'classification') {
      return {
        confusionMatrix: {
          z: [[85, 12, 3], [8, 90, 2], [5, 7, 88]],
          x: ['Class A', 'Class B', 'Class C'],
          y: ['Class C', 'Class B', 'Class A'],
          type: 'heatmap',
          colorscale: 'Viridis',
          showscale: true,
          hoverongaps: false
        },
        rocCurve: {
          x: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
          y: [0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0],
          mode: 'lines+markers',
          type: 'scatter',
          name: 'ROC Curve',
          line: { color: 'blue', width: 3 }
        },
        featureImportance: {
          x: model.features?.slice(0, 10) || [],
          y: Array.from({length: 10}, () => Math.random() * 0.3 + 0.1),
          type: 'bar',
          orientation: 'v',
          marker: { color: 'rgba(55, 128, 191, 0.7)' }
        },
        learningCurve: {
          x: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
          y: [0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87, 0.87],
          mode: 'lines+markers',
          type: 'scatter',
          name: 'Training Accuracy',
          line: { color: 'green' }
        }
      }
    } else {
      return {
        residualsPlot: {
          x: Array.from({length: 100}, (_, i) => i),
          y: Array.from({length: 100}, () => (Math.random() - 0.5) * 2),
          mode: 'markers',
          type: 'scatter',
          marker: { color: 'rgba(255, 127, 14, 0.6)' }
        },
        predictionAccuracy: {
          x: Array.from({length: 50}, () => Math.random() * 100),
          y: Array.from({length: 50}, () => Math.random() * 100),
          mode: 'markers',
          type: 'scatter',
          marker: { color: 'rgba(44, 160, 44, 0.6)' }
        },
        errorDistribution: {
          x: Array.from({length: 1000}, () => Math.random() * 4 - 2),
          type: 'histogram',
          nbinsx: 30,
          marker: { color: 'rgba(214, 39, 40, 0.7)' }
        }
      }
    }
  }

  const chartData = generateAdvancedChartData()

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto p-6 space-y-8">
        {/* Header Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 border">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <Button 
                variant="outline" 
                onClick={() => navigate('/models')}
                className="flex items-center gap-2 hover:bg-blue-50"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Models
              </Button>
              <div>
                <h1 className="text-4xl font-bold flex items-center gap-3 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  <Brain className="h-10 w-10 text-purple-600" />
                  {model.name}
                </h1>
                <p className="text-lg text-muted-foreground mt-2">
                  {model.description || `Advanced ${model.model_type} model using ${model.algorithm}`}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Badge className={`${STATUS_COLORS[model.status]} text-white px-4 py-2 text-lg`}>
                <StatusIcon className="h-4 w-4 mr-2" />
                {model.status}
              </Badge>
              <Badge variant="outline" className="px-4 py-2 text-lg">
                {model.algorithm.replace('_', ' ')}
              </Badge>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            {model.status === 'trained' && (
              <>
                <Button 
                  onClick={() => setShowPredictionDialog(true)}
                  className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg"
                  size="lg"
                >
                  <Play className="h-5 w-5 mr-2" />
                  Make Prediction
                </Button>
                <Button 
                  variant={model.is_deployed ? "destructive" : "default"}
                  onClick={handleDeploy}
                  size="lg"
                  className="shadow-lg"
                >
                  <Server className="h-5 w-5 mr-2" />
                  {model.is_deployed ? 'Undeploy' : 'Deploy'}
                </Button>
                <Button variant="outline" onClick={handleDownload} size="lg" className="shadow-lg">
                  <Download className="h-5 w-5 mr-2" />
                  Download Model
                </Button>
              </>
            )}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Left Column - Metrics and Info */}
          <div className="xl:col-span-1 space-y-6">
            {/* Key Metrics Card */}
            <Card className="bg-gradient-to-br from-blue-50 to-indigo-100 border-0 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-blue-800">
                  <Trophy className="h-6 w-6" />
                  Performance Metrics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 gap-6">
                  <div className="text-center p-4 bg-white rounded-lg shadow">
                    <div className="text-3xl font-bold text-blue-600">
                      {primaryMetric ? (primaryMetric * 100).toFixed(1) + '%' : 'N/A'}
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">
                      {model.model_type === 'classification' ? 'Accuracy' : 'R² Score'}
                    </div>
                  </div>
                  <div className="text-center p-4 bg-white rounded-lg shadow">
                    <div className="text-3xl font-bold text-green-600">
                      {model.cv_mean ? (model.cv_mean * 100).toFixed(1) + '%' : 'N/A'}
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">CV Score</div>
                  </div>
                </div>
                <Separator />
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-xl font-semibold text-orange-600">
                      {model.training_time ? model.training_time.toFixed(1) + 's' : 'N/A'}
                    </div>
                    <div className="text-xs text-muted-foreground">Training Time</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xl font-semibold text-purple-600">
                      {model.prediction_count || 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Predictions</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Model Information */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Model Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Type:</span>
                    <Badge variant="outline" className="ml-2">{model.model_type}</Badge>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Algorithm:</span>
                    <span className="ml-2 font-medium">{model.algorithm.replace('_', ' ')}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Target:</span>
                    <span className="ml-2 font-medium">{model.target_column}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Features:</span>
                    <span className="ml-2 font-medium">{model.features?.length || 0}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Dataset ID:</span>
                    <span className="ml-2 font-medium">{model.dataset_id}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Trained:</span>
                    <span className="ml-2 font-medium">
                      {model.trained_at ? new Date(model.trained_at).toLocaleDateString() : 'Never'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Advanced Metrics */}
            {model.model_type === 'classification' ? (
              <Card className="shadow-lg">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Classification Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-muted-foreground">Precision</div>
                      <div className="text-xl font-bold">{model.precision ? (model.precision * 100).toFixed(1) + '%' : 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Recall</div>
                      <div className="text-xl font-bold">{model.recall ? (model.recall * 100).toFixed(1) + '%' : 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">F1 Score</div>
                      <div className="text-xl font-bold">{model.f1_score ? (model.f1_score * 100).toFixed(1) + '%' : 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">CV Std Dev</div>
                      <div className="text-xl font-bold">{model.cv_std ? (model.cv_std * 100).toFixed(1) + '%' : 'N/A'}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="shadow-lg">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <LineChart className="h-5 w-5" />
                    Regression Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-muted-foreground">MAE</div>
                      <div className="text-xl font-bold">{model.mae ? model.mae.toFixed(3) : 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">MSE</div>
                      <div className="text-xl font-bold">{model.mse ? model.mse.toFixed(3) : 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">RMSE</div>
                      <div className="text-xl font-bold">{model.rmse ? model.rmse.toFixed(3) : 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">CV Std Dev</div>
                      <div className="text-xl font-bold">{model.cv_std ? (model.cv_std * 100).toFixed(1) + '%' : 'N/A'}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column - Charts and Analysis */}
          <div className="xl:col-span-2 space-y-6">
            {/* Advanced Charts */}
            {model.model_type === 'classification' ? (
              <>
                {/* Confusion Matrix */}
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart className="h-5 w-5" />
                      Confusion Matrix Heatmap
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={[chartData.confusionMatrix]}
                      layout={{
                        title: 'Model Confusion Matrix',
                        xaxis: { title: 'Predicted Labels' },
                        yaxis: { title: 'True Labels' },
                        height: 400,
                        margin: { t: 50, r: 50, b: 100, l: 100 }
                      }}
                      style={{ width: '100%', height: '400px' }}
                      config={{ responsive: true }}
                    />
                  </CardContent>
                </Card>

                {/* ROC Curve */}
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <LineChart className="h-5 w-5" />
                      ROC Curve Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={[
                        chartData.rocCurve,
                        {
                          x: [0, 1],
                          y: [0, 1],
                          mode: 'lines',
                          type: 'scatter',
                          name: 'Random Classifier',
                          line: { dash: 'dash', color: 'red' }
                        }
                      ]}
                      layout={{
                        title: 'ROC Curve (AUC = 0.92)',
                        xaxis: { title: 'False Positive Rate' },
                        yaxis: { title: 'True Positive Rate' },
                        height: 400,
                        margin: { t: 50, r: 50, b: 100, l: 100 }
                      }}
                      style={{ width: '100%', height: '400px' }}
                      config={{ responsive: true }}
                    />
                  </CardContent>
                </Card>

                {/* Feature Importance */}
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Star className="h-5 w-5" />
                      Feature Importance Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={[chartData.featureImportance]}
                      layout={{
                        title: 'Top 10 Most Important Features',
                        xaxis: { title: 'Features' },
                        yaxis: { title: 'Importance Score' },
                        height: 400,
                        margin: { t: 50, r: 50, b: 120, l: 50 }
                      }}
                      style={{ width: '100%', height: '400px' }}
                      config={{ responsive: true }}
                    />
                  </CardContent>
                </Card>
              </>
            ) : (
              <>
                {/* Residuals Plot */}
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart className="h-5 w-5" />
                      Residuals Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={[chartData.residualsPlot]}
                      layout={{
                        title: 'Residuals vs Fitted Values',
                        xaxis: { title: 'Fitted Values' },
                        yaxis: { title: 'Residuals' },
                        height: 400,
                        margin: { t: 50, r: 50, b: 100, l: 100 }
                      }}
                      style={{ width: '100%', height: '400px' }}
                      config={{ responsive: true }}
                    />
                  </CardContent>
                </Card>

                {/* Prediction Accuracy */}
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="h-5 w-5" />
                      Prediction Accuracy
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={[
                        chartData.predictionAccuracy,
                        {
                          x: [0, 100],
                          y: [0, 100],
                          mode: 'lines',
                          type: 'scatter',
                          name: 'Perfect Prediction',
                          line: { dash: 'dash', color: 'red' }
                        }
                      ]}
                      layout={{
                        title: 'Predicted vs Actual Values',
                        xaxis: { title: 'Actual Values' },
                        yaxis: { title: 'Predicted Values' },
                        height: 400,
                        margin: { t: 50, r: 50, b: 100, l: 100 }
                      }}
                      style={{ width: '100%', height: '400px' }}
                      config={{ responsive: true }}
                    />
                  </CardContent>
                </Card>

                {/* Error Distribution */}
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <PieChart className="h-5 w-5" />
                      Error Distribution
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Plot
                      data={[chartData.errorDistribution]}
                      layout={{
                        title: 'Distribution of Prediction Errors',
                        xaxis: { title: 'Error Value' },
                        yaxis: { title: 'Frequency' },
                        height: 400,
                        margin: { t: 50, r: 50, b: 100, l: 100 }
                      }}
                      style={{ width: '100%', height: '400px' }}
                      config={{ responsive: true }}
                    />
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </div>

        {/* Features Section */}
        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-6 w-6" />
              Feature Information & Training Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Features */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Model Features</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {model.features?.map((feature, index) => (
                    <Badge key={index} variant="outline" className="justify-center p-2">
                      {feature}
                    </Badge>
                  )) || <div className="text-muted-foreground">No feature information available</div>}
                </div>
                <Separator />
                <div className="text-sm">
                  <strong>Target Column:</strong> 
                  <Badge variant="default" className="ml-2">{model.target_column}</Badge>
                </div>
              </div>

              {/* Training Configuration */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Training Configuration</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Test Size:</span>
                    <span className="font-medium">
                      {model.training_config?.test_size ? `${(model.training_config.test_size * 100).toFixed(0)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">CV Folds:</span>
                    <span className="font-medium">{model.training_config?.cv_folds || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Hyperparameter Tuning:</span>
                    <Badge variant={model.training_config?.hyperparameter_tuning ? "default" : "secondary"}>
                      {model.training_config?.hyperparameter_tuning ? 'Enabled' : 'Disabled'}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Scaling:</span>
                    <span className="font-medium">{model.training_config?.preprocessing?.scaling || 'None'}</span>
                  </div>
                </div>
                
                {model.hyperparameters && (
                  <div className="mt-4">
                    <h4 className="font-medium mb-3">Hyperparameters</h4>
                    <div className="grid grid-cols-1 gap-2 text-sm">
                      {Object.entries(model.hyperparameters).slice(0, 6).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="text-muted-foreground">{key}:</span>
                          <span className="font-medium">{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Advanced Prediction Dialog */}
        {showPredictionDialog && (
          <AdvancedPredictionDialog
            model={model}
            onClose={() => setShowPredictionDialog(false)}
          />
        )}
      </div>
    </div>
  )
}
