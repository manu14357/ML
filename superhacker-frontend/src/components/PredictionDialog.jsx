import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Target, Brain, Zap, Loader2, CheckCircle, XCircle } from 'lucide-react'

const API_BASE = 'http://localhost:5000/api'

export function PredictionDialog({ model, isOpen, onClose }) {
  const [predictionData, setPredictionData] = useState({})
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async () => {
    if (!model || !model.features) return

    setLoading(true)
    setError(null)

    try {
      // Validate that all features are provided
      const missingFeatures = model.features.filter(feature => 
        predictionData[feature] === undefined || predictionData[feature] === ''
      )

      if (missingFeatures.length > 0) {
        setError(`Missing values for: ${missingFeatures.join(', ')}`)
        setLoading(false)
        return
      }

      const response = await fetch(`${API_BASE}/ml/models/${model.id}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: predictionData
        })
      })

      const data = await response.json()

      if (data.success) {
        setPredictions(data)
      } else {
        setError(data.message)
      }
    } catch (err) {
      setError('Prediction failed: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (feature, value) => {
    setPredictionData(prev => ({
      ...prev,
      [feature]: value
    }))
  }

  const resetForm = () => {
    setPredictionData({})
    setPredictions(null)
    setError(null)
  }

  if (!model) return null

  return (
    <Dialog open={isOpen} onOpenChange={(open) => {
      if (!open) {
        onClose()
        resetForm()
      }
    }}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Target className="h-6 w-6 text-blue-600" />
            Make Prediction - {model.name}
          </DialogTitle>
          <DialogDescription>
            Enter feature values to get predictions from your trained model
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Model Info */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Model Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <span className="text-sm text-muted-foreground">Algorithm</span>
                  <div className="font-medium">{model.algorithm.replace('_', ' ')}</div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Type</span>
                  <div className="font-medium">{model.model_type}</div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Accuracy</span>
                  <div className="font-medium">
                    {model.model_type === 'classification' 
                      ? (model.accuracy ? (model.accuracy * 100).toFixed(1) + '%' : 'N/A')
                      : (model.r2_score ? model.r2_score.toFixed(3) : 'N/A')
                    }
                  </div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Features</span>
                  <div className="font-medium">{model.features?.length || 0}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Feature Input Form */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Feature Values</CardTitle>
              <CardDescription>
                Enter values for all {model.features?.length || 0} features
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {model.features?.map((feature, index) => (
                  <div key={index} className="space-y-2">
                    <Label htmlFor={feature}>{feature}</Label>
                    <Input
                      id={feature}
                      type="number"
                      step="any"
                      placeholder="Enter value"
                      value={predictionData[feature] || ''}
                      onChange={(e) => handleInputChange(feature, parseFloat(e.target.value))}
                    />
                  </div>
                ))}
              </div>

              {error && (
                <Alert className="mt-4" variant="destructive">
                  <XCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <div className="flex gap-2 mt-6">
                <Button 
                  onClick={handlePredict} 
                  disabled={loading}
                  className="flex-1"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Predicting...
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Make Prediction
                    </>
                  )}
                </Button>
                <Button variant="outline" onClick={resetForm}>
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Prediction Results */}
          {predictions && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  Prediction Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Primary Prediction */}
                  <div className="text-center p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border">
                    <div className="text-3xl font-bold text-blue-600 mb-2">
                      {Array.isArray(predictions.predictions) 
                        ? predictions.predictions[0] 
                        : predictions.predictions
                      }
                    </div>
                    <div className="text-muted-foreground">
                      Predicted {model.model_type === 'classification' ? 'Class' : 'Value'}
                    </div>
                  </div>

                  {/* Classification Probabilities */}
                  {predictions.prediction_probabilities && (
                    <div>
                      <h4 className="font-medium mb-3">Class Probabilities</h4>
                      <div className="space-y-2">
                        {predictions.prediction_probabilities[0].map((prob, index) => (
                          <div key={index} className="flex items-center gap-3">
                            <Badge variant="outline" className="w-16 justify-center">
                              Class {index}
                            </Badge>
                            <div className="flex-1 bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-blue-600 h-2 rounded-full" 
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium w-16 text-right">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Model Info */}
                  <div className="text-sm text-muted-foreground">
                    <p>Model: {predictions.model_info.name}</p>
                    <p>Algorithm: {predictions.model_info.algorithm.replace('_', ' ')}</p>
                    <p>Type: {predictions.model_info.model_type}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
