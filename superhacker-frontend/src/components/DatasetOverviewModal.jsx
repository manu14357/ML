import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Database, X, Sparkles, ScatterChart, BarChart, LineChart, PieChart
} from 'lucide-react'

export function DatasetOverviewModal({ 
  selectedDataset, 
  columns, 
  showDatasetOverview, 
  setShowDatasetOverview, 
  onAutoGenerate 
}) {
  if (!showDatasetOverview || !selectedDataset) return null

  // Ensure proper column categorization
  const numericColumns = columns.filter(col => {
    return col.is_numeric || ['int64', 'float64', 'int32', 'float32', 'number'].includes(col.dtype?.toLowerCase?.() || col.dtype)
  })
  
  const categoricalColumns = columns.filter(col => {
    return col.is_categorical || ['object', 'category', 'string'].includes(col.dtype?.toLowerCase?.() || col.dtype)
  })
  
  const datetimeColumns = columns.filter(col => {
    return col.is_datetime || (col.dtype && col.dtype.toString().toLowerCase().includes('datetime'))
  })

  // Generate smarter chart recommendations based on column types
  const recommendations = []
  
  if (numericColumns.length >= 2) {
    recommendations.push({
      type: "scatter",
      icon: <ScatterChart className="h-4 w-4 text-purple-500" />,
      description: "Scatter plots for correlation analysis"
    })
    
    recommendations.push({
      type: "heatmap",
      icon: <Database className="h-4 w-4 text-orange-500" />,
      description: "Correlation heatmap for numeric variables"
    })
  }
  
  if (numericColumns.length >= 1) {
    recommendations.push({
      type: "histogram",
      icon: <BarChart className="h-4 w-4 text-blue-500" />,
      description: "Histograms for distribution analysis"
    })
    
    recommendations.push({
      type: "box",
      icon: <Database className="h-4 w-4 text-green-500" />,
      description: "Box plots for statistical summaries"
    })
  }
  
  if (datetimeColumns.length >= 1 && numericColumns.length >= 1) {
    recommendations.push({
      type: "line",
      icon: <LineChart className="h-4 w-4 text-green-500" />,
      description: "Time series analysis with date/time columns"
    })
  }
  
  if (categoricalColumns.length >= 1) {
    recommendations.push({
      type: "pie",
      icon: <PieChart className="h-4 w-4 text-orange-500" />,
      description: "Pie charts for category distribution"
    })
    
    recommendations.push({
      type: "bar",
      icon: <BarChart className="h-4 w-4 text-indigo-500" />,
      description: "Bar charts for category comparison"
    })
  }
  
  if (categoricalColumns.length >= 1 && numericColumns.length >= 1) {
    recommendations.push({
      type: "box",
      icon: <Database className="h-4 w-4 text-blue-500" />,
      description: "Box plots grouped by categories"
    })
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-y-auto m-4">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center">
                <Database className="h-5 w-5 mr-2" />
                Dataset Overview: {selectedDataset.name}
              </CardTitle>
              <CardDescription>
                Detailed analysis and column information
              </CardDescription>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setShowDatasetOverview(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Dataset Statistics */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Dataset Statistics</h3>
              <div className="grid grid-cols-2 gap-4">
                <Card className="p-4">
                  <div className="text-2xl font-bold text-blue-600">{columns.length}</div>
                  <div className="text-sm text-muted-foreground">Total Columns</div>
                </Card>
                <Card className="p-4">
                  <div className="text-2xl font-bold text-green-600">
                    {numericColumns.length}
                  </div>
                  <div className="text-sm text-muted-foreground">Numeric Columns</div>
                </Card>
                <Card className="p-4">
                  <div className="text-2xl font-bold text-purple-600">
                    {categoricalColumns.length}
                  </div>
                  <div className="text-sm text-muted-foreground">Categorical Columns</div>
                </Card>
                <Card className="p-4">
                  <div className="text-2xl font-bold text-orange-600">
                    {datetimeColumns.length}
                  </div>
                  <div className="text-sm text-muted-foreground">Date/Time Columns</div>
                </Card>
              </div>
              
              {/* Chart Recommendations */}
              <div className="mt-6">
                <h4 className="font-semibold mb-3">Recommended Chart Types</h4>
                <div className="space-y-2">
                  {recommendations.length > 0 ? (
                    recommendations.map((rec, index) => (
                      <div key={index} className="flex items-center space-x-2 text-sm">
                        {rec.icon}
                        <span>{rec.description}</span>
                      </div>
                    ))
                  ) : (
                    <div className="text-sm text-muted-foreground">
                      Select a dataset to see chart recommendations
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            {/* Column Details */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Column Details</h3>
              <ScrollArea className="h-96">
                <div className="space-y-3">
                  {columns.length > 0 ? columns.map(column => (
                    <Card key={column.name} className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{column.name}</h4>
                        <div className="flex items-center space-x-1">
                          {column.is_numeric && (
                            <Badge variant="secondary" className="text-xs bg-blue-100 text-blue-700">
                              NUM
                            </Badge>
                          )}
                          {column.is_categorical && (
                            <Badge variant="secondary" className="text-xs bg-green-100 text-green-700">
                              CAT
                            </Badge>
                          )}
                          {column.is_datetime && (
                            <Badge variant="secondary" className="text-xs bg-purple-100 text-purple-700">
                              DATE
                            </Badge>
                          )}
                        </div>
                      </div>
                      <div className="text-sm text-muted-foreground space-y-1">
                        <div>Type: {column.dtype || 'Unknown'}</div>
                        {column.non_null_count > 0 && (
                          <div>Non-null: {column.non_null_count.toLocaleString()}</div>
                        )}
                        {column.unique_count > 0 && (
                          <div>Unique: {column.unique_count.toLocaleString()}</div>
                        )}
                        {!column.non_null_count && !column.unique_count && (
                          <div className="text-xs text-gray-400">Detailed stats not available</div>
                        )}
                      </div>
                    </Card>
                  )) : (
                    <Card className="p-4">
                      <div className="text-center text-muted-foreground">
                        <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
                        <p>No column information available</p>
                      </div>
                    </Card>
                  )}
                </div>
              </ScrollArea>
            </div>
          </div>
        </CardContent>
        
        <CardFooter className="flex justify-between">
          <Button variant="outline" onClick={() => setShowDatasetOverview(false)}>
            Close
          </Button>
          <Button 
            onClick={() => {
              setShowDatasetOverview(false)
              onAutoGenerate()
            }}
            className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700"
          >
            <Sparkles className="h-4 w-4 mr-2" />
            Generate Recommended Charts
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
