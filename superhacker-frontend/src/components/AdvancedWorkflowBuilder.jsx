import React, { useState, useEffect, useRef, useCallback } from 'react'
import Plot from 'react-plotly.js'
import { 
  Play, 
  Square, 
  Save, 
  Download, 
  Upload, 
  Plus, 
  Trash2, 
  Settings, 
  Copy,
  Zap,
  Database,
  Filter,
  BarChart3,
  FileText,
  GitBranch,
  Target,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Clock,
  Eye,
  Edit3,
  X,
  ChevronRight,
  ChevronDown,
  Home,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Minimize2,
  ThumbsUp,
  Brain,
  Info
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Switch } from '@/components/ui/switch'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { toast } from 'sonner'

const API_BASE = 'http://localhost:5000/api'

// Icon mapping for different node types
const NODE_ICONS = {
  Database: Database,
  RefreshCw: RefreshCw,
  Filter: Filter,
  Zap: Zap,
  CheckCircle: CheckCircle,
  BarChart3: BarChart3,
  FileText: FileText,
  Target: Target,
  GitBranch: GitBranch,
  Eye: Eye,
  Download: Download
}

// Utility function to render charts (Plotly JSON or base64 images) with insights
const renderChart = (chartName, chartData) => {
  try {
    // Check if chartData has the new structure with chart and insight
    const hasInsightStructure = chartData && typeof chartData === 'object' && 
                               (chartData.chart !== undefined || chartData.insight !== undefined)
    
    let actualChartData = chartData
    let insightText = null
    
    if (hasInsightStructure) {
      actualChartData = chartData.chart
      insightText = chartData.insight
    }
    
    // Debug logging
    console.log(`Rendering chart: ${chartName}`, {
      type: typeof actualChartData,
      isObject: typeof actualChartData === 'object',
      hasData: actualChartData?.data !== undefined,
      hasLayout: actualChartData?.layout !== undefined,
      hasInsight: !!insightText,
      chartData: actualChartData
    })
    
    const renderChartContent = () => {
      // Handle Plotly chart objects (Event Detection charts come as objects)
      if (typeof actualChartData === 'object' && actualChartData !== null && (actualChartData.data || actualChartData.layout)) {
        return (
          <div className="plotly-chart-container">
            <Plot
              data={actualChartData.data || []}
              layout={{
                ...actualChartData.layout,
                autosize: true,
                margin: { l: 40, r: 40, t: 40, b: 40 },
                height: 400,
                font: { size: 10 }
              }}
              config={{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
              }}
              style={{ width: '100%', height: '400px' }}
              useResizeHandler={true}
            />
          </div>
        )
      }
      
      // If chartData is a string that looks like JSON, try to parse it as Plotly
      if (typeof actualChartData === 'string' && (actualChartData.trim().startsWith('{') || actualChartData.trim().startsWith('['))) {
        try {
          const plotlyData = JSON.parse(actualChartData)
          if (plotlyData && (plotlyData.data || plotlyData.layout)) {
            return (
              <div className="plotly-chart-container">
                <Plot
                  data={plotlyData.data || []}
                  layout={{
                    ...plotlyData.layout,
                    autosize: true,
                    margin: { l: 40, r: 40, t: 40, b: 40 },
                    height: 400,
                    font: { size: 10 }
                  }}
                  config={{
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                  }}
                  style={{ width: '100%', height: '400px' }}
                  useResizeHandler={true}
                />
              </div>
            )
          }
        } catch (parseError) {
          console.warn('Failed to parse as Plotly JSON:', parseError)
        }
      }
      
      // Fallback to base64 image rendering
      if (typeof actualChartData === 'string') {
        return (
          <div className="flex justify-center">
            <img 
              src={actualChartData.startsWith('data:image/png;base64,') ? actualChartData : `data:image/png;base64,${actualChartData}`} 
              alt={chartName}
              className="max-w-full h-auto rounded border shadow-sm"
              style={{ maxHeight: '400px' }}
            />
          </div>
        )
      }
      
      // If chart data is not a string, show unavailable
      return (
        <div className="text-sm text-gray-500 italic">Chart data unavailable</div>
      )
    }
    
    return (
      <div key={chartName} className="bg-gray-50 rounded-lg p-3 border">
        <h6 className="text-sm font-medium text-gray-700 mb-2 capitalize">
          {chartName.replace(/_/g, ' ')}
        </h6>
        
        {/* Render the chart */}
        {renderChartContent()}
        
        {/* Render insight if available */}
        {insightText && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-start gap-2">
              <Brain className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
              <div>
                <h6 className="text-sm font-medium text-blue-800 mb-1">Chart Insights</h6>
                <div className="text-sm text-blue-700 whitespace-pre-wrap">
                  {insightText}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  } catch (error) {
    console.error('Error rendering chart:', error)
    return (
      <div key={chartName} className="bg-gray-50 rounded-lg p-3 border">
        <h6 className="text-sm font-medium text-gray-700 mb-2 capitalize">
          {chartName.replace(/_/g, ' ')}
        </h6>
        <div className="text-sm text-red-500 italic">Error rendering chart</div>
      </div>
    )
  }
}

// Univariate Anomaly Panel Component with Column Selection
const UnivariateAnomalyPanel = ({ result }) => {
  const [selectedColumn, setSelectedColumn] = useState('')
  
  // Get available columns from the dataset info
  const availableColumns = result?.anomaly_results?.dataset_info?.columns_analyzed || []
  
  // Set default column to first numeric column
  useEffect(() => {
    if (availableColumns.length > 0 && !selectedColumn) {
      setSelectedColumn(availableColumns[0])
    }
  }, [availableColumns, selectedColumn])
  
  const handleColumnChange = (value) => {
    setSelectedColumn(value)
  }
  
  // Get column-specific charts
  const getColumnCharts = (column) => {
    if (!column || !result?.charts?.column_specific?.[column]) {
      return {}
    }
    return result.charts.column_specific[column]
  }
  
  const columnCharts = getColumnCharts(selectedColumn)
  
  return (
    <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
      <h5 className="font-bold text-yellow-700 mb-4 flex items-center text-lg">⚠️ Univariate Anomaly Detection</h5>
      
      {/* Column Selection Dropdown */}
      <div className="mb-4">
        <h6 className="font-semibold text-yellow-700 mb-2">🎯 Column Analysis</h6>
        <div className="bg-white p-3 rounded border">
          <div className="flex items-center gap-3 mb-3">
            <label className="text-sm font-medium text-gray-700">Select Column:</label>
            <select
              value={selectedColumn}
              onChange={(e) => handleColumnChange(e.target.value)}
              className="border border-gray-300 rounded px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent"
            >
              {availableColumns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
          
          {/* Column-specific summary */}
          {selectedColumn && result.anomaly_results.combined_analysis?.[selectedColumn] && (
            <div className="bg-yellow-50 p-3 rounded border border-yellow-200">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-sm text-gray-700">{selectedColumn}</span>
                <div className="flex gap-4 text-xs">
                  <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                    {result.anomaly_results.combined_analysis[selectedColumn].total_unique_anomalies || 0} anomalies
                  </span>
                  <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">
                    {result.anomaly_results.combined_analysis[selectedColumn].anomaly_percentage?.toFixed(2) || 0}% rate
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Column-Specific Charts */}
      {selectedColumn && columnCharts && Object.keys(columnCharts).length > 0 && (
        <div className="mb-4">
          <h6 className="font-semibold text-yellow-700 mb-2">📈 {selectedColumn} - Anomaly Analysis</h6>
          <div className="space-y-4">
            {Object.entries(columnCharts).map(([chartName, chartData]) => (
              chartData && renderChart(`${selectedColumn}_${chartName}`, chartData)
            ))}
          </div>
        </div>
      )}

      {/* Overall Analysis Summary - Always Visible */}
      <div className="mb-4">
        <div className="bg-white rounded border">
          <div className="p-3">
            <h6 className="font-semibold text-gray-700 mb-3">📊 Overall Analysis Summary</h6>
            
            {/* Overall Charts */}
            {result.charts && (
              <div className="space-y-4">
                {Object.entries(result.charts).filter(([name]) => name !== 'column_specific').map(([chartName, chartData]) => (
                  chartData && renderChart(chartName, chartData)
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {result.recommendations && result.recommendations.length > 0 && (
        <div className="mb-4">
          <h6 className="font-semibold text-yellow-700 mb-2">💡 Recommendations</h6>
          <div className="bg-white p-3 rounded border max-h-48 overflow-y-auto">
            <div className="space-y-2">
              {result.recommendations.map((recommendation, idx) => (
                <div key={idx} className="flex items-start space-x-2 p-2 bg-yellow-50 rounded">
                  <div className="flex-shrink-0 w-6 h-6 bg-yellow-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                    {idx + 1}
                  </div>
                  <div className="flex-1 text-sm text-gray-700">
                    {recommendation}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

const AdvancedWorkflowBuilder = () => {
  // State management
  const [workflows, setWorkflows] = useState([])
  const [currentWorkflow, setCurrentWorkflow] = useState(null)
  const [nodes, setNodes] = useState([])
  const [connections, setConnections] = useState([])
  const [selectedNode, setSelectedNode] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [executionResults, setExecutionResults] = useState(null)
  const [nodeTypes, setNodeTypes] = useState({})
  const [availableDatasets, setAvailableDatasets] = useState([])
  const [draggedNode, setDraggedNode] = useState(null)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectionStart, setConnectionStart] = useState(null)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const [activeTab, setActiveTab] = useState('workflow') // workflow, results
  const [hoveredConnection, setHoveredConnection] = useState(null)
  const [expandedCategories, setExpandedCategories] = useState({
    input: true,
    preprocessing: true,
    visualization: true,
    analysis: true,
    ml_supervised: true,
    ml_unsupervised: true,
    output: true
  })

  // State for node dragging
  const [isDraggingNode, setIsDraggingNode] = useState(false)
  const [draggedNodeId, setDraggedNodeId] = useState(null)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [nodeAnimations, setNodeAnimations] = useState({})
  const [dragStartPosition, setDragStartPosition] = useState({ x: 0, y: 0 })
  const [hasMoved, setHasMoved] = useState(false)
  const [mouseDownNode, setMouseDownNode] = useState(null)

  // State for canvas panning
  const [isPanning, setIsPanning] = useState(false)
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 })
  const [panStart, setPanStart] = useState({ x: 0, y: 0 })
  
  // State for canvas zooming
  const [zoomLevel, setZoomLevel] = useState(1)
  const [zoomOrigin, setZoomOrigin] = useState({ x: 0, y: 0 })

  // State for column data
  const [columnData, setColumnData] = useState({})
  
  // State for create workflow dialog
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [newWorkflowName, setNewWorkflowName] = useState('')
  const [newWorkflowDescription, setNewWorkflowDescription] = useState('')
  const [workflowTemplate, setWorkflowTemplate] = useState('blank')
  const [isCreatingWorkflow, setIsCreatingWorkflow] = useState(false)
  
  // State for node library search
  const [nodeLibrarySearch, setNodeLibrarySearch] = useState('')

  // State for AI Summary
  const [aiSummaryData, setAiSummaryData] = useState(null)
  const [aiSummaryLoading, setAiSummaryLoading] = useState(false)
  const [aiSummaryError, setAiSummaryError] = useState(null)
  
  // State for Streaming AI Summary
  const [streamingAiSummary, setStreamingAiSummary] = useState('')
  const [streamingTaskId, setStreamingTaskId] = useState(null)
  const [isStreamingActive, setIsStreamingActive] = useState(false)
  const [streamingError, setStreamingError] = useState(null)

  // Refs
  const canvasRef = useRef(null)

  // Load initial data
  useEffect(() => {
    loadWorkflows()
    loadNodeTypes()
    loadDatasets()
  }, [])

  // Canvas panning handlers
  const handleCanvasMouseDown = (e) => {
    // Check if the click is on a node (nodes have the class containing 'w-48')
    const isOnNode = e.target.closest('.absolute.w-48') !== null
    
    // Check if the click is on a connection line (SVG elements)
    const isOnConnection = e.target.closest('svg') !== null && e.target.tagName !== 'svg'
    
    // Only start panning if NOT clicking on nodes or connections
    if (!isOnNode && !isOnConnection) {
      // Prevent panning if already connecting or dragging a node
      if (isConnecting || isDraggingNode) {
        return
      }
      
      setIsPanning(true)
      setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y })
      e.preventDefault()
      e.stopPropagation()
      
      // Add visual feedback
      document.body.style.cursor = 'grabbing'
    }
  }

  // Canvas zoom handlers
  const handleCanvasWheel = useCallback((e) => {
    e.preventDefault()
    
    const delta = e.deltaY > 0 ? -0.1 : 0.1
    const newZoomLevel = Math.max(0.25, Math.min(3, zoomLevel + delta))
    
    // Visual feedback when hitting zoom limit
    if (newZoomLevel === zoomLevel && canvasRef.current) {
      canvasRef.current.classList.add('zoom-limit')
      setTimeout(() => {
        if (canvasRef.current) {
          canvasRef.current.classList.remove('zoom-limit')
        }
      }, 300)
      return
    }
    
    if (newZoomLevel !== zoomLevel) {
      const rect = canvasRef.current.getBoundingClientRect()
      const mouseX = e.clientX - rect.left
      const mouseY = e.clientY - rect.top
      
      setZoomLevel(newZoomLevel)
      setZoomOrigin({ x: mouseX, y: mouseY })
      
      // Adjust pan offset when zooming to prevent content from going out of bounds
      setPanOffset(prevOffset => {
        const canvasWidth = rect.width
        const canvasHeight = rect.height
        const maxPanX = Math.max(200, canvasWidth * (newZoomLevel - 1) * 0.5)
        const maxPanY = Math.max(200, canvasHeight * (newZoomLevel - 1) * 0.5)
        
        return {
          x: Math.max(-maxPanX, Math.min(maxPanX, prevOffset.x)),
          y: Math.max(-maxPanY, Math.min(maxPanY, prevOffset.y))
        }
      })
    }
  }, [zoomLevel])

  const zoomIn = useCallback(() => {
    const newZoomLevel = Math.min(3, zoomLevel + 0.25)
    
    // Visual feedback when hitting zoom limit
    if (newZoomLevel === zoomLevel && canvasRef.current) {
      canvasRef.current.classList.add('zoom-limit')
      setTimeout(() => {
        if (canvasRef.current) {
          canvasRef.current.classList.remove('zoom-limit')
        }
      }, 300)
      return
    }
    
    setZoomLevel(newZoomLevel)
    
    if (canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect()
      setZoomOrigin({ x: rect.width / 2, y: rect.height / 2 })
      
      // Adjust pan offset to stay within bounds
      setPanOffset(prevOffset => {
        const canvasWidth = rect.width
        const canvasHeight = rect.height
        const maxPanX = Math.max(200, canvasWidth * (newZoomLevel - 1) * 0.5)
        const maxPanY = Math.max(200, canvasHeight * (newZoomLevel - 1) * 0.5)
        
        return {
          x: Math.max(-maxPanX, Math.min(maxPanX, prevOffset.x)),
          y: Math.max(-maxPanY, Math.min(maxPanY, prevOffset.y))
        }
      })
    }
  }, [zoomLevel])

  const zoomOut = useCallback(() => {
    const newZoomLevel = Math.max(0.25, zoomLevel - 0.25)
    
    // Visual feedback when hitting zoom limit
    if (newZoomLevel === zoomLevel && canvasRef.current) {
      canvasRef.current.classList.add('zoom-limit')
      setTimeout(() => {
        if (canvasRef.current) {
          canvasRef.current.classList.remove('zoom-limit')
        }
      }, 300)
      return
    }
    
    setZoomLevel(newZoomLevel)
    
    if (canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect()
      setZoomOrigin({ x: rect.width / 2, y: rect.height / 2 })
      
      // Adjust pan offset to stay within bounds
      setPanOffset(prevOffset => {
        const canvasWidth = rect.width
        const canvasHeight = rect.height
        const maxPanX = Math.max(200, canvasWidth * (newZoomLevel - 1) * 0.5)
        const maxPanY = Math.max(200, canvasHeight * (newZoomLevel - 1) * 0.5)
        
        return {
          x: Math.max(-maxPanX, Math.min(maxPanX, prevOffset.x)),
          y: Math.max(-maxPanY, Math.min(maxPanY, prevOffset.y))
        }
      })
    }
  }, [zoomLevel])

  const resetZoom = useCallback(() => {
    setZoomLevel(1)
    setPanOffset({ x: 0, y: 0 })
    
    if (canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect()
      setZoomOrigin({ x: rect.width / 2, y: rect.height / 2 })
    }
  }, [])

  // Mouse tracking for connections, node dragging, and canvas panning
  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY })
      
      // Handle canvas panning
      if (isPanning) {
        const newOffset = {
          x: e.clientX - panStart.x,
          y: e.clientY - panStart.y
        }
        
        // Add intelligent limits based on canvas size and zoom level
        if (canvasRef.current) {
          const rect = canvasRef.current.getBoundingClientRect()
          const canvasWidth = rect.width
          const canvasHeight = rect.height
          
          // Calculate maximum pan distance based on zoom level
          // When zoomed in, allow more panning; when zoomed out, restrict it
          const maxPanX = Math.max(200, canvasWidth * (zoomLevel - 1) * 0.5)
          const maxPanY = Math.max(200, canvasHeight * (zoomLevel - 1) * 0.5)
          
          // Check if we're hitting boundaries
          const hitBoundaryX = newOffset.x <= -maxPanX || newOffset.x >= maxPanX
          const hitBoundaryY = newOffset.y <= -maxPanY || newOffset.y >= maxPanY
          
          // Apply boundaries
          newOffset.x = Math.max(-maxPanX, Math.min(maxPanX, newOffset.x))
          newOffset.y = Math.max(-maxPanY, Math.min(maxPanY, newOffset.y))
          
          // Visual feedback when hitting boundaries
          if (hitBoundaryX || hitBoundaryY) {
            canvasRef.current.style.borderColor = '#ef4444'
            setTimeout(() => {
              if (canvasRef.current) {
                canvasRef.current.style.borderColor = '#e5e7eb'
              }
            }, 200)
          }
        }
        
        console.log('Panning:', newOffset) // Debug log
        setPanOffset(newOffset)
      }
      
      // Check if we should start dragging (moved more than threshold)
      if (mouseDownNode && !isDraggingNode && !hasMoved) {
        const moveDistance = Math.sqrt(
          Math.pow(e.clientX - dragStartPosition.x, 2) + 
          Math.pow(e.clientY - dragStartPosition.y, 2)
        )
        
        if (moveDistance > 5) { // 5px threshold
          setHasMoved(true)
          setIsDraggingNode(true)
          setDraggedNodeId(mouseDownNode.id)
          
          // Calculate offset for smooth dragging
          if (canvasRef.current) {
            const rect = canvasRef.current.getBoundingClientRect()
            const offsetX = dragStartPosition.x - rect.left - mouseDownNode.x
            const offsetY = dragStartPosition.y - rect.top - mouseDownNode.y
            setDragOffset({ x: offsetX, y: offsetY })
          }
        }
      }
      
      // Handle node dragging
      if (isDraggingNode && draggedNodeId && canvasRef.current) {
        const rect = canvasRef.current.getBoundingClientRect()
        let newX = e.clientX - rect.left - dragOffset.x - panOffset.x
        let newY = e.clientY - rect.top - dragOffset.y - panOffset.y
        
        // Snap to grid for cleaner positioning
        newX = snapToGrid(newX, 20)
        newY = snapToGrid(newY, 20)
        
        // Allow nodes to be positioned anywhere within a large area
        const maxCanvasSize = 4000
        newX = Math.max(-maxCanvasSize, Math.min(newX, maxCanvasSize))
        newY = Math.max(-maxCanvasSize, Math.min(newY, maxCanvasSize))
        
        setNodes(prev => prev.map(node => 
          node.id === draggedNodeId 
            ? { ...node, x: newX, y: newY }
            : node
        ))
      }
    }

    const handleMouseUp = () => {
      // Handle canvas panning completion
      if (isPanning) {
        setIsPanning(false)
        document.body.style.cursor = 'auto' // Reset cursor
      }
      
      // Handle node selection (if no significant movement occurred)
      if (mouseDownNode && !hasMoved) {
        setSelectedNode(mouseDownNode)
        // Show a subtle success message for selection
        toast.success(`Selected ${mouseDownNode.name}`, { duration: 1000 })
      }
      
      // Handle drag completion
      if (isDraggingNode && hasMoved) {
        setIsDraggingNode(false)
        const currentDraggedNodeId = draggedNodeId
        setDraggedNodeId(null)
        setDragOffset({ x: 0, y: 0 })
        
        // Add bounce animation and success feedback
        if (currentDraggedNodeId) {
          setNodeAnimations(prev => ({
            ...prev,
            [currentDraggedNodeId]: 'bounce'
          }))
          
          setTimeout(() => {
            setNodeAnimations(prev => {
              const newAnimations = { ...prev }
              delete newAnimations[currentDraggedNodeId]
              return newAnimations
            })
          }, 600) // Match the bounce animation duration
          
          toast.success('Node positioned successfully', { duration: 1500 })
        }
      }
      
      // Reset mouse down state
      setMouseDownNode(null)
      setHasMoved(false)
      setDragStartPosition({ x: 0, y: 0 })
    }

    if (isConnecting || mouseDownNode || isPanning) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isConnecting, isDraggingNode, draggedNodeId, dragOffset, mouseDownNode, hasMoved, dragStartPosition, isPanning, panOffset, panStart, zoomLevel])

  const loadWorkflows = async () => {
    try {
      const response = await fetch(`${API_BASE}/workflows/workflows`)
      const data = await response.json()
      if (data.success) {
        setWorkflows(data.workflows)
      }
    } catch (error) {
      console.error('Error loading workflows:', error)
      toast.error('Failed to load workflows')
    }
  }

  const loadNodeTypes = async () => {
    try {
      const response = await fetch(`${API_BASE}/workflows/node-types`)
      const data = await response.json()
      if (data.success) {
        setNodeTypes(data.node_types)
      }
    } catch (error) {
      console.error('Error loading node types:', error)
      toast.error('Failed to load node types')
    }
  }

  const loadDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE}/workflows/datasets`)
      const data = await response.json()
      if (data.success) {
        setAvailableDatasets(data.datasets)
      }
    } catch (error) {
      console.error('Error loading datasets:', error)
      toast.error('Failed to load datasets')
    }
  }

  // Load column data for a dataset
  const loadDatasetColumns = async (datasetId) => {
    if (columnData[datasetId]) return columnData[datasetId] // Use cached data
    
    try {
      const response = await fetch(`${API_BASE}/data/datasets/${datasetId}`)
      const data = await response.json()
      if (data.success) {
        const columns = {
          all: data.columns.map(col => col.name),
          numeric: data.numeric_columns,
          categorical: data.categorical_columns,
          datetime: data.datetime_columns,
          recommendations: {} // No recommendations in this endpoint, can be added later
        }
        setColumnData(prev => ({ ...prev, [datasetId]: columns }))
        return columns
      }
    } catch (error) {
      console.error('Error loading dataset columns:', error)
    }
    return { all: [], numeric: [], categorical: [], datetime: [], recommendations: {} }
  }

  // Get available columns for current workflow
  const getAvailableColumns = () => {
    // Find data source nodes in current workflow to get dataset info
    const dataSourceNodes = nodes.filter(node => node.type === 'data_source')
    if (dataSourceNodes.length > 0) {
      const datasetId = dataSourceNodes[0].config?.dataset_id
      if (datasetId && columnData[datasetId]) {
        return columnData[datasetId]
      }
    }
    return { all: [], numeric: [], categorical: [], datetime: [], recommendations: {} }
  }

  const createNewWorkflow = async () => {
    setShowCreateDialog(true)
  }

  const handleCreateWorkflow = async () => {
    if (!newWorkflowName.trim()) {
      toast.error('Please enter a workflow name')
      return
    }

    setIsCreatingWorkflow(true)

    try {
      const response = await fetch(`${API_BASE}/workflows/workflows`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          name: newWorkflowName.trim(), 
          description: newWorkflowDescription.trim() || 'Created with Workflow Builder'
        })
      })
      const data = await response.json()
      if (data.success) {
        setCurrentWorkflow(data.workflow)
        setNodes([])
        setConnections([])
        setSelectedNode(null)
        setExecutionResults(null)
        
        // Reset dialog state
        setShowCreateDialog(false)
        setNewWorkflowName('')
        setNewWorkflowDescription('')
        setWorkflowTemplate('blank')
        
        toast.success('Workflow created successfully!')
        loadWorkflows()
        
        // Add template nodes if selected
        if (workflowTemplate === 'data_analysis') {
          // Add basic data analysis template
          setTimeout(() => {
            const templateNodes = [
              {
                id: `node_${Date.now()}_1`,
                type: 'data_source',
                name: 'Data Source',
                x: 100,
                y: 100,
                config: {}
              },
              {
                id: `node_${Date.now()}_2`,
                type: 'eda',
                name: 'Exploratory Data Analysis',
                x: 350,
                y: 100,
                config: {}
              },
              {
                id: `node_${Date.now()}_3`,
                type: 'visualization',
                name: 'Data Visualization',
                x: 600,
                y: 100,
                config: {}
              }
            ]
            setNodes(templateNodes)
            toast.success('Template nodes added to your workflow!')
          }, 500)
        } else if (workflowTemplate === 'ml_pipeline') {
          // Add ML pipeline template
          setTimeout(() => {
            const templateNodes = [
              {
                id: `node_${Date.now()}_1`,
                type: 'data_source',
                name: 'Data Source',
                x: 50,
                y: 100,
                config: {}
              },
              {
                id: `node_${Date.now()}_2`,
                type: 'preprocessing',
                name: 'Data Preprocessing',
                x: 300,
                y: 100,
                config: {}
              },
              {
                id: `node_${Date.now()}_3`,
                type: 'classification',
                name: 'ML Classification',
                x: 550,
                y: 100,
                config: {}
              },
              {
                id: `node_${Date.now()}_4`,
                type: 'model_evaluation',
                name: 'Model Evaluation',
                x: 800,
                y: 100,
                config: {}
              }
            ]
            setNodes(templateNodes)
            toast.success('ML Pipeline template added to your workflow!')
          }, 500)
        }
      }
    } catch (error) {
      console.error('Error creating workflow:', error)
      toast.error('Failed to create workflow')
    } finally {
      setIsCreatingWorkflow(false)
    }
  }

  const saveWorkflow = useCallback(async () => {
    if (!currentWorkflow) return

    try {
      const response = await fetch(`${API_BASE}/workflows/workflows/${currentWorkflow.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nodes,
          connections,
          status: 'draft'
        })
      })
      const data = await response.json()
      if (data.success) {
        toast.success('Workflow saved successfully')
      }
    } catch (error) {
      console.error('Error saving workflow:', error)
      toast.error('Failed to save workflow')
    }
  }, [currentWorkflow, nodes, connections])

  // Keyboard shortcuts handler
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Check if user is typing in an input/textarea
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return
      }
      
      // Handle keyboard shortcuts
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case '+':
          case '=':
            e.preventDefault()
            zoomIn()
            break
          case '-':
            e.preventDefault()
            zoomOut()
            break
          case '0':
            e.preventDefault()
            resetZoom()
            break
          case 's':
            e.preventDefault()
            saveWorkflow()
            break
          case 'n':
            e.preventDefault()
            // Focus on the new workflow input if it exists
            {
              const newWorkflowInput = document.querySelector('input[placeholder*="workflow name"]')
              if (newWorkflowInput) {
                newWorkflowInput.focus()
              }
            }
            break
          default:
            break
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [zoomIn, zoomOut, resetZoom, saveWorkflow])

  // Canvas wheel event handler (non-passive to allow preventDefault)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const handleWheel = (e) => {
      e.preventDefault()
      handleCanvasWheel(e)
    }

    canvas.addEventListener('wheel', handleWheel, { passive: false })
    return () => {
      canvas.removeEventListener('wheel', handleWheel)
    }
  }, [handleCanvasWheel])

  const runWorkflow = async () => {
    if (!currentWorkflow || nodes.length === 0) {
      toast.error('Please add nodes to the workflow before running')
      return
    }

    setIsRunning(true)
    setExecutionResults(null)

    try {
      const response = await fetch(`${API_BASE}/workflows/workflows/${currentWorkflow.id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, connections })
      })
      const data = await response.json()
      
      if (data.success) {
        setExecutionResults(data)
        setActiveTab('results') // Switch to results tab
        toast.success('Workflow executed successfully')
      } else {
        toast.error(`Workflow execution failed: ${data.error}`)
      }
    } catch (error) {
      console.error('Error running workflow:', error)
      toast.error('Failed to run workflow')
    } finally {
      setIsRunning(false)
    }
  }

  // AI Summary Generation Function
  const generateAISummary = async () => {
    if (!executionResults) {
      toast.error('No workflow results to analyze')
      return
    }

    setAiSummaryLoading(true)
    setAiSummaryError(null)

    try {
      // Prepare comprehensive data for AI analysis
      const comprehensiveData = {
        workflow_info: {
          name: currentWorkflow?.name || 'Untitled Workflow',
          nodes: nodes.map(node => ({
            id: node.id,
            name: node.name,
            type: node.type,
            config: node.config
          })),
          connections: connections
        },
        execution_results: executionResults.execution_results,
        summary: {
          total_nodes: nodes.length,
          completed_nodes: Object.values(executionResults.execution_results?.results || {}).filter(r => r.status === 'completed').length,
          execution_time: executionResults.execution_results?.summary?.execution_time || 0
        }
      }

      // Try the AI endpoint first, fall back to mock if it fails
      let data
      try {
        const response = await fetch(`${API_BASE}/ai_summary/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ comprehensive_data: comprehensiveData })
        })

        if (response.ok) {
          data = await response.json()
        } else {
          throw new Error('AI endpoint not available')
        }
      } catch (apiError) {
        console.log('AI endpoint not available, generating mock summary...')
        
        // Generate a mock AI summary based on the results
        const results = executionResults.execution_results?.results || {}
        const summary = executionResults.execution_results?.summary || {}
        
        // Analyze the workflow
        const completedNodes = Object.values(results).filter(r => r.status === 'completed').length
        const failedNodes = Object.values(results).filter(r => r.status === 'failed').length
        const totalExecutionTime = summary.execution_time || 0
        const successRate = (completedNodes / (completedNodes + failedNodes)) * 100 || 0
        
        // Get node types used
        const nodeTypesUsed = nodes.map(node => nodeTypes[node.type]?.name || node.type).filter(Boolean)
        const uniqueNodeTypes = [...new Set(nodeTypesUsed)]
        
        // Analyze data flow
        const dataNodes = nodes.filter(node => 
          nodeTypes[node.type]?.category === 'input' || 
          node.type === 'data_source'
        ).length
        
        const analysisNodes = nodes.filter(node => 
          nodeTypes[node.type]?.category === 'analysis' ||
          nodeTypes[node.type]?.category === 'ml_supervised' ||
          nodeTypes[node.type]?.category === 'ml_unsupervised'
        ).length
        
        const visualizationNodes = nodes.filter(node => 
          nodeTypes[node.type]?.category === 'visualization'
        ).length

        // Generate mock summary text
        const mockSummary = {
          overview: `Successfully executed a ${nodes.length}-node data science workflow "${currentWorkflow?.name || 'Untitled'}" with ${completedNodes} completed tasks in ${totalExecutionTime.toFixed(2)} seconds. The workflow achieved a ${successRate.toFixed(1)}% success rate using ${uniqueNodeTypes.length} different node types.`,
          
          insights: `Your workflow demonstrates a well-structured data science pipeline with ${dataNodes} data input node(s), ${analysisNodes} analysis/ML node(s), and ${visualizationNodes} visualization node(s). ${
            successRate === 100 ? 'All nodes executed successfully, indicating a robust pipeline design.' :
            successRate >= 80 ? 'Most nodes executed successfully with minimal failures.' :
            'There were some execution failures that may need attention.'
          } The execution time of ${totalExecutionTime.toFixed(2)} seconds suggests ${
            totalExecutionTime < 5 ? 'efficient processing' :
            totalExecutionTime < 30 ? 'reasonable performance' :
            'the workflow may benefit from optimization'
          }.`,
          
          recommendations: `${
            failedNodes > 0 ? `• Review and fix ${failedNodes} failed node(s) to improve success rate\n` : ''
          }• Consider adding data validation nodes if not present\n• ${
            visualizationNodes === 0 ? 'Add visualization nodes to better understand your results\n• ' :
            visualizationNodes < analysisNodes ? 'Consider adding more visualization nodes for comprehensive analysis\n• ' : ''
          }${
            totalExecutionTime > 30 ? 'Optimize data processing steps for better performance\n• ' : ''
          }Monitor execution logs for any warnings or optimization opportunities\n• Consider parameterizing node configurations for reusability`,
          
          summary: `This ${uniqueNodeTypes.join(', ')} workflow processed data through ${nodes.length} interconnected nodes, completing in ${totalExecutionTime.toFixed(2)}s with ${successRate.toFixed(1)}% success rate. ${
            completedNodes === nodes.length ? 'Excellent execution with no failures!' :
            failedNodes === 1 ? 'One node failed - check execution logs for details.' :
            failedNodes > 1 ? `${failedNodes} nodes failed - review configuration and data flow.` :
            'Good execution results.'
          }`
        }

        data = { success: true, summary: mockSummary }
      }
      
      if (data.success && data.summary) {
        setAiSummaryData(data.summary)
        setActiveTab('ai-summary') // Switch to AI summary tab
        toast.success('AI summary generated successfully!')
      } else {
        setAiSummaryError(data.error || 'Failed to generate AI summary')
        toast.error(`Failed to generate AI summary: ${data.error || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Error generating AI summary:', error)
      setAiSummaryError(error.message)
      toast.error('Failed to generate AI summary')
    } finally {
      setAiSummaryLoading(false)
    }
  }

  // Streaming AI Summary Generation Function
  const generateStreamingAISummary = async () => {
    if (!executionResults) {
      toast.error('No workflow results to analyze')
      return
    }

    setIsStreamingActive(true)
    setStreamingError(null)
    setStreamingAiSummary('')
    setStreamingTaskId(null)

    try {
      // Prepare comprehensive data for AI analysis - match expected format
      const comprehensiveData = {
        // Top-level nodes format expected by AI service
        nodes: nodes.reduce((acc, node) => {
          acc[node.id] = {
            id: node.id,
            name: node.name,
            type: node.type,
            config: node.config,
            // Add execution results for this node if available
            data: executionResults.execution_results?.results?.[node.id] || {}
          }
          return acc
        }, {}),
        workflow_context: {
          name: currentWorkflow?.name || 'Untitled Workflow',
          total_nodes: nodes.length,
          connections: connections,
          completed_nodes: Object.values(executionResults.execution_results?.results || {}).filter(r => r.status === 'completed').length,
          execution_time: executionResults.execution_results?.summary?.execution_time || 0,
          execution_results: executionResults.execution_results
        }
      }

      // Start background AI analysis
      let response, data
      try {
        response = await fetch(`${API_BASE}/ai_background/start_background_analysis`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ comprehensive_data: comprehensiveData })
        })
        data = await response.json()
      } catch (networkError) {
        console.log('Backend AI service unavailable, generating fallback streaming content...')
        toast.info('Backend AI service unavailable, using fallback analysis...')
        generateFallbackStreamingContent()
        return
      }
      
      if (data.success && data.task_id) {
        setStreamingTaskId(data.task_id)
        toast.success('Streaming AI analysis started!')
        
        // Start polling for results
        pollStreamingResults(data.task_id)
      } else {
        // Backend error - use fallback
        console.log('Backend returned error, generating fallback streaming content...')
        toast.info('Backend analysis failed, using fallback analysis...')
        generateFallbackStreamingContent()
      }
    } catch (error) {
      console.error('Error starting streaming AI summary:', error)
      
      // Use fallback streaming content instead of showing error
      console.log('Using fallback streaming content due to error...')
      toast.info('Using fallback streaming analysis...')
      generateFallbackStreamingContent()
    }
  }

  // Poll streaming results with fallback
  const pollStreamingResults = async (taskId) => {
    let attempts = 0
    const maxAttempts = 90 // 90 attempts = 3 minutes max (increased for comprehensive AI analysis)
    
    const poll = async () => {
      try {
        attempts++
        
        const response = await fetch(`${API_BASE}/ai_background/task_status/${taskId}`)
        const data = await response.json()
        
        if (data.success) {
          if (data.status === 'completed') {
            // Get the full streaming analysis result
            const resultResponse = await fetch(`${API_BASE}/ai_background/streaming_analysis/${taskId}`)
            const resultData = await resultResponse.json()
            
            if (resultData.success) {
              // Handle new comprehensive AI analysis format
              if (resultData.insights) {
                // New format with structured insights
                const aiSummary = formatComprehensiveAiAnalysis(resultData)
                setStreamingAiSummary(aiSummary)
                toast.success('Comprehensive AI analysis completed!')
              } else if (resultData.analysis_result) {
                // Legacy format
                setStreamingAiSummary(resultData.analysis_result)
                toast.success('Streaming AI analysis completed!')
              } else if (resultData.full_response) {
                // Fallback to full response
                setStreamingAiSummary(resultData.full_response)
                toast.success('AI analysis completed!')
              } else {
                throw new Error('No analysis content found in response')
              }
            } else {
              throw new Error('Failed to get streaming analysis result')
            }
            setIsStreamingActive(false)
          } else if (data.status === 'failed') {
            throw new Error(data.error || 'Streaming analysis failed')
          } else if (data.status === 'processing') {
            // Continue polling with progress feedback
            const progress = Math.min(90, (attempts / maxAttempts) * 100)
            
            // Show progress updates to user
            if (attempts % 15 === 0) { // Every 30 seconds
              toast.info(`AI analysis in progress... (${Math.round(progress)}%)`, { autoClose: 2000 })
            }
            
            if (attempts < maxAttempts) {
              // Update progress and continue polling
              setTimeout(poll, 2000) // Poll every 2 seconds
            } else {
              // Timeout reached - generate fallback streaming content
              console.log('Backend analysis timed out after 3 minutes, generating fallback streaming content...')
              toast.warning('AI analysis taking longer than expected, showing fallback analysis...')
              generateFallbackStreamingContent(true) // Pass timeout flag
            }
          }
        } else {
          throw new Error(data.error || 'Failed to get task status')
        }
      } catch (error) {
        console.error('Error polling streaming results:', error)
        
        // If it's a network error or API unavailable, try fallback
        if (error.message.includes('fetch') || error.message.includes('Failed to get task status')) {
          console.log('Network error detected, generating fallback streaming content...')
          generateFallbackStreamingContent(false) // Pass network error flag
        } else {
          setStreamingError(error.message)
          setIsStreamingActive(false)
          toast.error('Streaming analysis failed: ' + error.message)
        }
      }
    }
    
    poll()
  }

  // Format comprehensive AI analysis result into displayable content
  const formatComprehensiveAiAnalysis = (analysisData) => {
    const { insights, workflow_analysis, node_summary, metadata } = analysisData
    
    let formattedContent = `## 🤖 AI COMPREHENSIVE ANALYSIS\n\n`
    
    // Add metadata header
    if (metadata) {
      formattedContent += `**Analysis Timestamp:** ${new Date(metadata.analysis_timestamp).toLocaleString()}\n`
      formattedContent += `**AI Model:** ${metadata.ai_model_used || 'Advanced AI'}\n`
      formattedContent += `**Nodes Analyzed:** ${metadata.nodes_analyzed || 'Multiple'}\n`
      formattedContent += `**Complexity Level:** ${workflow_analysis?.complexity_level?.toUpperCase() || 'MODERATE'}\n\n`
    }
    
    // Add Data Summary
    if (insights?.data_summary) {
      formattedContent += `## 📊 DATA SUMMARY\n\n${insights.data_summary}\n\n`
    }
    
    // Add Integrated Analysis
    if (insights?.integrated_analysis) {
      formattedContent += `## 🔗 INTEGRATED DATA ANALYSIS\n\n${insights.integrated_analysis}\n\n`
    }
    
    // Add Analysis Results
    if (insights?.analysis_results) {
      formattedContent += `## 📈 ANALYSIS RESULTS\n\n${insights.analysis_results}\n\n`
    }
    
    // Add Key Insights
    if (insights?.key_insights && insights.key_insights.length > 0) {
      formattedContent += `## 💡 KEY INSIGHTS\n\n`
      insights.key_insights.forEach((insight, index) => {
        formattedContent += `${index + 1}. ${insight}\n`
      })
      formattedContent += `\n`
    }
    
    // Add Data Quality Assessment
    if (insights?.data_quality && insights.data_quality.length > 0) {
      formattedContent += `## 🔍 DATA QUALITY ASSESSMENT\n\n`
      insights.data_quality.forEach((quality, index) => {
        formattedContent += `${index + 1}. ${quality}\n`
      })
      formattedContent += `\n`
    }
    
    // Add Statistical Properties
    if (insights?.statistical_properties) {
      formattedContent += `## 📋 STATISTICAL PROPERTIES\n\n${insights.statistical_properties}\n\n`
    }
    
    // Add Node Summary
    if (node_summary) {
      formattedContent += `## 🏗️ WORKFLOW SUMMARY\n\n`
      formattedContent += `- **Total Nodes:** ${node_summary.total_nodes}\n`
      formattedContent += `- **Node Types:** ${node_summary.node_types?.join(', ') || 'Various'}\n`
      formattedContent += `- **Complexity:** ${node_summary.complexity_level?.toUpperCase() || 'MODERATE'}\n\n`
    }
    
    // Fallback to full response if structured content is empty
    if (formattedContent.trim() === '## 🤖 AI COMPREHENSIVE ANALYSIS\n\n') {
      return analysisData.full_response || 'AI analysis completed successfully.'
    }
    
    return formattedContent
  }

  // Generate fallback streaming content with simulated streaming effect
  const generateFallbackStreamingContent = (isTimeout = false) => {
    const results = executionResults.execution_results?.results || {}
    const summary = executionResults.execution_results?.summary || {}
    
    // Analyze the workflow
    const completedNodes = Object.values(results).filter(r => r.status === 'completed').length
    const failedNodes = Object.values(results).filter(r => r.status === 'failed').length
    const totalExecutionTime = summary.execution_time || 0
    const successRate = (completedNodes / (completedNodes + failedNodes)) * 100 || 0
    
    // Create comprehensive streaming analysis
    const streamingContent = `## 🔄 STREAMING ANALYSIS REPORT

**WORKFLOW PERFORMANCE ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **Execution Summary:**
• Workflow: "${currentWorkflow?.name || 'Untitled Workflow'}"
• Total Nodes: ${nodes.length}
• Completed: ${completedNodes}/${nodes.length} nodes
• Success Rate: ${successRate.toFixed(1)}%
• Execution Time: ${totalExecutionTime.toFixed(2)} seconds

📊 **Real-time Analysis:**
${successRate === 100 ? 
  '✅ EXCELLENT: All nodes executed successfully! This indicates a well-designed and robust data pipeline.' :
  successRate >= 80 ? 
  '✅ GOOD: Most nodes completed successfully with minimal issues.' :
  '⚠️ ATTENTION: Some nodes failed execution. Review the failed components for optimization opportunities.'
}

🔍 **Performance Insights:**
${totalExecutionTime < 5 ? 
  '⚡ FAST: Execution completed in under 5 seconds - highly optimized workflow.' :
  totalExecutionTime < 15 ? 
  '✅ EFFICIENT: Good performance with reasonable execution time.' :
  '⏳ CONSIDER OPTIMIZATION: Execution time suggests potential for performance improvements.'
}

🧠 **Data Flow Analysis:**
• Data Input Nodes: ${nodes.filter(n => nodeTypes[n.type]?.category === 'input' || n.type === 'data_source').length}
• Processing Nodes: ${nodes.filter(n => ['preprocessing', 'analysis', 'ml_supervised', 'ml_unsupervised'].includes(nodeTypes[n.type]?.category)).length}
• Output Nodes: ${nodes.filter(n => nodeTypes[n.type]?.category === 'visualization' || nodeTypes[n.type]?.category === 'output').length}

🔧 **Streaming Recommendations:**
${failedNodes > 0 ? `• PRIORITY: Fix ${failedNodes} failed node(s) to improve pipeline reliability\n` : ''}• Add data validation checkpoints for robust error handling
• Consider implementing caching for frequently accessed data
• Monitor memory usage during large dataset processing
${totalExecutionTime > 15 ? '• Optimize computational bottlenecks for better performance\n' : ''}• Implement logging for better debugging and monitoring

📈 **Next Steps:**
1. Review execution logs for any warnings or optimization opportunities
2. Consider adding more visualization nodes for comprehensive analysis
3. Implement parameterization for workflow reusability
4. Set up automated testing for continuous validation

**Analysis Generated:** ${new Date().toLocaleString()}
**Status:** ${isTimeout ? 'Comprehensive AI analysis taking longer than expected - showing workflow performance summary' : 'Network connectivity issue - showing fallback analysis'}`;

    // Simulate streaming effect
    let currentIndex = 0
    const streamText = (text) => {
      if (currentIndex < text.length) {
        const chunkSize = Math.floor(Math.random() * 50) + 30; // Random chunk size 30-80 chars
        const chunk = text.slice(0, currentIndex + chunkSize)
        setStreamingAiSummary(chunk)
        currentIndex += chunkSize
        
        // Continue streaming with random delay
        setTimeout(() => streamText(text), Math.floor(Math.random() * 300) + 100) // 100-400ms delay
      } else {
        // Streaming complete
        setIsStreamingActive(false)
        toast.success('Streaming analysis completed!')
      }
    }
    
    // Start streaming
    toast.info('Generating fallback streaming analysis...')
    streamText(streamingContent)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (!draggedNode) return

    const rect = canvasRef.current.getBoundingClientRect()
    let x = e.clientX - rect.left
    let y = e.clientY - rect.top

    // Snap to grid for cleaner positioning
    x = snapToGrid(x - 100, 20) // Offset for node width and snap to grid
    y = snapToGrid(y - 30, 20)  // Offset for node height and snap to grid
    
    // Keep nodes within canvas bounds
    const nodeWidth = 192 // w-48 = 192px
    const nodeHeight = 120 // approximate height
    x = Math.max(0, Math.min(x, rect.width - nodeWidth))
    y = Math.max(0, Math.min(y, rect.height - nodeHeight))

    const newNode = {
      id: `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: draggedNode.id,
      name: draggedNode.name,
      x: x,
      y: y,
      config: {}
    }

    setNodes(prev => [...prev, newNode])
    setDraggedNode(null)
    
    // Add entrance animation
    setNodeAnimations(prev => ({
      ...prev,
      [newNode.id]: 'bounce'
    }))
    
    setTimeout(() => {
      setNodeAnimations(prev => {
        const newAnimations = { ...prev }
        delete newAnimations[newNode.id]
        return newAnimations
      })
    }, 600)
    
    toast.success(`Added ${draggedNode.name} node`, { duration: 2000 })
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const startNodeConnection = (nodeId, port = 'output') => {
    setIsConnecting(true)
    setConnectionStart({ nodeId, port })
  }

  const finishNodeConnection = (targetNodeId, targetPort = 'input') => {
    if (!connectionStart || connectionStart.nodeId === targetNodeId) {
      setIsConnecting(false)
      setConnectionStart(null)
      return
    }

    // Check if connection already exists
    const existingConnection = connections.find(conn => 
      conn.source === connectionStart.nodeId && 
      conn.target === targetNodeId
    )

    if (existingConnection) {
      toast.error('Connection already exists between these nodes')
      setIsConnecting(false)
      setConnectionStart(null)
      return
    }

    const newConnection = {
      id: `conn_${Date.now()}`,
      source: connectionStart.nodeId,
      target: targetNodeId,
      sourcePort: connectionStart.port,
      targetPort: targetPort
    }

    setConnections(prev => [...prev, newConnection])
    setIsConnecting(false)
    setConnectionStart(null)
    toast.success('Nodes connected successfully')
  }

  const deleteNode = (nodeId) => {
    setNodes(prev => prev.filter(node => node.id !== nodeId))
    setConnections(prev => prev.filter(conn => 
      conn.source !== nodeId && conn.target !== nodeId
    ))
    if (selectedNode?.id === nodeId) {
      setSelectedNode(null)
    }
    toast.success('Node deleted')
  }

  const deleteConnection = (connectionId) => {
    setConnections(prev => prev.filter(conn => conn.id !== connectionId))
    toast.success('Connection deleted')
  }

  const updateNodeConfig = (nodeId, config) => {
    setNodes(prev => prev.map(node => 
      node.id === nodeId ? { ...node, config } : node
    ))
  }

  const handleNodeMouseDown = (e, node) => {
    e.stopPropagation()
    
    // Don't start drag on connection ports or buttons
    const target = e.target
    if (
      target.closest('[data-connection-port]') || 
      target.closest('button') ||
      e.button !== 0 // Only left mouse button
    ) {
      return
    }
    
    setMouseDownNode(node)
    setDragStartPosition({ x: e.clientX, y: e.clientY })
    setHasMoved(false)
  }

  // Snap to grid helper function
  const snapToGrid = (value, gridSize = 20) => {
    return Math.round(value / gridSize) * gridSize
  }

  // Updated getNodePosition - don't add panOffset since SVG is inside transformed container
  const getNodePosition = (nodeId) => {
    const node = nodes.find(n => n.id === nodeId)
    if (!node) return { x: 0, y: 0 }
    
    return { 
      x: node.x + 100, 
      y: node.y + 30 
    }
  }

  // Function to get descriptive category display names
  const getCategoryDisplayName = (category) => {
    const categoryNames = {
      'input': 'Input',
      'preprocessing': 'Preprocessing', 
      'visualization': 'Data Visualization',
      'analysis': 'Analysis',
      'ml_supervised': 'ML Supervised',
      'ml_unsupervised': 'ML Unsupervised',
      'output': 'Export Data'
    }
    return categoryNames[category] || category.replace('_', ' ')
  }

  const renderNodeLibrary = () => {
    const categories = {}
    Object.values(nodeTypes).forEach(nodeType => {
      // Filter out AI Data Summary nodes
      if (nodeType.id === 'ai_data_summary' || nodeType.name === 'AI Data Summary') {
        return // Skip AI Data Summary nodes
      }
      
      if (!categories[nodeType.category]) {
        categories[nodeType.category] = []
      }
      categories[nodeType.category].push(nodeType)
    })

    // Define the desired category order
    const categoryOrder = [
      'input',           // 1. INPUT - Data Sources
      'preprocessing',   // 2. PREPROCESSING - Data Cleaning & Preparation
      'visualization',   // 3. DATA VISUALIZATION - Charts & Plots
      'analysis',        // 4. ANALYSIS - Statistical Analysis & EDA
      'ml_supervised',   // 5. ML SUPERVISED - Supervised Learning Models
      'ml_unsupervised', // 6. ML UNSUPERVISED - Unsupervised Learning Models
      'output'           // 7. EXPORT DATA - Output & Model Deployment
    ]

    // Create ordered categories object
    const orderedCategories = {}
    categoryOrder.forEach(category => {
      if (categories[category]) {
        orderedCategories[category] = categories[category]
      }
    })
    
    // Add any remaining categories not in the defined order
    Object.keys(categories).forEach(category => {
      if (!orderedCategories[category]) {
        orderedCategories[category] = categories[category]
      }
    })

    // Filter nodes based on search
    const filteredCategories = {}
    if (nodeLibrarySearch.trim()) {
      categoryOrder.forEach(category => {
        if (orderedCategories[category]) {
          const filtered = orderedCategories[category].filter(node => 
            node.name.toLowerCase().includes(nodeLibrarySearch.toLowerCase()) ||
            node.description.toLowerCase().includes(nodeLibrarySearch.toLowerCase()) ||
            category.toLowerCase().includes(nodeLibrarySearch.toLowerCase())
          )
          if (filtered.length > 0) {
            filteredCategories[category] = filtered
          }
        }
      })
    } else {
      Object.assign(filteredCategories, orderedCategories)
    }

    return (
      <div className="node-library w-80 lg:w-80 md:w-72 sm:w-64 border-r bg-gray-50 flex flex-col h-full shrink-0">
        <div className="p-3 sm:p-4 border-b bg-white shrink-0">
          <h3 className="font-semibold text-base sm:text-lg text-gray-800 flex items-center space-x-2">
            <Database className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600" />
            <span className="hidden sm:inline">Node Library</span>
            <span className="sm:hidden">Nodes</span>
          </h3>
          <p className="text-xs text-gray-500 mt-1 hidden sm:block">Drag nodes to the canvas to build your workflow</p>
          
          {/* Search input */}
          <div className="mt-2 sm:mt-3 relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Database className="h-3 w-3 sm:h-4 sm:w-4 text-gray-400" />
            </div>
            <Input
              type="text"
              placeholder="Search nodes..."
              value={nodeLibrarySearch}
              onChange={(e) => setNodeLibrarySearch(e.target.value)}
              className="pl-8 sm:pl-10 h-7 sm:h-8 text-xs sm:text-sm bg-gray-50 border-gray-200 focus:bg-white"
            />
            {nodeLibrarySearch && (
              <button
                onClick={() => setNodeLibrarySearch('')}
                className="absolute inset-y-0 right-0 pr-2 sm:pr-3 flex items-center"
              >
                <X className="h-3 w-3 sm:h-4 sm:w-4 text-gray-400 hover:text-gray-600" />
              </button>
            )}
          </div>
        </div>
        
        {/* Scrollable node categories - this section will scroll independently */}
        <div className="flex-1 min-h-0">
          <ScrollArea className="h-full scrollbar-hidden">
            <div className="p-2 sm:p-4 space-y-2 sm:space-y-3">
              {Object.entries(filteredCategories).map(([category, categoryNodes]) => (
                <div key={category} className="mb-2">
                  <button
                    className="flex items-center w-full text-left p-2 sm:p-3 hover:bg-white hover:shadow-sm rounded-lg transition-all duration-200 bg-gray-100/50"
                    onClick={() => setExpandedCategories(prev => ({
                      ...prev,
                      [category]: !prev[category]
                    }))}
                  >
                    {expandedCategories[category] ? 
                      <ChevronDown className="w-3 h-3 sm:w-4 sm:h-4 mr-2 sm:mr-3 text-blue-600" /> : 
                      <ChevronRight className="w-3 h-3 sm:w-4 sm:h-4 mr-2 sm:mr-3 text-gray-500" />
                    }
                    <span className="font-medium text-gray-700 text-xs sm:text-sm">
                      {getCategoryDisplayName(category)}
                    </span>
                    <span className="ml-auto text-xs text-gray-500 bg-gray-200 px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-full">
                      {categoryNodes.length}
                    </span>
                  </button>
                  
                  {(expandedCategories[category] || nodeLibrarySearch.trim()) && (
                    <div className="mt-2 space-y-1 sm:space-y-2 pl-2 sm:pl-3">
                      {categoryNodes.map(nodeType => {
                        const IconComponent = NODE_ICONS[nodeType.icon] || Database
                        return (
                          <div
                            key={nodeType.id}
                            draggable
                            onDragStart={() => {
                              setDraggedNode(nodeType)
                              toast.info(`Drag ${nodeType.name} to the canvas`, { duration: 2000 })
                            }}
                            onDragEnd={() => {
                              setDraggedNode(null)
                            }}
                            className="group p-2 sm:p-3 border border-gray-200 rounded-lg cursor-move hover:shadow-md hover:border-blue-300 transition-all duration-200 bg-white transform hover:scale-[1.02] hover:-translate-y-0.5"
                          >
                            <div className="flex items-center space-x-2 sm:space-x-3">
                              <div 
                                className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg flex items-center justify-center text-white shadow-sm group-hover:shadow-md transition-shadow"
                                style={{ backgroundColor: nodeType.color }}
                              >
                                <IconComponent className="w-4 h-4 sm:w-5 sm:h-5" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-xs sm:text-sm text-gray-800 group-hover:text-blue-700 transition-colors">
                                  {nodeType.name}
                                </div>
                                <div className="text-xs text-gray-500 mt-0.5 line-clamp-2 hidden sm:block">
                                  {nodeType.description}
                                </div>
                              </div>
                              <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                                <div className="w-5 h-5 sm:w-6 sm:h-6 bg-blue-100 rounded-full flex items-center justify-center">
                                  <Plus className="w-3 h-3 text-blue-600" />
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              ))}
              
              {/* Empty state if no node types */}
              {Object.keys(filteredCategories).length === 0 && (
                <div className="text-center py-6 sm:py-8 text-gray-500">
                  {nodeLibrarySearch.trim() ? (
                    <>
                      <Database className="w-10 h-10 sm:w-12 sm:h-12 mx-auto mb-3 opacity-50" />
                      <p className="text-sm">No nodes found</p>
                      <p className="text-xs mt-1">Try a different search term</p>
                    </>
                  ) : (
                    <>
                      <Database className="w-10 h-10 sm:w-12 sm:h-12 mx-auto mb-3 opacity-50" />
                      <p className="text-sm">No node types available</p>
                      <p className="text-xs mt-1">Check your backend connection</p>
                    </>
                  )}
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      </div>
    )
  }

  const renderCanvas = () => {
    return (
      <div 
        ref={canvasRef}
        className={`flex-1 relative overflow-hidden canvas-container ${
          isPanning ? 'panning' : ''
        } ${
          isConnecting ? 'cursor-crosshair' : ''
        } ${
          isDraggingNode ? 'cursor-grabbing' : ''
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onMouseDown={(e) => {
          console.log('Canvas mouse down event:', e.target.className) // Debug log
          handleCanvasMouseDown(e)
        }}
        onMouseMove={(e) => {
          if (isPanning) {
            const newOffset = {
              x: e.clientX - panStart.x,
              y: e.clientY - panStart.y
            }
            
            // Add limits to prevent panning too far away
            const maxPanDistance = 2000 // Maximum distance in pixels
            newOffset.x = Math.max(-maxPanDistance, Math.min(maxPanDistance, newOffset.x))
            newOffset.y = Math.max(-maxPanDistance, Math.min(maxPanDistance, newOffset.y))
            
            setPanOffset(newOffset)
          }
        }}
        onMouseUp={() => {
          if (isPanning) {
            console.log('Canvas mouse up, stopping panning') // Debug log
            setIsPanning(false)
          }
        }}
        onMouseLeave={() => {
          if (isPanning) {
            console.log('Canvas mouse leave, stopping panning') // Debug log
            setIsPanning(false)
          }
        }}
        onClick={() => {
          if (isConnecting) {
            setIsConnecting(false)
            setConnectionStart(null)
          }
        }}
      >
        {/* Fixed background that always covers the entire canvas */}
        <div 
          className="absolute pointer-events-none"
          style={{
            left: '-200px',
            top: '-200px',
            right: '-200px',
            bottom: '-200px',
            backgroundColor: '#f8fafc',
            backgroundImage: `
              radial-gradient(circle at 10px 10px, rgba(59,130,246,0.12) 1px, transparent 1px),
              linear-gradient(rgba(59,130,246,0.06) 1px, transparent 1px),
              linear-gradient(90deg, rgba(59,130,246,0.06) 1px, transparent 1px)
            `,
            backgroundSize: '20px 20px, 20px 20px, 20px 20px',
            zIndex: 1
          }}
        />

        {/* Enhanced grid background that transforms with pan/zoom */}
        <div 
          className={`absolute transition-opacity duration-300 grid-background pointer-events-none ${
            isDraggingNode || draggedNode ? 'opacity-50' : 'opacity-25'
          }`}
          style={{
            left: '-1000px',
            top: '-1000px',
            right: '-1000px',
            bottom: '-1000px',
            backgroundImage: `
              radial-gradient(circle, rgba(59,130,246,0.3) 1px, transparent 1px),
              linear-gradient(rgba(59,130,246,0.12) 1px, transparent 1px),
              linear-gradient(90deg, rgba(59,130,246,0.12) 1px, transparent 1px)
            `,
            backgroundSize: `${20 * zoomLevel}px ${20 * zoomLevel}px, ${20 * zoomLevel}px ${20 * zoomLevel}px, ${20 * zoomLevel}px ${20 * zoomLevel}px`,
            transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel})`,
            transformOrigin: `${zoomOrigin.x + 1000}px ${zoomOrigin.y + 1000}px`,
            zIndex: 2
          }}
        />

        {/* Drop zone highlight when dragging from library */}
        {draggedNode && (
          <div className="absolute inset-0 bg-blue-50 bg-opacity-50 border-2 border-dashed border-blue-300 flex items-center justify-center" style={{ zIndex: 15 }}>
            <div className="text-center text-blue-600 font-medium">
              <Database className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-4 opacity-70 animate-pulse" />
              <p className="text-sm sm:text-lg">Drop to add {draggedNode.name} node</p>
            </div>
          </div>
        )}

        {/* Dragging indicator overlay */}
        {isDraggingNode && (
          <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 20 }}>
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white px-3 sm:px-4 py-2 rounded-lg shadow-lg">
              <div className="flex items-center space-x-2">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span className="text-sm">Moving node...</span>
              </div>
            </div>
          </div>
        )}

        {        /* Panning indicator */}
        {isPanning && (
          <div className="absolute top-4 right-4 bg-gray-800 text-white px-3 py-2 rounded-lg shadow-lg pointer-events-none z-50" style={{ zIndex: 25 }}>
            <div className="flex items-center space-x-2">
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span className="text-sm">Panning canvas...</span>
            </div>
          </div>
        )}

        {/* Zoom controls */}
        <div className="absolute bottom-4 right-4 bg-white rounded-lg shadow-lg border z-50 flex flex-col zoom-controls">
          <button
            onClick={zoomIn}
            className="p-2 hover:bg-gray-100 transition-colors border-b rounded-t-lg"
            title="Zoom In"
            disabled={zoomLevel >= 3}
          >
            <ZoomIn className="w-4 h-4 text-gray-600" />
          </button>
          <button
            onClick={zoomOut}
            className="p-2 hover:bg-gray-100 transition-colors border-b"
            title="Zoom Out"
            disabled={zoomLevel <= 0.25}
          >
            <ZoomOut className="w-4 h-4 text-gray-600" />
          </button>
          <button
            onClick={resetZoom}
            className="p-2 hover:bg-gray-100 transition-colors rounded-b-lg"
            title="Reset View"
          >
            <Minimize2 className="w-4 h-4 text-gray-600" />
          </button>
        </div>

        {/* Zoom level indicator */}
        <div className="absolute bottom-4 left-4 bg-white px-3 py-2 rounded-lg shadow-lg border z-40">
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <span>Zoom:</span>
            <span className="font-medium">{Math.round(zoomLevel * 100)}%</span>
          </div>
        </div>



        {/* Canvas content container - this will be transformed for panning and zooming */}
        <div 
          className={`absolute inset-0 ${isPanning ? '' : 'transition-transform duration-150 ease-out'}`}
          style={{
            transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel})`,
            transformOrigin: `${zoomOrigin.x}px ${zoomOrigin.y}px`,
            zIndex: 10
          }}
        >

        {/* Connections */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          <defs>
            {/* Enhanced gradient for connection lines */}
            <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.8" />
              <stop offset="50%" stopColor="#8B5CF6" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#EC4899" stopOpacity="0.8" />
            </linearGradient>
            
            {/* Animated gradient for active connections */}
            <linearGradient id="activeConnectionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10B981" stopOpacity="1">
                <animate attributeName="stop-color" 
                  values="#10B981;#06D6A0;#10B981" 
                  dur="2s" 
                  repeatCount="indefinite" />
                <animate attributeName="stop-opacity"
                  values="0.8;1;0.8"
                  dur="1s"
                  repeatCount="indefinite" />
              </stop>
              <stop offset="50%" stopColor="#3B82F6" stopOpacity="1">
                <animate attributeName="stop-color" 
                  values="#3B82F6;#06B6D4;#3B82F6" 
                  dur="2s" 
                  repeatCount="indefinite" />
              </stop>
              <stop offset="100%" stopColor="#8B5CF6" stopOpacity="1">
                <animate attributeName="stop-color" 
                  values="#8B5CF6;#A855F7;#8B5CF6" 
                  dur="2s" 
                  repeatCount="indefinite" />
                <animate attributeName="stop-opacity"
                  values="0.8;1;0.8"
                  dur="1s"
                  repeatCount="indefinite" />
              </stop>
            </linearGradient>

            {/* Connection arrow marker */}
            <marker id="arrowhead" markerWidth="12" markerHeight="8" 
              refX="11" refY="4" orient="auto" markerUnits="strokeWidth">
              <polygon points="0 0, 12 4, 0 8, 3 4" fill="#3B82F6" opacity="0.8" />
            </marker>

            {/* Animated arrow marker */}
            <marker id="animatedArrowhead" markerWidth="12" markerHeight="8" 
              refX="11" refY="4" orient="auto" markerUnits="strokeWidth">
              <polygon points="0 0, 12 4, 0 8, 3 4" fill="#10B981">
                <animateTransform
                  attributeName="transform"
                  attributeType="XML"
                  type="scale"
                  values="1;1.5;1"
                  dur="1s"
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="fill"
                  values="#10B981;#06D6A0;#10B981"
                  dur="2s"
                  repeatCount="indefinite"
                />
              </polygon>
            </marker>

            {/* Glow filter for active connections */}
            <filter id="connectionGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>

            {/* Pulse effect for connecting state */}
            <filter id="connectionPulse" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {connections.map(connection => {
            const sourcePos = getNodePosition(connection.source)
            const targetPos = getNodePosition(connection.target)
            
            // Calculate enhanced control points for smooth curves
            const dx = targetPos.x - sourcePos.x
            const dy = targetPos.y - sourcePos.y
            const distance = Math.sqrt(dx * dx + dy * dy)
            
            // Adaptive curve based on distance
            const curveFactor = Math.min(distance * 0.4, 150)
            const controlX1 = sourcePos.x + curveFactor
            const controlX2 = targetPos.x - curveFactor
            
            const isActive = executionResults && 
              executionResults.execution_results?.results && 
              executionResults.execution_results.results[connection.source]?.status === 'completed'
            
            return (
              <g key={connection.id}>
                {/* Connection glow effect */}
                <path
                  d={`M ${sourcePos.x} ${sourcePos.y} C ${controlX1} ${sourcePos.y} ${controlX2} ${targetPos.y} ${targetPos.x} ${targetPos.y}`}
                  stroke={isActive ? "#10B981" : "#3B82F6"}
                  strokeWidth="8"
                  fill="none"
                  opacity="0.2"
                  filter="url(#connectionGlow)"
                />
                
                {/* Connection shadow */}
                <path
                  d={`M ${sourcePos.x} ${sourcePos.y} C ${controlX1} ${sourcePos.y} ${controlX2} ${targetPos.y} ${targetPos.x} ${targetPos.y}`}
                  stroke="rgba(0,0,0,0.15)"
                  strokeWidth="4"
                  fill="none"
                  transform="translate(2,2)"
                />
                
                {/* Main connection line */}
                <path
                  d={`M ${sourcePos.x} ${sourcePos.y} C ${controlX1} ${sourcePos.y} ${controlX2} ${targetPos.y} ${targetPos.x} ${targetPos.y}`}
                  stroke={isActive ? "url(#activeConnectionGradient)" : "url(#connectionGradient)"}
                  strokeWidth="3"
                  fill="none"
                  markerEnd={isActive ? "url(#animatedArrowhead)" : "url(#arrowhead)"}
                  className="pointer-events-auto cursor-pointer transition-all duration-300"
                  filter={isActive ? "url(#connectionGlow)" : "none"}
                  onMouseEnter={() => setHoveredConnection(connection.id)}
                  onMouseLeave={() => setHoveredConnection(null)}
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteConnection(connection.id)
                  }}
                  style={{
                    strokeWidth: hoveredConnection === connection.id ? '5px' : '3px',
                    filter: hoveredConnection === connection.id ? 'url(#connectionGlow)' : (isActive ? 'url(#connectionGlow)' : 'none')
                  }}
                >
                  {/* Animated flow for active connections */}
                  {isActive && (
                    <animate
                      attributeName="stroke-dasharray"
                      values="0,20;20,20;40,20"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                  )}
                </path>

                {/* Data flow particles for active connections */}
                {isActive && (
                  <circle r="3" fill="#10B981" opacity="0.8">
                    <animateMotion
                      dur="3s"
                      repeatCount="indefinite"
                      path={`M ${sourcePos.x} ${sourcePos.y} C ${controlX1} ${sourcePos.y} ${controlX2} ${targetPos.y} ${targetPos.x} ${targetPos.y}`}
                    />
                    <animate
                      attributeName="opacity"
                      values="0;1;0"
                      dur="3s"
                      repeatCount="indefinite"
                    />
                  </circle>
                )}

                {/* Connection label on hover */}
                {hoveredConnection === connection.id && (
                  <g>
                    <rect
                      x={(sourcePos.x + targetPos.x) / 2 - 30}
                      y={(sourcePos.y + targetPos.y) / 2 - 12}
                      width="60"
                      height="24"
                      fill="white"
                      stroke="#3B82F6"
                      strokeWidth="2"
                      rx="6"
                      filter="url(#connectionGlow)"
                    />
                    <text
                      x={(sourcePos.x + targetPos.x) / 2}
                      y={(sourcePos.y + targetPos.y) / 2 + 4}
                      textAnchor="middle"
                      fontSize="10"
                      fill="#3B82F6"
                      fontWeight="bold"
                      className="pointer-events-none"
                    >
                      Click to delete
                    </text>
                  </g>
                )}
              </g>
            )
          })}
          
          {/* Enhanced temporary connection while dragging */}
          {isConnecting && connectionStart && canvasRef.current && (
            <g>
              {/* Connection glow */}
              <path
                d={`M ${getNodePosition(connectionStart.nodeId).x} ${getNodePosition(connectionStart.nodeId).y} L ${mousePosition.x - canvasRef.current.getBoundingClientRect().left - panOffset.x} ${mousePosition.y - canvasRef.current.getBoundingClientRect().top - panOffset.y}`}
                stroke="#6366F1"
                strokeWidth="8"
                fill="none"
                opacity="0.3"
                filter="url(#connectionPulse)"
              />
              
              {/* Main temp connection */}
              <path
                d={`M ${getNodePosition(connectionStart.nodeId).x} ${getNodePosition(connectionStart.nodeId).y} L ${mousePosition.x - canvasRef.current.getBoundingClientRect().left - panOffset.x} ${mousePosition.y - canvasRef.current.getBoundingClientRect().top - panOffset.y}`}
                stroke="#6366F1"
                strokeWidth="4"
                strokeDasharray="12,6"
                fill="none"
                opacity="0.8"
                filter="url(#connectionGlow)"
              >
                <animate
                  attributeName="stroke-dashoffset"
                  values="0;18;0"
                  dur="1.5s"
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="stroke-width"
                  values="3;5;3"
                  dur="2s"
                  repeatCount="indefinite"
                />
              </path>
              
              {/* Animated connection endpoint */}
              <circle
                cx={mousePosition.x - canvasRef.current.getBoundingClientRect().left - panOffset.x}
                cy={mousePosition.y - canvasRef.current.getBoundingClientRect().top - panOffset.y}
                r="8"
                fill="#6366F1"
                opacity="0.8"
                filter="url(#connectionGlow)"
              >
                <animate
                  attributeName="r"
                  values="6;12;6"
                  dur="1.5s"
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="opacity"
                  values="0.6;1;0.6"
                  dur="1s"
                  repeatCount="indefinite"
                />
              </circle>
              
              {/* Pulse rings around endpoint */}
              <circle
                cx={mousePosition.x - canvasRef.current.getBoundingClientRect().left - panOffset.x}
                cy={mousePosition.y - canvasRef.current.getBoundingClientRect().top - panOffset.y}
                r="15"
                fill="none"
                stroke="#6366F1"
                strokeWidth="2"
                opacity="0.4"
              >
                <animate
                  attributeName="r"
                  values="8;20;8"
                  dur="2s"
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="opacity"
                  values="0.6;0;0.6"
                  dur="2s"
                  repeatCount="indefinite"
                />
              </circle>
            </g>
          )}
        </svg>

        {/* Nodes */}
        {nodes.map(node => {
          const nodeType = nodeTypes[node.type]
          if (!nodeType) return null

          const IconComponent = NODE_ICONS[nodeType.icon] || Database
          const isSelected = selectedNode?.id === node.id
          const isDragging = draggedNodeId === node.id
          const animation = nodeAnimations[node.id]
          
          return (
            <div
              key={node.id}
              className={`absolute w-48 bg-white border-2 rounded-lg shadow-sm cursor-pointer transition-all duration-300 node-floating ${
                isSelected ? 'border-blue-500 shadow-lg scale-105' : 'border-gray-200 hover:border-gray-300'
              } ${
                isDragging ? 'shadow-2xl scale-110 z-50 drag-preview' : 'hover:shadow-md'
              } ${
                animation === 'bounce' ? 'node-bounce' : ''
              } ${
                animation === 'scale' ? 'scale-110' : ''
              }`}
              style={{ 
                left: node.x, 
                top: node.y,
                transform: isDragging ? 'rotate(2deg)' : 'rotate(0deg)',
                zIndex: isDragging ? 1000 : (isSelected ? 100 : 10)
              }}
              onClick={(e) => {
                e.stopPropagation()
                // Selection is now handled in mouse up event
              }}
              onMouseDown={(e) => {
                handleNodeMouseDown(e, node)
              }}
            >
              {/* Node header */}
              <div 
                className="p-3 rounded-t-lg text-white flex items-center justify-between"
                style={{ backgroundColor: nodeType.color }}
              >
                <div className="flex items-center space-x-2">
                  <IconComponent className="w-4 h-4" />
                  <span className="font-medium text-sm">{node.name}</span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteNode(node.id)
                  }}
                  className="hover:bg-black hover:bg-opacity-20 rounded p-1"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>

              {/* Node body */}
              <div className="p-3">
                <div className="text-xs text-gray-600 mb-2">{nodeType.description}</div>
                
                {/* Connection points */}
                <div className="flex justify-between items-center">
                  {/* Input port */}
                  {nodeType.inputs && nodeType.inputs.length > 0 && (
                    <div 
                      data-connection-port="input"
                      className={`w-4 h-4 border-2 border-white rounded-full cursor-pointer transition-all duration-200 -ml-8 ${
                        isConnecting && connectionStart && connectionStart.nodeId !== node.id
                          ? 'bg-green-500 hover:bg-green-600 scale-125 shadow-lg animate-pulse'
                          : 'bg-green-500 hover:bg-green-600 hover:scale-110'
                      }`}
                      onClick={(e) => {
                        e.stopPropagation()
                        if (isConnecting && connectionStart && connectionStart.nodeId !== node.id) {
                          finishNodeConnection(node.id, 'input')
                        }
                      }}
                      title="Input port - connect from output of another node"
                    >
                      {/* Input port indicator */}
                      <div className="w-full h-full rounded-full bg-white opacity-30"></div>
                    </div>
                  )}
                  
                  {/* Output port */}
                  {nodeType.outputs && nodeType.outputs.length > 0 && (
                    <div 
                      data-connection-port="output"
                      className={`w-4 h-4 border-2 border-white rounded-full cursor-pointer transition-all duration-200 -mr-8 ml-auto ${
                        !isConnecting
                          ? 'bg-blue-500 hover:bg-blue-600 hover:scale-110'
                          : 'bg-blue-600 scale-110 shadow-lg'
                      }`}
                      onClick={(e) => {
                        e.stopPropagation()
                        if (!isConnecting) {
                          startNodeConnection(node.id, 'output')
                        }
                      }}
                      title="Output port - drag to connect to input of another node"
                    >
                      {/* Output port indicator */}
                      <div className="w-full h-full rounded-full bg-white opacity-30"></div>
                    </div>
                  )}
                </div>

                {/* Execution status */}
                {executionResults && executionResults.execution_results?.results?.[node.id] && (
                  <div className="mt-2 pt-2 border-t">
                    <div className={`text-xs flex items-center space-x-1 ${
                      executionResults.execution_results.results[node.id].status === 'completed' 
                        ? 'text-green-600' 
                        : 'text-red-600'
                    }`}>
                      {executionResults.execution_results.results[node.id].status === 'completed' ? (
                        <CheckCircle className="w-3 h-3" />
                      ) : (
                        <AlertCircle className="w-3 h-3" />
                      )}
                      <span>{executionResults.execution_results.results[node.id].status}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )
        })}

        {/* Empty state */}
        {nodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-500">
              <Database className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-medium mb-2">Start Building Your Workflow</h3>
              <p className="text-sm mb-4">Drag nodes from the library to create your data science pipeline</p>
              <div className="text-xs text-gray-400 space-y-1">
                <p>💡 <strong>Tip:</strong> Click and drag the canvas to pan around</p>
                <p>🖱️ Use the mouse wheel to zoom in and out</p>
                <p>⌨️ Use zoom controls in the bottom-right corner</p>
              </div>
            </div>
          </div>
        )}
        </div>
      </div>
    )
  }

  const renderPropertiesPanel = () => {
    if (!selectedNode) {
      return (
        <div className="properties-panel w-80 lg:w-80 md:w-72 sm:w-64 border-l bg-gray-50 p-3 sm:p-4 shrink-0">
          <h3 className="font-semibold mb-4 text-sm sm:text-base">Properties</h3>
          <p className="text-gray-500 text-xs sm:text-sm">Select a node to configure its properties</p>
        </div>
      )
    }

    const nodeType = nodeTypes[selectedNode.type]
    if (!nodeType) return null

    return (
      <div className="properties-panel w-80 lg:w-80 md:w-72 sm:w-64 border-l bg-gray-50 p-3 sm:p-4 overflow-y-auto shrink-0">
        <h3 className="font-semibold mb-4 text-sm sm:text-base">Node Properties</h3>
        
        <div className="space-y-3 sm:space-y-4">
          <div>
            <Label className="text-xs sm:text-sm">Node Name</Label>
            <Input 
              value={selectedNode.name}
              onChange={(e) => {
                setNodes(prev => prev.map(node => 
                  node.id === selectedNode.id 
                    ? { ...node, name: e.target.value }
                    : node
                ))
                setSelectedNode(prev => ({ ...prev, name: e.target.value }))
              }}
              className="mt-1 h-8 sm:h-9 text-xs sm:text-sm"
            />
          </div>

          <Separator />

          {nodeType.parameters && nodeType.parameters.map(param => (
            <div key={param.name} className="space-y-2">
              <Label className="text-xs sm:text-sm">{param.description || param.name}</Label>
              
              {param.type === 'text' && (
                <Input
                  value={selectedNode.config[param.name] || param.default || ''}
                  onChange={(e) => {
                    const newConfig = { ...selectedNode.config, [param.name]: e.target.value }
                    updateNodeConfig(selectedNode.id, newConfig)
                    setSelectedNode(prev => ({ ...prev, config: newConfig }))
                  }}
                  placeholder={param.description}
                  className="h-8 sm:h-9 text-xs sm:text-sm"
                />
              )}

              {param.type === 'number' && (
                <Input
                  type="number"
                  value={selectedNode.config[param.name] || param.default || ''}
                  onChange={(e) => {
                    const newConfig = { ...selectedNode.config, [param.name]: parseFloat(e.target.value) || 0 }
                    updateNodeConfig(selectedNode.id, newConfig)
                    setSelectedNode(prev => ({ ...prev, config: newConfig }))
                  }}
                  min={param.min}
                  max={param.max}
                  className="h-8 sm:h-9 text-xs sm:text-sm"
                />
              )}

              {param.type === 'textarea' && (
                <Textarea
                  value={selectedNode.config[param.name] || param.default || ''}
                  onChange={(e) => {
                    const newConfig = { ...selectedNode.config, [param.name]: e.target.value }
                    updateNodeConfig(selectedNode.id, newConfig)
                    setSelectedNode(prev => ({ ...prev, config: newConfig }))
                  }}
                  placeholder={param.placeholder}
                  className="h-16 sm:h-20 text-xs sm:text-sm resize-none"
                />
              )}

              {param.type === 'select' && (
                <Select
                  value={selectedNode.config[param.name] || param.default || ''}
                  onValueChange={(value) => {
                    const newConfig = { ...selectedNode.config, [param.name]: value }
                    updateNodeConfig(selectedNode.id, newConfig)
                    setSelectedNode(prev => ({ ...prev, config: newConfig }))
                  }}
                >
                  <SelectTrigger className="h-8 sm:h-9 text-xs sm:text-sm">
                    <SelectValue placeholder={`Select ${param.name}`} />
                  </SelectTrigger>
                  <SelectContent>
                    {param.options && param.options.map(option => (
                      <SelectItem key={option} value={option}>
                        {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}

              {param.type === 'dataset_select' && (
                <Select
                  value={selectedNode.config[param.name] || ''}
                  onValueChange={async (value) => {
                    const newConfig = { ...selectedNode.config, [param.name]: value }
                    updateNodeConfig(selectedNode.id, newConfig)
                    setSelectedNode(prev => ({ ...prev, config: newConfig }))
                    
                    // Load column data for the selected dataset
                    if (value) {
                      await loadDatasetColumns(value)
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableDatasets.map(dataset => (
                      <SelectItem key={dataset.id} value={dataset.id.toString()}>
                        <div>
                          <div className="font-medium">{dataset.name}</div>
                          <div className="text-xs text-gray-500">
                            {dataset.rows_count} rows, {dataset.columns_count} columns
                          </div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}

              {param.type === 'boolean' && (
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={
                      selectedNode.config[param.name] !== undefined 
                        ? selectedNode.config[param.name] 
                        : (param.default !== undefined ? param.default : false)
                    }
                    onCheckedChange={(checked) => {
                      const newConfig = { ...selectedNode.config, [param.name]: checked }
                      updateNodeConfig(selectedNode.id, newConfig)
                      setSelectedNode(prev => ({ ...prev, config: newConfig }))
                    }}
                  />
                  <span className="text-sm">{param.description}</span>
                </div>
              )}

              {param.type === 'column_select' && (
                <div className="space-y-2">
                  <Select
                    value={selectedNode.config[param.name] || ''}
                    onValueChange={async (value) => {
                      // Handle clearing the selection
                      const newValue = value === '__CLEAR__' ? '' : value
                      const newConfig = { ...selectedNode.config, [param.name]: newValue }
                      
                      // Remove empty values from config to keep it clean
                      if (newValue === '') {
                        delete newConfig[param.name]
                      }
                      
                      updateNodeConfig(selectedNode.id, newConfig)
                      setSelectedNode(prev => ({ ...prev, config: newConfig }))
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={`Select ${param.name.replace('_', ' ')}`} />
                    </SelectTrigger>
                    <SelectContent>
                      {/* Add clear option if something is selected */}
                      {selectedNode.config[param.name] && (
                        <SelectItem value="__CLEAR__" className="text-red-600 font-medium">
                          <div className="flex items-center space-x-2">
                            <X className="w-4 h-4" />
                            <span>Clear selection</span>
                          </div>
                        </SelectItem>
                      )}
                      
                      {/* Separator if clear option is shown */}
                      {selectedNode.config[param.name] && (
                        <div className="border-t border-gray-200 my-1"></div>
                      )}
                      
                      {(() => {
                        const availableColumns = getAvailableColumns()
                        let columns = []
                        
                        // Get filtered columns based on parameter filter and node type
                        if (param.filter === 'numeric') {
                          columns = availableColumns.numeric
                        } else if (param.filter === 'categorical') {
                          columns = availableColumns.categorical
                        } else if (param.filter === 'target_suitable') {
                          // For target columns, exclude timestamp/datetime and ID-like columns
                          const excludeColumns = [...(availableColumns.datetime || []), 
                                                 ...availableColumns.all.filter(col => 
                                                   /^(id|index|timestamp|time|date|created|updated|modified)$/i.test(col) ||
                                                   /_id$/i.test(col) || 
                                                   /id_/i.test(col)
                                                 )]
                          columns = availableColumns.all.filter(col => !excludeColumns.includes(col))
                        } else {
                          columns = availableColumns.all
                        }
                        
                        return columns.map(column => (
                          <SelectItem key={column} value={column}>
                            <div className="flex items-center justify-between w-full">
                              <span>{column}</span>
                              {/* Show column type indicator */}
                              <span className="text-xs text-gray-500 ml-2">
                                {availableColumns.numeric?.includes(column) && '(num)'}
                                {availableColumns.categorical?.includes(column) && '(cat)'}
                                {availableColumns.datetime?.includes(column) && '(time)'}
                              </span>
                            </div>
                          </SelectItem>
                        ))
                      })()}
                    </SelectContent>
                  </Select>
                  
                  {/* Show current selection with clear button as alternative */}
                  {selectedNode.config[param.name] && (
                    <div className="flex items-center justify-between bg-blue-50 border border-blue-200 rounded-md p-2">
                      <span className="text-sm font-medium text-blue-700">
                        Selected: {selectedNode.config[param.name]}
                      </span>
                      <button
                        onClick={() => {
                          const newConfig = { ...selectedNode.config }
                          delete newConfig[param.name]
                          updateNodeConfig(selectedNode.id, newConfig)
                          setSelectedNode(prev => ({ ...prev, config: newConfig }))
                        }}
                        className="text-red-600 hover:text-red-800 p-1 hover:bg-red-100 rounded"
                        title="Clear selection"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                  
                  {/* Show validation messages */}
                  {(() => {
                    const isTargetColumn = param.name === 'target_column'
                    const selectedValue = selectedNode.config[param.name]
                    const availableColumns = getAvailableColumns()
                    
                    if (isTargetColumn && selectedValue) {
                      const isTimestamp = availableColumns.datetime?.includes(selectedValue)
                      const isIdLike = /^(id|index|timestamp|time|date|created|updated|modified)$/i.test(selectedValue) ||
                                      /_id$/i.test(selectedValue) || /id_/i.test(selectedValue)
                      
                      if (isTimestamp || isIdLike) {
                        return (
                          <div className="text-xs text-red-600 bg-red-50 p-2 rounded border border-red-200">
                            ⚠️ '{selectedValue}' is not suitable as a target column. 
                            {isTimestamp && ' Timestamp columns cannot be used as targets.'}
                            {isIdLike && ' ID columns are not suitable for prediction.'}
                            <br />
                            <span className="font-medium">Suggestions:</span> {
                              availableColumns.categorical?.length > 0 
                                ? `Try categorical columns: ${availableColumns.categorical.slice(0, 3).join(', ')}`
                                : availableColumns.numeric?.length > 0 
                                  ? `Try numeric columns: ${availableColumns.numeric.slice(0, 3).join(', ')}`
                                  : 'No suitable columns found'
                            }
                          </div>
                        )
                      }
                      
                      // Show success message for good selection
                      const isCategorical = availableColumns.categorical?.includes(selectedValue)
                      const isNumeric = availableColumns.numeric?.includes(selectedValue)
                      
                      if (selectedNode.type === 'classification' && isNumeric) {
                        return (
                          <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded border border-blue-200">
                            ℹ️ Numeric column selected for classification. The values will be automatically binned into categories for training.
                          </div>
                        )
                      }
                      
                      if (selectedNode.type === 'classification' && isCategorical) {
                        return (
                          <div className="text-xs text-green-600 bg-green-50 p-2 rounded border border-green-200">
                            ✅ Good choice! Categorical column is ideal for classification.
                          </div>
                        )
                      }
                      
                      if (selectedNode.type === 'regression' && isNumeric) {
                        return (
                          <div className="text-xs text-green-600 bg-green-50 p-2 rounded border border-green-200">
                            ✅ Good choice! Numeric column is ideal for regression.
                          </div>
                        )
                      }
                    }
                    
                    return null
                  })()}
                </div>
              )}

              {param.type === 'multi_column_select' && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="text-xs text-gray-600">
                      {param.description || `Select ${param.name.replace('_', ' ')}`}
                    </div>
                    {/* Clear All button */}
                    {(() => {
                      const selectedColumns = selectedNode.config[param.name] || []
                      return selectedColumns.length > 0 && (
                        <button
                          onClick={() => {
                            const newConfig = { ...selectedNode.config }
                            delete newConfig[param.name]
                            updateNodeConfig(selectedNode.id, newConfig)
                            setSelectedNode(prev => ({ ...prev, config: newConfig }))
                          }}
                          className="text-xs text-red-600 hover:text-red-800 hover:bg-red-50 px-2 py-1 rounded border border-red-200"
                        >
                          Clear All
                        </button>
                      )
                    })()}
                  </div>
                  
                  <div className="border rounded-md p-2 max-h-32 overflow-y-auto">
                    {(() => {
                      const availableColumns = getAvailableColumns()
                      let columns = []
                      
                      // Apply filters similar to column_select
                      if (param.filter === 'numeric') {
                        columns = availableColumns.numeric
                      } else if (param.filter === 'categorical') {
                        columns = availableColumns.categorical
                      } else if (param.filter === 'feature_suitable') {
                        // For feature columns, exclude timestamp and ID columns
                        const excludeColumns = [...(availableColumns.datetime || []), 
                                               ...availableColumns.all.filter(col => 
                                                 /^(id|index|timestamp|time|date|created|updated|modified)$/i.test(col) ||
                                                 /_id$/i.test(col) || 
                                                 /id_/i.test(col)
                                               )]
                        columns = availableColumns.all.filter(col => !excludeColumns.includes(col))
                      } else {
                        columns = availableColumns.all
                      }
                      
                      const selectedColumns = selectedNode.config[param.name] || []
                      
                      if (columns.length === 0) {
                        return (
                          <div className="text-sm text-gray-500 text-center py-2">
                            No suitable columns available
                          </div>
                        )
                      }
                      
                      return columns.map(column => (
                        <div key={column} className="flex items-center space-x-2 mb-1 hover:bg-gray-50 rounded px-1 py-1">
                          <input
                            type="checkbox"
                            id={`${param.name}_${column}`}
                            checked={selectedColumns.includes(column)}
                            onChange={(e) => {
                              let newSelectedColumns
                              if (e.target.checked) {
                                newSelectedColumns = [...selectedColumns, column]
                              } else {
                                newSelectedColumns = selectedColumns.filter(col => col !== column)
                              }
                              const newConfig = { ...selectedNode.config }
                              if (newSelectedColumns.length > 0) {
                                newConfig[param.name] = newSelectedColumns
                              } else {
                                delete newConfig[param.name]
                              }
                              updateNodeConfig(selectedNode.id, newConfig)
                              setSelectedNode(prev => ({ ...prev, config: newConfig }))
                            }}
                            className="h-4 w-4 text-blue-600 rounded"
                          />
                          <label 
                            htmlFor={`${param.name}_${column}`}
                            className="text-sm text-gray-700 cursor-pointer flex-1 flex items-center justify-between"
                          >
                            <span>{column}</span>
                            {/* Show column type indicator */}
                            <span className="text-xs text-gray-500">
                              {availableColumns.numeric?.includes(column) && '(num)'}
                              {availableColumns.categorical?.includes(column) && '(cat)'}
                              {availableColumns.datetime?.includes(column) && '(time)'}
                            </span>
                          </label>
                        </div>
                      ))
                    })()}
                  </div>
                  
                  {/* Selected columns summary */}
                  {(() => {
                    const selectedColumns = selectedNode.config[param.name] || []
                    return selectedColumns.length > 0 && (
                      <div className="bg-blue-50 border border-blue-200 rounded p-2">
                        <div className="text-xs text-blue-700 font-medium mb-1">
                          Selected ({selectedColumns.length}) columns:
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {selectedColumns.map(column => (
                            <span key={column} className="inline-flex items-center bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                              {column}
                              <button
                                onClick={() => {
                                  const newSelectedColumns = selectedColumns.filter(col => col !== column)
                                  const newConfig = { ...selectedNode.config }
                                  if (newSelectedColumns.length > 0) {
                                    newConfig[param.name] = newSelectedColumns
                                  } else {
                                    delete newConfig[param.name]
                                  }
                                  updateNodeConfig(selectedNode.id, newConfig)
                                  setSelectedNode(prev => ({ ...prev, config: newConfig }))
                                }}
                                className="ml-1 text-blue-600 hover:text-blue-800"
                                title={`Remove ${column}`}
                              >
                                <X className="w-3 h-3" />
                              </button>
                            </span>
                          ))}
                        </div>
                      </div>
                    )
                  })()}
                  
                  {/* Show warnings for feature selection */}
                  {(() => {
                    const isFeatureColumns = param.name === 'feature_columns'
                    const selectedColumns = selectedNode.config[param.name] || []
                    const availableColumns = getAvailableColumns()
                    
                    if (isFeatureColumns && selectedColumns.length > 0) {
                      const timeColumns = selectedColumns.filter(col => availableColumns.datetime?.includes(col))
                      const idColumns = selectedColumns.filter(col => 
                        /^(id|index|timestamp|time|date|created|updated|modified)$/i.test(col) ||
                        /_id$/i.test(col) || /id_/i.test(col)
                      )
                      
                      if (timeColumns.length > 0 || idColumns.length > 0) {
                        return (
                          <div className="text-xs text-amber-600 bg-amber-50 p-2 rounded border border-amber-200">
                            ⚠️ Some selected columns may not be suitable as features:
                            {timeColumns.length > 0 && <div>• Timestamp columns: {timeColumns.join(', ')}</div>}
                            {idColumns.length > 0 && <div>• ID columns: {idColumns.join(', ')}</div>}
                          </div>
                        )
                      }
                    }
                    
                    return null
                  })()}
                </div>
              )}

              {param.type === 'multiselect_columns' && (
                <div className="space-y-2">
                  <div className="text-xs text-gray-500">
                    Select multiple columns (leave empty for auto-detection)
                  </div>
                  <Input
                    value={selectedNode.config[param.name] || ''}
                    onChange={(e) => {
                      const newConfig = { ...selectedNode.config, [param.name]: e.target.value }
                      updateNodeConfig(selectedNode.id, newConfig)
                      setSelectedNode(prev => ({ ...prev, config: newConfig }))
                    }}
                    placeholder="Enter column names separated by commas, or leave empty for auto-detection"
                  />
                  <div className="text-xs text-blue-500">
                    Available columns: {getAvailableColumns().all.join(', ')}
                  </div>
                </div>
              )}

              {param.type === 'multiselect' && (
                <div className="text-sm text-gray-500">
                  Multi-select not yet implemented
                </div>
              )}

              {param.required && (
                <div className="text-xs text-red-500">Required</div>
              )}
            </div>
          ))}
        </div>
      </div>
    )
  }

  const renderResultsPanel = () => {
    if (!executionResults) {
      return (
        <div className="flex-1 flex items-center justify-center bg-gray-50">
          <div className="text-center text-gray-500">
            <BarChart3 className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-4 opacity-50" />
            <h3 className="text-base sm:text-lg font-medium mb-2">No Results Yet</h3>
            <p className="text-xs sm:text-sm">Run a workflow to see the execution results here</p>
          </div>
        </div>
      )
    }

    return (
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header for Results panel */}
        <div className="border-b bg-white px-4 py-3 shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <GitBranch className="w-5 h-5 text-blue-600" />
              <h3 className="font-medium text-blue-800">Execution Results</h3>
            </div>
            
            {/* AI Summary Quick Access Button */}
            <Button
              onClick={() => setActiveTab('ai-summary')}
              size="sm"
              variant="outline"
              className="border-purple-200 text-purple-600 hover:bg-purple-50"
            >
              <Brain className="w-4 h-4 mr-2" />
              AI Summary
            </Button>
          </div>
          
          <p className="text-xs text-gray-500 mt-1">
            View detailed execution results below.
          </p>
        </div>

        {/* Content */}
        {renderExecutionResults()}
      </div>
    )
  }

  const renderExecutionResults = () => {
    const results = executionResults.execution_results?.results || {}
    const summary = executionResults.execution_results?.summary || {}
    const logs = executionResults.execution_results?.logs || []

    return (
      <div className="flex-1 flex min-h-0 flex-col lg:flex-row">
        {/* Results sidebar */}
        <div className="w-full lg:w-80 border-b lg:border-b-0 lg:border-r bg-white p-3 sm:p-4 overflow-y-auto shrink-0 max-h-64 lg:max-h-none">
          <h3 className="font-semibold mb-3 sm:mb-4 text-sm sm:text-base">Execution Summary</h3>
          
          {/* Summary cards */}
          <div className="space-y-2 sm:space-y-3 mb-4 sm:mb-6">
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-sm text-blue-600 font-medium">Total Nodes</div>
              <div className="text-2xl font-bold text-blue-700">{summary.total_nodes || 0}</div>
            </div>
            
            <div className="bg-green-50 p-3 rounded-lg">
              <div className="text-sm text-green-600 font-medium">Completed</div>
              <div className="text-2xl font-bold text-green-700">{summary.nodes_completed || 0}</div>
            </div>
            
            {summary.nodes_failed > 0 && (
              <div className="bg-red-50 p-3 rounded-lg">
                <div className="text-sm text-red-600 font-medium">Failed</div>
                <div className="text-2xl font-bold text-red-700">{summary.nodes_failed}</div>
              </div>
            )}
            
            <div className="bg-purple-50 p-3 rounded-lg">
              <div className="text-sm text-purple-600 font-medium">Execution Time</div>
              <div className="text-2xl font-bold text-purple-700">
                {summary.execution_time ? `${summary.execution_time.toFixed(2)}s` : '0s'}
              </div>
            </div>
            
            <div className="bg-yellow-50 p-3 rounded-lg">
              <div className="text-sm text-yellow-600 font-medium">Success Rate</div>
              <div className="text-2xl font-bold text-yellow-700">
                {summary.success_rate ? `${summary.success_rate.toFixed(1)}%` : '0%'}
              </div>
            </div>
          </div>

          {/* Node results list */}
          <h4 className="font-medium mb-2">Node Results</h4>
          <div className="space-y-2">
            {Object.entries(results).filter(([nodeId, result]) => {
              const node = nodes.find(n => n.id === nodeId)
              const nodeType = nodeTypes[node?.type]
              // Filter out AI Data Summary nodes from results list
              return !(node?.type === 'ai_data_summary' || nodeType?.name === 'AI Data Summary')
            }).map(([nodeId, result]) => {
              const node = nodes.find(n => n.id === nodeId)
              const isCompleted = result.status === 'completed'
              
              return (
                <div key={nodeId} className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                  isCompleted ? 'border-green-200 bg-green-50 hover:bg-green-100' : 'border-red-200 bg-red-50 hover:bg-red-100'
                }`}>
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-sm">{node?.name || nodeId}</span>
                    <Badge variant={isCompleted ? 'default' : 'destructive'} className="text-xs">
                      {result.status}
                    </Badge>
                  </div>
                  
                  {result.result_summary && (
                    <div className="mt-2 text-xs text-gray-600">
                      <div>Type: {result.result_summary.type}</div>
                      {result.result_summary.shape && (
                        <div>Shape: {result.result_summary.shape[0]} × {result.result_summary.shape[1]}</div>
                      )}
                      {result.execution_time && (
                        <div>Time: {result.execution_time.toFixed(2)}s</div>
                      )}
                    </div>
                  )}
                  
                  {result.error && (
                    <div className="mt-2 text-xs text-red-600 font-mono">
                      {result.error}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Main results area */}
        <div className="flex-1 p-3 sm:p-6 bg-gray-50 overflow-y-auto min-h-0">
          <div className="space-y-4 sm:space-y-6">
            {/* Workflow visualization */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-sm sm:text-base">
                  <GitBranch className="w-4 h-4 sm:w-5 sm:h-5" />
                  <span>Workflow Execution Flow</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative h-24 sm:h-32 bg-gray-100 rounded-lg p-2 sm:p-4 overflow-x-auto">
                  <div className="flex items-center space-x-2 sm:space-x-4 h-full">
                    {(() => {
                      const filteredNodes = nodes.filter(node => {
                        // Filter out AI Data Summary nodes from workflow visualization
                        const nodeType = nodeTypes[node.type]
                        return !(node.type === 'ai_data_summary' || nodeType?.name === 'AI Data Summary')
                      })
                      
                      return filteredNodes.map((node, index) => {
                        const result = results[node.id]
                        const isCompleted = result?.status === 'completed'
                        const isFailed = result?.status === 'failed'
                      
                        return (
                          <div key={node.id} className="flex items-center">
                            <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-white font-medium text-xs ${
                              isCompleted ? 'bg-green-500' : 
                              isFailed ? 'bg-red-500' : 
                              'bg-gray-400'
                            }`}>
                              {isCompleted && <CheckCircle className="w-6 h-6" />}
                              {isFailed && <AlertCircle className="w-6 h-6" />}
                              {!result && <Clock className="w-6 h-6" />}
                            </div>
                            <div className="ml-2 text-xs">
                              <div className="font-medium">{node.name}</div>
                              {result?.execution_time && (
                                <div className="text-gray-500">{result.execution_time.toFixed(2)}s</div>
                              )}
                              {result?.charts && Object.keys(result.charts).length > 0 && (
                                <div className="text-blue-500">📊 {Object.keys(result.charts).length} chart(s)</div>
                              )}
                            </div>
                            
                            {index < filteredNodes.length - 1 && (
                              <div className="mx-4">
                                <ChevronRight className="w-5 h-5 text-gray-400" />
                              </div>
                            )}
                          </div>
                        )
                      })
                    })()}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Step-by-step workflow results */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-sm sm:text-base">
                  <FileText className="w-4 h-4 sm:w-5 sm:h-5" />
                  <span>Step-by-Step Results</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0 sm:p-6">
                <div className="max-h-96 sm:max-h-[700px] overflow-y-auto">
                  <div className="space-y-4 sm:space-y-6 p-3 sm:p-0">
                    {(() => {
                      // Create execution order based on workflow flow
                      const getExecutionOrder = () => {
                        // Find nodes without incoming connections (start nodes)
                        const startNodes = nodes.filter(node => 
                          !connections.some(conn => conn.target === node.id)
                        )
                        
                        if (startNodes.length === 0) return []
                        
                        const visited = new Set()
                        const order = []
                        
                        const dfs = (nodeId) => {
                          if (visited.has(nodeId)) return
                          visited.add(nodeId)
                          order.push(nodeId)
                          
                          // Find connected nodes
                          const outgoingConnections = connections.filter(conn => conn.source === nodeId)
                          outgoingConnections.forEach(conn => {
                            dfs(conn.target)
                          })
                        }
                        
                        startNodes.forEach(node => dfs(node.id))
                        return order
                      }
                      
                      const executionOrder = getExecutionOrder()
                      
                      return executionOrder.map((nodeId, stepIndex) => {
                        const node = nodes.find(n => n.id === nodeId)
                        const result = results[nodeId]
                        const nodeType = nodeTypes[node?.type]
                        
                        if (!node || !result) return null
                        
                        // Filter out AI Data Summary nodes from results display
                        if (node.type === 'ai_data_summary' || nodeType?.name === 'AI Data Summary') {
                          return null
                        }
                        
                        const isCompleted = result.status === 'completed'
                        const isFailed = result.status === 'failed'
                        
                        return (
                          <div key={nodeId} className="relative">
                            {/* Step connector line */}
                            {stepIndex > 0 && (
                              <div className="absolute -top-6 left-6 w-0.5 h-6 bg-gray-300"></div>
                            )}
                            
                            {/* Step header */}
                            <div className="flex items-center space-x-4 mb-4">
                              <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white font-bold ${
                                isCompleted ? 'bg-green-500' : 
                                isFailed ? 'bg-red-500' : 
                                'bg-gray-400'
                              }`}>
                                {stepIndex + 1}
                              </div>
                              <div className="flex-1">
                                <div className="flex items-center space-x-2">
                                  {nodeType && (
                                    <div 
                                      className="w-6 h-6 rounded flex items-center justify-center text-white"
                                      style={{ backgroundColor: nodeType.color }}
                                    >
                                      {React.createElement(NODE_ICONS[nodeType.icon] || Database, { className: "w-3 h-3" })}
                                    </div>
                                  )}
                                  <h3 className="text-lg font-semibold">{node.name}</h3>
                                  <Badge variant={isCompleted ? 'default' : 'destructive'}>
                                    {result.status}
                                  </Badge>
                                  {result.execution_time && (
                                    <Badge variant="outline" className="text-xs">
                                      {result.execution_time.toFixed(2)}s
                                    </Badge>
                                  )}
                                  {result.charts && Object.keys(result.charts).length > 0 && (
                                    <Badge variant="outline" className="text-xs bg-blue-50 text-blue-700">
                                      📊 {Array.isArray(result.charts) ? result.charts.length : Object.keys(result.charts).length} chart(s)
                                    </Badge>
                                  )}
                                </div>
                                <p className="text-sm text-gray-600 mt-1">{nodeType?.description || 'Processing step'}</p>
                              </div>
                            </div>
                            
                            {/* Step content */}
                            <div className="ml-16 space-y-4">
                              
                              {/* Output Summary */}
                              {result.result_summary && (
                                <div className="bg-gray-50 p-4 rounded-lg">
                                  <h5 className="font-medium text-sm mb-3">Output Summary</h5>
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                                    <div className="bg-white p-2 rounded border">
                                      <div className="text-xs text-gray-500">Type</div>
                                      <div className="font-medium">{result.result_summary.type}</div>
                                    </div>
                                    {result.result_summary.shape && (
                                      <div className="bg-white p-2 rounded border">
                                        <div className="text-xs text-gray-500">Shape</div>
                                        <div className="font-medium">{result.result_summary.shape[0]} × {result.result_summary.shape[1]}</div>
                                      </div>
                                    )}
                                    {result.result_summary.memory_usage && (
                                      <div className="bg-white p-2 rounded border">
                                        <div className="text-xs text-gray-500">Memory</div>
                                        <div className="font-medium">{result.result_summary.memory_usage}</div>
                                      </div>
                                    )}
                                    {result.execution_time && (
                                      <div className="bg-white p-2 rounded border">
                                        <div className="text-xs text-gray-500">Duration</div>
                                        <div className="font-medium">{result.execution_time.toFixed(2)}s</div>
                                      </div>
                                    )}
                                  </div>
                                  {result.result_summary.columns && (
                                    <div className="mt-3">
                                      <div className="text-xs text-gray-500 mb-1">Columns</div>
                                      <div className="text-sm font-medium">
                                        {result.result_summary.columns.slice(0, 5).join(', ')}
                                        {result.result_summary.columns.length > 5 ? ` (+${result.result_summary.columns.length - 5} more)` : ''}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                              
                              {/* Data Cleaning Node Details */}
                              {node?.type === 'data_cleaning' && result.result_summary && (
                                <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                                  <h5 className="font-bold text-green-700 mb-4 flex items-center text-lg">🧹 Data Cleaning Analysis</h5>
                                  
                                  {/* Key Metrics */}
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                                    {result.result_summary.data_quality_score !== undefined && (
                                      <div className="bg-white p-3 rounded border text-center shadow-sm">
                                        <div className="text-xs text-gray-500 mb-1">Data Quality Score</div>
                                        <div className="font-bold text-2xl text-green-700">{result.result_summary.data_quality_score.toFixed(1)}</div>
                                        <div className="text-xs text-gray-400">/100</div>
                                      </div>
                                    )}
                                    {result.result_summary.rows_removed !== undefined && (
                                      <div className="bg-white p-3 rounded border text-center shadow-sm">
                                        <div className="text-xs text-gray-500 mb-1">Rows Removed</div>
                                        <div className="font-bold text-2xl text-red-600">{result.result_summary.rows_removed.toLocaleString()}</div>
                                        <div className="text-xs text-gray-400">rows</div>
                                      </div>
                                    )}
                                    {result.result_summary.columns_removed !== undefined && (
                                      <div className="bg-white p-3 rounded border text-center shadow-sm">
                                        <div className="text-xs text-gray-500 mb-1">Columns Removed</div>
                                        <div className="font-bold text-2xl text-orange-600">{result.result_summary.columns_removed}</div>
                                        <div className="text-xs text-gray-400">columns</div>
                                      </div>
                                    )}
                                    {result.result_summary.operations_performed && (
                                      <div className="bg-white p-3 rounded border text-center shadow-sm">
                                        <div className="text-xs text-gray-500 mb-1">Operations</div>
                                        <div className="font-bold text-2xl text-blue-600">{result.result_summary.operations_performed.length}</div>
                                        <div className="text-xs text-gray-400">performed</div>
                                      </div>
                                    )}
                                  </div>

                                  {/* Data Shape Changes */}
                                  {(result.result_summary.original_shape && result.result_summary.final_shape) && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-green-700 mb-2">📊 Data Shape Changes</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="flex items-center justify-between">
                                          <div className="text-center">
                                            <div className="text-sm font-medium text-gray-700">Original</div>
                                            <div className="text-lg font-bold text-gray-800">
                                              {result.result_summary.original_shape[0].toLocaleString()} × {result.result_summary.original_shape[1]}
                                            </div>
                                            <div className="text-xs text-gray-500">rows × columns</div>
                                          </div>
                                          <div className="text-2xl text-gray-400">→</div>
                                          <div className="text-center">
                                            <div className="text-sm font-medium text-gray-700">Final</div>
                                            <div className="text-lg font-bold text-green-700">
                                              {result.result_summary.final_shape[0].toLocaleString()} × {result.result_summary.final_shape[1]}
                                            </div>
                                            <div className="text-xs text-gray-500">rows × columns</div>
                                          </div>
                                          <div className="text-center">
                                            <div className="text-sm font-medium text-gray-700">Reduction</div>
                                            <div className="text-lg font-bold text-red-600">
                                              {((1 - (result.result_summary.final_shape[0] / result.result_summary.original_shape[0])) * 100).toFixed(1)}%
                                            </div>
                                            <div className="text-xs text-gray-500">data reduced</div>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Detailed Operations */}
                                  {result.result_summary.operations_performed && result.result_summary.operations_performed.length > 0 && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-green-700 mb-2">🔧 Cleaning Operations Performed</h6>
                                      <div className="bg-white p-3 rounded border max-h-64 overflow-y-auto">
                                        <div className="space-y-2">
                                          {result.result_summary.operations_performed.map((operation, idx) => (
                                            <div key={idx} className="flex items-start space-x-2 p-2 bg-gray-50 rounded">
                                              <div className="flex-shrink-0 w-6 h-6 bg-green-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                                                {idx + 1}
                                              </div>
                                              <div className="flex-1 text-sm text-gray-700">
                                                {operation}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Data Quality Breakdown */}
                                  <div className="mb-2">
                                    <h6 className="font-semibold text-green-700 mb-2">📈 Data Quality Assessment</h6>
                                    <div className="bg-white p-3 rounded border">
                                      <div className="flex items-center space-x-2">
                                        <div className="flex-1 bg-gray-200 rounded-full h-3">
                                          <div 
                                            className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full transition-all duration-500"
                                            style={{ width: `${result.result_summary.data_quality_score || 0}%` }}
                                          ></div>
                                        </div>
                                        <div className="text-sm font-bold text-green-700">
                                          {(result.result_summary.data_quality_score || 0).toFixed(1)}%
                                        </div>
                                      </div>
                                      <div className="mt-2 text-xs text-gray-600">
                                        {result.result_summary.data_quality_score >= 90 ? '🟢 Excellent data quality' :
                                         result.result_summary.data_quality_score >= 75 ? '🟡 Good data quality' :
                                         result.result_summary.data_quality_score >= 50 ? '🟠 Fair data quality' :
                                         '🔴 Poor data quality - consider additional cleaning'}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              )}

                              {/* EDA Analysis Node Details */}
                              {node?.type === 'eda_analysis' && result.eda_results && (
                                <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg">
                                  <h5 className="font-bold text-purple-700 mb-4 flex items-center text-lg">🔍 Exploratory Data Analysis</h5>
                                  
                                  {/* Dataset Overview */}
                                  {result.eda_results.overview && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">📊 Dataset Overview</h6>
                                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Total Rows</div>
                                          <div className="font-bold text-2xl text-purple-700">
                                            {result.eda_results.overview.shape?.rows?.toLocaleString() || 
                                             result.eda_results.overview.rows?.toLocaleString() || 'N/A'}
                                          </div>
                                        </div>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Total Columns</div>
                                          <div className="font-bold text-2xl text-blue-600">
                                            {result.eda_results.overview.shape?.columns || 
                                             result.eda_results.overview.columns || 'N/A'}
                                          </div>
                                        </div>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Numeric Columns</div>
                                          <div className="font-bold text-2xl text-green-600">
                                            {result.eda_results.overview.numeric_columns || 
                                             result.eda_results.executive_summary?.variable_types?.numeric || 'N/A'}
                                          </div>
                                        </div>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Missing Values</div>
                                          <div className="font-bold text-2xl text-red-600">
                                            {result.eda_results.overview.missing_values || 
                                             result.eda_results.data_quality?.total_missing_values || 
                                             result.eda_results.overview.total_missing_values || '0'}
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Executive Summary */}
                                  {result.eda_results.executive_summary && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">📝 Executive Summary</h6>
                                      <div className="bg-white p-3 rounded border">
                                        {typeof result.eda_results.executive_summary === 'string' ? (
                                          <div className="text-sm text-gray-700 whitespace-pre-line font-mono">{result.eda_results.executive_summary}</div>
                                        ) : (
                                          <div className="space-y-2">
                                            {result.eda_results.executive_summary.key_insights && (
                                              <div>
                                                <div className="font-semibold text-sm text-gray-600 mb-1">Key Insights:</div>
                                                <ul className="text-sm text-gray-700 space-y-1">
                                                  {result.eda_results.executive_summary.key_insights.map((insight, idx) => (
                                                    <li key={idx} className="flex items-start space-x-2">
                                                      <span className="text-purple-600 mt-1">•</span>
                                                      <span>{insight}</span>
                                                    </li>
                                                  ))}
                                                </ul>
                                              </div>
                                            )}
                                            {result.eda_results.executive_summary.complexity && (
                                              <div className="text-sm text-gray-600">
                                                <span className="font-semibold">Complexity:</span> {result.eda_results.executive_summary.complexity.label}
                                              </div>
                                            )}
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Data Quality */}
                                  {result.eda_results.data_quality && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">🔍 Data Quality Assessment</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3">
                                          <div className="text-center">
                                            <div className="text-xs text-gray-500 mb-1">Quality Score</div>
                                            <div className="font-bold text-xl text-green-600">
                                              {result.eda_results.data_quality.overall_quality_score?.toFixed(1) || 'N/A'}%
                                            </div>
                                          </div>
                                          <div className="text-center">
                                            <div className="text-xs text-gray-500 mb-1">Missing Data</div>
                                            <div className="font-bold text-xl text-yellow-600">
                                              {result.eda_results.data_quality.completeness?.score?.toFixed(1) || 'N/A'}%
                                            </div>
                                          </div>
                                          <div className="text-center">
                                            <div className="text-xs text-gray-500 mb-1">Quality Grade</div>
                                            <div className="font-bold text-xl text-blue-600">
                                              {result.eda_results.data_quality.quality_grade || 'N/A'}
                                            </div>
                                          </div>
                                        </div>
                                        
                                        {/* Detailed Quality Metrics */}
                                        {(result.eda_results.data_quality.completeness || 
                                          result.eda_results.data_quality.consistency || 
                                          result.eda_results.data_quality.validity || 
                                          result.eda_results.data_quality.uniqueness) && (
                                          <div className="border-t pt-3">
                                            <div className="text-xs font-semibold text-gray-600 mb-2">Detailed Metrics:</div>
                                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                                              {result.eda_results.data_quality.completeness && (
                                                <div className="bg-green-50 p-2 rounded">
                                                  <div className="font-semibold text-green-700">Completeness</div>
                                                  <div className="text-green-600">{result.eda_results.data_quality.completeness.score?.toFixed(1)}%</div>
                                                </div>
                                              )}
                                              {result.eda_results.data_quality.consistency && (
                                                <div className="bg-blue-50 p-2 rounded">
                                                  <div className="font-semibold text-blue-700">Consistency</div>
                                                  <div className="text-blue-600">{result.eda_results.data_quality.consistency.score?.toFixed(1)}%</div>
                                                </div>
                                              )}
                                              {result.eda_results.data_quality.validity && (
                                                <div className="bg-orange-50 p-2 rounded">
                                                  <div className="font-semibold text-orange-700">Validity</div>
                                                  <div className="text-orange-600">{result.eda_results.data_quality.validity.score?.toFixed(1)}%</div>
                                                </div>
                                              )}
                                              {result.eda_results.data_quality.uniqueness && (
                                                <div className="bg-purple-50 p-2 rounded">
                                                  <div className="font-semibold text-purple-700">Uniqueness</div>
                                                  <div className="text-purple-600">{result.eda_results.data_quality.uniqueness.score?.toFixed(1)}%</div>
                                                </div>
                                              )}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Statistical Insights */}
                                  {result.eda_results.statistical_insights && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">📊 Statistical Insights</h6>
                                      <div className="bg-white p-3 rounded border max-h-64 overflow-y-auto">
                                        {result.eda_results.statistical_insights.numeric_summary && (
                                          <div className="mb-3">
                                            <div className="text-sm font-semibold text-gray-600 mb-2">Numeric Variables Summary:</div>
                                            <div className="overflow-x-auto">
                                              <table className="min-w-full text-xs">
                                                <thead className="bg-gray-50">
                                                  <tr>
                                                    <th className="px-2 py-1 text-left">Variable</th>
                                                    <th className="px-2 py-1 text-center">Mean</th>
                                                    <th className="px-2 py-1 text-center">Std</th>
                                                    <th className="px-2 py-1 text-center">Min</th>
                                                    <th className="px-2 py-1 text-center">Max</th>
                                                  </tr>
                                                </thead>
                                                <tbody>
                                                  {Object.entries(result.eda_results.statistical_insights.numeric_summary).map(([variable, stats]) => (
                                                    <tr key={variable} className="border-b">
                                                      <td className="px-2 py-1 font-medium text-gray-700">{variable}</td>
                                                      <td className="px-2 py-1 text-center">{stats.mean?.toFixed(2) || 'N/A'}</td>
                                                      <td className="px-2 py-1 text-center">{stats.std?.toFixed(2) || 'N/A'}</td>
                                                      <td className="px-2 py-1 text-center">{stats.min?.toFixed(2) || 'N/A'}</td>
                                                      <td className="px-2 py-1 text-center">{stats.max?.toFixed(2) || 'N/A'}</td>
                                                    </tr>
                                                  ))}
                                                </tbody>
                                              </table>
                                            </div>
                                          </div>
                                        )}
                                        
                                        {result.eda_results.statistical_insights.categorical_summary && Object.keys(result.eda_results.statistical_insights.categorical_summary).length > 0 && (
                                          <div>
                                            <div className="text-sm font-semibold text-gray-600 mb-2">Categorical Variables Summary:</div>
                                            <div className="space-y-2">
                                              {Object.entries(result.eda_results.statistical_insights.categorical_summary).map(([variable, stats]) => (
                                                <div key={variable} className="bg-gray-50 p-2 rounded">
                                                  <div className="font-medium text-gray-700 text-sm">{variable}</div>
                                                  <div className="text-xs text-gray-600">
                                                    Unique values: {stats.unique_values} | 
                                                    Most common: {Object.keys(stats.most_common || {})[0] || 'N/A'}
                                                  </div>
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Correlations */}
                                  {result.eda_results.correlations && result.eda_results.correlations.correlation_matrix && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">🔗 Correlation Analysis</h6>
                                      <div className="bg-white p-3 rounded border">
                                        {result.eda_results.correlations.high_correlations && result.eda_results.correlations.high_correlations.length > 0 ? (
                                          <div className="space-y-2">
                                            <div className="text-sm font-semibold text-gray-600">High Correlations Found:</div>
                                            {result.eda_results.correlations.high_correlations.slice(0, 5).map((corr, idx) => (
                                              <div key={idx} className="flex justify-between items-center p-2 bg-purple-50 rounded">
                                                <span className="text-sm text-gray-700">{corr.variable1} ↔ {corr.variable2}</span>
                                                <span className={`text-sm font-bold ${Math.abs(corr.correlation) > 0.8 ? 'text-red-600' : 'text-orange-600'}`}>
                                                  {corr.correlation?.toFixed(3)}
                                                </span>
                                              </div>
                                            ))}
                                          </div>
                                        ) : (
                                          <div className="text-sm text-gray-600">No high correlations detected (threshold: 0.7)</div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Recommendations */}
                                  {(result.recommendations && result.recommendations.length > 0) || 
                                   (result.eda_results.recommendations && result.eda_results.recommendations.length > 0) && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">💡 Recommendations</h6>
                                      <div className="bg-white p-3 rounded border max-h-48 overflow-y-auto">
                                        <div className="space-y-2">
                                          {(result.recommendations || result.eda_results.recommendations || []).map((recommendation, idx) => (
                                            <div key={idx} className="flex items-start space-x-2 p-2 bg-purple-50 rounded">
                                              <div className="flex-shrink-0 w-6 h-6 bg-purple-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                                                {idx + 1}
                                              </div>
                                              <div className="flex-1">
                                                <div className="text-sm font-medium text-gray-800">
                                                  {typeof recommendation === 'string' ? recommendation : recommendation.title || recommendation.description}
                                                </div>
                                                {typeof recommendation === 'object' && recommendation.actions && (
                                                  <div className="text-xs text-gray-600 mt-1">
                                                    Priority: {recommendation.priority} | Impact: {recommendation.impact}
                                                  </div>
                                                )}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Charts */}
                                  {result.charts && Object.keys(result.charts).length > 0 && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">📈 Visualizations</h6>
                                      <div className="space-y-4">
                                        {/* Render all charts directly */}
                                        {Object.entries(result.charts).map(([chartName, chartData]) => (
                                          chartData && chartName !== 'error' && renderChart(chartName, chartData)
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}

                              {/* Univariate Anomaly Detection Node Details */}
                              {node?.type === 'univariate_anomaly_detection' && result.anomaly_results && (
                                <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
                                  <h5 className="font-bold text-yellow-700 mb-4 flex items-center text-lg">⚠️ Univariate Anomaly Detection</h5>
                                  
                                  {/* Dataset Info */}
                                  {result.anomaly_results.dataset_info && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-yellow-700 mb-2">📊 Analysis Overview</h6>
                                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Total Rows</div>
                                          <div className="font-bold text-2xl text-yellow-700">{result.anomaly_results.dataset_info.total_rows?.toLocaleString() || 'N/A'}</div>
                                        </div>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Numeric Columns</div>
                                          <div className="font-bold text-2xl text-blue-600">{result.anomaly_results.dataset_info.numeric_columns || 'N/A'}</div>
                                        </div>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Columns Analyzed</div>
                                          <div className="font-bold text-2xl text-green-600">{result.anomaly_results.dataset_info.columns_analyzed?.length || 'N/A'}</div>
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Anomaly Summary */}
                                  {result.anomaly_results.combined_analysis && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-yellow-700 mb-2">🔍 Anomaly Summary</h6>
                                      <div className="bg-white p-3 rounded border max-h-64 overflow-y-auto">
                                        <div className="space-y-3">
                                          {Object.entries(result.anomaly_results.combined_analysis).map(([column, info]) => (
                                            <div key={column} className="border-b border-gray-200 pb-2 last:border-b-0">
                                              <div className="flex justify-between items-center mb-1">
                                                <span className="font-semibold text-sm text-gray-700">{column}</span>
                                                <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                                                  {info.total_unique_anomalies || 0} anomalies
                                                </span>
                                              </div>
                                              <div className="text-xs text-gray-600">
                                                Anomaly Rate: {info.anomaly_percentage?.toFixed(2) || 0}%
                                              </div>
                                              {info.high_confidence_anomalies && info.high_confidence_anomalies.length > 0 && (
                                                <div className="text-xs text-red-600 mt-1">
                                                  {info.high_confidence_anomalies.length} high-confidence anomalies
                                                </div>
                                              )}
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Charts with Column Selection */}
                                  {result.charts && Object.keys(result.charts).length > 0 && (
                                    <UnivariateAnomalyPanel result={result} />
                                  )}
                                </div>
                              )}

                              {/* Multivariate Anomaly Detection Node Details */}
                              {node?.type === 'multivariate_anomaly_detection' && result.anomaly_results && (
                                <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
                                  <h5 className="font-bold text-red-700 mb-4 flex items-center text-lg">🔍 Multivariate Anomaly Detection</h5>
                                  
                                  {/* Create scrollable container for all multivariate content */}
                                  <div className="max-h-[500px] overflow-y-auto space-y-4 pr-2">
                                    {/* Dataset Info */}
                                    {result.anomaly_results.dataset_info && (
                                      <div>
                                        <h6 className="font-semibold text-red-700 mb-2">📊 Analysis Overview</h6>
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                          <div className="bg-white p-3 rounded border text-center shadow-sm">
                                            <div className="text-xs text-gray-500 mb-1">Total Rows</div>
                                            <div className="font-bold text-2xl text-red-700">{result.anomaly_results.dataset_info.total_rows?.toLocaleString() || 'N/A'}</div>
                                          </div>
                                          <div className="bg-white p-3 rounded border text-center shadow-sm">
                                            <div className="text-xs text-gray-500 mb-1">Features Used</div>
                                            <div className="font-bold text-2xl text-blue-600">
                                              {result.anomaly_results?.dataset_info?.numeric_columns || 
                                               result.results?.dataset_info?.numeric_columns ||
                                               result.dataset_info?.columns_analyzed?.length || 'N/A'}
                                            </div>
                                          </div>
                                          <div className="bg-white p-3 rounded border text-center shadow-sm">
                                            <div className="text-xs text-gray-500 mb-1">Total Anomalies</div>
                                            <div className="font-bold text-2xl text-orange-600">
                                              {result.anomaly_results?.combined_analysis?.total_unique_anomalies || 
                                               result.results?.combined_analysis?.total_unique_anomalies ||
                                               result.combined_analysis?.total_unique_anomalies || 'N/A'}
                                            </div>
                                          </div>
                                          <div className="bg-white p-3 rounded border text-center shadow-sm">
                                            <div className="text-xs text-gray-500 mb-1">Anomaly Rate</div>
                                            <div className="font-bold text-2xl text-green-600">
                                              {(result.anomaly_results?.combined_analysis?.anomaly_percentage || 
                                                result.results?.combined_analysis?.anomaly_percentage ||
                                                result.combined_analysis?.anomaly_percentage)?.toFixed(2) || 'N/A'}%
                                            </div>
                                          </div>
                                        </div>
                                      </div>
                                    )}

                                  {/* Method Performance */}
                                  {result.anomaly_results.method_performance && (
                                    <div>
                                      <h6 className="font-semibold text-red-700 mb-2">🎯 Detection Methods Performance</h6>
                                      <div className="bg-white p-3 rounded border max-h-48 overflow-y-auto">
                                        <div className="space-y-3">
                                          {Object.entries(result.anomaly_results.method_performance).map(([method, performance]) => (
                                            <div key={method} className="border-b border-gray-200 pb-2 last:border-b-0">
                                              <div className="flex justify-between items-center mb-1">
                                                <span className="font-semibold text-sm text-gray-700 capitalize">{method.replace('_', ' ')}</span>
                                                <span className="text-xs bg-red-100 text-red-800 px-2 py-1 rounded">
                                                  {performance.anomalies_detected || 0} detected
                                                </span>
                                              </div>
                                              <div className="text-xs text-gray-600">
                                                Detection Rate: {performance.detection_rate?.toFixed(2) || 0}%
                                              </div>
                                              {performance.confidence_score && (
                                                <div className="text-xs text-blue-600 mt-1">
                                                  Confidence: {performance.confidence_score.toFixed(3)}
                                                </div>
                                              )}
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Recommendations */}
                                  {((result.recommendations || result.results?.recommendations) && 
                                    (result.recommendations?.length > 0 || result.results?.recommendations?.length > 0)) && (
                                    <div>
                                      <h6 className="font-semibold text-red-700 mb-2">💡 Recommendations</h6>
                                      <div className="bg-white p-3 rounded border max-h-48 overflow-y-auto">
                                        <div className="space-y-2">
                                          {(result.recommendations || result.results?.recommendations)?.map((recommendation, idx) => (
                                            <div key={idx} className="flex items-start space-x-2 p-2 bg-red-50 rounded">
                                              <div className="flex-shrink-0 w-6 h-6 bg-red-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                                                {idx + 1}
                                              </div>
                                              <div className="flex-1 text-sm text-gray-700">
                                                {recommendation}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Advanced Analysis Results */}
                                  {(result.mutual_information_analysis || result.results?.mutual_information_analysis) && (
                                    <div>
                                      <h6 className="font-semibold text-red-700 mb-2">🧠 Mutual Information Analysis</h6>
                                      <div className="bg-white p-3 rounded border">
                                        {(result.mutual_information_analysis?.summary || result.results?.mutual_information_analysis?.summary) && (
                                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Total Pairs</div>
                                              <div className="font-bold text-lg text-blue-600">
                                                {result.mutual_information_analysis?.summary?.total_pairs || result.results?.mutual_information_analysis?.summary?.total_pairs}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Max MI</div>
                                              <div className="font-bold text-lg text-green-600">
                                                {(result.mutual_information_analysis?.summary?.max_mutual_info || result.results?.mutual_information_analysis?.summary?.max_mutual_info)?.toFixed(3)}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Avg MI</div>
                                              <div className="font-bold text-lg text-orange-600">
                                                {(result.mutual_information_analysis?.summary?.avg_mutual_info || result.results?.mutual_information_analysis?.summary?.avg_mutual_info)?.toFixed(3)}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Min MI</div>
                                              <div className="font-bold text-lg text-purple-600">
                                                {(result.mutual_information_analysis?.summary?.min_mutual_info || result.results?.mutual_information_analysis?.summary?.min_mutual_info)?.toFixed(3)}
                                              </div>
                                            </div>
                                          </div>
                                        )}
                                        {(result.mutual_information_analysis?.top_relationships || result.results?.mutual_information_analysis?.top_relationships) && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Top Mutual Information Relationships:</div>
                                            <div className="space-y-1 max-h-32 overflow-y-auto">
                                              {(result.mutual_information_analysis?.top_relationships || result.results?.mutual_information_analysis?.top_relationships)?.map((rel, idx) => (
                                                <div key={idx} className="flex justify-between items-center text-xs bg-gray-50 p-2 rounded">
                                                  <span className="font-medium">{rel.relationship}</span>
                                                  <span className="font-bold text-blue-600">{rel.mutual_info?.toFixed(4)}</span>
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Hierarchical Clustering Results */}
                                  {(result.hierarchical_clustering || result.results?.hierarchical_clustering) && (
                                    <div>
                                      <h6 className="font-semibold text-red-700 mb-2">🌳 Hierarchical Clustering Analysis</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="mb-3">
                                          <div className="text-sm font-medium text-gray-700 mb-2">
                                            Number of Clusters: {result.hierarchical_clustering?.n_clusters || result.results?.hierarchical_clustering?.n_clusters}
                                          </div>
                                        </div>
                                        {(result.hierarchical_clustering?.clusters || result.results?.hierarchical_clustering?.clusters) && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Feature Clusters:</div>
                                            <div className="space-y-2 max-h-32 overflow-y-auto">
                                              {(result.hierarchical_clustering?.clusters || result.results?.hierarchical_clustering?.clusters)?.map((cluster, idx) => (
                                                <div key={idx} className="bg-gray-50 p-2 rounded">
                                                  <div className="flex justify-between items-center mb-1">
                                                    <span className="font-medium text-xs">Cluster {cluster.cluster_id}</span>
                                                    <span className="text-xs text-gray-500">{cluster.size} features</span>
                                                  </div>
                                                  <div className="text-xs text-gray-600">
                                                    {cluster.features.join(', ')}
                                                  </div>
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Cross-Correlation Analysis */}
                                  {(result.cross_correlation_analysis || result.results?.cross_correlation_analysis) && (
                                    <div>
                                      <h6 className="font-semibold text-red-700 mb-2">📊 Cross-Correlation Analysis</h6>
                                      <div className="bg-white p-3 rounded border">
                                        {(result.cross_correlation_analysis?.summary || result.results?.cross_correlation_analysis?.summary) && (
                                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Total Pairs</div>
                                              <div className="font-bold text-lg text-blue-600">
                                                {result.cross_correlation_analysis?.summary?.total_pairs || result.results?.cross_correlation_analysis?.summary?.total_pairs}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Significant</div>
                                              <div className="font-bold text-lg text-green-600">
                                                {result.cross_correlation_analysis?.summary?.significant_correlations || result.results?.cross_correlation_analysis?.summary?.significant_correlations}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Strong (&gt;0.7)</div>
                                              <div className="font-bold text-lg text-orange-600">
                                                {result.cross_correlation_analysis?.summary?.strong_correlations || result.results?.cross_correlation_analysis?.summary?.strong_correlations}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Avg Correlation</div>
                                              <div className="font-bold text-lg text-purple-600">
                                                {(result.cross_correlation_analysis?.summary?.avg_correlation || result.results?.cross_correlation_analysis?.summary?.avg_correlation)?.toFixed(3)}
                                              </div>
                                            </div>
                                          </div>
                                        )}
                                        {(result.cross_correlation_analysis?.top_correlations || result.results?.cross_correlation_analysis?.top_correlations) && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Top Correlations:</div>
                                            <div className="space-y-1 max-h-32 overflow-y-auto">
                                              {(result.cross_correlation_analysis?.top_correlations || result.results?.cross_correlation_analysis?.top_correlations)?.slice(0, 5).map((corr, idx) => (
                                                <div key={idx} className="flex justify-between items-center text-xs bg-gray-50 p-2 rounded">
                                                  <span className="font-medium">{corr.feature1} ↔ {corr.feature2}</span>
                                                  <span className={`font-bold ${corr.correlation >= 0 ? 'text-blue-600' : 'text-red-600'}`}>
                                                    {corr.correlation?.toFixed(4)}
                                                  </span>
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Variance Change Analysis */}
                                  {(result.variance_change_analysis || result.results?.variance_change_analysis) && (
                                    <div>
                                      <h6 className="font-semibold text-red-700 mb-2">📉 Variance Change Analysis</h6>
                                      <div className="bg-white p-3 rounded border">
                                        {(result.variance_change_analysis?.summary || result.results?.variance_change_analysis?.summary) && (
                                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Total Features</div>
                                              <div className="font-bold text-lg text-blue-600">
                                                {result.variance_change_analysis?.summary?.total_features || result.results?.variance_change_analysis?.summary?.total_features}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Unreliable</div>
                                              <div className="font-bold text-lg text-red-600">
                                                {result.variance_change_analysis?.summary?.unreliable_count || result.results?.variance_change_analysis?.summary?.unreliable_count}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Avg Change</div>
                                              <div className="font-bold text-lg text-orange-600">
                                                {(result.variance_change_analysis?.summary?.avg_variance_change || result.results?.variance_change_analysis?.summary?.avg_variance_change)?.toFixed(2)}
                                              </div>
                                            </div>
                                            <div className="text-center">
                                              <div className="text-xs text-gray-500">Threshold</div>
                                              <div className="font-bold text-lg text-purple-600">
                                                {(result.variance_change_analysis?.threshold || result.results?.variance_change_analysis?.threshold)?.toFixed(2)}
                                              </div>
                                            </div>
                                          </div>
                                        )}
                                        {((result.variance_change_analysis?.unreliable_sensors || result.results?.variance_change_analysis?.unreliable_sensors) && 
                                          (result.variance_change_analysis?.unreliable_sensors?.length > 0 || result.results?.variance_change_analysis?.unreliable_sensors?.length > 0)) && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Potentially Unreliable Sensors:</div>
                                            <div className="space-y-1">
                                              {(result.variance_change_analysis?.unreliable_sensors || result.results?.variance_change_analysis?.unreliable_sensors)?.map((sensor, idx) => (
                                                <div key={idx} className="inline-block bg-red-100 text-red-800 px-2 py-1 rounded text-xs mr-2 mb-1">
                                                  {sensor}
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Charts */}
                                  {((result.charts || result.results?.charts) && 
                                    Object.keys(result.charts || result.results?.charts || {}).length > 0) && (
                                    <div>
                                      <h6 className="font-semibold text-red-700 mb-2">📈 Multivariate Analysis Visualizations</h6>
                                      <div className="space-y-4">
                                        {Object.entries(result.charts || result.results?.charts || {}).map(([chartName, chartData]) => (
                                          chartData && renderChart(chartName, chartData)
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  </div> {/* End of scrollable container */}
                                </div>
                              )}

                              {/* Event Detection Node Details */}
                              {node?.type === 'event_detection' && result.event_results && (
                                <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg">
                                  <h5 className="font-bold text-purple-700 mb-4 flex items-center text-lg">⚡ Event Detection</h5>
                                  
                                  {/* Dataset Info */}
                                  {result.event_results.dataset_info && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">📊 Analysis Overview</h6>
                                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Total Rows</div>
                                          <div className="font-bold text-2xl text-purple-700">{result.event_results.dataset_info.total_rows?.toLocaleString() || 'N/A'}</div>
                                        </div>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Numeric Columns</div>
                                          <div className="font-bold text-2xl text-blue-600">{result.event_results.dataset_info.numeric_columns || 'N/A'}</div>
                                        </div>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Columns Analyzed</div>
                                          <div className="font-bold text-2xl text-green-600">{result.event_results.dataset_info.columns_analyzed?.length || 'N/A'}</div>
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Event Summary */}
                                  {result.event_results.comprehensive_summary && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">🔍 Event Summary</h6>
                                      <div className="bg-white p-3 rounded border max-h-64 overflow-y-auto">
                                        <div className="space-y-3">
                                          {result.event_results.comprehensive_summary.event_overview && Object.entries(result.event_results.comprehensive_summary.event_overview).map(([eventType, info]) => (
                                            eventType !== 'total_events_all_methods' && (
                                              <div key={eventType} className="border-b border-gray-200 pb-2 last:border-b-0">
                                                <div className="flex justify-between items-center mb-1">
                                                  <span className="font-semibold text-sm text-gray-700 capitalize">{eventType.replace('_', ' ')}</span>
                                                  <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">
                                                    {info.total_events || 0} events
                                                  </span>
                                                </div>
                                                <div className="text-xs text-gray-600">
                                                  Affected Columns: {info.affected_columns || 0}
                                                </div>
                                                <div className="text-xs text-gray-600">
                                                  Avg Events/Column: {info.avg_events_per_column?.toFixed(2) || 0}
                                                </div>
                                              </div>
                                            )
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Event Types Performance */}
                                  {result.event_results.combined_analysis && result.event_results.combined_analysis.method_comparison && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">🎯 Event Detection Performance</h6>
                                      <div className="bg-white p-3 rounded border max-h-48 overflow-y-auto">
                                        <div className="space-y-3">
                                          {Object.entries(result.event_results.combined_analysis.method_comparison).map(([method, performance]) => (
                                            <div key={method} className="border-b border-gray-200 pb-2 last:border-b-0">
                                              <div className="flex justify-between items-center mb-1">
                                                <span className="font-semibold text-sm text-gray-700 capitalize">{method.replace('_', ' ')}</span>
                                                <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">
                                                  {performance.total_events || 0} events
                                                </span>
                                              </div>
                                              <div className="text-xs text-gray-600">
                                                Affected Columns: {performance.affected_columns || 0}
                                              </div>
                                              <div className="text-xs text-gray-600">
                                                Max Events in Column: {performance.max_events_in_column || 0}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Recommendations */}
                                  {result.recommendations && result.recommendations.length > 0 && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">💡 Recommendations</h6>
                                      <div className="bg-white p-3 rounded border max-h-48 overflow-y-auto">
                                        <div className="space-y-2">
                                          {result.recommendations.map((recommendation, idx) => (
                                            <div key={idx} className="flex items-start space-x-2 p-2 bg-purple-50 rounded">
                                              <div className="flex-shrink-0 w-6 h-6 bg-purple-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                                                {idx + 1}
                                              </div>
                                              <div className="flex-1 text-sm text-gray-700">
                                                {typeof recommendation === 'object' ? recommendation.description : recommendation}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Charts */}
                                  {result.charts && Object.keys(result.charts).length > 0 && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-purple-700 mb-2">📈 Event Detection Visualizations</h6>
                                      <div className="space-y-4">
                                        {Object.entries(result.charts).map(([chartName, chartData]) => (
                                          chartData && renderChart(chartName, chartData)
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}

                              {/* Feature Engineering Node Details */}
                              {node?.type === 'feature_engineering' && result.result_summary && (
                                <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                                  <h5 className="font-bold text-blue-700 mb-4 flex items-center text-lg">🛠️ Feature Engineering Analysis</h5>
                                  
                                  {/* Key Metrics */}
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                                    {result.result_summary.features_added !== undefined && (
                                      <div className="bg-white p-3 rounded border text-center shadow-sm">
                                        <div className="text-xs text-gray-500 mb-1">Features Added</div>
                                        <div className="font-bold text-2xl text-blue-700">{result.result_summary.features_added}</div>
                                        <div className="text-xs text-gray-400">new features</div>
                                      </div>
                                    )}
                                    {result.result_summary.operations_performed && (
                                      <div className="bg-white p-3 rounded border text-center shadow-sm">
                                        <div className="text-xs text-gray-500 mb-1">Operations</div>
                                        <div className="font-bold text-2xl text-green-600">{result.result_summary.operations_performed.length}</div>
                                        <div className="text-xs text-gray-400">performed</div>
                                      </div>
                                    )}
                                    {result.result_summary.feature_types && (
                                      <>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Numerical</div>
                                          <div className="font-bold text-2xl text-purple-600">{result.result_summary.feature_types.numerical || 0}</div>
                                          <div className="text-xs text-gray-400">features</div>
                                        </div>
                                        <div className="bg-white p-3 rounded border text-center shadow-sm">
                                          <div className="text-xs text-gray-500 mb-1">Categorical</div>
                                          <div className="font-bold text-2xl text-orange-600">{result.result_summary.feature_types.categorical || 0}</div>
                                          <div className="text-xs text-gray-400">features</div>
                                        </div>
                                      </>
                                    )}
                                  </div>

                                  {/* Feature Shape Changes */}
                                  {(result.result_summary.original_shape && result.result_summary.final_shape) && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-blue-700 mb-2">📊 Feature Set Changes</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="flex items-center justify-between">
                                          <div className="text-center">
                                            <div className="text-sm font-medium text-gray-700">Original Features</div>
                                            <div className="text-lg font-bold text-gray-800">
                                              {result.result_summary.original_shape[1]}
                                            </div>
                                            <div className="text-xs text-gray-500">columns</div>
                                          </div>
                                          <div className="text-2xl text-gray-400">→</div>
                                          <div className="text-center">
                                            <div className="text-sm font-medium text-gray-700">Final Features</div>
                                            <div className="text-lg font-bold text-blue-700">
                                              {result.result_summary.final_shape[1]}
                                            </div>
                                            <div className="text-xs text-gray-500">columns</div>
                                          </div>
                                          <div className="text-center">
                                            <div className="text-sm font-medium text-gray-700">Increase</div>
                                            <div className="text-lg font-bold text-green-600">
                                              +{result.result_summary.features_added || 0}
                                            </div>
                                            <div className="text-xs text-gray-500">features</div>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Feature Type Distribution */}
                                  {result.result_summary.feature_types && (
                                    <div className="mb-4">
                                      <h6 className="font-semibold text-blue-700 mb-2">📈 Feature Type Distribution</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="grid grid-cols-3 gap-4">
                                          <div className="text-center">
                                            <div className="h-16 bg-purple-100 rounded-lg flex items-center justify-center mb-2">
                                              <span className="text-2xl font-bold text-purple-600">{result.result_summary.feature_types.numerical || 0}</span>
                                            </div>
                                            <div className="text-sm font-medium text-gray-700">Numerical</div>
                                            <div className="text-xs text-gray-500">Continuous values</div>
                                          </div>
                                          <div className="text-center">
                                            <div className="h-16 bg-orange-100 rounded-lg flex items-center justify-center mb-2">
                                              <span className="text-2xl font-bold text-orange-600">{result.result_summary.feature_types.categorical || 0}</span>
                                            </div>
                                            <div className="text-sm font-medium text-gray-700">Categorical</div>
                                            <div className="text-xs text-gray-500">Discrete categories</div>
                                          </div>
                                          <div className="text-center">
                                            <div className="h-16 bg-green-100 rounded-lg flex items-center justify-center mb-2">
                                              <span className="text-2xl font-bold text-green-600">{result.result_summary.feature_types.boolean || 0}</span>
                                            </div>
                                            <div className="text-sm font-medium text-gray-700">Boolean</div>
                                            <div className="text-xs text-gray-500">True/False values</div>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Detailed Operations */}
                                  {result.result_summary.operations_performed && result.result_summary.operations_performed.length > 0 && (
                                    <div className="mb-2">
                                      <h6 className="font-semibold text-blue-700 mb-2">🔧 Engineering Operations Performed</h6>
                                      <div className="bg-white p-3 rounded border max-h-64 overflow-y-auto">
                                        <div className="space-y-2">
                                          {result.result_summary.operations_performed.map((operation, idx) => (
                                            <div key={idx} className="flex items-start space-x-2 p-2 bg-gray-50 rounded">
                                              <div className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                                                {idx + 1}
                                              </div>
                                              <div className="flex-1 text-sm text-gray-700">
                                                {operation}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                              {/* Enhanced Statistics for Descriptive Stats */}
                              {node?.type === 'descriptive_stats' && result.result_summary && (
                                <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                                  <h5 className="font-medium text-blue-700 mb-4">Statistical Analysis Results</h5>
                                  
                                  {/* ...existing descriptive stats code... */}
                                  {/* Summary Information */}
                                  {result.result_summary.statistics_summary && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Dataset Overview</h6>
                                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                        {Object.entries(result.result_summary.statistics_summary).map(([key, value]) => (
                                          <div key={key} className="bg-white p-3 rounded border">
                                            <div className="text-xs text-gray-500 mb-1">{key.replace(/_/g, ' ')}</div>
                                            <div className="font-bold text-lg">
                                              {typeof value === 'number' ? value.toLocaleString() : String(value)}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Column Analysis */}
                                  {result.result_summary.column_info && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Column Types</h6>
                                      <div className="grid grid-cols-3 gap-3">
                                        <div className="bg-blue-100 text-blue-800 p-3 rounded border text-center">
                                          <div className="font-bold text-2xl">{result.result_summary.column_info.numeric || 0}</div>
                                          <div className="text-sm">Numeric</div>
                                        </div>
                                        <div className="bg-green-100 text-green-800 p-3 rounded border text-center">
                                          <div className="font-bold text-2xl">{result.result_summary.column_info.categorical || 0}</div>
                                          <div className="text-sm">Categorical</div>
                                        </div>
                                        <div className="bg-purple-100 text-purple-800 p-3 rounded border text-center">
                                          <div className="font-bold text-2xl">{result.result_summary.column_info.datetime || 0}</div>
                                          <div className="text-sm">DateTime</div>
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Detailed Statistics by Column */}
                                  {result.result_summary.basic_stats && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Statistical Details</h6>
                                      <div className="space-y-3 max-h-80 overflow-y-auto">
                                        {Object.entries(result.result_summary.basic_stats).map(([column, stats]) => (
                                          <div key={column} className="bg-white p-3 rounded border">
                                            <div className="font-medium text-sm mb-2 text-blue-700">{column}</div>
                                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                                              {Object.entries(stats).map(([statName, statValue]) => (
                                                <div key={statName} className="bg-gray-50 p-2 rounded">
                                                  <div className="text-gray-600 capitalize">{statName}:</div>
                                                  <div className="font-medium">
                                                    {typeof statValue === 'number' ? 
                                                      (Number.isInteger(statValue) ? statValue.toLocaleString() : statValue.toFixed(3)) : 
                                                      String(statValue)
                                                    }
                                                  </div>
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}

                                  {/* Data Types */}
                                  {result.result_summary.data_types && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Data Types</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                                          {Object.entries(result.result_summary.data_types).map(([column, dataType]) => (
                                            <div key={column} className="flex justify-between items-center py-1">
                                              <span className="font-medium">{column}:</span>
                                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                                dataType.includes('int') || dataType.includes('float') ? 'bg-blue-100 text-blue-700' :
                                                dataType.includes('object') ? 'bg-green-100 text-green-700' :
                                                dataType.includes('datetime') ? 'bg-purple-100 text-purple-700' :
                                                'bg-gray-100 text-gray-700'
                                              }`}>
                                                {dataType}
                                              </span>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Missing Values */}
                                  {result.result_summary.missing_values && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Missing Values Analysis</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                                          {Object.entries(result.result_summary.missing_values).map(([column, missingCount]) => (
                                            <div key={column} className="flex justify-between items-center py-1">
                                              <span className="font-medium">{column}:</span>
                                              <div className="flex items-center space-x-2">
                                                <span className="font-mono text-xs">{missingCount}</span>
                                                <span className={`px-2 py-1 rounded text-xs font-medium ${
                                                  missingCount > 0 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                                                }`}>
                                                  {missingCount > 0 ? 'Missing' : 'Complete'}
                                                </span>
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Top Correlations */}
                                  {result.result_summary.correlations && (
                                    <div>
                                      <h6 className="font-medium mb-3">Top Correlations</h6>
                                      <div className="bg-white p-3 rounded border">
                                        {(() => {
                                          const correlations = result.result_summary.correlations;
                                          const correlationPairs = [];
                                          
                                          Object.keys(correlations).forEach(col1 => {
                                            Object.keys(correlations[col1]).forEach(col2 => {
                                              if (col1 !== col2) {
                                                const corrValue = correlations[col1][col2];
                                                if (corrValue !== null && !isNaN(corrValue)) {
                                                  correlationPairs.push({
                                                    pair: `${col1} ↔ ${col2}`,
                                                    value: corrValue
                                                  });
                                                }
                                              }
                                            });
                                          });
                                          
                                          const topCorrelations = correlationPairs
                                            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
                                            .slice(0, 8);
                                          
                                          return (
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                                              {topCorrelations.map((corr, idx) => (
                                                <div key={idx} className="flex justify-between items-center py-2 px-3 bg-gray-50 rounded">
                                                  <span className="font-medium text-xs">{corr.pair}</span>
                                                  <span className={`px-2 py-1 rounded text-xs font-bold ${
                                                    Math.abs(corr.value) > 0.7 ? 'bg-red-100 text-red-700' :
                                                    Math.abs(corr.value) > 0.5 ? 'bg-orange-100 text-orange-700' :
                                                    Math.abs(corr.value) > 0.3 ? 'bg-yellow-100 text-yellow-700' :
                                                    'bg-gray-100 text-gray-700'
                                                  }`}>
                                                    {corr.value.toFixed(3)}
                                                  </span>
                                                </div>
                                              ))}
                                            </div>
                                          );
                                        })()}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}                              
                              {/* ENHANCED CLUSTERING RESULTS */}
                              {node?.type === 'clustering' && result.result_summary && (
                                <div className="bg-cyan-50 border border-cyan-200 p-6 rounded-lg">
                                  <h5 className="font-bold text-cyan-800 mb-6 text-xl flex items-center">
                                    🔬 Detailed Clustering Analysis
                                  </h5>
                                  
                                  {/* Algorithm Information */}
                                  <div className="bg-white p-4 rounded-lg mb-4 border">
                                    <h6 className="font-semibold mb-3 text-gray-800">Algorithm Configuration</h6>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                      <div className="text-center">
                                        <div className="text-sm text-gray-600">Algorithm</div>
                                        <div className="font-bold text-lg text-cyan-700">{result.result_summary.algorithm || 'Unknown'}</div>
                                      </div>
                                      {result.result_summary.feature_columns_used && (
                                        <div className="text-center">
                                          <div className="text-sm text-gray-600">Features Used</div>
                                          <div className="font-bold text-lg text-cyan-700">{result.result_summary.feature_columns_used.length}</div>
                                          <div className="text-xs text-gray-500 mt-1">{result.result_summary.feature_columns_used.join(', ')}</div>
                                        </div>
                                      )}
                                      {result.result_summary.scaled_features !== undefined && (
                                        <div className="text-center">
                                          <div className="text-sm text-gray-600">Feature Scaling</div>
                                          <div className="font-bold text-lg text-cyan-700">{result.result_summary.scaled_features ? 'Enabled' : 'Disabled'}</div>
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                  
                                  {/* Core Metrics */}
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                                    {result.result_summary.n_clusters !== undefined && (
                                      <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-4 rounded-lg text-center">
                                        <div className="font-bold text-3xl">{result.result_summary.n_clusters}</div>
                                        <div className="text-sm opacity-90">Clusters</div>
                                      </div>
                                    )}
                                    
                                    {result.result_summary.silhouette_score !== undefined && (
                                      <div className="bg-gradient-to-br from-green-500 to-green-600 text-white p-4 rounded-lg text-center">
                                        <div className="font-bold text-3xl">{(result.result_summary.silhouette_score * 100).toFixed(1)}%</div>
                                        <div className="text-sm opacity-90">Silhouette Score</div>
                                      </div>
                                    )}
                                    
                                    {result.result_summary.n_samples && (
                                      <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white p-4 rounded-lg text-center">
                                        <div className="font-bold text-3xl">{result.result_summary.n_samples.toLocaleString()}</div>
                                        <div className="text-sm opacity-90">Data Points</div>
                                      </div>
                                    )}
                                    
                                    {result.result_summary.inertia !== undefined && (
                                      <div className="bg-gradient-to-br from-orange-500 to-orange-600 text-white p-4 rounded-lg text-center">
                                        <div className="font-bold text-3xl">{result.result_summary.inertia.toFixed(0)}</div>
                                        <div className="text-sm opacity-90">Inertia (WCSS)</div>
                                      </div>
                                    )}
                                  </div>
                                  
                                  {/* Cluster Distribution */}
                                  {result.result_summary.cluster_distribution && (
                                    <div className="bg-white p-4 rounded-lg mb-4 border">
                                      <h6 className="font-semibold mb-3 text-gray-800">Cluster Distribution</h6>
                                      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                                        {Object.entries(result.result_summary.cluster_distribution)
                                          .sort(([a], [b]) => a === '-1' ? 1 : b === '-1' ? -1 : parseInt(a) - parseInt(b))
                                          .map(([cluster, count]) => (
                                          <div key={cluster} className="bg-cyan-50 border border-cyan-200 p-3 rounded-lg text-center">
                                            <div className="text-sm text-gray-600 mb-1">
                                              {cluster === '-1' ? 'Noise' : `Cluster ${cluster}`}
                                            </div>
                                            <div className="font-bold text-xl text-cyan-700">{count}</div>
                                            <div className="text-xs text-gray-500">
                                              {((count / result.result_summary.n_samples) * 100).toFixed(1)}%
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                              
                              {/* Enhanced Results for Classification/Regression */}
                              {(node?.type === 'classification' || node?.type === 'regression') && result.result_summary && (
                                <div className={`border p-4 rounded-lg ${
                                  node.type === 'classification' ? 'bg-green-50 border-green-200' : 'bg-blue-50 border-blue-200'
                                }`}>
                                  <h5 className={`font-medium mb-4 flex items-center ${
                                    node.type === 'classification' ? 'text-green-700' : 'text-blue-700'
                                  }`}>
                                    <Target className="w-5 h-5 mr-2" />
                                    {node.type === 'classification' ? 'Classification Results' : 'Regression Results'}
                                  </h5>
                                  
                                  {/* Model Performance Metrics */}
                                  {result.result_summary.metrics && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Model Performance</h6>
                                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        {Object.entries(result.result_summary.metrics).map(([metric, value]) => {
                                          const isPercent = metric.includes('accuracy') || metric.includes('precision') || 
                                                          metric.includes('recall') || metric.includes('f1') || 
                                                          metric.includes('score') || metric.includes('r2')
                                          const displayValue = typeof value === 'number' ? 
                                            (isPercent ? (value * 100).toFixed(2) + '%' : value.toFixed(4)) : 
                                            String(value)
                                          
                                          return (
                                            <div key={metric} className="bg-white p-3 rounded border">
                                              <div className="text-xs text-gray-500 mb-1 capitalize">
                                                {metric.replace(/_/g, ' ')}
                                              </div>
                                              <div className={`font-bold text-lg ${
                                                node.type === 'classification' ? 'text-green-700' : 'text-blue-700'
                                              }`}>
                                                {displayValue}
                                              </div>
                                            </div>
                                          )
                                        })}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Target Information */}
                                  {result.result_summary.target_info && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Target Column Analysis</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                          <div>
                                            <div className="text-sm font-medium text-gray-700">Column: {result.result_summary.target_info.column}</div>
                                            <div className="text-xs text-gray-500 mt-1">Type: {result.result_summary.target_info.type}</div>
                                          </div>
                                          
                                          {node.type === 'classification' && result.result_summary.target_info.classes && (
                                            <div>
                                              <div className="text-sm font-medium text-gray-700">Classes ({result.result_summary.target_info.classes.length}):</div>
                                              <div className="text-xs text-gray-600 mt-1">
                                                {result.result_summary.target_info.classes.slice(0, 5).join(', ')}
                                                {result.result_summary.target_info.classes.length > 5 && 
                                                  ` (+${result.result_summary.target_info.classes.length - 5} more)`
                                                }
                                              </div>
                                            </div>
                                          )}
                                          
                                          {result.result_summary.target_info.binning_applied && (
                                            <div className="col-span-2">
                                              <div className="text-xs bg-yellow-100 text-yellow-800 p-2 rounded border border-yellow-300">
                                                ℹ️ Target column was binned for classification: {result.result_summary.target_info.binning_method}
                                                {result.result_summary.target_info.bins && (
                                                  <div className="mt-1">Bins: {result.result_summary.target_info.bins.join(', ')}</div>
                                                )}
                                              </div>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Feature Information */}
                                  {result.result_summary.feature_info && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Feature Analysis</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3">
                                          <div className="text-center">
                                            <div className="font-bold text-2xl text-blue-600">
                                              {result.result_summary.feature_info.total_features || 0}
                                            </div>
                                            <div className="text-xs text-gray-600">Total Features</div>
                                          </div>
                                          <div className="text-center">
                                            <div className="font-bold text-2xl text-green-600">
                                              {result.result_summary.feature_info.numeric_features || 0}
                                            </div>
                                            <div className="text-xs text-gray-600">Numeric</div>
                                          </div>
                                          <div className="text-center">
                                            <div className="font-bold text-2xl text-purple-600">
                                              {result.result_summary.feature_info.categorical_features || 0}
                                            </div>
                                            <div className="text-xs text-gray-600">Categorical</div>
                                          </div>
                                        </div>
                                        
                                        {result.result_summary.feature_info.feature_names && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Features Used:</div>
                                            <div className="text-xs text-gray-600">
                                              {result.result_summary.feature_info.feature_names.slice(0, 10).join(', ')}
                                              {result.result_summary.feature_info.feature_names.length > 10 && 
                                                ` (+${result.result_summary.feature_info.feature_names.length - 10} more)`
                                              }
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Model Information */}
                                  {result.result_summary.model_info && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Model Details</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                          {result.result_summary.model_info.algorithm && (
                                            <div>
                                              <div className="text-xs text-gray-500">Algorithm</div>
                                              <div className="font-medium">{result.result_summary.model_info.algorithm}</div>
                                            </div>
                                          )}
                                          {result.result_summary.model_info.training_samples && (
                                            <div>
                                              <div className="text-xs text-gray-500">Training Samples</div>
                                              <div className="font-medium">{result.result_summary.model_info.training_samples.toLocaleString()}</div>
                                            </div>
                                          )}
                                          {result.result_summary.model_info.test_samples && (
                                            <div>
                                              <div className="text-xs text-gray-500">Test Samples</div>
                                              <div className="font-medium">{result.result_summary.model_info.test_samples.toLocaleString()}</div>
                                            </div>
                                          )}
                                          {result.result_summary.model_info.cross_validation && (
                                            <div>
                                              <div className="text-xs text-gray-500">Cross Validation</div>
                                              <div className="font-medium">{result.result_summary.model_info.cross_validation}</div>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Confusion Matrix for Classification */}
                                  {node.type === 'classification' && result.result_summary.confusion_matrix && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Confusion Matrix</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="text-xs text-gray-600 mb-2">
                                          True labels vs Predicted labels
                                        </div>
                                        <div className="overflow-x-auto">
                                          <table className="min-w-full border border-gray-200">
                                            <thead>
                                              <tr>
                                                <th className="border border-gray-200 p-2 bg-gray-50 text-xs"></th>
                                                {result.result_summary.confusion_matrix.labels.map(label => (
                                                  <th key={label} className="border border-gray-200 p-2 bg-green-100 text-xs font-medium">
                                                    {label}
                                                  </th>
                                                ))}
                                              </tr>
                                            </thead>
                                            <tbody>
                                              {result.result_summary.confusion_matrix.matrix.map((row, i) => (
                                                <tr key={i}>
                                                  <th className="border border-gray-200 p-2 bg-green-100 text-xs font-medium">
                                                    {result.result_summary.confusion_matrix.labels[i]}
                                                  </th>
                                                  {row.map((value, j) => (
                                                    <td key={j} className={`border border-gray-200 p-2 text-xs text-center ${
                                                      i === j ? 'bg-green-50 font-bold' : 'bg-white'
                                                    }`}>
                                                      {value}
                                                    </td>
                                                  ))}
                                                </tr>
                                              ))}
                                            </tbody>
                                          </table>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Clustering Results */}
                                  {console.log('Checking clustering condition:', {
                                    nodeType: node.type,
                                    hasResultSummary: !!result.result_summary,
                                    resultSummaryKeys: result.result_summary ? Object.keys(result.result_summary) : [],
                                    algorithm: result.result_summary?.algorithm
                                  })}
                                  {node.type === 'clustering' && result.result_summary && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3 text-lg text-blue-800 bg-blue-100 p-2 rounded">🔬 Clustering Results ✅</h6>
                                      <div className="bg-white p-3 rounded border space-y-4">
                                        
                                        {/* Algorithm and Basic Info */}
                                        <div className="bg-gray-50 p-3 rounded">
                                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            <div>
                                              <div className="text-xs text-gray-600">Algorithm</div>
                                              <div className="font-bold text-gray-800">{result.result_summary.algorithm || 'Unknown'}</div>
                                            </div>
                                            {result.result_summary.feature_columns_used && (
                                              <div>
                                                <div className="text-xs text-gray-600">Features Used</div>
                                                <div className="font-bold text-gray-800">{result.result_summary.feature_columns_used.join(', ')}</div>
                                              </div>
                                            )}
                                            {result.result_summary.scaled_features !== undefined && (
                                              <div>
                                                <div className="text-xs text-gray-600">Feature Scaling</div>
                                                <div className="font-bold text-gray-800">{result.result_summary.scaled_features ? 'Enabled' : 'Disabled'}</div>
                                              </div>
                                            )}
                                          </div>
                                        </div>
                                        
                                        {/* Core Clustering Metrics */}
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                          {result.result_summary.n_clusters !== undefined && (
                                            <div className="bg-cyan-50 p-3 rounded-lg text-center">
                                              <div className="font-bold text-2xl text-cyan-600">
                                                {result.result_summary.n_clusters}
                                              </div>
                                              <div className="text-xs text-gray-600">Clusters Found</div>
                                            </div>
                                          )}
                                          {result.result_summary.silhouette_score !== undefined && (
                                            <div className="bg-blue-50 p-3 rounded-lg text-center">
                                              <div className="font-bold text-2xl text-blue-600">
                                                {(result.result_summary.silhouette_score * 100).toFixed(1)}%
                                              </div>
                                              <div className="text-xs text-gray-600">Silhouette Score</div>
                                            </div>
                                          )}
                                          {result.result_summary.n_samples && (
                                            <div className="bg-purple-50 p-3 rounded-lg text-center">
                                              <div className="font-bold text-2xl text-purple-600">
                                                {result.result_summary.n_samples.toLocaleString()}
                                              </div>
                                              <div className="text-xs text-gray-600">Samples</div>
                                            </div>
                                          )}
                                          {result.result_summary.feature_columns_used && (
                                            <div className="bg-green-50 p-3 rounded-lg text-center">
                                              <div className="font-bold text-2xl text-green-600">
                                                {result.result_summary.feature_columns_used.length}
                                              </div>
                                              <div className="text-xs text-gray-600">Features Used</div>
                                            </div>
                                          )}
                                        </div>
                                        
                                        {/* Algorithm-Specific Metrics */}
                                        {(result.result_summary.metrics || result.result_summary.inertia !== undefined || result.result_summary.n_noise_points !== undefined || result.result_summary.aic !== undefined) && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Algorithm-Specific Metrics:</div>
                                            <div className="bg-gray-50 p-3 rounded">
                                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                                {/* K-Means Inertia */}
                                                {(result.result_summary.inertia !== undefined || result.result_summary.metrics?.inertia !== undefined) && (
                                                  <div className="text-center">
                                                    <div className="text-xs text-gray-600">Inertia (WCSS)</div>
                                                    <div className="font-bold text-orange-600">
                                                      {((result.result_summary.inertia ?? result.result_summary.metrics?.inertia) || 0).toFixed(2)}
                                                    </div>
                                                  </div>
                                                )}
                                                
                                                {/* DBSCAN Noise Points */}
                                                {(result.result_summary.n_noise_points !== undefined || result.result_summary.metrics?.n_noise_points !== undefined) && (
                                                  <div className="text-center">
                                                    <div className="text-xs text-gray-600">Noise Points</div>
                                                    <div className="font-bold text-red-600">
                                                      {result.result_summary.n_noise_points || result.result_summary.metrics?.n_noise_points}
                                                    </div>
                                                  </div>
                                                )}
                                                
                                                {/* DBSCAN Parameters */}
                                                {result.result_summary.eps !== undefined && (
                                                  <div className="text-center">
                                                    <div className="text-xs text-gray-600">Epsilon (eps)</div>
                                                    <div className="font-bold text-indigo-600">{result.result_summary.eps}</div>
                                                  </div>
                                                )}
                                                
                                                {result.result_summary.min_samples !== undefined && (
                                                  <div className="text-center">
                                                    <div className="text-xs text-gray-600">Min Samples</div>
                                                    <div className="font-bold text-indigo-600">{result.result_summary.min_samples}</div>
                                                  </div>
                                                )}
                                                
                                                {/* Gaussian Mixture AIC/BIC */}
                                                {(result.result_summary.aic !== undefined || result.result_summary.metrics?.aic !== undefined) && (
                                                  <div className="text-center">
                                                    <div className="text-xs text-gray-600">AIC Score</div>
                                                    <div className="font-bold text-teal-600">
                                                      {((result.result_summary.aic ?? result.result_summary.metrics?.aic) || 0).toFixed(2)}
                                                    </div>
                                                  </div>
                                                )}
                                                
                                                {(result.result_summary.bic !== undefined || result.result_summary.metrics?.bic !== undefined) && (
                                                  <div className="text-center">
                                                    <div className="text-xs text-gray-600">BIC Score</div>
                                                    <div className="font-bold text-teal-600">
                                                      {((result.result_summary.bic ?? result.result_summary.metrics?.bic) || 0).toFixed(2)}
                                                    </div>
                                                  </div>
                                                )}
                                              </div>
                                            </div>
                                          </div>
                                        )}
                                        
                                        {/* Cluster Distribution */}
                                        {result.result_summary.cluster_distribution && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Cluster Distribution:</div>
                                            <div className="bg-gray-50 p-3 rounded">
                                              <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
                                                {Object.entries(result.result_summary.cluster_distribution)
                                                  .sort(([a], [b]) => a === '-1' ? 1 : b === '-1' ? -1 : parseInt(a) - parseInt(b))
                                                  .map(([cluster, count]) => (
                                                  <div key={cluster} className="text-center p-2 bg-white rounded border">
                                                    <div className="text-xs text-gray-600 mb-1">
                                                      {cluster === '-1' ? 'Noise' : `Cluster ${cluster}`}
                                                    </div>
                                                    <div className="font-bold text-cyan-600">{count}</div>
                                                    <div className="text-xs text-gray-500">
                                                      ({((count / result.result_summary.n_samples) * 100).toFixed(1)}%)
                                                    </div>
                                                  </div>
                                                ))}
                                              </div>
                                            </div>
                                          </div>
                                        )}
                                        
                                        {/* Cluster Visualization */}
                                        {result.result_summary.cluster_plot && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Cluster Visualization:</div>
                                            <div className="bg-gray-50 p-3 rounded text-center">
                                              <img 
                                                src={`data:image/png;base64,${result.result_summary.cluster_plot}`} 
                                                alt="Cluster Plot" 
                                                className="max-w-full h-auto mx-auto rounded border"
                                                style={{ maxHeight: '400px' }}
                                              />
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Debug: Show all clustering data regardless of condition */}
                                  {node.type === 'clustering' && (
                                    <div className="mb-4 bg-red-100 border-4 border-red-400 p-6 rounded-lg">
                                      <h6 className="font-bold mb-3 text-red-800 text-xl">🐛 DEBUG: Clustering Data Analysis</h6>
                                      <div className="text-sm text-red-700 space-y-3 bg-white p-4 rounded">
                                        <div><strong>Node Type:</strong> "{node.type}"</div>
                                        <div><strong>Node Type Check:</strong> {node.type === 'clustering' ? '✅ MATCHES' : '❌ NO MATCH'}</div>
                                        <div><strong>Has Result Summary:</strong> {result.result_summary ? '✅ YES' : '❌ NO'}</div>
                                        <div><strong>Result Summary Type:</strong> {typeof result.result_summary}</div>
                                        {result.result_summary && (
                                          <div className="bg-green-50 p-3 rounded mt-2">
                                            <div><strong>Algorithm:</strong> {result.result_summary.algorithm || '❌ Not found'}</div>
                                            <div><strong>N Clusters:</strong> {result.result_summary.n_clusters || '❌ Not found'}</div>
                                            <div><strong>Silhouette Score:</strong> {result.result_summary.silhouette_score || '❌ Not found'}</div>
                                            <div><strong>N Samples:</strong> {result.result_summary.n_samples || '❌ Not found'}</div>
                                            <div><strong>Has Cluster Plot:</strong> {result.result_summary.cluster_plot ? '✅ YES' : '❌ NO'}</div>
                                            <div><strong>Cluster Distribution:</strong> {result.result_summary.cluster_distribution ? '✅ YES' : '❌ NO'}</div>
                                            <div><strong>Available Keys:</strong> {Object.keys(result.result_summary).join(', ')}</div>
                                          </div>
                                        )}
                                        <div className="bg-blue-50 p-3 rounded mt-2">
                                          <div><strong>Full Result Keys:</strong> {Object.keys(result).join(', ')}</div>
                                          <div><strong>Result Type:</strong> {result.result_type}</div>
                                          <div><strong>Result Status:</strong> {result.status}</div>
                                        </div>
                                        <div className="bg-yellow-50 p-3 rounded mt-2">
                                          <div><strong>Condition Result:</strong> {(node.type === 'clustering' && result.result_summary) ? '✅ SHOULD SHOW CLUSTERING' : '❌ CONDITION FAILED'}</div>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* ALWAYS SHOW CLUSTERING - FORCED DISPLAY */}
                                  {result.result_summary && result.result_summary.algorithm && (
                                    <div className="mb-4 bg-green-100 border-4 border-green-600 p-6 rounded-lg">
                                      <h6 className="font-bold mb-3 text-green-800 text-xl">🔬 FORCED CLUSTERING RESULTS ✅</h6>
                                      <div className="bg-white p-4 rounded space-y-4">
                                        
                                        {/* Algorithm and Basic Info */}
                                        <div className="bg-gray-50 p-3 rounded">
                                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            <div>
                                              <div className="text-xs text-gray-600">Algorithm</div>
                                              <div className="font-bold text-gray-800">{result.result_summary.algorithm || 'Unknown'}</div>
                                            </div>
                                            {result.result_summary.feature_columns_used && (
                                              <div>
                                                <div className="text-xs text-gray-600">Features Used</div>
                                                <div className="font-bold text-gray-800">{result.result_summary.feature_columns_used.join(', ')}</div>
                                              </div>
                                            )}
                                            {result.result_summary.scaled_features !== undefined && (
                                              <div>
                                                <div className="text-xs text-gray-600">Feature Scaling</div>
                                                <div className="font-bold text-gray-800">{result.result_summary.scaled_features ? 'Enabled' : 'Disabled'}</div>
                                              </div>
                                            )}
                                          </div>
                                        </div>
                                        
                                        {/* Core Clustering Metrics */}
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                          {result.result_summary.n_clusters !== undefined && (
                                            <div className="bg-cyan-50 p-3 rounded-lg text-center">
                                              <div className="font-bold text-2xl text-cyan-600">
                                                {result.result_summary.n_clusters}
                                              </div>
                                              <div className="text-xs text-gray-600">Clusters Found</div>
                                            </div>
                                          )}
                                          {result.result_summary.silhouette_score !== undefined && (
                                            <div className="bg-blue-50 p-3 rounded-lg text-center">
                                              <div className="font-bold text-2xl text-blue-600">
                                                {(result.result_summary.silhouette_score * 100).toFixed(1)}%
                                              </div>
                                              <div className="text-xs text-gray-600">Silhouette Score</div>
                                            </div>
                                          )}
                                          {result.result_summary.n_samples && (
                                            <div className="bg-purple-50 p-3 rounded-lg text-center">
                                              <div className="font-bold text-2xl text-purple-600">
                                                {result.result_summary.n_samples.toLocaleString()}
                                              </div>
                                              <div className="text-xs text-gray-600">Samples</div>
                                            </div>
                                          )}
                                          {result.result_summary.feature_columns_used && (
                                            <div className="bg-green-50 p-3 rounded-lg text-center">
                                              <div className="font-bold text-2xl text-green-600">
                                                {result.result_summary.feature_columns_used.length}
                                              </div>
                                              <div className="text-xs text-gray-600">Features Used</div>
                                            </div>
                                          )}
                                        </div>
                                        
                                        {/* Inertia for K-Means */}
                                        {result.result_summary.inertia !== undefined && (
                                          <div className="bg-orange-50 p-3 rounded-lg text-center">
                                            <div className="text-sm font-medium text-gray-700 mb-1">K-Means Inertia (WCSS)</div>
                                            <div className="font-bold text-xl text-orange-600">
                                              {result.result_summary.inertia.toFixed(2)}
                                            </div>
                                          </div>
                                        )}
                                        
                                        {/* Cluster Distribution */}
                                        {result.result_summary.cluster_distribution && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Cluster Distribution:</div>
                                            <div className="bg-gray-50 p-3 rounded">
                                              <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
                                                {Object.entries(result.result_summary.cluster_distribution)
                                                  .sort(([a], [b]) => a === '-1' ? 1 : b === '-1' ? -1 : parseInt(a) - parseInt(b))
                                                  .map(([cluster, count]) => (
                                                  <div key={cluster} className="text-center p-2 bg-white rounded border">
                                                    <div className="text-xs text-gray-600 mb-1">
                                                      {cluster === '-1' ? 'Noise' : `Cluster ${cluster}`}
                                                    </div>
                                                    <div className="font-bold text-cyan-600">{count}</div>
                                                    <div className="text-xs text-gray-500">
                                                      ({((count / result.result_summary.n_samples) * 100).toFixed(1)}%)
                                                    </div>
                                                  </div>
                                                ))}
                                              </div>
                                            </div>
                                          </div>
                                        )}
                                        
                                        {/* Cluster Visualization */}
                                        {result.result_summary.cluster_plot && (
                                          <div>
                                            <div className="text-sm font-medium text-gray-700 mb-2">Cluster Visualization:</div>
                                            <div className="bg-gray-50 p-3 rounded text-center">
                                              <img 
                                                src={`data:image/png;base64,${result.result_summary.cluster_plot}`} 
                                                alt="Cluster Plot" 
                                                className="max-w-full h-auto mx-auto rounded border"
                                                style={{ maxHeight: '400px' }}
                                              />
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Feature Importance */}
                                  {result.result_summary.feature_importance && (
                                    <div className="mb-4">
                                      <h6 className="font-medium mb-3">Feature Importance</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="space-y-2">
                                          {result.result_summary.feature_importance.slice(0, 10).map(([feature, importance]) => (
                                            <div key={feature} className="flex items-center justify-between">
                                              <span className="text-sm font-medium">{feature}</span>
                                              <div className="flex items-center space-x-2">
                                                <div className="w-20 bg-gray-200 rounded-full h-2">
                                                  <div 
                                                    className={`h-2 rounded-full ${
                                                      node.type === 'classification' ? 'bg-green-500' : 'bg-blue-500'
                                                    }`}
                                                    style={{ width: `${(importance * 100)}%` }}
                                                  ></div>
                                                </div>
                                                <span className="text-xs text-gray-600 w-12 text-right">
                                                  {(importance * 100).toFixed(1)}%
                                                </span>
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Predictions Sample */}
                                  {result.result_summary.predictions_sample && (
                                    <div>
                                      <h6 className="font-medium mb-3">Sample Predictions</h6>
                                      <div className="bg-white p-3 rounded border">
                                        <div className="overflow-x-auto">
                                          <table className="min-w-full text-xs">
                                            <thead>
                                              <tr className="bg-gray-50">
                                                <th className="border border-gray-200 p-2 text-left">Actual</th>
                                                <th className="border border-gray-200 p-2 text-left">Predicted</th>
                                                {node.type === 'classification' && (
                                                  <th className="border border-gray-200 p-2 text-left">Confidence</th>
                                                )}
                                              </tr>
                                            </thead>
                                            <tbody>
                                              {result.result_summary.predictions_sample.slice(0, 5).map((pred, idx) => (
                                                <tr key={idx} className={pred.actual === pred.predicted ? 'bg-green-50' : 'bg-red-50'}>
                                                  <td className="border border-gray-200 p-2">{pred.actual}</td>
                                                  <td className="border border-gray-200 p-2">{pred.predicted}</td>
                                                  {node.type === 'classification' && pred.confidence && (
                                                    <td className="border border-gray-200 p-2">
                                                      {(pred.confidence * 100).toFixed(1)}%
                                                    </td>
                                                  )}
                                                </tr>
                                              ))}
                                            </tbody>
                                          </table>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}

                              {/* Charts Section - Only for nodes without specific chart rendering */}
                              {result.charts && Object.keys(result.charts).length > 0 && 
                               node?.type !== 'eda_analysis' && 
                               node?.type !== 'univariate_anomaly_detection' && 
                               node?.type !== 'multivariate_anomaly_detection' && (
                                <div className="bg-white border rounded-lg p-4">
                                  <h5 className="font-medium mb-4">Generated Visualizations</h5>
                                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                    {(() => {
                                      if (Array.isArray(result.charts)) {
                                        return result.charts.map((chart, index) => 
                                          renderChart(`${chart.type}_${chart.column}_${index}`, chart.image)
                                        )
                                      } else if (typeof result.charts === 'object') {
                                        return Object.entries(result.charts).map(([chartName, chartData]) => 
                                          renderChart(chartName, chartData)
                                        )
                                      }
                                    })()}
                                  </div>
                                </div>
                              )}
                              
                              {/* Error Section */}
                              {result.error && (
                                <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
                                  <h5 className="font-medium text-red-700 mb-2">Error Details</h5>
                                  <pre className="text-sm text-red-600 whitespace-pre-wrap font-mono bg-white p-3 rounded border">
                                    {result.error}
                                  </pre>
                                </div>
                              )}
                            </div>
                          </div>
                        )
                      }).filter(Boolean)
                    })()}
                    
                    {/* Execution Logs - Always show section for debugging */}
                    <div className="mt-6 sm:mt-8 pt-4 sm:pt-6 border-t">
                      <h4 className="font-medium mb-3 sm:mb-4 flex items-center space-x-2 text-sm sm:text-base">
                        <FileText className="w-4 h-4" />
                        <span>Execution Logs</span>
                        <Badge variant="secondary" className="text-xs">{logs.length}</Badge>
                      </h4>
                      <div className="bg-black text-green-400 p-3 sm:p-4 rounded-lg font-mono text-xs sm:text-sm max-h-48 sm:max-h-64 overflow-y-auto border">
                        {logs.length === 0 ? (
                          <div className="text-gray-400 italic">No execution logs available</div>
                        ) : (
                          logs.map((log, index) => (
                            <div key={index} className={`mb-1 leading-relaxed ${
                              log.includes('ERROR') ? 'text-red-400' :
                              log.includes('WARNING') ? 'text-yellow-400' :
                              log.includes('INFO') ? 'text-blue-400' :
                              'text-green-400'
                            }`}>
                              <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    )
  }

  const renderCreateWorkflowDialog = () => {
    return (
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Plus className="w-5 h-5 text-blue-600" />
              <span>Create New Workflow</span>
            </DialogTitle>
            <DialogDescription>
              Create a new data science workflow. Choose a template to get started quickly or start with a blank canvas.
              <br />
              <span className="text-xs text-gray-400 mt-2 block">
                💡 Pro tip: Use <kbd className="px-1.5 py-0.5 text-xs bg-gray-100 rounded border">Ctrl+N</kbd> to quickly create a new workflow, 
                <kbd className="px-1.5 py-0.5 text-xs bg-gray-100 rounded border">Ctrl++</kbd> to zoom in, 
                <kbd className="px-1.5 py-0.5 text-xs bg-gray-100 rounded border">Ctrl+-</kbd> to zoom out
              </span>
            </DialogDescription>
          </DialogHeader>
          
          <div className="grid gap-6 py-4">
            {/* Workflow Name */}
            <div className="grid gap-2">
              <Label htmlFor="workflow-name" className="text-sm font-medium">
                Workflow Name *
              </Label>
              <Input
                id="workflow-name"
                placeholder="Enter a descriptive name for your workflow"
                value={newWorkflowName}
                onChange={(e) => setNewWorkflowName(e.target.value)}
                className={`w-full transition-colors ${
                  newWorkflowName.trim() ? 'border-green-300 focus:border-green-500' : ''
                }`}
                maxLength={100}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && newWorkflowName.trim()) {
                    handleCreateWorkflow()
                  }
                }}
              />
              <p className={`text-xs transition-colors ${
                newWorkflowName.length > 80 ? 'text-orange-500' : 'text-gray-500'
              }`}>
                {newWorkflowName.length}/100 characters
              </p>
            </div>

            {/* Workflow Description */}
            <div className="grid gap-2">
              <Label htmlFor="workflow-description" className="text-sm font-medium">
                Description (Optional)
              </Label>
              <Textarea
                id="workflow-description"
                placeholder="Describe what this workflow will accomplish..."
                value={newWorkflowDescription}
                onChange={(e) => setNewWorkflowDescription(e.target.value)}
                className="w-full h-20 resize-none"
                maxLength={500}
              />
              <p className={`text-xs transition-colors ${
                newWorkflowDescription.length > 400 ? 'text-orange-500' : 'text-gray-500'
              }`}>
                {newWorkflowDescription.length}/500 characters
              </p>
            </div>

            {/* Template Selection */}
            <div className="grid gap-3">
              <Label className="text-sm font-medium">Choose Template</Label>
              <div className="grid grid-cols-1 gap-3">
                {/* Blank Template */}
                <div 
                  className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                    workflowTemplate === 'blank' 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => setWorkflowTemplate('blank')}
                >
                  <div className="flex items-start space-x-3">
                    <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                      <FileText className="w-5 h-5 text-gray-600" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-sm">Blank Workflow</h4>
                      <p className="text-xs text-gray-500 mt-1">
                        Start with an empty canvas and build your workflow from scratch
                      </p>
                    </div>
                    {workflowTemplate === 'blank' && (
                      <CheckCircle className="w-5 h-5 text-blue-600" />
                    )}
                  </div>
                </div>

                {/* Data Analysis Template */}
                <div 
                  className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                    workflowTemplate === 'data_analysis' 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => setWorkflowTemplate('data_analysis')}
                >
                  <div className="flex items-start space-x-3">
                    <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                      <BarChart3 className="w-5 h-5 text-green-600" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-sm">Data Analysis</h4>
                      <p className="text-xs text-gray-500 mt-1">
                        Pre-configured with Data Source → EDA → Visualization nodes
                      </p>
                    </div>
                    {workflowTemplate === 'data_analysis' && (
                      <CheckCircle className="w-5 h-5 text-blue-600" />
                    )}
                  </div>
                </div>

                {/* ML Pipeline Template */}
                <div 
                  className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                    workflowTemplate === 'ml_pipeline' 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => setWorkflowTemplate('ml_pipeline')}
                >
                  <div className="flex items-start space-x-3">
                    <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                      <Zap className="w-5 h-5 text-purple-600" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-sm">Machine Learning Pipeline</h4>
                      <p className="text-xs text-gray-500 mt-1">
                        Complete ML workflow with Data → Preprocessing → Model → Evaluation
                      </p>
                    </div>
                    {workflowTemplate === 'ml_pipeline' && (
                      <CheckCircle className="w-5 h-5 text-blue-600" />
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Template Preview */}
            {workflowTemplate !== 'blank' && (
              <div className="bg-gray-50 rounded-lg p-4 border">
                <h4 className="text-sm font-medium mb-3 flex items-center space-x-2">
                  <Eye className="w-4 h-4" />
                  <span>Template Preview</span>
                </h4>
                <div className="space-y-2">
                  {workflowTemplate === 'data_analysis' && (
                    <div className="text-xs text-gray-600">
                      <p className="font-medium mb-2">This template will add:</p>
                      <div className="space-y-1">
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-blue-500 rounded"></div>
                          <span>Data Source node - Connect to your datasets</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-green-500 rounded"></div>
                          <span>EDA node - Explore and analyze your data</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-purple-500 rounded"></div>
                          <span>Visualization node - Create charts and insights</span>
                        </div>
                      </div>
                    </div>
                  )}
                  {workflowTemplate === 'ml_pipeline' && (
                    <div className="text-xs text-gray-600">
                      <p className="font-medium mb-2">This template will add:</p>
                      <div className="space-y-1">
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-blue-500 rounded"></div>
                          <span>Data Source - Load your training data</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-yellow-500 rounded"></div>
                          <span>Preprocessing - Clean and prepare data</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-red-500 rounded"></div>
                          <span>Classification - Train ML models</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 bg-green-500 rounded"></div>
                          <span>Evaluation - Assess model performance</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="flex justify-end space-x-3 pt-4 border-t">
            <Button 
              variant="outline" 
              onClick={() => {
                setShowCreateDialog(false)
                setNewWorkflowName('')
                setNewWorkflowDescription('')
                setWorkflowTemplate('blank')
              }}
            >
              Cancel
            </Button>
            <Button 
              onClick={handleCreateWorkflow}
              disabled={!newWorkflowName.trim() || isCreatingWorkflow}
              className="bg-blue-600 hover:bg-blue-700"
            >
              {isCreatingWorkflow ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4 mr-2" />
                  Create Workflow
                </>
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    )
  }

  // Render AI Summary Panel
  const renderAISummaryPanel = () => {
    return (
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header for AI Summary panel */}
        <div className="border-b bg-white px-4 py-3 shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-purple-600" />
              <h3 className="font-medium text-purple-800">AI Summary</h3>
            </div>
            
            {/* Generate Summary Button */}
            <Button 
              onClick={generateAISummary}
              disabled={!executionResults || aiSummaryLoading}
              size="sm"
              className="bg-purple-600 hover:bg-purple-700 text-white"
            >
              {aiSummaryLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4 mr-2" />
                  Generate AI Summary
                </>
              )}
            </Button>
          </div>
          
          <p className="text-xs text-gray-500 mt-1">
            Generate an AI-powered analysis of your workflow execution results.
          </p>
        </div>

        {/* Content */}
        <div className="flex-1 p-6 bg-gray-50 overflow-y-auto min-h-0">
          {!executionResults ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-gray-500">
                <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">No Workflow Results</h3>
                <p className="text-sm">Run a workflow first to generate an AI summary</p>
              </div>
            </div>
          ) : !aiSummaryData ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-gray-500">
                <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">Ready to Generate AI Summary</h3>
                <p className="text-sm mb-4">Click the "Generate AI Summary" button to analyze your workflow results</p>
                <Button 
                  onClick={generateAISummary}
                  disabled={aiSummaryLoading}
                  className="bg-purple-600 hover:bg-purple-700 text-white"
                >
                  {aiSummaryLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 mr-2" />
                      Generate AI Summary
                    </>
                  )}
                </Button>
              </div>
            </div>
          ) : aiSummaryError ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-red-500">
                <AlertCircle className="w-16 h-16 mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">Error Generating Summary</h3>
                <p className="text-sm mb-4">{aiSummaryError}</p>
                <Button 
                  onClick={generateAISummary}
                  disabled={aiSummaryLoading}
                  variant="outline"
                  className="border-red-200 text-red-600 hover:bg-red-50"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Try Again
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* AI Summary Content */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Brain className="w-5 h-5 text-purple-600" />
                    <span>AI Analysis Results</span>
                    <Badge variant="default" className="bg-purple-600">
                      ✨ Generated
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Summary Overview */}
                    {aiSummaryData.overview && (
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">📊 Overview</h4>
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                          <p className="text-gray-700 whitespace-pre-wrap">{aiSummaryData.overview}</p>
                        </div>
                      </div>
                    )}
                    
                    {/* Key Insights */}
                    {aiSummaryData.insights && (
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">💡 Key Insights</h4>
                        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                          <p className="text-gray-700 whitespace-pre-wrap">{aiSummaryData.insights}</p>
                        </div>
                      </div>
                    )}
                    
                    {/* Streaming AI Summary Section */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-gray-800">🔄 Streaming AI Analysis</h4>
                        <Button 
                          onClick={generateStreamingAISummary}
                          disabled={!executionResults || isStreamingActive}
                          size="sm"
                          variant="outline"
                          className="border-purple-200 text-purple-600 hover:bg-purple-50"
                        >
                          {isStreamingActive ? (
                            <>
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                              Streaming...
                            </>
                          ) : (
                            <>
                              <Zap className="w-4 h-4 mr-2" />
                              Start Streaming
                            </>
                          )}
                        </Button>
                      </div>
                      
                      {streamingError ? (
                        <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                          <div className="flex items-center space-x-2 text-red-600 mb-2">
                            <AlertCircle className="w-4 h-4" />
                            <span className="font-medium">Streaming Error</span>
                          </div>
                          <p className="text-red-700 text-sm">{streamingError}</p>
                        </div>
                      ) : isStreamingActive ? (
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                          <div className="flex items-center space-x-2 text-purple-600 mb-2">
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            <span className="font-medium">Generating Streaming Analysis...</span>
                          </div>
                          <p className="text-purple-700 text-sm">
                            {streamingTaskId ? `Task ID: ${streamingTaskId}` : 'Starting background analysis...'}
                          </p>
                          {streamingAiSummary && (
                            <div className="mt-3 p-3 bg-white rounded border">
                              <p className="text-gray-700 whitespace-pre-wrap text-sm">{streamingAiSummary}</p>
                            </div>
                          )}
                        </div>
                      ) : streamingAiSummary ? (
                        <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-200">
                          <div className="flex items-center space-x-2 text-indigo-600 mb-2">
                            <CheckCircle className="w-4 h-4" />
                            <span className="font-medium">Streaming Analysis Complete</span>
                          </div>
                          <div className="text-indigo-800 whitespace-pre-wrap text-sm">{streamingAiSummary}</div>
                        </div>
                      ) : (
                        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                          <div className="flex items-center space-x-2 text-gray-600 mb-2">
                            <Info className="w-4 h-4" />
                            <span className="font-medium">Ready for Streaming Analysis</span>
                          </div>
                          <p className="text-gray-700 text-sm">
                            Click "Start Streaming" to generate advanced AI analysis using background processing.
                          </p>
                        </div>
                      )}
                    </div>
                    
                    {/* Recommendations */}
                    {aiSummaryData.recommendations && (
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">🎯 Recommendations</h4>
                        <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                          <p className="text-gray-700 whitespace-pre-wrap">{aiSummaryData.recommendations}</p>
                        </div>
                      </div>
                    )}
                    
                    {/* Full Summary */}
                    {aiSummaryData.summary && !aiSummaryData.overview && (
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">📝 AI Summary</h4>
                        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                          <p className="text-gray-700 whitespace-pre-wrap">{aiSummaryData.summary}</p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Regenerate Button */}
                  <div className="mt-6 pt-4 border-t flex justify-end">
                    <Button 
                      onClick={generateAISummary}
                      disabled={aiSummaryLoading}
                      variant="outline"
                      size="sm"
                    >
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Regenerate Summary
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-gray-100 overflow-hidden">
      {/* Custom CSS for advanced animations */}
      <style>{`
        @keyframes nodeFloat {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-2px) rotate(0.5deg); }
        }
        
        @keyframes connectionPulse {
          0%, 100% { opacity: 0.8; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.05); }
        }
        
        @keyframes nodeBounce {
          0%, 20%, 53%, 80%, 100% { transform: translate3d(0,0,0); }
          40%, 43% { transform: translate3d(0,-10px,0); }
          70% { transform: translate3d(0,-5px,0); }
          90% { transform: translate3d(0,-2px,0); }
        }
        
        .node-floating:hover {
          animation: nodeFloat 2s ease-in-out infinite;
        }
        
        .connection-pulse {
          animation: connectionPulse 2s ease-in-out infinite;
        }
        
        .node-bounce {
          animation: nodeBounce 0.6s ease-out;
        }
        
        .drag-preview {
          pointer-events: none;
          transform: rotate(3deg) scale(1.1);
          box-shadow: 0 20px 30px rgba(0,0,0,0.3);
        }
        
        /* Custom scrollbar styles - make scrollbar invisible */
        .scrollbar-hidden {
          scrollbar-width: none; /* Firefox */
          -ms-overflow-style: none; /* IE and Edge */
        }
        
        .scrollbar-hidden::-webkit-scrollbar {
          display: none; /* Chrome, Safari, Opera */
        }
        
        /* Line clamp utility for text truncation */
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        /* Responsive canvas panning and zooming */
        .canvas-container {
          position: relative;
          overflow: hidden;
          cursor: grab;
          user-select: none;
          -webkit-user-select: none;
          -moz-user-select: none;
          -ms-user-select: none;
          border: 3px solid #e5e7eb;
          border-radius: 12px;
          background-color: #ffffff;
          box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05);
          transition: border-color 0.2s ease;
        }
        
        /* Zoom limit feedback animation */
        .canvas-container.zoom-limit {
          animation: zoomLimitPulse 0.3s ease-in-out;
        }
        
        @keyframes zoomLimitPulse {
          0% { transform: scale(1); border-color: #e5e7eb; }
          50% { transform: scale(1.002); border-color: #f59e0b; }
          100% { transform: scale(1); border-color: #e5e7eb; }
        }
        
        .canvas-container:active {
          cursor: grabbing;
        }
        
        .canvas-container.panning {
          cursor: grabbing;
        }
        
        .canvas-container.panning * {
          pointer-events: none;
        }
        
        .canvas-container.cursor-crosshair {
          cursor: crosshair;
        }
        
        .canvas-container.cursor-grabbing {
          cursor: grabbing;
        }
        
        /* Smooth panning and zooming transitions */
        .canvas-container .absolute.inset-0.transition-transform {
          transition: transform 0.1s ease-out;
        }
        
        .canvas-container.panning .absolute.inset-0.transition-transform {
          transition: none;
        }
        
        /* Prevent text selection during panning */
        .canvas-container.panning {
          -webkit-touch-callout: none;
          -webkit-user-select: none;
          -khtml-user-select: none;
          -moz-user-select: none;
          -ms-user-select: none;
          user-select: none;
        }
        
        /* Zoom controls styling */
        .zoom-controls button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .zoom-controls button:disabled:hover {
          background-color: transparent;
        }

        /* Responsive breakpoints */
        @media (max-width: 768px) {
          .workflow-header {
            flex-direction: column;
            gap: 0.75rem;
            padding: 0.75rem;
          }
          
          .workflow-header h1 {
            font-size: 1.125rem;
          }
          
          .workflow-controls {
            flex-wrap: wrap;
            gap: 0.5rem;
          }
          
          .workflow-controls .select-trigger {
            width: 100%;
            min-width: 200px;
          }
        }
        
        @media (max-width: 1024px) {
          .node-library {
            width: 280px;
          }
          
          .properties-panel {
            width: 280px;
          }
        }
        
        @media (max-width: 640px) {
          .node-library {
            width: 100%;
            max-width: 300px;
          }
          
          .properties-panel {
            width: 100%;
            max-width: 300px;
          }
        }
      `}</style>
      
      {/* Header */}
      <div className="bg-white border-b p-2 sm:p-4 flex items-center justify-between workflow-header shrink-0">
        <div className="flex items-center space-x-2 sm:space-x-4 min-w-0">
          <h1 className="text-lg sm:text-xl font-bold truncate">Workflow Builder</h1>
          {currentWorkflow && (
            <Badge variant="outline" className="hidden sm:inline-flex">{currentWorkflow.name}</Badge>
          )}
        </div>

        <div className="flex items-center space-x-1 sm:space-x-2 workflow-controls">
          <Select value={currentWorkflow?.id?.toString() || ''} onValueChange={(value) => {
            const workflow = workflows.find(w => w.id.toString() === value)
            if (workflow) {
              setCurrentWorkflow(workflow)
              setNodes(workflow.nodes || [])
              setConnections(workflow.connections || [])
            }
          }}>
            <SelectTrigger className="w-32 sm:w-48 select-trigger">
              <SelectValue placeholder="Select workflow" />
            </SelectTrigger>
            <SelectContent>
              {workflows.map(workflow => (
                <SelectItem key={workflow.id} value={workflow.id.toString()}>
                  {workflow.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button onClick={createNewWorkflow} variant="outline" size="sm" className="bg-blue-50 hover:bg-blue-100 border-blue-200 text-blue-700 hidden sm:flex">
            <Plus className="w-4 h-4 mr-2" />
            New Workflow
          </Button>
          
          <Button onClick={createNewWorkflow} variant="outline" size="sm" className="bg-blue-50 hover:bg-blue-100 border-blue-200 text-blue-700 sm:hidden">
            <Plus className="w-4 h-4" />
          </Button>

          <Button onClick={saveWorkflow} variant="outline" size="sm" disabled={!currentWorkflow} className="hidden sm:flex">
            <Save className="w-4 h-4 mr-2" />
            Save
          </Button>
          
          <Button onClick={saveWorkflow} variant="outline" size="sm" disabled={!currentWorkflow} className="sm:hidden">
            <Save className="w-4 h-4" />
          </Button>

          <Button 
            onClick={runWorkflow} 
            disabled={isRunning || !currentWorkflow || nodes.length === 0}
            className="bg-green-600 hover:bg-green-700"
            size="sm"
          >
            {isRunning ? (
              <>
                <Clock className="w-4 h-4 mr-1 sm:mr-2 animate-spin" />
                <span className="hidden sm:inline">Running...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-1 sm:mr-2" />
                <span className="hidden sm:inline">Run</span>
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
          <div className="border-b bg-white px-2 sm:px-4 shrink-0">
            <TabsList className="grid w-full max-w-96 grid-cols-3">
              <TabsTrigger value="workflow" className="flex items-center space-x-2 text-sm">
                <GitBranch className="w-3 h-3 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">Workflow</span>
                <span className="sm:hidden">Flow</span>
              </TabsTrigger>
              <TabsTrigger value="results" className="flex items-center space-x-2 text-sm relative">
                <BarChart3 className="w-3 h-3 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">Results</span>
                <span className="sm:hidden">Results</span>
                
                {/* Results Count Badge */}
                {executionResults && (
                  <Badge variant="secondary" className="ml-1 text-xs">
                    {Object.keys(executionResults.execution_results?.results || {}).length}
                  </Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="ai-summary" className="flex items-center space-x-2 text-sm relative">
                <Brain className="w-3 h-3 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">AI Summary</span>
                <span className="sm:hidden">AI</span>
                
                {/* AI Summary Badge */}
                {aiSummaryData && (
                  <Badge variant="default" className="ml-1 text-xs bg-blue-500">
                    ✨
                  </Badge>
                )}
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="workflow" className="flex-1 flex mt-0 min-h-0">
            {renderNodeLibrary()}
            {renderCanvas()}
            {renderPropertiesPanel()}
          </TabsContent>

          <TabsContent value="results" className="flex-1 flex mt-0 min-h-0">
            {renderResultsPanel()}
          </TabsContent>

          <TabsContent value="ai-summary" className="flex-1 flex mt-0 min-h-0">
            {renderAISummaryPanel()}
          </TabsContent>
        </Tabs>
      </div>
      
      {/* Create Workflow Dialog */}
      {renderCreateWorkflowDialog()}
    </div>
  )
}

export default AdvancedWorkflowBuilder
export { AdvancedWorkflowBuilder }
