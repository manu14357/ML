import { useState, useEffect, useRef, useCallback } from 'react'
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
  X
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { toast } from 'sonner'

// Advanced Node types for data science workflows
const NODE_TYPES = {
  // Input/Data Sources
  DATA_SOURCE: {
    id: 'data_source',
    name: 'Data Source',
    icon: Database,
    color: 'bg-blue-500',
    category: 'input',
    description: 'Load data from CSV, JSON, API, or database',
    params: ['file_path', 'source_type', 'connection_string']
  },
  API_SOURCE: {
    id: 'api_source',
    name: 'API Source',
    icon: RefreshCw,
    color: 'bg-blue-600',
    category: 'input',
    description: 'Fetch data from REST APIs',
    params: ['url', 'headers', 'auth_type']
  },
  DATABASE_SOURCE: {
    id: 'database_source',
    name: 'Database',
    icon: Database,
    color: 'bg-blue-700',
    category: 'input',
    description: 'Connect to SQL/NoSQL databases',
    params: ['connection_string', 'query', 'database_type']
  },
  
  // Data Preprocessing
  DATA_CLEANING: {
    id: 'data_cleaning',
    name: 'Data Cleaning',
    icon: Filter,
    color: 'bg-green-500',
    category: 'preprocessing',
    description: 'Handle missing values, outliers, duplicates',
    params: ['missing_strategy', 'outlier_method', 'duplicate_action']
  },
  FEATURE_ENGINEERING: {
    id: 'feature_engineering',
    name: 'Feature Engineering',
    icon: Zap,
    color: 'bg-yellow-500',
    category: 'preprocessing',
    description: 'Create, transform, and select features',
    params: ['feature_methods', 'scaling_type', 'encoding_type']
  },
  DATA_VALIDATION: {
    id: 'data_validation',
    name: 'Data Validation',
    icon: CheckCircle,
    color: 'bg-green-600',
    category: 'preprocessing',
    description: 'Validate data quality and schema',
    params: ['validation_rules', 'quality_checks', 'schema_validation']
  },
  
  // Statistical Analysis
  DESCRIPTIVE_STATS: {
    id: 'descriptive_stats',
    name: 'Descriptive Statistics',
    icon: BarChart3,
    color: 'bg-purple-500',
    category: 'analysis',
    description: 'Generate summary statistics and distributions',
    params: ['stat_types', 'group_by', 'percentiles']
  },
  CORRELATION_ANALYSIS: {
    id: 'correlation_analysis',
    name: 'Correlation Analysis',
    icon: GitBranch,
    color: 'bg-purple-600',
    category: 'analysis',
    description: 'Analyze correlations and relationships',
    params: ['correlation_method', 'significance_level', 'target_variables']
  },
  HYPOTHESIS_TESTING: {
    id: 'hypothesis_testing',
    name: 'Hypothesis Testing',
    icon: Target,
    color: 'bg-purple-700',
    category: 'analysis',
    description: 'Perform statistical hypothesis tests',
    params: ['test_type', 'alpha_level', 'variables']
  },
  
  // Machine Learning - Supervised
  CLASSIFICATION: {
    id: 'classification',
    name: 'Classification',
    icon: Target,
    color: 'bg-red-500',
    category: 'ml_supervised',
    description: 'Binary/multi-class classification models',
    params: ['algorithm', 'target_column', 'features', 'hyperparameters']
  },
  REGRESSION: {
    id: 'regression',
    name: 'Regression',
    icon: BarChart3,
    color: 'bg-red-600',
    category: 'ml_supervised',
    description: 'Linear/non-linear regression models',
    params: ['algorithm', 'target_column', 'features', 'regularization']
  },
  TIME_SERIES: {
    id: 'time_series',
    name: 'Time Series',
    icon: Clock,
    color: 'bg-red-700',
    category: 'ml_supervised',
    description: 'Time series forecasting models',
    params: ['model_type', 'time_column', 'forecast_horizon', 'seasonality']
  },
  
  // Machine Learning - Unsupervised
  CLUSTERING: {
    id: 'clustering',
    name: 'Clustering',
    icon: GitBranch,
    color: 'bg-indigo-500',
    category: 'ml_unsupervised',
    description: 'K-means, hierarchical, DBSCAN clustering',
    params: ['algorithm', 'n_clusters', 'features', 'distance_metric']
  },
  DIMENSIONALITY_REDUCTION: {
    id: 'dimensionality_reduction',
    name: 'Dimensionality Reduction',
    icon: Zap,
    color: 'bg-indigo-600',
    category: 'ml_unsupervised',
    description: 'PCA, t-SNE, UMAP for dimension reduction',
    params: ['method', 'n_components', 'features', 'random_state']
  },
  ANOMALY_DETECTION: {
    id: 'anomaly_detection',
    name: 'Anomaly Detection',
    icon: AlertCircle,
    color: 'bg-indigo-700',
    category: 'ml_unsupervised',
    description: 'Detect outliers and anomalies',
    params: ['method', 'contamination', 'features', 'threshold']
  },
  
  // Deep Learning
  NEURAL_NETWORK: {
    id: 'neural_network',
    name: 'Neural Network',
    icon: GitBranch,
    color: 'bg-pink-500',
    category: 'deep_learning',
    description: 'Deep neural networks for complex patterns',
    params: ['architecture', 'layers', 'activation', 'optimizer']
  },
  CNN: {
    id: 'cnn',
    name: 'CNN',
    icon: Eye,
    color: 'bg-pink-600',
    category: 'deep_learning',
    description: 'Convolutional neural networks for images',
    params: ['input_shape', 'conv_layers', 'pooling', 'filters']
  },
  RNN_LSTM: {
    id: 'rnn_lstm',
    name: 'RNN/LSTM',
    icon: RefreshCw,
    color: 'bg-pink-700',
    category: 'deep_learning',
    description: 'Recurrent networks for sequences',
    params: ['cell_type', 'units', 'sequence_length', 'dropout']
  },
  
  // Model Evaluation
  MODEL_EVALUATION: {
    id: 'model_evaluation',
    name: 'Model Evaluation',
    icon: CheckCircle,
    color: 'bg-orange-500',
    category: 'evaluation',
    description: 'Evaluate model performance with metrics',
    params: ['metrics', 'cross_validation', 'test_size', 'stratify'
    ]
  },
  FEATURE_IMPORTANCE: {
    id: 'feature_importance',
    name: 'Feature Importance',
    icon: BarChart3,
    color: 'bg-orange-600',
    category: 'evaluation',
    description: 'Analyze feature importance and SHAP values',
    params: ['method', 'top_features', 'visualization', 'explainer_type']
  },
  MODEL_COMPARISON: {
    id: 'model_comparison',
    name: 'Model Comparison',
    icon: GitBranch,
    color: 'bg-orange-700',
    category: 'evaluation',
    description: 'Compare multiple models performance',
    params: ['models', 'comparison_metrics', 'statistical_tests', 'visualization']
  },
  
  // Visualization
  BASIC_PLOTS: {
    id: 'basic_plots',
    name: 'Basic Plots',
    icon: BarChart3,
    color: 'bg-teal-500',
    category: 'visualization',
    description: 'Histograms, scatter plots, box plots',
    params: ['plot_type', 'x_column', 'y_column', 'group_by']
  },
  ADVANCED_PLOTS: {
    id: 'advanced_plots',
    name: 'Advanced Plots',
    icon: Eye,
    color: 'bg-teal-600',
    category: 'visualization',
    description: 'Heatmaps, pair plots, 3D visualizations',
    params: ['plot_type', 'features', 'color_scheme', 'interactive']
  },
  DASHBOARD: {
    id: 'dashboard',
    name: 'Dashboard',
    icon: BarChart3,
    color: 'bg-teal-700',
    category: 'visualization',
    description: 'Interactive dashboards and reports',
    params: ['dashboard_type', 'widgets', 'filters', 'refresh_rate']
  },
  
  // AI/ML Special
  AI_SUMMARY: {
    id: 'ai_summary',
    name: 'AI Data Summary',
    icon: Target,
    color: 'bg-cyan-500',
    category: 'analysis',
    description: 'AI-powered insights and data quality analysis',
    params: ['analysis_depth', 'include_recommendations']
  },
  AUTO_ML: {
    id: 'auto_ml',
    name: 'AutoML',
    icon: Zap,
    color: 'bg-cyan-600',
    category: 'ml_supervised',
    description: 'Automated machine learning pipeline',
    params: ['target_column', 'problem_type', 'time_limit']
  },
  
  // Output
  EXPORT_DATA: {
    id: 'export_data',
    name: 'Export Data',
    icon: FileText,
    color: 'bg-gray-500',
    category: 'output',
    description: 'Export processed data to various formats including PDF reports',
    params: ['format', 'filename', 'include_index', 'compress']
  },
  SAVE_MODEL: {
    id: 'save_model',
    name: 'Save Model',
    icon: Save,
    color: 'bg-gray-600',
    category: 'output',
    description: 'Save trained models for deployment',
    params: ['model_format', 'file_path', 'metadata', 'versioning']
  },
  DEPLOY_MODEL: {
    id: 'deploy_model',
    name: 'Deploy Model',
    icon: Upload,
    color: 'bg-gray-700',
    category: 'output',
    description: 'Deploy models to production endpoints',
    params: ['deployment_type', 'endpoint_url', 'scaling', 'monitoring']
  }
}

export function WorkflowBuilder() {
  const [workflows, setWorkflows] = useState([])
  const [currentWorkflow, setCurrentWorkflow] = useState(null)
  const [nodes, setNodes] = useState([])
  const [connections, setConnections] = useState([])
  const [selectedNode, setSelectedNode] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [draggedNodeType, setDraggedNodeType] = useState(null)
  const [showNewWorkflowDialog, setShowNewWorkflowDialog] = useState(false)
  const [showNodeConfigDialog, setShowNodeConfigDialog] = useState(false)
  const [newWorkflowForm, setNewWorkflowForm] = useState({ name: '', description: '' })
  const [availableDatasets, setAvailableDatasets] = useState([])
  const [nodeTypes, setNodeTypes] = useState({})
  const [executionLogs, setExecutionLogs] = useState([])
  const [executionResults, setExecutionResults] = useState({})
  const [showResultsPanel, setShowResultsPanel] = useState(false)
  const [expandedDetails, setExpandedDetails] = useState(new Set())
  
  const canvasRef = useRef(null)
  const [canvasOffset, setCanvasOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [connecting, setConnecting] = useState(null)

  useEffect(() => {
    fetchWorkflows()
    fetchAvailableDatasets()
    fetchNodeTypes()
  }, [])

  const fetchWorkflows = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/workflow/workflows')
      if (response.ok) {
        const data = await response.json()
        setWorkflows(data.workflows || [])
        console.log('Workflows loaded:', data.workflows?.length || 0)
      } else {
        console.error('Failed to fetch workflows:', response.status)
        toast.error('Failed to load workflows')
      }
    } catch (error) {
      console.error('Error fetching workflows:', error)
      toast.error('Backend connection failed. Using demo mode.')
      // Set some demo workflows for testing
      setWorkflows([
        {
          id: 1,
          name: 'Demo ML Pipeline',
          description: 'Sample machine learning workflow',
          status: 'draft',
          updated_at: new Date().toISOString()
        },
        {
          id: 2,
          name: 'Demo EDA Workflow',
          description: 'Sample exploratory data analysis',
          status: 'completed',
          updated_at: new Date().toISOString()
        }
      ])
    }
  }

  const fetchAvailableDatasets = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/workflow/datasets')
      if (response.ok) {
        const data = await response.json()
        setAvailableDatasets(data.datasets || [])
        console.log('Available datasets loaded:', data.datasets?.length || 0)
      } else {
        console.error('Failed to fetch datasets:', response.status)
      }
    } catch (error) {
      console.error('Error fetching datasets:', error)
    }
  }

  const fetchNodeTypes = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/workflow/node-types')
      if (response.ok) {
        const data = await response.json()
        setNodeTypes(data.node_types || {})
        console.log('Node types loaded:', Object.keys(data.node_types || {}).length)
      } else {
        console.error('Failed to fetch node types:', response.status)
      }
    } catch (error) {
      console.error('Error fetching node types:', error)
    }
  }

  const createNewWorkflow = async () => {
    if (!newWorkflowForm.name.trim()) {
      toast.error('Please enter a workflow name')
      return
    }

    try {
      const response = await fetch('http://localhost:5000/api/workflow/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newWorkflowForm)
      })

      if (response.ok) {
        const data = await response.json()
        setCurrentWorkflow(data.workflow)
        setNodes([])
        setConnections([])
        setNewWorkflowForm({ name: '', description: '' })
        setShowNewWorkflowDialog(false)
        fetchWorkflows()
        toast.success('Workflow created successfully!')
      } else {
        toast.error('Failed to create workflow')
      }
    } catch (error) {
      console.error('Error creating workflow:', error)
      toast.error('Error creating workflow')
    }
  }

  const saveWorkflow = useCallback(async () => {
    if (!currentWorkflow) return

    try {
      const response = await fetch(`http://localhost:5000/api/workflow/workflows/${currentWorkflow.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nodes: nodes,
          connections: connections
        })
      })

      if (response.ok) {
        toast.success('Workflow saved successfully!')
      } else {
        toast.error('Failed to save workflow')
      }
    } catch (error) {
      console.error('Error saving workflow:', error)
      toast.error('Error saving workflow')
    }
  }, [currentWorkflow, nodes, connections])

  const runWorkflow = useCallback(async () => {
    if (!currentWorkflow || nodes.length === 0) {
      toast.error('Please add nodes to the workflow before running')
      return
    }

    setIsRunning(true)
    setExecutionLogs([])
    setExecutionResults({})
    setExpandedDetails(new Set())
    
    try {
      const response = await fetch(`http://localhost:5000/api/workflow/workflows/${currentWorkflow.id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nodes: nodes,
          connections: connections
        })
      })

      const data = await response.json()
      console.log('Workflow execution response:', data)
      
      if (response.ok && data.success) {
        const executionResults = data.execution_results || {}
        const summary = executionResults.summary || {}
        const results = executionResults.results || {}
        const logs = executionResults.logs || []
        
        // Store execution data
        setExecutionResults(executionResults)
        setExecutionLogs(logs)
        setShowResultsPanel(true)
        
        // Show success message with summary
        const nodesCompleted = summary.nodes_completed || Object.keys(results).length
        const totalNodes = summary.total_nodes || nodes.length
        const executionTime = summary.execution_time || 0
        
        toast.success(`Workflow executed successfully! (${nodesCompleted}/${totalNodes} nodes completed)`)
        
        // Update node statuses based on execution results
        setNodes(prev => prev.map(node => ({
          ...node,
          status: results[node.id]?.status || 'completed',
          lastExecuted: results[node.id]?.timestamp || null,
          executionTime: results[node.id]?.execution_time || 0,
          resultSummary: results[node.id]?.result_summary || null
        })))
        
        // Show execution summary
        if (executionTime > 0) {
          toast.info(`Execution completed in ${executionTime.toFixed(2)} seconds`)
        }
        
      } else {
        toast.error(data.error || 'Workflow execution failed')
        
        const executionResults = data.execution_results || {}
        const logs = executionResults.logs || []
        const results = executionResults.results || {}
        
        // Store execution data even for failures
        setExecutionResults(executionResults)
        setExecutionLogs(logs)
        setShowResultsPanel(true)
        
        // Update failed nodes
        setNodes(prev => prev.map(node => ({
          ...node,
          status: results[node.id]?.status || 'error'
        })))
      }
    } catch (error) {
      console.error('Error running workflow:', error)
      toast.error('Network error: Unable to execute workflow')
    } finally {
      setIsRunning(false)
    }
  }, [currentWorkflow, nodes, connections])

  const handleDragStart = (nodeType) => {
    setDraggedNodeType(nodeType)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (!draggedNodeType || !canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left - canvasOffset.x
    const y = e.clientY - rect.top - canvasOffset.y

    const newNode = {
      id: `node_${Date.now()}`,
      type: draggedNodeType.id,
      name: draggedNodeType.name,
      x: Math.max(0, x - 75), // Center the node
      y: Math.max(0, y - 40),
      config: {},
      status: 'idle'
    }

    setNodes(prev => [...prev, newNode])
    setDraggedNodeType(null)
  }

  const handleNodeClick = (node) => {
    setSelectedNode(node)
    setShowNodeConfigDialog(true)
  }

  const handleNodeDelete = (nodeId) => {
    setNodes(prev => prev.filter(n => n.id !== nodeId))
    setConnections(prev => prev.filter(c => c.source !== nodeId && c.target !== nodeId))
  }

  const handleCanvasMouseDown = (e) => {
    if (e.target === canvasRef.current) {
      setIsDragging(true)
      setDragStart({ x: e.clientX - canvasOffset.x, y: e.clientY - canvasOffset.y })
    }
  }

  const handleCanvasMouseMove = (e) => {
    if (isDragging) {
      setCanvasOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    }
  }

  const handleCanvasMouseUp = () => {
    setIsDragging(false)
  }

  const getNodeIcon = (nodeType) => {
    const nodeConfig = Object.values(NODE_TYPES).find(type => type.id === nodeType)
    return nodeConfig?.icon || Database
  }

  const getNodeColor = (nodeType) => {
    const nodeConfig = Object.values(NODE_TYPES).find(type => type.id === nodeType)
    return nodeConfig?.color || 'bg-gray-500'
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'running':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const nodeCategories = {
    // 1. INPUT - Data Sources
    input: Object.values(NODE_TYPES).filter(type => type.category === 'input'),
    
    // 2. PREPROCESSING - Data Cleaning & Preparation  
    preprocessing: Object.values(NODE_TYPES).filter(type => type.category === 'preprocessing'),
    
    // 3. DATA VISUALIZATION - Charts & Plots
    visualization: Object.values(NODE_TYPES).filter(type => type.category === 'visualization'),
    
    // 4. ANALYSIS - Statistical Analysis & EDA
    analysis: Object.values(NODE_TYPES).filter(type => type.category === 'analysis'),
    
    // 5. ML SUPERVISED - Supervised Learning (includes deep learning supervised models)
    ml_supervised: [
      ...Object.values(NODE_TYPES).filter(type => type.category === 'ml_supervised'),
      ...Object.values(NODE_TYPES).filter(type => type.category === 'deep_learning'),
      ...Object.values(NODE_TYPES).filter(type => type.category === 'evaluation')
    ],
    
    // 6. ML UNSUPERVISED - Unsupervised Learning Models
    ml_unsupervised: Object.values(NODE_TYPES).filter(type => type.category === 'ml_unsupervised'),
    
    // 7. EXPORT DATA - Output & Model Deployment
    output: Object.values(NODE_TYPES).filter(type => type.category === 'output')
  }

  // Connection handling with improved functionality
  const startConnection = (nodeId) => {
    setConnecting({ from: nodeId, to: null })
    toast.info('Click another node to connect')
  }

  const completeConnection = (nodeId) => {
    if (connecting && connecting.from !== nodeId) {
      // Check if connection already exists
      const existingConnection = connections.find(
        conn => conn.source === connecting.from && conn.target === nodeId
      )
      
      if (existingConnection) {
        toast.error('Connection already exists between these nodes')
        setConnecting(null)
        return
      }
      
      const newConnection = {
        id: `conn_${Date.now()}`,
        source: connecting.from,
        target: nodeId
      }
      setConnections(prev => [...prev, newConnection])
      setConnecting(null)
      toast.success('Nodes connected successfully!')
    } else if (connecting && connecting.from === nodeId) {
      toast.error('Cannot connect a node to itself')
    }
  }

  const cancelConnection = () => {
    setConnecting(null)
    toast.info('Connection cancelled')
  }

  // const deleteConnection = (connectionId) => {
  //   setConnections(prev => prev.filter(conn => conn.id !== connectionId))
  //   toast.success('Connection deleted')
  // }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 's':
            e.preventDefault()
            if (currentWorkflow) saveWorkflow()
            break
          case 'r':
            e.preventDefault()
            if (currentWorkflow && nodes.length > 0) runWorkflow()
            break
          case 'n':
            e.preventDefault()
            setShowNewWorkflowDialog(true)
            break
          case 'Delete':
            e.preventDefault()
            if (selectedNode) handleNodeDelete(selectedNode.id)
            break
          default:
            break
        }
      }
      if (e.key === 'Escape') {
        if (connecting) cancelConnection()
        if (selectedNode) setSelectedNode(null)
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [currentWorkflow, nodes, selectedNode, connecting, runWorkflow, saveWorkflow])

  // Auto-save functionality
  useEffect(() => {
    if (currentWorkflow && (nodes.length > 0 || connections.length > 0)) {
      const autoSaveTimer = setTimeout(() => {
        saveWorkflow()
      }, 30000) // Auto-save every 30 seconds

      return () => clearTimeout(autoSaveTimer)
    }
  }, [currentWorkflow, nodes, connections, saveWorkflow])

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Advanced Workflow Builder</h1>
          <p className="text-muted-foreground">
            Create and manage data science workflows with drag-and-drop • {workflows.length} workflows • {nodes.length} nodes
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={() => setShowNewWorkflowDialog(true)}>
            <Plus className="h-4 w-4 mr-2" />
            New Workflow
          </Button>
          <Button 
            variant="outline" 
            onClick={() => {
              // Load ML Pipeline Template
              const mlTemplate = [
                { id: 'ds1', type: 'data_source', name: 'Data Source', x: 50, y: 100, config: {}, status: 'idle' },
                { id: 'dc1', type: 'data_cleaning', name: 'Data Cleaning', x: 300, y: 100, config: {}, status: 'idle' },
                { id: 'fe1', type: 'feature_engineering', name: 'Feature Engineering', x: 550, y: 100, config: {}, status: 'idle' },
                { id: 'cls1', type: 'classification', name: 'Classification', x: 800, y: 100, config: {}, status: 'idle' },
                { id: 'eval1', type: 'model_evaluation', name: 'Model Evaluation', x: 1050, y: 100, config: {}, status: 'idle' }
              ]
              const mlConnections = [
                { id: 'c1', source: 'ds1', target: 'dc1' },
                { id: 'c2', source: 'dc1', target: 'fe1' },
                { id: 'c3', source: 'fe1', target: 'cls1' },
                { id: 'c4', source: 'cls1', target: 'eval1' }
              ]
              setNodes(mlTemplate)
              setConnections(mlConnections)
              toast.success('ML Pipeline template loaded!')
            }}
          >
            <Target className="h-4 w-4 mr-2" />
            ML Template
          </Button>
          <Button 
            variant="outline" 
            onClick={() => {
              // Load EDA Template
              const edaTemplate = [
                { id: 'ds1', type: 'data_source', name: 'Data Source', x: 50, y: 100, config: {}, status: 'idle' },
                { id: 'dv1', type: 'data_validation', name: 'Data Validation', x: 300, y: 50, config: {}, status: 'idle' },
                { id: 'ds2', type: 'descriptive_stats', name: 'Descriptive Stats', x: 300, y: 150, config: {}, status: 'idle' },
                { id: 'ca1', type: 'correlation_analysis', name: 'Correlation Analysis', x: 550, y: 100, config: {}, status: 'idle' },
                { id: 'bp1', type: 'basic_plots', name: 'Basic Plots', x: 800, y: 50, config: {}, status: 'idle' },
                { id: 'ap1', type: 'advanced_plots', name: 'Advanced Plots', x: 800, y: 150, config: {}, status: 'idle' }
              ]
              const edaConnections = [
                { id: 'c1', source: 'ds1', target: 'dv1' },
                { id: 'c2', source: 'ds1', target: 'ds2' },
                { id: 'c3', source: 'ds2', target: 'ca1' },
                { id: 'c4', source: 'ca1', target: 'bp1' },
                { id: 'c5', source: 'ca1', target: 'ap1' }
              ]
              setNodes(edaTemplate)
              setConnections(edaConnections)
              toast.success('EDA template loaded!')
            }}
          >
            <BarChart3 className="h-4 w-4 mr-2" />
            EDA Template
          </Button>
          {currentWorkflow && (
            <>
              <Button variant="outline" onClick={saveWorkflow}>
                <Save className="h-4 w-4 mr-2" />
                Save
              </Button>
              <Button 
                variant="outline"
                onClick={() => {
                  const exportData = {
                    workflow: currentWorkflow,
                    nodes: nodes,
                    connections: connections
                  }
                  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `${currentWorkflow.name}_workflow.json`
                  a.click()
                  URL.revokeObjectURL(url)
                  toast.success('Workflow exported!')
                }}
              >
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
              <Button 
                onClick={runWorkflow} 
                disabled={isRunning || nodes.length === 0}
                className="bg-green-600 hover:bg-green-700"
              >
                {isRunning ? (
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                {isRunning ? 'Running...' : 'Run'}
              </Button>
              {Object.keys(executionResults).length > 0 && (
                <Button 
                  variant="outline"
                  onClick={() => setShowResultsPanel(!showResultsPanel)}
                >
                  <BarChart3 className="h-4 w-4 mr-2" />
                  {showResultsPanel ? 'Hide Results' : 'Show Results'}
                </Button>
              )}
              {connecting && (
                <Button 
                  variant="destructive"
                  size="sm"
                  onClick={cancelConnection}
                >
                  Cancel Connection
                </Button>
              )}
              <Button 
                variant="outline"
                onClick={() => {
                  toast.success(`
🎯 WorkflowBuilder Guide:
• Drag nodes from the library to canvas
• Click nodes to configure parameters
• Hover nodes to see connection points
• Blue dots (right) = output, Gray dots (left) = input
• Keyboard shortcuts: Ctrl+S (save), Ctrl+R (run), Ctrl+N (new)
• Use templates for quick start!
                  `, { duration: 10000 })
                }}
              >
                <Eye className="h-4 w-4 mr-2" />
                Help
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Node Palette */}
        <div className="w-80 border-r bg-muted/30 p-4 overflow-y-auto">
          <div className="space-y-6">
            <div>
              <h3 className="font-semibold mb-3">Current Workflow</h3>
              {currentWorkflow ? (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">{currentWorkflow.name}</CardTitle>
                    <CardDescription className="text-xs">
                      {currentWorkflow.description || 'No description'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{nodes.length} nodes</span>
                      <span>{connections.length} connections</span>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card className="border-dashed">
                  <CardContent className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <GitBranch className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">No workflow selected</p>
                      <Button 
                        variant="link" 
                        size="sm" 
                        onClick={() => setShowNewWorkflowDialog(true)}
                        className="mt-1"
                      >
                        Create new workflow
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            <div>
              <h3 className="font-semibold mb-3">Node Library - UPDATED</h3>
              <Tabs defaultValue="input" className="w-full">
                <TabsList className="grid w-full grid-cols-3 mb-2">
                  <TabsTrigger value="input" className="text-xs">Input</TabsTrigger>
                  <TabsTrigger value="preprocessing" className="text-xs">Preprocessing</TabsTrigger>
                  <TabsTrigger value="visualization" className="text-xs">Visualization</TabsTrigger>
                </TabsList>
                <TabsList className="grid w-full grid-cols-3 mb-4">
                  <TabsTrigger value="analysis" className="text-xs">Analysis</TabsTrigger>
                  <TabsTrigger value="ml_supervised" className="text-xs">ML Supervised</TabsTrigger>
                  <TabsTrigger value="ml_unsupervised" className="text-xs">ML Unsupervised</TabsTrigger>
                </TabsList>
                <TabsList className="grid w-full grid-cols-1">
                  <TabsTrigger value="output" className="text-xs">Export Data</TabsTrigger>
                </TabsList>
                
                {Object.entries(nodeCategories).map(([category, nodeTypes]) => (
                  <TabsContent key={category} value={category} className="space-y-2 mt-4">
                    {nodeTypes.map((nodeType) => {
                      const Icon = nodeType.icon
                      return (
                        <Card 
                          key={nodeType.id}
                          className="cursor-grab hover:shadow-md transition-shadow"
                          draggable
                          onDragStart={() => handleDragStart(nodeType)}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center space-x-3">
                              <div className={`p-2 rounded ${nodeType.color} text-white`}>
                                <Icon className="h-4 w-4" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium truncate">{nodeType.name}</p>
                                <p className="text-xs text-muted-foreground truncate">
                                  {nodeType.description}
                                </p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </TabsContent>
                ))}
              </Tabs>
            </div>

            {workflows.length > 0 && (
              <div>
                <h3 className="font-semibold mb-3">Recent Workflows</h3>
                <div className="space-y-2">
                  {workflows.slice(0, 5).map((workflow) => (
                    <Card 
                      key={workflow.id}
                      className="cursor-pointer hover:shadow-md transition-shadow"
                      onClick={() => setCurrentWorkflow(workflow)}
                    >
                      <CardContent className="p-3">
                        <div className="flex items-center justify-between">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">{workflow.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(workflow.updated_at).toLocaleDateString()}
                            </p>
                          </div>
                          <Badge variant="outline" className="ml-2">
                            {workflow.status || 'draft'}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Canvas */}
        <div className="flex-1 relative overflow-hidden">
          {/* Quick Actions Toolbar */}
          {currentWorkflow && (
            <div className="absolute top-4 left-4 z-20 flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setCanvasOffset({ x: 0, y: 0 })
                  toast.success('Canvas reset to center')
                }}
                className="bg-white/90 backdrop-blur"
              >
                <RefreshCw className="h-3 w-3" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setNodes([])
                  setConnections([])
                  toast.success('Canvas cleared')
                }}
                className="bg-white/90 backdrop-blur"
              >
                <Trash2 className="h-3 w-3" />
              </Button>
              <Badge variant="outline" className="bg-white/90 backdrop-blur">
                {nodes.length} nodes • {connections.length} connections
              </Badge>
            </div>
          )}

          {/* Connection indicator */}
          {connecting && (
            <div className="absolute top-4 right-4 z-20">
              <Badge variant="secondary" className="bg-blue-500 text-white animate-pulse">
                Click another node to connect
              </Badge>
            </div>
          )}

          <div
            ref={canvasRef}
            className="w-full h-full bg-grid-pattern relative cursor-move"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            style={{
              backgroundPosition: `${canvasOffset.x}px ${canvasOffset.y}px`,
              backgroundImage: `
                radial-gradient(circle, #e5e7eb 1px, transparent 1px)
              `,
              backgroundSize: '20px 20px'
            }}
          >
            {/* Render connections */}
            <svg className="absolute inset-0 pointer-events-none" style={{ zIndex: 0 }}>
              {connections.map((connection) => {
                const fromNode = nodes.find(n => n.id === connection.source)
                const toNode = nodes.find(n => n.id === connection.target)
                if (!fromNode || !toNode) return null
                
                const startX = fromNode.x + canvasOffset.x + 150
                const startY = fromNode.y + canvasOffset.y + 40
                const endX = toNode.x + canvasOffset.x
                const endY = toNode.y + canvasOffset.y + 40
                
                return (
                  <line
                    key={connection.id}
                    x1={startX}
                    y1={startY}
                    x2={endX}
                    y2={endY}
                    stroke="#6366f1"
                    strokeWidth="2"
                    markerEnd="url(#arrowhead)"
                  />
                )
              })}
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1" />
                </marker>
              </defs>
            </svg>

            {/* Render nodes */}
            {nodes.map((node) => {
              const Icon = getNodeIcon(node.type)
              const colorClass = getNodeColor(node.type)
              
              return (
                <div
                  key={node.id}
                  className="absolute bg-white border-2 border-gray-200 rounded-lg shadow-lg cursor-pointer hover:shadow-xl transition-shadow group"
                  style={{
                    left: node.x + canvasOffset.x,
                    top: node.y + canvasOffset.y,
                    width: 180,
                    zIndex: selectedNode?.id === node.id ? 10 : 1
                  }}
                  onClick={() => handleNodeClick(node)}
                >
                  {/* Input connection point */}
                  <div 
                    className="absolute -left-2 top-1/2 transform -translate-y-1/2 w-4 h-4 bg-gray-300 rounded-full border-2 border-white opacity-0 group-hover:opacity-100 transition-opacity cursor-crosshair"
                    onClick={(e) => {
                      e.stopPropagation()
                      if (connecting) {
                        completeConnection(node.id)
                      }
                    }}
                  />
                  
                  {/* Output connection point */}
                  <div 
                    className="absolute -right-2 top-1/2 transform -translate-y-1/2 w-4 h-4 bg-blue-500 rounded-full border-2 border-white opacity-0 group-hover:opacity-100 transition-opacity cursor-crosshair"
                    onClick={(e) => {
                      e.stopPropagation()
                      startConnection(node.id)
                    }}
                  />
                  
                  <div className={`p-3 ${colorClass} text-white rounded-t-md`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Icon className="h-4 w-4" />
                        <span className="text-sm font-medium truncate">{node.name}</span>
                      </div>
                      {getStatusIcon(node.status)}
                    </div>
                  </div>
                  <div className="p-3">
                    <p className="text-xs text-muted-foreground truncate mb-2">
                      {Object.keys(node.config || {}).length > 0 ? 'Configured' : 'Not configured'}
                    </p>
                    {node.config && Object.keys(node.config).length > 0 && (
                      <div className="text-xs space-y-1">
                        {Object.entries(node.config).slice(0, 2).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-muted-foreground">{key}:</span>
                            <span className="truncate ml-1">{String(value).slice(0, 10)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    <div className="flex items-center justify-between mt-3">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleNodeClick(node)
                        }}
                        className="p-1"
                      >
                        <Settings className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          const newNode = { ...node, id: `node_${Date.now()}`, x: node.x + 20, y: node.y + 20 }
                          setNodes(prev => [...prev, newNode])
                        }}
                        className="p-1"
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleNodeDelete(node.id)
                        }}
                        className="text-red-500 hover:text-red-700 p-1"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </div>
              )
            })}

            {/* Empty state */}
            {nodes.length === 0 && currentWorkflow && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <GitBranch className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">Start Building Your Workflow</h3>
                  <p className="text-muted-foreground mb-4">
                    Drag nodes from the library to create your data processing pipeline
                  </p>
                  <Badge variant="outline">
                    Tip: Start with a Data Source node
                  </Badge>
                </div>
              </div>
            )}

            {!currentWorkflow && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Plus className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">Create Your First Workflow</h3>
                  <p className="text-muted-foreground mb-4">
                    Get started by creating a new workflow
                  </p>
                  <Button onClick={() => setShowNewWorkflowDialog(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    New Workflow
                  </Button>
                </div>
              </div>
            )}

            {/* Mini-map for large workflows */}
            {nodes.length > 5 && (
              <div className="absolute bottom-4 right-4 z-20 w-32 h-24 bg-white/90 backdrop-blur border rounded-lg p-2">
                <div className="text-xs font-medium mb-1">Mini Map</div>
                <div className="relative w-full h-full bg-gray-100 rounded overflow-hidden">
                  {nodes.map((node) => (
                    <div
                      key={node.id}
                      className={`absolute w-1 h-1 ${getNodeColor(node.type)} rounded-full`}
                      style={{
                        left: `${Math.max(0, Math.min(100, (node.x / 20)))}%`,
                        top: `${Math.max(0, Math.min(100, (node.y / 20)))}%`
                      }}
                    />
                  ))}
                  <div
                    className="absolute border border-blue-500 bg-blue-500/20"
                    style={{
                      left: `${Math.max(0, Math.min(90, (-canvasOffset.x / 20)))}%`,
                      top: `${Math.max(0, Math.min(90, (-canvasOffset.y / 20)))}%`,
                      width: '10%',
                      height: '10%'
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Panel */}
      {/* Enhanced Results Panel with Tabs */}
      {showResultsPanel && (
        <div className="border-t bg-muted/30 max-h-[70vh] overflow-y-auto">
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Workflow Execution Results</h3>
              <div className="flex items-center space-x-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => {
                    const resultsData = {
                      summary: executionResults.summary,
                      results: executionResults.results,
                      logs: executionLogs,
                      timestamp: new Date().toISOString()
                    }
                    const blob = new Blob([JSON.stringify(resultsData, null, 2)], { type: 'application/json' })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = `workflow_results_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '')}.json`
                    a.click()
                    URL.revokeObjectURL(url)
                    toast.success('Results exported successfully!')
                  }}
                >
                  <Download className="h-3 w-3 mr-1" />
                  Export
                </Button>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setShowResultsPanel(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="data">Data Results</TabsTrigger>
                <TabsTrigger value="charts">Charts & Plots</TabsTrigger>
                <TabsTrigger value="logs">Execution Logs</TabsTrigger>
              </TabsList>
              
              <TabsContent value="overview" className="space-y-4">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Execution Summary */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Execution Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {executionResults.summary && (
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Total Nodes:</span>
                            <span className="font-medium">{executionResults.summary.total_nodes}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Completed:</span>
                            <span className="font-medium text-green-600">{executionResults.summary.nodes_completed}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Failed:</span>
                            <span className="font-medium text-red-600">{executionResults.summary.nodes_failed || 0}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Success Rate:</span>
                            <span className="font-medium">{executionResults.summary.success_rate?.toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Execution Time:</span>
                            <span className="font-medium">{executionResults.summary.execution_time?.toFixed(2)}s</span>
                          </div>
                          {executionResults.summary.charts_generated > 0 && (
                            <div className="flex justify-between">
                              <span>Charts Generated:</span>
                              <span className="font-medium">{executionResults.summary.charts_generated}</span>
                            </div>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Workflow Execution Flow */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Workflow Execution Flow</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {nodes.map((node) => {
                          const result = executionResults.results?.[node.id]
                          return (
                            <div key={node.id} className="flex items-center space-x-3">
                              <div className="flex items-center space-x-2 flex-1">
                                <div className={`w-2 h-2 rounded-full ${
                                  result?.status === 'completed' ? 'bg-green-500' : 
                                  result?.status === 'failed' ? 'bg-red-500' : 'bg-gray-300'
                                }`} />
                                <span className="text-sm font-medium truncate">{node.name}</span>
                              </div>
                              {result?.execution_time && (
                                <span className="text-xs text-muted-foreground">
                                  {result.execution_time.toFixed(2)}s
                                </span>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="data" className="space-y-4">
                <div className="space-y-4">
                  <h4 className="text-md font-semibold">Node Output Results</h4>
                  {executionResults.results && Object.entries(executionResults.results).map(([nodeId, result]) => {
                    const node = nodes.find(n => n.id === nodeId)
                    if (!node || result.status !== 'completed') return null
                    
                    return (
                      <Card key={nodeId}>
                        <CardHeader className="pb-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <div className={`p-1 rounded ${getNodeColor(node.type)} text-white`}>
                                {(() => {
                                  const Icon = getNodeIcon(node.type)
                                  return <Icon className="h-3 w-3" />
                                })()}
                              </div>
                              <CardTitle className="text-sm">{node.name}</CardTitle>
                              <Badge variant="outline" className="text-xs">
                                {result.status}
                              </Badge>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {/* Output Summary */}
                            <div>
                              <h5 className="text-xs font-medium mb-2">Output Summary</h5>
                              <div className="text-xs space-y-1 bg-muted p-2 rounded">
                                <div>Type: <span className="font-medium">{result.result_summary?.type}</span></div>
                                {result.result_summary?.shape && (
                                  <div>Shape: <span className="font-medium">{result.result_summary.shape.join(' rows × ')} columns</span></div>
                                )}
                                {result.result_summary?.columns && (
                                  <div>Columns: <span className="font-medium">{result.result_summary.columns.slice(0, 5).join(', ')}{result.result_summary.columns.length > 5 ? '...' : ''}</span></div>
                                )}
                                {result.result_summary?.memory_usage && (
                                  <div>Memory: <span className="font-medium">{result.result_summary.memory_usage}</span></div>
                                )}
                                <div>Executed at: <span className="font-medium">{new Date(result.timestamp).toLocaleString()}</span></div>
                              </div>
                            </div>

                            {/* Data Preview for DataFrames */}
                            {result.result_summary?.type === 'DataFrame' && result.result_summary?.shape && (
                              <div>
                                <h5 className="text-xs font-medium mb-2">Dataset Information</h5>
                                <div className="text-xs bg-muted p-3 rounded">
                                  <div className="mb-2 text-muted-foreground">
                                    Dataset shape: {result.result_summary.shape.join(' × ')}
                                  </div>
                                  {result.result_summary.columns && (
                                    <div>
                                      <div className="font-medium mb-1">Columns ({result.result_summary.columns.length}):</div>
                                      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-1">
                                        {result.result_summary.columns.map((col, idx) => (
                                          <div key={idx} className="bg-background p-1 rounded text-xs truncate border">
                                            {col}
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Statistics Preview for Descriptive Stats */}
                            {node.type === 'descriptive_stats' && (
                              <div>
                                <h5 className="text-xs font-medium mb-2">Statistical Summary</h5>
                                <div className="text-xs bg-muted p-3 rounded space-y-3">
                                  {/* Display generated statistics categories */}
                                  {result.result_summary?.keys && (
                                    <div>
                                      <div className="font-medium mb-2">Generated Statistics:</div>
                                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                        {result.result_summary.keys.map((key, idx) => (
                                          <div key={idx} className="bg-background p-2 rounded border">
                                            <span className="font-medium">{key.replace(/_/g, ' ').toUpperCase()}</span>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Display basic statistical info if available */}
                                  {result.result_summary?.statistics_summary && (
                                    <div>
                                      <div className="font-medium mb-2">Summary Information:</div>
                                      <div className="grid grid-cols-2 gap-2 text-xs">
                                        {Object.entries(result.result_summary.statistics_summary).map(([key, value]) => (
                                          <div key={key} className="bg-background p-2 rounded border">
                                            <div className="font-medium">{key.replace(/_/g, ' ')}</div>
                                            <div className="text-muted-foreground">
                                              {typeof value === 'number' ? value.toLocaleString() : String(value)}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Display count of numeric vs categorical columns */}
                                  {result.result_summary?.column_info && (
                                    <div>
                                      <div className="font-medium mb-2">Column Analysis:</div>
                                      <div className="grid grid-cols-3 gap-2">
                                        <div className="bg-background p-2 rounded border text-center">
                                          <div className="font-medium">{result.result_summary.column_info.numeric || 0}</div>
                                          <div className="text-muted-foreground">Numeric</div>
                                        </div>
                                        <div className="bg-background p-2 rounded border text-center">
                                          <div className="font-medium">{result.result_summary.column_info.categorical || 0}</div>
                                          <div className="text-muted-foreground">Categorical</div>
                                        </div>
                                        <div className="bg-background p-2 rounded border text-center">
                                          <div className="font-medium">{result.result_summary.column_info.datetime || 0}</div>
                                          <div className="text-muted-foreground">DateTime</div>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Charts generated indicator */}
                                  {result.charts && (
                                    <div className="mt-2 pt-2 border-t">
                                      <div className="flex items-center justify-between">
                                        <span className="text-muted-foreground">Charts Generated:</span>
                                        <span className="font-medium">
                                          {Array.isArray(result.charts) ? result.charts.length : Object.keys(result.charts).length} charts
                                        </span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    )
                  })}
                </div>
              </TabsContent>

              <TabsContent value="charts" className="space-y-4">
                <div className="space-y-4">
                  <h4 className="text-md font-semibold">Generated Charts & Visualizations</h4>
                  
                  {(() => {
                    // Collect all charts from all nodes
                    const allCharts = []
                    if (executionResults.results) {
                      Object.entries(executionResults.results).forEach(([nodeId, result]) => {
                        const node = nodes.find(n => n.id === nodeId)
                        
                        if (result.charts) {
                          if (Array.isArray(result.charts)) {
                            // Format 1: Array of chart objects
                            result.charts.forEach((chart, index) => {
                              allCharts.push({
                                ...chart,
                                nodeId,
                                nodeName: node?.name,
                                chartIndex: index
                              })
                            })
                          } else if (typeof result.charts === 'object') {
                            // Format 2: Object with chart names as keys
                            Object.entries(result.charts).forEach(([chartName, chartData], index) => {
                              allCharts.push({
                                type: chartName,
                                column: chartName.split('_').slice(1).join('_'),
                                image: chartData,
                                nodeId,
                                nodeName: node?.name,
                                chartIndex: index
                              })
                            })
                          }
                        }
                      })
                    }

                    if (allCharts.length === 0) {
                      return (
                        <Card>
                          <CardContent className="flex items-center justify-center py-12">
                            <div className="text-center">
                              <BarChart3 className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                              <h3 className="text-lg font-semibold mb-2">No Visualizations Generated</h3>
                              <p className="text-sm text-muted-foreground mb-4">
                                Your workflow executed successfully, but no charts were generated.
                              </p>
                              <div className="text-xs text-muted-foreground">
                                <p>• Add visualization nodes (Basic Plots, Advanced Plots) to generate charts</p>
                                <p>• Enable plot generation in Descriptive Statistics nodes</p>
                                <p>• Configure chart parameters in node settings</p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    }

                    return (
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {allCharts.map((chart) => (
                          <Card key={`${chart.nodeId}-${chart.chartIndex}`} className="overflow-hidden">
                            <CardHeader className="pb-3">
                              <div className="flex items-center justify-between">
                                <CardTitle className="text-sm">
                                  {chart.type?.replace(/_/g, ' ').toUpperCase()} - {chart.column || 'Multiple Columns'}
                                </CardTitle>
                                <Badge variant="outline" className="text-xs">
                                  {chart.nodeName}
                                </Badge>
                              </div>
                            </CardHeader>
                            <CardContent className="p-0">
                              <div className="relative bg-white">
                                <img 
                                  src={`data:image/png;base64,${chart.image}`}
                                  alt={`${chart.type} for ${chart.column}`}
                                  className="w-full h-auto"
                                  style={{ maxHeight: '400px', objectFit: 'contain' }}
                                />
                                <div className="absolute top-2 right-2 opacity-0 hover:opacity-100 transition-opacity">
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => {
                                      // Download chart
                                      const link = document.createElement('a')
                                      link.href = `data:image/png;base64,${chart.image}`
                                      link.download = `${chart.type}_${chart.column || 'chart'}_${new Date().getTime()}.png`
                                      link.click()
                                      toast.success('Chart downloaded successfully!')
                                    }}
                                    className="bg-white/90 backdrop-blur shadow-lg"
                                  >
                                    <Download className="h-3 w-3" />
                                  </Button>
                                </div>
                              </div>
                              <div className="p-3 bg-muted/50">
                                <div className="text-xs text-muted-foreground space-y-1">
                                  <div>Chart Type: <span className="font-medium">{chart.type || 'Unknown'}</span></div>
                                  <div>Column: <span className="font-medium">{chart.column || 'Multiple'}</span></div>
                                  <div>Generated by: <span className="font-medium">{chart.nodeName}</span></div>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    )
                  })()}
                </div>
              </TabsContent>

              <TabsContent value="logs" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Execution Logs</CardTitle>
                    <CardDescription>
                      Detailed execution logs for debugging and monitoring
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-black text-green-400 p-4 rounded-lg font-mono text-xs max-h-96 overflow-y-auto">
                      {executionLogs.length > 0 ? (
                        executionLogs.map((log, index) => (
                          <div key={index} className="mb-1 leading-relaxed">
                            <span className="text-gray-500 mr-2">[{String(index + 1).padStart(3, '0')}]</span>
                            <span className={
                              log.includes('ERROR') ? 'text-red-400' :
                              log.includes('WARNING') ? 'text-yellow-400' :
                              log.includes('completed successfully') ? 'text-green-400' :
                              'text-gray-300'
                            }>
                              {log}
                            </span>
                          </div>
                        ))
                      ) : (
                        <div className="text-gray-500 text-center py-4">
                          No execution logs available
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      )}

      <Dialog open={showNewWorkflowDialog} onOpenChange={setShowNewWorkflowDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Workflow</DialogTitle>
            <DialogDescription>
              Create a new data processing workflow
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="workflow-name">Workflow Name</Label>
              <Input
                id="workflow-name"
                value={newWorkflowForm.name}
                onChange={(e) => setNewWorkflowForm(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Enter workflow name"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="workflow-description">Description (Optional)</Label>
              <Textarea
                id="workflow-description"
                value={newWorkflowForm.description}
                onChange={(e) => setNewWorkflowForm(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Describe your workflow"
                rows={3}
              />
            </div>
          </div>
          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={() => setShowNewWorkflowDialog(false)}>
              Cancel
            </Button>
            <Button onClick={createNewWorkflow}>
              Create Workflow
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Node Configuration Dialog */}
      <Dialog open={showNodeConfigDialog} onOpenChange={setShowNodeConfigDialog}>
        <DialogContent className="sm:max-w-[700px] max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Configure Node: {selectedNode?.name}</DialogTitle>
            <DialogDescription>
              Configure the settings for this workflow node
            </DialogDescription>
          </DialogHeader>
          {selectedNode && (
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label>Node Type</Label>
                <div className="flex items-center space-x-2">
                  <div className={`p-2 rounded ${getNodeColor(selectedNode.type)} text-white`}>
                    {(() => {
                      const Icon = getNodeIcon(selectedNode.type)
                      return <Icon className="h-4 w-4" />
                    })()}
                  </div>
                  <div>
                    <span className="font-medium">{selectedNode.name}</span>
                    <p className="text-xs text-muted-foreground">
                      {Object.values(NODE_TYPES).find(type => type.id === selectedNode.type)?.description}
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="grid gap-2">
                <Label htmlFor="node-name">Node Name</Label>
                <Input
                  id="node-name"
                  value={selectedNode.name}
                  onChange={(e) => {
                    setSelectedNode(prev => ({ ...prev, name: e.target.value }))
                    setNodes(prev => prev.map(n => 
                      n.id === selectedNode.id ? { ...n, name: e.target.value } : n
                    ))
                  }}
                />
              </div>

              <div className="grid gap-2">
                <Label>Node Parameters</Label>
                <div className="space-y-3 p-4 border rounded-md bg-muted/50 max-h-60 overflow-y-auto">
                  {(() => {
                    const nodeType = Object.values(NODE_TYPES).find(type => type.id === selectedNode.type)
                    const backendNodeType = nodeTypes[selectedNode.type]
                    
                    // Use backend node type parameters if available, otherwise fall back to frontend
                    const parameters = backendNodeType?.parameters || nodeType?.params || []
                    
                    if (parameters.length === 0) {
                      return <p className="text-sm text-muted-foreground">No parameters available for this node type.</p>
                    }
                    
                    return parameters.map((param) => {
                      const paramName = param.name || param
                      const paramType = param.type || 'text'
                      const paramRequired = param.required || false
                      const paramOptions = param.options || []
                      
                      return (
                        <div key={paramName} className="grid gap-2">
                          <Label htmlFor={`param-${paramName}`} className="text-sm capitalize">
                            {(param.description || paramName).replace(/_/g, ' ')}
                            {paramRequired && <span className="text-red-500 ml-1">*</span>}
                          </Label>
                          
                          {/* Special handling for dataset selection */}
                          {paramName === 'dataset_id' && selectedNode.type === 'data_source' ? (
                            <Select
                              value={selectedNode.config?.[paramName] || ''}
                              onValueChange={(value) => {
                                const newConfig = { ...selectedNode.config, [paramName]: value }
                                setSelectedNode(prev => ({ ...prev, config: newConfig }))
                                setNodes(prev => prev.map(n => 
                                  n.id === selectedNode.id ? { ...n, config: newConfig } : n
                                ))
                              }}
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Select a dataset" />
                              </SelectTrigger>
                              <SelectContent>
                                {availableDatasets.map((dataset) => (
                                  <SelectItem key={dataset.id} value={dataset.id.toString()}>
                                    <div className="flex flex-col">
                                      <span className="font-medium">{dataset.name}</span>
                                      <span className="text-xs text-muted-foreground">
                                        {dataset.rows_count} rows × {dataset.columns_count} cols ({dataset.file_type})
                                      </span>
                                    </div>
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          ) : paramType === 'select' && paramOptions.length > 0 ? (
                            <Select
                              value={selectedNode.config?.[paramName] || ''}
                              onValueChange={(value) => {
                                const newConfig = { ...selectedNode.config, [paramName]: value }
                                setSelectedNode(prev => ({ ...prev, config: newConfig }))
                                setNodes(prev => prev.map(n => 
                                  n.id === selectedNode.id ? { ...n, config: newConfig } : n
                                ))
                              }}
                            >
                              <SelectTrigger>
                                <SelectValue placeholder={`Select ${paramName.replace(/_/g, ' ')}`} />
                              </SelectTrigger>
                              <SelectContent>
                                {paramOptions.map((option) => (
                                  <SelectItem key={option} value={option}>
                                    {option.replace(/_/g, ' ').toUpperCase()}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          ) : paramType === 'boolean' ? (
                            <Select
                              value={selectedNode.config?.[paramName] || 'false'}
                              onValueChange={(value) => {
                                const newConfig = { ...selectedNode.config, [paramName]: value === 'true' }
                                setSelectedNode(prev => ({ ...prev, config: newConfig }))
                                setNodes(prev => prev.map(n => 
                                  n.id === selectedNode.id ? { ...n, config: newConfig } : n
                                ))
                              }}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="true">Yes</SelectItem>
                                <SelectItem value="false">No</SelectItem>
                              </SelectContent>
                            </Select>
                          ) : paramType === 'textarea' ? (
                            <Textarea
                              id={`param-${paramName}`}
                              placeholder={param.description || `Enter ${paramName.replace(/_/g, ' ')}`}
                              value={selectedNode.config?.[paramName] || ''}
                              onChange={(e) => {
                                const newConfig = { ...selectedNode.config, [paramName]: e.target.value }
                                setSelectedNode(prev => ({ ...prev, config: newConfig }))
                                setNodes(prev => prev.map(n => 
                                  n.id === selectedNode.id ? { ...n, config: newConfig } : n
                                ))
                              }}
                              rows={3}
                              className="text-sm"
                            />
                          ) : (
                            <Input
                              id={`param-${paramName}`}
                              type={paramType === 'number' ? 'number' : 'text'}
                              step={paramType === 'number' ? '0.01' : undefined}
                              placeholder={param.description || `Enter ${paramName.replace(/_/g, ' ')}`}
                              value={selectedNode.config?.[paramName] || ''}
                              onChange={(e) => {
                                const value = paramType === 'number' ? parseFloat(e.target.value) || e.target.value : e.target.value
                                const newConfig = { ...selectedNode.config, [paramName]: value }
                                setSelectedNode(prev => ({ ...prev, config: newConfig }))
                                setNodes(prev => prev.map(n => 
                                  n.id === selectedNode.id ? { ...n, config: newConfig } : n
                                ))
                              }}
                              className="text-sm"
                            />
                          )}
                          
                          {param.description && (
                            <p className="text-xs text-muted-foreground">{param.description}</p>
                          )}
                        </div>
                      )
                    })
                  })()}
                </div>
              </div>

              {/* Advanced Configuration */}
              <div className="grid gap-2">
                <Label>Advanced Settings</Label>
                <Tabs defaultValue="general" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="general">General</TabsTrigger>
                    <TabsTrigger value="performance">Performance</TabsTrigger>
                    <TabsTrigger value="validation">Validation</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="general" className="space-y-3 mt-4">
                    <div className="grid gap-2">
                      <Label htmlFor="node-description">Description</Label>
                      <Textarea
                        id="node-description"
                        placeholder="Describe what this node does..."
                        value={selectedNode.config?.description || ''}
                        onChange={(e) => {
                          const newConfig = { ...selectedNode.config, description: e.target.value }
                          setSelectedNode(prev => ({ ...prev, config: newConfig }))
                          setNodes(prev => prev.map(n => 
                            n.id === selectedNode.id ? { ...n, config: newConfig } : n
                          ))
                        }}
                        rows={2}
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="cache-results">Cache Results</Label>
                      <Select
                        value={selectedNode.config?.cache_results || 'false'}
                        onValueChange={(value) => {
                          const newConfig = { ...selectedNode.config, cache_results: value }
                          setSelectedNode(prev => ({ ...prev, config: newConfig }))
                          setNodes(prev => prev.map(n => 
                            n.id === selectedNode.id ? { ...n, config: newConfig } : n
                          ))
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="true">Yes</SelectItem>
                          <SelectItem value="false">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="performance" className="space-y-3 mt-4">
                    <div className="grid gap-2">
                      <Label htmlFor="parallel-processing">Parallel Processing</Label>
                      <Select
                        value={selectedNode.config?.parallel_processing || 'auto'}
                        onValueChange={(value) => {
                          const newConfig = { ...selectedNode.config, parallel_processing: value }
                          setSelectedNode(prev => ({ ...prev, config: newConfig }))
                          setNodes(prev => prev.map(n => 
                            n.id === selectedNode.id ? { ...n, config: newConfig } : n
                          ))
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Auto</SelectItem>
                          <SelectItem value="single">Single Thread</SelectItem>
                          <SelectItem value="multi">Multi Thread</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="memory-limit">Memory Limit (MB)</Label>
                      <Input
                        id="memory-limit"
                        type="number"
                        placeholder="1024"
                        value={selectedNode.config?.memory_limit || ''}
                        onChange={(e) => {
                          const newConfig = { ...selectedNode.config, memory_limit: e.target.value }
                          setSelectedNode(prev => ({ ...prev, config: newConfig }))
                          setNodes(prev => prev.map(n => 
                            n.id === selectedNode.id ? { ...n, config: newConfig } : n
                          ))
                        }}
                      />
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="validation" className="space-y-3 mt-4">
                    <div className="grid gap-2">
                      <Label htmlFor="data-validation">Data Validation</Label>
                      <Select
                        value={selectedNode.config?.data_validation || 'basic'}
                        onValueChange={(value) => {
                          const newConfig = { ...selectedNode.config, data_validation: value }
                          setSelectedNode(prev => ({ ...prev, config: newConfig }))
                          setNodes(prev => prev.map(n => 
                            n.id === selectedNode.id ? { ...n, config: newConfig } : n
                          ))
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          <SelectItem value="basic">Basic</SelectItem>
                          <SelectItem value="strict">Strict</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="error-handling">Error Handling</Label>
                      <Select
                        value={selectedNode.config?.error_handling || 'stop'}
                        onValueChange={(value) => {
                          const newConfig = { ...selectedNode.config, error_handling: value }
                          setSelectedNode(prev => ({ ...prev, config: newConfig }))
                          setNodes(prev => prev.map(n => 
                            n.id === selectedNode.id ? { ...n, config: newConfig } : n
                          ))
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="stop">Stop on Error</SelectItem>
                          <SelectItem value="skip">Skip and Continue</SelectItem>
                          <SelectItem value="retry">Retry</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </TabsContent>
                </Tabs>
              </div>
            </div>
          )}
          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={() => setShowNodeConfigDialog(false)}>
              Cancel
            </Button>
            <Button 
              onClick={() => {
                setShowNodeConfigDialog(false)
                toast.success('Node configuration saved!')
              }}
            >
              Save Configuration
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

