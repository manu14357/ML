import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { toast } from 'sonner'
import { 
  BarChart3, Plus, Eye, PieChart, LineChart, ScatterChart, Activity, 
  Database, Download, Share, Settings, RefreshCw, Search, Filter,
  TrendingUp, TrendingDown, BarChart, Brain, Layers, Zap, 
  Globe, Monitor, Palette, Code, Play, Save, Copy, X,
  Box, Target, Grid, Sparkles, Map, Layout,
  Maximize, Minimize, RotateCcw, Paintbrush, Bot
} from 'lucide-react'
import Plot from 'react-plotly.js'
import { fetchRealChartData, generateRealPlotlyConfig } from '@/utils/realDataUtils'
import { DatasetOverviewModal } from './DatasetOverviewModal'

const API_BASE = 'http://localhost:5000/api'

// Chart type configurations
const CHART_TYPES = {
  line: {
    name: 'Line Chart',
    icon: LineChart,
    description: 'Perfect for time series and trends',
    color: 'bg-blue-500',
    gradient: 'from-blue-400 to-blue-600'
  },
  bar: {
    name: 'Bar Chart',
    icon: BarChart3,
    description: 'Compare categories and values',
    color: 'bg-green-500',
    gradient: 'from-green-400 to-green-600'
  },
  scatter: {
    name: 'Scatter Plot',
    icon: ScatterChart,
    description: 'Explore relationships between variables',
    color: 'bg-purple-500',
    gradient: 'from-purple-400 to-purple-600'
  },
  pie: {
    name: 'Pie Chart',
    icon: PieChart,
    description: 'Show parts of a whole',
    color: 'bg-orange-500',
    gradient: 'from-orange-400 to-orange-600'
  },
  heatmap: {
    name: 'Heatmap',
    icon: Grid,
    description: 'Visualize correlation matrices',
    color: 'bg-red-500',
    gradient: 'from-red-400 to-red-600'
  },
  box: {
    name: 'Box Plot',
    icon: Box,
    description: 'Statistical distribution analysis',
    color: 'bg-indigo-500',
    gradient: 'from-indigo-400 to-indigo-600'
  },
  histogram: {
    name: 'Histogram',
    icon: BarChart,
    description: 'Data distribution patterns',
    color: 'bg-teal-500',
    gradient: 'from-teal-400 to-teal-600'
  },
  violin: {
    name: 'Violin Plot',
    icon: Activity,
    description: 'Density and distribution combined',
    color: 'bg-pink-500',
    gradient: 'from-pink-400 to-pink-600'
  },
  surface: {
    name: '3D Surface',
    icon: Layers,
    description: '3D data visualization',
    color: 'bg-cyan-500',
    gradient: 'from-cyan-400 to-cyan-600'
  },
  radar: {
    name: 'Radar Chart',
    icon: Target,
    description: 'Multi-dimensional comparison',
    color: 'bg-yellow-500',
    gradient: 'from-yellow-400 to-yellow-600'
  }
}

// Color schemes
const COLOR_SCHEMES = {
  viridis: ['#440154', '#31688e', '#35b779', '#fde725'],
  plasma: ['#0d0887', '#7e03a8', '#cc4778', '#f89441', '#f0f921'],
  rainbow: ['#ff0000', '#ff8000', '#ffff00', '#80ff00', '#00ff00', '#00ff80', '#00ffff', '#0080ff', '#0000ff', '#8000ff'],
  ocean: ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
  sunset: ['#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF', '#EAECCC', '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D', '#A50026'],
  earth: ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']
}

export function Visualizations() {
  // State management
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [columns, setColumns] = useState([])
  const [charts, setCharts] = useState([])
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterType, setFilterType] = useState('all')
  const [currentTab, setCurrentTab] = useState('gallery')
  const [aiInsight, setAiInsight] = useState({ chartId: null, insight: '', loading: false })
  
  // New advanced features state
  const [autoGenerating, setAutoGenerating] = useState(false)
  const [datasetAnalysis, setDatasetAnalysis] = useState(null)
  const [recommendedCharts, setRecommendedCharts] = useState([])
  const [showDatasetOverview, setShowDatasetOverview] = useState(false)
  
  // Chart creation state
  const [chartConfig, setChartConfig] = useState({
    type: 'line',
    title: '',
    xColumn: '',
    yColumn: '',
    colorColumn: 'none',
    sizeColumn: 'none',
    colorScheme: 'viridis',
    height: 500,
    width: 800
  })

  // Real data state for live preview
  const [realChartData, setRealChartData] = useState(null)
  const [loadingRealData, setLoadingRealData] = useState(false)

  // Dashboard state
  const [dashboards, setDashboards] = useState([])
  const [selectedDashboard, setSelectedDashboard] = useState(null)
  const [dashboardConfig, setDashboardConfig] = useState({
    name: '',
    description: '',
    layout: 'grid',
    refreshInterval: 0,
    theme: 'light'
  })
  const [showCreateDashboard, setShowCreateDashboard] = useState(false)
  const [selectedChartsForDashboard, setSelectedChartsForDashboard] = useState([])
  const [dashboardCharts, setDashboardCharts] = useState([])
  const [isDashboardEditMode, setIsDashboardEditMode] = useState(false)


  // Load datasets on component mount
  useEffect(() => {
    fetchDatasets()
    loadSavedCharts()
    loadSavedDashboards()
  }, [])

  // Fetch real data when chart configuration changes
  useEffect(() => {
    const fetchRealDataForPreview = async () => {
      if (selectedDataset && chartConfig.xColumn && (chartConfig.yColumn || ['pie', 'histogram', 'box'].includes(chartConfig.type))) {
        setLoadingRealData(true)
        try {
          const realData = await fetchRealChartData(selectedDataset.id, chartConfig)
          if (realData && typeof realData === 'object') {
            setRealChartData(realData)
          } else {
            console.warn('Invalid real data received:', realData)
            setRealChartData(null)
          }
        } catch (error) {
          console.error('Error fetching real data for preview:', error)
          setRealChartData(null)
        } finally {
          setLoadingRealData(false)
        }
      } else {
        setRealChartData(null)
      }
    }

    fetchRealDataForPreview()
  }, [selectedDataset, chartConfig])

  const fetchDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE}/data/datasets`)
      if (response.ok) {
        const data = await response.json()
        setDatasets(data.datasets || [])
      }
    } catch (error) {
      console.error('Error fetching datasets:', error)
      toast.error('Failed to load datasets')
    }
  }

  const loadSavedCharts = () => {
    // Load charts from localStorage for now
    const saved = localStorage.getItem('superhacker_charts')
    if (saved) {
      try {
        setCharts(JSON.parse(saved))
      } catch (error) {
        console.error('Error loading saved charts:', error)
      }
    }
  }

  const saveChart = (chart) => {
    const newChart = {
      id: Date.now().toString(),
      ...chart,
      createdAt: new Date().toISOString()
    }
    const updatedCharts = [...charts, newChart]
    setCharts(updatedCharts)
    localStorage.setItem('superhacker_charts', JSON.stringify(updatedCharts))
    toast.success('Chart saved successfully!')
  }

  const deleteChart = (chartId) => {
    const updatedCharts = charts.filter(chart => chart.id !== chartId)
    setCharts(updatedCharts)
    localStorage.setItem('superhacker_charts', JSON.stringify(updatedCharts))
    toast.success('Chart deleted')
  }

  const fetchDatasetColumns = async (datasetId) => {
    try {
      setLoading(true)
      const response = await fetch(`${API_BASE}/data/datasets/${datasetId}`)
      if (response.ok) {
        const data = await response.json()
        const dataset = data.dataset
        
        // Extract columns from dataset info with enhanced metadata
        let columnsList = []
        
        console.log('Dataset columns_info:', dataset.columns_info)
        console.log('Dataset data_types:', dataset.data_types)
        
        if (dataset.columns_info) {
          try {
            const columnsInfo = typeof dataset.columns_info === 'string' 
              ? JSON.parse(dataset.columns_info) 
              : dataset.columns_info
            
            const dataTypes = dataset.data_types ? 
              (typeof dataset.data_types === 'string' ? JSON.parse(dataset.data_types) : dataset.data_types) 
              : {}
            
            if (Array.isArray(columnsInfo)) {
              // Handle array format with column metadata
              columnsList = columnsInfo.map(col => {
                const colName = typeof col === 'string' ? col : col.name
                const dtype = dataTypes[colName] || (typeof col === 'object' ? col.dtype : 'object')
                
                return {
                  name: colName,
                  type: dtype,
                  dtype: dtype,
                  is_numeric: ['int64', 'float64', 'int32', 'float32', 'number'].includes(dtype),
                  is_categorical: ['object', 'category', 'string'].includes(dtype),
                  is_datetime: dtype?.includes('datetime') || dtype === 'datetime64[ns]',
                  non_null_count: typeof col === 'object' ? parseInt(col.non_null_count) || 0 : 0,
                  unique_count: typeof col === 'object' ? parseInt(col.unique_count) || 0 : 0
                }
              })
            } else if (typeof columnsInfo === 'object') {
              // Handle object format
              columnsList = Object.entries(columnsInfo).map(([name, info]) => {
                const dtype = dataTypes[name] || (typeof info === 'object' ? info.dtype : 'object')
                
                return {
                  name,
                  type: dtype,
                  dtype: dtype,
                  is_numeric: ['int64', 'float64', 'int32', 'float32', 'number'].includes(dtype),
                  is_categorical: ['object', 'category', 'string'].includes(dtype),
                  is_datetime: dtype?.includes('datetime') || dtype === 'datetime64[ns]',
                  non_null_count: typeof info === 'object' ? parseInt(info.non_null_count) || 0 : 0,
                  unique_count: typeof info === 'object' ? parseInt(info.unique_count) || 0 : 0
                }
              })
            }
          } catch (parseError) {
            console.error('Error parsing columns_info:', parseError)
            // Fallback to data_types if available
            if (dataset.data_types) {
              const dataTypes = typeof dataset.data_types === 'string'
                ? JSON.parse(dataset.data_types)
                : dataset.data_types
              columnsList = Object.entries(dataTypes).map(([name, dtype]) => ({
                name,
                type: dtype,
                dtype,
                is_numeric: ['int64', 'float64', 'int32', 'float32', 'number'].includes(dtype),
                is_categorical: ['object', 'category', 'string'].includes(dtype),
                is_datetime: dtype?.includes('datetime') || dtype === 'datetime64[ns]',
                non_null_count: 0,
                unique_count: 0
              }))
            }
          }
        }
        
        console.log('Processed columns:', columnsList)
        setColumns(columnsList)
        setSelectedDataset(dataset)
      }
    } catch (error) {
      console.error('Error fetching dataset columns:', error)
      toast.error('Failed to load dataset columns')
    } finally {
      setLoading(false)
    }
  }

  const getAIInsight = async (chart) => {
    setAiInsight({ chartId: chart.id, insight: '', loading: true })
    try {
      // Get detailed column information for better insights
      const xColumnInfo = columns.find(col => col.name === chart.xColumn)
      const yColumnInfo = columns.find(col => col.name === chart.yColumn)
      
      const response = await fetch(`${API_BASE}/visualization/insight`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chart_type: chart.type,
          dataset_name: chart.datasetName,
          dataset_id: chart.datasetId,
          x_column: chart.xColumn,
          y_column: chart.yColumn,
          title: chart.title,
          x_column_type: xColumnInfo?.type || 'unknown',
          y_column_type: yColumnInfo?.type || 'unknown',
          is_x_numeric: xColumnInfo?.is_numeric || false,
          is_y_numeric: yColumnInfo?.is_numeric || false,
          is_x_datetime: xColumnInfo?.is_datetime || false,
          is_y_datetime: yColumnInfo?.is_datetime || false,
          color_column: chart.colorColumn !== 'none' ? chart.colorColumn : null,
          size_column: chart.sizeColumn !== 'none' ? chart.sizeColumn : null,
          color_scheme: chart.colorScheme
        })
      })

      if (response.ok) {
        const data = await response.json()
        setAiInsight({ chartId: chart.id, insight: data.insight, loading: false })
      } else {
        toast.error('Failed to get AI insight.')
        setAiInsight({ chartId: chart.id, insight: '', loading: false })
      }
    } catch (error) {
      console.error('Error getting AI insight:', error)
      toast.error('An error occurred while fetching AI insight.')
      setAiInsight({ chartId: chart.id, insight: '', loading: false })
    }
  }

  // Auto-generate recommended charts based on dataset analysis
  const autoGenerateCharts = async () => {
    if (!selectedDataset) {
      toast.error('Please select a dataset first')
      return
    }

    setAutoGenerating(true)
    try {
      // Analyze dataset for chart recommendations
      const response = await fetch(`${API_BASE}/visualization/analyze/${selectedDataset.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chart_types: ['correlation_heatmap', 'distribution_plots', 'scatter_matrix', 'trend_analysis']
        })
      })

      if (response.ok) {
        const responseText = await response.text()
        let data
        
        try {
          // Clean up any NaN values before parsing JSON
          const cleanedText = responseText.replace(/\bNaN\b/g, 'null')
          data = JSON.parse(cleanedText)
        } catch (parseError) {
          console.error('JSON parsing error:', parseError)
          console.error('Response text:', responseText)
          throw new Error('Invalid JSON response from server')
        }
        
        setDatasetAnalysis(data.analysis)
        
        // Generate recommended charts
        const recommendations = data.recommendations || []
        const newCharts = []

        for (const rec of recommendations) {
          // Ensure we have valid plot config
          if (!rec.plot_config || !rec.x_column) {
            console.warn('Skipping recommendation with invalid config:', rec)
            continue
          }

          const chartConfig = {
            type: rec.chart_type,
            title: rec.title,
            xColumn: rec.x_column,
            yColumn: rec.y_column,
            colorColumn: rec.color_column || 'none',
            sizeColumn: rec.size_column || 'none',
            colorScheme: rec.color_scheme || 'viridis',
            height: 500,
            width: 800,
            plotConfig: rec.plot_config
          }

          const newChart = {
            id: Date.now().toString() + '_' + Math.random().toString(36).substr(2, 9),
            ...chartConfig,
            datasetId: selectedDataset.id,
            datasetName: selectedDataset.name,
            createdAt: new Date().toISOString(),
            isAutoGenerated: true,
            hasRealData: true // Auto-generated charts use real data
          }

          newCharts.push(newChart)
        }

        // Add to charts list
        const updatedCharts = [...charts, ...newCharts]
        setCharts(updatedCharts)
        localStorage.setItem('superhacker_charts', JSON.stringify(updatedCharts))
        
        setRecommendedCharts(newCharts)
        toast.success(`Generated ${newCharts.length} recommended charts!`)
        setCurrentTab('gallery')
      } else {
        const errorText = await response.text()
        console.error('Failed to analyze dataset:', errorText)
        toast.error('Failed to analyze dataset')
      }
    } catch (error) {
      console.error('Error auto-generating charts:', error)
      toast.error('Error generating charts. Creating basic charts instead.')
      
      // Fallback: create basic charts with available numeric columns
      const numericColumns = columns.filter(col => col.is_numeric)
      
      if (numericColumns.length >= 2) {
        // Create a correlation heatmap
        const correlationChart = createBasicChart('heatmap', 'Correlation Matrix', numericColumns[0].name, numericColumns[1].name)
        saveChart(correlationChart)
      }
      
      if (numericColumns.length >= 1) {
        // Create distribution plot
        const distributionChart = createBasicChart('histogram', `${numericColumns[0].name} Distribution`, numericColumns[0].name, '')
        saveChart(distributionChart)
      }
      
      if (numericColumns.length >= 2) {
        // Create scatter plot
        const scatterChart = createBasicChart('scatter', `${numericColumns[0].name} vs ${numericColumns[1].name}`, numericColumns[0].name, numericColumns[1].name)
        saveChart(scatterChart)
      }
      
      toast.success('Created basic visualization suite!')
    } finally {
      setAutoGenerating(false)
    }
  }

  // Create a basic chart configuration
  const createBasicChart = (type, title, xColumn, yColumn) => {
    const plotConfig = generatePlotlyConfig()
    return {
      type,
      title,
      xColumn,
      yColumn,
      colorColumn: 'none',
      sizeColumn: 'none',
      colorScheme: 'viridis',
      height: 500,
      width: 800,
      plotConfig
    }
  }

  // Show dataset overview with column statistics
  const showDatasetDetails = () => {
    setShowDatasetOverview(true)
  }

  // Generate plotly config with real data or fallback to mock data
  const generatePreviewPlotlyConfig = () => {    
    if (!selectedDataset || !chartConfig.xColumn) return null

    // If we have real data, use it; otherwise fall back to mock data
    if (realChartData) {
      return generateRealPlotlyConfig(chartConfig, realChartData, COLOR_SCHEMES)
    }

    // Fallback to mock data (existing implementation)
    return generatePlotlyConfig()
  }

  const generatePlotlyConfig = () => {
    const { type, xColumn, yColumn, colorColumn, sizeColumn, title, colorScheme, height, width } = chartConfig
    
    if (!selectedDataset || !xColumn) return null

    // Generate sample data for demo (in real implementation, this would fetch actual data)
    const sampleSize = 100
    const data = Array.from({ length: sampleSize }, (_, i) => ({
      x: Math.random() * 100,
      y: Math.random() * 100 + Math.sin(i / 10) * 20,
      color: Math.random(),
      size: Math.random() * 20 + 5
    }))

    const colors = COLOR_SCHEMES[colorScheme] || COLOR_SCHEMES.viridis

    let plotData = []
    let layout = {
      title: title || `${type.charAt(0).toUpperCase() + type.slice(1)} Chart`,
      xaxis: { title: xColumn },
      yaxis: { title: yColumn },
      height,
      width,
      autosize: true,
      margin: { l: 60, r: 60, t: 80, b: 60 },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white'
    }

    switch (type) {
      case 'line':
        plotData = [{
          x: data.map(d => d.x),
          y: data.map(d => d.y),
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: colors[0], width: 3 },
          marker: { size: 6 }
        }]
        break

      case 'bar':
        plotData = [{
          x: data.slice(0, 20).map((_, i) => `Category ${i + 1}`),
          y: data.slice(0, 20).map(d => d.y),
          type: 'bar',
          marker: { color: colors }
        }]
        break

      case 'scatter':
        plotData = [{
          x: data.map(d => d.x),
          y: data.map(d => d.y),
          mode: 'markers',
          type: 'scatter',
          marker: {
            size: sizeColumn && sizeColumn !== 'none' ? data.map(d => d.size) : 8,
            color: colorColumn && colorColumn !== 'none' ? data.map(d => d.color) : colors[0],
            colorscale: colorScheme === 'viridis' ? 'Viridis' : 'Plasma',
            showscale: !!(colorColumn && colorColumn !== 'none')
          }
        }]
        break

      case 'pie': {
        const pieData = data.slice(0, 8)
        plotData = [{
          values: pieData.map(d => d.y),
          labels: pieData.map((_, i) => `Segment ${i + 1}`),
          type: 'pie',
          marker: { colors }
        }]
        break
      }

      case 'heatmap': {
        const heatmapData = Array.from({ length: 10 }, () =>
          Array.from({ length: 10 }, () => Math.random())
        )
        plotData = [{
          z: heatmapData,
          type: 'heatmap',
          colorscale: colorScheme === 'viridis' ? 'Viridis' : 'Plasma'
        }]
        break
      }

      case 'box':
        plotData = [{
          y: data.map(d => d.y),
          type: 'box',
          name: yColumn,
          marker: { color: colors[0] }
        }]
        break

      case 'histogram':
        plotData = [{
          x: data.map(d => d.x),
          type: 'histogram',
          marker: { color: colors[0] },
          opacity: 0.7
        }]
        break

      case 'violin':
        plotData = [{
          y: data.map(d => d.y),
          type: 'violin',
          name: yColumn,
          line: { color: colors[0] }
        }]
        break

      case 'surface': {
        const surfaceData = Array.from({ length: 20 }, (_, i) =>
          Array.from({ length: 20 }, (_, j) =>
            Math.sin(i / 5) * Math.cos(j / 5) * 10
          )
        )
        plotData = [{
          z: surfaceData,
          type: 'surface',
          colorscale: colorScheme === 'viridis' ? 'Viridis' : 'Plasma'
        }]
        layout.scene = {
          xaxis: { title: xColumn },
          yaxis: { title: yColumn },
          zaxis: { title: 'Z' }
        }
        break
      }

      case 'radar': {
        const categories = ['Speed', 'Reliability', 'Comfort', 'Safety', 'Efficiency']
        plotData = [{
          type: 'scatterpolar',
          r: categories.map(() => Math.random() * 100),
          theta: categories,
          fill: 'toself',
          name: 'Series 1',
          line: { color: colors[0] }
        }]
        layout.polar = {
          radialaxis: { visible: true, range: [0, 100] }
        }
        break
      }
    }

    return { data: plotData, layout }
  }

  const createChart = async () => {
    if (!selectedDataset || !chartConfig.xColumn) {
      toast.error('Please select a dataset and X column')
      return
    }

    // Use real data if available, otherwise fall back to generated config
    let plotConfig
    if (realChartData) {
      try {
        plotConfig = generateRealPlotlyConfig(chartConfig, realChartData, COLOR_SCHEMES)
      } catch (error) {
        console.error('Error generating real plot config:', error)
        plotConfig = generatePlotlyConfig()
      }
    } else {
      plotConfig = generatePlotlyConfig()
    }
    
    if (!plotConfig || !plotConfig.data) {
      toast.error('Unable to generate chart configuration')
      return
    }

    const newChart = {
      ...chartConfig,
      datasetId: selectedDataset.id,
      datasetName: selectedDataset.name,
      plotConfig,
      hasRealData: !!realChartData // Track whether chart uses real data
    }

    saveChart(newChart)
    setCurrentTab('gallery')
    
    // Reset form
    setChartConfig({
      type: 'line',
      title: '',
      xColumn: '',
      yColumn: '',
      colorColumn: 'none',
      sizeColumn: 'none',
      colorScheme: 'viridis',
      height: 500,
      width: 800
    })
    
    // Clear real data
    setRealChartData(null)
  }

  // Dashboard management functions
  const loadSavedDashboards = () => {
    const saved = localStorage.getItem('superhacker_dashboards')
    if (saved) {
      try {
        setDashboards(JSON.parse(saved))
      } catch (error) {
        console.error('Error loading saved dashboards:', error)
      }
    }
  }

  const saveDashboard = (dashboard) => {
    const newDashboard = {
      id: Date.now().toString(),
      ...dashboard,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
    const updatedDashboards = [...dashboards, newDashboard]
    setDashboards(updatedDashboards)
    localStorage.setItem('superhacker_dashboards', JSON.stringify(updatedDashboards))
    toast.success('Dashboard saved successfully!')
    return newDashboard
  }

  // eslint-disable-next-line no-unused-vars
  const updateDashboard = (dashboardId, updates) => {
    const updatedDashboards = dashboards.map(dashboard =>
      dashboard.id === dashboardId
        ? { ...dashboard, ...updates, updatedAt: new Date().toISOString() }
        : dashboard
    )
    setDashboards(updatedDashboards)
    localStorage.setItem('superhacker_dashboards', JSON.stringify(updatedDashboards))
    toast.success('Dashboard updated successfully!')
  }

  const deleteDashboard = (dashboardId) => {
    const updatedDashboards = dashboards.filter(dashboard => dashboard.id !== dashboardId)
    setDashboards(updatedDashboards)
    localStorage.setItem('superhacker_dashboards', JSON.stringify(updatedDashboards))
    toast.success('Dashboard deleted')
  }

  const createDashboard = () => {
    if (!dashboardConfig.name.trim()) {
      toast.error('Please enter a dashboard name')
      return
    }

    const newDashboard = {
      ...dashboardConfig,
      charts: selectedChartsForDashboard.map(chartId => {
        const chart = charts.find(c => c.id === chartId)
        return {
          id: chartId,
          x: Math.floor(Math.random() * 4) * 3, // Grid position
          y: Math.floor(Math.random() * 4) * 2,
          w: 6, // Width in grid units
          h: 4, // Height in grid units
          title: chart?.title || 'Untitled Chart'
        }
      })
    }

    saveDashboard(newDashboard)
    setShowCreateDashboard(false)
    setDashboardConfig({
      name: '',
      description: '',
      layout: 'grid',
      refreshInterval: 0,
      theme: 'light'
    })
    setSelectedChartsForDashboard([])
  }

  const toggleChartForDashboard = (chartId) => {
    setSelectedChartsForDashboard(prev =>
      prev.includes(chartId)
        ? prev.filter(id => id !== chartId)
        : [...prev, chartId]
    )
  }

  const loadDashboardCharts = (dashboard) => {
    const dashboardChartData = dashboard.charts.map(chartLayout => {
      const chart = charts.find(c => c.id === chartLayout.id)
      return {
        ...chartLayout,
        chartData: chart
      }
    })
    setDashboardCharts(dashboardChartData)
    setSelectedDashboard(dashboard)
  }

  const filteredCharts = charts.filter(chart => {
    if (filterType !== 'all' && chart.type !== filterType) {
      return false
    }
    if (searchTerm && !chart.title.toLowerCase().includes(searchTerm.toLowerCase())) {
      return false
    }
    return true
  })

  const renderChartCard = (chart) => (
    <Card key={chart.id} className="group hover:shadow-lg transition-all duration-300 border-0 shadow-md overflow-hidden w-full max-w-5xl mx-auto">
      {/* Header with Chart Info and Actions */}
      <CardHeader className={`bg-gradient-to-r ${CHART_TYPES[chart.type]?.gradient || 'from-gray-400 to-gray-600'} text-white py-4`}>
        <div className="flex items-center justify-between">
          <div className="flex-grow">
            <div className="flex items-center space-x-2">
              <CardTitle className="text-lg font-semibold truncate pr-2" title={chart.title}>
                {chart.title || 'Untitled Chart'}
              </CardTitle>
              {chart.hasRealData && (
                <Badge variant="secondary" className="bg-green-500/20 text-green-100 text-xs px-1.5 py-0.5">
                  Real Data
                </Badge>
              )}
            </div>
            <p className="text-white/80 text-sm mt-0.5">
              {CHART_TYPES[chart.type]?.name || 'Chart'} • {chart.datasetName}
            </p>
          </div>
          <div className="flex items-center space-x-2 flex-shrink-0">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => getAIInsight(chart)}
              className="bg-white/20 hover:bg-white/30 text-white border-0 p-2"
              title="Get AI Insight"
            >
              <Bot className="w-4 h-4" />
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={() => deleteChart(chart.id)}
              className="bg-red-500/20 hover:bg-red-500/40 text-white border-0 p-2"
              title="Delete Chart"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      {/* Chart Visualization */}
      <CardContent className="p-0">
        <div className="bg-gray-50 dark:bg-gray-800/20 h-64 flex items-center justify-center relative">
          {chart.plotConfig && chart.plotConfig.data ? (
            <Plot
              data={chart.plotConfig.data}
              layout={{ 
                ...chart.plotConfig.layout, 
                autosize: true, 
                paper_bgcolor: 'transparent', 
                plot_bgcolor: 'transparent',
                height: 260,
                margin: { l: 40, r: 40, t: 20, b: 40 },
                font: { 
                  color: document.body.classList.contains('dark') ? '#E5E7EB' : '#1F2937',
                  size: 10
                }
              }}
              style={{ width: '100%', height: '260px' }}
              useResizeHandler={true}
              config={{ responsive: true, displayModeBar: false, staticPlot: false }}
            />
          ) : (
            <div className="text-center text-muted-foreground">
              <BarChart3 className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">Chart data unavailable</p>
            </div>
          )}
        </div>
      </CardContent>

      {/* AI Insight Section */}
      {aiInsight.chartId === chart.id && (
        <CardContent className="p-0">
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/30 dark:to-blue-900/30 border-t border-purple-200 dark:border-purple-700/50">
            <div className="p-4">
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0">
                  <div className="w-6 h-6 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center">
                    <Bot className="w-3 h-3 text-white" />
                  </div>
                </div>
                <div className="flex-grow min-w-0">
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2 flex items-center text-sm">
                    AI-Powered Business Insight
                    <Badge variant="secondary" className="ml-2 bg-purple-100 text-purple-700 dark:bg-purple-800 dark:text-purple-200 text-xs">
                      Beta
                    </Badge>
                  </h4>
                  
                  {aiInsight.loading ? (
                    <div className="flex items-center space-x-3 py-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-purple-500 border-t-transparent"></div>
                      <span className="text-gray-600 dark:text-gray-300 text-xs">
                        Analyzing your data to generate actionable insights...
                      </span>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-purple-200 dark:border-purple-700/50 shadow-sm">
                        <div className="flex items-start space-x-2">
                          <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-1.5 flex-shrink-0"></div>
                          <div className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed prose prose-sm max-w-none">
                            {aiInsight.insight.split('\n').map((line, index) => {
                              // Handle headers (###)
                              if (line.startsWith('### ')) {
                                return (
                                  <h3 key={index} className="text-base font-semibold text-gray-800 dark:text-gray-200 mt-4 mb-2 first:mt-0">
                                    {line.replace('### ', '')}
                                  </h3>
                                );
                              }
                              
                              // Handle bullet points with **bold** text
                              if (line.startsWith('* **')) {
                                const match = line.match(/^\* \*\*(.*?)\*\*: (.*)$/);
                                if (match) {
                                  return (
                                    <div key={index} className="flex items-start space-x-2 mb-2">
                                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                                      <div>
                                        <span className="font-semibold text-gray-800 dark:text-gray-200">{match[1]}</span>
                                        <span className="text-gray-700 dark:text-gray-300">: {match[2]}</span>
                                      </div>
                                    </div>
                                  );
                                }
                              }
                              
                              // Handle regular bullet points
                              if (line.startsWith('* ')) {
                                return (
                                  <div key={index} className="flex items-start space-x-2 mb-2">
                                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                                    <span className="text-gray-700 dark:text-gray-300">{line.replace('* ', '')}</span>
                                  </div>
                                );
                              }
                              
                              // Handle bold text inline
                              if (line.includes('**') && line.trim() !== '') {
                                const parts = line.split('**');
                                return (
                                  <p key={index} className="mb-2">
                                    {parts.map((part, i) => 
                                      i % 2 === 1 ? (
                                        <span key={i} className="font-semibold text-gray-800 dark:text-gray-200">{part}</span>
                                      ) : (
                                        <span key={i} className="text-gray-700 dark:text-gray-300">{part}</span>
                                      )
                                    )}
                                  </p>
                                );
                              }
                              
                              // Handle empty lines
                              if (line.trim() === '') {
                                return <div key={index} className="h-2"></div>;
                              }
                              
                              // Handle regular text
                              return (
                                <p key={index} className="text-gray-700 dark:text-gray-300 mb-2">
                                  {line}
                                </p>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 pt-1">
                        <span className="flex items-center space-x-1">
                          <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                          <span>Generated by AI</span>
                        </span>
                        <span>
                          {new Date().toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      )}

      {/* Footer with Metadata */}
      <CardContent className="p-4 bg-gray-50 dark:bg-gray-800/30 border-t border-gray-100 dark:border-gray-800">
        <div className="flex flex-wrap items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <div className="flex items-center flex-wrap gap-3">
            <span className="flex items-center space-x-1.5">
              <Database className="w-4 h-4 text-gray-400" />
              <span>{chart.xColumn} × {chart.yColumn}</span>
            </span>
            <Badge variant="outline" className="text-xs font-medium">
              {chart.colorScheme}
            </Badge>
            {chart.hasRealData && (
              <Badge variant="outline" className="text-xs bg-green-50 text-green-700 border-green-200 font-medium">
                Real Data
              </Badge>
            )}
          </div>
          <span className="text-xs">
            {new Date(chart.createdAt).toLocaleDateString()}
          </span>
        </div>
      </CardContent>
    </Card>
  )

  // Main component render
  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-gray-200 font-sans">
      <div className="w-full max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Visualizations
            </h1>
            <p className="text-xl text-muted-foreground mt-2">
              Create stunning interactive charts and dashboards with Plotly
            </p>
          </div>
          <div className="flex items-center space-x-3">
            {selectedDataset && (
              <Button variant="outline" onClick={showDatasetDetails}>
                <Database className="h-4 w-4 mr-2" />
                Dataset Info
              </Button>
            )}
            {selectedDataset && (
              <Button 
                variant="outline" 
                onClick={autoGenerateCharts}
                disabled={autoGenerating}
                className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white border-0"
              >
                {autoGenerating ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Auto-Generate
                  </>
                )}
              </Button>
            )}
            <Button variant="outline" onClick={loadSavedCharts}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button 
              className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
              onClick={() => setCurrentTab('create')}
            >
              <Plus className="h-4 w-4 mr-2" />
              Create Chart
            </Button>
          </div>
        </div>

        {/* Chart Gallery */}
        <Tabs value={currentTab} onValueChange={setCurrentTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="gallery">Chart Gallery</TabsTrigger>
            <TabsTrigger value="create">Create Chart</TabsTrigger>
            <TabsTrigger value="templates">Templates</TabsTrigger>
            <TabsTrigger value="dashboards">Dashboards</TabsTrigger>
          </TabsList>

          <TabsContent value="gallery" className="space-y-6">
            {/* Filters */}
            <div className="flex items-center space-x-4">
              <div className="relative flex-1 max-w-sm">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search charts..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="w-48">
                  <Filter className="h-4 w-4 mr-2" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  {Object.entries(CHART_TYPES).map(([key, config]) => (
                    <SelectItem key={key} value={key}>
                      {config.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Charts Grid */}
            {filteredCharts.length === 0 ? (
              <Card className="py-12">
                <CardContent className="text-center">
                  <BarChart3 className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No Charts Yet</h3>
                  <p className="text-muted-foreground mb-4">
                    Create your first visualization to get started
                  </p>
                  <Button onClick={() => setCurrentTab('create')}>
                    <Plus className="h-4 w-4 mr-2" />
                    Create First Chart
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 gap-10">
                {filteredCharts.map(renderChartCard)}
              </div>
            )}
          </TabsContent>

          <TabsContent value="create" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Configuration Panel */}
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-semibold flex items-center">
                    <Sparkles className="h-5 w-5 mr-2" />
                    Chart Configuration
                  </h3>
                  <Badge variant="secondary">
                    {CHART_TYPES[chartConfig.type]?.name}
                  </Badge>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label>Dataset</Label>
                    <Select 
                      value={selectedDataset?.id?.toString() || ''}
                      onValueChange={(value) => fetchDatasetColumns(value)}
                      disabled={loading}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a dataset" />
                      </SelectTrigger>
                      <SelectContent>
                        {datasets.map(dataset => (
                          <SelectItem key={dataset.id} value={dataset.id.toString()}>
                            {dataset.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label>Chart Type</Label>
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      {Object.entries(CHART_TYPES).map(([key, config]) => {
                        const Icon = config.icon
                        return (
                          <Button
                            key={key}
                            variant={chartConfig.type === key ? "default" : "outline"}
                            className={`justify-start h-auto p-3 ${
                              chartConfig.type === key 
                                ? `bg-gradient-to-r ${config.gradient} text-white` 
                                : ''
                            }`}
                            onClick={() => setChartConfig(prev => ({ ...prev, type: key }))}
                          >
                            <Icon className="h-4 w-4 mr-2" />
                            <div className="text-left">
                              <div className="text-sm font-medium">{config.name}</div>
                            </div>
                          </Button>
                        )
                      })}
                    </div>
                  </div>

                  <div>
                    <Label>Chart Title</Label>
                    <Input
                      value={chartConfig.title}
                      onChange={(e) => setChartConfig(prev => ({ ...prev, title: e.target.value }))}
                      placeholder="Enter chart title"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>X Column</Label>
                      <Select 
                        value={chartConfig.xColumn}
                        onValueChange={(value) => setChartConfig(prev => ({ ...prev, xColumn: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select X column" />
                        </SelectTrigger>
                        <SelectContent>
                          {columns.map(column => (
                            <SelectItem key={column.name} value={column.name}>
                              <div className="flex items-center justify-between w-full">
                                <span>{column.name}</span>
                                <div className="flex items-center space-x-1 ml-2">
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
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label>Y Column</Label>
                      <Select 
                        value={chartConfig.yColumn}
                        onValueChange={(value) => setChartConfig(prev => ({ ...prev, yColumn: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select Y column" />
                        </SelectTrigger>
                        <SelectContent>
                          {columns.map(column => (
                            <SelectItem key={column.name} value={column.name}>
                              <div className="flex items-center justify-between w-full">
                                <span>{column.name}</span>
                                <div className="flex items-center space-x-1 ml-2">
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
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {['scatter', 'bubble'].includes(chartConfig.type) && (
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Color Column (Optional)</Label>
                        <Select 
                          value={chartConfig.colorColumn}
                          onValueChange={(value) => setChartConfig(prev => ({ ...prev, colorColumn: value }))}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select color column" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">None</SelectItem>
                            {columns.map(column => (
                              <SelectItem key={column.name} value={column.name}>
                                <div className="flex items-center justify-between w-full">
                                  <span>{column.name}</span>
                                  <div className="flex items-center space-x-1 ml-2">
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
                                  </div>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label>Size Column (Optional)</Label>
                        <Select 
                          value={chartConfig.sizeColumn}
                          onValueChange={(value) => setChartConfig(prev => ({ ...prev, sizeColumn: value }))}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select size column" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">None</SelectItem>
                            {columns.filter(col => col.is_numeric).map(column => (
                              <SelectItem key={column.name} value={column.name}>
                                <div className="flex items-center justify-between w-full">
                                  <span>{column.name}</span>
                                  <Badge variant="secondary" className="text-xs bg-blue-100 text-blue-700 ml-2">
                                    NUM
                                  </Badge>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  )}

                  <div>
                    <Label>Color Scheme</Label>
                    <Select 
                      value={chartConfig.colorScheme}
                      onValueChange={(value) => setChartConfig(prev => ({ ...prev, colorScheme: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(COLOR_SCHEMES).map(([key, colors]) => (
                          <SelectItem key={key} value={key}>
                            <div className="flex items-center">
                              <div className="flex mr-2">
                                {colors.slice(0, 4).map((color, i) => (
                                  <div
                                    key={i}
                                    className="w-3 h-3 rounded-full border border-gray-300"
                                    style={{ backgroundColor: color }}
                                  />
                                ))}
                              </div>
                              {key.charAt(0).toUpperCase() + key.slice(1)}
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>Height (px)</Label>
                      <Input
                        type="number"
                        value={chartConfig.height}
                        onChange={(e) => setChartConfig(prev => ({ ...prev, height: parseInt(e.target.value) }))}
                        min="300"
                        max="1000"
                      />
                    </div>
                    <div>
                      <Label>Width (px)</Label>
                      <Input
                        type="number"
                        value={chartConfig.width}
                        onChange={(e) => setChartConfig(prev => ({ ...prev, width: parseInt(e.target.value) }))}
                        min="400"
                        max="1200"
                      />
                    </div>
                  </div>
                </div>

                <div className="flex justify-end space-x-2">
                  <Button variant="outline" onClick={() => setCurrentTab('gallery')}>
                    <X className="h-4 w-4 mr-2" />
                    Cancel
                  </Button>
                  <Button onClick={createChart} disabled={!selectedDataset || !chartConfig.xColumn}>
                    <Save className="h-4 w-4 mr-2" />
                    Create Chart
                  </Button>
                </div>
              </div>

              {/* Preview Panel */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="text-lg font-medium">Live Preview</Label>
                  <Badge variant="secondary">
                    {CHART_TYPES[chartConfig.type]?.name}
                  </Badge>
                </div>
                
                <div className="border rounded-lg p-6 bg-gray-50 min-h-[500px] flex items-center justify-center">
                  {selectedDataset && chartConfig.xColumn ? (
                    <div className="w-full h-full">
                      {loadingRealData ? (
                        <div className="flex items-center justify-center h-[450px]">
                          <div className="text-center">
                            <RefreshCw className="h-8 w-8 mx-auto mb-4 animate-spin text-blue-500" />
                            <p className="text-lg font-medium">Loading real data...</p>
                            <p className="text-sm text-muted-foreground">Fetching data from {selectedDataset.name}</p>
                          </div>
                        </div>
                      ) : (
                        (() => {
                          const plotConfig = generatePreviewPlotlyConfig()
                          return plotConfig ? (
                            <div className="space-y-2">
                              {realChartData && (
                                <div className="flex items-center justify-between text-sm text-muted-foreground bg-green-50 p-2 rounded">
                                  <span className="flex items-center">
                                    <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                                    Using real data from {selectedDataset.name}
                                  </span>
                                  <span>
                                    {realChartData.x?.length || realChartData.values?.length || 0} data points
                                  </span>
                                </div>
                              )}
                              <Plot
                                data={plotConfig.data}
                                layout={{
                                  ...plotConfig.layout,
                                  height: 450,
                                  width: undefined,
                                  autosize: true
                                }}
                                config={{
                                  responsive: true,
                                  displayModeBar: true,
                                  displaylogo: false,
                                  modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                                }}
                                style={{ width: '100%', height: '450px' }}
                              />
                            </div>
                          ) : (
                            <div className="text-center text-muted-foreground">
                              <BarChart3 className="h-16 w-16 mx-auto mb-4 opacity-50" />
                              <p className="text-lg font-medium mb-2">No Data Available</p>
                              <p>Unable to load chart data for the selected configuration</p>
                            </div>
                          )
                        })()
                      )}
                    </div>
                  ) : (
                    <div className="text-center text-muted-foreground">
                      <BarChart3 className="h-16 w-16 mx-auto mb-4 opacity-50" />
                      <p className="text-lg font-medium mb-2">Chart Preview</p>
                      <p>Select a dataset and columns to see live preview</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="templates" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(CHART_TYPES).map(([key, config]) => {
                const Icon = config.icon
                return (
                  <Card key={key} className="group hover:shadow-lg transition-all duration-300 cursor-pointer border-0 shadow-md">
                    <CardHeader className={`bg-gradient-to-r ${config.gradient} text-white rounded-t-lg`}>
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-white/20 rounded-lg">
                          <Icon className="h-6 w-6" />
                        </div>
                        <div>
                          <CardTitle>{config.name}</CardTitle>
                          <p className="text-white/80 text-sm">{config.description}</p>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-6">
                      <div className="space-y-3">
                        <div className="text-sm text-muted-foreground">
                          Perfect for: {config.description}
                        </div>
                        <Button 
                          className="w-full" 
                          variant="outline"
                          onClick={() => {
                            setChartConfig(prev => ({ ...prev, type: key }))
                            setCurrentTab('create')
                          }}
                        >
                          Use Template
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </TabsContent>

          <TabsContent value="dashboards" className="space-y-6">
            {!selectedDashboard ? (
              // Dashboard Gallery View
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-2xl font-bold">Interactive Dashboards</h3>
                    <p className="text-muted-foreground">Combine multiple charts into interactive dashboards</p>
                  </div>
                  <Button 
                    onClick={() => setShowCreateDashboard(true)}
                    className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Create Dashboard
                  </Button>
                </div>

                {dashboards.length === 0 ? (
                  <Card className="py-16">
                    <CardContent className="text-center">
                      <Layout className="h-20 w-20 mx-auto mb-6 text-muted-foreground" />
                      <h3 className="text-xl font-semibold mb-3">No Dashboards Yet</h3>
                      <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                        Create your first dashboard by combining multiple charts into a unified view
                      </p>
                      <Button 
                        onClick={() => setShowCreateDashboard(true)}
                        className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700"
                      >
                        <Plus className="h-4 w-4 mr-2" />
                        Create First Dashboard
                      </Button>
                    </CardContent>
                  </Card>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {dashboards.map((dashboard) => (
                      <Card key={dashboard.id} className="group hover:shadow-lg transition-all duration-300 cursor-pointer border-0 shadow-md">
                        <CardHeader className="bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-t-lg">
                          <div className="flex items-center justify-between">
                            <div>
                              <CardTitle className="text-lg">{dashboard.name}</CardTitle>
                              <p className="text-white/80 text-sm">{dashboard.description || 'No description'}</p>
                            </div>
                            <Badge variant="secondary" className="bg-white/20 text-white border-0">
                              {dashboard.charts?.length || 0} charts
                            </Badge>
                          </div>
                        </CardHeader>
                        
                        <CardContent className="p-6">
                          <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
                              <div className="flex items-center">
                                <Grid className="h-3 w-3 mr-1" />
                                {dashboard.layout || 'Grid'} layout
                              </div>
                              <div className="flex items-center">
                                <RefreshCw className="h-3 w-3 mr-1" />
                                {dashboard.refreshInterval > 0 ? `${dashboard.refreshInterval}s` : 'Manual'}
                              </div>
                            </div>
                            
                            <div className="text-sm text-muted-foreground">
                              Updated {new Date(dashboard.updatedAt || dashboard.createdAt).toLocaleDateString()}
                            </div>
                            
                            <div className="flex space-x-2">
                              <Button 
                                size="sm" 
                                className="flex-1"
                                onClick={() => loadDashboardCharts(dashboard)}
                              >
                                <Eye className="h-3 w-3 mr-1" />
                                View
                              </Button>
                              <Button size="sm" variant="outline">
                                <Settings className="h-3 w-3 mr-1" />
                                Edit
                              </Button>
                              <Button 
                                size="sm" 
                                variant="destructive"
                                onClick={() => deleteDashboard(dashboard.id)}
                              >
                                <X className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}

                {/* Create Dashboard Dialog */}
                {showCreateDashboard && (
                  <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <Card className="w-full max-w-4xl max-h-[90vh] overflow-y-auto m-4">
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <div>
                            <CardTitle className="flex items-center">
                              <Sparkles className="h-5 w-5 mr-2" />
                              Create Dashboard
                            </CardTitle>
                            <CardDescription>Combine charts into an interactive dashboard</CardDescription>
                          </div>
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => setShowCreateDashboard(false)}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      </CardHeader>
                      
                      <CardContent className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="space-y-4">
                            <div>
                              <Label>Dashboard Name</Label>
                              <Input
                                value={dashboardConfig.name}
                                onChange={(e) => setDashboardConfig(prev => ({ ...prev, name: e.target.value }))}
                                placeholder="Enter dashboard name"
                              />
                            </div>
                            
                            <div>
                              <Label>Description (Optional)</Label>
                              <Input
                                value={dashboardConfig.description}
                                onChange={(e) => setDashboardConfig(prev => ({ ...prev, description: e.target.value }))}
                                placeholder="Enter description"
                              />
                            </div>
                            
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <Label>Layout Style</Label>
                                <Select 
                                  value={dashboardConfig.layout}
                                  onValueChange={(value) => setDashboardConfig(prev => ({ ...prev, layout: value }))}
                                >
                                  <SelectTrigger>
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectItem value="grid">Grid Layout</SelectItem>
                                    <SelectItem value="masonry">Masonry Layout</SelectItem>
                                    <SelectItem value="tabs">Tabbed Layout</SelectItem>
                                  </SelectContent>
                                </Select>
                              </div>
                              
                              <div>
                                <Label>Auto Refresh (seconds)</Label>
                                <Select 
                                  value={dashboardConfig.refreshInterval.toString()}
                                  onValueChange={(value) => setDashboardConfig(prev => ({ ...prev, refreshInterval: parseInt(value) }))}
                                >
                                  <SelectTrigger>
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectItem value="0">Manual</SelectItem>
                                    <SelectItem value="30">30 seconds</SelectItem>
                                    <SelectItem value="60">1 minute</SelectItem>
                                    <SelectItem value="300">5 minutes</SelectItem>
                                    <SelectItem value="900">15 minutes</SelectItem>
                                  </SelectContent>
                                </Select>
                              </div>
                            </div>
                          </div>
                          
                          <div>
                            <Label>Select Charts ({selectedChartsForDashboard.length})</Label>
                            <ScrollArea className="h-72">
                              <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-900/50 min-h-[200px]">
                                {charts.length > 0 ? (
                                  <div className="space-y-3">
                                    {charts.map(chart => {
                                      const ChartIcon = CHART_TYPES[chart.type]?.icon
                                      return (
                                        <div 
                                          key={chart.id}
                                          className={`flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors ${selectedChartsForDashboard.includes(chart.id) ? 'bg-blue-100 dark:bg-blue-900/50 border-blue-400' : 'bg-white dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800'}`}
                                          onClick={() => toggleChartForDashboard(chart.id)}
                                        >
                                          <div className="flex items-center space-x-3">
                                            <div className={`p-1.5 rounded-md bg-gradient-to-r ${CHART_TYPES[chart.type]?.gradient || 'from-gray-400 to-gray-600'}`}>
                                              {ChartIcon && <ChartIcon className="h-4 w-4 text-white" />}
                                            </div>
                                            <div>
                                              <p className="font-medium text-sm">{chart.title}</p>
                                              <p className="text-xs text-muted-foreground">{chart.datasetName}</p>
                                            </div>
                                          </div>
                                          {selectedChartsForDashboard.includes(chart.id) && (
                                            <div className="h-5 w-5 rounded-full bg-blue-500 text-white flex items-center justify-center">
                                              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                              </svg>
                                            </div>
                                          )}
                                        </div>
                                      )
                                    })}
                                  </div>
                                ) : (
                                  <div className="text-center py-10">
                                    <p className="text-muted-foreground">No charts available. Create some charts first.</p>
                                  </div>
                                )}
                              </div>
                            </ScrollArea>
                          </div>
                        </div>
                      </CardContent>
                      
                      <CardFooter className="flex justify-end space-x-3">
                        <Button variant="outline" onClick={() => setShowCreateDashboard(false)}>Cancel</Button>
                        <Button 
                          onClick={createDashboard}
                          className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700"
                        >
                          <Plus className="h-4 w-4 mr-2" />
                          Create Dashboard
                        </Button>
                      </CardFooter>
                    </Card>
                  </div>
                )}
              </div>
            ) : (
              // Single Dashboard View
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Button variant="outline" onClick={() => setSelectedDashboard(null)}>
                      &larr; Back to Dashboards
                    </Button>
                    <div>
                      <h3 className="text-2xl font-bold">{selectedDashboard.name}</h3>
                      <p className="text-muted-foreground">{selectedDashboard.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <Button variant="outline" onClick={() => setIsDashboardEditMode(!isDashboardEditMode)}>
                      {isDashboardEditMode ? 'Save Layout' : 'Edit Layout'}
                    </Button>
                    <Button>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Refresh
                    </Button>
                  </div>
                </div>
                
                <div className="grid grid-cols-12 gap-6">
                  {dashboardCharts.map(({ chartData, ...layoutProps }) => (
                    <div 
                      key={layoutProps.id} 
                      className="col-span-12 sm:col-span-6 lg:col-span-4"
                    >
                      {chartData ? (
                        <Card className="h-full flex flex-col">
                          <CardHeader>
                            <CardTitle>{chartData.title}</CardTitle>
                          </CardHeader>
                          <CardContent className="flex-grow">
                            <Plot
                              data={chartData.plotConfig.data}
                              layout={{ ...chartData.plotConfig.layout, autosize: true }}
                              style={{ width: '100%', height: '100%' }}
                              useResizeHandler={true}
                              config={{ responsive: true }}
                            />
                          </CardContent>
                        </Card>
                      ) : (
                        <Card className="h-full flex items-center justify-center">
                          <CardContent>
                            <p>Chart data not found</p>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* Dataset Overview Modal */}
      {showDatasetOverview && selectedDataset && (
        <DatasetOverviewModal
          selectedDataset={selectedDataset}
          columns={columns}
          showDatasetOverview={showDatasetOverview}
          setShowDatasetOverview={setShowDatasetOverview}
          onAutoGenerate={autoGenerateCharts}
        />
      )}
    </div>
  )
}

