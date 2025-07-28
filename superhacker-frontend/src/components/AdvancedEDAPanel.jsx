import React, { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react'
import { 
  ChevronUp, 
  ChevronDown,
  BarChart3,
  PieChart,
  Activity,
  Target,
  Zap,
  Eye,
  Brain,
  Layers,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Info,
  Download,
  RefreshCw,
  X,
  Maximize2,
  Minimize2,
  Filter,
  Search
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import Plotly from 'plotly.js-dist-min'

const OptimizedChart = memo(({ chartData, chartId, title, description, category, index, totalCharts }) => {
  const chartRef = useRef(null)
  const [isVisible, setIsVisible] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // Utility function to format numbers intelligently
  const formatNumber = useCallback((num) => {
    if (typeof num !== 'number' || isNaN(num)) return num
    
    // For very large or very small numbers, use scientific notation
    if (Math.abs(num) >= 1e6 || (Math.abs(num) < 0.001 && Math.abs(num) > 0)) {
      return num.toExponential(2)
    }
    
    // For regular numbers, use fixed decimal places
    if (Math.abs(num) >= 1000) {
      return num.toFixed(1) // 1 decimal for thousands
    } else if (Math.abs(num) >= 1) {
      return num.toFixed(3) // 3 decimals for regular numbers
    } else {
      return num.toFixed(4) // 4 decimals for small numbers
    }
  }, [])

  // Enhanced number formatting for problematic charts like Q-Q plots
  const aggressiveFormatNumber = useCallback((num) => {
    if (typeof num !== 'number' || isNaN(num)) return num
    
    // Always limit to 3 decimal places for Q-Q plots and statistical charts
    if (Math.abs(num) >= 1e6 || Math.abs(num) < 1e-6) {
      return parseFloat(num.toExponential(2))
    }
    
    // Force 3 decimal places maximum
    return parseFloat(num.toFixed(3))
  }, [])

  // Utility function to get smart tick format based on data range
  const getSmartTickFormat = useCallback((data) => {
    if (!data || !Array.isArray(data) || data.length === 0) return '.3f'
    
    const numericData = data.filter(val => typeof val === 'number' && !isNaN(val))
    if (numericData.length === 0) return '.3f'
    
    const max = Math.max(...numericData)
    const min = Math.min(...numericData)
    const range = Math.abs(max - min)
    
    // For very large numbers, use scientific notation
    if (max >= 1e6 || min <= -1e6 || (range > 0 && range < 1e-3)) {
      return '.2e'
    }
    
    // For large numbers, use fewer decimals
    if (range >= 1000) {
      return '.1f'
    }
    
    // For small ranges, use more decimals
    if (range < 1) {
      return '.4f'
    }
    
    // Default formatting
    return '.3f'
  }, [])

  // Detect if this is a multi-plot chart
  const isMultiPlot = useMemo(() => {
    if (!chartData?.data) return false
    
    // Check for subplot indicators
    const hasSubplots = chartData.data.some(trace => 
      trace.xaxis && trace.xaxis !== 'x' || 
      trace.yaxis && trace.yaxis !== 'y' ||
      (chartData.layout && chartData.layout.xaxis2) ||
      (chartData.layout && chartData.layout.yaxis2)
    )
    
    // Check for multiple traces that might indicate subplots
    const multipleTraces = chartData.data.length > 3
    
    // Check title for subplot indicators
    const titleIndicatesSubplots = title && (
      title.toLowerCase().includes('advanced analysis') ||
      title.toLowerCase().includes('multiple') ||
      title.toLowerCase().includes('comparison') ||
      title.toLowerCase().includes('vs ') ||
      title.toLowerCase().includes('subplot')
    )
    
    return hasSubplots || (multipleTraces && titleIndicatesSubplots)
  }, [chartData, title])

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
          if (chartRef.current) {
            observer.unobserve(chartRef.current)
          }
        }
      },
      {
        rootMargin: '0px 0px 200px 0px', // Start loading when 200px away from viewport
      }
    )

    if (chartRef.current) {
      observer.observe(chartRef.current)
    }

    return () => {
      if (chartRef.current) {
        // eslint-disable-next-line react-hooks/exhaustive-deps
        observer.unobserve(chartRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (isVisible && chartRef.current && chartData && chartData.data && chartData.layout) {
      setIsLoading(true)
      
      // Calculate appropriate height based on chart complexity
      const baseHeight = isMultiPlot ? 800 : 500
      const dataLength = chartData.data ? chartData.data.length : 1
      const calculatedHeight = isMultiPlot ? Math.max(baseHeight, dataLength * 150) : baseHeight
      
      // Adjust margins for multi-plot charts
      const margins = isMultiPlot 
        ? { l: 80, r: 60, t: 60, b: 80 }
        : { l: 60, r: 40, t: 40, b: 60 }
      
      // Determine smart formatting based on data and chart type
      const xData = chartData.data.flatMap(trace => trace.x || [])
      const yData = chartData.data.flatMap(trace => trace.y || [])
      
      // Check if this is a Q-Q plot for special handling
      const isQQPlot = title && (
        title.toLowerCase().includes('q-q') ||
        title.toLowerCase().includes('qq') ||
        title.toLowerCase().includes('quantile') ||
        title.toLowerCase().includes('normal') ||
        title.toLowerCase().includes('probability')
      ) || chartData.data.some(trace => 
        trace.name && (
          trace.name.toLowerCase().includes('q-q') ||
          trace.name.toLowerCase().includes('qq') ||
          trace.name.toLowerCase().includes('quantile') ||
          trace.name.toLowerCase().includes('theoretical') ||
          trace.name.toLowerCase().includes('sample')
        )
      )
      
      // For Q-Q plots, use more aggressive formatting
      const xFormat = isQQPlot ? '.3f' : getSmartTickFormat(xData)
      const yFormat = isQQPlot ? '.3f' : getSmartTickFormat(yData)
      
      const layout = {
        ...chartData.layout,
        title: '', // Title is handled outside
        autosize: true,
        height: calculatedHeight,
        margin: { ...margins, ...chartData.layout.margin },
        font: { 
          family: 'Inter, system-ui, -apple-system, sans-serif', 
          size: isMultiPlot ? 12 : 13, 
          color: '#374151',
          ...chartData.layout.font 
        },
        paper_bgcolor: 'rgba(255, 255, 255, 1)',
        plot_bgcolor: 'rgba(248, 250, 252, 1)',
        // Global number formatting settings
        separators: '.,',
        // Default axis formatting for primary axes
        xaxis: {
          ...chartData.layout.xaxis,
          gridcolor: 'rgba(148, 163, 184, 0.25)',
          linecolor: 'rgba(100, 116, 139, 0.4)',
          tickfont: { size: isMultiPlot ? 10 : 11, color: '#64748b' },
          showgrid: true,
          zeroline: false,
          titlefont: { size: isMultiPlot ? 11 : 12, color: '#374151' },
          tickformat: xFormat,
          hoverformat: xFormat,
          // Force custom tick formatting for better control
          tickmode: chartData.layout.xaxis?.tickmode || 'auto',
          exponentformat: 'e' // Use scientific notation when needed
        },
        yaxis: {
          ...chartData.layout.yaxis,
          gridcolor: 'rgba(148, 163, 184, 0.25)',
          linecolor: 'rgba(100, 116, 139, 0.4)',
          tickfont: { size: isMultiPlot ? 10 : 11, color: '#64748b' },
          showgrid: true,
          zeroline: false,
          titlefont: { size: isMultiPlot ? 11 : 12, color: '#374151' },
          tickformat: yFormat,
          hoverformat: yFormat,
          // Force custom tick formatting for better control
          tickmode: chartData.layout.yaxis?.tickmode || 'auto',
          exponentformat: 'e' // Use scientific notation when needed
        },
        // Enhanced grid styling for better visibility in complex charts with smart number formatting
        ...Object.fromEntries(
          Array.from({ length: 10 }, (_, i) => i + 1).flatMap(i => [
            [`xaxis${i === 1 ? '' : i}`, {
              ...chartData.layout[`xaxis${i === 1 ? '' : i}`],
              gridcolor: 'rgba(148, 163, 184, 0.25)',
              linecolor: 'rgba(100, 116, 139, 0.4)',
              tickfont: { size: isMultiPlot ? 10 : 11, color: '#64748b' },
              showgrid: true,
              zeroline: false,
              titlefont: { size: isMultiPlot ? 11 : 12, color: '#374151' },
              tickformat: xFormat, // Use smart formatting
              hoverformat: xFormat,
              tickmode: 'auto',
              exponentformat: 'e'
            }],
            [`yaxis${i === 1 ? '' : i}`, {
              ...chartData.layout[`yaxis${i === 1 ? '' : i}`],
              gridcolor: 'rgba(148, 163, 184, 0.25)',
              linecolor: 'rgba(100, 116, 139, 0.4)',
              tickfont: { size: isMultiPlot ? 10 : 11, color: '#64748b' },
              showgrid: true,
              zeroline: false,
              titlefont: { size: isMultiPlot ? 11 : 12, color: '#374151' },
              tickformat: yFormat, // Use smart formatting
              hoverformat: yFormat,
              tickmode: 'auto',
              exponentformat: 'e'
            }]
          ]).filter(([key]) => chartData.layout[key])
        )
      }
      
      // Set the chart container height dynamically
      if (chartRef.current) {
        chartRef.current.style.height = `${calculatedHeight}px`
      }
      
      // Format trace data for better number display
      const formattedData = chartData.data.map(trace => {
        const formattedTrace = { ...trace }
        
        // Check if this is a Q-Q plot for more aggressive formatting
        const isQQPlot = title && (
          title.toLowerCase().includes('q-q') ||
          title.toLowerCase().includes('qq') ||
          title.toLowerCase().includes('quantile')
        )
        
        const numberFormatter = isQQPlot ? aggressiveFormatNumber : formatNumber
        
        // Format y-axis data
        if (trace.y && Array.isArray(trace.y)) {
          formattedTrace.y = trace.y.map(val => 
            typeof val === 'number' ? numberFormatter(val) : val
          )
        }
        
        // Format x-axis data if numeric
        if (trace.x && Array.isArray(trace.x)) {
          formattedTrace.x = trace.x.map(val => 
            typeof val === 'number' ? numberFormatter(val) : val
          )
        }
        
        // Format z-axis data for heatmaps and 3D plots
        if (trace.z && Array.isArray(trace.z)) {
          formattedTrace.z = trace.z.map(row => 
            Array.isArray(row) ? 
              row.map(val => typeof val === 'number' ? numberFormatter(val) : val) : 
              (typeof row === 'number' ? numberFormatter(row) : row)
          )
        }
        
        // Format text values for scatter plots and Q-Q plots
        if (trace.text && Array.isArray(trace.text)) {
          formattedTrace.text = trace.text.map(val => 
            typeof val === 'string' && val.match(/^-?\d+\.?\d*$/) ? 
              numberFormatter(parseFloat(val)) : val
          )
        }
        
        // Format hover text specifically
        if (trace.hovertext && Array.isArray(trace.hovertext)) {
          formattedTrace.hovertext = trace.hovertext.map(val => 
            typeof val === 'string' && val.match(/^-?\d+\.?\d*$/) ? 
              numberFormatter(parseFloat(val)) : val
          )
        }
        
        // Enhance hover template for better number display
        if (trace.hovertemplate) {
          formattedTrace.hovertemplate = trace.hovertemplate
            .replace(/%{[xy]}(?![\w])/g, (match) => match + ':.3f')
            .replace(/%{z}(?![\w])/g, '%{z:.3f')
            .replace(/%{text}(?![\w])/g, '%{text}') // Keep text as formatted
        }
        
        // For Q-Q plots, format marker text and names
        if (trace.mode && trace.mode.includes('markers') && trace.name) {
          // Check if this looks like a Q-Q plot based on trace name or title
          const isQQPlotTrace = isQQPlot || (trace.name && (
            trace.name.toLowerCase().includes('q-q') ||
            trace.name.toLowerCase().includes('qq') ||
            trace.name.toLowerCase().includes('quantile')
          ))
          
          if (isQQPlotTrace && trace.marker && trace.marker.size) {
            // Format marker sizes if they're numeric
            if (Array.isArray(trace.marker.size)) {
              formattedTrace.marker = {
                ...trace.marker,
                size: trace.marker.size.map(val => 
                  typeof val === 'number' ? numberFormatter(val) : val
                )
              }
            }
          }
        }
        
        return formattedTrace
      })
      
      Plotly.newPlot(chartRef.current, formattedData, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
        displaylogo: false,
        autosizable: true,
        toImageButtonOptions: {
          format: 'png',
          filename: `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}`,
          height: calculatedHeight,
          width: 1200,
          scale: 1
        }
      }).then(() => {
        // Force update tick formatting after plot is rendered
        const update = {}
        
        // Apply formatting to all axes
        Object.keys(layout).forEach(key => {
          if (key.startsWith('xaxis') || key.startsWith('yaxis')) {
            const axisNum = key.replace(/[xy]axis/, '') || ''
            const axisKey = key.startsWith('x') ? `xaxis${axisNum}` : `yaxis${axisNum}`
            const format = key.startsWith('x') ? xFormat : yFormat
            
            update[`${axisKey}.tickformat`] = format
            update[`${axisKey}.hoverformat`] = format
          }
        })
        
        // Apply the formatting update
        if (Object.keys(update).length > 0) {
          Plotly.relayout(chartRef.current, update)
        }
        
        setIsLoading(false)
      }).catch(() => {
        setIsLoading(false)
      })
    }
  }, [isVisible, chartData, title, isMultiPlot, formatNumber, getSmartTickFormat, aggressiveFormatNumber])

  // Get color scheme based on category
  const getCategoryColors = (cat) => {
    const colorMap = {
      'numerical_charts': { bg: 'from-blue-50 to-indigo-50', border: 'border-blue-200', accent: 'text-blue-600' },
      'categorical_charts': { bg: 'from-green-50 to-emerald-50', border: 'border-green-200', accent: 'text-green-600' },
      'data_quality_charts': { bg: 'from-yellow-50 to-amber-50', border: 'border-yellow-200', accent: 'text-yellow-600' },
      'outlier_charts': { bg: 'from-red-50 to-rose-50', border: 'border-red-200', accent: 'text-red-600' },
      'relationship_charts': { bg: 'from-purple-50 to-violet-50', border: 'border-purple-200', accent: 'text-purple-600' },
      'default': { bg: 'from-gray-50 to-slate-50', border: 'border-gray-200', accent: 'text-gray-600' }
    }
    return colorMap[cat] || colorMap.default
  }

  const colors = getCategoryColors(category)

  const baseHeight = isMultiPlot ? 800 : 500
  const dataLength = chartData?.data ? chartData.data.length : 1
  const dynamicHeight = isMultiPlot ? Math.max(baseHeight, dataLength * 150) : baseHeight

  return (
    <div className={`w-full mb-8 bg-gradient-to-br ${colors.bg} rounded-2xl ${colors.border} border-2 shadow-lg hover:shadow-xl transition-all duration-300`}>
      {/* Chart Header */}
      <div className="p-6 border-b border-gray-200/60">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <div className={`w-2 h-2 rounded-full ${colors.accent.replace('text-', 'bg-')} animate-pulse`}></div>
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Chart {index + 1} of {totalCharts}
              </span>
              {isMultiPlot && (
                <span className="text-xs font-semibold text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                  Multi-Plot
                </span>
              )}
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-1 leading-tight">
              {title}
            </h3>
            {description && (
              <p className="text-sm text-gray-600 leading-relaxed">
                {description}
              </p>
            )}
            {isMultiPlot && (
              <p className="text-xs text-blue-600 mt-2 font-medium">
                📊 This chart contains multiple subplots for comprehensive analysis
              </p>
            )}
          </div>
          <div className={`p-2 rounded-lg ${colors.bg} ${colors.border} border`}>
            <BarChart3 className={`w-5 h-5 ${colors.accent}`} />
          </div>
        </div>
      </div>

      {/* Chart Content */}
      <div className="p-6">
        <div
          ref={chartRef}
          id={chartId}
          className="w-full bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
          style={{ 
            minHeight: `${dynamicHeight}px`, 
            height: `${dynamicHeight}px`,
            maxHeight: isMultiPlot ? '1200px' : '600px'
          }}
        >
          {!isVisible ? (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-8 h-8 animate-pulse" />
                </div>
                <p className="font-medium text-gray-500">Preparing Chart...</p>
                <p className="text-xs text-gray-400 mt-1">Scroll down to load</p>
                {isMultiPlot && (
                  <p className="text-xs text-blue-500 mt-2">⏳ Complex visualization loading...</p>
                )}
              </div>
            </div>
          ) : isLoading ? (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
                <p className="font-medium text-gray-600">
                  {isMultiPlot ? 'Rendering Complex Chart...' : 'Rendering Chart...'}
                </p>
                <p className="text-xs text-gray-400 mt-1">{title}</p>
                {isMultiPlot && (
                  <p className="text-xs text-blue-500 mt-2">⚡ Processing multiple subplots...</p>
                )}
              </div>
            </div>
          ) : null}
        </div>
      </div>

      {/* Chart Footer */}
      <div className="px-6 py-4 bg-gray-50/50 border-t border-gray-200/60">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500 flex items-center gap-2">
            <Activity className="w-4 h-4" />
            {category?.replace('_', ' ').toUpperCase() || 'ANALYSIS'}
            {isMultiPlot && (
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-medium ml-2">
                📊 Multi-Plot Chart
              </span>
            )}
          </span>
          <span className="flex items-center gap-1">
            <div className="w-1 h-1 bg-green-500 rounded-full"></div>
            {isMultiPlot ? 'Complex Chart Loaded' : 'Loaded'}
          </span>
        </div>
        
        {isMultiPlot && (
          <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-start gap-3">
              <div className="w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-white text-xs font-bold">💡</span>
              </div>
              <div className="text-xs text-blue-700 leading-relaxed">
                <p className="font-medium mb-1">Multi-Plot Chart Tips:</p>
                <ul className="space-y-1 text-blue-600">
                  <li>• This chart contains multiple related visualizations</li>
                  <li>• Numbers are automatically formatted for optimal readability</li>
                  <li>• Hover over data points for detailed information</li>
                  <li>• Chart automatically sized for optimal subplot visibility</li>
                  <li>• Each subplot can be analyzed independently</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
})
OptimizedChart.displayName = 'OptimizedChart'

export function AdvancedEDAPanel({ edaData, loading, dataset, onClose, onRefresh }) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [isMaximized, setIsMaximized] = useState(false)
  const [activeSection, setActiveSection] = useState('distributions')
  const [searchTerm, setSearchTerm] = useState('')
  const [chartFilter, setChartFilter] = useState('all')
  
  // Recovery and debugging state
  const [manualDatasetId, setManualDatasetId] = useState('15') // Default to dataset 15 for testing
  const [recoveredDataset, setRecoveredDataset] = useState(null)
  
  const panelRef = useRef(null)

  // Function to fetch dataset by ID when dataset prop is null
  const fetchDatasetById = async (datasetId) => {
    try {
      console.log('🚑 Fetching dataset by ID:', datasetId)
      const response = await fetch(`http://localhost:5000/api/data/datasets/${datasetId}`)
      
      if (response.ok) {
        const data = await response.json()
        console.log('✅ Dataset fetched successfully:', data.dataset)
        setRecoveredDataset(data.dataset)
        return data.dataset
      } else {
        console.error('❌ Failed to fetch dataset:', response.status)
        return null
      }
    } catch (error) {
      console.error('❌ Error fetching dataset:', error)
      return null
    }
  }

  // Get the active dataset (prop or recovered)
  const getActiveDataset = () => {
    if (dataset) return dataset
    if (recoveredDataset) return recoveredDataset
    return null
  }

  // Debug logging
  useEffect(() => {
    console.log('🔍 AdvancedEDAPanel props changed:', {
      hasEdaData: !!edaData,
      edaDataKeys: edaData ? Object.keys(edaData) : 'No data',
      hasDataset: !!dataset,
      datasetKeys: dataset ? Object.keys(dataset) : 'No dataset',
      datasetId: dataset?.id || 'No ID found',
      loading,
      timestamp: new Date().toISOString()
    })
    
    // Special warning if dataset becomes null
    if (!dataset) {
      console.warn('⚠️ Dataset is null/undefined in AdvancedEDAPanel')
    }
    
    if (edaData?.charts) {
      console.log('Available chart categories:', Object.keys(edaData.charts))
      Object.entries(edaData.charts).forEach(([category, charts]) => {
        console.log(`Category ${category}: ${charts?.length || 0} charts`)
      })
    }
  }, [edaData, dataset, loading])

  const formatNumber = (num) => {
    if (typeof num !== 'number') return 'N/A'
    return num.toLocaleString()
  }

  const chartCategories = edaData?.charts ? Object.keys(edaData.charts) : []
  const filteredCategories = chartFilter === 'all' 
    ? chartCategories 
    : chartCategories.filter(cat => {
        // Map frontend filter terms to backend category names
        const filterMap = {
          'numerical': 'numerical_charts',
          'categorical': 'categorical_charts', 
          'quality': 'data_quality_charts',
          'outlier': 'outlier_charts',
          'relationship': 'relationship_charts'
        }
        return cat === filterMap[chartFilter]
      })

  if (loading) {
    return (
      <div 
        className="fixed bottom-0 right-0 bg-white border-t shadow-lg transition-all duration-300 z-40 left-64"
        style={{
          height: isExpanded ? '90vh' : '64px'
        }}
      >
        <div className="flex items-center justify-between p-4 border-b bg-gray-50">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
              <RefreshCw className="w-4 h-4 text-blue-600 animate-spin" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Generating EDA Report</h3>
              <p className="text-sm text-gray-500">Analyzing dataset and creating visualizations...</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </Button>
        </div>
        {isExpanded && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <RefreshCw className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
              <p className="text-gray-600">Processing your data...</p>
            </div>
          </div>
        )}
      </div>
    )
  }

  if (!edaData) {
    return null
  }

  return (
    <div 
      ref={panelRef}
      className={`fixed bottom-0 right-0 bg-white border-t-2 border-blue-200 shadow-2xl transition-all duration-500 ease-in-out z-30 ${
        isMaximized ? 'top-0 left-64' : isExpanded ? 'left-64' : 'left-64'
      }`}
      style={{
        backdropFilter: 'blur(10px)',
        boxShadow: '0 -10px 40px rgba(0, 0, 0, 0.15)',
        borderTopLeftRadius: '20px',
        borderTopRightRadius: '20px',
        height: isMaximized ? '100vh' : isExpanded ? '90vh' : '64px'
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 lg:p-6 border-b-2 border-gray-100 bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50 rounded-t-2xl">
        <div className="flex items-center space-x-4 min-w-0 flex-1">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg flex-shrink-0 ring-4 ring-blue-100">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-3 mb-1">
              <h3 className="text-xl lg:text-2xl font-bold text-gray-800 truncate">
                EDA Report
              </h3>
              <Badge variant="secondary" className="bg-gradient-to-r from-green-100 to-green-200 text-green-700 px-3 py-1 text-sm font-semibold ring-1 ring-green-300">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                Live Analysis
              </Badge>
            </div>
            <div className="flex items-center gap-3 text-sm text-gray-600">
              <div className="flex items-center gap-2 bg-white px-3 py-1 rounded-lg shadow-sm border">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="font-medium truncate max-w-32 lg:max-w-none">
                  {dataset?.name || `Dataset ${dataset?.id || dataset?.dataset_id || 'Unknown'}`}
                </span>
              </div>
              <div className="flex items-center gap-2 bg-white px-3 py-1 rounded-lg shadow-sm border">
                <Layers className="w-3 h-3 text-gray-500" />
                <span className="font-medium">{formatNumber(edaData.overview?.shape?.rows || 0)} rows</span>
              </div>
              <div className="hidden sm:flex items-center gap-2 bg-white px-3 py-1 rounded-lg shadow-sm border">
                <BarChart3 className="w-3 h-3 text-gray-500" />
                <span className="font-medium">{edaData.overview?.shape?.columns || 0} columns</span>
              </div>
              {/* Debug badge - only show if dataset has no clear ID */}
              {dataset && !dataset.id && !dataset.dataset_id && (
                <div className="flex items-center gap-2 bg-red-100 px-3 py-1 rounded-lg shadow-sm border border-red-200">
                  <AlertTriangle className="w-3 h-3 text-red-500" />
                  <span className="font-medium text-red-700 text-xs">No ID</span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3 flex-shrink-0">
          {/* Search and Filter - Hidden on small screens, shown on large */}
          <div className="hidden xl:flex items-center space-x-3 bg-white rounded-2xl p-3 shadow-lg border border-gray-200">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <Input
                placeholder="Search insights..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 w-48 h-10 border-0 bg-gray-50 focus:bg-white transition-colors rounded-xl"
              />
            </div>
            <div className="w-px h-6 bg-gray-200"></div>
            <Select value={chartFilter} onValueChange={setChartFilter}>
              <SelectTrigger className="w-40 h-10 border-0 bg-gray-50 transition-colors rounded-xl">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Charts</SelectItem>
                <SelectItem value="numerical">Numerical</SelectItem>
                <SelectItem value="categorical">Categorical</SelectItem>
                <SelectItem value="quality">Data Quality</SelectItem>
                <SelectItem value="outlier">Outliers</SelectItem>
                <SelectItem value="relationship">Relationships</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Mobile Filter - Shown on small screens */}
          <div className="flex xl:hidden items-center">
            <Select value={chartFilter} onValueChange={setChartFilter}>
              <SelectTrigger className="w-36 h-10 border border-gray-300 bg-white transition-colors rounded-xl shadow-md">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Charts</SelectItem>
                <SelectItem value="numerical">Numerical</SelectItem>
                <SelectItem value="categorical">Categorical</SelectItem>
                <SelectItem value="quality">Data Quality</SelectItem>
                <SelectItem value="outlier">Outliers</SelectItem>
                <SelectItem value="relationship">Relationships</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center space-x-2 bg-white rounded-2xl p-2 shadow-lg border border-gray-200">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={onRefresh}
              className="h-10 w-10 transition-colors rounded-xl"
              title="Refresh Data"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMaximized(!isMaximized)}
              className="h-10 w-10 transition-colors rounded-xl"
              title={isMaximized ? "Minimize" : "Maximize"}
            >
              {isMaximized ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="h-10 w-10 transition-colors rounded-xl"
              title={isExpanded ? "Collapse" : "Expand"}
            >
              {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
            </Button>

            <div className="w-px h-6 bg-gray-200"></div>

            <Button 
              variant="ghost" 
              size="sm" 
              onClick={onClose}
              className="h-10 w-10 transition-colors rounded-xl"
              title="Close Panel"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="flex h-full bg-gradient-to-br from-gray-50 to-white">
          {/* Sidebar Navigation - Enhanced Design */}
          <div className="w-56 lg:w-60 xl:w-64 border-r-2 border-gray-200 bg-gradient-to-b from-white via-gray-50 to-gray-100 p-3 lg:p-4 overflow-y-auto shadow-lg">
            <div className="space-y-3">
              <div className="mb-6">
                <h4 className="text-sm font-bold text-gray-700 uppercase tracking-wider mb-4 flex items-center">
                  <div className="w-2 h-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mr-2 animate-pulse"></div>
                  Analysis Sections
                </h4>
              </div>
              {[
                { id: 'overview', label: 'Overview', icon: Eye, description: 'Dataset summary & metrics', color: 'blue', gradient: 'from-blue-500 to-blue-600', bgGradient: 'from-blue-50 to-blue-100' },
                { id: 'quality', label: 'Data Quality', icon: CheckCircle, description: 'Quality assessment', color: 'green', gradient: 'from-green-500 to-green-600', bgGradient: 'from-green-50 to-green-100' },
                { id: 'distributions', label: 'Distributions', icon: BarChart3, description: 'Variable patterns', color: 'purple', gradient: 'from-purple-500 to-purple-600', bgGradient: 'from-purple-50 to-purple-100' },
                { id: 'relationships', label: 'Relationships', icon: Activity, description: 'Correlations', color: 'orange', gradient: 'from-orange-500 to-orange-600', bgGradient: 'from-orange-50 to-orange-100' },
                { id: 'outliers', label: 'Outliers', icon: AlertTriangle, description: 'Anomaly detection', color: 'red', gradient: 'from-red-500 to-red-600', bgGradient: 'from-red-50 to-red-100' },
                { id: 'insights', label: 'AI Insights', icon: Zap, description: 'AI recommendations', color: 'yellow', gradient: 'from-yellow-500 to-yellow-600', bgGradient: 'from-yellow-50 to-yellow-100' },
                { id: 'debug', label: 'Debug', icon: Info, description: 'Technical details', color: 'gray', gradient: 'from-gray-500 to-gray-600', bgGradient: 'from-gray-50 to-gray-100' }
              ].map((section) => {
                const isActive = activeSection === section.id
                
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full text-left p-2 lg:p-3 rounded-xl transition-all duration-300 border-2 group transform shadow-md ${
                      isActive 
                        ? `bg-gradient-to-r ${section.gradient} text-white border-white shadow-xl scale-[1.01] ring-2 ring-white ring-opacity-30` 
                        : `border-gray-200 bg-gradient-to-br ${section.bgGradient} text-gray-700 shadow-lg`
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className={`p-2 lg:p-2.5 rounded-xl transition-all duration-300 ${
                        isActive 
                          ? 'bg-white bg-opacity-20 backdrop-blur-sm shadow-md' 
                          : `bg-gradient-to-br ${section.gradient} bg-opacity-15 shadow-sm`
                      }`}>
                        <section.icon className={`w-4 h-4 lg:w-5 lg:h-5 transition-all duration-300 ${
                          isActive ? 'text-white drop-shadow-sm' : `text-${section.color}-600`
                        }`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className={`font-bold text-sm lg:text-base mb-1 transition-colors ${
                          isActive ? 'text-white drop-shadow-sm' : 'text-gray-800'
                        }`}>
                          {section.label}
                        </div>
                        <div className={`text-xs lg:text-sm leading-relaxed transition-colors ${
                          isActive ? 'text-white text-opacity-90 drop-shadow-sm' : 'text-gray-600'
                        }`}>
                          {section.description}
                        </div>
                      </div>
                      {isActive && (
                        <div className="flex items-center">
                          <div className="w-2 h-2 bg-white rounded-full animate-pulse shadow-sm"></div>
                        </div>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1 overflow-hidden bg-gradient-to-br from-gray-50 via-white to-blue-50/20">
            <ScrollArea className="h-full">
              <div className="p-4 sm:p-6 lg:p-8 xl:p-12 pb-24 lg:pb-32 xl:pb-40">
                {activeSection === 'overview' && (
                  <OverviewSection edaData={edaData} />
                )}
                {activeSection === 'quality' && (
                  <QualitySection edaData={edaData} />
                )}
                {activeSection === 'distributions' && (
                  <DistributionsSection edaData={edaData} filteredCategories={filteredCategories} />
                )}
                {activeSection === 'relationships' && (
                  <RelationshipsSection edaData={edaData} />
                )}
                {activeSection === 'outliers' && (
                  <OutliersSection edaData={edaData} />
                )}
                {activeSection === 'insights' && (
                  <InsightsSection 
                    edaData={edaData} 
                    dataset={dataset} 
                    getActiveDataset={getActiveDataset}
                    fetchDatasetById={fetchDatasetById}
                    manualDatasetId={manualDatasetId}
                    setManualDatasetId={setManualDatasetId}
                  />
                )}
                {activeSection === 'debug' && (
                  <DebugSection edaData={edaData} dataset={dataset} />
                )}
              </div>
            </ScrollArea>
          </div>
        </div>
      )}
    </div>
  )
}

// Overview Section Component
function OverviewSection({ edaData }) {
  const overview = edaData.overview || {}
  
  return (
    <div className="space-y-10 lg:space-y-12">
      <div className="text-center lg:text-left">
        <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4 bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
          Dataset Overview
        </h2>
        <p className="text-xl lg:text-2xl text-gray-600 max-w-3xl leading-relaxed">
          Comprehensive analysis and key metrics of your dataset structure
        </p>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 transition-all duration-300 rounded-2xl border-2">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-bold text-blue-700 uppercase tracking-wide">Total Rows</p>
                <p className="text-2xl font-bold text-blue-900 mt-1">
                  {(overview.shape?.rows || 0).toLocaleString()}
                </p>
                <p className="text-xs text-blue-600 mt-1">Data points</p>
              </div>
              <div className="p-2 bg-blue-200 rounded-lg shadow-md">
                <Layers className="w-6 h-6 text-blue-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200 transition-all duration-300 rounded-2xl border-2">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-bold text-green-700 uppercase tracking-wide">Variables</p>
                <p className="text-2xl font-bold text-green-900 mt-1">
                  {overview.shape?.columns || 0}
                </p>
                <p className="text-xs text-green-600 mt-1">Features</p>
              </div>
              <div className="p-2 bg-green-200 rounded-lg shadow-md">
                <BarChart3 className="w-6 h-6 text-green-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 transition-all duration-300 rounded-2xl border-2">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-bold text-purple-700 uppercase tracking-wide">Data Density</p>
                <p className="text-2xl font-bold text-purple-900 mt-1">
                  {(overview.data_density || 0).toFixed(1)}%
                </p>
                <p className="text-xs text-purple-600 mt-1">Complete</p>
              </div>
              <div className="p-2 bg-purple-200 rounded-lg shadow-md">
                <Target className="w-6 h-6 text-purple-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200 transition-all duration-300 rounded-2xl border-2">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-bold text-orange-700 uppercase tracking-wide">Memory Usage</p>
                <p className="text-2xl font-bold text-orange-900 mt-1">
                  {(overview.memory_usage?.total_mb || 0).toFixed(2)}
                </p>
                <p className="text-xs text-orange-600 mt-1">MB</p>
              </div>
              <div className="p-2 bg-orange-200 rounded-lg shadow-md">
                <Zap className="w-6 h-6 text-orange-700" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Column Details Table */}
      {overview.column_details && (
        <Card className="shadow-2xl border-0 rounded-3xl">
          <CardHeader className="bg-gradient-to-r from-gray-100 via-white to-gray-100 border-b-2 border-gray-200 p-8 rounded-t-3xl">
            <CardTitle className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-blue-100 rounded-2xl">
                  <Info className="w-7 h-7 lg:w-8 lg:h-8 text-blue-600" />
                </div>
                <span className="text-2xl lg:text-3xl font-bold text-gray-800">Column Information</span>
              </div>
              <Badge variant="secondary" className="self-start sm:self-auto px-4 py-2 text-lg font-semibold rounded-xl">
                {Object.keys(overview.column_details).length} columns
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <div className="max-h-[600px] overflow-y-auto">
                <table className="w-full min-w-[700px]">
                  <thead className="bg-gradient-to-r from-gray-200 via-gray-100 to-gray-200 border-b-2 border-gray-300 sticky top-0 z-10">
                    <tr>
                      <th className="text-left p-4 lg:p-6 font-bold text-gray-800 text-base lg:text-lg">
                        Column Name
                      </th>
                      <th className="text-left p-4 lg:p-6 font-bold text-gray-800 text-base lg:text-lg">
                        Data Type
                      </th>
                      <th className="text-left p-4 lg:p-6 font-bold text-gray-800 text-base lg:text-lg">
                        Non-Null Count
                      </th>
                      <th className="text-left p-4 lg:p-6 font-bold text-gray-800 text-base lg:text-lg">
                        Unique Values
                      </th>
                      <th className="text-left p-4 lg:p-6 font-bold text-gray-800 text-base lg:text-lg">
                        Memory (MB)
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(overview.column_details).map(([col, details], index) => (
                      <tr 
                        key={col} 
                        className={`border-b-2 border-gray-200 transition-colors duration-200 ${
                          index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                        }`}
                      >
                        <td className="p-4 lg:p-6">
                          <div className="font-bold text-gray-900 text-base lg:text-lg break-all">
                            {col}
                          </div>
                        </td>
                        <td className="p-4 lg:p-6">
                          <Badge 
                            variant="secondary" 
                            className="bg-blue-100 text-blue-800 font-mono text-sm px-3 py-2 lg:px-4 lg:py-2 rounded-xl"
                          >
                            {details.dtype}
                          </Badge>
                        </td>
                        <td className="p-4 lg:p-6 text-gray-700 font-semibold text-base lg:text-lg">
                          {details.non_null_count?.toLocaleString() || 'N/A'}
                        </td>
                        <td className="p-4 lg:p-6 text-gray-700 font-semibold text-base lg:text-lg">
                          {details.unique_count?.toLocaleString() || 'N/A'}
                        </td>
                        <td className="p-4 lg:p-6 text-gray-700 font-semibold text-base lg:text-lg">
                          {(details.memory_mb || 0).toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            
            {/* Mobile-friendly summary for small screens */}
            <div className="block sm:hidden border-t bg-gray-100 p-6 rounded-b-3xl">
              <p className="text-sm text-gray-600 text-center font-medium">
                Scroll horizontally to view all columns • {Object.keys(overview.column_details).length} total columns
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

// Quality Section Component  
function QualitySection({ edaData }) {
  const overview = edaData?.overview || {}
  
  // Calculate quality metrics
  const totalRows = overview.shape?.rows || 0
  const dataDensity = overview.data_density || 0
  
  // Extract missing data info from column details
  const missingDataInfo = overview.column_details ? 
    Object.entries(overview.column_details).map(([column, details]) => ({
      column,
      missingCount: totalRows - (details.non_null_count || 0),
      missingPercentage: totalRows > 0 ? ((totalRows - (details.non_null_count || 0)) / totalRows * 100) : 0,
      dataType: details.dtype,
      uniqueValues: details.unique_count || 0
    })).sort((a, b) => b.missingPercentage - a.missingPercentage) : []

  const averageMissingRate = missingDataInfo.length > 0 ? 
    missingDataInfo.reduce((sum, item) => sum + item.missingPercentage, 0) / missingDataInfo.length : 0

  return (
    <div className="space-y-8">
      <div className="text-center lg:text-left">
        <h2 className="text-3xl font-bold text-gray-900 mb-3 bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
          Data Quality Assessment
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl">
          Comprehensive analysis of data quality metrics and patterns
        </p>
      </div>

      {/* Quality Metrics Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-gradient-to-br from-green-50 to-emerald-100 border-green-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-green-700 uppercase tracking-wide">Data Completeness</p>
                <p className="text-3xl font-bold text-green-900 mt-2">{dataDensity.toFixed(1)}%</p>
                <p className="text-xs text-green-600 mt-1">Overall quality</p>
              </div>
              <div className="p-3 bg-green-200 rounded-xl">
                <CheckCircle className="w-8 h-8 text-green-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-blue-700 uppercase tracking-wide">Missing Rate</p>
                <p className="text-3xl font-bold text-blue-900 mt-2">{averageMissingRate.toFixed(1)}%</p>
                <p className="text-xs text-blue-600 mt-1">Average per column</p>
              </div>
              <div className="p-3 bg-blue-200 rounded-xl">
                <AlertTriangle className="w-8 h-8 text-blue-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-purple-700 uppercase tracking-wide">Complete Columns</p>
                <p className="text-3xl font-bold text-purple-900 mt-2">
                  {missingDataInfo.filter(item => item.missingPercentage === 0).length}
                </p>
                <p className="text-xs text-purple-600 mt-1">No missing data</p>
              </div>
              <div className="p-3 bg-purple-200 rounded-xl">
                <Target className="w-8 h-8 text-purple-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-orange-700 uppercase tracking-wide">Issues Found</p>
                <p className="text-3xl font-bold text-orange-900 mt-2">
                  {missingDataInfo.filter(item => item.missingPercentage > 10).length}
                </p>
                <p className="text-xs text-orange-600 mt-1">Columns &gt;10% missing</p>
              </div>
              <div className="p-3 bg-orange-200 rounded-xl">
                <AlertTriangle className="w-8 h-8 text-orange-700" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Missing Data Details */}
        <Card className="shadow-lg border-0">
          <CardHeader className="bg-gradient-to-r from-gray-50 to-gray-100 border-b">
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="w-6 h-6 text-orange-600" />
              <span>Missing Data Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="space-y-4">
              {missingDataInfo.slice(0, 10).map((item) => (
                <div key={item.column} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg transition-colors">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-800 truncate">{item.column}</span>
                      <Badge variant="secondary" className="text-xs">
                        {item.dataType}
                      </Badge>
                    </div>
                    <div className="flex items-center space-x-2 mt-1">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            item.missingPercentage > 20 ? 'bg-red-500' :
                            item.missingPercentage > 10 ? 'bg-orange-500' :
                            item.missingPercentage > 0 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${Math.max(item.missingPercentage, 2)}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-600 font-medium min-w-0">
                        {item.missingPercentage.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="text-right text-sm text-gray-600 ml-4">
                    <div>{item.missingCount.toLocaleString()} missing</div>
                    <div className="text-xs">{item.uniqueValues.toLocaleString()} unique</div>
                  </div>
                </div>
              ))}
              {missingDataInfo.length === 0 && (
                <div className="text-center py-8">
                  <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-2" />
                  <p className="text-gray-600">No missing data information available</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
        
        {/* Data Quality Chart */}
        <div className="space-y-6">
          {(edaData?.charts?.data_quality_charts || []).length > 0 ? (
            (edaData.charts.data_quality_charts).map((chart, index) => (
              <OptimizedChart
                key={`quality-${index}`}
                chartId={`chart-quality_charts-${index}`}
                chartData={chart.chart}
                title={chart.title}
                description="Visual representation of data quality metrics and completeness analysis"
                category="data_quality_charts"
                index={index}
                totalCharts={edaData.charts.data_quality_charts.length}
              />
            ))
          ) : (
            <div className="bg-white rounded-2xl border-2 border-gray-200 p-12 text-center shadow-lg">
              <div className="max-w-md mx-auto">
                <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <BarChart3 className="w-10 h-10 text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Data Quality Visualization</h3>
                <p className="text-gray-500 leading-relaxed mb-4">
                  Quality metrics visualization will appear here when data quality charts are available
                </p>
                <Button 
                  onClick={() => {
                    console.log('Chart data not available for rendering.')
                  }}
                  variant="outline" 
                  size="sm"
                  disabled
                  className="opacity-50"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  No Chart Data
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Distributions Section Component
function DistributionsSection({ edaData, filteredCategories }) {
  // Map backend category names to friendly display names
  const getCategoryDisplayName = (category) => {
    const nameMap = {
      'numerical_charts': 'Numerical Variables',
      'categorical_charts': 'Categorical Variables', 
      'data_quality_charts': 'Data Quality',
      'outlier_charts': 'Outlier Detection',
      'relationship_charts': 'Relationships'
    }
    return nameMap[category] || category.replace(/_/g, ' ')
  }

  const getCategoryDescription = (category) => {
    const descMap = {
      'numerical_charts': 'Distribution patterns and statistical properties of numerical variables',
      'categorical_charts': 'Frequency distributions and patterns in categorical data', 
      'data_quality_charts': 'Assessment of data completeness, consistency, and quality metrics',
      'outlier_charts': 'Detection and visualization of anomalous data points',
      'relationship_charts': 'Correlations and associations between different variables'
    }
    return descMap[category] || 'Statistical analysis and visualization'
  }

  return (
    <div className="space-y-10">
      <div className="text-center lg:text-left">
        <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4 bg-gradient-to-r from-purple-600 via-pink-600 to-indigo-600 bg-clip-text text-transparent">
          Variable Distributions
        </h2>
        <p className="text-xl lg:text-2xl text-gray-600 max-w-4xl leading-relaxed">
          Statistical distributions and patterns in your data variables
        </p>
      </div>

      {edaData.charts && filteredCategories.map((category) => {
        const charts = edaData.charts[category]
        if (!Array.isArray(charts) || charts.length === 0) return null

        return (
          <div key={category} className="space-y-6">
            {/* Category Header */}
            <div className="bg-white rounded-2xl border-2 border-gray-200 p-6 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">
                    {getCategoryDisplayName(category)}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {getCategoryDescription(category)}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant="secondary" className="bg-gradient-to-r from-purple-100 to-pink-100 text-purple-800 px-4 py-2 text-base font-semibold rounded-xl">
                    {charts.length} chart{charts.length !== 1 ? 's' : ''}
                  </Badge>
                </div>
              </div>
            </div>

            {/* Charts - One per row */}
            <div className="space-y-6">
              {charts.map((chart, index) => (
                <OptimizedChart
                  key={`${category}-${index}`}
                  chartId={`chart-${category}-${index}`}
                  chartData={chart.chart}
                  title={chart.title || `${getCategoryDisplayName(category)} - Chart ${index + 1}`}
                  description={chart.description}
                  category={category}
                  index={index}
                  totalCharts={charts.length}
                />
              ))}
            </div>
          </div>
        )
      })}
      
      {filteredCategories.length === 0 && (
        <div className="bg-white rounded-2xl border-2 border-gray-200 p-12 text-center shadow-lg">
          <div className="max-w-md mx-auto">
            <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <BarChart3 className="w-10 h-10 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">No Charts Available</h3>
            <p className="text-gray-500 leading-relaxed">
              No charts match the current filter selection. Try changing the filter or selecting "All Charts".
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

// Relationships Section Component
function RelationshipsSection({ edaData }) {
  const overview = edaData?.overview || {}
  const relationshipCharts = edaData?.charts?.relationship_charts || []
  
  // Get numerical columns for correlation analysis
  const numericalColumns = overview.column_details ? 
    Object.entries(overview.column_details)
      .filter(([, details]) => ['int64', 'float64', 'int32', 'float32'].includes(details.dtype))
      .map(([column]) => column) : []

  return (
    <div className="space-y-8">
      <div className="text-center lg:text-left">
        <h2 className="text-3xl font-bold text-gray-900 mb-3 bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
          Feature Relationships
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl">
          Correlations and associations between variables in your dataset
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-gradient-to-br from-orange-50 to-red-100 border-orange-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-orange-700 uppercase tracking-wide">Numerical Features</p>
                <p className="text-3xl font-bold text-orange-900 mt-2">{numericalColumns.length}</p>
                <p className="text-xs text-orange-600 mt-1">Available for correlation</p>
              </div>
              <div className="p-3 bg-orange-200 rounded-xl">
                <Activity className="w-8 h-8 text-orange-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-purple-700 uppercase tracking-wide">Possible Pairs</p>
                <p className="text-3xl font-bold text-purple-900 mt-2">
                  {numericalColumns.length > 1 ? (numericalColumns.length * (numericalColumns.length - 1) / 2) : 0}
                </p>
                <p className="text-xs text-purple-600 mt-1">Correlation pairs</p>
              </div>
              <div className="p-3 bg-purple-200 rounded-xl">
                <Target className="w-8 h-8 text-purple-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-blue-700 uppercase tracking-wide">Total Variables</p>
                <p className="text-3xl font-bold text-blue-900 mt-2">{overview.shape?.columns || 0}</p>
                <p className="text-xs text-blue-600 mt-1">All data types</p>
              </div>
              <div className="p-3 bg-blue-200 rounded-xl">
                <BarChart3 className="w-8 h-8 text-blue-700" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-8">
        {/* Correlation Matrix */}
        <div>
          <div className="bg-white rounded-2xl border-2 border-gray-200 p-6 shadow-lg mb-6">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-orange-100 rounded-xl">
                <Activity className="w-6 h-6 text-orange-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-900">Correlation Analysis</h3>
                <p className="text-gray-600">Statistical relationships and correlations between numerical variables</p>
              </div>
            </div>
          </div>

          {relationshipCharts.length > 0 ? (
            <div className="space-y-6">
              {relationshipCharts.map((chart, index) => (
                <OptimizedChart
                  key={`relationship-${index}`}
                  chartId={`chart-relationship_charts-${index}`}
                  chartData={chart.chart}
                  title={chart.title}
                  description="Correlation matrix showing statistical relationships between numerical variables"
                  category="relationship_charts"
                  index={index}
                  totalCharts={relationshipCharts.length}
                />
              ))}
            </div>
          ) : (
            <div className="bg-white rounded-2xl border-2 border-gray-200 p-12 text-center shadow-lg">
              <div className="max-w-md mx-auto">
                <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Activity className="w-10 h-10 text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Correlation Matrix</h3>
                <p className="text-gray-500 leading-relaxed mb-4">
                  {numericalColumns.length < 2 ? 
                    'Need at least 2 numerical columns for correlation analysis' :
                    'Correlation heatmap will be displayed here'
                  }
                </p>
                {numericalColumns.length > 0 && (
                  <div className="max-w-md mx-auto">
                    <p className="text-sm text-gray-500 mb-3">Available numerical columns:</p>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {numericalColumns.slice(0, 10).map(col => (
                        <Badge key={col} variant="secondary" className="text-xs">
                          {col}
                        </Badge>
                      ))}
                      {numericalColumns.length > 10 && (
                        <Badge variant="secondary" className="text-xs">
                          +{numericalColumns.length - 10} more
                        </Badge>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Outliers Section Component
function OutliersSection({ edaData }) {
  const overview = edaData?.overview || {}
  const outlierCharts = edaData?.charts?.outlier_charts || []
  
  // Get numerical columns for outlier analysis
  const numericalColumns = overview.column_details ? 
    Object.entries(overview.column_details)
      .filter(([, details]) => ['int64', 'float64', 'int32', 'float32'].includes(details.dtype))
      .map(([column]) => column) : []

  return (
    <div className="space-y-8">
      <div className="text-center lg:text-left">
        <h2 className="text-3xl font-bold text-gray-900 mb-3 bg-gradient-to-r from-red-600 to-pink-600 bg-clip-text text-transparent">
          Outlier Detection
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl">
          Identification and analysis of anomalous data points in your dataset
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-gradient-to-br from-red-50 to-pink-100 border-red-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-red-700 uppercase tracking-wide">Analysis Methods</p>
                <p className="text-3xl font-bold text-red-900 mt-2">{outlierCharts.length}</p>
                <p className="text-xs text-red-600 mt-1">Detection techniques</p>
              </div>
              <div className="p-3 bg-red-200 rounded-xl">
                <AlertTriangle className="w-8 h-8 text-red-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-orange-700 uppercase tracking-wide">Numerical Vars</p>
                <p className="text-3xl font-bold text-orange-900 mt-2">{numericalColumns.length}</p>
                <p className="text-xs text-orange-600 mt-1">Available for analysis</p>
              </div>
              <div className="p-3 bg-orange-200 rounded-xl">
                <BarChart3 className="w-8 h-8 text-orange-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-yellow-50 to-yellow-100 border-yellow-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-yellow-700 uppercase tracking-wide">IQR Method</p>
                <p className="text-3xl font-bold text-yellow-900 mt-2">✓</p>
                <p className="text-xs text-yellow-600 mt-1">Statistical outliers</p>
              </div>
              <div className="p-3 bg-yellow-200 rounded-xl">
                <Target className="w-8 h-8 text-yellow-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-purple-700 uppercase tracking-wide">Z-Score</p>
                <p className="text-3xl font-bold text-purple-900 mt-2">✓</p>
                <p className="text-xs text-purple-600 mt-1">Standard deviation</p>
              </div>
              <div className="p-3 bg-purple-200 rounded-xl">
                <TrendingUp className="w-8 h-8 text-purple-700" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Outlier Analysis Charts */}
      <div className="space-y-8">
        <div className="bg-white rounded-2xl border-2 border-gray-200 p-6 shadow-lg">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-red-100 rounded-xl">
              <AlertTriangle className="w-6 h-6 text-red-600" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-900">Outlier Analysis</h3>
              <p className="text-gray-600">Detection and visualization of anomalous data points using statistical methods</p>
            </div>
          </div>
        </div>

        {outlierCharts.length > 0 ? (
          <div className="space-y-6">
            {outlierCharts.map((chart, index) => (
              <OptimizedChart
                key={`outlier-${index}`}
                chartId={`chart-outlier_charts-${index}`}
                chartData={chart.chart}
                title={chart.title || `Outlier Analysis ${index + 1}`}
                description="Statistical outlier detection using IQR and Z-score methods"
                category="outlier_charts"
                index={index}
                totalCharts={outlierCharts.length}
              />
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-2xl border-2 border-gray-200 p-12 text-center shadow-lg">
            <div className="max-w-md mx-auto">
              <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <AlertTriangle className="w-10 h-10 text-gray-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Outlier Detection</h3>
              <p className="text-gray-500 leading-relaxed mb-4">
                {numericalColumns.length === 0 ? 
                  'No numerical columns available for outlier detection' :
                  'Outlier analysis charts will be displayed here'
                }
              </p>
              {numericalColumns.length > 0 && (
                <div className="max-w-md mx-auto">
                  <p className="text-sm text-gray-500 mb-3">Outlier detection methods:</p>
                  <div className="grid grid-cols-1 gap-2 text-sm">
                    <div className="flex items-center justify-center space-x-2 p-3 bg-blue-50 rounded-lg">
                      <Target className="w-4 h-4 text-blue-600" />
                      <span>Interquartile Range (IQR) Method</span>
                    </div>
                    <div className="flex items-center justify-center space-x-2 p-3 bg-green-50 rounded-lg">
                      <TrendingUp className="w-4 h-4 text-green-600" />
                      <span>Z-Score Analysis (±3 standard deviations)</span>
                    </div>
                    <div className="flex items-center justify-center space-x-2 p-3 bg-purple-50 rounded-lg">
                      <BarChart3 className="w-4 h-4 text-purple-600" />
                      <span>Box Plot Visualization</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Insights Section Component
function InsightsSection({ edaData, dataset, getActiveDataset, fetchDatasetById, manualDatasetId, setManualDatasetId }) {
  const [aiInsights, setAiInsights] = useState(null)
  const [isLoadingAI, setIsLoadingAI] = useState(false)
  const [aiError, setAiError] = useState(null)
  const [showFallback, setShowFallback] = useState(false)
  const [advancedAnalysis, setAdvancedAnalysis] = useState(null)
  const [isLoadingAdvanced, setIsLoadingAdvanced] = useState(false)
  const [advancedError, setAdvancedError] = useState(null)

  // Track dataset history for debugging
  const datasetHistory = useRef([])
  const previousDataset = useRef(null)

  // Advanced dataset tracking
  useEffect(() => {
    const timestamp = new Date().toISOString()
    const datasetSnapshot = {
      timestamp,
      hasDataset: !!dataset,
      datasetId: dataset?.id,
      datasetName: dataset?.name,
      datasetKeys: dataset ? Object.keys(dataset) : [],
      previousHadDataset: !!previousDataset.current,
      previousDatasetId: previousDataset.current?.id
    }
    
    datasetHistory.current.push(datasetSnapshot)
    
    // Keep only last 10 entries
    if (datasetHistory.current.length > 10) {
      datasetHistory.current = datasetHistory.current.slice(-10)
    }
    
    // Check for state transitions
    if (previousDataset.current && !dataset) {
      console.error('🚨 DATASET LOST IN INSIGHTS SECTION! Previous dataset was:', previousDataset.current)
      console.error('🚨 Dataset history:', datasetHistory.current)
      console.trace('Stack trace when dataset was lost:')
    } else if (!previousDataset.current && dataset) {
      console.log('✅ Dataset gained in insights section:', dataset)
    }
    
    previousDataset.current = dataset
  }, [dataset])

  const generateAIInsights = async () => {
    let activeDataset = getActiveDataset()
    
    console.log('🔍 AI Insights Debug - Dataset object:', activeDataset)
    console.log('🔍 Dataset full object structure:', JSON.stringify(activeDataset, null, 2))
    
    // If no dataset available, try to recover using manual ID
    if (!activeDataset) {
      console.warn('⚠️ No dataset available, attempting recovery with ID:', manualDatasetId)
      if (manualDatasetId) {
        activeDataset = await fetchDatasetById(manualDatasetId)
      }
    }
    
    // Final check if dataset exists
    if (!activeDataset) {
      console.error('❌ Dataset is null/undefined for AI insights')
      setAiError('No dataset available. Please select a dataset first and ensure it loads properly.')
      setShowFallback(true)
      return
    }
    
    // Use the primary dataset ID field
    const datasetId = activeDataset.id
    
    console.log('🔍 ID validation check:', {
      datasetId,
      datasetExists: !!activeDataset,
      datasetKeys: activeDataset ? Object.keys(activeDataset) : [],
      hasId: Object.prototype.hasOwnProperty.call(activeDataset, 'id'),
      idValue: activeDataset?.id,
      idType: typeof activeDataset?.id
    })
    
    console.log('✅ Resolved dataset ID:', datasetId, 'Type:', typeof datasetId)
    
    if (!datasetId) {
      console.error('❌ No dataset ID found in any field')
      console.error('❌ Dataset object keys:', dataset ? Object.keys(dataset) : 'Dataset is null/undefined')
      console.error('❌ Dataset properties:', Object.getOwnPropertyNames(dataset))
      setAiError('Dataset ID not found. The dataset object is missing a valid ID field (id, dataset_id, pk, etc.)')
      setShowFallback(true)
      return
    }

    console.log('✅ Using dataset ID:', datasetId)
    setIsLoadingAI(true)
    setAiError(null)
    
    try {
      // Double-check dataset is still available before making request
      if (!activeDataset) {
        throw new Error('Dataset became unavailable during request')
      }
      
      console.log('🚀 Making AI insights request to:', `http://localhost:5000/api/ai/datasets/${datasetId}/ai-insights`)
      const response = await fetch(`http://localhost:5000/api/ai/datasets/${datasetId}/ai-insights`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      console.log('📡 AI insights response status:', response.status)
      const data = await response.json()
      console.log('📊 AI insights response data:', data)
      
      if (data.success) {
        setAiInsights(data.insights.insights)
        setShowFallback(false)
        console.log('✅ AI insights loaded successfully')
      } else {
        throw new Error(data.error || 'Failed to generate AI insights')
      }
    } catch (error) {
      console.error('❌ Error generating AI insights:', error)
      setAiError(error.message)
      setShowFallback(true)
    } finally {
      setIsLoadingAI(false)
    }
  }

  const generateAdvancedAnalysis = async () => {
    let activeDataset = getActiveDataset()
    
    console.log('🔍 Advanced Analysis Debug - Dataset object:', activeDataset)
    console.log('🔍 Dataset full object structure:', JSON.stringify(activeDataset, null, 2))
    
    // If no dataset available, try to recover using manual ID
    if (!activeDataset) {
      console.warn('⚠️ No dataset available, attempting recovery with ID:', manualDatasetId)
      if (manualDatasetId) {
        activeDataset = await fetchDatasetById(manualDatasetId)
      }
    }
    
    // Final check if dataset exists
    if (!activeDataset) {
      console.error('❌ Dataset is null/undefined for advanced analysis')
      setAdvancedError('No dataset available. Please select a dataset first and ensure it loads properly.')
      return
    }
    
    // Use the primary dataset ID field
    const datasetId = activeDataset.id
    
    console.log('✅ Resolved dataset ID for advanced analysis:', datasetId, 'Type:', typeof datasetId)
    
    if (!datasetId) {
      console.error('❌ No dataset ID found for advanced analysis')
      console.error('❌ Dataset object keys:', activeDataset ? Object.keys(activeDataset) : 'Dataset is null/undefined')
      console.error('❌ Dataset properties:', Object.getOwnPropertyNames(activeDataset))
      setAdvancedError('Dataset ID not found. The dataset object is missing a valid ID field.')
      return
    }

    console.log('✅ Using dataset ID for advanced analysis:', datasetId)
    setIsLoadingAdvanced(true)
    setAdvancedError(null)
    
    try {
      // Double-check dataset is still available before making request
      if (!activeDataset) {
        throw new Error('Dataset became unavailable during request')
      }
      
      console.log('🚀 Making advanced analysis request to:', `http://localhost:5000/api/ai/datasets/${datasetId}/advanced-analysis`)
      const response = await fetch(`http://localhost:5000/api/ai/datasets/${datasetId}/advanced-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      console.log('📡 Advanced analysis response status:', response.status)
      const data = await response.json()
      console.log('📊 Advanced analysis response data:', data)
      
      if (data.success) {
        setAdvancedAnalysis(data.analysis)
        console.log('✅ Advanced analysis loaded successfully')
      } else {
        throw new Error(data.error || 'Failed to generate advanced analysis')
      }
    } catch (error) {
      console.error('❌ Error generating advanced analysis:', error)
      setAdvancedError(error.message)
    } finally {
      setIsLoadingAdvanced(false)
    }
  }

  const generateFallbackInsights = () => {
    const overview = edaData?.overview || {}
    const insights = []
    const totalRows = overview.shape?.rows || 0
    const totalColumns = overview.shape?.columns || 0
    const dataDensity = overview.data_density || 0
    
    // Data size insights
    if (totalRows > 100000) {
      insights.push({
        type: 'info',
        category: 'Dataset Size',
        title: 'Large Dataset Detected',
        description: `Your dataset contains ${totalRows.toLocaleString()} rows and ${totalColumns} columns, which is excellent for robust analysis and machine learning models.`,
        recommendation: 'Consider using sampling techniques for initial exploration to improve performance.',
        priority: 'medium'
      })
    } else if (totalRows < 1000) {
      insights.push({
        type: 'warning',
        category: 'Dataset Size',
        title: 'Small Dataset',
        description: `With only ${totalRows.toLocaleString()} rows and ${totalColumns} columns, statistical conclusions may have limited reliability.`,
        recommendation: 'Consider collecting more data or using techniques like bootstrapping for robust analysis.',
        priority: 'high'
      })
    }

    // Data quality insights
    if (dataDensity > 95) {
      insights.push({
        type: 'success',
        category: 'Data Quality',
        title: 'Excellent Data Completeness',
        description: `Your dataset has ${dataDensity.toFixed(1)}% data density, indicating very few missing values.`,
        recommendation: 'Proceed with analysis - minimal data cleaning required.',
        priority: 'low'
      })
    } else if (dataDensity < 80) {
      insights.push({
        type: 'error',
        category: 'Data Quality',
        title: 'Significant Missing Data',
        description: `Data density is only ${dataDensity.toFixed(1)}%, indicating substantial missing values.`,
        recommendation: 'Implement missing data imputation strategies before proceeding with analysis.',
        priority: 'high'
      })
    }

    // Add some general insights if no specific ones
    if (insights.length === 0) {
      insights.push({
        type: 'info',
        category: 'General',
        title: 'Dataset Ready for Analysis',
        description: `Your dataset with ${totalRows.toLocaleString()} rows and ${totalColumns} columns appears to be in good condition for exploratory data analysis.`,
        recommendation: 'Proceed with statistical analysis and visualization to uncover patterns.',
        priority: 'low'
      })
    }

    return insights.slice(0, 6) // Limit to 6 insights
  }

  const renderAIInsights = () => {
    if (!aiInsights) return null

    return (
      <div className="space-y-8">
        {/* Key Findings */}
        {aiInsights.key_findings && (
          <Card className="shadow-2xl border-0 rounded-3xl">
            <CardHeader className="bg-gradient-to-r from-blue-100 via-white to-blue-100 border-b-2 border-blue-200 p-8 rounded-t-3xl">
              <CardTitle className="flex items-center space-x-3">
                <Brain className="w-8 h-8 text-blue-600" />
                <span className="text-2xl font-bold">AI Key Findings</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-8">
              <div className="grid grid-cols-1 gap-4">
                {aiInsights.key_findings.map((finding, index) => (
                  <div key={index} className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                    <p className="text-gray-800 font-medium">{finding}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Data Quality Assessment */}
        {aiInsights.data_quality_assessment && (
          <Card className="shadow-2xl border-0 rounded-3xl">
            <CardHeader className="bg-gradient-to-r from-green-100 via-white to-green-100 border-b-2 border-green-200 p-8 rounded-t-3xl">
              <CardTitle className="flex items-center space-x-3">
                <CheckCircle className="w-8 h-8 text-green-600" />
                <span className="text-2xl font-bold">AI Quality Assessment</span>
                <Badge className={`ml-4 px-3 py-1 text-lg font-bold ${
                  aiInsights.data_quality_assessment.overall_score === 'A' ? 'bg-green-500' :
                  aiInsights.data_quality_assessment.overall_score === 'B' ? 'bg-blue-500' :
                  aiInsights.data_quality_assessment.overall_score === 'C' ? 'bg-yellow-500' :
                  aiInsights.data_quality_assessment.overall_score === 'D' ? 'bg-orange-500' : 'bg-red-500'
                } text-white`}>
                  Grade: {aiInsights.data_quality_assessment.overall_score}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                  <h4 className="text-lg font-bold text-gray-800 mb-4">Main Issues</h4>
                  {aiInsights.data_quality_assessment.main_issues?.map((issue, index) => (
                    <div key={index} className="p-3 bg-red-50 rounded-lg border border-red-200 mb-3">
                      <p className="text-red-800 font-medium">{issue}</p>
                    </div>
                  ))}
                </div>
                <div>
                  <h4 className="text-lg font-bold text-gray-800 mb-4">Recommendations</h4>
                  {aiInsights.data_quality_assessment.recommendations?.map((rec, index) => (
                    <div key={index} className="p-3 bg-green-50 rounded-lg border border-green-200 mb-3">
                      <p className="text-green-800 font-medium">{rec}</p>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Analysis Recommendations & Business Insights */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          {aiInsights.analysis_recommendations && (
            <Card className="shadow-2xl border-0 rounded-3xl">
              <CardHeader className="bg-gradient-to-r from-purple-100 via-white to-purple-100 border-b-2 border-purple-200 p-8 rounded-t-3xl">
                <CardTitle className="flex items-center space-x-3">
                  <Target className="w-8 h-8 text-purple-600" />
                  <span className="text-xl font-bold">Analysis Recommendations</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-4">
                  {aiInsights.analysis_recommendations.map((rec, index) => (
                    <div key={index} className="p-4 bg-purple-50 rounded-xl border border-purple-200">
                      <p className="text-purple-800 font-medium">{rec}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {aiInsights.business_insights && (
            <Card className="shadow-2xl border-0 rounded-3xl">
              <CardHeader className="bg-gradient-to-r from-orange-100 via-white to-orange-100 border-b-2 border-orange-200 p-8 rounded-t-3xl">
                <CardTitle className="flex items-center space-x-3">
                  <TrendingUp className="w-8 h-8 text-orange-600" />
                  <span className="text-xl font-bold">Business Insights</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-4">
                  {aiInsights.business_insights.map((insight, index) => (
                    <div key={index} className="p-4 bg-orange-50 rounded-xl border border-orange-200">
                      <p className="text-orange-800 font-medium">{insight}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Next Steps */}
        {aiInsights.next_steps && (
          <Card className="shadow-2xl border-0 rounded-3xl">
            <CardHeader className="bg-gradient-to-r from-indigo-100 via-white to-indigo-100 border-b-2 border-indigo-200 p-8 rounded-t-3xl">
              <CardTitle className="flex items-center space-x-3">
                <Zap className="w-8 h-8 text-indigo-600" />
                <span className="text-2xl font-bold">Recommended Next Steps</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {aiInsights.next_steps.map((step, index) => (
                  <div key={index} className="p-6 bg-indigo-50 rounded-2xl border border-indigo-200 transition-shadow">
                    <div className="flex items-start space-x-4">
                      <div className="w-8 h-8 bg-indigo-500 text-white rounded-full flex items-center justify-center font-bold text-sm">
                        {index + 1}
                      </div>
                      <p className="text-indigo-800 font-medium">{step}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="text-center lg:text-left">
        <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4 bg-gradient-to-r from-yellow-600 via-orange-600 to-red-600 bg-clip-text text-transparent">
          AI-Powered Insights
        </h2>
        <p className="text-xl lg:text-2xl text-gray-600 max-w-3xl leading-relaxed">
          Intelligent analysis and actionable recommendations powered by advanced AI
        </p>
      </div>

      {/* Dataset Warning */}
      {!dataset && (
        <Card className="bg-gradient-to-r from-yellow-50 to-orange-50 border-l-4 border-yellow-400">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-8 h-8 text-yellow-600 flex-shrink-0" />
              <div>
                <h3 className="text-lg font-semibold text-yellow-800 mb-2">Dataset Not Available</h3>
                <p className="text-yellow-700">
                  No dataset is currently selected or the dataset information is missing. 
                  Please select a dataset from the Data Management section to enable AI-powered analysis.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Real-time Dataset Debug Card */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-400">
        <CardContent className="p-6">
          <div className="flex items-start space-x-3">
            <Info className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-blue-800 mb-3">Current Dataset State</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-blue-700">Dataset Present:</span>
                  <span className={`ml-2 px-2 py-1 rounded ${dataset ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {dataset ? 'Yes' : 'No'}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-blue-700">Dataset Type:</span>
                  <span className="ml-2 text-blue-600">{typeof dataset}</span>
                </div>
                <div>
                  <span className="font-medium text-blue-700">Dataset ID:</span>
                  <span className={`ml-2 px-2 py-1 rounded ${dataset?.id ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {dataset?.id || 'NOT FOUND'}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-blue-700">Dataset Name:</span>
                  <span className="ml-2 text-blue-600">{dataset?.name || 'N/A'}</span>
                </div>
                <div>
                  <span className="font-medium text-blue-700">Available Keys:</span>
                  <span className="ml-2 text-blue-600 text-xs">
                    {dataset ? Object.keys(dataset).slice(0, 5).join(', ') + (Object.keys(dataset).length > 5 ? '...' : '') : 'None'}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-blue-700">Helper Result:</span>
                  <span className={`ml-2 px-2 py-1 rounded ${dataset?.id ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {dataset?.id || 'No ID Found'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Controls */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start flex-wrap">
        <Button 
          onClick={generateAIInsights}
          disabled={!getActiveDataset() || isLoadingAI || isLoadingAdvanced}
          className="px-8 py-4 bg-gradient-to-r from-yellow-600 via-orange-600 to-red-600 text-white rounded-2xl transition-all duration-300 shadow-xl text-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoadingAI ? (
            <>
              <RefreshCw className="w-5 h-5 mr-3 animate-spin" />
              Generating AI Insights...
            </>
          ) : !getActiveDataset() ? (
            <>
              <AlertTriangle className="w-5 h-5 mr-3" />
              No Dataset Available
            </>
          ) : (
            <>
              <Brain className="w-5 h-5 mr-3" />
              Generate AI Insights
            </>
          )}
        </Button>

        <Button 
          onClick={generateAdvancedAnalysis}
          disabled={!getActiveDataset() || isLoadingAI || isLoadingAdvanced}
          className="px-8 py-4 bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 text-white rounded-2xl transition-all duration-300 shadow-xl text-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoadingAdvanced ? (
            <>
              <RefreshCw className="w-5 h-5 mr-3 animate-spin" />
              Advanced Analysis...
            </>
          ) : !getActiveDataset() ? (
            <>
              <AlertTriangle className="w-5 h-5 mr-3" />
              No Dataset Available
            </>
          ) : (
            <>
              <Zap className="w-5 h-5 mr-3" />
              Use AI (Advanced)
            </>
          )}
        </Button>
        
        {(aiError || showFallback) && (
          <Button 
            onClick={() => setShowFallback(true)}
            variant="outline"
            className="px-6 py-3 border-2 border-gray-300 rounded-xl transition-colors"
          >
            <Eye className="w-5 h-5 mr-2" />
            View Basic Insights
          </Button>
        )}

        {/* Debug Dataset Button */}
        <Button 
          onClick={() => {
            console.log('🔍 === COMPREHENSIVE DATASET DEBUG ===')
            console.log('🔍 Dataset prop:', dataset)
            console.log('🔍 Dataset type:', typeof dataset)
            console.log('🔍 Dataset is array:', Array.isArray(dataset))
            console.log('🔍 Dataset keys:', dataset ? Object.keys(dataset) : 'No dataset')
            console.log('🔍 Dataset properties:', dataset ? Object.getOwnPropertyNames(dataset) : 'No dataset')
            console.log('🔍 All possible IDs:', {
              id: dataset?.id,
              dataset_id: dataset?.dataset_id,
              pk: dataset?.pk,
              key: dataset?.key,
              _id: dataset?._id,
              datasetId: dataset?.datasetId
            })
            console.log('🔍 ID types:', {
              id: typeof dataset?.id,
              dataset_id: typeof dataset?.dataset_id,
              pk: typeof dataset?.pk,
              key: typeof dataset?.key,
              _id: typeof dataset?._id,
              datasetId: typeof dataset?.datasetId
            })
            console.log('🔍 Dataset ID result:', dataset?.id)
            console.log('🔍 Dataset history:', datasetHistory.current)
            console.log('🔍 Previous dataset:', previousDataset.current)
            console.log('🔍 Full dataset JSON:', JSON.stringify(dataset, null, 2))
            console.log('🔍 === END COMPREHENSIVE DEBUG ===')
            
            // Also show in UI
            const historyInfo = datasetHistory.current.map(h => 
              `${h.timestamp.split('T')[1].split('.')[0]}: ${h.hasDataset ? `✓ ID:${h.datasetId}` : '✗ No dataset'}`
            ).join('\n')
            
            alert(`Dataset Debug Info:
- Has Dataset: ${dataset ? 'Yes' : 'No'}
- Dataset Type: ${typeof dataset}
- Dataset ID: ${dataset?.id || 'NOT FOUND'}
- Available Keys: ${dataset ? Object.keys(dataset).join(', ') : 'None'}
- Dataset ID Result: ${dataset?.id}

Dataset History (last 10 changes):
${historyInfo}

Check console for full details.`)
          }}
          variant="outline"
          className="px-6 py-3 border-2 border-blue-300 rounded-xl transition-colors"
        >
          <Info className="w-5 h-5 mr-2" />
          Debug Dataset ID
        </Button>
      </div>

      {/* Manual Dataset Recovery Controls */}
      <Card className="border-2 border-orange-200 bg-orange-50 mt-6">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center text-orange-800">
            <AlertTriangle className="w-5 h-5 mr-2" />
            Dataset Recovery (Testing Mode)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex gap-3 items-center">
              <input
                type="number"
                value={manualDatasetId}
                onChange={(e) => setManualDatasetId(e.target.value)}
                placeholder="Dataset ID (e.g., 15)"
                className="px-3 py-2 border border-gray-300 rounded flex-1 max-w-xs"
              />
              <Button
                onClick={() => fetchDatasetById(manualDatasetId)}
                className="px-4 py-2 bg-orange-600 text-white rounded"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Recover Dataset
              </Button>
            </div>
            
            <div className="text-sm text-orange-700">
              <p>💡 If dataset is showing as null, enter the dataset ID and click "Recover Dataset" to manually fetch it.</p>
              <p>🔍 Known working dataset ID: <strong>15</strong> (complex_test_dataset.csv)</p>
            </div>
            
            {getActiveDataset() && (
              <div className="p-3 bg-green-100 border border-green-300 rounded">
                <div className="text-sm text-green-800">
                  ✅ <strong>Active Dataset Found:</strong><br/>
                  ID: {getActiveDataset()?.id}<br/>
                  Name: {getActiveDataset()?.name}<br/>
                  Status: {getActiveDataset()?.status}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Error State */}
      {aiError && !showFallback && (
        <Card className="shadow-lg border-red-200 border-2">
          <CardContent className="p-8 text-center">
            <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-red-800 mb-2">AI Analysis Error</h3>
            <p className="text-red-600 mb-4">{aiError}</p>
            <div className="bg-red-50 p-4 rounded-lg mb-4 text-left">
              <p className="text-sm text-red-700 mb-2"><strong>Troubleshooting:</strong></p>
              <ul className="text-sm text-red-600 space-y-1">
                <li>• Make sure a dataset is selected and EDA has been run</li>
                <li>• Check that the backend server is running on port 5000</li>
                <li>• Verify the AI service is properly configured</li>
                <li>• Try refreshing the page and selecting the dataset again</li>
              </ul>
            </div>
            <Button onClick={() => setShowFallback(true)} variant="outline" className="border-red-300">
              View Basic Insights Instead
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Loading State */}
      {isLoadingAI && (
        <Card className="shadow-lg">
          <CardContent className="p-12 text-center">
            <Brain className="w-16 h-16 text-blue-500 mx-auto mb-4 animate-pulse" />
            <h3 className="text-xl font-bold text-gray-800 mb-2">AI is analyzing your data...</h3>
            <p className="text-gray-600 mb-4">This may take a few moments while our AI processes your dataset</p>
            <div className="w-64 mx-auto bg-gray-200 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{width: '70%'}}></div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Advanced Loading State */}
      {isLoadingAdvanced && (
        <Card className="shadow-lg border-purple-200 border-2">
          <CardContent className="p-12 text-center">
            <Zap className="w-16 h-16 text-purple-500 mx-auto mb-4 animate-pulse" />
            <h3 className="text-xl font-bold text-gray-800 mb-2">Advanced AI Analysis in Progress...</h3>
            <p className="text-gray-600 mb-4">Our most powerful AI is performing deep analysis of your dataset</p>
            <div className="w-64 mx-auto bg-gray-200 rounded-full h-2">
              <div className="bg-purple-500 h-2 rounded-full animate-pulse" style={{width: '85%'}}></div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* AI Insights */}
      {aiInsights && !showFallback && renderAIInsights()}

      {/* Advanced Analysis Results */}
      {advancedAnalysis && !showFallback && (
        <Card className="shadow-2xl border-0 rounded-3xl bg-gradient-to-br from-purple-50 to-indigo-50">
          <CardHeader className="bg-gradient-to-r from-purple-100 via-white to-indigo-100 border-b-2 border-purple-200 p-8 rounded-t-3xl">
            <CardTitle className="flex items-center space-x-3">
              <Zap className="w-8 h-8 text-purple-600" />
              <span className="text-2xl font-bold">Advanced AI Analysis</span>
              <Badge className="ml-4 px-3 py-1 text-lg font-bold bg-purple-500 text-white">
                NVIDIA AI
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-8">
            <div className="space-y-6">
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-purple-200">
                <h4 className="text-lg font-bold text-purple-800 mb-4 flex items-center">
                  <Brain className="w-5 h-5 mr-2" />
                  Deep Analysis Results
                </h4>
                <div className="prose max-w-none text-gray-700">
                  <pre className="whitespace-pre-wrap text-sm leading-relaxed bg-gray-50 p-4 rounded-xl border">
                    {advancedAnalysis.detailed_analysis || 'Advanced analysis completed successfully'}
                  </pre>
                </div>
                <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
                  <span>Analysis Type: {advancedAnalysis.analysis_type}</span>
                  <span>Model: {advancedAnalysis.model_used}</span>
                  <span>Generated: {new Date(advancedAnalysis.timestamp).toLocaleString()}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Advanced Analysis Error State */}
      {advancedError && (
        <Card className="shadow-lg border-purple-200 border-2">
          <CardContent className="p-8 text-center">
            <AlertTriangle className="w-16 h-16 text-purple-500 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-purple-800 mb-2">Advanced Analysis Error</h3>
            <p className="text-purple-600 mb-4">{advancedError}</p>
            <Button onClick={() => setAdvancedError(null)} variant="outline" className="border-purple-300">
              Dismiss
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Fallback Insights */}
      {showFallback && (
        <div className="space-y-8">
          <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6">
            <div className="flex items-center space-x-3">
              <Info className="w-6 h-6 text-yellow-600" />
              <div>
                <h4 className="font-bold text-yellow-800">Basic Statistical Insights</h4>
                <p className="text-yellow-700">AI insights are not available. Showing basic statistical analysis.</p>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {generateFallbackInsights().map((insight, index) => (
              <Card key={index} className={`shadow-lg border-2 ${
                insight.type === 'success' ? 'border-green-200 bg-green-50' :
                insight.type === 'warning' ? 'border-yellow-200 bg-yellow-50' :
                insight.type === 'error' ? 'border-red-200 bg-red-50' : 'border-blue-200 bg-blue-50'
              }`}>
                <CardContent className="p-6">
                  <div className="flex items-start space-x-4">
                    <div className={`p-2 rounded-lg ${
                      insight.type === 'success' ? 'bg-green-200' :
                      insight.type === 'warning' ? 'bg-yellow-200' :
                      insight.type === 'error' ? 'bg-red-200' : 'bg-blue-200'
                    }`}>
                      {insight.type === 'success' ? <CheckCircle className="w-6 h-6 text-green-600" /> :
                       insight.type === 'warning' ? <AlertTriangle className="w-6 h-6 text-yellow-600" /> :
                       insight.type === 'error' ? <X className="w-6 h-6 text-red-600" /> :
                       <Info className="w-6 h-6 text-blue-600" />}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <h4 className="font-bold text-gray-800">{insight.title}</h4>
                        <Badge variant="secondary" className="text-xs">
                          {insight.category}
                        </Badge>
                      </div>
                      <p className="text-gray-600 mb-3">{insight.description}</p>
                      <div className="bg-white p-3 rounded-lg border">
                        <p className="text-sm font-medium text-gray-700">
                          <strong>Recommendation:</strong> {insight.recommendation}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Debug Section Component
function DebugSection({ edaData, dataset }) {
  return (
    <div className="space-y-8">
      <div className="text-center lg:text-left">
        <h2 className="text-3xl font-bold text-gray-900 mb-3 bg-gradient-to-r from-gray-600 to-slate-600 bg-clip-text text-transparent">
          Debug Information
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl">
          Technical details and raw data structure for troubleshooting
        </p>
      </div>
      
      {/* Testing Section */}
      <Card className="shadow-lg border-0">
        <CardHeader className="bg-gradient-to-r from-green-50 to-green-100 border-b">
          <CardTitle className="flex items-center space-x-2">
            <Target className="w-6 h-6 text-green-600" />
            <span>API Testing & Diagnostics</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Button 
                onClick={async () => {
                  try {
                    console.log('🧪 Testing backend connection...')
                    const response = await fetch('http://localhost:5000/api/data/datasets')
                    console.log('📡 Backend test response:', {
                      status: response.status,
                      ok: response.ok,
                      statusText: response.statusText
                    })
                    if (response.ok) {
                      const data = await response.json()
                      console.log('✅ Backend connected! Datasets count:', data.datasets?.length || 0)
                    } else {
                      console.error('❌ Backend connection failed')
                    }
                  } catch (error) {
                    console.error('❌ Backend connection error:', error)
                  }
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg"
              >
                <Activity className="w-4 h-4 mr-2" />
                Test Backend Connection
              </Button>
              
              <Button 
                onClick={async () => {
                  if (!dataset?.id) {
                    console.error('❌ No dataset available for EDA testing')
                    return
                  }
                  try {
                    console.log('🧪 Testing EDA endpoint...')
                    const response = await fetch(`http://localhost:5000/api/data/datasets/${dataset.id}/eda`)
                    console.log('📡 EDA test response:', {
                      status: response.status,
                      ok: response.ok,
                      statusText: response.statusText
                    })
                    if (response.ok) {
                      const data = await response.json()
                      console.log('✅ EDA endpoint working! Charts available:', Object.keys(data.charts || {}))
                    } else {
                      console.error('❌ EDA endpoint failed')
                    }
                  } catch (error) {
                    console.error('❌ EDA endpoint error:', error)
                  }
                }}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg"
                disabled={!dataset?.id}
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                Test EDA Endpoint
              </Button>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-700 mb-2">
                <strong>Dataset Status:</strong> {dataset ? 'Available' : 'No dataset selected'}
              </p>
              {dataset && (
                <p className="text-sm text-gray-700">
                  <strong>Dataset ID:</strong> {dataset.id || 'No ID found'}
                </p>
              )}
              <p className="text-xs text-gray-500 mt-2">
                Check browser console for detailed test results and network requests
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Dataset Information */}
      <Card className="shadow-lg border-0">
        <CardHeader className="bg-gradient-to-r from-purple-50 to-purple-100 border-b">
          <CardTitle className="flex items-center space-x-2">
            <Layers className="w-6 h-6 text-purple-600" />
            <span>Dataset Information</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Dataset Object:</h4>
              <div className="p-3 bg-purple-50 rounded-lg">
                <p className="text-sm text-purple-800 font-mono">
                  Type: {typeof dataset} | Available: {dataset ? 'Yes' : 'No'}
                </p>
                {dataset && (
                  <div className="mt-2 space-y-1">
                    <p className="text-sm text-purple-700">Keys: {Object.keys(dataset).join(', ')}</p>
                    <p className="text-sm text-purple-700">
                      ID Fields: id={dataset.id || 'null'}, dataset_id={dataset.dataset_id || 'null'}, 
                      pk={dataset.pk || 'null'}, key={dataset.key || 'null'}
                    </p>
                  </div>
                )}
              </div>
            </div>
            
            {dataset && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">Dataset Properties:</h4>
                <div className="bg-purple-50 rounded-lg p-4 max-h-64 overflow-y-auto">
                  <pre className="text-xs text-purple-700 font-mono whitespace-pre-wrap">
                    {JSON.stringify(dataset, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
      
      {/* EDA Data Information */}
      <Card className="shadow-lg border-0">
        <CardHeader className="bg-gradient-to-r from-gray-50 to-gray-100 border-b">
          <CardTitle className="flex items-center space-x-2">
            <Info className="w-6 h-6 text-gray-600" />
            <span>EDA Data Structure</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Data Keys:</h4>
              <div className="p-3 bg-gray-100 rounded-lg">
                <p className="text-sm text-gray-600 font-mono">
                  {edaData ? Object.keys(edaData).join(', ') : 'No data available'}
                </p>
              </div>
            </div>
            
            {edaData?.charts && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-3">Chart Categories:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {Object.entries(edaData.charts).map(([category, charts]) => (
                    <div key={category} className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <div className="font-medium text-blue-800">{category}</div>
                      <div className="text-sm text-blue-600">
                        {Array.isArray(charts) ? charts.length : 0} charts
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {edaData?.charts && (
              <div>
                <h4 className="font-semibold text-gray-800 mb-3">Sample Chart Data:</h4>
                <div className="bg-gray-100 rounded-lg p-4 max-h-96 overflow-y-auto">
                  <pre className="text-xs text-gray-700 font-mono whitespace-pre-wrap">
                    {JSON.stringify(
                      Object.fromEntries(
                        Object.entries(edaData.charts).map(([category, charts]) => [
                          category, 
                          Array.isArray(charts) && charts.length > 0 
                            ? {
                                title: charts[0].title,
                                hasChart: !!charts[0].chart,
                                hasData: !!charts[0].chart?.data,
                                hasLayout: !!charts[0].chart?.layout,
                                dataLength: charts[0].chart?.data?.length || 0
                              }
                            : 'No charts available'
                        ])
                      ),
                      null,
                      2
                    )}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
