import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { toast } from 'sonner'
import { 
  Monitor, Activity, RefreshCw, Settings, Cpu, HardDrive, Zap, 
  Thermometer, Network, Database, AlertTriangle, CheckCircle, 
  XCircle, Clock, TrendingUp, TrendingDown, Wifi, Server,
  MemoryStick, Battery, Shield, Eye, EyeOff, Bell, BellOff,
  Download, Upload, Calendar, BarChart3, LineChart, Play, Pause,
  Maximize2, Minimize2, RotateCcw, Filter, Search, AlertCircle, Save
} from 'lucide-react'
import Plot from 'react-plotly.js'

const API_BASE = 'http://localhost:5000/api'

// Mock system data generator for demo purposes
const generateSystemMetrics = () => {
  const now = Date.now()
  return {
    timestamp: now,
    cpu: {
      usage: Math.random() * 100,
      cores: 8,
      frequency: 2.8 + Math.random() * 1.2,
      temperature: 45 + Math.random() * 30,
      processes: Math.floor(150 + Math.random() * 50)
    },
    memory: {
      total: 16384, // MB
      used: Math.floor(Math.random() * 12000) + 2000,
      available: 0,
      cached: Math.floor(Math.random() * 2000) + 500,
      buffers: Math.floor(Math.random() * 500) + 100
    },
    disk: {
      total: 500000, // MB
      used: Math.floor(Math.random() * 300000) + 100000,
      available: 0,
      readSpeed: Math.random() * 200,
      writeSpeed: Math.random() * 150,
      iops: Math.floor(Math.random() * 1000) + 100
    },
    network: {
      downloadSpeed: Math.random() * 100,
      uploadSpeed: Math.random() * 50,
      latency: Math.random() * 50 + 10,
      packetsIn: Math.floor(Math.random() * 10000),
      packetsOut: Math.floor(Math.random() * 8000),
      errors: Math.floor(Math.random() * 5)
    },
    system: {
      uptime: Math.floor(Math.random() * 86400 * 7), // seconds
      loadAverage: [
        Math.random() * 4,
        Math.random() * 4,
        Math.random() * 4
      ],
      processes: Math.floor(150 + Math.random() * 50),
      threads: Math.floor(800 + Math.random() * 200)
    }
  }
}

export function SystemMonitor() {
  // State management
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [historicalData, setHistoricalData] = useState([])
  const [refreshInterval, setRefreshInterval] = useState(1000)
  const [alertsEnabled, setAlertsEnabled] = useState(true)
  const [expandedView, setExpandedView] = useState(false)
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [alerts, setAlerts] = useState([])
  const [thresholds, setThresholds] = useState({
    cpu: 80,
    memory: 85,
    disk: 90,
    temperature: 70
  })

  // Alert checking
  const checkAlerts = useCallback((metrics) => {
    const newAlerts = []
    
    if (metrics.cpu.usage > thresholds.cpu) {
      newAlerts.push({
        id: Date.now() + 'cpu',
        type: 'warning',
        metric: 'CPU',
        message: `CPU usage is high (${metrics.cpu.usage.toFixed(1)}%)`,
        timestamp: new Date().toISOString()
      })
    }
    
    const memoryUsagePercent = (metrics.memory.used / metrics.memory.total) * 100
    if (memoryUsagePercent > thresholds.memory) {
      newAlerts.push({
        id: Date.now() + 'memory',
        type: 'warning',
        metric: 'Memory',
        message: `Memory usage is high (${memoryUsagePercent.toFixed(1)}%)`,
        timestamp: new Date().toISOString()
      })
    }
    
    if (metrics.cpu.temperature > thresholds.temperature) {
      newAlerts.push({
        id: Date.now() + 'temp',
        type: 'critical',
        metric: 'Temperature',
        message: `CPU temperature is high (${metrics.cpu.temperature.toFixed(1)}°C)`,
        timestamp: new Date().toISOString()
      })
    }
    
    if (newAlerts.length > 0) {
      setAlerts(prev => [...newAlerts, ...prev].slice(0, 50)) // Keep last 50 alerts
      newAlerts.forEach(alert => {
        toast.error(`System Alert: ${alert.message}`)
      })
    }
  }, [thresholds])

  // Real-time data fetching
  const fetchSystemMetrics = useCallback(async () => {
    try {
      // Call the real backend API
      const response = await fetch(`${API_BASE}/system/metrics`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch metrics')
      }
      
      const newMetrics = data.metrics
      
      // Calculate derived values
      newMetrics.memory.available = newMetrics.memory.total - newMetrics.memory.used
      newMetrics.disk.available = newMetrics.disk.total - newMetrics.disk.used
      
      setMetrics(newMetrics)
      
      // Add to historical data
      setHistoricalData(prev => {
        const updated = [...prev, newMetrics].slice(-100) // Keep last 100 points
        return updated
      })
      
      // Check for alerts
      if (alertsEnabled) {
        checkAlerts(newMetrics)
      }
      
    } catch (error) {
      console.error('Error fetching system metrics:', error)
      toast.error('Failed to fetch system metrics: ' + error.message)
      
      // Fallback to mock data if backend is unavailable
      const fallbackMetrics = generateSystemMetrics()
      fallbackMetrics.memory.available = fallbackMetrics.memory.total - fallbackMetrics.memory.used
      fallbackMetrics.disk.available = fallbackMetrics.disk.total - fallbackMetrics.disk.used
      setMetrics(fallbackMetrics)
      setHistoricalData(prev => [...prev, fallbackMetrics].slice(-100))
    }
  }, [alertsEnabled, checkAlerts])

  // Backend API functions
  const fetchSystemAlerts = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/system/alerts`)
      if (response.ok) {
        const data = await response.json()
        if (data.success) {
          setAlerts(data.alerts)
        }
      }
    } catch (error) {
      console.error('Error fetching system alerts:', error)
    }
  }, [])

  const fetchMonitoringConfig = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/system/monitoring-config`)
      if (response.ok) {
        const data = await response.json()
        if (data.success) {
          const config = data.config
          setRefreshInterval(config.refresh_interval)
          setThresholds(config.alert_thresholds)
          setAutoRefresh(config.auto_refresh)
          setAlertsEnabled(config.alerts_enabled)
        }
      }
    } catch (error) {
      console.error('Error fetching monitoring config:', error)
    }
  }, [])

  const saveMonitoringConfig = async (config) => {
    try {
      const response = await fetch(`${API_BASE}/system/monitoring-config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
      })
      
      if (response.ok) {
        toast.success('Configuration saved successfully')
      } else {
        throw new Error('Failed to save configuration')
      }
    } catch (error) {
      console.error('Error saving monitoring config:', error)
      toast.error('Failed to save configuration')
    }
  }

  const handleSaveConfig = () => {
    const config = {
      refresh_interval: refreshInterval,
      alert_thresholds: thresholds,
      auto_refresh: autoRefresh,
      alerts_enabled: alertsEnabled
    }
    saveMonitoringConfig(config)
  }

  // Auto-refresh effect
  useEffect(() => {
    let interval
    if (isMonitoring && autoRefresh) {
      interval = setInterval(fetchSystemMetrics, refreshInterval)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isMonitoring, autoRefresh, refreshInterval, fetchSystemMetrics])

  // Initial data fetch
  useEffect(() => {
    fetchSystemMetrics()
    fetchSystemAlerts()
    fetchMonitoringConfig()
  }, [fetchSystemMetrics, fetchSystemAlerts, fetchMonitoringConfig])

  const formatUptime = (seconds) => {
    const days = Math.floor(seconds / 86400)
    const hours = Math.floor((seconds % 86400) / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`
    if (hours > 0) return `${hours}h ${minutes}m`
    return `${minutes}m`
  }

  const formatBytes = (bytes) => {
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    if (bytes === 0) return '0 B'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`
  }

  const getStatusColor = (value, threshold) => {
    if (value > threshold) return 'text-red-500'
    if (value > threshold * 0.8) return 'text-yellow-500'
    return 'text-green-500'
  }

  const getStatusIcon = (value, threshold) => {
    if (value > threshold) return <XCircle className="h-4 w-4 text-red-500" />
    if (value > threshold * 0.8) return <AlertTriangle className="h-4 w-4 text-yellow-500" />
    return <CheckCircle className="h-4 w-4 text-green-500" />
  }

  const generateTimeSeriesChart = (dataKey, title, unit = '%') => {
    if (historicalData.length === 0) return null
    
    const trace = {
      x: historicalData.map(d => new Date(d.timestamp)),
      y: historicalData.map(d => {
        switch (dataKey) {
          case 'cpu':
            return d.cpu.usage
          case 'memory':
            return (d.memory.used / d.memory.total) * 100
          case 'disk':
            return (d.disk.used / d.disk.total) * 100
          case 'temperature':
            return d.cpu.temperature
          case 'network_down':
            return d.network.downloadSpeed
          case 'network_up':
            return d.network.uploadSpeed
          default:
            return 0
        }
      }),
      type: 'scatter',
      mode: 'lines',
      name: title,
      line: { 
        color: dataKey === 'cpu' ? '#3b82f6' : 
               dataKey === 'memory' ? '#10b981' :
               dataKey === 'disk' ? '#f59e0b' :
               dataKey === 'temperature' ? '#ef4444' :
               dataKey === 'network_down' ? '#8b5cf6' : '#06b6d4',
        width: 2
      },
      fill: 'tonexty'
    }
    
    return {
      data: [trace],
      layout: {
        title: title,
        xaxis: { title: 'Time' },
        yaxis: { title: unit },
        height: 300,
        margin: { l: 50, r: 50, t: 50, b: 50 },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
      }
    }
  }

  if (!metrics) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Loading system metrics...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`p-6 space-y-6 ${expandedView ? 'min-h-screen' : ''}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 via-green-600 to-purple-600 bg-clip-text text-transparent">
            System Monitor
          </h1>
          <p className="text-xl text-muted-foreground mt-2">
            Real-time system performance monitoring and alerting
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Button
            variant="outline"
            onClick={() => setExpandedView(!expandedView)}
          >
            {expandedView ? <Minimize2 className="h-4 w-4 mr-2" /> : <Maximize2 className="h-4 w-4 mr-2" />}
            {expandedView ? 'Compact' : 'Expand'}
          </Button>
          <Button
            variant="outline"
            onClick={fetchSystemMetrics}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button
            onClick={() => setIsMonitoring(!isMonitoring)}
            className={isMonitoring ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
          >
            {isMonitoring ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            {isMonitoring ? 'Stop' : 'Start'} Monitoring
          </Button>
        </div>
      </div>

      {/* Quick Status Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-r from-blue-50 to-blue-100">
          <CardContent className="flex items-center p-4">
            <Cpu className="h-8 w-8 text-blue-600 mr-3" />
            <div>
              <p className="text-sm font-medium text-blue-800">CPU Usage</p>
              <p className="text-2xl font-bold text-blue-900">{metrics.cpu.usage.toFixed(1)}%</p>
            </div>
            {getStatusIcon(metrics.cpu.usage, thresholds.cpu)}
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-green-50 to-green-100">
          <CardContent className="flex items-center p-4">
            <MemoryStick className="h-8 w-8 text-green-600 mr-3" />
            <div>
              <p className="text-sm font-medium text-green-800">Memory</p>
              <p className="text-2xl font-bold text-green-900">
                {((metrics.memory.used / metrics.memory.total) * 100).toFixed(1)}%
              </p>
            </div>
            {getStatusIcon((metrics.memory.used / metrics.memory.total) * 100, thresholds.memory)}
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-orange-50 to-orange-100">
          <CardContent className="flex items-center p-4">
            <HardDrive className="h-8 w-8 text-orange-600 mr-3" />
            <div>
              <p className="text-sm font-medium text-orange-800">Disk Usage</p>
              <p className="text-2xl font-bold text-orange-900">
                {((metrics.disk.used / metrics.disk.total) * 100).toFixed(1)}%
              </p>
            </div>
            {getStatusIcon((metrics.disk.used / metrics.disk.total) * 100, thresholds.disk)}
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-red-50 to-red-100">
          <CardContent className="flex items-center p-4">
            <Thermometer className="h-8 w-8 text-red-600 mr-3" />
            <div>
              <p className="text-sm font-medium text-red-800">Temperature</p>
              <p className="text-2xl font-bold text-red-900">{metrics.cpu.temperature.toFixed(1)}°C</p>
            </div>
            {getStatusIcon(metrics.cpu.temperature, thresholds.temperature)}
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* CPU Details */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Cpu className="h-5 w-5 mr-2" />
                  CPU Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Usage</span>
                    <span className={getStatusColor(metrics.cpu.usage, thresholds.cpu)}>
                      {metrics.cpu.usage.toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={metrics.cpu.usage} className="h-2" />
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Cores</p>
                    <p className="font-semibold">{metrics.cpu.cores}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Frequency</p>
                    <p className="font-semibold">{metrics.cpu.frequency.toFixed(1)} GHz</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Temperature</p>
                    <p className={`font-semibold ${getStatusColor(metrics.cpu.temperature, thresholds.temperature)}`}>
                      {metrics.cpu.temperature.toFixed(1)}°C
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Processes</p>
                    <p className="font-semibold">{metrics.cpu.processes}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Memory Details */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <MemoryStick className="h-5 w-5 mr-2" />
                  Memory Usage
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Used</span>
                    <span className={getStatusColor((metrics.memory.used / metrics.memory.total) * 100, thresholds.memory)}>
                      {formatBytes(metrics.memory.used * 1024 * 1024)} / {formatBytes(metrics.memory.total * 1024 * 1024)}
                    </span>
                  </div>
                  <Progress value={(metrics.memory.used / metrics.memory.total) * 100} className="h-2" />
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Available</p>
                    <p className="font-semibold">{formatBytes(metrics.memory.available * 1024 * 1024)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Cached</p>
                    <p className="font-semibold">{formatBytes(metrics.memory.cached * 1024 * 1024)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Buffers</p>
                    <p className="font-semibold">{formatBytes(metrics.memory.buffers * 1024 * 1024)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Usage %</p>
                    <p className={`font-semibold ${getStatusColor((metrics.memory.used / metrics.memory.total) * 100, thresholds.memory)}`}>
                      {((metrics.memory.used / metrics.memory.total) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Disk Usage */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <HardDrive className="h-5 w-5 mr-2" />
                  Storage
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Used</span>
                    <span className={getStatusColor((metrics.disk.used / metrics.disk.total) * 100, thresholds.disk)}>
                      {formatBytes(metrics.disk.used * 1024 * 1024)} / {formatBytes(metrics.disk.total * 1024 * 1024)}
                    </span>
                  </div>
                  <Progress value={(metrics.disk.used / metrics.disk.total) * 100} className="h-2" />
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Read Speed</p>
                    <p className="font-semibold">{metrics.disk.readSpeed.toFixed(1)} MB/s</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Write Speed</p>
                    <p className="font-semibold">{metrics.disk.writeSpeed.toFixed(1)} MB/s</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">IOPS</p>
                    <p className="font-semibold">{metrics.disk.iops}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Available</p>
                    <p className="font-semibold">{formatBytes(metrics.disk.available * 1024 * 1024)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* System Info */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Server className="h-5 w-5 mr-2" />
                  System Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Uptime</p>
                    <p className="font-semibold">{formatUptime(metrics.system.uptime)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Load Average</p>
                    <p className="font-semibold">
                      {metrics.system.loadAverage.map(load => load.toFixed(2)).join(', ')}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Processes</p>
                    <p className="font-semibold">{metrics.system.processes}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Threads</p>
                    <p className="font-semibold">{metrics.system.threads}</p>
                  </div>
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Last Updated</span>
                  <span className="text-sm font-medium">
                    {new Date(metrics.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {historicalData.length > 0 && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle>CPU Usage Over Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {(() => {
                      const chartConfig = generateTimeSeriesChart('cpu', 'CPU Usage', '%')
                      return chartConfig ? (
                        <Plot
                          data={chartConfig.data}
                          layout={chartConfig.layout}
                          config={{ responsive: true, displayModeBar: false }}
                          style={{ width: '100%' }}
                        />
                      ) : null
                    })()}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Memory Usage Over Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {(() => {
                      const chartConfig = generateTimeSeriesChart('memory', 'Memory Usage', '%')
                      return chartConfig ? (
                        <Plot
                          data={chartConfig.data}
                          layout={chartConfig.layout}
                          config={{ responsive: true, displayModeBar: false }}
                          style={{ width: '100%' }}
                        />
                      ) : null
                    })()}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Temperature Over Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {(() => {
                      const chartConfig = generateTimeSeriesChart('temperature', 'CPU Temperature', '°C')
                      return chartConfig ? (
                        <Plot
                          data={chartConfig.data}
                          layout={chartConfig.layout}
                          config={{ responsive: true, displayModeBar: false }}
                          style={{ width: '100%' }}
                        />
                      ) : null
                    })()}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Disk Usage Over Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {(() => {
                      const chartConfig = generateTimeSeriesChart('disk', 'Disk Usage', '%')
                      return chartConfig ? (
                        <Plot
                          data={chartConfig.data}
                          layout={chartConfig.layout}
                          config={{ responsive: true, displayModeBar: false }}
                          style={{ width: '100%' }}
                        />
                      ) : null
                    })()}
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </TabsContent>

        <TabsContent value="network" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Network className="h-5 w-5 mr-2" />
                  Network Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <Download className="h-6 w-6 mx-auto mb-2 text-blue-600" />
                    <p className="text-sm text-muted-foreground">Download</p>
                    <p className="text-lg font-bold text-blue-600">
                      {metrics.network.downloadSpeed.toFixed(1)} Mbps
                    </p>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <Upload className="h-6 w-6 mx-auto mb-2 text-green-600" />
                    <p className="text-sm text-muted-foreground">Upload</p>
                    <p className="text-lg font-bold text-green-600">
                      {metrics.network.uploadSpeed.toFixed(1)} Mbps
                    </p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Latency</p>
                    <p className="font-semibold">{metrics.network.latency.toFixed(1)} ms</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Packets In</p>
                    <p className="font-semibold">{metrics.network.packetsIn.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Packets Out</p>
                    <p className="font-semibold">{metrics.network.packetsOut.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Errors</p>
                    <p className={`font-semibold ${metrics.network.errors > 0 ? 'text-red-500' : 'text-green-500'}`}>
                      {metrics.network.errors}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {historicalData.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Network Traffic Over Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <Plot
                    data={[
                      {
                        x: historicalData.map(d => new Date(d.timestamp)),
                        y: historicalData.map(d => d.network.downloadSpeed),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Download',
                        line: { color: '#3b82f6' }
                      },
                      {
                        x: historicalData.map(d => new Date(d.timestamp)),
                        y: historicalData.map(d => d.network.uploadSpeed),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Upload',
                        line: { color: '#10b981' }
                      }
                    ]}
                    layout={{
                      title: 'Network Speed',
                      xaxis: { title: 'Time' },
                      yaxis: { title: 'Mbps' },
                      height: 300,
                      margin: { l: 50, r: 50, t: 50, b: 50 }
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%' }}
                  />
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center">
                      <AlertTriangle className="h-5 w-5 mr-2" />
                      Recent Alerts
                    </span>
                    <Badge variant={alertsEnabled ? "default" : "secondary"}>
                      {alertsEnabled ? 'Enabled' : 'Disabled'}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[400px]">
                    {alerts.length === 0 ? (
                      <div className="text-center py-8">
                        <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
                        <p className="text-muted-foreground">No alerts at this time</p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {alerts.map((alert) => (
                          <div key={alert.id} className={`p-3 rounded-lg border ${
                            alert.type === 'critical' ? 'bg-red-50 border-red-200' :
                            alert.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                            'bg-blue-50 border-blue-200'
                          }`}>
                            <div className="flex items-start justify-between">
                              <div className="flex items-start space-x-2">
                                {alert.type === 'critical' ? 
                                  <XCircle className="h-5 w-5 text-red-500 mt-0.5" /> :
                                  <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5" />
                                }
                                <div>
                                  <p className="font-medium">{alert.metric} Alert</p>
                                  <p className="text-sm text-muted-foreground">{alert.message}</p>
                                </div>
                              </div>
                              <span className="text-xs text-muted-foreground">
                                {new Date(alert.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Alert Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="alerts-enabled">Enable Alerts</Label>
                  <Switch
                    id="alerts-enabled"
                    checked={alertsEnabled}
                    onCheckedChange={setAlertsEnabled}
                  />
                </div>
                
                <Separator />
                
                <div className="space-y-3">
                  <div>
                    <Label htmlFor="cpu-threshold">CPU Threshold (%)</Label>
                    <Input
                      id="cpu-threshold"
                      type="number"
                      value={thresholds.cpu}
                      onChange={(e) => setThresholds(prev => ({ ...prev, cpu: Number(e.target.value) }))}
                      min="1"
                      max="100"
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="memory-threshold">Memory Threshold (%)</Label>
                    <Input
                      id="memory-threshold"
                      type="number"
                      value={thresholds.memory}
                      onChange={(e) => setThresholds(prev => ({ ...prev, memory: Number(e.target.value) }))}
                      min="1"
                      max="100"
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="disk-threshold">Disk Threshold (%)</Label>
                    <Input
                      id="disk-threshold"
                      type="number"
                      value={thresholds.disk}
                      onChange={(e) => setThresholds(prev => ({ ...prev, disk: Number(e.target.value) }))}
                      min="1"
                      max="100"
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="temp-threshold">Temperature Threshold (°C)</Label>
                    <Input
                      id="temp-threshold"
                      type="number"
                      value={thresholds.temperature}
                      onChange={(e) => setThresholds(prev => ({ ...prev, temperature: Number(e.target.value) }))}
                      min="30"
                      max="100"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="h-5 w-5 mr-2" />
                  Monitoring Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="auto-refresh">Auto Refresh</Label>
                  <Switch
                    id="auto-refresh"
                    checked={autoRefresh}
                    onCheckedChange={setAutoRefresh}
                  />
                </div>
                
                <div>
                  <Label htmlFor="refresh-interval">Refresh Interval</Label>
                  <Select value={refreshInterval.toString()} onValueChange={(value) => setRefreshInterval(Number(value))}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="500">0.5 seconds</SelectItem>
                      <SelectItem value="1000">1 second</SelectItem>
                      <SelectItem value="2000">2 seconds</SelectItem>
                      <SelectItem value="5000">5 seconds</SelectItem>
                      <SelectItem value="10000">10 seconds</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <Label htmlFor="time-range">Chart Time Range</Label>
                  <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="15m">15 minutes</SelectItem>
                      <SelectItem value="1h">1 hour</SelectItem>
                      <SelectItem value="6h">6 hours</SelectItem>
                      <SelectItem value="24h">24 hours</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <Separator />
                
                <div className="space-y-2">
                  <Button 
                    className="w-full"
                    onClick={handleSaveConfig}
                  >
                    <Save className="h-4 w-4 mr-2" />
                    Save Configuration
                  </Button>
                  
                  <Button 
                    variant="outline" 
                    className="w-full"
                    onClick={() => {
                      setHistoricalData([])
                      setAlerts([])
                      toast.success('Historical data cleared')
                    }}
                  >
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Clear Historical Data
                  </Button>
                  
                  <Button 
                    variant="outline" 
                    className="w-full"
                    onClick={() => {
                      const data = {
                        metrics,
                        historicalData,
                        alerts,
                        thresholds,
                        timestamp: new Date().toISOString()
                      }
                      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `system-monitor-${new Date().toISOString().split('T')[0]}.json`
                      a.click()
                      URL.revokeObjectURL(url)
                      toast.success('Data exported successfully')
                    }}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Export Data
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Monitoring Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Monitoring Active</span>
                  <Badge variant={isMonitoring ? "default" : "secondary"}>
                    {isMonitoring ? 'Active' : 'Inactive'}
                  </Badge>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Data Points Collected</span>
                  <span className="font-semibold">{historicalData.length}</span>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Active Alerts</span>
                  <span className="font-semibold">{alerts.length}</span>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Refresh Rate</span>
                  <span className="font-semibold">{refreshInterval}ms</span>
                </div>
                
                <Separator />
                
                <div className="text-sm text-muted-foreground">
                  <p className="mb-2">System monitoring provides real-time insights into your system's performance, helping you identify bottlenecks and potential issues before they become critical.</p>
                  <p>Configure alert thresholds to receive notifications when system resources exceed safe limits.</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

