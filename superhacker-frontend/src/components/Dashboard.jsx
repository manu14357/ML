import { useState, useEffect } from 'react'
import { 
  Activity, 
  Database, 
  Brain, 
  GitBranch, 
  BarChart3, 
  TrendingUp,
  TrendingDown,
  Cpu,
  HardDrive,
  MemoryStick,
  Clock,
  AlertTriangle,
  CheckCircle2,
  Zap
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8']

export function Dashboard() {
  const [stats, setStats] = useState(null)
  const [overview, setOverview] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        
        // Fetch stats and overview data
        const [statsResponse, overviewResponse] = await Promise.all([
          fetch('http://localhost:5000/api/system/stats'),
          fetch('http://localhost:5000/api/dashboard/overview')
        ])

        if (!statsResponse.ok || !overviewResponse.ok) {
          throw new Error('Failed to fetch dashboard data')
        }

        const statsData = await statsResponse.json()
        const overviewData = await overviewResponse.json()

        setStats(statsData.stats)
        setOverview(overviewData.overview)
        setError(null)
      } catch (err) {
        console.error('Dashboard fetch error:', err)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    // Refresh data every 30 seconds
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="pb-2">
                <div className="h-4 bg-muted rounded w-3/4"></div>
              </CardHeader>
              <CardContent>
                <div className="h-8 bg-muted rounded w-1/2 mb-2"></div>
                <div className="h-3 bg-muted rounded w-full"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="flex items-center text-destructive">
              <AlertTriangle className="h-5 w-5 mr-2" />
              Dashboard Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">{error}</p>
            <p className="text-sm text-muted-foreground mt-2">
              Make sure the backend server is running on http://localhost:5000
            </p>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Prepare chart data
  const algorithmData = stats?.ml?.algorithms_available ? 
    Object.entries(stats.ml.algorithms_available).map(([name, count]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value: count
    })) : []

  const systemMetrics = [
    { name: 'CPU', value: stats?.system?.cpu_usage || 0, color: '#0088FE' },
    { name: 'Memory', value: stats?.system?.memory_usage || 0, color: '#00C49F' },
    { name: 'Disk', value: stats?.system?.disk_usage || 0, color: '#FFBB28' }
  ]

  return (
    <div className="p-3 sm:p-4 md:p-6 space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight truncate">
            Dashboard
          </h1>
          <p className="text-muted-foreground text-sm sm:text-base">
            Welcome to your advanced data science platform
          </p>
        </div>
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2">
          <Badge variant="outline" className="text-green-600 border-green-600 text-xs">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            System Healthy
          </Badge>
          <Badge variant="secondary" className="text-xs">
            <Clock className="h-3 w-3 mr-1" />
            <span className="hidden sm:inline">Last updated: </span>
            {new Date().toLocaleTimeString()}
          </Badge>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <Card className="border-l-4 border-l-blue-500">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Datasets</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{overview?.totals?.datasets || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats?.data?.recent_datasets || 0} added recently
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-green-500">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ML Models</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{overview?.totals?.models || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats?.ml?.deployed_models || 0} deployed
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-purple-500">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Workflows</CardTitle>
            <GitBranch className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{overview?.totals?.workflows || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats?.workflow?.running_workflows || 0} running
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-orange-500">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Visualizations</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{overview?.totals?.visualizations || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats?.visualization?.dashboards || 0} dashboards
            </p>
          </CardContent>
        </Card>
      </div>

      {/* System Performance */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center text-lg">
              <Activity className="h-5 w-5 mr-2" />
              System Performance
            </CardTitle>
            <CardDescription>Real-time system metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {systemMetrics.map((metric) => (
              <div key={metric.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{metric.name} Usage</span>
                  <span className="text-sm text-muted-foreground">{metric.value.toFixed(1)}%</span>
                </div>
                <Progress value={metric.value} className="h-2" />
              </div>
            ))}
            <div className="pt-2 border-t">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center">
                  <Clock className="h-4 w-4 mr-1" />
                  Uptime
                </span>
                <span>{stats?.system?.uptime_hours || 0}h</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center text-lg">
              <Brain className="h-5 w-5 mr-2" />
              ML Algorithms Available
            </CardTitle>
            <CardDescription>Distribution of available algorithms</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-48 sm:h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={algorithmData}
                    cx="50%"
                    cy="50%"
                    innerRadius={30}
                    outerRadius={60}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {algorithmData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-4">
              {algorithmData.map((entry, index) => (
                <div key={entry.name} className="flex items-center text-sm">
                  <div 
                    className="w-3 h-3 rounded-full mr-2 flex-shrink-0" 
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="truncate">{entry.name}: {entry.value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Activity Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 sm:gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Data Quality</CardTitle>
            <CardDescription>Average data quality metrics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Quality Score</span>
                <span className="font-medium">{overview?.quality_metrics?.avg_data_quality || 0}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">EDA Generated</span>
                <span className="font-medium">{overview?.quality_metrics?.datasets_with_eda || 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Workflow Success</span>
                <span className="font-medium">{overview?.quality_metrics?.workflow_success_rate || 0}%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Recent Activity</CardTitle>
            <CardDescription>Latest platform activity</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm flex items-center">
                  <Database className="h-4 w-4 mr-2 text-blue-500" />
                  New Datasets
                </span>
                <Badge variant="secondary">{overview?.recent_activity?.new_datasets || 0}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm flex items-center">
                  <Brain className="h-4 w-4 mr-2 text-green-500" />
                  New Models
                </span>
                <Badge variant="secondary">{overview?.recent_activity?.new_models || 0}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm flex items-center">
                  <GitBranch className="h-4 w-4 mr-2 text-purple-500" />
                  New Workflows
                </span>
                <Badge variant="secondary">{overview?.recent_activity?.new_workflows || 0}</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">System Status</CardTitle>
            <CardDescription>Current system state</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Running Workflows</span>
                <Badge variant={overview?.status?.running_workflows > 0 ? "default" : "secondary"}>
                  {overview?.status?.running_workflows || 0}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Deployed Models</span>
                <Badge variant={overview?.status?.deployed_models > 0 ? "default" : "secondary"}>
                  {overview?.status?.deployed_models || 0}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Errors (24h)</span>
                <Badge variant={stats?.system?.error_count_24h > 0 ? "destructive" : "secondary"}>
                  {stats?.system?.error_count_24h || 0}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

