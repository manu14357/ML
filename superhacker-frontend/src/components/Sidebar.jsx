import { Link, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Database, 
  GitBranch, 
  Brain, 
  BarChart3, 
  Monitor,
  Menu,
  X,
  Activity,
  AlertCircle,
  CheckCircle,
  Moon,
  Sun,
  Settings
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useTheme } from '@/components/ThemeProvider'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Data Management', href: '/data', icon: Database },
  { name: 'Workflow Builder', href: '/workflows', icon: GitBranch },
  { name: 'ML Models', href: '/models', icon: Brain },
  { name: 'Visualizations', href: '/visualizations', icon: BarChart3 },
  { name: 'System Monitor', href: '/system', icon: Monitor },
]

export function Sidebar({ open, onToggle, systemHealth, isMobile }) {
  const location = useLocation()
  const { theme, setTheme } = useTheme()

  const getHealthIcon = () => {
    if (!systemHealth) return <Activity className="h-4 w-4 text-muted-foreground" />
    
    switch (systemHealth.status) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'unhealthy':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <Activity className="h-4 w-4 text-yellow-500" />
    }
  }

  const getHealthBadge = () => {
    if (!systemHealth) return null
    
    const variant = systemHealth.status === 'healthy' ? 'default' : 'destructive'
    return (
      <Badge variant={variant} className="text-xs">
        {systemHealth.status}
      </Badge>
    )
  }

  return (
    <div className={cn(
      "fixed left-0 top-0 h-full bg-sidebar border-r border-sidebar-border transition-all duration-300 z-50",
      isMobile && open ? "w-64" : isMobile ? "w-0" : open ? "w-64" : "w-16",
      isMobile && !open && "overflow-hidden"
    )}>
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className={cn(
          "flex h-16 items-center border-b border-sidebar-border",
          open ? "justify-between px-4" : "justify-center px-2"
        )}>
          {(open || !isMobile) && (
            <div className="flex items-center space-x-2">
              {open && (
                <span className="font-bold text-sidebar-foreground truncate">
                  SuperHacker
                </span>
              )}
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className={cn(
              "h-8 w-8 text-sidebar-foreground hover:bg-sidebar-accent flex-shrink-0",
              !open && !isMobile && "mx-auto"
            )}
          >
            {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-2 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <Link
                key={item.name}
                to={item.href}
                onClick={isMobile ? () => onToggle() : undefined}
                className={cn(
                  "flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                  isActive 
                    ? "bg-sidebar-primary text-sidebar-primary-foreground" 
                    : "text-sidebar-foreground",
                  !open && !isMobile && "justify-center px-2"
                )}
                title={!open ? item.name : undefined}
              >
                <item.icon className={cn(
                  "h-5 w-5 flex-shrink-0", 
                  open ? "mr-3" : ""
                )} />
                {open && <span className="truncate">{item.name}</span>}
              </Link>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="border-t border-sidebar-border p-4 space-y-2">
          {/* System Health */}
          <div className={cn(
            "flex items-center rounded-lg p-2 bg-sidebar-accent",
            (!open && !isMobile) && "justify-center"
          )}>
            {getHealthIcon()}
            {open && (
              <div className="ml-3 flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-sidebar-foreground">
                    System
                  </span>
                  {getHealthBadge()}
                </div>
                {systemHealth && (
                  <p className="text-xs text-sidebar-foreground/70 truncate">
                    {systemHealth.status === 'healthy' ? 'All services running' : 'Issues detected'}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Theme Toggle */}
          <div className="flex items-center justify-between">
            {open && (
              <span className="text-sm font-medium text-sidebar-foreground">
                Theme
              </span>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
              className="h-8 w-8 text-sidebar-foreground hover:bg-sidebar-accent flex-shrink-0"
            >
              {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

