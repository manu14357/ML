import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Sidebar } from '@/components/Sidebar'
import { Dashboard } from '@/components/Dashboard'
import { DataManagement } from '@/components/DataManagement'
import { AdvancedWorkflowBuilder } from '@/components/AdvancedWorkflowBuilder'
import { MLModels } from '@/components/MLModels'
import { ModelDetailsPage } from '@/components/ModelDetailsPage'
import { Visualizations } from '@/components/Visualizations'
import { SystemMonitor } from '@/components/SystemMonitor'
import { Toaster } from '@/components/ui/toaster'
import { ThemeProvider } from '@/components/ThemeProvider'
import './App.css'
import './responsive.css'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [systemHealth, setSystemHealth] = useState(null)
  const [isMobile, setIsMobile] = useState(false)

  // Check system health on app load
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/system/health')
        const data = await response.json()
        setSystemHealth(data)
      } catch (error) {
        console.error('Failed to check system health:', error)
        setSystemHealth({ status: 'unhealthy', error: 'Backend not available' })
      }
    }

    checkHealth()
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  // Handle responsive behavior
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 1024
      setIsMobile(mobile)
      
      // Auto-close sidebar on mobile and tablets
      if (mobile) {
        setSidebarOpen(false)
      } else {
        // Auto-open sidebar on desktop screens
        setSidebarOpen(true)
      }
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return (
    <ThemeProvider defaultTheme="light" storageKey="superhacker-theme">
      <Router>
        <div className="relative flex h-screen bg-background">
          {/* Sidebar */}
          <Sidebar 
            open={sidebarOpen} 
            onToggle={() => setSidebarOpen(!sidebarOpen)}
            systemHealth={systemHealth}
            isMobile={isMobile}
          />
          
          {/* Mobile overlay */}
          {isMobile && sidebarOpen && (
            <div 
              className="fixed inset-0 bg-black/50 z-40 lg:hidden"
              onClick={() => setSidebarOpen(false)}
            />
          )}
          
          {/* Main content */}
          <main className={`flex-1 overflow-hidden transition-all duration-300 ${
            isMobile 
              ? 'ml-0' 
              : sidebarOpen 
                ? 'ml-64' 
                : 'ml-16'
          }`}>
            <div className="h-full overflow-auto">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/data" element={<DataManagement />} />
                <Route path="/workflows" element={<AdvancedWorkflowBuilder />} />
                <Route path="/models" element={<MLModels />} />
                <Route path="/models/:modelId" element={<ModelDetailsPage />} />
                <Route path="/ml-models/:modelId" element={<ModelDetailsPage />} />
                <Route path="/visualizations" element={<Visualizations />} />
                <Route path="/system" element={<SystemMonitor />} />
              </Routes>
            </div>
          </main>
          
          <Toaster />
        </div>
      </Router>
    </ThemeProvider>
  )
}

export default App

