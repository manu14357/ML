import { useState, useEffect, useRef, useMemo } from 'react'
import { 
  Upload, 
  Database, 
  FileText, 
  BarChart3, 
  Download,
  Eye,
  Trash2,
  Plus,
  Search,
  Filter,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
  X,
  FileUp,
  Loader2,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { toast } from 'sonner'
import Plotly from 'plotly.js-dist-min';
import { AdvancedEDAPanel } from './AdvancedEDAPanel';

export function DataManagement() {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [selectedDatasetId, setSelectedDatasetId] = useState(null) // Backup ID reference
  const [selectedEDAData, setSelectedEDAData] = useState(null)
  const [loadingEDA, setLoadingEDA] = useState(false)
  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [showEDAPanel, setShowEDAPanel] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  
  // Preview pagination state
  const [previewPage, setPreviewPage] = useState(1)
  const [previewPageSize, setPreviewPageSize] = useState(20)
  const [loadingMorePreview, setLoadingMorePreview] = useState(false)
  const [hasMorePreviewPages, setHasMorePreviewPages] = useState(true)
  
  // Column navigation state
  const [columnViewIndex, setColumnViewIndex] = useState(0)
  const maxColumnsToShow = 6 // Maximum number of columns to display at once
  
  // Get columns array from preview data
  const datasetColumns = useMemo(() => {
    if (selectedDataset?.preview?.[0]) {
      return Object.keys(selectedDataset.preview[0]);
    }
    return [];
  }, [selectedDataset?.preview]);
  
  // Debug effect to track selectedDataset changes
  useEffect(() => {
    console.log('🔍 selectedDataset changed:', {
      hasDataset: !!selectedDataset,
      datasetId: selectedDataset?.id,
      datasetName: selectedDataset?.name,
      timestamp: new Date().toISOString()
    })
    if (selectedDataset) {
      console.log('✅ selectedDataset full object:', selectedDataset)
      setSelectedDatasetId(selectedDataset.id) // Keep backup of ID
    } else {
      console.log('❌ selectedDataset is null/undefined')
      console.trace('Stack trace when selectedDataset becomes null:')
    }
  }, [selectedDataset])
  
  // Upload form state
  const [uploadForm, setUploadForm] = useState({
    file: null,
    name: '',
    description: '',
    generateEDA: true
  })
  
  const fileInputRef = useRef(null)

  useEffect(() => {
    fetchDatasets()
  }, [])

  // Diagnostic function to track dataset state
  const debugDatasetState = () => {
    console.log('🔍 Current Dataset State Diagnostic:', {
      datasets_length: datasets.length,
      selectedDataset_exists: !!selectedDataset,
      selectedDataset_id: selectedDataset?.id,
      selectedDatasetId_backup: selectedDatasetId,
      showEDAPanel: showEDAPanel,
      loadingEDA: loadingEDA,
      selectedEDAData_exists: !!selectedEDAData,
      datasets_sample: datasets.slice(0, 3).map(d => ({ id: d.id, name: d.name }))
    })
  }

  // Create a stable dataset reference for EDA
  const createStableDatasetRef = (dataset) => {
    if (!dataset) return null
    
    // Create a stable reference with all required fields
    return {
      id: dataset.id,
      name: dataset.name,
      description: dataset.description,
      file_type: dataset.file_type,
      file_size: dataset.file_size,
      rows_count: dataset.rows_count,
      columns_count: dataset.columns_count,
      status: dataset.status,
      created_at: dataset.created_at,
      eda_generated: dataset.eda_generated,
      data_quality_score: dataset.data_quality_score,
      // Include any other fields that might be needed
      ...dataset
    }
  }

  const fetchDatasets = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:5000/api/data/datasets')
      if (response.ok) {
        const data = await response.json()
        console.log('Datasets API response:', data)
        console.log('Individual datasets:', data.datasets?.map(d => ({
          id: d.id,
          name: d.name,
          allKeys: Object.keys(d)
        })))
        setDatasets(data.datasets || [])
      } else {
        toast.error('Failed to fetch datasets')
      }
    } catch (error) {
      console.error('Error fetching datasets:', error)
      toast.error('Error connecting to server')
    } finally {
      setLoading(false)
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }

  const handleFileSelect = (file) => {
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/json']
    const maxSize = 100 * 1024 * 1024 // 100MB
    
    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
      toast.error('File type not supported. Please upload CSV, Excel, or JSON files.')
      return
    }
    
    if (file.size > maxSize) {
      toast.error('File size too large. Maximum size is 100MB.')
      return
    }
    
    setUploadForm(prev => ({
      ...prev,
      file: file,
      name: file.name.replace(/\.[^/.]+$/, '') // Remove extension
    }))
    setShowUploadDialog(true)
  }

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!uploadForm.file) {
      toast.error('Please select a file to upload')
      return
    }

    setUploading(true)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', uploadForm.file)
      formData.append('name', uploadForm.name)
      formData.append('description', uploadForm.description)
      formData.append('generate_eda', uploadForm.generateEDA.toString())

      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90))
      }, 200)

      const response = await fetch('http://localhost:5000/api/data/upload', {
        method: 'POST',
        body: formData
      })

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (response.ok) {
        toast.success('Dataset uploaded and processed successfully!')
        setShowUploadDialog(false)
        setUploadForm({ file: null, name: '', description: '', generateEDA: true })
        fetchDatasets() // Refresh the list
      } else {
        const error = await response.json()
        toast.error(error.error || 'Upload failed')
      }
    } catch (error) {
      console.error('Upload error:', error)
      toast.error('Upload failed. Please try again.')
    } finally {
      setUploading(false)
      setUploadProgress(0)
    }
  }

  const handleDelete = async (datasetId) => {
    if (!confirm('Are you sure you want to delete this dataset?')) return

    try {
      const response = await fetch(`http://localhost:5000/api/data/datasets/${datasetId}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        toast.success('Dataset deleted successfully')
        fetchDatasets()
      } else {
        toast.error('Failed to delete dataset')
      }
    } catch (error) {
      console.error('Delete error:', error)
      toast.error('Error deleting dataset')
    }
  }

  const handlePreview = async (dataset) => {
    try {
      // Reset pagination when opening a new dataset preview
      setPreviewPage(1)
      setHasMorePreviewPages(true)
      setLoadingMorePreview(true)
      setColumnViewIndex(0) // Reset column view index
      
      const response = await fetch(`http://localhost:5000/api/data/datasets/${dataset.id}/preview?page=${1}&limit=${previewPageSize}`)
      if (response.ok) {
        const data = await response.json()
        
        // Ensure preview is an array before setting it
        let preview = data.preview;
        
        // Handle different potential response formats
        if (!Array.isArray(preview)) {
          console.warn('API preview data is not an array:', preview);
          
          // If it's an object with data property, try to use that
          if (preview && typeof preview === 'object' && Array.isArray(preview.data)) {
            preview = preview.data;
          } else if (preview && typeof preview === 'object') {
            // If it's just a single object, wrap it in an array
            preview = [preview];
          } else {
            // Otherwise set it to empty array to avoid errors
            preview = [];
            toast.warning("Preview data format is not compatible. Showing empty preview.");
          }
        }
        
        // Set hasMorePreviewPages based on API response or data length
        const total = data.total || dataset.rows_count || 0;
        setHasMorePreviewPages(preview.length >= previewPageSize && preview.length < total);
        
        setSelectedDataset({ ...dataset, preview });
      } else {
        // Handle error response
        let errorMessage = 'Failed to load dataset preview';
        try {
          const errorData = await response.json();
          if (errorData && errorData.error) {
            errorMessage = errorData.error;
          }
        } catch {
          // If we can't parse error JSON, use status text
          errorMessage = response.statusText || errorMessage;
        }
        
        toast.error(errorMessage);
        console.error('Preview API error:', response.status, errorMessage);
        
        // Still set the dataset but with empty preview
        setSelectedDataset({ ...dataset, preview: [] });
      }
    } catch (error) {
      console.error('Preview error:', error)
      toast.error('Error loading preview: ' + (error.message || 'Unknown error'))
      setSelectedDataset({ ...dataset, preview: [] });
    } finally {
      setLoadingMorePreview(false)
    }
  }

  const handleLoadMorePreview = async (datasetId) => {
    if (!datasetId || loadingMorePreview) return;
    
    try {
      setLoadingMorePreview(true);
      
      // If we don't have a dataset yet, do a fresh preview load
      if (!selectedDataset) {
        const dataset = datasets.find(d => d.id === datasetId);
        if (dataset) {
          return handlePreview(dataset);
        }
        return;
      }
      
      const response = await fetch(
        `http://localhost:5000/api/data/datasets/${datasetId}/preview?page=${previewPage}&limit=${previewPageSize}`
      );
      
      if (response.ok) {
        const data = await response.json();
        
        // Ensure preview is an array
        let preview = data.preview;
        if (!Array.isArray(preview)) {
          console.warn('API preview data is not an array on page load:', preview);
          if (preview && typeof preview === 'object' && Array.isArray(preview.data)) {
            preview = preview.data;
          } else if (preview && typeof preview === 'object') {
            preview = [preview];
          } else {
            preview = [];
            toast.warning("Retrieved preview data has an unexpected format. Some data may not be displayed correctly.");
          }
        }
        
        // Update pagination state
        const total = data.total || selectedDataset.rows_count || 0;
        setHasMorePreviewPages(preview.length >= previewPageSize && (previewPage * previewPageSize) < total);
        
        // Append to or replace existing preview data based on the page
        if (previewPage === 1) {
          setSelectedDataset(prev => ({ ...prev, preview }));
          setColumnViewIndex(0); // Reset column view index when getting first page
        } else {
          setSelectedDataset(prev => ({ 
            ...prev, 
            preview: prev.preview ? [...prev.preview, ...preview] : preview 
          }));
        }
      } else {
        // Handle error response
        let errorMessage = 'Failed to load more preview data';
        try {
          const errorData = await response.json();
          if (errorData && errorData.error) {
            errorMessage = errorData.error;
          }
        } catch {
          // If we can't parse error JSON, use status text
          errorMessage = response.statusText || errorMessage;
        }
        
        toast.error(errorMessage);
        console.error('Load more preview API error:', response.status, errorMessage);
      }
    } catch (error) {
      console.error('Load more preview error:', error);
      toast.error('Error loading more preview data: ' + (error.message || 'Unknown error'));
    } finally {
      setLoadingMorePreview(false);
    }
  }

  const handleViewEDA = async (dataset) => {
    try {
      console.log('🔍 handleViewEDA called with dataset:', dataset) // Debug log
      console.log('🔍 Dataset ID check:', {
        hasId: !!dataset.id,
        id: dataset.id,
        type: typeof dataset.id,
        allKeys: Object.keys(dataset),
        datasetName: dataset.name,
        datasetStatus: dataset.status
      })
      console.log('🔍 Full dataset object:', JSON.stringify(dataset, null, 2))
      
      // Validate dataset has required fields
      if (!dataset || !dataset.id) {
        throw new Error('Invalid dataset: missing ID')
      }

      // Enhanced dataset ID validation
      const datasetId = dataset.id
      if (!datasetId || datasetId === null || datasetId === undefined || datasetId === '') {
        console.error('❌ Dataset ID validation failed:', {
          id: dataset.id,
          type: typeof dataset.id,
          stringified: JSON.stringify(dataset.id)
        })
        throw new Error('Dataset ID is invalid or empty')
      }
      
      console.log('✅ Dataset ID validated successfully:', datasetId)
      
      // Create stable dataset reference with guaranteed ID
      const stableDataset = createStableDatasetRef(dataset)
      
      // Double-check the stable reference has the ID
      if (!stableDataset.id) {
        console.error('❌ Stable dataset reference missing ID')
        throw new Error('Failed to create stable dataset reference with valid ID')
      }
      
      console.log('✅ Setting selectedDataset to stable reference:', stableDataset)
      setSelectedDataset(stableDataset) // Set the selected dataset first
      setLoadingEDA(true)
      setShowEDAPanel(true) // Show panel immediately to provide feedback
      
      const response = await fetch(`http://localhost:5000/api/data/datasets/${dataset.id}/eda`)
      const data = await response.json()
      
      console.log('EDA API response:', { status: response.status, data }) // Debug log
      
      if (response.ok) {
        setSelectedEDAData(data)
      } else if (response.status === 404 && data.can_generate) {
        // Show panel with option to generate EDA
        setSelectedEDAData(null)
      } else {
        throw new Error(data.error || 'Failed to load EDA data')
      }
    } catch (error) {
      console.error('EDA error:', error)
      toast.error(error.message || 'Error loading EDA data')
      // Keep panel open but clear EDA data to show error state
      setSelectedEDAData(null)
    } finally {
      setLoadingEDA(false)
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'processing':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status) => {
    const variants = {
      ready: 'default',
      processing: 'secondary',
      error: 'destructive'
    }
    return <Badge variant={variants[status] || 'secondary'}>{status}</Badge>
  }

  const filteredDatasets = datasets.filter(dataset =>
    dataset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    dataset.description?.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const totalSize = datasets.reduce((sum, dataset) => sum + (dataset.file_size || 0), 0)
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const edaCount = datasets.filter(d => d.eda_generated).length
  const processingCount = datasets.filter(d => d.status === 'processing').length

  return (
    <div className={`p-3 sm:p-4 md:p-6 space-y-4 sm:space-y-6 ${showEDAPanel ? 'pb-[70vh]' : ''}`}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Data Management</h1>
          <p className="text-sm sm:text-base text-muted-foreground">
            Upload, process, and manage your datasets with automatic EDA
          </p>
        </div>
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={debugDatasetState}
            title="Debug Dataset State"
            className="w-full sm:w-auto"
          >
            🔍 Debug
          </Button>
          <Button onClick={() => setShowUploadDialog(true)} className="w-full sm:w-auto">
            <Plus className="h-4 w-4 mr-2" />
            Upload Dataset
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Datasets</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{datasets.length}</div>
            <p className="text-xs text-muted-foreground">
              {datasets.filter(d => {
                const today = new Date()
                const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000)
                return new Date(d.created_at) > weekAgo
              }).length} added this week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Size</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatFileSize(totalSize)}</div>
            <p className="text-xs text-muted-foreground">
              Across all datasets
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">EDA Generated</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{edaCount}</div>
            <p className="text-xs text-muted-foreground">
              Automatic analysis ready
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing</CardTitle>
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{processingCount}</div>
            <p className="text-xs text-muted-foreground">
              Currently processing
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Upload Area */}
      <Card 
        className={`border-dashed border-2 transition-colors ${
          dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Upload className={`h-12 w-12 mb-4 ${dragActive ? 'text-primary' : 'text-muted-foreground'}`} />
          <h3 className="text-lg font-semibold mb-2">Upload Your Dataset</h3>
          <p className="text-muted-foreground text-center mb-4">
            Drag and drop your CSV, Excel, or JSON files here, or click to browse
          </p>
          <Button onClick={() => fileInputRef.current?.click()}>
            <Upload className="h-4 w-4 mr-2" />
            Choose Files
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".csv,.xlsx,.xls,.json"
            onChange={handleFileInputChange}
          />
          <p className="text-xs text-muted-foreground mt-2">
            Supports CSV, Excel, JSON files up to 100MB
          </p>
        </CardContent>
      </Card>

      {/* Datasets Table */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <CardTitle>Your Datasets</CardTitle>
              <CardDescription>Manage and analyze your uploaded data</CardDescription>
            </div>
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
              <Input 
                placeholder="Search datasets..." 
                className="w-full sm:w-64"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              <Button variant="outline" size="icon" onClick={fetchDatasets} className="w-full sm:w-auto">
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead className="hidden md:table-cell">Type</TableHead>
                  <TableHead className="hidden lg:table-cell">Size</TableHead>
                <TableHead>Rows</TableHead>
                <TableHead>Columns</TableHead>
                <TableHead>Quality</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={9} className="text-center py-8">
                    <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2" />
                    Loading datasets...
                  </TableCell>
                </TableRow>
              ) : filteredDatasets.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={9} className="text-center py-8 text-muted-foreground">
                    {searchTerm ? 'No datasets match your search.' : 'No datasets found. Upload your first dataset to get started.'}
                  </TableCell>
                </TableRow>
              ) : (
                filteredDatasets.map((dataset) => (
                  <TableRow key={dataset.id}>
                    <TableCell className="font-medium">{dataset.name}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{dataset.file_type?.toUpperCase()}</Badge>
                    </TableCell>
                    <TableCell>{formatFileSize(dataset.file_size || 0)}</TableCell>
                    <TableCell>{dataset.rows_count?.toLocaleString() || '-'}</TableCell>
                    <TableCell>{dataset.columns_count || '-'}</TableCell>
                    <TableCell>
                      {dataset.data_quality_score ? (
                        <div className="flex items-center space-x-2">
                          <Progress value={dataset.data_quality_score} className="w-16 h-2" />
                          <span className="text-sm">{Math.round(dataset.data_quality_score)}%</span>
                        </div>
                      ) : '-'}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(dataset.status)}
                        {getStatusBadge(dataset.status)}
                      </div>
                    </TableCell>
                    <TableCell>{new Date(dataset.created_at).toLocaleDateString()}</TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <Button variant="ghost" size="icon" onClick={() => handlePreview(dataset)}>
                          <Eye className="h-4 w-4" />
                        </Button>
                        {dataset.eda_generated && (                        <Button 
                          variant="ghost" 
                          size="icon"
                          onClick={() => handleViewEDA(dataset)}
                          disabled={loadingEDA}
                          title="View EDA Analysis"
                        >
                          {loadingEDA ? (
                            <RefreshCw className="h-4 w-4 animate-spin" />
                          ) : (
                            <BarChart3 className="h-4 w-4" />
                          )}
                        </Button>
                        )}
                        <Button variant="ghost" size="icon">
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="text-destructive"
                          onClick={() => handleDelete(dataset.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
          </div>
        </CardContent>
      </Card>

      {/* Upload Dialog */}
      <Dialog open={showUploadDialog} onOpenChange={setShowUploadDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Upload Dataset</DialogTitle>
            <DialogDescription>
              Configure your dataset upload and processing options.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            {uploadForm.file && (
              <div className="flex items-center space-x-2 p-2 bg-muted rounded">
                <FileUp className="h-4 w-4" />
                <span className="text-sm">{uploadForm.file.name}</span>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-6 w-6"
                  onClick={() => setUploadForm(prev => ({ ...prev, file: null }))}
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            )}
            
            <div className="grid gap-2">
              <Label htmlFor="name">Dataset Name</Label>
              <Input
                id="name"
                value={uploadForm.name}
                onChange={(e) => setUploadForm(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Enter dataset name"
              />
            </div>
            
            <div className="grid gap-2">
              <Label htmlFor="description">Description (Optional)</Label>
              <Textarea
                id="description"
                value={uploadForm.description}
                onChange={(e) => setUploadForm(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Describe your dataset"
                rows={3}
              />
            </div>
            
            <div className="flex items-center space-x-2">
              <Switch
                id="generate-eda"
                checked={uploadForm.generateEDA}
                onCheckedChange={(checked) => setUploadForm(prev => ({ ...prev, generateEDA: checked }))}
              />
              <Label htmlFor="generate-eda">Generate automatic EDA</Label>
            </div>
            
            {uploading && (
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm">Uploading and processing...</span>
                </div>
                <Progress value={uploadProgress} className="w-full" />
              </div>
            )}
          </div>
          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={() => setShowUploadDialog(false)} disabled={uploading}>
              Cancel
            </Button>
            <Button onClick={handleUpload} disabled={!uploadForm.file || uploading}>
              {uploading ? 'Uploading...' : 'Upload'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Dataset Preview Dialog */}
      {selectedDataset && !showEDAPanel && (
        <Dialog 
          open={!!selectedDataset && !showEDAPanel} 
          onOpenChange={(open) => {
            if (!open) {
              setSelectedDataset(null);
              setPreviewPage(1);
              setHasMorePreviewPages(true);
              setColumnViewIndex(0); // Reset column view
            }
          }}
        >
          <DialogContent className="sm:max-w-[90vw] lg:max-w-[80vw] max-h-[85vh] w-full">
            <DialogHeader>
              <DialogTitle>{selectedDataset.name}</DialogTitle>
              <DialogDescription>
                Dataset preview and information
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Rows</Label>
                  <p className="text-sm text-muted-foreground">{selectedDataset.rows_count?.toLocaleString()}</p>
                </div>
                <div>
                  <Label>Columns</Label>
                  <p className="text-sm text-muted-foreground">{selectedDataset.columns_count}</p>
                </div>
                <div>
                  <Label>Size</Label>
                  <p className="text-sm text-muted-foreground">{formatFileSize(selectedDataset.file_size)}</p>
                </div>
                <div>
                  <Label>Quality Score</Label>
                  <p className="text-sm text-muted-foreground">{Math.round(selectedDataset.data_quality_score || 0)}%</p>
                </div>
              </div>
              
              {selectedDataset.preview && Array.isArray(selectedDataset.preview) && selectedDataset.preview.length > 0 ? (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <Label className="text-lg font-medium">Data Preview</Label>
                    <div className="flex items-center space-x-2">
                      {datasetColumns.length > maxColumnsToShow && (
                        <div className="flex items-center mr-2 bg-muted rounded-md px-3 py-1">
                          <Button 
                            variant="ghost" 
                            size="icon"
                            disabled={columnViewIndex === 0}
                            onClick={() => setColumnViewIndex(0)}
                            title="First columns"
                          >
                            <ChevronLeft className="h-4 w-4 mr-1" />
                            <ChevronLeft className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="icon"
                            disabled={columnViewIndex === 0}
                            onClick={() => setColumnViewIndex(prev => Math.max(prev - maxColumnsToShow, 0))}
                            title="Previous columns"
                          >
                            <ChevronLeft className="h-4 w-4" />
                          </Button>
                          <span className="text-sm mx-2">
                            {columnViewIndex + 1}-{Math.min(columnViewIndex + maxColumnsToShow, datasetColumns.length)} of {datasetColumns.length}
                          </span>
                          <Button 
                            variant="ghost" 
                            size="icon"
                            disabled={columnViewIndex + maxColumnsToShow >= datasetColumns.length}
                            onClick={() => setColumnViewIndex(prev => Math.min(prev + maxColumnsToShow, datasetColumns.length - maxColumnsToShow))}
                            title="Next columns"
                          >
                            <ChevronRight className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="icon"
                            disabled={columnViewIndex + maxColumnsToShow >= datasetColumns.length}
                            onClick={() => setColumnViewIndex(Math.max(datasetColumns.length - maxColumnsToShow, 0))}
                            title="Last columns"
                          >
                            <ChevronRight className="h-4 w-4 mr-1" />
                            <ChevronRight className="h-4 w-4" />
                          </Button>
                        </div>
                      )}
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => {
                          if (selectedDataset?.id) {
                            // Keep current page, just reload the data
                            handleLoadMorePreview(selectedDataset.id);
                          }
                        }}
                        disabled={loadingMorePreview}
                      >
                        {loadingMorePreview ? (
                          <><RefreshCw className="h-4 w-4 mr-2 animate-spin" /> Loading...</>
                        ) : (
                          <>Refresh Data</>
                        )}
                      </Button>
                      <Select 
                        value={previewPageSize.toString()}
                        onValueChange={(value) => {
                          const newSize = parseInt(value);
                          setPreviewPageSize(newSize);
                          setPreviewPage(1); // Reset to first page
                          if (selectedDataset?.id) {
                            setTimeout(() => {
                              handleLoadMorePreview(selectedDataset.id);
                            }, 0);
                          }
                        }}
                      >
                        <SelectTrigger className="w-[130px]">
                          <SelectValue placeholder="Rows per page" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="10">10 rows</SelectItem>
                          <SelectItem value="20">20 rows</SelectItem>
                          <SelectItem value="50">50 rows</SelectItem>
                          <SelectItem value="100">100 rows</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <div className="mt-2 border rounded-md shadow-sm overflow-hidden">
                    <div className="overflow-auto max-h-[40vh]">
                      <Table className="min-w-full table-fixed">
                        <TableHeader className="bg-muted sticky top-0 z-10">
                          <TableRow>
                            {/* Column navigation buttons */}
                            {columnViewIndex > 0 && (
                              <TableHead className="w-10 p-0 bg-primary/10 border-r">
                                <Button 
                                  variant="ghost" 
                                  size="icon"
                                  className="h-full w-full rounded-none"
                                  onClick={() => setColumnViewIndex(prev => Math.max(prev - maxColumnsToShow, 0))}
                                >
                                  <ChevronLeft className="h-4 w-4" />
                                </Button>
                              </TableHead>
                            )}
                            
                            {/* Visible columns */}
                            {datasetColumns.slice(columnViewIndex, columnViewIndex + maxColumnsToShow).map(col => (
                              <TableHead key={col} className="py-3 px-4 font-semibold text-sm whitespace-nowrap">
                                {col}
                              </TableHead>
                            ))}
                            
                            {/* More columns indicator */}
                            {columnViewIndex + maxColumnsToShow < datasetColumns.length && (
                              <TableHead className="w-10 p-0 bg-primary/10 border-l">
                                <Button 
                                  variant="ghost" 
                                  size="icon"
                                  className="h-full w-full rounded-none"
                                  onClick={() => setColumnViewIndex(prev => Math.min(prev + maxColumnsToShow, datasetColumns.length - maxColumnsToShow))}
                                >
                                  <ChevronRight className="h-4 w-4" />
                                </Button>
                              </TableHead>
                            )}
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {selectedDataset.preview.map((row, idx) => (
                            <TableRow key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-muted/30'}>
                              {/* Column navigation for rows */}
                              {columnViewIndex > 0 && (
                                <TableCell className="w-10 p-0 bg-primary/5 border-r">
                                  {/* Spacer cell */}
                                </TableCell>
                              )}
                              
                              {/* Visible row data */}
                              {datasetColumns.slice(columnViewIndex, columnViewIndex + maxColumnsToShow).map((col, colIdx) => (
                                <TableCell key={colIdx} className="py-2 px-4 text-sm border-b">
                                  {row[col] === null || row[col] === undefined ? 
                                    <span className="text-muted-foreground italic">null</span> : 
                                    String(row[col])}
                                </TableCell>
                              ))}
                              
                              {/* More columns indicator for rows */}
                              {columnViewIndex + maxColumnsToShow < datasetColumns.length && (
                                <TableCell className="w-10 p-0 bg-primary/5 border-l">
                                  {/* Spacer cell */}
                                </TableCell>
                              )}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between mt-2">
                    <div className="flex flex-col space-y-1">
                      <p className="text-sm text-muted-foreground">
                        Showing {selectedDataset.preview.length} of {selectedDataset.rows_count?.toLocaleString() || '?'} rows
                      </p>
                      {datasetColumns.length > maxColumnsToShow && (
                        <p className="text-sm text-muted-foreground">
                          Showing columns {columnViewIndex + 1}-{Math.min(columnViewIndex + maxColumnsToShow, datasetColumns.length)} of {datasetColumns.length} 
                          <Button 
                            variant="link" 
                            size="sm" 
                            className="h-auto p-0 mx-1 text-primary" 
                            onClick={() => setColumnViewIndex(0)}
                          >
                            Reset view
                          </Button>
                        </p>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => {
                          const newPage = Math.max(previewPage - 1, 1);
                          setPreviewPage(newPage);
                          if (selectedDataset?.id) {
                            handleLoadMorePreview(selectedDataset.id);
                          }
                        }}
                        disabled={previewPage <= 1 || loadingMorePreview}
                      >
                        Previous
                      </Button>
                      <span className="text-sm mx-2">
                        Page {previewPage}
                      </span>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => {
                          const newPage = previewPage + 1;
                          setPreviewPage(newPage);
                          if (selectedDataset?.id) {
                            handleLoadMorePreview(selectedDataset.id);
                          }
                        }}
                        disabled={!hasMorePreviewPages || loadingMorePreview}
                      >
                        Next
                      </Button>
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <Label className="text-lg font-medium">Data Preview</Label>
                  <div className="mt-2 p-8 border rounded text-center text-muted-foreground bg-muted/20">
                    {loadingMorePreview ? (
                      <div className="flex flex-col items-center p-8">
                        <RefreshCw className="h-8 w-8 animate-spin mb-4 text-primary" />
                        <p className="text-lg font-medium">Loading preview data...</p>
                        <p className="text-muted-foreground mt-1">This may take a moment for large datasets</p>
                      </div>
                    ) : selectedDataset.preview === null ? (
                      <div className="flex flex-col items-center p-8">
                        <FileText className="h-12 w-12 mb-4 text-muted-foreground" />
                        <p className="text-lg font-medium">No preview available for this dataset.</p>
                        <p className="text-muted-foreground mt-1 mb-3">The preview data hasn't been loaded yet.</p>
                        <Button 
                          className="mt-2"
                          onClick={() => {
                            if (selectedDataset?.id) {
                              setPreviewPage(1);
                              handleLoadMorePreview(selectedDataset.id);
                            }
                          }}
                        >
                          Load Preview Data
                        </Button>
                      </div>
                    ) : Array.isArray(selectedDataset.preview) && selectedDataset.preview.length === 0 ? (
                      <div className="flex flex-col items-center p-8">
                        <AlertCircle className="h-12 w-12 mb-4 text-muted-foreground" />
                        <p className="text-lg font-medium">No preview data available</p>
                        <p className="text-muted-foreground mt-1">Unable to display preview for this dataset.</p>
                      </div>
                    ) : null}
                  </div>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      )}

      {/* Advanced EDA Results Panel */}
      {showEDAPanel && (
        <AdvancedEDAPanel
          edaData={selectedEDAData}
          loading={loadingEDA}
          dataset={selectedDataset}
          onClose={() => {
            console.log('🔍 AdvancedEDAPanel onClose called - clearing state')
            debugDatasetState()
            setShowEDAPanel(false);
            setLoadingEDA(false);
            setSelectedEDAData(null);
            setSelectedDataset(null);
            setSelectedDatasetId(null);
          }}
          onRefresh={() => {
            console.log('🔍 AdvancedEDAPanel onRefresh called with selectedDataset:', selectedDataset)
            debugDatasetState()
            if (selectedDataset) {
              handleViewEDA(selectedDataset);
            } else {
              console.error('❌ Cannot refresh - selectedDataset is null')
              // Try to find the dataset from the datasets list using backup ID
              if (selectedDatasetId) {
                console.log('🔍 Attempting to recover dataset using backup ID:', selectedDatasetId)
                const foundDataset = datasets.find(d => d.id === selectedDatasetId)
                if (foundDataset) {
                  console.log('✅ Recovered dataset:', foundDataset)
                  const stableDataset = createStableDatasetRef(foundDataset)
                  setSelectedDataset(stableDataset)
                  handleViewEDA(stableDataset)
                } else {
                  console.error('❌ Could not find dataset with ID:', selectedDatasetId)
                }
              }
            }
          }}
        />
      )}
    </div>
  )
}
