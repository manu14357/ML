// Real data fetching utility for chart previews
export const fetchRealChartData = async (datasetId, chartConfig) => {
  const API_BASE = 'http://localhost:5000/api'
  
  try {
    // Show more detailed logging in development
    console.log('Fetching real data for chart:', chartConfig)
    
    const response = await fetch(`${API_BASE}/visualization/data/${datasetId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chart_type: chartConfig.type,
        x_column: chartConfig.xColumn,
        y_column: chartConfig.yColumn,
        color_column: chartConfig.colorColumn !== 'none' ? chartConfig.colorColumn : null,
        size_column: chartConfig.sizeColumn !== 'none' ? chartConfig.sizeColumn : null,
        sample_size: 200 // Limit for preview performance
      })
    })

    if (response.ok) {
      const responseText = await response.text()
      let result
      
      try {
        // Clean up any NaN values before parsing JSON
        const cleanedText = responseText
          .replace(/\bNaN\b/g, 'null')
          .replace(/\bInfinity\b/g, 'null')
          .replace(/\b-Infinity\b/g, 'null')
        
        result = JSON.parse(cleanedText)
        
        // Further clean the data to remove null values that were NaN
        if (result.data) {
          result.data = cleanDataValues(result.data)
        }
        
        // Log the cleaned data for debugging
        console.log('Cleaned chart data:', result.data)
        
        return result.data
      } catch (parseError) {
        console.error('JSON parsing error in fetchRealChartData:', parseError)
        console.error('Response text:', responseText)
        return null
      }
    } else {
      console.error('Failed to fetch real data', response.status, response.statusText)
      return null
    }
  } catch (error) {
    console.error('Error fetching real chart data:', error)
    return null
  }
}

// Helper function to clean data values
const cleanDataValues = (data) => {
  if (!data || typeof data !== 'object') return data
  
  const cleaned = { ...data }
  
  // Clean arrays by removing null values (that were originally NaN)
  const keysToClean = ['x', 'y', 'color', 'size', 'values', 'labels']
  keysToClean.forEach(key => {
    if (Array.isArray(cleaned[key])) {
      // Filter out null, undefined, NaN values
      cleaned[key] = cleaned[key].filter(val => 
        val !== null && 
        val !== undefined && 
        (typeof val !== 'number' || !Number.isNaN(val))
      )
      
      // Additional cleaning for numerical arrays (handle Infinity)
      if (cleaned[key].some(val => typeof val === 'number')) {
        cleaned[key] = cleaned[key].map(val => 
          typeof val === 'number' && !Number.isFinite(val) ? null : val
        )
      }
    }
  })
  
  // Clean nested structures like grouped_data for box plots
  if (cleaned.grouped_data && typeof cleaned.grouped_data === 'object') {
    Object.keys(cleaned.grouped_data).forEach(group => {
      if (Array.isArray(cleaned.grouped_data[group])) {
        cleaned.grouped_data[group] = cleaned.grouped_data[group].filter(val => 
          val !== null && 
          val !== undefined && 
          (typeof val !== 'number' || !Number.isNaN(val))
        )
        
        // Handle Infinity values
        cleaned.grouped_data[group] = cleaned.grouped_data[group].map(val => 
          typeof val === 'number' && !Number.isFinite(val) ? null : val
        )
      }
    })
  }
  
  // Clean 2D arrays like heatmap z data
  if (Array.isArray(cleaned.z) && cleaned.z.length > 0 && Array.isArray(cleaned.z[0])) {
    cleaned.z = cleaned.z.map(row => 
      row.map(val => 
        val === null || val === undefined || 
        (typeof val === 'number' && (Number.isNaN(val) || !Number.isFinite(val))) 
          ? 0 : val
      )
    )
  }
  
  return cleaned
}

// Generate Plotly config with real data
export const generateRealPlotlyConfig = (chartConfig, realData, colorSchemes) => {
  // eslint-disable-next-line no-unused-vars
  const { type, xColumn, yColumn, colorColumn, sizeColumn, title, colorScheme, height, width } = chartConfig
  
  if (!realData || !xColumn) return null

  const colors = colorSchemes[colorScheme] || colorSchemes.viridis

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
        x: realData.x || [],
        y: realData.y || [],
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: colors[0], width: 3 },
        marker: { 
          size: 6,
          color: realData.color || colors[0],
          colorscale: colorScheme === 'viridis' ? 'Viridis' : 'Plasma',
          showscale: !!(realData.color)
        }
      }]
      break

    case 'bar':
      // For bar charts, use actual category names from the data
      if (realData.labels && realData.values) {
        // Pie chart data format (for categorical data)
        plotData = [{
          x: realData.labels,
          y: realData.values,
          type: 'bar',
          marker: { color: colors }
        }]
      } else {
        // Regular x,y data
        plotData = [{
          x: realData.x || [],
          y: realData.y || [],
          type: 'bar',
          marker: { color: colors[0] }
        }]
      }
      break

    case 'scatter':
      plotData = [{
        x: realData.x || [],
        y: realData.y || [],
        mode: 'markers',
        type: 'scatter',
        marker: {
          size: realData.size || 8,
          color: realData.color || colors[0],
          colorscale: colorScheme === 'viridis' ? 'Viridis' : 'Plasma',
          showscale: !!(realData.color),
          opacity: 0.7
        }
      }]
      break

    case 'pie':
      if (realData.labels && realData.values) {
        plotData = [{
          values: realData.values,
          labels: realData.labels,
          type: 'pie',
          marker: { colors }
        }]
      }
      break

    case 'heatmap':
      if (realData.z && realData.x && realData.y) {
        plotData = [{
          z: realData.z,
          x: realData.x,
          y: realData.y,
          type: 'heatmap',
          colorscale: colorScheme === 'viridis' ? 'Viridis' : 'Plasma'
        }]
      }
      break

    case 'box':
      if (realData.grouped_data) {
        // Multiple box plots grouped by category
        plotData = Object.entries(realData.grouped_data).map(([group, values]) => ({
          y: values,
          type: 'box',
          name: group,
          boxpoints: 'outliers'
        }))
      } else {
        plotData = [{
          y: realData.y || [],
          type: 'box',
          name: yColumn,
          marker: { color: colors[0] }
        }]
      }
      break

    case 'histogram':
      plotData = [{
        x: realData.x || [],
        type: 'histogram',
        marker: { color: colors[0] },
        opacity: 0.7
      }]
      break

    case 'violin':
      plotData = [{
        y: realData.y || [],
        type: 'violin',
        name: yColumn,
        line: { color: colors[0] }
      }]
      break

    default:
      // Fallback to simple scatter
      plotData = [{
        x: realData.x || [],
        y: realData.y || [],
        type: 'scatter',
        mode: 'markers',
        marker: { color: colors[0] }
      }]
  }

  return { data: plotData, layout }
}
