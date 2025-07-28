import React from 'react';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

const PlotlyChart = ({ 
  data, 
  layout = {}, 
  config = {},
  onInitialized,
  onError,
  loading = false
}) => {
  try {
    // Handle different data formats
    let plotData = data;
    
    if (typeof data === 'string') {
      try {
        plotData = JSON.parse(data);
      } catch (e) {
        console.error('Failed to parse chart data:', e);
        onError?.('Invalid chart data format');
        return <div className="text-red-500 p-4">Error: Invalid chart data format</div>;
      }
    }
    
    // Ensure we have valid data
    if (!plotData) {
      return <div className="text-gray-500 p-4">No data available for chart</div>;
    }
    
    // Handle loading state
    if (loading) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      );
    }

    // Prepare layout with defaults
    const chartLayout = {
      autosize: true,
      margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 },
      showlegend: true,
      hovermode: 'closest',
      ...(plotData.layout || {}),
      ...layout
    };

    // Prepare data
    const chartData = Array.isArray(plotData) ? plotData : 
                      (plotData.data || []);

    return (
      <div className="w-full h-full min-h-[400px]">
        <Plot
          data={chartData}
          layout={chartLayout}
          config={{
            responsive: true,
            displayModeBar: true,
            scrollZoom: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['select2d', 'lasso2d'],
            ...config
          }}
          onInitialized={onInitialized}
          onError={(err) => {
            console.error('Plotly error:', err);
            onError?.(err);
          }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </div>
    );
  } catch (error) {
    console.error('Error in PlotlyChart:', error);
    return (
      <div className="text-red-500 p-4 border border-red-200 bg-red-50 rounded">
        <p>Failed to render chart: {error.message}</p>
      </div>
    );
  }
};

export default PlotlyChart;
