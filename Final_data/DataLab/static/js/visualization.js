/**
 * DataLab Visualization Module
 * Provides interactive data visualization capabilities
 */

// Initialize visualization module when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log('Visualization module loaded');
  
  // Global variables
  const datasetId = document.querySelector('.dataset-info-card') ? 
    parseInt(document.querySelector('.dataset-info-card').dataset.datasetId) : null;
  
  let datasetColumns = [];
  let datasetData = null;
  let dashboardItems = [];
  let currentTab = 'auto-viz';
  
  // Initialize visualization components
  initTabs();
  initAutoViz();
  initEDA();
  initCorrelationAnalysis();
  initAnomalyDetection();
  initCustomViz();
  initDashboard();
  initTooltips();
  
  // Fetch dataset metadata and columns
  if (datasetId) {
    fetchDatasetInfo(datasetId);
  }
  
  /**
   * Initialize tab switching
   */
  function initTabs() {
    const tabButtons = document.querySelectorAll('#vizTabs button');
    
    tabButtons.forEach(button => {
      button.addEventListener('click', function(e) {
        currentTab = this.id.replace('-tab', '');
        console.log('Switched to tab:', currentTab);
        
        // Load tab-specific data if needed
        if (currentTab === 'eda' && datasetColumns && !document.querySelector('#univariateFeature').options.length > 1) {
          populateEdaFeatures();
        } else if (currentTab === 'correlation' && !document.querySelector('#correlationMatrix').hasChildNodes()) {
          populateFeatureSelection();
        } else if (currentTab === 'anomaly' && !document.querySelector('#anomalyFeature').options.length > 1) {
          populateAnomalyFeatures();
        } else if (currentTab === 'custom-viz' && !document.querySelector('#xAxis').options.length > 1) {
          populateChartFeatures();
        }
      });
    });
    
    // Initialize create first visualization button
    const createFirstVizBtn = document.getElementById('createFirstVizBtn');
    if (createFirstVizBtn) {
      createFirstVizBtn.addEventListener('click', function() {
        document.getElementById('custom-viz-tab').click();
      });
    }
  }
  
  /**
   * Fetch dataset information and columns
   */
  function fetchDatasetInfo(datasetId) {
    showLoading(true);
    
    fetch(`/api/dataset/${datasetId}/info`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        console.log('Dataset info received:', data);
        datasetColumns = data.columns;
        
        // Populate all feature selection dropdowns
        populateFeatureSelection();
        populateEdaFeatures();
        populateAnomalyFeatures();
        populateChartFeatures();
        
        showLoading(false);
      })
      .catch(error => {
        console.error('Error fetching dataset info:', error);
        showError('Failed to load dataset information. Please try again.');
        showLoading(false);
      });
  }
  
  /**
   * Fetch dataset data for visualization
   */
  function fetchDatasetData(datasetId, limit = 1000) {
    showLoading(true);
    
    fetch(`/api/dataset/${datasetId}/data?limit=${limit}`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        console.log('Dataset data received, rows:', data.length);
        datasetData = data;
        
        // If auto-viz tab is active, generate visualizations
        if (currentTab === 'auto-viz') {
          generateAutoVisualizations();
        }
        
        showLoading(false);
      })
      .catch(error => {
        console.error('Error fetching dataset data:', error);
        showError('Failed to load dataset data. Please try again.');
        showLoading(false);
      });
  }
  
  /**
   * Show/hide loading indicator
   */
  function showLoading(show) {
    const loadingElements = document.querySelectorAll('.viz-loading');
    
    loadingElements.forEach(el => {
      if (show) {
        el.style.display = 'flex';
      } else {
        el.style.display = 'none';
      }
    });
  }
  
  /**
   * Show error message
   */
  function showError(message) {
    // Implement an error toast or notification
    console.error('ERROR:', message);
    alert(message); // Simple fallback
  }
  
  /**
   * Detect data type of a column
   */
  function detectColumnType(column) {
    if (!datasetData || datasetData.length === 0) return 'unknown';
    
    const sample = datasetData.slice(0, 100);
    const values = sample.map(row => row[column]).filter(val => val !== null && val !== undefined);
    
    if (values.length === 0) return 'unknown';
    
    // Check if all values are numeric
    const numericValues = values.filter(val => !isNaN(parseFloat(val)) && isFinite(val));
    if (numericValues.length === values.length) return 'numeric';
    
    // Check for date values
    const datePattern = /^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}[-/]\d{1,2}[-/]\d{4}/;
    const dateValues = values.filter(val => datePattern.test(String(val)));
    if (dateValues.length > values.length * 0.8) return 'date';
    
    // Check for categorical values
    const uniqueValues = new Set(values);
    if (uniqueValues.size <= Math.min(10, values.length * 0.2)) return 'categorical';
    
    // Default to text
    return 'text';
  }
  
  /**
   * Initialize tooltips for visualization explanations
   */
  function initTooltips() {
    const vizTooltip = document.getElementById('vizTooltip');
    const closeVizTooltip = document.getElementById('closeVizTooltip');
    
    if (closeVizTooltip) {
      closeVizTooltip.addEventListener('click', function() {
        vizTooltip.classList.remove('show');
      });
    }
    
    // Close tooltip when clicking outside
    document.addEventListener('click', function(event) {
      if (vizTooltip && vizTooltip.classList.contains('show')) {
        if (!vizTooltip.contains(event.target) && !event.target.classList.contains('viz-explanation-btn')) {
          vizTooltip.classList.remove('show');
        }
      }
    });
  }
  
  /**
   * Show explanation tooltip for a visualization
   */
  function showVizExplanation(title, explanation, interpretation, insights) {
    const vizTooltip = document.getElementById('vizTooltip');
    const vizExplanation = document.getElementById('vizExplanation');
    const vizInterpretation = document.getElementById('vizInterpretation');
    const vizInsightsList = document.getElementById('vizInsightsList');
    
    if (vizTooltip && vizExplanation && vizInterpretation && vizInsightsList) {
      // Update tooltip content
      document.querySelector('.viz-tooltip-header h6').textContent = title;
      vizExplanation.textContent = explanation;
      vizInterpretation.textContent = interpretation;
      
      // Update insights list
      vizInsightsList.innerHTML = '';
      if (Array.isArray(insights)) {
        insights.forEach(insight => {
          const li = document.createElement('li');
          li.textContent = insight;
          vizInsightsList.appendChild(li);
        });
      }
      
      // Show tooltip
      vizTooltip.classList.add('show');
    }
  }
  
  // Add event listener for export dashboard button
  const exportDashboardBtn = document.getElementById('exportDashboardBtn');
  if (exportDashboardBtn) {
    exportDashboardBtn.addEventListener('click', exportDashboard);
  }
  
  /**
   * Export dashboard to PNG
   */
  function exportDashboard() {
    const dashboardGrid = document.getElementById('dashboardGrid');
    
    if (!dashboardGrid || dashboardItems.length === 0) {
      showError('No visualizations to export');
      return;
    }
    
    // Implementation would use html2canvas or similar library
    alert('Dashboard export functionality will be implemented with html2canvas');
  }
  
  /**
   * Initialize auto-visualization tab
   */
  function initAutoViz() {
    const generateBtn = document.getElementById('generateAutoVizBtn');
    const maxChartsRange = document.getElementById('maxChartsRange');
    const maxChartsValue = document.getElementById('maxChartsValue');
    
    if (generateBtn) {
      generateBtn.addEventListener('click', function() {
        if (!datasetData) {
          fetchDatasetData(datasetId);
        } else {
          generateAutoVisualizations();
        }
      });
    }
    
    if (maxChartsRange && maxChartsValue) {
      maxChartsRange.addEventListener('input', function() {
        maxChartsValue.textContent = this.value;
      });
    }
  }
  
  /**
   * Generate automatic visualizations based on data characteristics
   */
  function generateAutoVisualizations() {
    const autoVizContainer = document.getElementById('autoVizContainer');
    const maxCharts = parseInt(document.getElementById('maxChartsRange').value);
    
    // Get selected chart types
    const chartTypes = [];
    if (document.getElementById('distributionCheck').checked) chartTypes.push('distribution');
    if (document.getElementById('relationshipCheck').checked) chartTypes.push('relationship');
    if (document.getElementById('compositionCheck').checked) chartTypes.push('composition');
    if (document.getElementById('comparisonCheck').checked) chartTypes.push('comparison');
    
    if (!datasetData || !autoVizContainer) {
      showError('Dataset data is not available');
      return;
    }
    
    showLoading(true);
    
    // Clear previous visualizations
    autoVizContainer.innerHTML = '';
    
    // Analyze dataset to determine appropriate visualizations
    const vizSuggestions = generateVizSuggestions(chartTypes);
    console.log('Visualization suggestions:', vizSuggestions);
    
    // Limit to max charts
    const selectedViz = vizSuggestions.slice(0, maxCharts);
    
    // Create visualization cards
    selectedViz.forEach((viz, index) => {
      const vizCard = document.createElement('div');
      vizCard.className = 'col-md-6 col-lg-4';
      vizCard.innerHTML = `
        <div class="viz-card">
          <div class="viz-card-header">
            <div class="d-flex justify-content-between align-items-center">
              <h6 class="mb-0">${viz.title}</h6>
              <button class="viz-explanation-btn" data-viz-index="${index}">
                <i class="fas fa-info-circle"></i>
              </button>
            </div>
          </div>
          <div class="viz-card-body">
            <div id="autoViz${index}" class="w-100 h-100"></div>
          </div>
          <div class="viz-card-footer">
            <small class="text-muted">${viz.description}</small>
          </div>
        </div>
      `;
      
      autoVizContainer.appendChild(vizCard);
      
      // Add event listener for explanation button
      const explanationBtn = vizCard.querySelector('.viz-explanation-btn');
      if (explanationBtn) {
        explanationBtn.addEventListener('click', function() {
          const vizIndex = parseInt(this.dataset.vizIndex);
          const viz = selectedViz[vizIndex];
          showVizExplanation(
            viz.title,
            viz.explanation,
            viz.interpretation,
            viz.insights
          );
        });
      }
      
      // Render the visualization
      setTimeout(() => {
        renderVisualization(viz, `autoViz${index}`);
      }, 100 * index); // Stagger rendering to prevent browser freeze
    });
    
    showLoading(false);
  }
  
  /**
   * Generate visualization suggestions based on data analysis
   */
  function generateVizSuggestions(chartTypes) {
    const suggestions = [];
    
    if (!datasetData || datasetData.length === 0 || !datasetColumns || datasetColumns.length === 0) {
      return suggestions;
    }
    
    // Identify numeric and categorical columns
    const numericColumns = datasetColumns.filter(col => 
      detectColumnType(col.name) === 'numeric'
    ).map(col => col.name);
    
    const categoricalColumns = datasetColumns.filter(col => 
      detectColumnType(col.name) === 'categorical'
    ).map(col => col.name);
    
    const dateColumns = datasetColumns.filter(col => 
      detectColumnType(col.name) === 'date'
    ).map(col => col.name);
    
    console.log('Column types:', {
      numeric: numericColumns,
      categorical: categoricalColumns,
      date: dateColumns
    });
    
    // 1. Distribution Charts
    if (chartTypes.includes('distribution') && numericColumns.length > 0) {
      // Histogram for each numeric column
      numericColumns.slice(0, 3).forEach(col => {
        suggestions.push({
          type: 'histogram',
          title: `Distribution of ${col}`,
          description: `Frequency distribution showing the range of values`,
          columns: [col],
          chartType: 'histogram',
          explanation: `This histogram shows the distribution of values for ${col}.`,
          interpretation: `Look for normal distributions, skewness, or multiple peaks that may indicate distinct subgroups.`,
          insights: [
            `The most common values are in the middle range.`,
            `Check for outliers in the extremes of the distribution.`
          ]
        });
      });
      
      // Box plot for numeric columns
      if (numericColumns.length >= 2) {
        suggestions.push({
          type: 'boxplot',
          title: `Box Plot Summary`,
          description: `Box plots showing distribution and outliers`,
          columns: numericColumns.slice(0, 5),
          chartType: 'boxplot',
          explanation: `Box plots show the median, quartiles, and outliers for each feature.`,
          interpretation: `Compare the medians and spreads across different features to identify patterns.`,
          insights: [
            `Boxes represent the middle 50% of data (IQR).`,
            `Points outside the whiskers are potential outliers.`
          ]
        });
      }
    }
    
    // 2. Relationship Charts
    if (chartTypes.includes('relationship') && numericColumns.length >= 2) {
      // Scatter plots for pairs of numeric columns
      for (let i = 0; i < Math.min(3, numericColumns.length - 1); i++) {
        const col1 = numericColumns[i];
        const col2 = numericColumns[i + 1];
        
        suggestions.push({
          type: 'scatter',
          title: `${col1} vs ${col2}`,
          description: `Scatter plot showing relationship between variables`,
          columns: [col1, col2],
          chartType: 'scatter',
          explanation: `This scatter plot shows the relationship between ${col1} and ${col2}.`,
          interpretation: `Look for patterns such as linear relationships, clusters, or outliers.`,
          insights: [
            `Dots that form a diagonal line suggest correlation.`,
            `Distinct clusters may indicate subgroups in your data.`
          ]
        });
      }
      
      // Correlation heatmap if enough numeric columns
      if (numericColumns.length >= 3) {
        suggestions.push({
          type: 'heatmap',
          title: `Correlation Matrix`,
          description: `Heatmap showing correlation between numeric variables`,
          columns: numericColumns.slice(0, 8),
          chartType: 'heatmap',
          explanation: `This heatmap shows the correlation coefficients between numeric variables.`,
          interpretation: `Darker red indicates strong positive correlation, darker blue indicates strong negative correlation.`,
          insights: [
            `Highly correlated features may contain redundant information.`,
            `Look for unexpected correlations that might reveal hidden patterns.`
          ]
        });
      }
    }
    
    // 3. Composition Charts
    if (chartTypes.includes('composition') && categoricalColumns.length > 0) {
      // Pie chart for categorical columns
      categoricalColumns.slice(0, 2).forEach(col => {
        suggestions.push({
          type: 'pie',
          title: `${col} Distribution`,
          description: `Pie chart showing proportion of categories`,
          columns: [col],
          chartType: 'pie',
          explanation: `This pie chart shows the distribution of categories in ${col}.`,
          interpretation: `The size of each slice represents the proportion of that category in the dataset.`,
          insights: [
            `Identify dominant categories that represent majority of the data.`,
            `Very small slices may be candidates for grouping into "Other".`
          ]
        });
      });
      
      // Treemap for hierarchical categorical data
      if (categoricalColumns.length >= 2) {
        suggestions.push({
          type: 'treemap',
          title: `${categoricalColumns[0]} by ${categoricalColumns[1]}`,
          description: `Treemap showing hierarchical composition`,
          columns: [categoricalColumns[0], categoricalColumns[1]],
          chartType: 'treemap',
          explanation: `This treemap shows the hierarchical breakdown of ${categoricalColumns[0]} by ${categoricalColumns[1]}.`,
          interpretation: `The size of each rectangle represents the proportion of that category combination.`,
          insights: [
            `Larger rectangles represent more frequent combinations.`,
            `Look for unexpected patterns in the hierarchy.`
          ]
        });
      }
    }
    
    // 4. Comparison Charts
    if (chartTypes.includes('comparison')) {
      // Bar charts for categorical vs numeric
      if (categoricalColumns.length > 0 && numericColumns.length > 0) {
        for (let i = 0; i < Math.min(2, categoricalColumns.length); i++) {
          const catCol = categoricalColumns[i];
          const numCol = numericColumns[0];
          
          suggestions.push({
            type: 'bar',
            title: `${numCol} by ${catCol}`,
            description: `Bar chart comparing numeric values across categories`,
            columns: [catCol, numCol],
            chartType: 'bar',
            explanation: `This bar chart compares the average ${numCol} across different ${catCol} categories.`,
            interpretation: `Taller bars indicate higher average values for that category.`,
            insights: [
              `Look for significant differences between categories.`,
              `Consider if the differences match your expectations or reveal unexpected patterns.`
            ]
          });
        }
      }
      
      // Line chart for date vs numeric
      if (dateColumns.length > 0 && numericColumns.length > 0) {
        suggestions.push({
          type: 'line',
          title: `${numericColumns[0]} over Time`,
          description: `Line chart showing trends over time`,
          columns: [dateColumns[0], numericColumns[0]],
          chartType: 'line',
          explanation: `This line chart shows the trend of ${numericColumns[0]} over time.`,
          interpretation: `Rising lines indicate increasing values, falling lines indicate decreasing values.`,
          insights: [
            `Look for seasonal patterns or cycles.`,
            `Sudden changes may indicate important events or anomalies.`
          ]
        });
      }
    }
    
    return suggestions;
  }
  
  /**
   * Render visualization using appropriate library
   */
  function renderVisualization(vizConfig, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Sample data for development - would normally use datasetData
    const sampleData = datasetData || generateSampleData(vizConfig);
    
    switch (vizConfig.chartType) {
      case 'histogram':
        renderHistogram(container, vizConfig, sampleData);
        break;
      case 'boxplot':
        renderBoxPlot(container, vizConfig, sampleData);
        break;
      case 'scatter':
        renderScatterPlot(container, vizConfig, sampleData);
        break;
      case 'heatmap':
        renderHeatmap(container, vizConfig, sampleData);
        break;
      case 'pie':
        renderPieChart(container, vizConfig, sampleData);
        break;
      case 'treemap':
        renderTreemap(container, vizConfig, sampleData);
        break;
      case 'bar':
        renderBarChart(container, vizConfig, sampleData);
        break;
      case 'line':
        renderLineChart(container, vizConfig, sampleData);
        break;
      default:
        container.innerHTML = '<div class="viz-placeholder"><p>Visualization type not supported</p></div>';
    }
  }
  
  /**
   * Generate sample data for development/testing
   */
  function generateSampleData(vizConfig) {
    // This would be replaced with actual data in production
    return []; // Placeholder
  }
  
  /**
   * Render histogram using Plotly
   */
  function renderHistogram(container, vizConfig, data) {
    const column = vizConfig.columns[0];
    const values = data.map(row => parseFloat(row[column])).filter(val => !isNaN(val));
    
    const trace = {
      x: values,
      type: 'histogram',
      marker: {
        color: '#4e73df',
        line: {
          color: '#3a58c8',
          width: 1
        }
      },
      opacity: 0.75
    };
    
    const layout = {
      title: '',
      margin: {t: 10, r: 10, l: 40, b: 40},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: {title: 'Frequency'},
      xaxis: {title: column},
      bargap: 0.05
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true, displayModeBar: false});
  }
  
  /**
   * Render box plot using Plotly
   */
  function renderBoxPlot(container, vizConfig, data) {
    const columns = vizConfig.columns;
    const traces = columns.map(col => {
      const values = data.map(row => parseFloat(row[col])).filter(val => !isNaN(val));
      return {
        y: values,
        type: 'box',
        name: col,
        marker: {
          color: '#4e73df'
        },
        boxmean: true
      };
    });
    
    const layout = {
      title: '',
      margin: {t: 10, r: 10, l: 50, b: 60},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      showlegend: false
    };
    
    Plotly.newPlot(container, traces, layout, {responsive: true, displayModeBar: false});
  }
  
  /**
   * Render scatter plot using Plotly
   */
  function renderScatterPlot(container, vizConfig, data) {
    const xColumn = vizConfig.columns[0];
    const yColumn = vizConfig.columns[1];
    
    const xValues = data.map(row => parseFloat(row[xColumn])).filter(val => !isNaN(val));
    const yValues = data.map(row => parseFloat(row[yColumn])).filter(val => !isNaN(val));
    
    // For simplicity, we'll just take the first n matching pairs
    const n = Math.min(xValues.length, yValues.length);
    
    const trace = {
      x: xValues.slice(0, n),
      y: yValues.slice(0, n),
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: '#4e73df',
        size: 8,
        opacity: 0.7
      }
    };
    
    const layout = {
      title: '',
      margin: {t: 10, r: 10, l: 50, b: 50},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {title: xColumn},
      yaxis: {title: yColumn}
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true, displayModeBar: false});
  }
  
  /**
   * Render heatmap using Plotly
   */
  function renderHeatmap(container, vizConfig, data) {
    const columns = vizConfig.columns;
    
    // Calculate correlation matrix
    const matrix = [];
    const values = {};
    
    // Extract values for each column
    columns.forEach(col => {
      values[col] = data.map(row => parseFloat(row[col])).filter(val => !isNaN(val));
    });
    
    // Calculate correlations
    columns.forEach(col1 => {
      const row = [];
      columns.forEach(col2 => {
        // This is a simplified correlation calculation
        // In production, use proper correlation formula
        const corr = col1 === col2 ? 1 : Math.random() * 2 - 1; // Dummy correlation for demo
        row.push(corr);
      });
      matrix.push(row);
    });
    
    const trace = {
      z: matrix,
      x: columns,
      y: columns,
      type: 'heatmap',
      colorscale: 'RdBu',
      zmin: -1,
      zmax: 1
    };
    
    const layout = {
      title: '',
      margin: {t: 25, r: 10, l: 100, b: 100},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      height: 400
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true, displayModeBar: false});
  }
  
  /**
   * Render pie chart using Plotly
   */
  function renderPieChart(container, vizConfig, data) {
    const column = vizConfig.columns[0];
    
    // Count frequencies of each category
    const counts = {};
    data.forEach(row => {
      const value = row[column];
      counts[value] = (counts[value] || 0) + 1;
    });
    
    // Convert to arrays for plotting
    const labels = Object.keys(counts);
    const values = Object.values(counts);
    
    const trace = {
      labels: labels,
      values: values,
      type: 'pie',
      textinfo: 'percent',
      insidetextorientation: 'radial',
      marker: {
        colors: ['#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b', '#858796']
      }
    };
    
    const layout = {
      title: '',
      margin: {t: 10, r: 10, l: 10, b: 10},
      paper_bgcolor: 'rgba(0,0,0,0)',
      showlegend: true,
      legend: {
        orientation: 'h',
        y: -0.2
      }
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true, displayModeBar: false});
  }
  
  /**
   * Render treemap using Plotly
   */
  function renderTreemap(container, vizConfig, data) {
    const col1 = vizConfig.columns[0];
    const col2 = vizConfig.columns[1];
    
    // Count combinations of categories
    const counts = {};
    data.forEach(row => {
      const key = `${row[col1]}-${row[col2]}`;
      counts[key] = (counts[key] || 0) + 1;
    });
    
    const labels = Object.keys(counts).map(k => k.replace('-', '<br>'));
    const values = Object.values(counts);
    
    const trace = {
      type: 'treemap',
      labels: labels,
      parents: labels.map(() => ''),
      values: values,
      textinfo: 'label+value',
      marker: {
        colorscale: 'Blues'
      }
    };
    
    const layout = {
      title: '',
      margin: {t: 0, r: 0, l: 0, b: 0},
      paper_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true, displayModeBar: false});
  }
  
  /**
   * Render bar chart using Plotly
   */
  function renderBarChart(container, vizConfig, data) {
    const catColumn = vizConfig.columns[0];
    const numColumn = vizConfig.columns[1];
    
    // Group by category and calculate averages
    const groups = {};
    data.forEach(row => {
      const category = row[catColumn];
      const value = parseFloat(row[numColumn]);
      
      if (!isNaN(value)) {
        if (!groups[category]) {
          groups[category] = {sum: 0, count: 0};
        }
        groups[category].sum += value;
        groups[category].count += 1;
      }
    });
    
    // Calculate averages
    const categories = Object.keys(groups);
    const averages = categories.map(cat => groups[cat].sum / groups[cat].count);
    
    const trace = {
      x: categories,
      y: averages,
      type: 'bar',
      marker: {
        color: '#4e73df',
        opacity: 0.8
      }
    };
    
    const layout = {
      title: '',
      margin: {t: 10, r: 10, l: 60, b: 80},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
        title: catColumn,
        tickangle: -45
      },
      yaxis: {
        title: `Average ${numColumn}`
      }
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true, displayModeBar: false});
  }
  
  /**
   * Render line chart using Plotly
   */
  function renderLineChart(container, vizConfig, data) {
    const dateColumn = vizConfig.columns[0];
    const valueColumn = vizConfig.columns[1];
    
    // Sort data by date
    const sortedData = [...data].sort((a, b) => {
      return new Date(a[dateColumn]) - new Date(b[dateColumn]);
    });
    
    const dates = sortedData.map(row => row[dateColumn]);
    const values = sortedData.map(row => parseFloat(row[valueColumn])).filter(val => !isNaN(val));
    
    const trace = {
      x: dates,
      y: values,
      type: 'scatter',
      mode: 'lines+markers',
      marker: {
        color: '#4e73df',
        size: 5
      },
      line: {
        width: 2,
        color: '#4e73df'
      }
    };
    
    const layout = {
      title: '',
      margin: {t: 10, r: 10, l: 60, b: 60},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
        title: 'Date',
        tickangle: -45
      },
      yaxis: {
        title: valueColumn
      }
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true, displayModeBar: false});
  }
  
  /**
   * Initialize EDA (Exploratory Data Analysis) tab
   */
  function initEDA() {
    const generateEdaBtn = document.getElementById('generateEdaBtn');
    const edaTypeRadios = document.querySelectorAll('input[name="edaType"]');
    
    // EDA type change handlers
    edaTypeRadios.forEach(radio => {
      radio.addEventListener('change', function() {
        showEdaOptions(this.value);
      });
    });
    
    // Generate EDA button
    if (generateEdaBtn) {
      generateEdaBtn.addEventListener('click', function() {
        generateEDA();
      });
    }
    
    // Initialize with default option
    showEdaOptions('univariate');
  }

  /**
   * Show/hide EDA option sections based on selected analysis type
   */
  function showEdaOptions(edaType) {
    const optionSections = document.querySelectorAll('.eda-options');
    
    // Hide all option sections
    optionSections.forEach(section => {
      section.style.display = 'none';
    });
    
    // Show selected option section
    const targetSection = document.getElementById(`${edaType}Options`);
    if (targetSection) {
      targetSection.style.display = 'block';
    }
  }

  /**
   * Populate EDA feature dropdowns
   */
  function populateEdaFeatures() {
    const featureDropdowns = [
      'univariateFeature',
      'bivariateFeature1', 
      'bivariateFeature2',
      'distFeature'
    ];
    
    featureDropdowns.forEach(dropdownId => {
      const dropdown = document.getElementById(dropdownId);
      if (!dropdown) return;
      
      // Clear existing options except first one
      while (dropdown.children.length > 1) {
        dropdown.removeChild(dropdown.lastChild);
      }
      
      if (!datasetColumns) return;
      
      // Handle different data structures for datasetColumns
      let columns = [];
      if (Array.isArray(datasetColumns)) {
        columns = datasetColumns;
      } else if (datasetColumns.columns && Array.isArray(datasetColumns.columns)) {
        columns = datasetColumns.columns;
      } else if (typeof datasetColumns === 'object') {
        columns = Object.keys(datasetColumns);
      }
      
      // Add options for each column
      columns.forEach(col => {
        const colName = typeof col === 'string' ? col : (col.name || col);
        if (colName) {
          const option = document.createElement('option');
          option.value = colName;
          option.textContent = colName;
          dropdown.appendChild(option);
        }
      });
    });
    
    // Populate multivariate feature checkboxes
    populateMultivariateFeatures();
  }

  /**
   * Populate multivariate feature checkboxes
   */
  function populateMultivariateFeatures() {
    const container = document.getElementById('multivariateFeatureSelection');
    if (!container || !datasetColumns) return;
    
    container.innerHTML = '';
    
    // Handle different data structures for datasetColumns
    let columns = [];
    if (Array.isArray(datasetColumns)) {
      columns = datasetColumns;
    } else if (datasetColumns.columns && Array.isArray(datasetColumns.columns)) {
      columns = datasetColumns.columns;
    } else if (typeof datasetColumns === 'object') {
      columns = Object.keys(datasetColumns);
    }
    
    if (columns.length === 0) {
      container.innerHTML = '<div class="form-text">No features available</div>';
      return;
    }
    
    // Add checkboxes for each column
    columns.forEach(col => {
      const colName = typeof col === 'string' ? col : (col.name || col);
      if (colName) {
        const checkbox = document.createElement('div');
        checkbox.className = 'form-check';
        checkbox.innerHTML = `
          <input class="form-check-input multivariate-feature" type="checkbox" value="${colName}" id="multi_${colName.replace(/[^a-zA-Z0-9]/g, '_')}">
          <label class="form-check-label" for="multi_${colName.replace(/[^a-zA-Z0-9]/g, '_')}">
            ${colName}
          </label>
        `;
        container.appendChild(checkbox);
      }
    });
  }

  /**
   * Generate EDA analysis based on selected type
   */
  function generateEDA() {
    const selectedType = document.querySelector('input[name="edaType"]:checked')?.value;
    if (!selectedType) {
      showError('Please select an analysis type');
      return;
    }
    
    const edaContainer = document.getElementById('edaContainer');
    const edaPlaceholder = document.getElementById('edaPlaceholder');
    
    if (edaPlaceholder) {
      edaPlaceholder.style.display = 'none';
    }
    
    showLoading(true);
    
    switch (selectedType) {
      case 'univariate':
        generateUnivariateAnalysis();
        break;
      case 'bivariate':
        generateBivariateAnalysis();
        break;
      case 'multivariate':
        generateMultivariateAnalysis();
        break;
      case 'missing':
        generateMissingValueAnalysis();
        break;
      case 'distribution':
        generateDistributionAnalysis();
        break;
      default:
        showError('Unknown analysis type selected');
        showLoading(false);
    }
  }

  /**
   * Generate univariate analysis
   */
  function generateUnivariateAnalysis() {
    const feature = document.getElementById('univariateFeature').value;
    if (!feature) {
      showError('Please select a feature for univariate analysis');
      showLoading(false);
      return;
    }
    
    const plotTypes = [];
    if (document.getElementById('histogramCheck').checked) plotTypes.push('histogram');
    if (document.getElementById('boxplotCheck').checked) plotTypes.push('boxplot');
    if (document.getElementById('violinCheck').checked) plotTypes.push('violin');
    if (document.getElementById('countplotCheck').checked) plotTypes.push('countplot');
    
    fetch(`/api/dataset/${datasetId}/eda/univariate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature, plot_types: plotTypes })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        displayUnivariateResults(data);
      } else {
        showError(data.message || 'Failed to generate univariate analysis');
      }
      showLoading(false);
    })
    .catch(error => {
      console.error('Error in univariate analysis:', error);
      showError('Failed to generate univariate analysis');
      showLoading(false);
    });
  }

  /**
   * Generate bivariate analysis
   */
  function generateBivariateAnalysis() {
    const feature1 = document.getElementById('bivariateFeature1').value;
    const feature2 = document.getElementById('bivariateFeature2').value;
    
    if (!feature1 || !feature2) {
      showError('Please select both features for bivariate analysis');
      showLoading(false);
      return;
    }
    
    const plotTypes = [];
    if (document.getElementById('scatterCheck').checked) plotTypes.push('scatter');
    if (document.getElementById('hexbinCheck').checked) plotTypes.push('hexbin');
    if (document.getElementById('groupedBoxCheck').checked) plotTypes.push('grouped_box');
    if (document.getElementById('barplotCheck').checked) plotTypes.push('barplot');
    if (document.getElementById('contingencyCheck').checked) plotTypes.push('contingency');
    
    fetch(`/api/dataset/${datasetId}/eda/bivariate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature1, feature2, plot_types: plotTypes })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        displayBivariateResults(data);
      } else {
        showError(data.message || 'Failed to generate bivariate analysis');
      }
      showLoading(false);
    })
    .catch(error => {
      console.error('Error in bivariate analysis:', error);
      showError('Failed to generate bivariate analysis');
      showLoading(false);
    });
  }

  /**
   * Generate multivariate analysis
   */
  function generateMultivariateAnalysis() {
    const selectedFeatures = Array.from(document.querySelectorAll('.multivariate-feature:checked'))
      .map(checkbox => checkbox.value);
    
    if (selectedFeatures.length < 2) {
      showError('Please select at least 2 features for multivariate analysis');
      showLoading(false);
      return;
    }
    
    if (selectedFeatures.length > 6) {
      showError('Please select maximum 6 features for multivariate analysis');
      showLoading(false);
      return;
    }
    
    const plotTypes = [];
    if (document.getElementById('pairplotCheck').checked) plotTypes.push('pairplot');
    if (document.getElementById('parallelCheck').checked) plotTypes.push('parallel');
    if (document.getElementById('pcaCheck').checked) plotTypes.push('pca');
    
    fetch(`/api/dataset/${datasetId}/eda/multivariate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: selectedFeatures, plot_types: plotTypes })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        displayMultivariateResults(data);
      } else {
        showError(data.message || 'Failed to generate multivariate analysis');
      }
      showLoading(false);
    })
    .catch(error => {
      console.error('Error in multivariate analysis:', error);
      showError('Failed to generate multivariate analysis');
      showLoading(false);
    });
  }

  /**
   * Generate missing value analysis
   */
  function generateMissingValueAnalysis() {
    const plotTypes = [];
    if (document.getElementById('missingMatrixCheck').checked) plotTypes.push('matrix');
    if (document.getElementById('missingBarCheck').checked) plotTypes.push('bar');
    if (document.getElementById('missingHeatmapCheck').checked) plotTypes.push('heatmap');
    
    fetch(`/api/dataset/${datasetId}/eda/missing`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plot_types: plotTypes })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        displayMissingValueResults(data);
      } else {
        showError(data.message || 'Failed to generate missing value analysis');
      }
      showLoading(false);
    })
    .catch(error => {
      console.error('Error in missing value analysis:', error);
      showError('Failed to generate missing value analysis');
      showLoading(false);
    });
  }

  /**
   * Generate distribution comparison analysis
   */
  function generateDistributionAnalysis() {
    const feature = document.getElementById('distFeature').value;
    const distType = document.getElementById('distType').value;
    
    if (!feature) {
      showError('Please select a feature for distribution analysis');
      showLoading(false);
      return;
    }
    
    fetch(`/api/dataset/${datasetId}/eda/distribution`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature, dist_type: distType })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        displayDistributionResults(data);
      } else {
        showError(data.message || 'Failed to generate distribution analysis');
      }
      showLoading(false);
    })
    .catch(error => {
      console.error('Error in distribution analysis:', error);
      showError('Failed to generate distribution analysis');
      showLoading(false);
    });
  }

  /**
   * Display univariate analysis results
   */
  function displayUnivariateResults(data) {
    const container = document.getElementById('edaVisualizationGrid');
    container.innerHTML = '';
    
    // Update title
    document.getElementById('edaTitle').textContent = `Univariate Analysis: ${data.feature}`;
    
    // Display summary statistics
    displaySummaryStats(data.summary, data.feature);
    
    // Display insights
    if (data.insights && data.insights.length > 0) {
      displayEdaInsights(data.insights);
    }
  }

  /**
   * Display bivariate analysis results
   */
  function displayBivariateResults(data) {
    const container = document.getElementById('edaVisualizationGrid');
    container.innerHTML = '';
    
    // Update title
    document.getElementById('edaTitle').textContent = `Bivariate Analysis: ${data.feature1} vs ${data.feature2}`;
    
    // Display insights
    if (data.insights && data.insights.length > 0) {
      displayEdaInsights(data.insights);
    }
  }

  /**
   * Display multivariate analysis results
   */
  function displayMultivariateResults(data) {
    const container = document.getElementById('edaVisualizationGrid');
    container.innerHTML = '';
    
    // Update title
    document.getElementById('edaTitle').textContent = `Multivariate Analysis: ${data.features.join(', ')}`;
    
    // Display insights
    if (data.insights && data.insights.length > 0) {
      displayEdaInsights(data.insights);
    }
  }

  /**
   * Display missing value analysis results
   */
  function displayMissingValueResults(data) {
    const container = document.getElementById('edaVisualizationGrid');
    container.innerHTML = '';
    
    // Update title
    document.getElementById('edaTitle').textContent = 'Missing Value Analysis';
    
    // Display insights
    if (data.insights && data.insights.length > 0) {
      displayEdaInsights(data.insights);
    }
  }

  /**
   * Display distribution analysis results
   */
  function displayDistributionResults(data) {
    const container = document.getElementById('edaVisualizationGrid');
    container.innerHTML = '';
    
    // Update title
    document.getElementById('edaTitle').textContent = `Distribution Analysis: ${data.feature}`;
    
    // Display insights
    if (data.insights && data.insights.length > 0) {
      displayEdaInsights(data.insights);
    }
  }

  /**
   * Display summary statistics table
   */
  function displaySummaryStats(summary, feature) {
    const summaryStatsDiv = document.getElementById('summaryStats');
    const summaryStatsHeader = document.getElementById('summaryStatsHeader');
    const summaryStatsBody = document.getElementById('summaryStatsBody');
    
    if (!summaryStatsDiv || !summary) return;
    
    // Update header
    summaryStatsHeader.innerHTML = `<th>Statistic</th><th>${feature}</th>`;
    
    // Clear body
    summaryStatsBody.innerHTML = '';
    
    // Add statistics rows
    Object.entries(summary).forEach(([stat, value]) => {
      if (value !== null && value !== undefined) {
        const row = document.createElement('tr');
        let displayValue = value;
        
        if (typeof value === 'number') {
          displayValue = value.toFixed(3);
        } else if (typeof value === 'object') {
          displayValue = JSON.stringify(value);
        }
        
        row.innerHTML = `
          <td><strong>${stat.replace(/_/g, ' ').toUpperCase()}</strong></td>
          <td>${displayValue}</td>
        `;
        summaryStatsBody.appendChild(row);
      }
    });
    
    // Show summary stats
    summaryStatsDiv.style.display = 'block';
  }

  /**
   * Display EDA insights
   */
  function displayEdaInsights(insights) {
    const container = document.getElementById('edaVisualizationGrid');
    
    if (insights && insights.length > 0) {
      const insightsCard = document.createElement('div');
      insightsCard.className = 'col-12 mb-3';
      insightsCard.innerHTML = `
        <div class="card">
          <div class="card-header">
            <h6 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Key Insights</h6>
          </div>
          <div class="card-body">
            <ul class="list-unstyled mb-0">
              ${insights.map(insight => `<li class="mb-2"><i class="fas fa-check-circle text-success me-2"></i>${insight}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
      container.appendChild(insightsCard);
    }
  }

  /**
   * Initialize correlation analysis tab
   */
  function initCorrelationAnalysis() {
    const generateBtn = document.getElementById('generateCorrelationBtn');
    const thresholdSlider = document.getElementById('correlationThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    
    if (generateBtn) {
      generateBtn.addEventListener('click', function() {
        if (!datasetData) {
          fetchDatasetData(datasetId, 1000);
        } else {
          generateCorrelationMatrix();
        }
      });
    }
    
    if (thresholdSlider && thresholdValue) {
      thresholdSlider.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
        // If we already have a correlation matrix, update it
        if (document.querySelector('#correlationMatrix').hasChildNodes()) {
          generateCorrelationMatrix();
        }
      });
    }
  }
  
  /**
   * Populate feature selection checkboxes for correlation analysis
   */
  function populateFeatureSelection() {
    const featureSelection = document.getElementById('featureSelection');
    if (!featureSelection) {
      return;
    }
    
    // Clear existing features
    featureSelection.innerHTML = '';
    
    if (!datasetColumns) {
      return;
    }
    
    // Handle different data structures for datasetColumns
    let columns = [];
    
    if (Array.isArray(datasetColumns)) {
      columns = datasetColumns;
    } else if (datasetColumns.columns && Array.isArray(datasetColumns.columns)) {
      columns = datasetColumns.columns;
    } else if (typeof datasetColumns === 'object') {
      columns = Object.keys(datasetColumns);
    }
    
    if (columns.length === 0) {
      featureSelection.innerHTML = '<div class="form-text">No features found for correlation analysis</div>';
      return;
    }
    
    // Add checkboxes for each column
    columns.forEach(col => {
      const colName = typeof col === 'string' ? col : (col.name || col);
      
      if (colName) {
        const checkbox = document.createElement('div');
        checkbox.className = 'form-check';
        checkbox.innerHTML = `
          <input class="form-check-input feature-checkbox" type="checkbox" value="${colName}" id="check_${colName.replace(/[^a-zA-Z0-9]/g, '_')}" checked>
          <label class="form-check-label" for="check_${colName.replace(/[^a-zA-Z0-9]/g, '_')}">
            ${colName}
          </label>
        `;
        featureSelection.appendChild(checkbox);
      }
    });
  }
  
  /**
   * Generate correlation matrix
   */
  function generateCorrelationMatrix() {
    const correlationMatrix = document.getElementById('correlationMatrix');
    const correlationInsights = document.getElementById('correlationInsights');
    const method = document.getElementById('correlationMethod').value;
    const threshold = parseFloat(document.getElementById('correlationThreshold').value);
    
    // Get selected features
    const selectedFeatures = Array.from(document.querySelectorAll('.feature-checkbox:checked'))
      .map(checkbox => checkbox.value);
    
    if (selectedFeatures.length < 2) {
      showError('Please select at least 2 features for correlation analysis');
      return;
    }
    
    if (!datasetData || !correlationMatrix) {
      showError('Dataset data is not available');
      return;
    }
    
    showLoading(true);
    
    // Calculate correlation matrix
    const correlations = calculateCorrelations(selectedFeatures, method);
    
    // Render the correlation heatmap
    renderCorrelationHeatmap(correlationMatrix, correlations, selectedFeatures);
    
    // Generate and display insights
    if (correlationInsights) {
      const insights = generateCorrelationInsights(correlations, selectedFeatures, threshold);
      displayCorrelationInsights(correlationInsights, insights);
    }
    
    showLoading(false);
  }
  
  /**
   * Calculate correlation matrix
   */
  function calculateCorrelations(features, method) {
    const result = [];
    
    // Extract values for each feature
    const values = {};
    features.forEach(feature => {
      values[feature] = datasetData
        .map(row => parseFloat(row[feature]))
        .filter(val => !isNaN(val));
    });
    
    // Calculate correlation for each pair of features
    features.forEach((feature1, i) => {
      const row = [];
      features.forEach((feature2, j) => {
        let correlation;
        if (i === j) {
          correlation = 1; // Perfect correlation with self
        } else {
          correlation = calculatePairCorrelation(values[feature1], values[feature2], method);
        }
        row.push(correlation);
      });
      result.push(row);
    });
    
    return result;
  }
  
  /**
   * Calculate correlation between two arrays
   */
  function calculatePairCorrelation(array1, array2, method) {
    // For simplicity, using Pearson correlation
    // In production, implement other methods (Spearman, Kendall)
    
    // Get common length
    const n = Math.min(array1.length, array2.length);
    if (n < 2) return 0;
    
    // Calculate means
    const sum1 = array1.slice(0, n).reduce((acc, val) => acc + val, 0);
    const sum2 = array2.slice(0, n).reduce((acc, val) => acc + val, 0);
    const mean1 = sum1 / n;
    const mean2 = sum2 / n;
    
    // Calculate correlation
    let num = 0;
    let den1 = 0;
    let den2 = 0;
    
    for (let i = 0; i < n; i++) {
      const val1 = array1[i] - mean1;
      const val2 = array2[i] - mean2;
      num += val1 * val2;
      den1 += val1 * val1;
      den2 += val2 * val2;
    }
    
    if (den1 === 0 || den2 === 0) return 0;
    
    return num / Math.sqrt(den1 * den2);
  }
  
  /**
   * Render correlation heatmap
   */
  function renderCorrelationHeatmap(container, correlations, features) {
    const trace = {
      z: correlations,
      x: features,
      y: features,
      type: 'heatmap',
      colorscale: 'RdBu',
      zmin: -1,
      zmax: 1,
      hoverongaps: false
    };
    
    const layout = {
      title: '',
      margin: {t: 25, r: 10, l: 120, b: 120},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      height: 400,
      xaxis: {
        tickangle: -45
      }
    };
    
    Plotly.newPlot(container, [trace], layout, {responsive: true});
  }
  
  /**
   * Generate insights from correlation matrix
   */
  function generateCorrelationInsights(correlations, features, threshold) {
    const insights = [];
    
    // Find strong correlations (positive and negative)
    for (let i = 0; i < features.length; i++) {
      for (let j = i + 1; j < features.length; j++) {
        const corr = correlations[i][j];
        const absCorr = Math.abs(corr);
        
        if (absCorr >= threshold) {
          let strength;
          let strengthClass;
          
          if (absCorr >= 0.8) {
            strength = 'Strong';
            strengthClass = 'strength-high';
          } else if (absCorr >= 0.5) {
            strength = 'Moderate';
            strengthClass = 'strength-medium';
          } else {
            strength = 'Weak';
            strengthClass = 'strength-low';
          }
          
          const direction = corr > 0 ? 'positive' : 'negative';
          
          insights.push({
            feature1: features[i],
            feature2: features[j],
            correlation: corr,
            strength,
            strengthClass,
            direction
          });
        }
      }
    }
    
    // Sort by absolute correlation (descending)
    insights.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
    
    return insights;
  }
  
  /**
   * Display correlation insights
   */
  function displayCorrelationInsights(container, insights) {
    if (!container) return;
    
    if (insights.length === 0) {
      container.innerHTML = '<p class="text-muted">No significant correlations found with the current threshold.</p>';
      return;
    }
    
    container.innerHTML = '';
    
    insights.forEach(insight => {
      const insightElement = document.createElement('div');
      insightElement.className = 'correlation-insight';
      insightElement.innerHTML = `
        <div class="insight-heading">
          <span class="correlation-strength ${insight.strengthClass}">${insight.strength} ${insight.direction}</span> correlation
        </div>
        <div class="insight-details">
          <strong>${insight.feature1}</strong> and <strong>${insight.feature2}</strong>
          (r = ${insight.correlation.toFixed(2)})
        </div>
        <div class="insight-interpretation">
          ${getCorrelationInterpretation(insight)}
        </div>
      `;
      container.appendChild(insightElement);
    });
  }
  
  /**
   * Get interpretation for a correlation insight
   */
  function getCorrelationInterpretation(insight) {
    if (insight.correlation > 0) {
      return `As ${insight.feature1} increases, ${insight.feature2} tends to increase as well.`;
    } else {
      return `As ${insight.feature1} increases, ${insight.feature2} tends to decrease.`;
    }
  }
  
  /**
   * Initialize anomaly detection tab
   */
  function initAnomalyDetection() {
    const detectBtn = document.getElementById('detectAnomaliesBtn');
    const sensitivitySlider = document.getElementById('anomalySensitivity');
    const sensitivityValue = document.getElementById('sensitivityValue');
    
    if (detectBtn) {
      detectBtn.addEventListener('click', function() {
        if (!datasetData) {
          fetchDatasetData(datasetId, 2000);
        } else {
          detectAnomalies();
        }
      });
    }
    
    if (sensitivitySlider && sensitivityValue) {
      sensitivitySlider.addEventListener('input', function() {
        sensitivityValue.textContent = this.value;
      });
    }
  }
  
  /**
   * Populate feature dropdown for anomaly detection
   */
  function populateAnomalyFeatures() {
    const featureSelect = document.getElementById('anomalyFeature');
    if (!featureSelect) {
      return;
    }
    
    // Clear existing options
    featureSelect.innerHTML = '<option value="">Select feature for anomaly detection</option>';
    
    if (!datasetColumns) {
      return;
    }
    
    // Handle different data structures for datasetColumns
    let columns = [];
    
    if (Array.isArray(datasetColumns)) {
      columns = datasetColumns;
    } else if (datasetColumns.columns && Array.isArray(datasetColumns.columns)) {
      columns = datasetColumns.columns;
    } else if (typeof datasetColumns === 'object') {
      columns = Object.keys(datasetColumns);
    }
    
    // Filter numeric columns and add to select
    columns.forEach(col => {
      const colName = typeof col === 'string' ? col : (col.name || col);
      
      if (colName) {
        const option = document.createElement('option');
        option.value = colName;
        option.textContent = colName;
        featureSelect.appendChild(option);
      }
    });
  }
  
  /**
   * Detect anomalies in selected feature
   */
  function detectAnomalies() {
    const anomalyPlot = document.getElementById('anomalyPlot');
    const anomalyCountElem = document.getElementById('anomalyCount');
    const anomalyPercentageElem = document.getElementById('anomalyPercentage');
    const anomalyDetails = document.getElementById('anomalyDetails');
    
    const feature = document.getElementById('anomalyFeature').value;
    const method = document.getElementById('anomalyMethod').value;
    const sensitivity = parseFloat(document.getElementById('anomalySensitivity').value);
    
    if (!feature) {
      showError('Please select a feature for anomaly detection');
      return;
    }
    
    if (!datasetData || !anomalyPlot) {
      showError('Dataset data is not available');
      return;
    }
    
    showLoading(true);
    
    // Extract values for the selected feature
    const values = datasetData
      .map(row => parseFloat(row[feature]))
      .filter(val => !isNaN(val));
    
    // Detect anomalies using the selected method
    const anomalies = findAnomalies(values, method, sensitivity);
    
    // Calculate statistics
    const anomalyIndices = Object.keys(anomalies).map(i => parseInt(i));
    const anomalyCount = anomalyIndices.length;
    const anomalyPercentage = (anomalyCount / values.length) * 100;
    
    // Update statistics display
    anomalyCountElem.textContent = anomalyCount;
    anomalyPercentageElem.textContent = `${anomalyPercentage.toFixed(2)}%`;
    
    // Render anomaly plot
    renderAnomalyPlot(anomalyPlot, values, anomalies, feature);
    
    // Display anomaly details
    displayAnomalyDetails(anomalyDetails, values, anomalies, feature);
    
    showLoading(false);
  }
  
  /**
   * Find anomalies in a dataset
   */
  function findAnomalies(values, method, sensitivity) {
    const anomalies = {};
    
    switch (method) {
      case 'statistical':
        // IQR method (box plot)
        const q1 = calculateQuantile(values, 0.25);
        const q3 = calculateQuantile(values, 0.75);
        const iqr = q3 - q1;
        const lowerBound = q1 - sensitivity * iqr;
        const upperBound = q3 + sensitivity * iqr;
        
        values.forEach((value, index) => {
          if (value < lowerBound || value > upperBound) {
            anomalies[index] = value;
          }
        });
        break;
        
      case 'zscore':
        // Z-score method
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
        const stdDev = Math.sqrt(squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length);
        
        values.forEach((value, index) => {
          const zscore = Math.abs((value - mean) / stdDev);
          if (zscore > sensitivity) {
            anomalies[index] = value;
          }
        });
        break;
        
      case 'isolation':
        // Simple implementation of isolation forest concept
        // In production, use a proper library for this
        const deviationThreshold = calculateQuantile(values, 0.75) - calculateQuantile(values, 0.25);
        
        values.forEach((value, index) => {
          let isAnomaly = true;
          let neighborCount = 0;
          
          // Find if the point has nearby neighbors (simplified approach)
          values.forEach(otherValue => {
            if (Math.abs(value - otherValue) <= deviationThreshold * sensitivity) {
              neighborCount++;
            }
          });
          
          // If it has enough neighbors, it's not an anomaly
          if (neighborCount > values.length * 0.05) {
            isAnomaly = false;
          }
          
          if (isAnomaly) {
            anomalies[index] = value;
          }
        });
        break;
    }
    
    return anomalies;
  }
  
  /**
   * Calculate quantile value
   */
  function calculateQuantile(values, q) {
    const sorted = [...values].sort((a, b) => a - b);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    
    if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
      return sorted[base];
    }
  }
  
  /**
   * Render anomaly detection plot
   */
  function renderAnomalyPlot(container, values, anomalies, feature) {
    // Prepare data points
    const indices = Array.from({length: values.length}, (_, i) => i);
    const anomalyIndices = Object.keys(anomalies).map(i => parseInt(i));
    const anomalyValues = anomalyIndices.map(i => values[i]);
    
    // Normal points trace
    const normalIndices = indices.filter(i => !anomalyIndices.includes(i));
    const normalValues = normalIndices.map(i => values[i]);
    
    const normalTrace = {
      x: normalIndices,
      y: normalValues,
      mode: 'markers',
      type: 'scatter',
      name: 'Normal',
      marker: {
        color: '#4e73df',
        size: 8,
        opacity: 0.7
      }
    };
    
    // Anomaly points trace
    const anomalyTrace = {
      x: anomalyIndices,
      y: anomalyValues,
      mode: 'markers',
      type: 'scatter',
      name: 'Anomaly',
      marker: {
        color: '#e74a3b',
        size: 10,
        opacity: 0.8,
        symbol: 'circle-open',
        line: {
          color: '#e74a3b',
          width: 2
        }
      }
    };
    
    const layout = {
      title: '',
      margin: {t: 10, r: 10, l: 60, b: 40},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
        title: 'Data Point Index',
        showgrid: true,
        gridcolor: '#f5f5f5'
      },
      yaxis: {
        title: feature,
        showgrid: true,
        gridcolor: '#f5f5f5'
      },
      legend: {
        orientation: 'h',
        y: 1.1
      }
    };
    
    Plotly.newPlot(container, [normalTrace, anomalyTrace], layout, {responsive: true});
  }
  
  /**
   * Display anomaly details
   */
  function displayAnomalyDetails(container, values, anomalies, feature) {
    if (!container) return;
    
    if (Object.keys(anomalies).length === 0) {
      container.innerHTML = '<p class="text-muted">No anomalies detected with current settings.</p>';
      return;
    }
    
    container.innerHTML = '';
    
    // Get anomaly details
    const anomalyIndices = Object.keys(anomalies).map(i => parseInt(i));
    const anomalyValues = anomalyIndices.map(i => values[i]);
    
    // Calculate statistics for context
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const median = calculateQuantile(values, 0.5);
    
    // Display details
    const detailsElement = document.createElement('div');
    detailsElement.innerHTML = `
      <div class="mb-3">
        <strong>Average ${feature}:</strong> ${mean.toFixed(2)} | 
        <strong>Median ${feature}:</strong> ${median.toFixed(2)}
      </div>
      <div class="anomaly-list">
        <table class="table table-sm">
          <thead>
            <tr>
              <th>Index</th>
              <th>Value</th>
              <th>Deviation</th>
            </tr>
          </thead>
          <tbody>
            ${anomalyIndices.slice(0, 10).map((index, i) => {
              const value = anomalies[index];
              const deviation = ((value - mean) / mean * 100).toFixed(2);
              return `
                <tr class="anomaly-item">
                  <td>${index}</td>
                  <td class="anomaly-value">${value.toFixed(2)}</td>
                  <td>${deviation}% from mean</td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
        ${anomalyIndices.length > 10 ? `<p class="text-muted">Showing 10 of ${anomalyIndices.length} anomalies</p>` : ''}
      </div>
    `;
    container.appendChild(detailsElement);
  }
  
  /**
   * Initialize custom visualization tab
   */
  function initCustomViz() {
    const generateBtn = document.getElementById('generateCustomVizBtn');
    const chartTypeSelect = document.getElementById('chartType');
    const addToDashboardBtn = document.getElementById('addToDashboardBtn');
    const exportPngBtn = document.getElementById('exportPngBtn');
    const exportSvgBtn = document.getElementById('exportSvgBtn');
    const exportCsvBtn = document.getElementById('exportCsvBtn');
    
    // Current chart state
    let currentCustomChart = null;
    
    if (generateBtn) {
      generateBtn.addEventListener('click', function() {
        if (!datasetData) {
          fetchDatasetData(datasetId, 1000);
        } else {
          generateCustomVisualization();
        }
      });
    }
    
    if (chartTypeSelect) {
      chartTypeSelect.addEventListener('change', function() {
        updateCustomChartOptions(this.value);
      });
    }
    
    if (addToDashboardBtn) {
      addToDashboardBtn.addEventListener('click', function() {
        addCurrentChartToDashboard();
      });
    }
    
    // Export buttons
    if (exportPngBtn) {
      exportPngBtn.addEventListener('click', function() {
        exportChart('png');
      });
    }
    
    if (exportSvgBtn) {
      exportSvgBtn.addEventListener('click', function() {
        exportChart('svg');
      });
    }
    
    if (exportCsvBtn) {
      exportCsvBtn.addEventListener('click', function() {
        exportChart('csv');
      });
    }
    
    /**
     * Update form options based on chart type
     */
    function updateCustomChartOptions(chartType) {
      const yAxisContainer = document.getElementById('yAxisContainer');
      const groupContainer = document.getElementById('groupContainer');
      const trendlineContainer = document.getElementById('trendlineContainer');
      const animationContainer = document.getElementById('animationContainer');
      
      // Show/hide options based on chart type
      switch (chartType) {
        case 'pie':
        case 'treemap':
        case 'donut':
          // These charts don't need Y axis
          if (yAxisContainer) yAxisContainer.style.display = 'none';
          if (groupContainer) groupContainer.style.display = 'block';
          if (trendlineContainer) trendlineContainer.style.display = 'none';
          if (animationContainer) animationContainer.style.display = 'block';
          break;
          
        case 'histogram':
        case 'density':
          // These charts only need X axis
          if (yAxisContainer) yAxisContainer.style.display = 'none';
          if (groupContainer) groupContainer.style.display = 'none';
          if (trendlineContainer) trendlineContainer.style.display = 'none';
          if (animationContainer) animationContainer.style.display = 'block';
          break;
          
        case 'scatter':
        case 'line':
          // These support trend lines
          if (yAxisContainer) yAxisContainer.style.display = 'block';
          if (groupContainer) groupContainer.style.display = 'block';
          if (trendlineContainer) trendlineContainer.style.display = 'block';
          if (animationContainer) animationContainer.style.display = 'block';
          break;
          
        case 'heatmap':
        case 'bubble':
          // Special cases for multi-dimensional data
          if (yAxisContainer) yAxisContainer.style.display = 'block';
          if (groupContainer) groupContainer.style.display = 'block';
          if (trendlineContainer) trendlineContainer.style.display = 'none';
          if (animationContainer) animationContainer.style.display = 'block';
          break;
          
        default:
          // Most charts need both X and Y
          if (yAxisContainer) yAxisContainer.style.display = 'block';
          if (groupContainer) groupContainer.style.display = 'block';
          if (trendlineContainer) trendlineContainer.style.display = 'none';
          if (animationContainer) animationContainer.style.display = 'block';
      }
      
      // Update trend line options
      const showTrendline = document.getElementById('showTrendline');
      if (showTrendline) {
        showTrendline.disabled = !['scatter', 'line'].includes(chartType);
      }
      
      // Update axis labels based on chart type
      const xAxisLabel = document.querySelector('label[for="xAxis"]');
      const yAxisLabel = document.querySelector('label[for="yAxis"]');
      
      if (xAxisLabel) {
        switch (chartType) {
          case 'pie':
          case 'donut':
            xAxisLabel.textContent = 'Categories';
            break;
          case 'histogram':
            xAxisLabel.textContent = 'Values';
            break;
          default:
            xAxisLabel.textContent = 'X-axis';
        }
      }
      
      if (yAxisLabel) {
        switch (chartType) {
          case 'bar':
          case 'line':
            yAxisLabel.textContent = 'Y-axis (Values)';
            break;
          case 'scatter':
            yAxisLabel.textContent = 'Y-axis (Scatter)';
            break;
          default:
            yAxisLabel.textContent = 'Y-axis';
        }
      }
    }
    
    /**
     * Generate custom visualization
     */
    function generateCustomVisualization() {
      const chartType = document.getElementById('chartType').value;
      const xAxis = document.getElementById('xAxis').value;
      const yAxis = document.getElementById('yAxis').value;
      const groupBy = document.getElementById('groupBy').value;
      const colorTheme = document.getElementById('colorTheme').value;
      const showTrendline = document.getElementById('showTrendline').checked;
      const showLabels = document.getElementById('showLabels').checked;
      const useAnimation = document.getElementById('useAnimation').checked;
      const chartTitle = document.getElementById('chartTitle').value;
      
      const container = document.getElementById('customVizResult');
      
      // Validate inputs
      if (!xAxis) {
        showError('Please select a feature for X-axis');
        return;
      }
      
      if (['bar', 'line', 'scatter'].includes(chartType) && !yAxis) {
        showError('Please select a feature for Y-axis');
        return;
      }
      
      if (!datasetData || !container) {
        showError('Dataset data is not available');
        return;
      }
      
      showLoading(true);
      
      // Create chart configuration
      const chartConfig = {
        chartType,
        xAxis,
        yAxis,
        groupBy,
        colorTheme,
        showTrendline,
        showLabels,
        useAnimation,
        title: chartTitle
      };
      
      // Render the chart
      currentCustomChart = renderCustomChart(container, chartConfig, datasetData);
      
      // Enable add to dashboard button
      if (addToDashboardBtn) {
        addToDashboardBtn.disabled = false;
      }
      
      showLoading(false);
    }
    
    /**
     * Add current chart to dashboard
     */
    function addCurrentChartToDashboard() {
      if (!currentCustomChart) {
        showError('Please generate a chart first');
        return;
      }
      
      // Create dashboard item
      const chartConfig = currentCustomChart.config;
      const chartTitle = chartConfig.title || `${chartConfig.xAxis} ${chartConfig.chartType}`;
      
      // Create dashboard item and add to dashboard
      const dashboardItem = {
        id: 'chart_' + Date.now(),
        title: chartTitle,
        type: chartConfig.chartType,
        config: chartConfig,
        size: 'normal' // or 'large'
      };
      
      addItemToDashboard(dashboardItem);
      
      // Switch to dashboard tab
      document.getElementById('dashboard-tab').click();
    }
    
    /**
     * Export current chart
     */
    function exportChart(format) {
      if (!currentCustomChart) {
        showError('Please generate a chart first');
        return;
      }
      
      const chartTitle = currentCustomChart.config.title || 'chart';
      const fileName = chartTitle.replace(/\s+/g, '_').toLowerCase();
      
      // This would use Plotly's export functions in production
      alert(`Export ${fileName} as ${format} - Will be implemented with Plotly's toImage functionality`);
    }
  }
  
  /**
   * Populate feature dropdowns for custom chart
   */
  function populateChartFeatures() {
    const xAxisSelect = document.getElementById('xAxis');
    const yAxisSelect = document.getElementById('yAxis');
    const groupBySelect = document.getElementById('groupBy');
    
    // Clear existing options
    if (xAxisSelect) xAxisSelect.innerHTML = '<option value="">Select X-axis feature</option>';
    if (yAxisSelect) yAxisSelect.innerHTML = '<option value="">Select Y-axis feature</option>';
    if (groupBySelect) groupBySelect.innerHTML = '<option value="">None</option>';
    
    if (!datasetColumns) {
      return;
    }
    
    // Handle different data structures for datasetColumns
    let columns = [];
    
    if (Array.isArray(datasetColumns)) {
      columns = datasetColumns;
    } else if (datasetColumns.columns && Array.isArray(datasetColumns.columns)) {
      columns = datasetColumns.columns;
    } else if (typeof datasetColumns === 'object') {
      columns = Object.keys(datasetColumns);
    }
    
    // Add all columns to X-axis
    if (xAxisSelect) {
      columns.forEach(col => {
        const colName = typeof col === 'string' ? col : (col.name || col);
        if (colName) {
          const option = document.createElement('option');
          option.value = colName;
          option.textContent = colName;
          xAxisSelect.appendChild(option);
        }
      });
    }
    
    // Add all columns to Y-axis
    if (yAxisSelect) {
      columns.forEach(col => {
        const colName = typeof col === 'string' ? col : (col.name || col);
        if (colName) {
          const option = document.createElement('option');
          option.value = colName;
          option.textContent = colName;
          yAxisSelect.appendChild(option);
        }
      });
    }
    
    // Add all columns to group by
    if (groupBySelect) {
      columns.forEach(col => {
        const colName = typeof col === 'string' ? col : (col.name || col);
        if (colName) {
          const option = document.createElement('option');
          option.value = colName;
          option.textContent = colName;
          groupBySelect.appendChild(option);
        }
      });
    }
    
    // Initialize chart options based on default chart type
    const chartTypeSelect = document.getElementById('chartType');
    if (chartTypeSelect) {
      updateCustomChartOptions(chartTypeSelect.value);
    }
  }
  
  /**
   * Render custom chart
   */
  function renderCustomChart(container, config, data) {
    if (!container) return null;
    
    // Extract configuration
    const { chartType, xAxis, yAxis, groupBy, colorTheme, showTrendline, showLabels, title } = config;
    
    // Define color schemes based on theme
    const colorSchemes = {
      default: ['#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b', '#858796'],
      pastel: ['#ff9999', '#99ff99', '#9999ff', '#ffff99', '#ff99ff', '#99ffff'],
      viridis: ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
      plasma: ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
      inferno: ['#000004', '#420a68', '#932667', '#dd513a', '#fca50a', '#fcffa4']
    };
    
    const colors = colorSchemes[colorTheme] || colorSchemes.default;
    
    // Create layout with title and margins - enhanced for better readability
    const layout = {
      title: {
        text: title,
        font: {
          size: 18,
          family: 'Arial, sans-serif',
          color: '#2c3e50'
        }
      },
      width: 800,
      height: 600,
      margin: {t: title ? 60 : 20, r: 40, l: 80, b: 80},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(248, 249, 250, 0.8)',
      font: {
        size: 14,
        family: 'Arial, sans-serif'
      },
      xaxis: {
        title: {
          text: xAxis,
          font: {
            size: 16,
            family: 'Arial, sans-serif',
            color: '#495057'
          }
        },
        showgrid: true,
        gridcolor: '#e9ecef',
        gridwidth: 1,
        tickfont: {
          size: 12,
          family: 'Arial, sans-serif'
        }
      },
      yaxis: {
        title: {
          text: yAxis,
          font: {
            size: 16,
            family: 'Arial, sans-serif',
            color: '#495057'
          }
        },
        showgrid: true,
        gridcolor: '#e9ecef',
        gridwidth: 1,
        tickfont: {
          size: 12,
          family: 'Arial, sans-serif'
        }
      }
    };
    
    // Set up animation
    const chartOptions = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    if (config.useAnimation) {
      chartOptions.animate = true;
      chartOptions.animation = {
        duration: 1000,
        easing: 'cubic-in-out'
      };
    }
    
    // Render based on chart type
    let trace;
    
    switch (chartType) {
      case 'bar':
        trace = renderCustomBarChart(data, xAxis, yAxis, groupBy, colors, showLabels);
        break;
      case 'line':
        trace = renderCustomLineChart(data, xAxis, yAxis, groupBy, colors, showTrendline, showLabels);
        break;
      case 'scatter':
        trace = renderCustomScatterPlot(data, xAxis, yAxis, groupBy, colors, showTrendline, showLabels);
        break;
      case 'pie':
        trace = renderCustomPieChart(data, xAxis, groupBy, colors, showLabels);
        // Adjust layout for pie chart - larger margins for better readability
        layout.margin = {t: title ? 60 : 20, r: 40, l: 40, b: 40};
        break;
      case 'histogram':
        trace = renderCustomHistogram(data, xAxis, colors, showLabels);
        break;
      case 'heatmap':
        trace = renderCustomHeatmap(data, xAxis, yAxis, groupBy, colors);
        break;
      case 'boxplot':
        trace = renderCustomBoxPlot(data, xAxis, yAxis, groupBy, colors);
        break;
      default:
        container.innerHTML = '<div class="viz-placeholder"><p>Chart type not supported</p></div>';
        return null;
    }
    
    // Render the chart
    Plotly.newPlot(container, Array.isArray(trace) ? trace : [trace], layout, chartOptions);
    
    // Return chart info for later reference
    return {
      container,
      config,
      plot: container // Reference to Plotly element
    };
  }
  
  /**
   * Render custom bar chart
   */
  function renderCustomBarChart(data, xAxis, yAxis, groupBy, colors, showLabels) {
    // If no groupBy, create simple bar chart
    if (!groupBy) {
      // Group by X axis and calculate average of Y axis
      const groups = {};
      data.forEach(row => {
        const xValue = row[xAxis];
        const yValue = parseFloat(row[yAxis]);
        
        if (!isNaN(yValue)) {
          if (!groups[xValue]) {
            groups[xValue] = {sum: 0, count: 0};
          }
          groups[xValue].sum += yValue;
          groups[xValue].count += 1;
        }
      });
      
      // Calculate averages
      const xValues = Object.keys(groups);
      const yValues = xValues.map(x => groups[x].sum / groups[x].count);
      
      return {
        x: xValues,
        y: yValues,
        type: 'bar',
        marker: {
          color: colors[0],
          opacity: 0.8
        },
        text: showLabels ? yValues.map(y => y.toFixed(2)) : undefined,
        textposition: 'auto',
        hoverinfo: 'x+y'
      };
    } else {
      // Create grouped bar chart
      // Group by both X axis and groupBy
      const groups = {};
      const groupByValues = new Set();
      
      data.forEach(row => {
        const xValue = row[xAxis];
        const groupValue = row[groupBy];
        const yValue = parseFloat(row[yAxis]);
        
        if (!isNaN(yValue)) {
          groupByValues.add(groupValue);
          
          if (!groups[xValue]) {
            groups[xValue] = {};
          }
          
          if (!groups[xValue][groupValue]) {
            groups[xValue][groupValue] = {sum: 0, count: 0};
          }
          
          groups[xValue][groupValue].sum += yValue;
          groups[xValue][groupValue].count += 1;
        }
      });
      
      // Convert to traces, one for each group value
      const xValues = Object.keys(groups);
      const traces = Array.from(groupByValues).map((groupValue, i) => {
        const yValues = xValues.map(x => 
          groups[x][groupValue] 
            ? groups[x][groupValue].sum / groups[x][groupValue].count 
            : null
        );
        
        return {
          x: xValues,
          y: yValues,
          type: 'bar',
          name: groupValue,
          marker: {
            color: colors[i % colors.length],
            opacity: 0.8
          },
          text: showLabels ? yValues.map(y => y !== null ? y.toFixed(2) : '') : undefined,
          textposition: 'auto',
          hoverinfo: 'x+y+name'
        };
      });
      
      return traces;
    }
  }
  
  /**
   * Render custom line chart
   */
  function renderCustomLineChart(data, xAxis, yAxis, groupBy, colors, showTrendline, showLabels) {
    // Simplified line chart implementation
    if (!groupBy) {
      // Create simple line chart
      // Group by X axis and calculate average of Y axis
      const groups = {};
      data.forEach(row => {
        const xValue = row[xAxis];
        const yValue = parseFloat(row[yAxis]);
        
        if (!isNaN(yValue)) {
          if (!groups[xValue]) {
            groups[xValue] = {sum: 0, count: 0};
          }
          groups[xValue].sum += yValue;
          groups[xValue].count += 1;
        }
      });
      
      // Calculate averages and sort by x-axis
      const points = Object.keys(groups)
        .map(x => ({
          x,
          y: groups[x].sum / groups[x].count
        }))
        .sort((a, b) => {
          // Try to sort numerically or as dates if possible
          const dateA = new Date(a.x);
          const dateB = new Date(b.x);
          
          if (!isNaN(dateA) && !isNaN(dateB)) {
            return dateA - dateB;
          }
          
          const numA = parseFloat(a.x);
          const numB = parseFloat(b.x);
          
          if (!isNaN(numA) && !isNaN(numB)) {
            return numA - numB;
          }
          
          return a.x.localeCompare(b.x);
        });
      
      const xValues = points.map(p => p.x);
      const yValues = points.map(p => p.y);
      
      // Create trace for line chart
      const trace = {
        x: xValues,
        y: yValues,
        type: 'scatter',
        mode: showLabels ? 'lines+markers+text' : 'lines+markers',
        name: yAxis,
        line: {
          color: colors[0],
          width: 2
        },
        marker: {
          color: colors[0],
          size: 6
        },
        text: showLabels ? yValues.map(y => y.toFixed(2)) : undefined,
        textposition: 'top',
        hoverinfo: 'x+y'
      };
      
      // Add trendline if requested
      if (showTrendline) {
        const trendline = calculateTrendline(xValues, yValues);
        
        const trendTrace = {
          x: [xValues[0], xValues[xValues.length - 1]],
          y: [trendline.start, trendline.end],
          type: 'scatter',
          mode: 'lines',
          name: 'Trend',
          line: {
            color: '#e74a3b',
            width: 2,
            dash: 'dash'
          },
          hoverinfo: 'none'
        };
        
        return [trace, trendTrace];
      }
      
      return trace;
    } else {
      // Create multi-line chart grouped by groupBy
      // Group by both X axis and groupBy
      const groups = {};
      const groupByValues = new Set();
      
      data.forEach(row => {
        const xValue = row[xAxis];
        const groupValue = row[groupBy];
        const yValue = parseFloat(row[yAxis]);
        
        if (!isNaN(yValue)) {
          groupByValues.add(groupValue);
          
          if (!groups[groupValue]) {
            groups[groupValue] = {};
          }
          
          if (!groups[groupValue][xValue]) {
            groups[groupValue][xValue] = {sum: 0, count: 0};
          }
          
          groups[groupValue][xValue].sum += yValue;
          groups[groupValue][xValue].count += 1;
        }
      });
      
      // Convert to traces, one for each group value
      const traces = Array.from(groupByValues).map((groupValue, i) => {
        // Calculate averages and sort by x-axis
        const points = Object.keys(groups[groupValue])
          .map(x => ({
            x,
            y: groups[groupValue][x].sum / groups[groupValue][x].count
          }))
          .sort((a, b) => {
            // Try to sort numerically or as dates if possible
            const dateA = new Date(a.x);
            const dateB = new Date(b.x);
            
            if (!isNaN(dateA) && !isNaN(dateB)) {
              return dateA - dateB;
            }
            
            const numA = parseFloat(a.x);
            const numB = parseFloat(b.x);
            
            if (!isNaN(numA) && !isNaN(numB)) {
              return numA - numB;
            }
            
            return a.x.localeCompare(b.x);
          });
        
        const xValues = points.map(p => p.x);
        const yValues = points.map(p => p.y);
        
        return {
          x: xValues,
          y: yValues,
          type: 'scatter',
          mode: showLabels ? 'lines+markers+text' : 'lines+markers',
          name: groupValue,
          line: {
            color: colors[i % colors.length],
            width: 2
          },
          marker: {
            color: colors[i % colors.length],
            size: 6
          },
          text: showLabels ? yValues.map(y => y.toFixed(2)) : undefined,
          textposition: 'top',
          hoverinfo: 'x+y+name'
        };
      });
      
      return traces;
    }
  }
  
  /**
   * Calculate trendline for a set of points
   */
  function calculateTrendline(xValues, yValues) {
    // For date or string x values, use indices
    const xNumerical = typeof xValues[0] === 'number' 
      ? xValues 
      : xValues.map((_, i) => i);
    
    // Simple linear regression
    const n = xNumerical.length;
    
    // Calculate means
    const meanX = xNumerical.reduce((sum, x) => sum + x, 0) / n;
    const meanY = yValues.reduce((sum, y) => sum + y, 0) / n;
    
    // Calculate slope and intercept
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 0; i < n; i++) {
      numerator += (xNumerical[i] - meanX) * (yValues[i] - meanY);
      denominator += Math.pow(xNumerical[i] - meanX, 2);
    }
    
    const slope = denominator !== 0 ? numerator / denominator : 0;
    const intercept = meanY - slope * meanX;
    
    // Calculate start and end points
    const start = slope * xNumerical[0] + intercept;
    const end = slope * xNumerical[n - 1] + intercept;
    
    return { slope, intercept, start, end };
  }
  
  /**
   * Render custom scatter plot
   */
  function renderCustomScatterPlot(data, xAxis, yAxis, groupBy, colors, showTrendline, showLabels) {
    // Extract values
    const xValues = data.map(row => parseFloat(row[xAxis])).filter(val => !isNaN(val));
    const yValues = data.map(row => parseFloat(row[yAxis])).filter(val => !isNaN(val));
    
    // Basic trace for scatter plot
    const trace = {
      x: xValues,
      y: yValues,
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: colors[0],
        size: 8,
        opacity: 0.7
      },
      hoverinfo: 'x+y'
    };
    
    // Add trendline if requested
    if (showTrendline) {
      const trendline = calculateTrendline(xValues, yValues);
      
      const trendTrace = {
        x: [Math.min(...xValues), Math.max(...xValues)],
        y: [
          trendline.slope * Math.min(...xValues) + trendline.intercept,
          trendline.slope * Math.max(...xValues) + trendline.intercept
        ],
        type: 'scatter',
        mode: 'lines',
        name: 'Trend',
        line: {
          color: '#e74a3b',
          width: 2,
          dash: 'dash'
        },
        hoverinfo: 'none'
      };
      
      return [trace, trendTrace];
    }
    
    return trace;
  }
  
  /**
   * Render custom pie chart
   */
  function renderCustomPieChart(data, xAxis, groupBy, colors, showLabels) {
    // Count frequencies
    const counts = {};
    data.forEach(row => {
      const value = row[xAxis];
      counts[value] = (counts[value] || 0) + 1;
    });
    
    // Convert to arrays for plotting
    const labels = Object.keys(counts);
    const values = Object.values(counts);
    
    return {
      labels: labels,
      values: values,
      type: 'pie',
      textinfo: showLabels ? 'label+percent' : 'percent',
      hoverinfo: 'label+value+percent',
      insidetextorientation: 'radial',
      marker: {
        colors: colors
      }
    };
  }
  
  /**
   * Render custom histogram
   */
  function renderCustomHistogram(data, xAxis, colors, showLabels) {
    const values = data.map(row => parseFloat(row[xAxis])).filter(val => !isNaN(val));
    
    return {
      x: values,
      type: 'histogram',
      marker: {
        color: colors[0],
        line: {
          color: 'white',
          width: 1
        }
      },
      opacity: 0.75,
      hoverinfo: 'x+y',
      histnorm: '',
      autobinx: true
    };
  }
  
  /**
   * Render custom heatmap
   */
  function renderCustomHeatmap(data, xAxis, yAxis, groupBy, colors) {
    // Simple heatmap implementation for two numeric columns
    const xValues = data.map(row => parseFloat(row[xAxis])).filter(val => !isNaN(val));
    const yValues = data.map(row => parseFloat(row[yAxis])).filter(val => !isNaN(val));
    
    // For simplicity, create a 2D histogram
    // In production, use a proper 2D histogram implementation
    
    // Define bins for x and y axes
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    const xBins = 10;
    const yBins = 10;
    
    const xStep = (xMax - xMin) / xBins;
    const yStep = (yMax - yMin) / yBins;
    
    // Initialize 2D array for histogram
    const histogram = Array(yBins).fill().map(() => Array(xBins).fill(0));
    
    // Count points in each bin
    for (let i = 0; i < Math.min(xValues.length, yValues.length); i++) {
      const x = xValues[i];
      const y = yValues[i];
      
      const xBin = Math.min(Math.floor((x - xMin) / xStep), xBins - 1);
      const yBin = Math.min(Math.floor((y - yMin) / yStep), yBins - 1);
      
      histogram[yBin][xBin]++;
    }
    
    // Create x and y bin centers for display
    const xBinCenters = Array(xBins).fill().map((_, i) => xMin + (i + 0.5) * xStep);
    const yBinCenters = Array(yBins).fill().map((_, i) => yMin + (i + 0.5) * yStep);
    
    return {
      z: histogram,
      x: xBinCenters,
      y: yBinCenters,
      type: 'heatmap',
      colorscale: 'Viridis'
    };
  }
  
  /**
   * Render custom box plot
   */
  function renderCustomBoxPlot(data, xAxis, yAxis, groupBy, colors) {
    if (!groupBy) {
      // Simple box plot
      const values = data.map(row => parseFloat(row[yAxis])).filter(val => !isNaN(val));
      
      return {
        y: values,
        type: 'box',
        name: yAxis,
        marker: {
          color: colors[0]
        },
        boxmean: true
      };
    } else {
      // Grouped box plot
      const groups = {};
      const groupByValues = new Set();
      
      data.forEach(row => {
        const groupValue = row[groupBy];
        const yValue = parseFloat(row[yAxis]);
        
        if (!isNaN(yValue)) {
          groupByValues.add(groupValue);
          
          if (!groups[groupValue]) {
            groups[groupValue] = [];
          }
          
          groups[groupValue].push(yValue);
        }
      });
      
      // Create traces for each group
      return Array.from(groupByValues).map((groupValue, i) => ({
        y: groups[groupValue],
        type: 'box',
        name: groupValue,
        marker: {
          color: colors[i % colors.length]
        },
        boxmean: true
      }));
    }
  }
  
  /**
   * Initialize dashboard tab
   */
  function initDashboard() {
    const saveDashboardBtn = document.getElementById('saveDashboardBtn');
    const resetDashboardBtn = document.getElementById('resetDashboardBtn');
    const dashboardGrid = document.getElementById('dashboardGrid');
    const dashboardEmptyState = document.getElementById('dashboardEmptyState');
    
    // Load any saved dashboard items
    loadDashboard();
    
    if (saveDashboardBtn) {
      saveDashboardBtn.addEventListener('click', function() {
        saveDashboard();
      });
    }
    
    if (resetDashboardBtn) {
      resetDashboardBtn.addEventListener('click', function() {
        resetDashboard();
      });
    }
    
    /**
     * Load dashboard from localStorage
     */
    function loadDashboard() {
      try {
        const savedDashboard = localStorage.getItem(`dashboard_${datasetId}`);
        if (savedDashboard) {
          dashboardItems = JSON.parse(savedDashboard);
          renderDashboard();
        }
      } catch (e) {
        console.error('Error loading dashboard:', e);
      }
    }
    
    /**
     * Save dashboard to localStorage
     */
    function saveDashboard() {
      try {
        localStorage.setItem(`dashboard_${datasetId}`, JSON.stringify(dashboardItems));
        alert('Dashboard saved successfully');
      } catch (e) {
        console.error('Error saving dashboard:', e);
        showError('Failed to save dashboard');
      }
    }
    
    /**
     * Reset dashboard to empty state
     */
    function resetDashboard() {
      if (confirm('Are you sure you want to reset the dashboard? All visualizations will be removed.')) {
        dashboardItems = [];
        renderDashboard();
        localStorage.removeItem(`dashboard_${datasetId}`);
      }
    }
    
    /**
     * Render the dashboard
     */
    function renderDashboard() {
      if (!dashboardGrid || !dashboardEmptyState) return;
      
      // Show/hide empty state
      if (dashboardItems.length === 0) {
        dashboardEmptyState.style.display = 'flex';
        dashboardGrid.style.display = 'none';
        return;
      }
      
      dashboardEmptyState.style.display = 'none';
      dashboardGrid.style.display = 'grid';
      
      // Clear existing items
      dashboardGrid.innerHTML = '';
      
      // Render each item
      dashboardItems.forEach(item => {
        renderDashboardItem(item);
      });
    }
    
    /**
     * Render a dashboard item
     */
    function renderDashboardItem(item) {
      const itemElement = document.createElement('div');
      itemElement.className = `dashboard-item ${item.size === 'large' ? 'large' : ''}`;
      itemElement.dataset.itemId = item.id;
      
      itemElement.innerHTML = `
        <div class="dashboard-item-header">
          <h6 class="mb-0">${item.title}</h6>
          <div class="dashboard-item-controls">
            <button class="btn btn-sm btn-link resize-btn" title="Toggle size">
              <i class="fas ${item.size === 'large' ? 'fa-compress-alt' : 'fa-expand-alt'}"></i>
            </button>
            <button class="btn btn-sm btn-link remove-btn" title="Remove">
              <i class="fas fa-times"></i>
            </button>
          </div>
        </div>
        <div class="dashboard-item-body">
          <div id="${item.id}_chart" class="w-100 h-100"></div>
        </div>
      `;
      
      dashboardGrid.appendChild(itemElement);
      
      // Add event listeners
      const resizeBtn = itemElement.querySelector('.resize-btn');
      const removeBtn = itemElement.querySelector('.remove-btn');
      
      if (resizeBtn) {
        resizeBtn.addEventListener('click', function() {
          toggleItemSize(item.id);
        });
      }
      
      if (removeBtn) {
        removeBtn.addEventListener('click', function() {
          removeItem(item.id);
        });
      }
      
      // Render the chart
      const chartContainer = document.getElementById(`${item.id}_chart`);
      if (chartContainer && datasetData) {
        renderCustomChart(chartContainer, item.config, datasetData);
      }
    }
    
    /**
     * Toggle dashboard item size
     */
    function toggleItemSize(itemId) {
      const itemIndex = dashboardItems.findIndex(item => item.id === itemId);
      if (itemIndex !== -1) {
        dashboardItems[itemIndex].size = dashboardItems[itemIndex].size === 'large' ? 'normal' : 'large';
        renderDashboard();
      }
    }
    
    /**
     * Remove dashboard item
     */
    function removeItem(itemId) {
      const itemIndex = dashboardItems.findIndex(item => item.id === itemId);
      if (itemIndex !== -1) {
        dashboardItems.splice(itemIndex, 1);
        renderDashboard();
      }
    }
  }
  
  /**
   * Add an item to the dashboard
   */
  function addItemToDashboard(item) {
    dashboardItems.push(item);
    
    // If dashboard is visible, re-render it
    if (currentTab === 'dashboard') {
      renderDashboard();
    }
  }
}); 