let currentData = null;
let performanceChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('dataset');
    const fileDropZone = document.getElementById('fileDropZone');
    const analyzeBtn = document.querySelector('.analyze-btn');
    
    // File upload handlers
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop handlers
    fileDropZone.addEventListener('dragover', handleDragOver);
    fileDropZone.addEventListener('dragleave', handleDragLeave);
    fileDropZone.addEventListener('drop', handleFileDrop);
    
    // Form submission
    uploadForm.addEventListener('submit', handleFormSubmit);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        displayFileInfo(file);
        enableAnalyzeButton();
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'text/csv') {
        document.getElementById('dataset').files = files;
        displayFileInfo(files[0]);
        enableAnalyzeButton();
    }
}

function displayFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'flex';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function enableAnalyzeButton() {
    const analyzeBtn = document.querySelector('.analyze-btn');
    analyzeBtn.disabled = false;
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('dataset');
    if (!fileInput.files[0]) {
        showNotification('Please select a CSV file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('dataset', fileInput.files[0]);
    
    showLoading();
    updateStatus('Analyzing', 'warning');
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        currentData = data;
        displayResults(data);
        updateStatus('Complete', 'success');
        
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
        updateStatus('Error', 'error');
    } finally {
        hideLoading();
    }
}

function showLoading() {
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loadingText');
    const progressFill = document.getElementById('progressFill');
    
    loading.classList.remove('hidden');
    
    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
        
        if (progress < 30) {
            loadingText.textContent = 'Preprocessing data and feature analysis...';
        } else if (progress < 60) {
            loadingText.textContent = 'Training and evaluating models...';
        } else {
            loadingText.textContent = 'Generating insights and recommendations...';
        }
    }, 200);
    
    // Clear interval when loading is done
    loading.dataset.interval = interval;
}

function hideLoading() {
    const loading = document.getElementById('loading');
    const interval = loading.dataset.interval;
    
    if (interval) {
        clearInterval(interval);
    }
    
    // Complete progress bar
    document.getElementById('progressFill').style.width = '100%';
    
    setTimeout(() => {
        loading.classList.add('hidden');
    }, 500);
}

function updateStatus(text, type) {
    const statusText = document.querySelector('.status-text');
    const statusIndicator = document.querySelector('.status-indicator');
    
    statusText.textContent = text;
    
    // Remove existing classes
    statusIndicator.classList.remove('success', 'warning', 'error');
    
    // Add new class based on type
    if (type === 'success') {
        statusIndicator.style.background = 'var(--success-color)';
    } else if (type === 'warning') {
        statusIndicator.style.background = 'var(--warning-color)';
    } else if (type === 'error') {
        statusIndicator.style.background = 'var(--danger-color)';
    }
}

function displayResults(data) {
    displayBestModel(data.best_model);
    displayDatasetStats(data.dataset_analysis);
    displayPerformanceChart(data.recommendations);
    displayModelRecommendations(data.recommendations);
    
    const results = document.getElementById('results');
    results.classList.remove('hidden');
    results.classList.add('fade-in');
}

function displayBestModel(bestModel) {
    const bestModelCard = document.getElementById('bestModelCard');
    
    if (!bestModel) {
        bestModelCard.innerHTML = '<p>No model recommendations available</p>';
        return;
    }
    
    bestModelCard.innerHTML = `
        <div class="best-model-name">${bestModel.name}</div>
        <div class="best-model-score">${bestModel.score.toFixed(1)}% Suitability</div>
        <div class="best-model-reason">${bestModel.why_best}</div>
    `;
}

function displayDatasetStats(analysis) {
    const statsContainer = document.getElementById('datasetStats');
    
    const stats = [
        { label: 'Samples', value: analysis.n_samples.toLocaleString(), highlight: analysis.data_size_category === 'large' },
        { label: 'Features', value: analysis.n_features, highlight: analysis.dimensionality === 'high' },
        { label: 'Task Type', value: analysis.task_type.charAt(0).toUpperCase() + analysis.task_type.slice(1) },
        { label: 'Missing %', value: analysis.missing_percentage.toFixed(1) + '%' },
        { label: 'Numerical', value: analysis.numerical_features },
        { label: 'Categorical', value: analysis.categorical_features },
        { label: 'Data Size', value: analysis.data_size_category.charAt(0).toUpperCase() + analysis.data_size_category.slice(1) },
        { label: 'Dimensionality', value: analysis.dimensionality.charAt(0).toUpperCase() + analysis.dimensionality.slice(1) }
    ];
    
    statsContainer.innerHTML = `
        <div class="stats-grid">
            ${stats.map(stat => `
                <div class="stat-card ${stat.highlight ? 'highlight' : ''}">
                    <div class="stat-value">${stat.value}</div>
                    <div class="stat-label">${stat.label}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function displayPerformanceChart(recommendations) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    // Destroy existing chart
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    const labels = recommendations.map(r => r.model);
    const scores = recommendations.map(r => (r.performance.mean_score * 100));
    const errors = recommendations.map(r => (r.performance.std_score * 100));
    const trainingTimes = recommendations.map(r => r.performance.training_time || 0);
    
    performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Performance Score (%)',
                data: scores,
                backgroundColor: recommendations.map((_, i) => 
                    i === 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(37, 99, 235, 0.8)'
                ),
                borderColor: recommendations.map((_, i) => 
                    i === 0 ? 'rgba(34, 197, 94, 1)' : 'rgba(37, 99, 235, 1)'
                ),
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        afterBody: function(context) {
                            const index = context[0].dataIndex;
                            return [
                                `Std Dev: ±${errors[index].toFixed(2)}%`,
                                `Training Time: ${trainingTimes[index].toFixed(2)}s`
                            ];
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 45
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function displayModelRecommendations(recommendations) {
    const container = document.getElementById('modelCards');
    
    container.innerHTML = recommendations.slice(1).map((rec, index) => {
        const rank = index + 2;
        const modelIcons = {
            'Random Forest': 'fas fa-tree',
            'Gradient Boosting': 'fas fa-chart-line',
            'Logistic Regression': 'fas fa-function',
            'Linear Regression': 'fas fa-function',
            'SVM': 'fas fa-vector-square',
            'Decision Tree': 'fas fa-sitemap',
            'KNN': 'fas fa-users',
            'Naive Bayes': 'fas fa-brain',
            'Neural Network': 'fas fa-network-wired',
            'AdaBoost': 'fas fa-rocket',
            'Ridge Regression': 'fas fa-mountain',
            'Lasso Regression': 'fas fa-lasso',
            'ElasticNet': 'fas fa-expand-arrows-alt'
        };
        
        const additionalMetrics = rec.performance.additional_metrics || {};
        
        return `
            <div class="model-card rank-${rank} slide-up" style="animation-delay: ${index * 0.1}s" data-model="${rec.model}">
                <div class="model-header">
                    <div class="model-info">
                        <div class="model-icon">
                            <i class="${modelIcons[rec.model] || 'fas fa-cog'}"></i>
                        </div>
                        <div class="model-details">
                            <h4>#${rank} ${rec.model}</h4>
                            <div class="model-type">${currentData.dataset_analysis.task_type}</div>
                        </div>
                    </div>
                    <div class="suitability-score">
                        <i class="fas fa-star"></i>
                        ${rec.suitability_score.toFixed(1)}%
                    </div>
                </div>
                
                <div class="model-body">
                    <div class="performance-metrics">
                        <div class="metric">
                            <div class="metric-value">${(rec.performance.mean_score * 100).toFixed(1)}%</div>
                            <div class="metric-label">CV Score</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">±${(rec.performance.std_score * 100).toFixed(1)}%</div>
                            <div class="metric-label">Std Dev</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${rec.performance.training_time.toFixed(2)}s</div>
                            <div class="metric-label">Train Time</div>
                        </div>
                        ${Object.keys(additionalMetrics).length > 0 ? `
                            <div class="metric">
                                <div class="metric-value">${Object.values(additionalMetrics)[0].toFixed(3)}</div>
                                <div class="metric-label">${Object.keys(additionalMetrics)[0].replace('_', ' ')}</div>
                            </div>
                        ` : ''}
                    </div>
                    
                    ${rec.gpt_analysis ? `
                        <div class="gpt-analysis">
                            <h5><i class="fas fa-robot"></i> GPT Analysis</h5>
                            <p>${rec.gpt_analysis}</p>
                        </div>
                    ` : `
                        <div class="no-gpt-notice">
                            <p><i class="fas fa-info-circle"></i> Add OpenAI API key for AI-powered insights</p>
                        </div>
                    `}
                    
                    <div class="justification">
                        <h5><i class="fas fa-chart-bar"></i> Statistical Justification</h5>
                        <p>${rec.justification}</p>
                    </div>
                    
                    <div class="model-actions">
                        <button class="action-btn tune-btn" onclick="openTuningModal('${rec.model}')" ${!rec.can_tune ? 'disabled' : ''}>
                            <i class="fas fa-sliders-h"></i>
                            ${rec.can_tune ? 'Tune Parameters' : 'No Tuning Available'}
                        </button>
                        <button class="action-btn finalize-btn" onclick="finalizeModel('${rec.model}')">
                            <i class="fas fa-download"></i>
                            Finalize Model
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

async function openTuningModal(modelName) {
    const modal = document.getElementById('tuningModal');
    const content = document.getElementById('tuningContent');
    
    content.innerHTML = `
        <div class="tuning-interface">
            <div class="tuning-header">
                <h4>Hyperparameter Tuning for ${modelName}</h4>
                <p>Running GridSearchCV optimization...</p>
            </div>
            
            <div class="tuning-progress">
                <div class="progress-bar">
                    <div class="progress-fill" id="tuningProgress" style="width: 0%"></div>
                </div>
                <p id="tuningStatus">Starting hyperparameter optimization...</p>
            </div>
            
            <div class="tuning-results" id="tuningResults" style="display: none;">
                <h5>Optimization Complete!</h5>
                <div class="results-grid">
                    <div class="result-item">
                        <strong>Best Score:</strong>
                        <span id="bestScore">-</span>
                    </div>
                    <div class="result-item">
                        <strong>Combinations:</strong>
                        <span id="combinations">-</span>
                    </div>
                    <div class="result-item">
                        <strong>Time Taken:</strong>
                        <span id="tuningTime">-</span>
                    </div>
                </div>
                <div class="best-params" id="bestParams"></div>
            </div>
        </div>
    `;
    
    modal.classList.remove('hidden');
    
    // Start real tuning
    await performRealTuning(modelName);
}

async function performRealTuning(modelName) {
    const progressBar = document.getElementById('tuningProgress');
    const statusText = document.getElementById('tuningStatus');
    const resultsDiv = document.getElementById('tuningResults');
    
    try {
        // Animate progress while tuning
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 2;
            if (progress > 90) progress = 90;
            progressBar.style.width = progress + '%';
            
            if (progress < 30) {
                statusText.textContent = 'Setting up parameter grid...';
            } else if (progress < 60) {
                statusText.textContent = 'Testing parameter combinations...';
            } else {
                statusText.textContent = 'Cross-validating best parameters...';
            }
        }, 200);
        
        // Make real API call with DataLab integration
        const datasetId = window.datasetInfo ? window.datasetInfo.id : '';
        const tuneUrl = datasetId ? `/ml/api/tune/${datasetId}` : '/ml/tune';
        const response = await fetch(tuneUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: modelName
            })
        });
        
        const data = await response.json();
        
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        window.currentTuningModel = modelName;
        showRealTuningResults(data.tuning_results, data.updated_performance);
        
    } catch (error) {
        statusText.textContent = 'Error: ' + error.message;
        statusText.style.color = 'var(--danger-color)';
    }
}

function showRealTuningResults(results, updatedPerformance) {
    const resultsDiv = document.getElementById('tuningResults');
    const bestScore = document.getElementById('bestScore');
    const combinations = document.getElementById('combinations');
    const tuningTime = document.getElementById('tuningTime');
    const bestParams = document.getElementById('bestParams');
    
    bestScore.textContent = results.best_score.toFixed(3);
    combinations.textContent = results.total_combinations;
    tuningTime.textContent = results.tuning_time.toFixed(1) + 's';
    
    bestParams.innerHTML = `
        <h6>Optimal Parameters:</h6>
        <div class="param-grid">
            ${Object.entries(results.best_params).map(([key, value]) => `
                <div class="param-item">
                    <strong>${key}:</strong> ${value}
                </div>
            `).join('')}
        </div>
    `;
    
    resultsDiv.style.display = 'block';
    window.currentTuningResults = { results, updatedPerformance };
    
    // Automatically update model metrics
    setTimeout(() => {
        applyTuningAutomatically(window.currentTuningModel, updatedPerformance);
        closeTuningModal();
    }, 2000);
}

function applyTuningAutomatically(modelName, updatedPerformance) {
    const modelCard = document.querySelector(`[data-model="${modelName}"]`);
    if (modelCard && updatedPerformance) {
        // Update all metrics
        const metrics = modelCard.querySelectorAll('.metric');
        
        // Update CV Score
        if (metrics[0]) {
            const scoreElement = metrics[0].querySelector('.metric-value');
            scoreElement.textContent = (updatedPerformance.mean_score * 100).toFixed(1) + '%';
            scoreElement.style.color = 'var(--success-color)';
            scoreElement.style.fontWeight = 'bold';
        }
        
        // Update Std Dev
        if (metrics[1]) {
            const stdElement = metrics[1].querySelector('.metric-value');
            stdElement.textContent = '±' + (updatedPerformance.std_score * 100).toFixed(1) + '%';
        }
        
        // Update Training Time
        if (metrics[2]) {
            const timeElement = metrics[2].querySelector('.metric-value');
            timeElement.textContent = updatedPerformance.training_time.toFixed(2) + 's';
        }
        
        // Add tuned indicator
        const modelHeader = modelCard.querySelector('.model-details h4');
        if (!modelHeader.textContent.includes('🔧')) {
            modelHeader.innerHTML += ' <span style="color: var(--success-color);">🔧 Tuned</span>';
        }
        
        showNotification(`${modelName} metrics updated with optimized parameters!`, 'success');
    }
}

function closeTuningModal() {
    document.getElementById('tuningModal').classList.add('hidden');
}

async function finalizeModel(modelName) {
    try {
        // Get model parameters (from tuning results if available)
        let modelParams = {};
        if (window.currentTuningResults && window.currentTuningModel === modelName) {
            modelParams = window.currentTuningResults.results.best_params;
        }
        
        // Get dataset info
        const datasetInfo = currentData ? currentData.dataset_analysis : {};
        
        // Use DataLab endpoints with dataset ID
        const datasetId = window.datasetInfo ? window.datasetInfo.id : '';
        const exportUrl = datasetId ? `/ml/api/export-notebook/${datasetId}` : '/ml/export-notebook';
        const response = await fetch(exportUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: modelName,
                model_params: modelParams,
                dataset_info: datasetInfo
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Download notebook
        const blob = new Blob([JSON.stringify(data.notebook_content, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        // Open JupyterLite in new tab with better handling
        try {
            const jupyterWindow = window.open('/ml/jupyterlite', '_blank', 'width=1200,height=800,resizable=yes,scrollbars=yes');
            
            if (jupyterWindow) {
                // Focus the new window
                jupyterWindow.focus();
                showNotification(`${modelName} notebook downloaded! JupyterLite opened - upload the notebook to start coding.`, 'success');
            } else {
                // Popup was blocked
                showNotification(`${modelName} notebook downloaded! Please allow popups and click here to open JupyterLite.`, 'warning');
                
                // Add a click handler to try again
                setTimeout(() => {
                    const notification = document.querySelector('.notification:last-child');
                    if (notification) {
                        notification.style.cursor = 'pointer';
                        notification.onclick = () => {
                            window.open('/ml/jupyterlite', '_blank');
                            notification.remove();
                        };
                    }
                }, 100);
            }
        } catch (error) {
            console.error('Error opening JupyterLite:', error);
            showNotification(`${modelName} notebook downloaded! Click "Python Notebook" in the sidebar to open JupyterLite.`, 'info');
        }
        
        // Store notebook info for easy access
        if (typeof(Storage) !== "undefined") {
            localStorage.setItem('datalab_latest_notebook', JSON.stringify({
                filename: data.filename,
                model_name: modelName,
                created_at: new Date().toISOString(),
                content: data.notebook_content
            }));
        }
        
    } catch (error) {
        showNotification('Error exporting notebook: ' + error.message, 'error');
    }
}

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? 'var(--success-color)' : type === 'error' ? 'var(--danger-color)' : 'var(--primary-color)'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        z-index: 3000;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        animation: slideInRight 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .model-card.selected {
        border-color: var(--success-color) !important;
        box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.2) !important;
    }
    
    .tuning-interface {
        text-align: center;
    }
    
    .tuning-header h4 {
        margin-bottom: 0.5rem;
        color: var(--text-primary);
    }
    
    .tuning-header p {
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    .tuning-progress {
        margin-bottom: 2rem;
    }
    
    .tuning-progress p {
        margin-top: 1rem;
        color: var(--text-secondary);
    }
    
    .results-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .result-item {
        background: var(--light-color);
        padding: 1rem;
        border-radius: var(--radius-md);
        text-align: center;
    }
    
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .param-item {
        background: var(--light-color);
        padding: 0.5rem;
        border-radius: var(--radius-sm);
        font-size: 0.875rem;
    }
    
    .finalize-btn {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
    }
    
    .finalize-btn:hover {
        background: linear-gradient(135deg, var(--primary-dark), #1e40af);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
`;
document.head.appendChild(style);