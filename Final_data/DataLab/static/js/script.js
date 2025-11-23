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
    console.log('displayResults called with:', data);

    if (!data || !data.recommendations || data.recommendations.length === 0) {
        console.error('No recommendations to display');
        showError('No model recommendations available. Please check your dataset.');
        return;
    }

    try {
        displayBestModel(data.best_model);
        displayDatasetStats(data.dataset_analysis);
        displayPerformanceChart(data.recommendations);
        displayExplanations(data.explanations);
        displayModelRecommendations(data.recommendations);

        const results = document.getElementById('results');
        results.classList.remove('hidden');
        results.classList.add('fade-in');

        console.log('Results displayed successfully');
    } catch (error) {
        console.error('Error displaying results:', error);
        showError('Error displaying results: ' + error.message);
    }
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

function displayExplanations(explanations) {
    const explanationsContent = document.getElementById('explanationsContent');

    if (!explanations || Object.keys(explanations).length === 0) {
        console.log('No explanations available');
        document.getElementById('explanationsSection').style.display = 'none';
        return;
    }

    // Convert markdown-like formatting to HTML
    function formatExplanation(text) {
        if (!text) return '';

        // Convert markdown headers
        text = text.replace(/^### (.*$)/gim, '<h4>$1</h4>');
        text = text.replace(/^## (.*$)/gim, '<h3>$1</h3>');
        text = text.replace(/^# (.*$)/gim, '<h2>$1</h2>');

        // Convert bold text
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert bullet points
        text = text.replace(/^- (.*$)/gim, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

        // Convert line breaks
        text = text.replace(/\n\n/g, '</p><p>');
        text = '<p>' + text + '</p>';

        return text;
    }

    // Order of sections for display
    const sectionOrder = ['summary', 'sampling', 'task_detection', 'best_model', 'cross_validation', 'hyperparameter_tuning'];

    let htmlContent = '<div class="explanation-grid">';

    sectionOrder.forEach(key => {
        if (explanations[key]) {
            htmlContent += `
                <div class="explanation-card">
                    <div class="explanation-content">
                        ${formatExplanation(explanations[key])}
                    </div>
                </div>
            `;
        }
    });

    htmlContent += '</div>';
    explanationsContent.innerHTML = htmlContent;
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
    
    container.innerHTML = recommendations.map((rec, index) => {
        const rank = index + 1;
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

                    <div class="justification">
                        <h5><i class="fas fa-chart-bar"></i> Why This Model?</h5>
                        <p>${rec.justification}</p>
                    </div>

                    <div class="model-explanation">
                        <h5><i class="fas fa-info-circle"></i> Understanding This Result</h5>
                        <p><strong>Performance:</strong> This model achieved a ${(rec.performance.mean_score * 100).toFixed(1)}% accuracy score through 5-fold cross-validation, meaning it was tested 5 times on different portions of your data.</p>
                        <p><strong>Stability:</strong> The standard deviation of ±${(rec.performance.std_score * 100).toFixed(1)}% shows how consistent the model performs across different test sets. Lower values mean more predictable, reliable results.</p>
                        <p><strong>Speed:</strong> Training completed in ${rec.performance.training_time.toFixed(2)} seconds. ${rec.performance.training_time < 1 ? 'This is very fast!' : rec.performance.training_time < 5 ? 'This is reasonably fast.' : 'This takes more time but may offer better accuracy.'}</p>
                        <p><strong>Suitability Score (${rec.suitability_score.toFixed(1)}%):</strong> This score considers multiple factors:</p>
                        <ul>
                            <li>Model accuracy and performance (40%)</li>
                            <li>Training and prediction speed (30%)</li>
                            <li>Result consistency and stability (20%)</li>
                            <li>Model complexity and interpretability (10%)</li>
                        </ul>
                        ${rank === 1 ? '<p style="color: var(--success-color); font-weight: 600;">🏆 This is the recommended model for your dataset based on the best overall balance of accuracy, speed, and reliability.</p>' : ''}
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
    // Store original performance for comparison
    const modelRec = currentData.recommendations.find(r => r.model === modelName);
    window.originalPerformance = modelRec ? modelRec.performance : null;
    
    const modal = document.getElementById('tuningModal');
    const content = document.getElementById('tuningContent');
    
    content.innerHTML = `
        <div class="tuning-interface">
            <div class="tuning-header">
                <h4>Hyperparameter Tuning for ${modelName}</h4>
                <p>Select tuning depth based on your time budget</p>
            </div>
            
            <div class="tuning-options">
                <div class="tuning-option" onclick="startTuning('${modelName}', 'normal')">
                    <div class="option-icon"><i class="fas fa-bolt"></i></div>
                    <h5>Normal Tuning</h5>
                    <p class="option-time">~10-30 seconds</p>
                    <p class="option-desc">Quick optimization with essential parameters</p>
                </div>
                <div class="tuning-option" onclick="startTuning('${modelName}', 'semi_deep')">
                    <div class="option-icon"><i class="fas fa-chart-line"></i></div>
                    <h5>Semi-Deep Tuning</h5>
                    <p class="option-time">~30-120 seconds</p>
                    <p class="option-desc">Moderate comprehensive search with many combinations</p>
                </div>
                <div class="tuning-option" onclick="startTuning('${modelName}', 'deep')">
                    <div class="option-icon"><i class="fas fa-brain"></i></div>
                    <h5>Deep Tuning</h5>
                    <p class="option-time">~2-5 minutes</p>
                    <p class="option-desc">Highly comprehensive search for optimal results</p>
                </div>
            </div>
            
            <div class="tuning-progress" id="tuningProgress" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill" id="tuningProgressBar" style="width: 0%"></div>
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
}

function startTuning(modelName, tuningLevel) {
    document.querySelector('.tuning-options').style.display = 'none';
    document.getElementById('tuningProgress').style.display = 'block';
    performRealTuning(modelName, tuningLevel);
}

async function performRealTuning(modelName, tuningLevel) {
    const progressBar = document.getElementById('tuningProgressBar');
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
                model_name: modelName,
                tuning_level: tuningLevel,
                original_performance: window.originalPerformance
            })
        });
        
        const data = await response.json();
        
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        window.currentTuningModel = modelName;
        window.currentTuningLevel = tuningLevel;
        showRealTuningResults(data.tuning_results, data.updated_performance, data.improved, data.improvement_pct, data.original_score, data.new_score, tuningLevel);
        
    } catch (error) {
        statusText.textContent = 'Error: ' + error.message;
        statusText.style.color = 'var(--danger-color)';
    }
}

function showRealTuningResults(results, updatedPerformance, improved, improvementPct, originalScore, newScore, tuningLevel) {
    const resultsDiv = document.getElementById('tuningResults');
    const bestScore = document.getElementById('bestScore');
    const combinations = document.getElementById('combinations');
    const tuningTime = document.getElementById('tuningTime');
    const bestParams = document.getElementById('bestParams');
    
    bestScore.textContent = results.best_score.toFixed(3);
    combinations.textContent = results.total_combinations;
    tuningTime.textContent = results.tuning_time.toFixed(1) + 's';
    
    // Smart suggestions based on tuning level and results
    let suggestionHtml = '';
    if (improved) {
        if (tuningLevel === 'normal') {
            suggestionHtml = `<div class="improvement-success">
                <i class="fas fa-check-circle"></i>
                <strong>Performance Improved!</strong>
                <p>Score increased from ${(originalScore * 100).toFixed(2)}% to ${(newScore * 100).toFixed(2)}%</p>
                <p class="improvement-pct">+${improvementPct.toFixed(2)}% improvement</p>
                <p class="suggestion">Try Semi-Deep Tuning for potentially better results!</p>
            </div>`;
        } else if (tuningLevel === 'semi_deep') {
            suggestionHtml = `<div class="improvement-success">
                <i class="fas fa-check-circle"></i>
                <strong>Performance Improved!</strong>
                <p>Score increased from ${(originalScore * 100).toFixed(2)}% to ${(newScore * 100).toFixed(2)}%</p>
                <p class="improvement-pct">+${improvementPct.toFixed(2)}% improvement</p>
                <p class="suggestion">Consider Deep Tuning for maximum optimization!</p>
            </div>`;
        } else {
            suggestionHtml = `<div class="improvement-success">
                <i class="fas fa-check-circle"></i>
                <strong>Performance Improved!</strong>
                <p>Score increased from ${(originalScore * 100).toFixed(2)}% to ${(newScore * 100).toFixed(2)}%</p>
                <p class="improvement-pct">+${improvementPct.toFixed(2)}% improvement</p>
                <p class="suggestion">Model fully optimized with Deep Tuning!</p>
            </div>`;
        }
    } else {
        if (tuningLevel === 'normal') {
            suggestionHtml = `<div class="improvement-warning">
                <i class="fas fa-info-circle"></i>
                <strong>No Improvement</strong>
                <p>Normal tuning did not improve performance. Original score: ${(originalScore * 100).toFixed(2)}%</p>
                <p class="suggestion">Recommendation: Try Semi-Deep Tuning for more thorough search.</p>
            </div>`;
        } else if (tuningLevel === 'semi_deep') {
            suggestionHtml = `<div class="improvement-warning">
                <i class="fas fa-info-circle"></i>
                <strong>No Improvement</strong>
                <p>Semi-Deep tuning did not improve performance. Original score: ${(originalScore * 100).toFixed(2)}%</p>
                <p class="suggestion">Recommendation: Try Deep Tuning for comprehensive optimization.</p>
            </div>`;
        } else {
            suggestionHtml = `<div class="improvement-warning">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>No Improvement</strong>
                <p>Deep tuning did not improve performance. Original score: ${(originalScore * 100).toFixed(2)}%</p>
                <p class="suggestion">Model is already well-optimized. Consider feature engineering or trying different models.</p>
            </div>`;
        }
    }
    const improvementHtml = suggestionHtml;
    
    bestParams.innerHTML = improvementHtml + `
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
    window.currentTuningResults = { results, updatedPerformance, improved };
    
    // Show smart popup notification
    if (improved) {
        const levelName = tuningLevel === 'normal' ? 'Normal' : tuningLevel === 'semi_deep' ? 'Semi-Deep' : 'Deep';
        showNotification(`${window.currentTuningModel} improved by ${improvementPct.toFixed(1)}%! ${levelName} tuning successful.`, 'success');
    } else {
        if (tuningLevel === 'normal') {
            showNotification(`${window.currentTuningModel}: No improvement with Normal tuning. Try Semi-Deep tuning.`, 'warning');
        } else if (tuningLevel === 'semi_deep') {
            showNotification(`${window.currentTuningModel}: No improvement with Semi-Deep tuning. Try Deep tuning.`, 'warning');
        } else {
            showNotification(`${window.currentTuningModel}: Model already optimized. No deep tuning needed.`, 'info');
        }
    }
    
    // Don't auto-close modal - let user close it manually
    // Automatically update model metrics in background
    applyTuningAutomatically(window.currentTuningModel, updatedPerformance, improved);
}

function applyTuningAutomatically(modelName, updatedPerformance, improved) {
    const modelCard = document.querySelector(`[data-model="${modelName}"]`);
    if (modelCard && updatedPerformance) {
        // Update all metrics in the card
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
        
        // Add tuned indicator ONLY if improved
        if (improved) {
            const modelHeader = modelCard.querySelector('.model-details h4');
            if (!modelHeader.textContent.includes('Tuned')) {
                modelHeader.innerHTML += ' <span style="color: var(--success-color); font-size: 0.8em;">✓ Tuned</span>';
            }
        }
        
        // Update currentData with new performance
        if (currentData && currentData.recommendations) {
            const modelIndex = currentData.recommendations.findIndex(r => r.model === modelName);
            if (modelIndex !== -1) {
                currentData.recommendations[modelIndex].performance = updatedPerformance;
                
                // Recalculate suitability score
                const newSuitability = calculateSuitabilityScore(modelName, updatedPerformance);
                currentData.recommendations[modelIndex].suitability_score = newSuitability;
                
                // Re-sort recommendations
                currentData.recommendations.sort((a, b) => b.suitability_score - a.suitability_score);
                
                // Update best model if changed
                const newBestModel = currentData.recommendations[0];
                currentData.best_model = {
                    name: newBestModel.model,
                    score: newBestModel.suitability_score,
                    performance: newBestModel.performance,
                    why_best: `Achieved highest suitability score of ${newBestModel.suitability_score.toFixed(1)}% with ${newBestModel.performance.mean_score.toFixed(3)} CV score and ${newBestModel.performance.training_time.toFixed(2)}s training time.`
                };
                
                // Update best model display
                displayBestModel(currentData.best_model);
                
                // Update performance chart
                displayPerformanceChart(currentData.recommendations);
            }
        }
        
        const message = improved ? 
            `${modelName} metrics updated! Rankings refreshed with improved performance.` :
            `${modelName} metrics updated. No performance improvement from tuning.`;
        showNotification(message, improved ? 'success' : 'info');
    }
}

function calculateSuitabilityScore(modelName, performance) {
    // Simple suitability calculation based on performance
    let baseScore = performance.mean_score * 100;
    
    // Adjust for stability
    if (performance.std_score < 0.05) baseScore += 5;
    else if (performance.std_score > 0.15) baseScore -= 5;
    
    // Adjust for training time
    if (performance.training_time < 1) baseScore += 3;
    else if (performance.training_time > 10) baseScore -= 3;
    
    return Math.max(0, Math.min(100, baseScore));
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
        
        // Download the notebook file
        const blob = new Blob([data.notebook_content], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        // Show success message with instructions
        showNotification(`${modelName} notebook downloaded! Open JupyterLite and upload the file to start coding.`, 'success');
        
        // Open JupyterLite in new tab after a short delay
        setTimeout(() => {
            const jupyterWindow = window.open('/ml/jupyterlite', '_blank', 'width=1200,height=800,resizable=yes,scrollbars=yes');
            if (jupyterWindow) {
                jupyterWindow.focus();
            }
        }, 500);
        
    } catch (error) {
        showNotification('Error exporting notebook: ' + error.message, 'error');
    }
}

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? 'var(--success-color)' : type === 'error' ? 'var(--danger-color)' : type === 'warning' ? '#f59e0b' : 'var(--primary-color)'};
        color: white;
        padding: 1rem 1.5rem;
        padding-right: 3rem;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        z-index: 3000;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        animation: slideInRight 0.3s ease-out;
        max-width: 400px;
    `;
    
    document.body.appendChild(notification);
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
    
    .improvement-success {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .improvement-success i {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .improvement-success strong {
        display: block;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .improvement-pct {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    .improvement-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .improvement-warning i {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .improvement-warning strong {
        display: block;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .tuning-options {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .tuning-option {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 2px solid #dee2e6;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .tuning-option:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border-color: var(--primary-color);
    }
    
    .option-icon {
        font-size: 2.5rem;
        color: var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .tuning-option h5 {
        margin: 0.5rem 0;
        color: var(--text-primary);
    }
    
    .option-time {
        color: var(--success-color);
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .option-desc {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin: 0;
    }
    
    .suggestion {
        margin-top: 1rem;
        padding: 0.75rem;
        background: rgba(255,255,255,0.2);
        border-radius: var(--radius-sm);
        font-weight: bold;
    }
    
    .notification-close {
        position: absolute;
        right: 0.5rem;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(255,255,255,0.2);
        border: none;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s;
    }
    
    .notification-close:hover {
        background: rgba(255,255,255,0.3);
    }
    
    .notification {
        position: relative;
    }
`;
document.head.appendChild(style);