// Workflow Management JavaScript

class WorkflowManager {
    constructor() {
        this.datasetId = window.datasetInfo.id;
        this.columns = [];
        this.currentTab = 'builder';
        this.pipeline = {
            steps: [],
            nextStepId: 1
        };
        this.selectedStepId = null;
        this.pipelines = [];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadDatasetColumns();
        this.loadPipelines();
        this.showTab('builder');
        this.setupDragAndDrop();
    }
    
    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.currentTarget.dataset.tab;
                this.showTab(tab);
            });
        });
        
        // Builder controls
        document.getElementById('save-pipeline-btn').addEventListener('click', () => {
            this.savePipeline();
        });
        
        document.getElementById('clear-pipeline-btn').addEventListener('click', () => {
            this.clearPipeline();
        });
        
        // Execution controls
        document.getElementById('execute-pipeline-btn').addEventListener('click', () => {
            this.executePipeline();
        });
        
        // Export controls
        document.getElementById('export-notebook-btn').addEventListener('click', () => {
            this.exportNotebook();
        });
        
        document.getElementById('export-docs-btn').addEventListener('click', () => {
            this.exportDocumentation();
        });
        
        document.getElementById('share-pipeline-btn').addEventListener('click', () => {
            this.showShareModal();
        });
        
        // Version controls
        document.getElementById('create-version-btn').addEventListener('click', () => {
            this.showVersionModal();
        });
        
        // Documentation controls
        document.getElementById('generate-docs-btn').addEventListener('click', () => {
            this.generateDocumentation();
        });
        
        // Modal controls
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.closeModal(e.target.closest('.modal'));
            });
        });
        
        document.getElementById('generate-share-btn').addEventListener('click', () => {
            this.generateShare();
        });
        
        document.getElementById('save-version-btn').addEventListener('click', () => {
            this.saveVersion();
        });
        
        document.getElementById('copy-share-btn').addEventListener('click', () => {
            this.copyShareContent();
        });
        
        // Documentation tabs
        document.querySelectorAll('.doc-tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.showDocTab(e.target.dataset.docTab);
            });
        });
    }
    
    showTab(tabName) {
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Show/hide tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.style.display = 'none';
        });
        document.getElementById(`${tabName}-tab`).style.display = 'block';
        
        this.currentTab = tabName;
        
        // Load data for specific tabs
        if (tabName === 'execution' || tabName === 'export' || tabName === 'version' || tabName === 'documentation') {
            this.populatePipelineSelectors();
        }
    }
    
    setupDragAndDrop() {
        // Make step items draggable
        document.querySelectorAll('.step-item').forEach(item => {
            item.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', e.target.dataset.stepType);
                e.target.classList.add('dragging');
            });
            
            item.addEventListener('dragend', (e) => {
                e.target.classList.remove('dragging');
            });
        });
        
        // Setup drop zone
        const canvas = document.getElementById('pipeline-canvas');
        
        canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            canvas.classList.add('drag-over');
        });
        
        canvas.addEventListener('dragleave', () => {
            canvas.classList.remove('drag-over');
        });
        
        canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            canvas.classList.remove('drag-over');
            
            const stepType = e.dataTransfer.getData('text/plain');
            this.addStep(stepType);
        });
    }
    
    async loadDatasetColumns() {
        try {
            const response = await fetch(`/api/dataset/${this.datasetId}/info`);
            const data = await response.json();
            
            if (data.success) {
                this.columns = data.columns;
            } else {
                console.error('Failed to load dataset columns');
            }
        } catch (error) {
            console.error('Error loading columns:', error);
        }
    }
    
    addStep(stepType) {
        const step = {
            id: this.pipeline.nextStepId++,
            type: stepType,
            name: this.getStepName(stepType),
            description: this.getStepDescription(stepType),
            parameters: this.getDefaultParameters(stepType)
        };
        
        this.pipeline.steps.push(step);
        this.renderPipeline();
        this.selectStep(step.id);
    }
    
    getStepName(stepType) {
        const names = {
            'missing_value_handling': 'Handle Missing Values',
            'outlier_removal': 'Remove Outliers',
            'data_cleaning': 'Clean Data',
            'scaling': 'Scale Features',
            'encoding': 'Encode Categories',
            'feature_creation': 'Create Features',
            'feature_selection': 'Select Features',
            'dimensionality_reduction': 'Reduce Dimensions'
        };
        return names[stepType] || stepType.replace('_', ' ').toUpperCase();
    }
    
    getStepDescription(stepType) {
        const descriptions = {
            'missing_value_handling': 'Handle missing values using various strategies',
            'outlier_removal': 'Remove or treat outliers in the data',
            'data_cleaning': 'Clean data by removing duplicates and empty rows/columns',
            'scaling': 'Scale numerical features to a common range',
            'encoding': 'Encode categorical variables to numerical format',
            'feature_creation': 'Create new features from existing ones',
            'feature_selection': 'Select the most important features',
            'dimensionality_reduction': 'Reduce the number of features while preserving information'
        };
        return descriptions[stepType] || 'Preprocessing step';
    }
    
    getDefaultParameters(stepType) {
        const defaults = {
            'missing_value_handling': {
                strategy: 'mean',
                columns: []
            },
            'outlier_removal': {
                method: 'iqr',
                columns: []
            },
            'data_cleaning': {
                remove_duplicates: true,
                remove_empty_rows: false,
                remove_empty_columns: false
            },
            'scaling': {
                method: 'standard',
                columns: []
            },
            'encoding': {
                method: 'onehot',
                columns: []
            },
            'feature_creation': {
                operation: 'polynomial',
                columns: [],
                degree: 2
            },
            'feature_selection': {
                method: 'correlation',
                target: '',
                n_features: 10
            },
            'dimensionality_reduction': {
                method: 'pca',
                n_components: 2
            }
        };
        return defaults[stepType] || {};
    }
    
    renderPipeline() {
        const canvas = document.getElementById('pipeline-canvas');
        
        if (this.pipeline.steps.length === 0) {
            canvas.innerHTML = `
                <div class="drop-zone">
                    <i class="fas fa-arrow-down"></i>
                    <p>Drop pipeline steps here</p>
                </div>
            `;
            return;
        }
        
        canvas.innerHTML = '';
        
        this.pipeline.steps.forEach((step, index) => {
            const stepElement = this.createStepElement(step, index + 1);
            canvas.appendChild(stepElement);
        });
    }
    
    createStepElement(step, stepNumber) {
        const div = document.createElement('div');
        div.className = 'pipeline-step';
        div.dataset.stepId = step.id;
        
        const parametersSummary = this.getParametersSummary(step);
        
        div.innerHTML = `
            <div class="step-header">
                <div class="step-title">
                    <div class="step-number">${stepNumber}</div>
                    <i class="fas fa-${this.getStepIcon(step.type)}"></i>
                    ${step.name}
                </div>
                <div class="step-actions">
                    <button class="step-btn edit" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="step-btn delete" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="step-description">${step.description}</div>
            <div class="step-parameters">${parametersSummary}</div>
        `;
        
        // Add event listeners
        div.addEventListener('click', () => {
            this.selectStep(step.id);
        });
        
        div.querySelector('.edit').addEventListener('click', (e) => {
            e.stopPropagation();
            this.selectStep(step.id);
        });
        
        div.querySelector('.delete').addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteStep(step.id);
        });
        
        return div;
    }
    
    getStepIcon(stepType) {
        const icons = {
            'missing_value_handling': 'fill-drip',
            'outlier_removal': 'search-minus',
            'data_cleaning': 'broom',
            'scaling': 'balance-scale',
            'encoding': 'code',
            'feature_creation': 'plus-circle',
            'feature_selection': 'funnel-dollar',
            'dimensionality_reduction': 'compress-alt'
        };
        return icons[stepType] || 'cog';
    }
    
    getParametersSummary(step) {
        const params = step.parameters;
        const summaries = [];
        
        if (params.strategy) {
            summaries.push(`Strategy: ${params.strategy}`);
        }
        if (params.method) {
            summaries.push(`Method: ${params.method}`);
        }
        if (params.columns && params.columns.length > 0) {
            summaries.push(`Columns: ${params.columns.slice(0, 3).join(', ')}${params.columns.length > 3 ? '...' : ''}`);
        }
        if (params.target) {
            summaries.push(`Target: ${params.target}`);
        }
        
        return summaries.join(' | ') || 'Default parameters';
    }
    
    selectStep(stepId) {
        // Update visual selection
        document.querySelectorAll('.pipeline-step').forEach(step => {
            step.classList.remove('selected');
        });
        
        const stepElement = document.querySelector(`[data-step-id="${stepId}"]`);
        if (stepElement) {
            stepElement.classList.add('selected');
        }
        
        this.selectedStepId = stepId;
        this.showStepConfiguration(stepId);
    }
    
    showStepConfiguration(stepId) {
        const step = this.pipeline.steps.find(s => s.id === stepId);
        if (!step) return;
        
        const configDiv = document.getElementById('step-config');
        configDiv.innerHTML = this.generateStepConfigForm(step);
        
        // Add event listeners for form changes
        configDiv.addEventListener('change', (e) => {
            this.updateStepParameter(stepId, e.target);
        });
        
        configDiv.addEventListener('input', (e) => {
            this.updateStepParameter(stepId, e.target);
        });
    }
    
    generateStepConfigForm(step) {
        let formHtml = `
            <div class="config-header">
                <div class="config-title">
                    <i class="fas fa-${this.getStepIcon(step.type)}"></i>
                    ${step.name}
                </div>
                <div class="config-description">${step.description}</div>
            </div>
            <div class="config-form">
        `;
        
        // Generate form fields based on step type
        if (step.type === 'missing_value_handling') {
            formHtml += `
                <div class="form-group">
                    <label for="strategy">Strategy:</label>
                    <select id="strategy" class="form-control" data-param="strategy">
                        <option value="mean" ${step.parameters.strategy === 'mean' ? 'selected' : ''}>Mean</option>
                        <option value="median" ${step.parameters.strategy === 'median' ? 'selected' : ''}>Median</option>
                        <option value="mode" ${step.parameters.strategy === 'mode' ? 'selected' : ''}>Mode</option>
                        <option value="drop" ${step.parameters.strategy === 'drop' ? 'selected' : ''}>Drop Rows</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="columns">Columns (leave empty for all numeric):</label>
                    <select multiple id="columns" class="form-control multi-select" data-param="columns">
                        ${this.generateColumnOptions(step.parameters.columns)}
                    </select>
                </div>
            `;
        } else if (step.type === 'scaling') {
            formHtml += `
                <div class="form-group">
                    <label for="method">Method:</label>
                    <select id="method" class="form-control" data-param="method">
                        <option value="standard" ${step.parameters.method === 'standard' ? 'selected' : ''}>Standard (Z-score)</option>
                        <option value="minmax" ${step.parameters.method === 'minmax' ? 'selected' : ''}>Min-Max</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="columns">Columns (leave empty for all numeric):</label>
                    <select multiple id="columns" class="form-control multi-select" data-param="columns">
                        ${this.generateColumnOptions(step.parameters.columns)}
                    </select>
                </div>
            `;
        } else if (step.type === 'encoding') {
            formHtml += `
                <div class="form-group">
                    <label for="method">Method:</label>
                    <select id="method" class="form-control" data-param="method">
                        <option value="onehot" ${step.parameters.method === 'onehot' ? 'selected' : ''}>One-Hot Encoding</option>
                        <option value="label" ${step.parameters.method === 'label' ? 'selected' : ''}>Label Encoding</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="columns">Columns (leave empty for all categorical):</label>
                    <select multiple id="columns" class="form-control multi-select" data-param="columns">
                        ${this.generateColumnOptions(step.parameters.columns)}
                    </select>
                </div>
            `;
        } else if (step.type === 'outlier_removal') {
            formHtml += `
                <div class="form-group">
                    <label for="method">Method:</label>
                    <select id="method" class="form-control" data-param="method">
                        <option value="iqr" ${step.parameters.method === 'iqr' ? 'selected' : ''}>IQR Method</option>
                        <option value="zscore" ${step.parameters.method === 'zscore' ? 'selected' : ''}>Z-Score</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="columns">Columns (leave empty for all numeric):</label>
                    <select multiple id="columns" class="form-control multi-select" data-param="columns">
                        ${this.generateColumnOptions(step.parameters.columns)}
                    </select>
                </div>
            `;
        } else if (step.type === 'feature_creation') {
            formHtml += `
                <div class="form-group">
                    <label for="operation">Operation:</label>
                    <select id="operation" class="form-control" data-param="operation">
                        <option value="polynomial" ${step.parameters.operation === 'polynomial' ? 'selected' : ''}>Polynomial Features</option>
                        <option value="interaction" ${step.parameters.operation === 'interaction' ? 'selected' : ''}>Interaction Features</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="degree">Degree (for polynomial):</label>
                    <select id="degree" class="form-control" data-param="degree">
                        <option value="2" ${step.parameters.degree === 2 ? 'selected' : ''}>2</option>
                        <option value="3" ${step.parameters.degree === 3 ? 'selected' : ''}>3</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="columns">Columns:</label>
                    <select multiple id="columns" class="form-control multi-select" data-param="columns">
                        ${this.generateColumnOptions(step.parameters.columns)}
                    </select>
                </div>
            `;
        } else if (step.type === 'feature_selection') {
            formHtml += `
                <div class="form-group">
                    <label for="method">Method:</label>
                    <select id="method" class="form-control" data-param="method">
                        <option value="correlation" ${step.parameters.method === 'correlation' ? 'selected' : ''}>Correlation</option>
                        <option value="f_score" ${step.parameters.method === 'f_score' ? 'selected' : ''}>F-Score</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="target">Target Column:</label>
                    <select id="target" class="form-control" data-param="target">
                        <option value="">Select target...</option>
                        ${this.generateColumnOptions([step.parameters.target])}
                    </select>
                </div>
                <div class="form-group">
                    <label for="n_features">Number of Features:</label>
                    <input type="number" id="n_features" class="form-control" 
                           value="${step.parameters.n_features || 10}" 
                           min="1" max="50" data-param="n_features">
                </div>
            `;
        } else if (step.type === 'data_cleaning') {
            formHtml += `
                <div class="form-group">
                    <div class="checkbox-group">
                        <label>
                            <input type="checkbox" data-param="remove_duplicates" 
                                   ${step.parameters.remove_duplicates ? 'checked' : ''}>
                            Remove duplicate rows
                        </label>
                        <label>
                            <input type="checkbox" data-param="remove_empty_rows" 
                                   ${step.parameters.remove_empty_rows ? 'checked' : ''}>
                            Remove empty rows
                        </label>
                        <label>
                            <input type="checkbox" data-param="remove_empty_columns" 
                                   ${step.parameters.remove_empty_columns ? 'checked' : ''}>
                            Remove empty columns
                        </label>
                    </div>
                </div>
            `;
        }
        
        formHtml += `</div>`;
        return formHtml;
    }
    
    generateColumnOptions(selectedColumns = []) {
        return this.columns.map(col => {
            const selected = selectedColumns.includes(col.name) ? 'selected' : '';
            return `<option value="${col.name}" ${selected}>${col.name} (${col.type})</option>`;
        }).join('');
    }
    
    updateStepParameter(stepId, element) {
        const step = this.pipeline.steps.find(s => s.id === stepId);
        if (!step) return;
        
        const paramName = element.dataset.param;
        let value;
        
        if (element.type === 'checkbox') {
            value = element.checked;
        } else if (element.multiple) {
            value = Array.from(element.selectedOptions).map(option => option.value);
        } else if (element.type === 'number') {
            value = parseInt(element.value);
        } else {
            value = element.value;
        }
        
        step.parameters[paramName] = value;
        
        // Update the pipeline display
        this.renderPipeline();
        this.selectStep(stepId);
    }
    
    deleteStep(stepId) {
        this.pipeline.steps = this.pipeline.steps.filter(step => step.id !== stepId);
        this.renderPipeline();
        
        // Clear configuration if this step was selected
        if (this.selectedStepId === stepId) {
            this.selectedStepId = null;
            document.getElementById('step-config').innerHTML = `
                <div class="no-selection">
                    <i class="fas fa-hand-pointer"></i>
                    <p>Select a step to configure</p>
                </div>
            `;
        }
    }
    
    clearPipeline() {
        if (confirm('Are you sure you want to clear all pipeline steps?')) {
            this.pipeline.steps = [];
            this.selectedStepId = null;
            this.renderPipeline();
            document.getElementById('step-config').innerHTML = `
                <div class="no-selection">
                    <i class="fas fa-hand-pointer"></i>
                    <p>Select a step to configure</p>
                </div>
            `;
        }
    }
    
    async savePipeline() {
        const name = document.getElementById('pipeline-name').value.trim();
        const description = document.getElementById('pipeline-description').value.trim();
        
        if (!name) {
            alert('Please enter a pipeline name');
            return;
        }
        
        if (this.pipeline.steps.length === 0) {
            alert('Please add at least one step to the pipeline');
            return;
        }
        
        this.showLoading(true);
        
        try {
            const response = await fetch(`/workflow/api/${this.datasetId}/create_pipeline`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: name,
                    description: description,
                    steps: this.pipeline.steps
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert('Pipeline saved successfully!');
                this.loadPipelines();
            } else {
                alert('Error saving pipeline: ' + data.message);
            }
        } catch (error) {
            console.error('Error saving pipeline:', error);
            alert('Error saving pipeline');
        } finally {
            this.showLoading(false);
        }
    }
    
    async loadPipelines() {
        try {
            const response = await fetch(`/workflow/api/${this.datasetId}/pipelines`);
            const data = await response.json();
            
            if (data.success) {
                this.pipelines = data.pipelines;
            }
        } catch (error) {
            console.error('Error loading pipelines:', error);
        }
    }
    
    populatePipelineSelectors() {
        const selectors = ['execution-pipeline', 'export-pipeline', 'version-pipeline', 'doc-pipeline'];
        
        selectors.forEach(selectorId => {
            const selector = document.getElementById(selectorId);
            if (selector) {
                // Keep first option
                const firstOption = selector.querySelector('option');
                selector.innerHTML = '';
                if (firstOption) {
                    selector.appendChild(firstOption);
                }
                
                // Add pipeline options
                this.pipelines.forEach(pipeline => {
                    const option = document.createElement('option');
                    option.value = pipeline.id;
                    option.textContent = `${pipeline.name} (v${pipeline.version})`;
                    selector.appendChild(option);
                });
            }
        });
    }
    
    async executePipeline() {
        const pipelineId = document.getElementById('execution-pipeline').value;
        
        if (!pipelineId) {
            alert('Please select a pipeline to execute');
            return;
        }
        
        this.showLoading(true);
        const startTime = Date.now();
        
        try {
            const response = await fetch(`/workflow/api/pipeline/${pipelineId}/execute`, {
                method: 'POST'
            });
            
            const data = await response.json();
            const endTime = Date.now();
            const executionTime = endTime - startTime;
            
            if (data.success) {
                this.displayExecutionResults(data, executionTime);
            } else {
                this.displayExecutionError(data, executionTime);
            }
        } catch (error) {
            console.error('Error executing pipeline:', error);
            this.displayExecutionError({ message: error.message }, Date.now() - startTime);
        } finally {
            this.showLoading(false);
        }
    }
    
    displayExecutionResults(data, executionTime) {
        const resultsDiv = document.getElementById('execution-results');
        resultsDiv.style.display = 'block';
        
        // Update status
        const statusDiv = document.querySelector('.execution-status');
        statusDiv.className = 'execution-status success';
        statusDiv.textContent = '✅ Success';
        
        // Update metrics
        document.getElementById('input-shape').textContent = 
            `${data.execution_record.input_shape[0]} × ${data.execution_record.input_shape[1]}`;
        document.getElementById('output-shape').textContent = 
            `${data.execution_record.output_shape[0]} × ${data.execution_record.output_shape[1]}`;
        document.getElementById('execution-time').textContent = `${executionTime}ms`;
        document.getElementById('execution-status').textContent = 'Success';
        
        // Display execution log
        this.displayExecutionLog(data.execution_record.execution_log);
        
        // Display output preview
        this.displayOutputPreview(data.output_preview);
    }
    
    displayExecutionError(data, executionTime) {
        const resultsDiv = document.getElementById('execution-results');
        resultsDiv.style.display = 'block';
        
        // Update status
        const statusDiv = document.querySelector('.execution-status');
        statusDiv.className = 'execution-status error';
        statusDiv.textContent = '❌ Failed';
        
        // Update metrics
        document.getElementById('input-shape').textContent = '-';
        document.getElementById('output-shape').textContent = '-';
        document.getElementById('execution-time').textContent = `${executionTime}ms`;
        document.getElementById('execution-status').textContent = 'Failed';
        
        // Display error log
        const logContainer = document.getElementById('log-container');
        logContainer.innerHTML = `
            <div class="log-entry error">
                <div class="log-timestamp">${new Date().toISOString()}</div>
                <div>ERROR: ${data.message}</div>
            </div>
        `;
        
        // Clear output preview
        document.getElementById('output-table-container').innerHTML = '<p class="text-muted">No output available</p>';
    }
    
    displayExecutionLog(logs) {
        const logContainer = document.getElementById('log-container');
        logContainer.innerHTML = '';
        
        logs.forEach(log => {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${log.status}`;
            logEntry.innerHTML = `
                <div class="log-timestamp">${new Date(log.timestamp).toLocaleString()}</div>
                <div><strong>${log.step_name}</strong> (${log.step_type})</div>
                <div>Input: ${log.input_shape[0]} × ${log.input_shape[1]} → Output: ${log.output_shape[0]} × ${log.output_shape[1]}</div>
                ${log.error ? `<div style="color: #e53e3e;">Error: ${log.error}</div>` : ''}
            `;
            logContainer.appendChild(logEntry);
        });
    }
    
    displayOutputPreview(data) {
        const container = document.getElementById('output-table-container');
        
        if (!data || data.length === 0) {
            container.innerHTML = '<p class="text-muted">No preview data available</p>';
            return;
        }
        
        const headers = Object.keys(data[0]);
        
        let tableHtml = `
            <table>
                <thead>
                    <tr>
                        ${headers.map(header => `<th>${header}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
        `;
        
        data.forEach(row => {
            tableHtml += '<tr>';
            headers.forEach(header => {
                const value = row[header];
                const displayValue = typeof value === 'number' ? value.toFixed(3) : value;
                tableHtml += `<td>${displayValue}</td>`;
            });
            tableHtml += '</tr>';
        });
        
        tableHtml += '</tbody></table>';
        container.innerHTML = tableHtml;
    }
    
    async exportNotebook() {
        const pipelineId = document.getElementById('export-pipeline').value;
        
        if (!pipelineId) {
            alert('Please select a pipeline to export');
            return;
        }
        
        try {
            const response = await fetch(`/workflow/api/pipeline/${pipelineId}/export_notebook`, {
                method: 'POST'
            });
            
            if (response.ok) {
                // Trigger download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = response.headers.get('content-disposition')?.split('filename=')[1] || 'pipeline.ipynb';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                alert('Notebook exported successfully!');
            } else {
                const data = await response.json();
                alert('Error exporting notebook: ' + data.message);
            }
        } catch (error) {
            console.error('Error exporting notebook:', error);
            alert('Error exporting notebook');
        }
    }
    
    async exportDocumentation() {
        const pipelineId = document.getElementById('export-pipeline').value;
        
        if (!pipelineId) {
            alert('Please select a pipeline to export');
            return;
        }
        
        try {
            const response = await fetch(`/workflow/api/pipeline/${pipelineId}/export_documentation`, {
                method: 'POST'
            });
            
            if (response.ok) {
                // Trigger download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = response.headers.get('content-disposition')?.split('filename=')[1] || 'pipeline_docs.zip';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                alert('Documentation exported successfully!');
            } else {
                const data = await response.json();
                alert('Error exporting documentation: ' + data.message);
            }
        } catch (error) {
            console.error('Error exporting documentation:', error);
            alert('Error exporting documentation');
        }
    }
    
    showShareModal() {
        const pipelineId = document.getElementById('export-pipeline').value;
        
        if (!pipelineId) {
            alert('Please select a pipeline to share');
            return;
        }
        
        document.getElementById('share-modal').style.display = 'flex';
        document.getElementById('share-result').style.display = 'none';
    }
    
    async generateShare() {
        const pipelineId = document.getElementById('export-pipeline').value;
        const shareType = document.querySelector('input[name="share-type"]:checked').value;
        
        try {
            const response = await fetch(`/workflow/api/pipeline/${pipelineId}/share`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ type: shareType })
            });
            
            const data = await response.json();
            
            if (data.success) {
                const resultDiv = document.getElementById('share-result');
                const contentArea = document.getElementById('share-content');
                
                if (shareType === 'link') {
                    contentArea.value = window.location.origin + data.share_link;
                } else {
                    contentArea.value = JSON.stringify(data.config, null, 2);
                }
                
                resultDiv.style.display = 'block';
            } else {
                alert('Error generating share: ' + data.message);
            }
        } catch (error) {
            console.error('Error generating share:', error);
            alert('Error generating share');
        }
    }
    
    copyShareContent() {
        const content = document.getElementById('share-content');
        content.select();
        document.execCommand('copy');
        alert('Content copied to clipboard!');
    }
    
    showVersionModal() {
        const pipelineId = document.getElementById('version-pipeline').value;
        
        if (!pipelineId) {
            alert('Please select a pipeline');
            return;
        }
        
        document.getElementById('version-modal').style.display = 'flex';
    }
    
    async saveVersion() {
        const pipelineId = document.getElementById('version-pipeline').value;
        const notes = document.getElementById('version-notes').value;
        
        try {
            const response = await fetch(`/workflow/api/pipeline/${pipelineId}/version`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ notes: notes })
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert(`Version ${data.new_version} created successfully!`);
                this.closeModal(document.getElementById('version-modal'));
                this.loadPipelines();
                this.displayVersionHistory(pipelineId);
            } else {
                alert('Error creating version: ' + data.message);
            }
        } catch (error) {
            console.error('Error creating version:', error);
            alert('Error creating version');
        }
    }
    
    displayVersionHistory(pipelineId) {
        const pipeline = this.pipelines.find(p => p.id == pipelineId);
        if (!pipeline || !pipeline.version_history) return;
        
        const historyDiv = document.getElementById('version-history');
        const timelineDiv = document.getElementById('version-timeline');
        
        historyDiv.style.display = 'block';
        timelineDiv.innerHTML = '';
        
        // Add current version
        const currentItem = document.createElement('div');
        currentItem.className = 'version-item';
        currentItem.innerHTML = `
            <div class="version-header">
                <div class="version-number">${pipeline.version} (Current)</div>
                <div class="version-date">${new Date(pipeline.updated_at).toLocaleDateString()}</div>
            </div>
            <div class="version-notes">Current version</div>
        `;
        timelineDiv.appendChild(currentItem);
        
        // Add version history
        pipeline.version_history.forEach(version => {
            const versionItem = document.createElement('div');
            versionItem.className = 'version-item';
            versionItem.innerHTML = `
                <div class="version-header">
                    <div class="version-number">${version.version}</div>
                    <div class="version-date">${new Date(version.timestamp).toLocaleDateString()}</div>
                </div>
                <div class="version-notes">${version.notes || 'No notes provided'}</div>
            `;
            timelineDiv.appendChild(versionItem);
        });
    }
    
    async generateDocumentation() {
        const pipelineId = document.getElementById('doc-pipeline').value;
        
        if (!pipelineId) {
            alert('Please select a pipeline');
            return;
        }
        
        const pipeline = this.pipelines.find(p => p.id == pipelineId);
        if (!pipeline) return;
        
        const previewDiv = document.getElementById('documentation-preview');
        previewDiv.style.display = 'block';
        
        // Generate documentation content
        this.displayDocumentation(pipeline);
    }
    
    displayDocumentation(pipeline) {
        // Overview tab
        const overviewContent = document.getElementById('overview-content');
        overviewContent.innerHTML = `
            <h1>${pipeline.name}</h1>
            <p><strong>Description:</strong> ${pipeline.description}</p>
            <p><strong>Version:</strong> ${pipeline.version}</p>
            <p><strong>Created:</strong> ${new Date(pipeline.created_at).toLocaleDateString()}</p>
            <p><strong>Last Updated:</strong> ${new Date(pipeline.updated_at).toLocaleDateString()}</p>
            <p><strong>Number of Steps:</strong> ${pipeline.steps.length}</p>
        `;
        
        // Steps tab
        const stepsContent = document.getElementById('steps-content');
        let stepsHtml = '<h2>Pipeline Steps</h2>';
        
        pipeline.steps.forEach((step, index) => {
            stepsHtml += `
                <h3>Step ${index + 1}: ${step.name}</h3>
                <p><strong>Type:</strong> ${step.type}</p>
                <p><strong>Description:</strong> ${step.description}</p>
                <p><strong>Parameters:</strong></p>
                <pre>${JSON.stringify(step.parameters, null, 2)}</pre>
                <hr>
            `;
        });
        
        stepsContent.innerHTML = stepsHtml;
        
        // Execution tab
        const executionContent = document.getElementById('execution-content');
        let executionHtml = '<h2>Execution History</h2>';
        
        if (pipeline.execution_history && pipeline.execution_history.length > 0) {
            pipeline.execution_history.forEach((execution, index) => {
                const status = execution.status === 'success' ? '✅' : '❌';
                executionHtml += `
                    <h3>Execution ${index + 1} ${status}</h3>
                    <p><strong>Timestamp:</strong> ${new Date(execution.timestamp).toLocaleString()}</p>
                    <p><strong>Status:</strong> ${execution.status}</p>
                `;
                
                if (execution.status === 'success') {
                    executionHtml += `
                        <p><strong>Input Shape:</strong> ${execution.input_shape[0]} × ${execution.input_shape[1]}</p>
                        <p><strong>Output Shape:</strong> ${execution.output_shape[0]} × ${execution.output_shape[1]}</p>
                    `;
                } else {
                    executionHtml += `<p><strong>Error:</strong> ${execution.error}</p>`;
                }
                
                executionHtml += '<hr>';
            });
        } else {
            executionHtml += '<p>No executions recorded yet.</p>';
        }
        
        executionContent.innerHTML = executionHtml;
    }
    
    showDocTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.doc-tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-doc-tab="${tabName}"]`).classList.add('active');
        
        // Show/hide tab content
        document.querySelectorAll('.doc-tab-content').forEach(content => {
            content.style.display = 'none';
        });
        document.getElementById(`doc-${tabName}`).style.display = 'block';
    }
    
    closeModal(modal) {
        modal.style.display = 'none';
    }
    
    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = show ? 'flex' : 'none';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WorkflowManager();
});