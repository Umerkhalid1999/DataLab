// GLOBAL VARIABLES
// These variables will be accessible throughout the file
let transformationSteps = []; // Array to store all transformation steps
let datasetColumns = [];      // Array to store dataset column names
let currentDatasetId = null;  // The ID of the current dataset being processed

// INITIALIZATION
// This runs when the page is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Preprocessing JS loaded');

    // STEP 1: GET DATASET INFORMATION
    // First check if variables were set in the HTML (preferred method)
    if (window.currentDatasetId) {
        currentDatasetId = window.currentDatasetId;
        console.log('Dataset ID from HTML:', currentDatasetId);
    }

    if (window.datasetColumns && window.datasetColumns.length > 0) {
        datasetColumns = window.datasetColumns;
        console.log('Dataset columns from HTML:', datasetColumns);
    }

    // Fallback: Try to get dataset ID from the container's data attribute
    if (!currentDatasetId) {
        const container = document.querySelector('.preprocessing-container');
        if (container && container.dataset && container.dataset.id) {
            currentDatasetId = container.dataset.id;
            console.log('Dataset ID from container attribute:', currentDatasetId);
        }
    }

    // STEP 2: SETUP EVENT LISTENERS FOR STATIC ELEMENTS
    // These are for buttons that exist when the page loads

    // "Add Custom Step" button
    const addCustomBtn = document.getElementById('addCustomTransformBtn');
    if (addCustomBtn) {
        addCustomBtn.addEventListener('click', function() {
            console.log('Add Custom Step clicked');
            showCustomTransformModal();
        });
    }

    // "Preview Transformations" button
    const previewBtn = document.getElementById('previewTransformBtn');
    if (previewBtn) {
        previewBtn.addEventListener('click', function() {
            console.log('Preview Transformations clicked');
            previewTransformations();
        });
    }

    // "Apply Transformations" button
    const applyBtn = document.getElementById('applyTransformBtn');
    if (applyBtn) {
        applyBtn.addEventListener('click', function() {
            console.log('Apply Transformations clicked');
            applyTransformations();
        });
    }

    // Transform type dropdown in the modal
    const transformType = document.getElementById('transformType');
    if (transformType) {
        transformType.addEventListener('change', function() {
            updateTransformParams(this.value);
        });
    }

    // "Save Transformation" button in the modal
    const saveBtn = document.getElementById('saveTransformBtn');
    if (saveBtn) {
        saveBtn.addEventListener('click', function() {
            saveCustomTransform();
        });
    }

    // STEP 3: SETUP GLOBAL EVENT LISTENER FOR DYNAMIC ELEMENTS
    // This handles buttons that are created after the page loads
    document.addEventListener('click', function(e) {
        // Check if the clicked element or its parent has the 'add-suggestion-btn' class
        const btn = e.target.closest('.add-suggestion-btn');
        if (btn) {
            e.preventDefault();
            console.log('Add to Pipeline clicked (global handler)');

            // Get the transformation type and columns from data attributes
            const type = btn.getAttribute('data-type');
            const columns = JSON.parse(btn.getAttribute('data-columns'));

            console.log('Adding transformation:', type, columns);
            addTransformationStep(type, columns);
        }
    });
});

// FUNCTION: Show the Custom Transform Modal
function showCustomTransformModal() {
    console.log('Opening custom transform modal');

    // Reset the form
    const form = document.getElementById('customTransformForm');
    if (form) form.reset();

    // Clear the parameters container
    const paramsContainer = document.getElementById('transformParamsContainer');
    if (paramsContainer) paramsContainer.innerHTML = '';

    // Populate the columns dropdown
    const columnsSelect = document.getElementById('transformColumns');
    if (columnsSelect) {
        columnsSelect.innerHTML = '';
        datasetColumns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            columnsSelect.appendChild(option);
        });
    }

    // Show the modal using Bootstrap
    const modalElement = document.getElementById('transformStepModal');
    if (modalElement) {
        try {
            // Try to get existing instance or create new one
            let modal = bootstrap.Modal.getInstance(modalElement);
            if (!modal) {
                modal = new bootstrap.Modal(modalElement);
            }
            modal.show();
        } catch (e) {
            console.error('Error showing modal:', e);
            alert('Could not open the modal. Make sure Bootstrap JS is loaded.');
        }
    } else {
        console.error('Modal element not found');
    }
}

// FUNCTION: Update Transform Parameters Based on Selected Type
function updateTransformParams(transformType) {
    console.log('Updating parameters for transform type:', transformType);

    const paramsContainer = document.getElementById('transformParamsContainer');
    if (!paramsContainer) return;

    paramsContainer.innerHTML = '';

    switch(transformType) {
        case 'imputation':
            paramsContainer.innerHTML = `
                <div class="mb-3">
                    <label for="imputationMethod" class="form-label">Imputation Method</label>
                    <select class="form-select" id="imputationMethod" name="method" required>
                        <option value="mean">Mean (for numerical)</option>
                        <option value="median">Median (for numerical)</option>
                        <option value="mode">Mode (for categorical)</option>
                        <option value="constant">Constant Value</option>
                    </select>
                </div>
                <div class="mb-3 d-none" id="constantValueGroup">
                    <label for="constantValue" class="form-label">Constant Value</label>
                    <input type="text" class="form-control" id="constantValue" name="constant_value">
                </div>
            `;

            // Show/hide constant value based on method
            document.getElementById('imputationMethod').addEventListener('change', function() {
                const constantGroup = document.getElementById('constantValueGroup');
                if (this.value === 'constant') {
                    constantGroup.classList.remove('d-none');
                } else {
                    constantGroup.classList.add('d-none');
                }
            });
            break;

        case 'scaling':
            paramsContainer.innerHTML = `
                <div class="mb-3">
                    <label for="scalingMethod" class="form-label">Scaling Method</label>
                    <select class="form-select" id="scalingMethod" name="method" required>
                        <option value="standard">Standard (Z-score)</option>
                        <option value="minmax">Min-Max (0-1)</option>
                        <option value="robust">Robust (median/IQR)</option>
                    </select>
                </div>
            `;
            break;

        case 'encoding':
            paramsContainer.innerHTML = `
                <div class="mb-3">
                    <label for="encodingMethod" class="form-label">Encoding Method</label>
                    <select class="form-select" id="encodingMethod" name="method" required>
                        <option value="onehot">One-Hot Encoding</option>
                        <option value="ordinal">Ordinal Encoding</option>
                        <option value="target">Target Encoding</option>
                    </select>
                </div>
            `;
            break;

        case 'outlier':
            paramsContainer.innerHTML = `
                <div class="mb-3">
                    <label for="outlierMethod" class="form-label">Outlier Treatment</label>
                    <select class="form-select" id="outlierMethod" name="method" required>
                        <option value="winsorize">Winsorization (clip values)</option>
                        <option value="remove">Remove Outliers</option>
                        <option value="transform">Log Transformation</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="outlierThreshold" class="form-label">Threshold (IQR multiplier)</label>
                    <input type="number" class="form-control" id="outlierThreshold" name="threshold" value="1.5" step="0.1" min="0.1" max="5">
                </div>
            `;
            break;

        case 'transformation':
            paramsContainer.innerHTML = `
                <div class="mb-3">
                    <label for="transformationMethod" class="form-label">Transformation Method</label>
                    <select class="form-select" id="transformationMethod" name="method" required>
                        <option value="log">Logarithmic</option>
                        <option value="sqrt">Square Root</option>
                        <option value="boxcox">Box-Cox</option>
                        <option value="yeojohnson">Yeo-Johnson</option>
                    </select>
                </div>
            `;
            break;
    }
}

// FUNCTION: Save Custom Transform from Modal
function saveCustomTransform() {
    console.log('Saving custom transformation');

    const form = document.getElementById('customTransformForm');
    if (!form) return;

    const formData = new FormData(form);
    const transformData = Object.fromEntries(formData.entries());

    // Convert columns to array (handle multiple selected options)
    transformData.columns = Array.from(formData.getAll('columns'));

    // Validate
    if (transformData.columns.length === 0) {
        alert('Please select at least one column');
        return;
    }

    // Add to transformation steps
    addTransformationStep(transformData.type, transformData.columns, transformData);

    // Close modal
    try {
        const modalElement = document.getElementById('transformStepModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        if (modal) modal.hide();
    } catch (e) {
        console.error('Error closing modal:', e);
    }
}

// FUNCTION: Add Transformation Step
function addTransformationStep(type, columns, params = {}) {
    console.log('Adding transformation step:', type, columns, params);

    const stepId = Date.now(); // Generate unique ID using timestamp
    const step = {
        id: stepId,
        type: type,
        columns: columns,
        params: params,
        description: getStepDescription(type, columns, params)
    };

    transformationSteps.push(step);
    renderTransformationSteps();
}

// FUNCTION: Generate Description for a Step
function getStepDescription(type, columns, params) {
    const colText = columns.length > 3 ?
        `${columns.length} columns` :
        columns.join(', ');

    switch(type) {
        case 'imputation':
            return `Impute missing values in ${colText} using ${params.method || 'mean'}`;
        case 'scaling':
            return `Scale ${colText} using ${params.method || 'standard'} method`;
        case 'encoding':
            return `Encode ${colText} using ${params.method || 'onehot'} encoding`;
        case 'outlier':
            return `Handle outliers in ${colText} using ${params.method || 'winsorize'}`;
        case 'transformation':
            return `Transform ${colText} using ${params.method || 'log'} method`;
        default:
            return `Apply ${type} to ${colText}`;
    }
}

// FUNCTION: Render Transformation Steps in the UI
function renderTransformationSteps() {
    console.log('Rendering transformation steps:', transformationSteps.length);

    const pipelineContainer = document.getElementById('transformationPipeline');
    if (!pipelineContainer) return;

    pipelineContainer.innerHTML = '';

    if (transformationSteps.length === 0) {
        pipelineContainer.innerHTML = `
            <div class="alert alert-info">
                No transformations added yet. Click "Add Custom Step" or select from suggested transformations.
            </div>
        `;
        return;
    }

    transformationSteps.forEach((step, index) => {
        const stepElement = document.createElement('div');
        stepElement.className = 'pipeline-step card mb-2';
        stepElement.dataset.stepId = step.id;

        stepElement.innerHTML = `
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h6 class="mb-0">Step ${index + 1}: ${step.type}</h6>
                        <small class="text-muted">${step.description}</small>
                    </div>
                    <div>
                        <button class="btn btn-sm btn-outline-primary view-step-btn me-1" title="View Details">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger remove-step-btn" title="Remove Step">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;

        pipelineContainer.appendChild(stepElement);

        // Add event listeners
        stepElement.querySelector('.view-step-btn').addEventListener('click', () => {
            viewStepDetails(step);
        });

        stepElement.querySelector('.remove-step-btn').addEventListener('click', () => {
            removeStep(step.id);
        });
    });
}

// FUNCTION: View Step Details
function viewStepDetails(step) {
    console.log('Viewing step details:', step);

    // Simple version - can be enhanced to show in a modal
    alert(`Step Details:
Type: ${step.type}
Columns: ${step.columns.join(', ')}
Params: ${JSON.stringify(step.params, null, 2)}`);
}

// FUNCTION: Remove a Step
function removeStep(stepId) {
    console.log('Removing step:', stepId);

    transformationSteps = transformationSteps.filter(step => step.id !== stepId);
    renderTransformationSteps();
}

// FUNCTION: Preview Transformations
function previewTransformations() {
    console.log('Previewing transformations');

    if (transformationSteps.length === 0) {
        alert('No transformations to preview. Please add some transformations first.');
        return;
    }

    const previewContainer = document.getElementById('transformationStepsPreview');
    if (!previewContainer) return;

    previewContainer.innerHTML = '<div class="text-center my-4"><i class="fas fa-spinner fa-spin fa-2x"></i><p>Generating preview...</p></div>';

    // Show preview section
    const previewSection = document.getElementById('transformationPreview');
    if (previewSection) previewSection.classList.remove('d-none');

    // Get task type and target from form
    const taskType = document.getElementById('taskType')?.value || 'classification';
    const target = document.getElementById('targetColumn')?.value || '';

    // Prepare data for API
    const transformData = {
        task_type: taskType,
        target: target,
        transformations: transformationSteps,
        pycaret_params: getPyCaretParams()
    };

    // Call API to preview transformations
    fetch(`/api/preprocessing/transform/${currentDatasetId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(transformData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayTransformationPreview(data.preprocessing_steps);
        } else {
            previewContainer.innerHTML = `
                <div class="alert alert-danger">
                    Error generating preview: ${data.message || 'Unknown error'}
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        previewContainer.innerHTML = `
            <div class="alert alert-danger">
                Error generating preview: ${error.message}
            </div>
        `;
    });
}

// FUNCTION: Get PyCaret Parameters
function getPyCaretParams() {
    // Get basic PyCaret setup parameters from UI
    return {
        normalize: transformationSteps.some(step => step.type === 'scaling'),
        transformation: transformationSteps.some(step => step.type === 'transformation'),
//        ignore_low_variance: true,
//        remove_multicollinearity: true,
//        multicollinearity_threshold: 0.9,
//        fix_imbalance: true
    };
}

// FUNCTION: Display Transformation Preview
function displayTransformationPreview(steps) {
    console.log('Displaying transformation preview', steps?.length);

    const previewContainer = document.getElementById('transformationStepsPreview');
    if (!previewContainer) return;

    previewContainer.innerHTML = '';

    if (!steps || steps.length === 0) {
        previewContainer.innerHTML = `
            <div class="alert alert-info">
                No transformation steps to display
            </div>
        `;
        return;
    }

    steps.forEach((step, index) => {
        if (step.step === 'initial') return;

        const stepElement = document.createElement('div');
        stepElement.className = 'preview-step card mb-3';

        stepElement.innerHTML = `
            <div class="card-header">
                <h6>Step ${index}: ${step.description}</h6>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Before Transformation</h6>
                        <pre class="bg-light p-2">${JSON.stringify(step.changes.before, null, 2)}</pre>
                    </div>
                    <div class="col-md-6">
                        <h6>After Transformation</h6>
                        <pre class="bg-light p-2">${JSON.stringify(step.changes.after, null, 2)}</pre>
                    </div>
                </div>
                ${step.visualization ? `
                <div class="row mt-3">
                    <div class="col-md-12">
                        <h6>Visual Comparison</h6>
                        <img src="data:image/png;base64,${step.visualization}"
                             class="img-fluid" alt="Transformation visualization">
                    </div>
                </div>
                ` : ''}
            </div>
        `;

        previewContainer.appendChild(stepElement);
    });
}

// FUNCTION: Apply Transformations
function applyTransformations() {
    console.log('Applying transformations');

    if (transformationSteps.length === 0) {
        alert('No transformations to apply. Please add some transformations first.');
        return;
    }

    const applyBtn = document.getElementById('applyTransformBtn');
    if (applyBtn) {
        applyBtn.disabled = true;
        applyBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Applying...';
    }

    // Get task type and target from form
    const taskType = document.getElementById('taskType')?.value || 'classification';
    const target = document.getElementById('targetColumn')?.value || '';

    // Prepare data for API - simplify to reduce potential issues
    const transformData = {
        task_type: taskType,
        target: target,
        transformations: transformationSteps,
        // Simplified parameters to avoid compatibility issues
        pycaret_params: {
            normalize: transformationSteps.some(step => step.type === 'scaling')
        }
    };

    console.log('Sending transform request with data:', JSON.stringify(transformData));

    // Call API to apply transformations with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

    fetch(`/api/preprocessing/transform/${currentDatasetId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(transformData),
        signal: controller.signal
    })
    .then(response => {
        clearTimeout(timeoutId);
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`Server returned ${response.status}: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Show success message
            alert('Transformations applied successfully! New dataset created.');

            // Redirect to new dataset or refresh page
            window.location.href = `/data_preview/${data.new_dataset_id}`;
        } else {
            alert(`Error applying transformations: ${data.message || 'Unknown error'}`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        if (error.name === 'AbortError') {
            alert('Request timed out. The transformation may be too complex or the server is busy.');
        } else {
            alert(`Error applying transformations: ${error.message}`);
        }
    })
    .finally(() => {
        if (applyBtn) {
            applyBtn.disabled = false;
            applyBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Apply Transformations';
        }
    });
}