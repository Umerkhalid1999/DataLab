// dashboard.js - Client-side JavaScript for DataLab dashboard functionality
// Place this file in your static/js directory

document.addEventListener('DOMContentLoaded', function() {
    // Fix sidebar styling issues immediately
    fixSidebarStyling();

    // Check if user is authenticated via Firebase
    checkAuthState();

    // Initialize dashboard components
    initializeUpload();
    initializeSearch();
    initializeDeleteActions();
    initializeCleanDataset();
    initializeVisualizationNav();

    // Initialize theme selector
    initializeTheme();

    // Initialize Bootstrap tooltips for quality score info
    initializeTooltips();
});

// Initialize Bootstrap tooltips
function initializeTooltips() {
    // Check if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl, {
                html: false,
                trigger: 'hover focus'
            });
        });
    }
}

// Fix sidebar styling issues
function fixSidebarStyling() {
    // Target all sidebar elements
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.style.backgroundColor = '#1F2937';
        sidebar.style.color = 'white';

        // Fix sidebar elements
        const allElements = sidebar.querySelectorAll('*');
        allElements.forEach(el => {
            // For direct icons/links in sidebar
            if (el.tagName === 'I' || el.tagName === 'A' || el.tagName === 'SVG') {
                if (el.parentElement === sidebar) {
                    el.style.color = 'white';
                }
            }

            // Fix sidebar header
            if (el.classList.contains('sidebar-header')) {
                const headerElements = el.querySelectorAll('*');
                headerElements.forEach(hEl => {
                    if (hEl.tagName === 'H2') {
                        hEl.style.color = 'white';
                    }
                    if (hEl.tagName === 'I') {
                        hEl.style.color = 'white';
                    }
                });
            }

            // Fix nav items
            if (el.classList.contains('nav-item')) {
                const links = el.querySelectorAll('a');
                links.forEach(link => {
                    link.style.color = '#D1D5DB';

                    // Fix icons inside links
                    const icons = link.querySelectorAll('i');
                    icons.forEach(icon => {
                        icon.style.color = '#D1D5DB';
                    });

                    // Hover and active states handled by CSS
                });
            }
        });
    }

    // Also fix the vertical sidebar in Image 1
    const verticalSidebarIcons = document.querySelectorAll('body > div > a, body > div > i, body > nav > a, body > nav > i');
    verticalSidebarIcons.forEach(icon => {
        icon.style.color = 'white';
    });
}

// Initialize theme switcher
function initializeTheme() {
    const themeSelector = document.getElementById('theme');
    if (themeSelector) {
        // Set initial theme
        if (localStorage.getItem('theme') === 'dark') {
            document.body.classList.add('dark');
            themeSelector.value = 'dark';
        } else {
            themeSelector.value = 'light';
        }

        // Handle theme switching
        themeSelector.addEventListener('change', function() {
            if (this.value === 'dark') {
                document.body.classList.add('dark');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.classList.remove('dark');
                localStorage.setItem('theme', 'light');
            }

            // Always ensure sidebar stays dark
            fixSidebarStyling();
        });
    }
}

// Check authentication state (only for dashboard-specific logic)
function checkAuthState() {
    if (window.firebaseAuth && window.firebaseReady) {
        window.firebaseAuth.onAuthStateChanged(window.firebaseAuth.auth, (user) => {
            if (!user) {
                // Don't auto-redirect - let server-side handle authentication
                console.log("Dashboard: User not authenticated on client-side");
            } else {
                console.log("Dashboard: User authenticated on client-side");
            }
        });
    } else if (!window.firebaseAuth) {
        console.error("Firebase Auth not initialized");
    } else {
        // Wait for Firebase to be ready
        console.log("Dashboard: Waiting for Firebase to be ready...");
        setTimeout(checkAuthState, 100);
    }
}

// Initialize file upload functionality
function initializeUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadProgressBar = document.getElementById('uploadProgressBar');
    const uploadMessage = document.getElementById('uploadMessage');

    if (!uploadArea || !fileInput) return;

    // Setup drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        }, false);
    });

    // Handle file drop
    uploadArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            uploadFile(files[0]);
        }
    }, false);

    // Connect button to file input
    const uploadButton = document.querySelector('.file-upload-btn');
    if (uploadButton) {
        uploadButton.addEventListener('click', function(e) {
            e.preventDefault();
            fileInput.click();
        });
    }

    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });

    // Upload file function
    function uploadFile(file) {
        // Check if file type is allowed
        const allowedTypes = ['csv', 'json', 'txt', 'xlsx', 'xls', 'jpg', 'png'];
        const fileExtension = file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(fileExtension)) {
            uploadMessage.textContent = 'File type not allowed. Please upload CSV, JSON, XLSX, TXT, or image files.';
            uploadMessage.classList.remove('d-none', 'alert-success');
            uploadMessage.classList.add('alert-danger');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Show progress
        uploadProgress.classList.remove('d-none');
        uploadProgressBar.style.width = '0%';
        uploadProgressBar.textContent = '0%';
        uploadMessage.classList.add('d-none');

        // Send AJAX request
        const xhr = new XMLHttpRequest();

        xhr.open('POST', '/upload_dataset', true);

        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                uploadProgressBar.style.width = percentComplete + '%';
                uploadProgressBar.textContent = percentComplete + '%';

                // Update color based on progress
                if (percentComplete < 50) {
                    uploadProgressBar.classList.remove('bg-success', 'bg-warning');
                    uploadProgressBar.classList.add('bg-info');
                } else if (percentComplete < 85) {
                    uploadProgressBar.classList.remove('bg-info', 'bg-success');
                    uploadProgressBar.classList.add('bg-warning');
                } else {
                    uploadProgressBar.classList.remove('bg-info', 'bg-warning');
                    uploadProgressBar.classList.add('bg-success');
                }
            }
        };

        xhr.onload = function() {
            uploadProgress.classList.add('d-none');

            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);

                    if (response.success) {
                        // Show success message
                        uploadMessage.innerHTML = `
                            <div>
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <i class="fas fa-check-circle me-2"></i>
                                        ${response.message || 'Dataset uploaded successfully!'}
                                    </div>
                                    <div>
                                        <a href="/visualization/${response.dataset_id}" class="btn btn-sm btn-primary me-2">
                                            <i class="fas fa-chart-bar me-1"></i>Visualize
                                        </a>
                                        <button class="btn btn-sm btn-outline-secondary" onclick="window.location.reload()">
                                            <i class="fas fa-redo me-1"></i>Refresh
                                        </button>
                                    </div>
                                </div>
                            </div>
                        `;
                        uploadMessage.classList.remove('d-none', 'alert-danger');
                        uploadMessage.classList.add('alert-success');

                        // Auto-refresh after 3 seconds
                        setTimeout(function() {
                            window.location.reload();
                        }, 3000);
                    } else {
                        // Show error message
                        uploadMessage.textContent = response.message || 'Upload failed';
                        uploadMessage.classList.remove('d-none', 'alert-success');
                        uploadMessage.classList.add('alert-danger');
                    }
                } catch (e) {
                    console.error('Error parsing response:', e);
                    uploadMessage.textContent = 'An error occurred during upload.';
                    uploadMessage.classList.remove('d-none', 'alert-success');
                    uploadMessage.classList.add('alert-danger');
                }
            } else {
                // Show error message
                uploadMessage.textContent = 'Server error: ' + xhr.status;
                uploadMessage.classList.remove('d-none', 'alert-success');
                uploadMessage.classList.add('alert-danger');
            }

            // Reset file input
            fileInput.value = '';
        };

        xhr.onerror = function() {
            uploadMessage.textContent = 'Network error during upload.';
            uploadMessage.classList.remove('d-none', 'alert-success');
            uploadMessage.classList.add('alert-danger');
        };

        xhr.send(formData);
    }
}

// Initialize dataset search functionality
function initializeSearch() {
    const datasetSearch = document.getElementById('datasetSearch');

    if (!datasetSearch) return;

    datasetSearch.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const datasetItems = document.querySelectorAll('.dataset-item');

        datasetItems.forEach(item => {
            const datasetName = item.querySelector('.card-header h6').textContent.toLowerCase();

            if (datasetName.includes(searchTerm)) {
                item.style.display = 'block';
            } else {
                item.style.display = 'none';
            }
        });
    });
}

// Initialize delete dataset functionality
function initializeDeleteActions() {
    const confirmDeleteBtn = document.getElementById('confirmDelete');
    const deleteModal = document.getElementById('deleteDatasetModal');
    const deleteDatasetName = document.getElementById('deleteDatasetName');

    if (!confirmDeleteBtn || !deleteModal) return;

    let currentDatasetId = null;
    let currentDatasetCard = null;
    let currentDatasetName = 'this dataset';

    // Ensure every dropdown has a delete option (covers cached/older HTML)
    document.querySelectorAll('.dataset-card').forEach(card => {
        const menu = card.querySelector('.dropdown-menu');
        if (!menu) return;
        if (menu.querySelector('.delete-dataset')) return;

        const datasetId = card.getAttribute('data-id');
        const datasetName = card.getAttribute('data-name') || (card.querySelector('h6')?.textContent?.trim()) || 'this dataset';

        const divider = document.createElement('li');
        divider.innerHTML = '<hr class="dropdown-divider">';
        const li = document.createElement('li');
        li.innerHTML = `<a class="dropdown-item text-danger delete-dataset" href="#" data-id="${datasetId}" data-name="${datasetName}"><i class="fas fa-trash-alt me-2"></i>Remove Dataset</a>`;

        menu.appendChild(divider);
        menu.appendChild(li);
    });

    // Event delegation for any delete-dataset trigger (works on dynamically added cards)
    document.addEventListener('click', function(e) {
        const trigger = e.target.closest('.delete-dataset');
        if (!trigger) return;

        e.preventDefault();
        currentDatasetId = trigger.getAttribute('data-id');
        currentDatasetName = trigger.getAttribute('data-name') || 'this dataset';
        currentDatasetCard = trigger.closest('.dataset-card');

        if (deleteDatasetName) {
            deleteDatasetName.textContent = currentDatasetName;
        }

        const modal = new bootstrap.Modal(deleteModal);
        modal.show();
    });

    // Confirm delete action
    confirmDeleteBtn.addEventListener('click', function() {
        if (!currentDatasetId) return;

        confirmDeleteBtn.disabled = true;
        confirmDeleteBtn.textContent = 'Deleting...';

        // Send delete request
        fetch(`/delete_dataset/${currentDatasetId}`, {
            method: 'DELETE',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Hide modal and update UI optimistically
                const modal = bootstrap.Modal.getInstance(deleteModal);
                if (modal) modal.hide();
                if (currentDatasetCard) {
                    currentDatasetCard.parentElement.remove();
                }

                // Show toast message
                showToast(data.message || `Deleted ${currentDatasetName}`, 'success', 3000);

                // If no datasets remain, reload to refresh empty state
                const remaining = document.querySelectorAll('.dataset-card');
                if (!remaining.length) {
                    window.location.reload();
                }
            } else {
                showToast('Error: ' + (data.message || 'Could not delete dataset'), 'danger');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('An error occurred while deleting the dataset.', 'danger');
        })
        .finally(() => {
            confirmDeleteBtn.disabled = false;
            confirmDeleteBtn.textContent = 'Delete';
        });
    });
}

// Initialize clean dataset functionality with modal UI
function initializeCleanDataset() {
    const cleanButtons = document.querySelectorAll('.clean-dataset');
    if (!cleanButtons.length) return;

    cleanButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const datasetId = this.getAttribute('data-id');
            const datasetName = this.getAttribute('data-name');
            showCleaningModal(datasetId, datasetName);
        });
    });
}

// Show cleaning options modal
function showCleaningModal(datasetId, datasetName) {
    const modalHTML = `
        <div class="modal fade" id="cleaningModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title"><i class="fas fa-robot me-2"></i>AI-Powered Preprocessing: ${datasetName}</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-info">
                            <i class="fas fa-brain me-2"></i>
                            <strong>Intelligent Analysis:</strong> AI will analyze your dataset, detect issues, and apply only necessary transformations with complete transparency.
                        </div>
                        <div id="cleaningProgress" class="d-none mt-3">
                            <div class="progress" style="height: 30px;">
                                <div class="progress-bar progress-bar-striped progress-bar-animated bg-info" style="width: 100%">
                                    <i class="fas fa-cog fa-spin me-2"></i>AI Analyzing Dataset...
                                </div>
                            </div>
                        </div>
                        <div id="cleaningResults" class="d-none mt-3"></div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-success" id="applyCleaningBtn">
                            <i class="fas fa-magic me-2"></i>Start AI Preprocessing
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    const existingModal = document.getElementById('cleaningModal');
    if (existingModal) existingModal.remove();
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    const modal = new bootstrap.Modal(document.getElementById('cleaningModal'));
    modal.show();
    
    document.getElementById('applyCleaningBtn').addEventListener('click', function() {
        applyCleaningOperations(datasetId, modal);
    });
}

// Apply cleaning operations
function applyCleaningOperations(datasetId, modal) {
    document.getElementById('cleaningProgress').classList.remove('d-none');
    document.getElementById('applyCleaningBtn').disabled = true;
    
    fetch(`/api/clean_dataset/${datasetId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('cleaningProgress').classList.add('d-none');
        
        if (data.success) {
            displayIntelligentResults(data);
        } else {
            alert('Error: ' + data.message);
            document.getElementById('applyCleaningBtn').disabled = false;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred: ' + error);
        document.getElementById('cleaningProgress').classList.add('d-none');
        document.getElementById('applyCleaningBtn').disabled = false;
    });
}

// Display cleaning results with AI explanation
function displayCleaningResults(data) {
    const resultsDiv = document.getElementById('cleaningResults');
    resultsDiv.classList.remove('d-none');
    
    // Check if dataset was already clean
    if (data.transformations.length === 0) {
        resultsDiv.innerHTML = `
            <div class="alert alert-success">
                <h5><i class="fas fa-check-circle me-2"></i>Dataset Already Clean!</h5>
                <p class="mb-0">No data quality issues detected. Your dataset is in excellent condition.</p>
            </div>
            <div class="card">
                <div class="card-header bg-success text-white">
                    <i class="fas fa-thumbs-up me-2"></i>AI Analysis
                </div>
                <div class="card-body">
                    <p class="mb-0">${data.ai_explanation}</p>
                </div>
            </div>
            <div class="mt-3 text-center">
                <button class="btn btn-secondary" data-bs-dismiss="modal">
                    <i class="fas fa-times me-2"></i>Close
                </button>
            </div>
        `;
        return;
    }
    
    // Show issues found
    let issuesHTML = '<h6>Issues Detected:</h6><ul class="list-group mb-3">';
    for (const [issue, details] of Object.entries(data.issues_found)) {
        if (issue === 'missing_values') {
            issuesHTML += `<li class="list-group-item list-group-item-warning">
                <strong>Missing Values:</strong> ${details.total} (${details.percentage}%)</li>`;
        } else if (issue === 'duplicates') {
            issuesHTML += `<li class="list-group-item list-group-item-warning">
                <strong>Duplicates:</strong> ${details.count} rows (${details.percentage}%)</li>`;
        } else if (issue === 'outliers') {
            issuesHTML += `<li class="list-group-item list-group-item-warning">
                <strong>Outliers:</strong> ${details.total} values</li>`;
        } else if (issue === 'high_variance') {
            issuesHTML += `<li class="list-group-item list-group-item-info">
                <strong>High Variance:</strong> ${details.columns.length} columns need scaling</li>`;
        }
    }
    issuesHTML += '</ul>';
    
    // Show transformations applied
    let transformationsHTML = '<h6>Transformations Applied:</h6><ul class="list-group mb-3">';
    data.transformations.forEach(t => {
        transformationsHTML += `<li class="list-group-item list-group-item-success">
            <strong>${t.type}:</strong> ${t.reason}</li>`;
    });
    transformationsHTML += '</ul>';
    
    resultsDiv.innerHTML = `
        <div class="alert alert-success">
            <h5><i class="fas fa-check-circle me-2"></i>Smart Cleaning Complete!</h5>
            <p><strong>Quality Score:</strong> ${data.quality_improvement.original}% → ${data.quality_improvement.new}% 
            <span class="badge bg-success">+${data.quality_improvement.improvement}%</span></p>
        </div>
        ${issuesHTML}
        ${transformationsHTML}
        <div class="card">
            <div class="card-header bg-info text-white">
                <i class="fas fa-robot me-2"></i>AI Analysis
            </div>
            <div class="card-body">
                <p class="mb-0" style="white-space: pre-wrap;">${data.ai_explanation}</p>
            </div>
        </div>
        <div class="mt-3 text-center">
            <button class="btn btn-primary" onclick="window.location.reload()">
                <i class="fas fa-redo me-2"></i>Refresh Dashboard
            </button>
        </div>
    `;
    
    document.getElementById('applyCleaningBtn').classList.add('d-none');
}

// Initialize visualization navigation
function initializeVisualizationNav() {
    const visualizationsNavLink = document.getElementById('visualizationsNavLink');
    
    if (!visualizationsNavLink) return;
    
    visualizationsNavLink.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Get the first available dataset for visualization
        const firstDatasetCard = document.querySelector('.dataset-card');
        
        if (firstDatasetCard) {
            const datasetId = firstDatasetCard.getAttribute('data-id');
            if (datasetId) {
                window.location.href = `/visualization/${datasetId}`;
            } else {
                alert('Please select a dataset first to visualize.');
            }
        } else {
            alert('No datasets available. Please upload a dataset first.');
        }
    });
}

// Toast notification system
function showToast(message, type = 'info', duration = 5000) {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }

    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;

    toastContainer.appendChild(toast);

    // Show toast
    const bsToast = new bootstrap.Toast(toast, { delay: duration });
    bsToast.show();

    // Remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

// Display intelligent cleaning results
function displayIntelligentResults(data) {
    const resultsDiv = document.getElementById('cleaningResults');
    if (!resultsDiv) return;

    resultsDiv.classList.remove('d-none');

    displayCleaningResults(data);
}


// Update dataset quality score in real-time
function updateDatasetQualityScore(datasetId, updatedDataset) {
    const datasetCard = document.querySelector(`.dataset-card[data-id="${datasetId}"]`);
    if (!datasetCard) return;
    
    // Update quality score display
    const qualityScoreSpan = datasetCard.querySelector('.fw-bold');
    if (qualityScoreSpan && qualityScoreSpan.textContent.includes('%')) {
        qualityScoreSpan.textContent = `${updatedDataset.quality_score}%`;
    }
    
    // Update quality bar
    const qualityBar = datasetCard.querySelector('.quality-bar');
    if (qualityBar) {
        qualityBar.style.width = `${updatedDataset.quality_score}%`;
        
        // Update color based on score
        if (updatedDataset.quality_score >= 80) {
            qualityBar.style.backgroundColor = '#28a745';
        } else if (updatedDataset.quality_score >= 60) {
            qualityBar.style.backgroundColor = '#ffc107';
        } else {
            qualityBar.style.backgroundColor = '#dc3545';
        }
    }
    
    // Update quality components data attribute
    const qualityInfoIcon = datasetCard.querySelector('.quality-score-info');
    if (qualityInfoIcon && updatedDataset.quality_components) {
        qualityInfoIcon.setAttribute('data-quality-score', updatedDataset.quality_score);
        qualityInfoIcon.setAttribute('data-quality-components', JSON.stringify(updatedDataset.quality_components));
    }
    
    // Update rows and columns
    const rowsDiv = datasetCard.querySelector('.col-6:nth-child(1) .fw-bold');
    if (rowsDiv) rowsDiv.textContent = updatedDataset.rows;
    
    const columnsDiv = datasetCard.querySelector('.col-6:nth-child(2) .fw-bold');
    if (columnsDiv) columnsDiv.textContent = updatedDataset.columns;
}
