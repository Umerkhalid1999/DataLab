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
    initializeVisualizationNav();

    // Initialize theme selector
    initializeTheme();
});

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
            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);

                    if (response.success) {
                        handleUploadSuccess(response);
                    } else {
                        // Show error message
                        uploadMessage.textContent = response.message;
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
    const deleteButtons = document.querySelectorAll('.delete-dataset');
    const confirmDeleteBtn = document.getElementById('confirmDelete');
    const deleteModal = document.getElementById('deleteDatasetModal');

    if (!deleteButtons.length || !confirmDeleteBtn || !deleteModal) return;

    let currentDatasetId = null;

    // Set up event listeners for delete buttons
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            currentDatasetId = this.getAttribute('data-id');
            const modal = new bootstrap.Modal(deleteModal);
            modal.show();
        });
    });

    // Confirm delete action
    confirmDeleteBtn.addEventListener('click', function() {
        if (!currentDatasetId) return;

        // Create and append CSRF token
        const formData = new FormData();

        // Send delete request
        fetch(`/delete_dataset/${currentDatasetId}`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Hide modal and refresh page
                const modal = bootstrap.Modal.getInstance(deleteModal);
                if (modal) modal.hide();
                window.location.reload();
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while deleting the dataset.');
        });
    });
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

// Enhanced upload success with better integration
function handleUploadSuccess(response) {
    // Show success message with visualization option
    uploadMessage.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <i class="fas fa-check-circle me-2"></i>
                ${response.message}
            </div>
            <div>
                <a href="/visualization/${response.dataset_id}" class="btn btn-sm btn-primary me-2">
                    <i class="fas fa-chart-bar me-1"></i>Visualize Now
                </a>
                <button class="btn btn-sm btn-outline-secondary" onclick="window.location.reload()">
                    <i class="fas fa-redo me-1"></i>Continue
                </button>
            </div>
        </div>
    `;
    uploadMessage.classList.remove('d-none', 'alert-danger');
    uploadMessage.classList.add('alert-success');

    // Show toast notification
    showToast(
        `📊 Dataset "${response.dataset.name}" uploaded successfully! <a href="/visualization/${response.dataset_id}" class="text-white text-decoration-underline">Click here to visualize</a>`,
        'success',
        7000
    );

    // Auto-refresh after 8 seconds if user doesn't click
    setTimeout(function() {
        if (document.contains(uploadMessage)) {
            window.location.reload();
        }
    }, 8000);
}