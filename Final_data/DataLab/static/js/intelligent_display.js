// Display intelligent preprocessing results with complete breakdown
function displayIntelligentResults(data) {
    const resultsDiv = document.getElementById('cleaningResults');
    resultsDiv.classList.remove('d-none');
    
    // Check if dataset was already clean
    if (data.transformations.length === 0) {
        resultsDiv.innerHTML = `
            <div class="alert alert-success">
                <h4><i class="fas fa-check-circle me-2"></i>Dataset Already Clean!</h4>
                <p class="mb-0">AI Analysis: No data quality issues detected. Your dataset is in excellent condition.</p>
            </div>
            <div class="text-center mt-3">
                <button class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
        `;
        return;
    }
    
    // Build comprehensive report
    let html = `
        <div class="alert alert-success">
            <h4><i class="fas fa-check-circle me-2"></i>Intelligent Preprocessing Complete!</h4>
            <p><strong>Quality Score:</strong> ${data.quality_improvement.original}% → ${data.quality_improvement.new}% 
            <span class="badge bg-success">+${data.quality_improvement.improvement}%</span></p>
        </div>
    `;
    
    // Issues Detected Section
    if (data.analysis && data.analysis.issues_detected) {
        html += '<div class="card mb-3"><div class="card-header bg-warning text-dark"><h5 class="mb-0"><i class="fas fa-exclamation-triangle me-2"></i>Issues Detected</h5></div><div class="card-body">';
        
        for (const [issueType, details] of Object.entries(data.analysis.issues_detected)) {
            html += `<div class="mb-3"><h6 class="text-capitalize">${issueType.replace('_', ' ')}</h6>`;
            
            if (issueType === 'missing_values') {
                html += `<p class="mb-1"><strong>Total Missing:</strong> ${details.total_missing} | <strong>Severity:</strong> <span class="badge bg-${details.severity === 'HIGH' ? 'danger' : 'warning'}">${details.severity}</span></p>`;
                html += '<ul class="small">';
                for (const [col, info] of Object.entries(details.affected_columns)) {
                    html += `<li><strong>${col}:</strong> ${info.count} missing (${info.percentage}%)</li>`;
                }
                html += '</ul>';
            } else if (issueType === 'duplicates') {
                html += `<p><strong>Count:</strong> ${details.count} rows (${details.percentage}%) | <strong>Severity:</strong> <span class="badge bg-${details.severity === 'HIGH' ? 'danger' : 'warning'}">${details.severity}</span></p>`;
            } else if (issueType === 'outliers') {
                html += `<p class="mb-1"><strong>Total Outliers:</strong> ${details.total_outliers}</p>`;
                html += '<ul class="small">';
                for (const [col, info] of Object.entries(details.affected_columns)) {
                    html += `<li><strong>${col}:</strong> ${info.count} outliers (${info.percentage}%) - Range: ${info.normal_range}</li>`;
                }
                html += '</ul>';
            } else if (issueType === 'scale_issues') {
                html += `<p><strong>Columns needing normalization:</strong> ${Object.keys(details.affected_columns).length}</p>`;
            }
            
            html += '</div>';
        }
        
        html += '</div></div>';
    }
    
    // Transformations Applied Section
    if (data.transformations && data.transformations.length > 0) {
        html += '<div class="card mb-3"><div class="card-header bg-success text-white"><h5 class="mb-0"><i class="fas fa-cogs me-2"></i>Transformations Applied</h5></div><div class="card-body">';
        
        data.transformations.forEach((t) => {
            html += `
                <div class="transformation-step mb-3 p-3 border-start border-4 border-success bg-light">
                    <h6><span class="badge bg-success me-2">Step ${t.step}</span>${t.operation}</h6>
                    <p class="mb-1"><strong>Reason:</strong> ${t.reason}</p>
            `;
            
            if (t.column) html += `<p class="mb-1"><strong>Column:</strong> ${t.column}</p>`;
            if (t.columns) html += `<p class="mb-1"><strong>Columns:</strong> ${t.columns.join(', ')}</p>`;
            if (t.strategy) html += `<p class="mb-1"><strong>Strategy:</strong> ${t.strategy}</p>`;
            if (t.method) html += `<p class="mb-1"><strong>Method:</strong> ${t.method}</p>`;
            if (t.filled !== undefined) html += `<p class="mb-0"><strong>Values Filled:</strong> ${t.filled}</p>`;
            if (t.removed !== undefined) html += `<p class="mb-0"><strong>Rows Removed:</strong> ${t.removed}</p>`;
            if (t.capped_values !== undefined) html += `<p class="mb-0"><strong>Values Capped:</strong> ${t.capped_values}</p>`;
            
            html += '</div>';
        });
        
        html += '</div></div>';
    }
    
    // AI Insights Section
    if (data.ai_insights) {
        html += '<div class="card mb-3"><div class="card-header bg-info text-white"><h5 class="mb-0"><i class="fas fa-robot me-2"></i>AI Expert Analysis</h5></div><div class="card-body">';
        
        for (const [issueType, explanation] of Object.entries(data.ai_insights)) {
            html += `
                <div class="mb-3">
                    <h6 class="text-capitalize text-info">${issueType.replace('_', ' ')}</h6>
                    <p class="mb-0" style="white-space: pre-wrap;">${explanation}</p>
                </div>
            `;
        }
        
        html += '</div></div>';
    }
    
    // Summary Section
    if (data.summary) {
        html += `
            <div class="card mb-3">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Summary</h5>
                </div>
                <div class="card-body">
                    <div class="row text-center">
                        <div class="col-md-3">
                            <h3 class="text-primary">${data.summary.total_issues_found}</h3>
                            <p class="small mb-0">Issues Found</p>
                        </div>
                        <div class="col-md-3">
                            <h3 class="text-success">${data.summary.total_transformations}</h3>
                            <p class="small mb-0">Transformations</p>
                        </div>
                        <div class="col-md-3">
                            <h3 class="text-info">${data.summary.final_shape[0]}</h3>
                            <p class="small mb-0">Final Rows</p>
                        </div>
                        <div class="col-md-3">
                            <h3 class="text-warning">${data.summary.final_shape[1]}</h3>
                            <p class="small mb-0">Final Columns</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += `
        <div class="text-center mt-3">
            <button class="btn btn-success btn-lg me-2" onclick="downloadNotebook()">
                <i class="fas fa-download me-2"></i>Download Notebook
            </button>
            <button class="btn btn-primary btn-lg" onclick="window.location.reload()">
                <i class="fas fa-redo me-2"></i>Refresh Dashboard
            </button>
        </div>
    `;
    
    // Store notebook data globally
    window.cleaningNotebook = data.notebook;
    
    resultsDiv.innerHTML = html;
    document.getElementById('applyCleaningBtn').classList.add('d-none');
}

function downloadNotebook() {
    if (!window.cleaningNotebook) {
        alert('Notebook data not available');
        return;
    }
    
    const blob = new Blob([JSON.stringify(window.cleaningNotebook, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'data_cleaning_' + new Date().getTime() + '.ipynb';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
