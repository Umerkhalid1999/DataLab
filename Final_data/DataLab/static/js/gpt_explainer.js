// GPT-3.5-turbo Explainer for ML and Feature Engineering

function explainMLResults(modelName, performance, datasetInfo) {
    const btn = event.target;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Explaining...';
    btn.disabled = true;
    
    fetch('/api/explain/ml', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            model_name: modelName,
            performance: performance,
            dataset_info: datasetInfo
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showExplanationModal('ML Model Explanation', data.explanation);
        } else {
            alert('Error: ' + (data.error || 'Failed to get explanation'));
        }
    })
    .catch(err => alert('Error: ' + err))
    .finally(() => {
        btn.innerHTML = originalText;
        btn.disabled = false;
    });
}

function explainFeatureResults(operation, results) {
    const btn = event.target;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Explaining...';
    btn.disabled = true;
    
    fetch('/api/explain/feature', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            operation: operation,
            results: results
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showExplanationModal('Feature Engineering Explanation', data.explanation);
        } else {
            alert('Error: ' + (data.error || 'Failed to get explanation'));
        }
    })
    .catch(err => alert('Error: ' + err))
    .finally(() => {
        btn.innerHTML = originalText;
        btn.disabled = false;
    });
}

function showExplanationModal(title, content) {
    let modal = document.getElementById('gptExplanationModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'gptExplanationModal';
        modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;z-index:9999';
        modal.innerHTML = `
            <div style="background:white;border-radius:8px;max-width:800px;width:90%;max-height:80vh;overflow:auto;box-shadow:0 4px 20px rgba(0,0,0,0.3)">
                <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:20px;border-radius:8px 8px 0 0;display:flex;justify-content:space-between;align-items:center">
                    <h3 style="margin:0"><i class="fas fa-robot"></i> <span id="gptModalTitle"></span></h3>
                    <button onclick="document.getElementById('gptExplanationModal').remove()" style="background:none;border:none;color:white;font-size:24px;cursor:pointer;padding:0;width:30px;height:30px">&times;</button>
                </div>
                <div style="padding:20px;max-height:60vh;overflow-y:auto">
                    <div id="gptModalContent" style="line-height:1.8;color:#333;font-size:15px"></div>
                </div>
                <div style="padding:20px;border-top:1px solid #eee;text-align:right">
                    <button onclick="document.getElementById('gptExplanationModal').remove()" style="background:#6c757d;color:white;border:none;padding:10px 20px;border-radius:5px;cursor:pointer">Close</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        modal.onclick = (e) => { if(e.target === modal) modal.remove(); };
    }
    
    document.getElementById('gptModalTitle').textContent = title;
    
    // Format content with proper HTML
    let formatted = content
        // Bold text between **
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Numbered lists
        .replace(/^(\d+\.)\s+(.+)$/gm, '<div style="margin:10px 0"><strong>$1</strong> $2</div>')
        // Bullet points
        .replace(/^[•\-]\s+(.+)$/gm, '<div style="margin:5px 0;padding-left:20px">• $1</div>')
        // Headings (lines ending with :)
        .replace(/^(.+):$/gm, '<h4 style="color:#667eea;margin:15px 0 10px 0">$1:</h4>')
        // Paragraphs
        .replace(/\n\n/g, '</p><p style="margin:10px 0">')
        // Line breaks
        .replace(/\n/g, '<br>');
    
    document.getElementById('gptModalContent').innerHTML = '<p style="margin:10px 0">' + formatted + '</p>';
    modal.style.display = 'flex';
}

// Add explain buttons to ML results
function addMLExplainButtons() {
    const modelCards = document.querySelectorAll('.model-card');
    modelCards.forEach(card => {
        if (!card.querySelector('.explain-btn')) {
            const modelName = card.dataset.modelName;
            const performance = JSON.parse(card.dataset.performance || '{}');
            const datasetInfo = JSON.parse(card.dataset.datasetInfo || '{}');
            
            const btn = document.createElement('button');
            btn.className = 'btn btn-sm btn-outline-primary explain-btn mt-2';
            btn.innerHTML = '<i class="fas fa-robot me-1"></i>Explain Results';
            btn.onclick = () => explainMLResults(modelName, performance, datasetInfo);
            
            const cardBody = card.querySelector('.card-body');
            if (cardBody) cardBody.appendChild(btn);
        }
    });
}

// Add explain buttons to feature engineering results
function addFeatureExplainButtons() {
    const resultSections = document.querySelectorAll('.feature-result-section');
    resultSections.forEach(section => {
        if (!section.querySelector('.explain-btn')) {
            const operation = section.dataset.operation;
            const results = JSON.parse(section.dataset.results || '{}');
            
            const btn = document.createElement('button');
            btn.className = 'btn btn-sm btn-outline-primary explain-btn mt-2';
            btn.innerHTML = '<i class="fas fa-robot me-1"></i>Explain Results';
            btn.onclick = () => explainFeatureResults(operation, results);
            
            section.appendChild(btn);
        }
    });
}
