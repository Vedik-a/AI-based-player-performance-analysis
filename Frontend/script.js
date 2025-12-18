// Configuration
const API_BASE_URL = 'http://localhost:5000/api';
let selectedFile = null;
let lastReportData = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
    setupUploadArea();
});

// Setup drag and drop upload area
function setupUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    
    uploadArea.addEventListener('click', () => {
        document.getElementById('imageInput').click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

// Handle file selection
function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file
    if (!['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)) {
        showError('Invalid file format. Please upload PNG or JPG image.');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Maximum size is 10 MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImage = document.getElementById('previewImage');
        previewImage.src = e.target.result;
        document.getElementById('previewSection').style.display = 'block';
        clearResults();
        hideError();
        // Analyze the image automatically
        analyzeUploadedImage(file);
    };
    reader.readAsDataURL(file);
}

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        const statusIndicator = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (data.model_loaded) {
            statusIndicator.classList.add('ready');
            statusText.textContent = 'Model Ready';
        } else {
            statusIndicator.classList.remove('ready');
            statusText.textContent = 'Model Not Trained';
        }
    } catch (error) {
        const statusIndicator = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        statusIndicator.classList.remove('ready');
        statusText.textContent = 'Connection Error';
        console.error('Error checking status:', error);
    }
}

// Analyze uploaded image (removed trainModel function - model is pre-trained)
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const predictionName = document.getElementById('predictionName');
    const predictionConfidence = document.getElementById('predictionConfidence');
    const confidenceChart = document.getElementById('confidenceChart');
    const injuryContainer = document.getElementById('injuryRiskContainer');
    const injuryLevel = document.getElementById('injuryRiskLevel');
    const injuryScore = document.getElementById('injuryRiskScore');
    const injuryReasons = document.getElementById('injuryRiskReasons');

    // Update prediction
    predictionName.textContent = data.prediction;
    predictionConfidence.textContent = `Confidence: ${data.confidence}%`;

    // Create confidence breakdown
    confidenceChart.innerHTML = '';
    const sortedPredictions = Object.entries(data.all_predictions)
        .sort((a, b) => b[1] - a[1]);

    sortedPredictions.forEach(([shotName, confidence]) => {
        const barDiv = document.createElement('div');
        barDiv.className = 'confidence-bar';

        const labelDiv = document.createElement('div');
        labelDiv.className = 'confidence-label';
        labelDiv.textContent = shotName;

        const barBgDiv = document.createElement('div');
        barBgDiv.className = 'confidence-bar-bg';

        const barFillDiv = document.createElement('div');
        barFillDiv.className = 'confidence-bar-fill';
        barFillDiv.style.width = `${confidence}%`;

        const valueSpan = document.createElement('span');
        valueSpan.className = 'confidence-value';
        valueSpan.textContent = `${confidence.toFixed(1)}%`;

        barFillDiv.appendChild(valueSpan);
        barBgDiv.appendChild(barFillDiv);
        barDiv.appendChild(labelDiv);
        barDiv.appendChild(barBgDiv);
        confidenceChart.appendChild(barDiv);
    });

    // store last report data so user can download it
    lastReportData = data;
    // show download button
    const downloadBtn = document.getElementById('downloadReportBtn');
    if (downloadBtn) downloadBtn.style.display = 'block';

    resultsSection.style.display = 'block';
}

// Display injury risk if present
function displayInjuryRisk(injury) {
    const injuryContainer = document.getElementById('injuryRiskContainer');
    const injuryLevel = document.getElementById('injuryRiskLevel');
    const injuryScore = document.getElementById('injuryRiskScore');
    const injuryReasons = document.getElementById('injuryRiskReasons');

    if (!injury) {
        injuryContainer.style.display = 'none';
        return;
    }

    injuryLevel.textContent = `${injury.level}`;
    injuryScore.textContent = injury.score !== null ? `Risk score: ${injury.score}%` : '';
    injuryReasons.innerHTML = '';
    if (Array.isArray(injury.reasons)) {
        injury.reasons.forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            injuryReasons.appendChild(li);
        });
    }

    injuryContainer.style.display = 'block';
}

// Display player improvement tips
function displayPlayerTips(tips) {
    const playerTipsContainer = document.getElementById('playerTipsContainer');
    const playerTipsList = document.getElementById('playerTipsList');
    
    if (!tips || tips.length === 0) {
        playerTipsContainer.style.display = 'none';
        return;
    }
    
    playerTipsList.innerHTML = '';
    
    tips.forEach(tip => {
        const card = document.createElement('div');
        card.className = `player-tip-card ${tip.priority.toLowerCase()}`;
        
        const priority = tip.priority.toLowerCase();
        let icon = '💡';
        if (priority === 'success') {
            icon = '✓';
        } else if (priority === 'critical') {
            icon = '⚠️';
        } else if (priority === 'high') {
            icon = '⚡';
        }
        
        card.innerHTML = `
            <div class="player-tip-header">
                <div class="player-tip-category">${tip.category}</div>
                <span class="player-tip-priority ${priority}">${tip.priority}</span>
            </div>
            <div class="player-tip-icon ${priority}">${icon}</div>
            <div class="player-tip-text">${tip.tip}</div>
        `;
        
        playerTipsList.appendChild(card);
    });
    
    playerTipsContainer.style.display = 'block';
}

// Clear results
function clearResults() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('playerTipsContainer').style.display = 'none';
    const downloadBtn = document.getElementById('downloadReportBtn');
    if (downloadBtn) downloadBtn.style.display = 'none';
    lastReportData = null;
}

// Show/hide error
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

// Show status messages
function showSuccessMessage(message) {
    const statusMessage = document.getElementById('trainingStatus');
    statusMessage.textContent = message;
    statusMessage.className = 'status-message success';
}

function showErrorMessage(message) {
    const statusMessage = document.getElementById('trainingStatus');
    statusMessage.textContent = message;
    statusMessage.className = 'status-message error';
}

// Shot types removed from UI

// Analyze uploaded image for immediate feedback (prediction, tips, injury risk)
async function analyzeUploadedImage(file) {
    try {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Show prediction UI and confidence
            displayResults(data);
            // Show player tips and injury risk when available
            if (data.player_tips) displayPlayerTips(data.player_tips);
            if (data.injury_risk) displayInjuryRisk(data.injury_risk);

            // Cache report data and show download button
            lastReportData = data;
            const downloadBtn = document.getElementById('downloadReportBtn');
            if (downloadBtn) downloadBtn.style.display = 'block';
        } else {
            // Show the backend analysis error to the user so they understand why injury analysis may be missing
            showError(data.message || 'Analysis failed: no pose detected');
            // hide any previous results
            clearResults();
        }
    } catch (error) {
        console.error('Error analyzing image:', error);
    }
}

// Create and download a JSON report containing prediction and tips
function downloadReport() {
    if (!lastReportData) {
        showError('No report available to download');
        return;
    }

    const report = {
        generated_at: new Date().toISOString(),
        prediction: lastReportData.prediction || null,
        confidence: lastReportData.confidence || null,
        all_predictions: lastReportData.all_predictions || null,
        player_tips: lastReportData.player_tips || [],
        injury_risk: lastReportData.injury_risk || null
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const filename = `cricket-report-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        URL.revokeObjectURL(url);
        a.remove();
    }, 1000);
}

// Note: Image quality analysis UI removed. Player tips displayed via displayPlayerTips().