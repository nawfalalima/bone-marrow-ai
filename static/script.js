const dropZone = document.getElementById('drop-zone');
const input = document.getElementById('image-input');
const previewWrapper = document.getElementById('preview-wrapper');
const imagePreview = document.getElementById('image-preview');
const statusMessage = document.getElementById('status-message');
const analyzeBtn = document.getElementById('analyze-btn');
const resultCard = document.getElementById('result-card');
const predictionName = document.getElementById('prediction-name');
const confidenceValue = document.getElementById('confidence-value');
const confidenceBar = document.getElementById('confidence-bar');
const probabilityList = document.getElementById('probability-list');
const removeImageBtn = document.getElementById('remove-image');
const fileTrigger = document.querySelector('.file-trigger');

let selectedFile = null;

const showStatus = (message, type = '') => {
    statusMessage.textContent = message;
    statusMessage.className = 'status-message';
    if (type) {
        statusMessage.classList.add(type);
    }
    statusMessage.classList.remove('hidden');
};

const hideStatus = () => {
    statusMessage.classList.add('hidden');
};

const resetResult = () => {
    resultCard.classList.add('hidden');
    predictionName.textContent = '--';
    confidenceValue.textContent = '0%';
    confidenceBar.style.width = '0%';
    probabilityList.innerHTML = '';
};

const setPreview = (file) => {
    if (!file) return;

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        previewWrapper.classList.remove('hidden');
        dropZone.classList.add('hidden');
        analyzeBtn.disabled = false;
        hideStatus();
        resetResult();
    };
    reader.readAsDataURL(file);
};

const clearSelection = () => {
    input.value = '';
    selectedFile = null;
    previewWrapper.classList.add('hidden');
    dropZone.classList.remove('hidden');
    analyzeBtn.disabled = true;
    imagePreview.removeAttribute('src');
    resetResult();
    hideStatus();
};

const renderProbabilityBars = (probabilities) => {
    const entries = Object.entries(probabilities);
    probabilityList.innerHTML = entries.map(([label, value], index) => {
        const isTopPrediction = index === 0;
        return `
            <div class="probability-row ${isTopPrediction ? 'highlight' : ''}">
                <span class="class-name">${label}</span>
                <div class="probability-track">
                    <div class="probability-fill" style="width: ${value}%"></div>
                </div>
                <span class="probability-percent">${Number(value).toFixed(2)}%</span>
            </div>
        `;
    }).join('');
};

fileTrigger.addEventListener('click', () => input.click());
input.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        setPreview(file);
    }
});

removeImageBtn.addEventListener('click', clearSelection);

['dragenter', 'dragover'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add('drag-over');
    });
});

['dragleave', 'drop'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove('drag-over');
    });
});

dropZone.addEventListener('drop', (event) => {
    const file = event.dataTransfer.files[0];
    if (file) {
        input.files = event.dataTransfer.files;
        setPreview(file);
    }
});

dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        input.click();
    }
});

analyzeBtn.disabled = true;

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showStatus('Please upload an image before analysis.', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    analyzeBtn.disabled = true;
    showStatus('Analyzing image...', 'loading');
    resultCard.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Prediction failed.');
        }

        predictionName.textContent = data.prediction;
        confidenceValue.textContent = `${Number(data.confidence).toFixed(2)}%`;
        confidenceBar.style.width = `${data.confidence}%`;
        renderProbabilityBars(data.probabilities);
        resultCard.classList.remove('hidden');
        hideStatus();
    } catch (error) {
        showStatus(error.message || 'Unable to analyze the uploaded image.', 'error');
    } finally {
        analyzeBtn.disabled = false;
    }
});
