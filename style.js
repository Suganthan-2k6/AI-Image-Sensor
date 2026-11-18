const API_URL = 'http://localhost:8000/detect';

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const resultsContainer = document.getElementById('resultsContainer');
    const loader = document.getElementById('loader');

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Click to upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    async function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'flex';
            resultsContainer.style.display = 'none';
        };
        reader.readAsDataURL(file);

        // Show loader
        loader.classList.add('active');

        // Upload and analyze
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze image. Make sure the backend server is running.');
        } finally {
            loader.classList.remove('active');
        }
    }

    function displayResults(data) {
        resultsContainer.style.display = 'block';

        // Verdict
        const verdict = document.getElementById('verdict');
        const isAi = data.verdict === 'AI-Generated';
        verdict.textContent = isAi ? '🤖 AI-Generated Image' : '📸 Real Image';
        verdict.className = `verdict ${isAi ? 'ai' : 'real'}`;

        // Score
        const score = data.overall_ai_score || 0;
        document.getElementById('scoreFill').style.width = `${score}%`;
        document.getElementById('scoreText').textContent = `${score}%`;

        // Details
        document.getElementById('filename').textContent = data.filename || '-';
        document.getElementById('filesize').textContent = 
            data.file_size ? `${(data.file_size / 1024).toFixed(1)} KB` : '-';
        
        const pred = data.ai_detection?.prediction || 'unknown';
        document.getElementById('prediction').textContent = 
            pred === 'ai' ? 'AI-Generated' : 'Real';
        document.getElementById('prediction').style.color = 
            pred === 'ai' ? '#ff4757' : '#2ed573';
        
        document.getElementById('confidence').textContent = 
            data.ai_detection?.confidence ? `${data.ai_detection.confidence}%` : '-';

        // Raw JSON for detailed view
        document.getElementById('rawResults').textContent = 
            JSON.stringify(data, null, 2);
    }
});

function toggleDetails() {
    const panel = document.getElementById('detailPanel');
    const btn = document.querySelector('.toggle-btn');
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        btn.textContent = '▲ Hide Detailed Analysis';
    } else {
        panel.style.display = 'none';
        btn.textContent = '▼ Show Detailed Analysis';
    }
}

function resetApp() {
    document.getElementById('previewContainer').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('fileInput').value = '';
    document.getElementById('previewImage').src = '';
}