document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const previewArea = document.getElementById('previewArea');
    const imagePreview = document.getElementById('imagePreview');
    const changeImageBtn = document.getElementById('changeImageBtn');
    const predictBtn = document.getElementById('predictBtn');
    
    const loadingState = document.getElementById('loadingState');
    const uploadSection = document.querySelector('.upload-section');
    const resultsSection = document.getElementById('resultsSection');
    
    const errorToast = document.getElementById('errorToast');
    const newDiagnosisBtn = document.getElementById('newDiagnosisBtn');

    const remediesWrapper = document.getElementById('remediesWrapper');
    const toggleRemediesBtn = document.getElementById('toggleRemediesBtn');
    const remediesContainer = document.getElementById('remediesContainer');
    const remImmediate = document.getElementById('remImmediate');
    const remOrganic = document.getElementById('remOrganic');
    const remChemical = document.getElementById('remChemical');

    let selectedFile = null;

    // --- Drag and Drop Handlers ---
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
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // --- Click Upload Handler ---
    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            showError('Please upload a valid JPG or PNG image.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewArea.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // --- Buttons ---
    changeImageBtn.addEventListener('click', () => {
        selectedFile = null;
        imageInput.value = '';
        previewArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    newDiagnosisBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        changeImageBtn.click();
    });

    toggleRemediesBtn.addEventListener('click', () => {
        if (remediesContainer.classList.contains('hidden')) {
            remediesContainer.classList.remove('hidden');
            toggleRemediesBtn.textContent = 'Hide Treatment Plan';
        } else {
            remediesContainer.classList.add('hidden');
            toggleRemediesBtn.textContent = 'Show Treatment Plan';
        }
    });

    // --- Prediction API Call ---
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI Updates for Loading
        previewArea.classList.add('hidden');
        dropZone.classList.add('hidden');
        document.getElementById('uploadTitle').classList.add('hidden');
        document.getElementById('uploadDesc').classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            displayResults(data);

        } catch (err) {
            showError(err.message);
            // Reset UI
            loadingState.classList.add('hidden');
            previewArea.classList.remove('hidden');
            document.getElementById('uploadTitle').classList.remove('hidden');
            document.getElementById('uploadDesc').classList.remove('hidden');
        }
    });

    function displayResults(data) {
        // Hide Upload Section, Show Results
        uploadSection.classList.add('hidden');
        loadingState.classList.add('hidden');
        
        // Reset Titles
        document.getElementById('uploadTitle').classList.remove('hidden');
        document.getElementById('uploadDesc').classList.remove('hidden');

        // Main Prediction
        const diseaseNameEl = document.getElementById('diseaseName');
        const mainCard = document.getElementById('mainResultCard');
        
        diseaseNameEl.textContent = data.disease;
        
        // Handle "Unknown" styling or "Healthy"
        if (data.is_unknown) {
            mainCard.style.backgroundColor = '#7F8C8D'; // Grey for unknown
            document.getElementById('confidenceValueBadge').classList.add('hidden');
        } else {
            document.getElementById('confidenceValueBadge').classList.remove('hidden');
            document.getElementById('confidenceValue').textContent = data.confidence + '%';
            
            if (data.disease.toLowerCase().includes('healthy')) {
                mainCard.style.backgroundColor = '#4CAF50'; // Green for healthy
            } else {
                mainCard.style.backgroundColor = '#E74C3C'; // Red for disease
            }
        }

        // Top 2 Prediction
        const top2Container = document.getElementById('top2Container');
        if (data.top_2 && !data.is_unknown) {
            document.getElementById('top2DiseaseName').textContent = data.top_2.disease;
            document.getElementById('top2ConfidenceValue').textContent = data.top_2.confidence + '%';
            top2Container.classList.remove('hidden');
        } else {
            top2Container.classList.add('hidden');
        }

        // Remedies
        if (data.remedy && !data.is_unknown) {
            remImmediate.textContent = data.remedy.immediate || 'N/A';
            remOrganic.textContent = data.remedy.organic || 'N/A';
            remChemical.textContent = data.remedy.chemical || 'N/A';
            
            remediesWrapper.classList.remove('hidden');
            remediesContainer.classList.add('hidden'); // hidden by default until clicked
            toggleRemediesBtn.textContent = 'Show Treatment Plan';
        } else {
            remediesWrapper.classList.add('hidden');
        }

        resultsSection.classList.remove('hidden');
    }

    function showError(msg) {
        errorToast.textContent = msg;
        errorToast.classList.remove('hidden');
        setTimeout(() => {
            errorToast.classList.add('hidden');
        }, 4000);
    }
});
