class BLExtractor {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.currentFile = null;
        this.extractionData = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAPIStatus();
        this.checkCapabilities();
    }

    setupEventListeners() {
        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const removeFileBtn = document.getElementById('remove-file');
        const extractBtn = document.getElementById('extract-btn');

        // Upload area events
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));

        // File input
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        removeFileBtn.addEventListener('click', this.removeFile.bind(this));

        // Extract button
        extractBtn.addEventListener('click', this.extractData.bind(this));

        // Results actions
        document.getElementById('download-json').addEventListener('click', this.downloadJSON.bind(this));
        document.getElementById('new-extraction').addEventListener('click', this.newExtraction.bind(this));

        // Tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
    }

    async checkAPIStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const statusElement = document.getElementById('api-status');
            
            if (response.ok) {
                statusElement.className = 'status-item online';
                statusElement.innerHTML = '<i class="fas fa-circle"></i><span>API: En ligne</span>';
            } else {
                throw new Error('API offline');
            }
        } catch (error) {
            const statusElement = document.getElementById('api-status');
            statusElement.className = 'status-item offline';
            statusElement.innerHTML = '<i class="fas fa-circle"></i><span>API: Hors ligne</span>';
        }
    }

    async checkCapabilities() {
        try {
            const response = await fetch(`${this.apiBase}/capabilities`);
            const data = await response.json();
            
            const gpuStatus = document.getElementById('gpu-status');
            const capabilities = data.capabilities;
            
            if (capabilities.gpu_acceleration && capabilities.nvidia_gpu) {
                gpuStatus.className = 'status-item online';
                gpuStatus.innerHTML = `<i class="fas fa-microchip"></i><span>GPU: Actif (${capabilities.gpu_memory_mb}MB)</span>`;
            } else if (capabilities.nvidia_gpu && !capabilities.gpu_acceleration) {
                gpuStatus.className = 'status-item warning';
                gpuStatus.innerHTML = `<i class="fas fa-microchip"></i><span>GPU: Détecté mais non utilisé</span>`;
            } else {
                gpuStatus.className = 'status-item offline';
                gpuStatus.innerHTML = '<i class="fas fa-microchip"></i><span>GPU: Non disponible</span>';
            }
        } catch (error) {
            console.error('Error checking capabilities:', error);
            const gpuStatus = document.getElementById('gpu-status');
            gpuStatus.className = 'status-item offline';
            gpuStatus.innerHTML = '<i class="fas fa-microchip"></i><span>GPU: Erreur détection</span>';
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        const allowedTypes = ['application/pdf', 'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
        const maxSize = 50 * 1024 * 1024; // 50MB

        if (!allowedTypes.includes(file.type)) {
            this.showError('Type de fichier non supporté. Veuillez télécharger un PDF ou une image.');
            return;
        }

        if (file.size > maxSize) {
            this.showError('Fichier trop volumineux. Taille maximum: 50MB');
            return;
        }

        this.currentFile = file;
        this.showFileInfo(file);
        this.enableExtraction();
    }

    showFileInfo(file) {
        const uploadContent = document.querySelector('.upload-content');
        const fileInfo = document.getElementById('file-info');
        const fileName = fileInfo.querySelector('.file-name');
        const fileSize = fileInfo.querySelector('.file-size');

        uploadContent.style.display = 'none';
        fileInfo.style.display = 'flex';
        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);
    }

    removeFile() {
        this.currentFile = null;
        const uploadContent = document.querySelector('.upload-content');
        const fileInfo = document.getElementById('file-info');
        const fileInput = document.getElementById('file-input');

        uploadContent.style.display = 'block';
        fileInfo.style.display = 'none';
        fileInput.value = '';
        this.disableExtraction();
    }

    enableExtraction() {
        const extractBtn = document.getElementById('extract-btn');
        extractBtn.disabled = false;
    }

    disableExtraction() {
        const extractBtn = document.getElementById('extract-btn');
        extractBtn.disabled = true;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async extractData() {
        if (!this.currentFile) return;

        const formData = new FormData();
        formData.append('file', this.currentFile);
        formData.append('ocr_method', document.getElementById('ocr-method').value);
        formData.append('use_docling', document.getElementById('use-docling').checked);
        formData.append('use_llm', document.getElementById('use-llm').checked);

        this.showLoading();
        const startTime = Date.now();

        try {
            const response = await fetch(`${this.apiBase}/extract`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const endTime = Date.now();
            const extractionTime = ((endTime - startTime) / 1000).toFixed(2);

            this.extractionData = data;
            this.showResults(data, extractionTime);
        } catch (error) {
            console.error('Extraction error:', error);
            this.showError('Erreur lors de l\'extraction: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    showLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingMessage = document.getElementById('loading-message');
        const progressFill = document.getElementById('progress-fill');

        loadingOverlay.style.display = 'flex';
        
        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            
            progressFill.style.width = progress + '%';
            
            if (progress < 30) {
                loadingMessage.textContent = 'Analyse du fichier...';
            } else if (progress < 60) {
                loadingMessage.textContent = 'Extraction OCR en cours...';
            } else if (progress < 90) {
                loadingMessage.textContent = 'Traitement avec LLM...';
            }
        }, 500);

        this.progressInterval = interval;
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        const progressFill = document.getElementById('progress-fill');
        
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        progressFill.style.width = '100%';
        setTimeout(() => {
            loadingOverlay.style.display = 'none';
            progressFill.style.width = '0%';
        }, 500);
    }

    showResults(data, extractionTime) {
        const resultsSection = document.getElementById('results-section');
        const uploadSection = document.querySelector('.upload-section');
        
        // Update extraction info
        document.getElementById('extraction-method').textContent = data.extraction_method || 'N/A';
        document.getElementById('confidence-score').textContent = data.confidence_score ? 
            `${(data.confidence_score * 100).toFixed(1)}%` : 'N/A';
        document.getElementById('extraction-time').textContent = `${extractionTime}s`;

        // Populate structured data
        this.populateStructuredData(data);
        
        // Populate JSON
        document.getElementById('json-output').textContent = JSON.stringify(data, null, 2);

        // Show results
        uploadSection.style.display = 'none';
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    populateStructuredData(data) {
        // Parties
        const partiesData = document.getElementById('parties-data');
        partiesData.innerHTML = '';
        
        const parties = [
            { label: 'Expéditeur', value: data.shipper?.name || 'N/A' },
            { label: 'Destinataire', value: data.consignee?.name || 'N/A' },
            { label: 'Partie à notifier', value: data.notify_party?.name || 'N/A' }
        ];

        parties.forEach(party => {
            const item = this.createDataItem(party.label, party.value);
            partiesData.appendChild(item);
        });

        // Ports
        const portsData = document.getElementById('ports-data');
        portsData.innerHTML = '';
        
        const ports = [
            { label: 'Port de chargement', value: data.port_of_loading || 'N/A' },
            { label: 'Port de déchargement', value: data.port_of_discharge || 'N/A' },
            { label: 'Lieu de livraison', value: data.place_of_delivery || 'N/A' }
        ];

        ports.forEach(port => {
            const item = this.createDataItem(port.label, port.value);
            portsData.appendChild(item);
        });

        // Transport
        const transportData = document.getElementById('transport-data');
        transportData.innerHTML = '';
        
        const transport = [
            { label: 'Navire', value: data.vessel_name || 'N/A' },
            { label: 'Voyage', value: data.voyage_number || 'N/A' },
            { label: 'Numéro BL', value: data.bl_number || 'N/A' },
            { label: 'Date d\'émission', value: data.date_of_issue || 'N/A' }
        ];

        transport.forEach(item => {
            const element = this.createDataItem(item.label, item.value);
            transportData.appendChild(element);
        });

        // Cargo
        const cargoData = document.getElementById('cargo-data');
        cargoData.innerHTML = '';
        
        const cargo = [
            { label: 'Description', value: data.description_of_goods || 'N/A' },
            { label: 'Poids brut', value: data.gross_weight || 'N/A' },
            { label: 'Nombre de conteneurs', value: data.number_of_containers || 'N/A' },
            { label: 'Conteneurs', value: data.container_numbers?.join(', ') || 'N/A' }
        ];

        cargo.forEach(item => {
            const element = this.createDataItem(item.label, item.value);
            cargoData.appendChild(element);
        });
    }

    createDataItem(label, value) {
        const item = document.createElement('div');
        item.className = 'data-item';
        
        const labelElement = document.createElement('span');
        labelElement.className = 'label';
        labelElement.textContent = label;
        
        const valueElement = document.createElement('span');
        valueElement.className = value === 'N/A' ? 'value empty' : 'value';
        valueElement.textContent = value;
        
        item.appendChild(labelElement);
        item.appendChild(valueElement);
        
        return item;
    }

    switchTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    downloadJSON() {
        if (!this.extractionData) return;
        
        const dataStr = JSON.stringify(this.extractionData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = 'extraction_results.json';
        link.click();
        
        URL.revokeObjectURL(url);
    }

    newExtraction() {
        const resultsSection = document.getElementById('results-section');
        const uploadSection = document.querySelector('.upload-section');
        
        resultsSection.style.display = 'none';
        uploadSection.style.display = 'block';
        
        this.removeFile();
        this.extractionData = null;
        
        uploadSection.scrollIntoView({ behavior: 'smooth' });
    }

    showError(message) {
        alert(message); // Simple error handling - could be enhanced with better UI
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new BLExtractor();
});