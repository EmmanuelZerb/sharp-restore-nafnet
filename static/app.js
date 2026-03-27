/**
 * Sharp Restore — Image Deblurring Frontend
 */

class SharpRestore {
    constructor() {
        this.selectedModel = 'nafnet32';
        this.currentJobId = null;
        this.pollInterval = null;

        this.init();
    }

    init() {
        this.cacheElements();
        this.bindEvents();
        this.checkStatus();
    }

    cacheElements() {
        // Sections
        this.uploadSection = document.getElementById('uploadSection');
        this.processingSection = document.getElementById('processingSection');
        this.resultSection = document.getElementById('resultSection');

        // Upload
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.modelSelector = document.getElementById('modelSelector');

        // Processing
        this.processingPreview = document.getElementById('processingPreview');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');

        // Result
        this.beforeImage = document.getElementById('beforeImage');
        this.afterImage = document.getElementById('afterImage');
        this.comparisonWrapper = document.getElementById('comparisonWrapper');
        this.comparisonSlider = document.getElementById('comparisonSlider');
        this.statTime = document.getElementById('statTime');
        this.statModel = document.getElementById('statModel');
        this.btnNewImage = document.getElementById('btnNewImage');
        this.btnDownload = document.getElementById('btnDownload');

        // Status
        this.statusIndicator = document.getElementById('statusIndicator');
    }

    bindEvents() {
        // Upload zone click
        this.uploadZone.addEventListener('click', () => this.fileInput.click());

        // File input change
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadZone.addEventListener('drop', (e) => this.handleDrop(e));

        // Model selector
        this.modelSelector.addEventListener('click', (e) => this.handleModelSelect(e));

        // Result actions
        this.btnNewImage.addEventListener('click', () => this.resetToUpload());
        this.btnDownload.addEventListener('click', () => this.downloadResult());

        // Comparison slider
        this.initComparisonSlider();
    }

    // ========================================
    // Status Check
    // ========================================

    async checkStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            this.statusIndicator.classList.remove('online', 'offline');

            if (data.gpu_available) {
                this.statusIndicator.classList.add('online');
                this.statusIndicator.querySelector('.status-text').textContent =
                    `${data.gpu_name} • ${data.vram_gb}GB`;
            } else {
                this.statusIndicator.classList.add('offline');
                this.statusIndicator.querySelector('.status-text').textContent = 'CPU Mode';
            }

            // Update model buttons based on available weights
            data.models.forEach(model => {
                const btn = this.modelSelector.querySelector(`[data-model="${model.name}"]`);
                if (btn && !model.weights_available) {
                    btn.classList.add('no-weights');
                }
            });
        } catch (error) {
            this.statusIndicator.classList.add('offline');
            this.statusIndicator.querySelector('.status-text').textContent = 'Offline';
        }
    }

    // ========================================
    // File Handling
    // ========================================

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image file (JPG, PNG, WebP, or BMP)');
            return;
        }

        // Validate file size (20MB max)
        if (file.size > 20 * 1024 * 1024) {
            alert('File size must be less than 20MB');
            return;
        }

        // Show processing section
        this.showSection('processing');

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.processingPreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Upload and process
        await this.uploadAndProcess(file);
    }

    // ========================================
    // Model Selection
    // ========================================

    handleModelSelect(e) {
        const btn = e.target.closest('.selector-option');
        if (!btn) return;

        // Update selection
        this.modelSelector.querySelectorAll('.selector-option').forEach(b => {
            b.classList.remove('active');
        });
        btn.classList.add('active');

        this.selectedModel = btn.dataset.model;
    }

    // ========================================
    // Upload & Processing
    // ========================================

    async uploadAndProcess(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_type', this.selectedModel);

        try {
            this.updateProgress(5, 'Uploading...');

            const response = await fetch('/api/restore', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const data = await response.json();
            this.currentJobId = data.job_id;

            this.updateProgress(10, 'Processing...');
            this.startPolling();

        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload image. Please try again.');
            this.resetToUpload();
        }
    }

    startPolling() {
        this.pollInterval = setInterval(() => this.pollJobStatus(), 500);
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    async pollJobStatus() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/job/${this.currentJobId}`);
            const job = await response.json();

            this.updateProgress(job.progress, this.getProgressMessage(job.progress));

            if (job.status === 'completed') {
                this.stopPolling();
                this.showResult(job);
            } else if (job.status === 'failed') {
                this.stopPolling();
                alert(`Processing failed: ${job.error}`);
                this.resetToUpload();
            }

        } catch (error) {
            console.error('Polling error:', error);
        }
    }

    getProgressMessage(progress) {
        if (progress < 20) return 'Loading image...';
        if (progress < 40) return 'Loading model...';
        if (progress < 90) return 'Restoring image...';
        return 'Finalizing...';
    }

    updateProgress(percent, message) {
        this.progressFill.style.width = `${percent}%`;
        this.progressText.textContent = message;
    }

    // ========================================
    // Result Display
    // ========================================

    async showResult(job) {
        // Load images
        this.beforeImage.src = `/api/image/input/${this.currentJobId}`;
        this.afterImage.src = `/api/image/output/${this.currentJobId}`;

        // Wait for images to load
        await Promise.all([
            new Promise(resolve => this.beforeImage.onload = resolve),
            new Promise(resolve => this.afterImage.onload = resolve)
        ]);

        // Update stats
        this.statTime.textContent = `${job.processing_time}s`;
        this.statModel.textContent = this.getModelDisplayName(this.selectedModel);

        // Reset slider position
        this.setSliderPosition(50);

        // Show result section
        this.showSection('result');
    }

    getModelDisplayName(model) {
        const names = {
            'nafnet32': 'Standard',
            'nafnet64': 'Quality',
            'nafnet_deblur': 'Maximum'
        };
        return names[model] || model;
    }

    // ========================================
    // Comparison Slider
    // ========================================

    initComparisonSlider() {
        let isDragging = false;

        const startDrag = (e) => {
            isDragging = true;
            e.preventDefault();
        };

        const stopDrag = () => {
            isDragging = false;
        };

        const doDrag = (e) => {
            if (!isDragging) return;

            const rect = this.comparisonWrapper.getBoundingClientRect();
            let x = (e.clientX || e.touches[0].clientX) - rect.left;
            let percent = (x / rect.width) * 100;
            percent = Math.max(0, Math.min(100, percent));

            this.setSliderPosition(percent);
        };

        // Mouse events
        this.comparisonSlider.addEventListener('mousedown', startDrag);
        document.addEventListener('mouseup', stopDrag);
        document.addEventListener('mousemove', doDrag);

        // Touch events
        this.comparisonSlider.addEventListener('touchstart', startDrag);
        document.addEventListener('touchend', stopDrag);
        document.addEventListener('touchmove', doDrag);

        // Click on wrapper to move slider
        this.comparisonWrapper.addEventListener('click', (e) => {
            const rect = this.comparisonWrapper.getBoundingClientRect();
            let x = e.clientX - rect.left;
            let percent = (x / rect.width) * 100;
            percent = Math.max(0, Math.min(100, percent));
            this.setSliderPosition(percent);
        });
    }

    setSliderPosition(percent) {
        const beforeContainer = this.comparisonWrapper.querySelector('.comparison-before');
        beforeContainer.style.width = `${percent}%`;
        this.comparisonSlider.style.left = `${percent}%`;
    }

    // ========================================
    // Actions
    // ========================================

    downloadResult() {
        if (!this.currentJobId) return;

        const link = document.createElement('a');
        link.href = `/api/image/output/${this.currentJobId}`;
        link.download = `restored_${this.currentJobId}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    resetToUpload() {
        this.stopPolling();
        this.currentJobId = null;
        this.fileInput.value = '';
        this.progressFill.style.width = '0%';
        this.showSection('upload');
    }

    // ========================================
    // Section Management
    // ========================================

    showSection(section) {
        this.uploadSection.classList.add('hidden');
        this.processingSection.classList.add('hidden');
        this.resultSection.classList.add('hidden');

        switch (section) {
            case 'upload':
                this.uploadSection.classList.remove('hidden');
                break;
            case 'processing':
                this.processingSection.classList.remove('hidden');
                break;
            case 'result':
                this.resultSection.classList.remove('hidden');
                break;
        }
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SharpRestore();
});
