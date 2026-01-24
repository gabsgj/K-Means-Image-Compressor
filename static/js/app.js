/**
 * K-Means Image Compressor - Frontend Application
 * ================================================
 * Handles image upload, compression settings, and UI interactions.
 */

// ============================================================================
// State Management
// ============================================================================
const state = {
    currentImageId: null,
    compressedImageId: null,
    originalPreviewUrl: null,
    compressedPreviewUrl: null,
    downloadUrl: null,
    isProcessing: false,
    originalDimensions: { width: 0, height: 0 },
    settings: {
        nColors: 16,
        maxIters: 10,
        targetWidth: null,
        targetHeight: null,
        resizeMode: 'fit',
        targetSizeKb: null,
        enableTargetSize: false,
        lockAspectRatio: true,
        cropData: null
    },
    cropper: null
};

// ============================================================================
// DOM Elements
// ============================================================================
const elements = {
    // Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadSection: document.getElementById('uploadSection'),
    
    // Settings
    settingsSection: document.getElementById('settingsSection'),
    presetButtons: document.querySelectorAll('.preset-btn'),
    nColorsSlider: document.getElementById('nColors'),
    nColorsValue: document.getElementById('nColorsValue'),
    maxItersSlider: document.getElementById('maxIters'),
    maxItersValue: document.getElementById('maxItersValue'),
    
    // Dimension Settings
    dimensionToggle: document.getElementById('dimensionToggle'),
    dimensionContent: document.getElementById('dimensionContent'),
    dimensionPreset: document.getElementById('dimensionPreset'),
    customDimensions: document.getElementById('customDimensions'),
    targetWidth: document.getElementById('targetWidth'),
    targetHeight: document.getElementById('targetHeight'),
    lockAspectBtn: document.getElementById('lockAspectBtn'),
    resizeModeSection: document.getElementById('resizeModeSection'),
    modeBtns: document.querySelectorAll('.mode-btn'),
    
    // File Size Settings
    filesizeToggle: document.getElementById('filesizeToggle'),
    filesizeContent: document.getElementById('filesizeContent'),
    enableTargetSize: document.getElementById('enableTargetSize'),
    filesizeControls: document.getElementById('filesizeControls'),
    targetSizeKb: document.getElementById('targetSizeKb'),
    sizePresetBtns: document.querySelectorAll('.size-preset-btn'),
    
    // Crop Modal
    cropModal: document.getElementById('cropModal'),
    cropImage: document.getElementById('cropImage'),
    cropDimensions: document.getElementById('cropDimensions'),
    cropAspectInfo: document.getElementById('cropAspectInfo'),
    closeCropModal: document.getElementById('closeCropModal'),
    cancelCropBtn: document.getElementById('cancelCropBtn'),
    applyCropBtn: document.getElementById('applyCropBtn'),
    aspectBtns: document.querySelectorAll('.aspect-btn'),
    rotateLeftBtn: document.getElementById('rotateLeftBtn'),
    rotateRightBtn: document.getElementById('rotateRightBtn'),
    flipHBtn: document.getElementById('flipHBtn'),
    flipVBtn: document.getElementById('flipVBtn'),
    resetCropBtn: document.getElementById('resetCropBtn'),
    
    // Images
    originalImage: document.getElementById('originalImage'),
    originalSize: document.getElementById('originalSize'),
    originalInfo: document.getElementById('originalInfo'),
    compressedImage: document.getElementById('compressedImage'),
    compressedSize: document.getElementById('compressedSize'),
    compressionStats: document.getElementById('compressionStats'),
    
    // Buttons
    resetBtn: document.getElementById('resetBtn'),
    compressBtn: document.getElementById('compressBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    
    // Result & Comparison
    resultCard: document.getElementById('resultCard'),
    comparisonSection: document.getElementById('comparisonSection'),
    compareOriginal: document.getElementById('compareOriginal'),
    compareCompressed: document.getElementById('compareCompressed'),
    comparisonOverlay: document.getElementById('comparisonOverlay'),
    comparisonSlider: document.getElementById('comparisonSlider'),
    
    // Loading
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    loadingSubtext: document.getElementById('loadingSubtext'),
    
    // Toast
    toastContainer: document.getElementById('toastContainer')
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format file size in human-readable format
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };
    
    toast.innerHTML = `
        <i class="fas ${icons[type] || icons.info}"></i>
        <span class="toast-message">${message}</span>
        <button class="toast-close"><i class="fas fa-times"></i></button>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
    
    // Close button
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.remove();
    });
}

/**
 * Show/hide loading overlay
 */
function setLoading(isLoading, text = 'Processing...', subtext = '') {
    state.isProcessing = isLoading;
    
    if (isLoading) {
        elements.loadingOverlay.classList.remove('hidden');
        elements.loadingText.textContent = text;
        elements.loadingSubtext.textContent = subtext;
    } else {
        elements.loadingOverlay.classList.add('hidden');
    }
    
    elements.compressBtn.disabled = isLoading;
}

/**
 * API request wrapper
 */
async function apiRequest(endpoint, options = {}) {
    const response = await fetch(`/api${endpoint}`, {
        ...options,
        headers: {
            ...options.headers
        }
    });
    
    const data = await response.json();
    
    if (!response.ok || !data.success) {
        throw new Error(data.error || 'An error occurred');
    }
    
    return data;
}

// ============================================================================
// Upload Handlers
// ============================================================================

/**
 * Handle file selection
 */
async function handleFileSelect(file) {
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showToast('Please select a valid image file (PNG, JPG, GIF, BMP, WebP)', 'error');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showToast('File size must be less than 16MB', 'error');
        return;
    }
    
    try {
        setLoading(true, 'Uploading image...', 'Please wait');
        
        const formData = new FormData();
        formData.append('image', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error);
        }
        
        // Update state
        state.currentImageId = data.image_id;
        state.originalPreviewUrl = data.preview_url;
        
        // Update UI
        elements.originalImage.src = data.preview_url;
        elements.originalSize.textContent = formatFileSize(data.file_size);
        elements.originalInfo.innerHTML = `
            <strong>${data.dimensions.width} × ${data.dimensions.height}</strong> pixels | 
            ${data.format || 'Unknown format'}
        `;
        
        // Store original dimensions
        state.originalDimensions = {
            width: data.dimensions.width,
            height: data.dimensions.height
        };
        
        // Show settings section
        elements.uploadSection.classList.add('hidden');
        elements.settingsSection.classList.remove('hidden');
        
        // Reset result card
        elements.resultCard.classList.add('hidden');
        elements.comparisonSection.classList.add('hidden');
        
        showToast('Image uploaded successfully!', 'success');
        
    } catch (error) {
        console.error('Upload error:', error);
        showToast(error.message || 'Failed to upload image', 'error');
    } finally {
        setLoading(false);
    }
}

// ============================================================================
// Compression Handler
// ============================================================================

/**
 * Compress the uploaded image
 */
async function compressImage() {
    if (!state.currentImageId || state.isProcessing) return;
    
    try {
        const hasAdvancedSettings = 
            state.settings.targetWidth || 
            state.settings.targetHeight || 
            state.settings.enableTargetSize ||
            state.settings.cropData;
        
        let loadingSubtext = `Using ${state.settings.nColors} colors`;
        if (hasAdvancedSettings) {
            loadingSubtext = 'Applying advanced compression settings';
        }
        
        setLoading(true, 'Compressing image...', loadingSubtext);
        
        // Build request body
        const requestBody = {
            image_id: state.currentImageId,
            n_colors: state.settings.nColors,
            max_iters: state.settings.maxIters
        };
        
        // Use advanced endpoint if dimension/size settings are specified
        const endpoint = hasAdvancedSettings ? '/api/compress-advanced' : '/api/compress';
        
        if (hasAdvancedSettings) {
            if (state.settings.targetWidth) {
                requestBody.target_width = state.settings.targetWidth;
            }
            if (state.settings.targetHeight) {
                requestBody.target_height = state.settings.targetHeight;
            }
            requestBody.resize_mode = state.settings.resizeMode;
            
            if (state.settings.enableTargetSize && state.settings.targetSizeKb) {
                requestBody.target_size_kb = state.settings.targetSizeKb;
            }
            
            if (state.settings.cropData) {
                requestBody.crop_data = state.settings.cropData;
            }
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error);
        }
        
        // Update state (add cache-busting timestamp)
        const timestamp = Date.now();
        state.compressedImageId = data.compressed_id;
        state.compressedPreviewUrl = `${data.preview_url}?t=${timestamp}`;
        state.downloadUrl = data.download_url;
        
        // Update UI
        elements.compressedImage.src = state.compressedPreviewUrl;
        elements.compressedSize.textContent = formatFileSize(data.stats.compressed_file_size);
        
        // Build stats HTML
        let statsHtml = `
            <div class="stat-item">
                <div class="stat-value">${data.stats.file_compression_ratio}x</div>
                <div class="stat-label">Compression Ratio</div>
            </div>
        `;
        
        if (data.stats.reduced_colors) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-value">${data.stats.reduced_colors}</div>
                    <div class="stat-label">Colors Used</div>
                </div>
            `;
        } else if (data.stats.colors_used) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-value">${data.stats.colors_used}</div>
                    <div class="stat-label">Colors Used</div>
                </div>
            `;
        }
        
        if (data.stats.iterations) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-value">${data.stats.iterations}</div>
                    <div class="stat-label">Iterations</div>
                </div>
            `;
        }
        
        if (data.stats.processing_time_ms) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-value">${data.stats.processing_time_ms.toFixed(0)}ms</div>
                    <div class="stat-label">Processing Time</div>
                </div>
            `;
        }
        
        if (data.stats.final_dimensions) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-value">${data.stats.final_dimensions[0]}×${data.stats.final_dimensions[1]}</div>
                    <div class="stat-label">Final Size</div>
                </div>
            `;
        }
        
        if (data.stats.achieved_size_kb) {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-value">${data.stats.achieved_size_kb.toFixed(1)}KB</div>
                    <div class="stat-label">File Size</div>
                </div>
            `;
        }
        
        elements.compressionStats.innerHTML = statsHtml;
        
        // Show result card
        elements.resultCard.classList.remove('hidden');
        
        // Setup comparison
        setupComparison();
        
        let successMessage = `Compressed! ${data.stats.file_compression_ratio}x smaller`;
        if (data.stats.operations && data.stats.operations.length > 0) {
            successMessage = `Compressed with ${data.stats.operations.join(', ')}`;
        }
        showToast(successMessage, 'success');
        
    } catch (error) {
        console.error('Compression error:', error);
        showToast(error.message || 'Failed to compress image', 'error');
    } finally {
        setLoading(false);
    }
}

// ============================================================================
// Dimension & Crop Handlers
// ============================================================================

/**
 * Initialize collapsible sections
 */
function initCollapsibles() {
    // Dimension toggle
    if (elements.dimensionToggle) {
        elements.dimensionToggle.addEventListener('click', () => {
            elements.dimensionToggle.classList.toggle('active');
            elements.dimensionContent.classList.toggle('active');
        });
    }
    
    // File size toggle
    if (elements.filesizeToggle) {
        elements.filesizeToggle.addEventListener('click', () => {
            elements.filesizeToggle.classList.toggle('active');
            elements.filesizeContent.classList.toggle('active');
        });
    }
}

/**
 * Update target dimension from width (maintaining aspect ratio)
 */
function updateHeightFromWidth(width) {
    if (state.settings.lockAspectRatio && state.originalDimensions.width > 0) {
        const aspectRatio = state.originalDimensions.height / state.originalDimensions.width;
        const newHeight = Math.round(width * aspectRatio);
        elements.targetHeight.value = newHeight;
        state.settings.targetHeight = newHeight;
    }
}

/**
 * Update target dimension from height (maintaining aspect ratio)
 */
function updateWidthFromHeight(height) {
    if (state.settings.lockAspectRatio && state.originalDimensions.height > 0) {
        const aspectRatio = state.originalDimensions.width / state.originalDimensions.height;
        const newWidth = Math.round(height * aspectRatio);
        elements.targetWidth.value = newWidth;
        state.settings.targetWidth = newWidth;
    }
}

/**
 * Initialize dimension settings handlers
 */
function initDimensionSettings() {
    // Dimension preset dropdown
    if (elements.dimensionPreset) {
        elements.dimensionPreset.addEventListener('change', (e) => {
            const value = e.target.value;
            
            if (value === 'custom') {
                elements.customDimensions.classList.remove('hidden');
                elements.resizeModeSection.classList.remove('hidden');
            } else if (value === '') {
                // Original size
                elements.customDimensions.classList.add('hidden');
                state.settings.targetWidth = null;
                state.settings.targetHeight = null;
                elements.targetWidth.value = '';
                elements.targetHeight.value = '';
            } else {
                const [width, height] = value.split(',').map(Number);
                state.settings.targetWidth = width;
                state.settings.targetHeight = height;
                elements.targetWidth.value = width;
                elements.targetHeight.value = height;
                elements.customDimensions.classList.remove('hidden');
                elements.resizeModeSection.classList.remove('hidden');
            }
        });
    }
    
    // Custom dimension inputs
    if (elements.targetWidth) {
        elements.targetWidth.addEventListener('input', (e) => {
            const width = parseInt(e.target.value) || null;
            state.settings.targetWidth = width;
            if (width) {
                updateHeightFromWidth(width);
            }
            // Set preset to custom
            elements.dimensionPreset.value = 'custom';
        });
    }
    
    if (elements.targetHeight) {
        elements.targetHeight.addEventListener('input', (e) => {
            const height = parseInt(e.target.value) || null;
            state.settings.targetHeight = height;
            if (height) {
                updateWidthFromHeight(height);
            }
            // Set preset to custom
            elements.dimensionPreset.value = 'custom';
        });
    }
    
    // Lock aspect ratio button
    if (elements.lockAspectBtn) {
        elements.lockAspectBtn.addEventListener('click', () => {
            state.settings.lockAspectRatio = !state.settings.lockAspectRatio;
            elements.lockAspectBtn.classList.toggle('active', state.settings.lockAspectRatio);
            elements.lockAspectBtn.innerHTML = state.settings.lockAspectRatio 
                ? '<i class="fas fa-lock"></i>' 
                : '<i class="fas fa-lock-open"></i>';
        });
        // Set initial state
        elements.lockAspectBtn.classList.add('active');
    }
    
    // Resize mode buttons
    if (elements.modeBtns) {
        elements.modeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                elements.modeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.settings.resizeMode = btn.dataset.mode;
                
                // If crop mode selected, open crop modal
                if (btn.dataset.mode === 'crop' && state.originalPreviewUrl) {
                    openCropModal();
                }
            });
        });
    }
}

/**
 * Initialize file size settings handlers
 */
function initFileSizeSettings() {
    // Enable target size checkbox
    if (elements.enableTargetSize) {
        elements.enableTargetSize.addEventListener('change', (e) => {
            state.settings.enableTargetSize = e.target.checked;
            if (elements.filesizeControls) {
                elements.filesizeControls.classList.toggle('hidden', !e.target.checked);
            }
        });
    }
    
    // Target size input
    if (elements.targetSizeKb) {
        elements.targetSizeKb.addEventListener('input', (e) => {
            const size = parseInt(e.target.value) || null;
            state.settings.targetSizeKb = size;
            
            // Deselect preset buttons
            elements.sizePresetBtns.forEach(b => b.classList.remove('active'));
        });
    }
    
    // Size preset buttons
    if (elements.sizePresetBtns) {
        elements.sizePresetBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                elements.sizePresetBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const size = parseInt(btn.dataset.size);
                state.settings.targetSizeKb = size;
                elements.targetSizeKb.value = size;
            });
        });
    }
}

/**
 * Open the crop modal
 */
function openCropModal() {
    if (!state.originalPreviewUrl) {
        showToast('Please upload an image first', 'error');
        return;
    }
    
    elements.cropModal.classList.remove('hidden');
    elements.cropImage.src = state.originalPreviewUrl;
    
    // Initialize Cropper.js after image loads
    elements.cropImage.onload = () => {
        initCropper();
    };
}

/**
 * Close the crop modal
 */
function closeCropModal() {
    elements.cropModal.classList.add('hidden');
    
    if (state.cropper) {
        state.cropper.destroy();
        state.cropper = null;
    }
}

/**
 * Initialize Cropper.js
 */
function initCropper() {
    if (state.cropper) {
        state.cropper.destroy();
    }
    
    // Determine initial aspect ratio and crop box from target dimensions
    let aspectRatio = NaN; // Free aspect ratio by default
    let initialCropBox = null;
    
    if (state.settings.targetWidth && state.settings.targetHeight) {
        aspectRatio = state.settings.targetWidth / state.settings.targetHeight;
        elements.cropAspectInfo.textContent = `Target: ${state.settings.targetWidth}×${state.settings.targetHeight}`;
        
        // Calculate crop box that matches target dimensions within the image
        const imgWidth = state.originalDimensions.width;
        const imgHeight = state.originalDimensions.height;
        const targetW = state.settings.targetWidth;
        const targetH = state.settings.targetHeight;
        
        // Scale crop box to fit within image while maintaining aspect ratio
        let cropWidth, cropHeight;
        if (targetW <= imgWidth && targetH <= imgHeight) {
            // Target fits within image, use exact dimensions
            cropWidth = targetW;
            cropHeight = targetH;
        } else {
            // Scale down to fit
            const scaleW = imgWidth / targetW;
            const scaleH = imgHeight / targetH;
            const scale = Math.min(scaleW, scaleH);
            cropWidth = Math.floor(targetW * scale);
            cropHeight = Math.floor(targetH * scale);
        }
        
        // Center the crop box
        initialCropBox = {
            width: cropWidth,
            height: cropHeight,
            x: Math.floor((imgWidth - cropWidth) / 2),
            y: Math.floor((imgHeight - cropHeight) / 2)
        };
    }
    
    state.cropper = new Cropper(elements.cropImage, {
        aspectRatio: aspectRatio,
        viewMode: 1,
        dragMode: 'crop',
        autoCropArea: 0.8,
        responsive: true,
        restore: true,
        guides: true,
        center: true,
        highlight: true,
        cropBoxMovable: true,
        cropBoxResizable: true,
        toggleDragModeOnDblclick: true,
        ready() {
            // Set initial crop box based on target dimensions
            if (initialCropBox && state.cropper) {
                state.cropper.setData({
                    x: initialCropBox.x,
                    y: initialCropBox.y,
                    width: initialCropBox.width,
                    height: initialCropBox.height
                });
            }
        },
        crop(event) {
            const width = Math.round(event.detail.width);
            const height = Math.round(event.detail.height);
            elements.cropDimensions.textContent = `Selection: ${width} × ${height}`;
        }
    });
}

/**
 * Initialize crop modal handlers
 */
function initCropModal() {
    // Close modal buttons
    if (elements.closeCropModal) {
        elements.closeCropModal.addEventListener('click', closeCropModal);
    }
    
    if (elements.cancelCropBtn) {
        elements.cancelCropBtn.addEventListener('click', () => {
            closeCropModal();
            // Reset to fit mode
            elements.modeBtns.forEach(b => b.classList.remove('active'));
            document.querySelector('.mode-btn[data-mode="fit"]')?.classList.add('active');
            state.settings.resizeMode = 'fit';
        });
    }
    
    // Apply crop button
    if (elements.applyCropBtn) {
        elements.applyCropBtn.addEventListener('click', () => {
            if (state.cropper) {
                const cropData = state.cropper.getData();
                state.settings.cropData = {
                    x: Math.round(cropData.x),
                    y: Math.round(cropData.y),
                    width: Math.round(cropData.width),
                    height: Math.round(cropData.height)
                };
                
                // Update target dimensions to crop dimensions
                state.settings.targetWidth = state.settings.cropData.width;
                state.settings.targetHeight = state.settings.cropData.height;
                elements.targetWidth.value = state.settings.cropData.width;
                elements.targetHeight.value = state.settings.cropData.height;
                
                showToast('Crop area applied!', 'success');
            }
            closeCropModal();
        });
    }
    
    // Aspect ratio buttons
    if (elements.aspectBtns) {
        elements.aspectBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                elements.aspectBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const aspect = btn.dataset.aspect;
                let ratio = NaN;
                
                if (aspect === 'free') {
                    ratio = NaN;
                } else if (aspect === 'custom') {
                    if (state.settings.targetWidth && state.settings.targetHeight) {
                        ratio = state.settings.targetWidth / state.settings.targetHeight;
                    }
                } else {
                    const [w, h] = aspect.split(':').map(Number);
                    ratio = w / h;
                }
                
                if (state.cropper) {
                    state.cropper.setAspectRatio(ratio);
                }
            });
        });
    }
    
    // Rotate buttons
    if (elements.rotateLeftBtn) {
        elements.rotateLeftBtn.addEventListener('click', () => {
            if (state.cropper) state.cropper.rotate(-90);
        });
    }
    
    if (elements.rotateRightBtn) {
        elements.rotateRightBtn.addEventListener('click', () => {
            if (state.cropper) state.cropper.rotate(90);
        });
    }
    
    // Flip buttons
    if (elements.flipHBtn) {
        elements.flipHBtn.addEventListener('click', () => {
            if (state.cropper) {
                const scaleX = state.cropper.getData().scaleX || 1;
                state.cropper.scaleX(-scaleX);
            }
        });
    }
    
    if (elements.flipVBtn) {
        elements.flipVBtn.addEventListener('click', () => {
            if (state.cropper) {
                const scaleY = state.cropper.getData().scaleY || 1;
                state.cropper.scaleY(-scaleY);
            }
        });
    }
    
    // Reset crop button
    if (elements.resetCropBtn) {
        elements.resetCropBtn.addEventListener('click', () => {
            if (state.cropper) state.cropper.reset();
        });
    }
    
    // Close modal on overlay click
    if (elements.cropModal) {
        elements.cropModal.addEventListener('click', (e) => {
            if (e.target === elements.cropModal) {
                closeCropModal();
            }
        });
    }
}

// ============================================================================
// Comparison Slider
// ============================================================================

/**
 * Setup the before/after comparison slider
 * The overlay (left side) shows the ORIGINAL, the background (right side) shows COMPRESSED
 * This creates the standard "before | after" comparison where sliding reveals the change
 */
function setupComparison() {
    if (!state.originalPreviewUrl || !state.compressedPreviewUrl) return;
    
    // Background (revealed on right) = Compressed (After)
    // Overlay (visible on left) = Original (Before)
    elements.compareOriginal.src = state.compressedPreviewUrl;
    elements.compareCompressed.src = state.originalPreviewUrl;
    elements.comparisonSection.classList.remove('hidden');
    
    // Initialize slider position
    updateComparisonSlider(50);
}

/**
 * Update comparison slider position
 */
function updateComparisonSlider(percentage) {
    elements.comparisonOverlay.style.width = `${percentage}%`;
    elements.comparisonSlider.style.left = `${percentage}%`;
}

/**
 * Handle comparison slider drag
 */
function initComparisonSlider() {
    const container = document.getElementById('comparisonContainer');
    if (!container) return;
    
    let isDragging = false;
    
    function handleDrag(e) {
        if (!isDragging) return;
        
        const rect = container.querySelector('.comparison-wrapper').getBoundingClientRect();
        let x = (e.clientX || e.touches[0].clientX) - rect.left;
        let percentage = (x / rect.width) * 100;
        percentage = Math.max(0, Math.min(100, percentage));
        
        updateComparisonSlider(percentage);
    }
    
    container.addEventListener('mousedown', (e) => {
        isDragging = true;
        handleDrag(e);
    });
    
    container.addEventListener('touchstart', (e) => {
        isDragging = true;
        handleDrag(e);
    });
    
    document.addEventListener('mousemove', handleDrag);
    document.addEventListener('touchmove', handleDrag);
    
    document.addEventListener('mouseup', () => isDragging = false);
    document.addEventListener('touchend', () => isDragging = false);
}

// ============================================================================
// Event Listeners
// ============================================================================

/**
 * Initialize all event listeners
 */
function initEventListeners() {
    // Upload area click
    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });
    
    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });
    
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        handleFileSelect(e.dataTransfer.files[0]);
    });
    
    // Preset buttons
    elements.presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            elements.presetButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update settings
            const colors = parseInt(btn.dataset.colors);
            const iters = parseInt(btn.dataset.iters);
            
            state.settings.nColors = colors;
            state.settings.maxIters = iters;
            
            // Update sliders
            elements.nColorsSlider.value = colors;
            elements.nColorsValue.textContent = colors;
            elements.maxItersSlider.value = iters;
            elements.maxItersValue.textContent = iters;
        });
    });
    
    // Color slider
    elements.nColorsSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        state.settings.nColors = value;
        elements.nColorsValue.textContent = value;
        
        // Deselect preset buttons
        elements.presetButtons.forEach(b => b.classList.remove('active'));
    });
    
    // Iterations slider
    elements.maxItersSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        state.settings.maxIters = value;
        elements.maxItersValue.textContent = value;
        
        // Deselect preset buttons
        elements.presetButtons.forEach(b => b.classList.remove('active'));
    });
    
    // Compress button
    elements.compressBtn.addEventListener('click', compressImage);
    
    // Reset button
    elements.resetBtn.addEventListener('click', () => {
        state.currentImageId = null;
        state.compressedImageId = null;
        state.originalPreviewUrl = null;
        state.compressedPreviewUrl = null;
        state.downloadUrl = null;
        state.originalDimensions = { width: 0, height: 0 };
        state.settings.targetWidth = null;
        state.settings.targetHeight = null;
        state.settings.resizeMode = 'fit';
        state.settings.targetSizeKb = null;
        state.settings.enableTargetSize = false;
        state.settings.cropData = null;
        
        elements.fileInput.value = '';
        elements.settingsSection.classList.add('hidden');
        elements.uploadSection.classList.remove('hidden');
        elements.resultCard.classList.add('hidden');
        elements.comparisonSection.classList.add('hidden');
        
        // Reset dimension settings
        if (elements.dimensionPreset) elements.dimensionPreset.value = '';
        if (elements.customDimensions) elements.customDimensions.classList.add('hidden');
        if (elements.targetWidth) elements.targetWidth.value = '';
        if (elements.targetHeight) elements.targetHeight.value = '';
        elements.modeBtns?.forEach(b => b.classList.remove('active'));
        document.querySelector('.mode-btn[data-mode="fit"]')?.classList.add('active');
        
        // Reset file size settings
        if (elements.enableTargetSize) elements.enableTargetSize.checked = false;
        if (elements.filesizeControls) elements.filesizeControls.classList.add('hidden');
        if (elements.targetSizeKb) elements.targetSizeKb.value = 500;
        elements.sizePresetBtns?.forEach(b => b.classList.remove('active'));
        document.querySelector('.size-preset-btn[data-size="500"]')?.classList.add('active');
        
        // Collapse advanced sections
        elements.dimensionToggle?.classList.remove('active');
        elements.dimensionContent?.classList.remove('active');
        elements.filesizeToggle?.classList.remove('active');
        elements.filesizeContent?.classList.remove('active');
        
        // Reset to balanced preset
        elements.presetButtons.forEach(b => b.classList.remove('active'));
        document.querySelector('.preset-btn[data-colors="16"]').classList.add('active');
        state.settings.nColors = 16;
        state.settings.maxIters = 10;
        elements.nColorsSlider.value = 16;
        elements.nColorsValue.textContent = 16;
        elements.maxItersSlider.value = 10;
        elements.maxItersValue.textContent = 10;
    });
    
    // Download button
    elements.downloadBtn.addEventListener('click', () => {
        if (state.downloadUrl) {
            window.location.href = state.downloadUrl;
        }
    });
    
    // Initialize comparison slider
    initComparisonSlider();
}

// ============================================================================
// Initialize Application
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    initCollapsibles();
    initDimensionSettings();
    initFileSizeSettings();
    initCropModal();
    console.log('K-Means Image Compressor initialized');
});
