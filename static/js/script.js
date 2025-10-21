// Global variables
let currentImages = [];

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Form submissions
    document.getElementById('singleImageForm').addEventListener('submit', generateSingleImage);
    document.getElementById('batchImageForm').addEventListener('submit', generateBatchImages);
    document.getElementById('downloadAllBtn').addEventListener('click', downloadAllImages);

    // Check server health
    checkServerHealth();
});

// Check server health
async function checkServerHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            showNotification('Server is running and ready!', 'success');
        } else {
            showNotification('Server is experiencing issues', 'warning');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        showNotification('Cannot connect to server', 'danger');
    }
}

// Generate single image
async function generateSingleImage(event) {
    event.preventDefault();
    
    const formData = new FormData();
    const noiseSeed = document.getElementById('noiseSeed').value;
    
    if (noiseSeed) {
        formData.append('noise_seed', noiseSeed);
    }
    
    formData.append('num_images', '1');
    
    await generateImages(formData, 'single');
}

// Generate batch images
async function generateBatchImages(event) {
    event.preventDefault();
    
    const batchSize = document.getElementById('batchSize').value;
    const formData = new FormData();
    formData.append('batch_size', batchSize);
    
    await generateImages(formData, 'batch');
}

// Main image generation function
async function generateImages(formData, type) {
    showLoading(true);
    hideError();
    hideResults();
    
    try {
        const url = type === 'batch' ? '/generate_batch' : '/generate';
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data, type);
            showNotification('Images generated successfully!', 'success');
        } else {
            showError(data.error || 'Failed to generate images');
        }
    } catch (error) {
        console.error('Generation error:', error);
        showError('Network error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Display results
function displayResults(data, type) {
    const resultsSection = document.getElementById('resultsSection');
    const imageResults = document.getElementById('imageResults');
    
    currentImages = data.images || [];
    
    if (type === 'batch') {
        // Display batch as grid
        imageResults.innerHTML = `
            <div class="text-center">
                <h5>Generated ${data.batch_size} Images</h5>
                <img src="${data.grid}" alt="Generated batch" class="img-fluid rounded shadow">
            </div>
        `;
    } else {
        // Display single image with details
        if (currentImages.length > 0) {
            imageResults.innerHTML = `
                <div class="single-image-container">
                    <h5 class="text-center mb-3">Generated Medical Image</h5>
                    <img src="${currentImages[0].data}" alt="Generated medical image" 
                         class="img-fluid rounded shadow generated-image">
                    <div class="text-center mt-3">
                        <button onclick="downloadSingleImage(${currentImages[0].id})" 
                                class="btn btn-outline-primary btn-sm">
                            <i class="fas fa-download"></i> Download This Image
                        </button>
                    </div>
                </div>
            `;
        }
    }
    
    resultsSection.classList.remove('d-none');
}

// Download single image
async function downloadSingleImage(imageId) {
    try {
        const response = await fetch(`/download_image/${imageId}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `medical_image_${imageId + 1}.png`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            showNotification('Image downloaded successfully!', 'success');
        } else {
            showError('Failed to download image');
        }
    } catch (error) {
        console.error('Download error:', error);
        showError('Download failed: ' + error.message);
    }
}

// Download all images
async function downloadAllImages() {
    if (currentImages.length === 0) {
        showError('No images to download');
        return;
    }
    
    for (let i = 0; i < currentImages.length; i++) {
        await downloadSingleImage(i);
        // Small delay to avoid overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 500));
    }
}

// UI Helper functions
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (show) {
        spinner.classList.remove('d-none');
    } else {
        spinner.classList.add('d-none');
    }
}

function hideResults() {
    document.getElementById('resultsSection').classList.add('d-none');
}

function hideError() {
    document.getElementById('errorSection').classList.add('d-none');
}

function showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorSection.classList.remove('d-none');
}

function showNotification(message, type) {
    // Create toast notification
    const toastContainer = document.createElement('div');
    toastContainer.className = `toast align-items-center text-white bg-${type} border-0`;
    toastContainer.setAttribute('role', 'alert');
    toastContainer.setAttribute('aria-live', 'assertive');
    toastContainer.setAttribute('aria-atomic', 'true');
    
    toastContainer.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toastContainer);
    
    const toast = new bootstrap.Toast(toastContainer);
    toast.show();
    
    // Remove toast after it's hidden
    toastContainer.addEventListener('hidden.bs.toast', function() {
        document.body.removeChild(toastContainer);
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+G or Cmd+G to generate single image
    if ((event.ctrlKey || event.metaKey) && event.key === 'g') {
        event.preventDefault();
        document.getElementById('generateSingle').click();
    }
});