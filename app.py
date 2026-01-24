"""
Flask Image Compression Application
====================================
A modern web application for compressing images using K-Means clustering.

Features:
- Drag-and-drop image upload
- Adjustable compression settings
- Real-time preview
- Before/After comparison
- Download compressed images
- RESTful API endpoints

Author: Your Name
License: MIT
"""

import os
import io
import uuid
import logging
from datetime import datetime
from functools import wraps
from typing import Optional

from flask import (
    Flask, 
    render_template, 
    request, 
    jsonify, 
    send_file, 
    url_for,
    redirect
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from kmeans_compressor import KMeansCompressor, InitMethod, compress_image, ImageProcessor, ResizeMode
from config import config, get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration based on environment
env = os.environ.get('FLASK_ENV', 'development')
if env == 'production':
    app.config.from_object(config['production'])
elif env == 'testing':
    app.config.from_object(config['testing'])
else:
    app.config.from_object(config['development'])

CORS(app)

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['COMPRESSED_FOLDER'], exist_ok=True)


# ============================================================================
# Utility Functions
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename to prevent collisions."""
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'png'
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{unique_id}.{ext}"


def cleanup_old_files(folder: str, max_age_hours: int = 1):
    """Remove files older than max_age_hours."""
    try:
        now = datetime.now()
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                file_age = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_hours = (now - file_age).total_seconds() / 3600
                if age_hours > max_age_hours:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def handle_errors(f):
    """Decorator for consistent error handling in API routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    return decorated_function


# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the about page with algorithm explanation."""
    return render_template('about.html')


@app.route('/api-docs')
def api_docs():
    """Render the API documentation page."""
    return render_template('api_docs.html')


# ============================================================================
# API Routes
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancers."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': app.config.get('VERSION', '1.0.0')
    })


@app.route('/api/upload', methods=['POST'])
@handle_errors
def upload_image():
    """
    Upload an image for compression.
    
    Accepts multipart/form-data with an 'image' file field.
    Returns the uploaded image ID and preview URL.
    """
    # Cleanup old files periodically
    cleanup_old_files(app.config['UPLOAD_FOLDER'])
    cleanup_old_files(app.config['COMPRESSED_FOLDER'])
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'File type not allowed. Supported types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({
            'success': False,
            'error': f'File too large. Maximum size: {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
        }), 400
    
    # Generate unique filename and save
    filename = generate_unique_filename(secure_filename(file.filename))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Get image dimensions
    with Image.open(filepath) as img:
        width, height = img.size
        format_type = img.format
    
    logger.info(f"Uploaded image: {filename} ({width}x{height})")
    
    return jsonify({
        'success': True,
        'image_id': filename,
        'preview_url': url_for('get_original_image', image_id=filename, _external=True),
        'dimensions': {
            'width': width,
            'height': height
        },
        'format': format_type,
        'file_size': file_size
    })


@app.route('/api/compress', methods=['POST'])
@handle_errors
def compress_image_api():
    """
    Compress an uploaded image using K-Means clustering.
    
    Request JSON body:
        - image_id: The ID of the uploaded image
        - n_colors: Number of colors (2-256, default: 16)
        - max_iters: Maximum iterations (1-100, default: 10)
        - quality: Output quality for JPEG (1-100, default: 95)
    
    Returns compression results and download URL.
    """
    data = request.get_json()
    
    if not data or 'image_id' not in data:
        return jsonify({
            'success': False,
            'error': 'image_id is required'
        }), 400
    
    image_id = data['image_id']
    n_colors = min(max(int(data.get('n_colors', 16)), 2), 256)
    max_iters = min(max(int(data.get('max_iters', 10)), 1), 100)
    quality = min(max(int(data.get('quality', 95)), 1), 100)
    
    # Load original image
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
    
    if not os.path.exists(original_path):
        return jsonify({
            'success': False,
            'error': 'Image not found. Please upload again.'
        }), 404
    
    # Open and convert image to numpy array
    with Image.open(original_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original_array = np.array(img)
    
    logger.info(f"Compressing image {image_id} with {n_colors} colors, {max_iters} iterations")
    
    # Compress the image
    compressor = KMeansCompressor(
        n_colors=n_colors,
        max_iters=max_iters,
        init_method=InitMethod.KMEANS_PLUS_PLUS
    )
    
    result = compressor.compress(original_array)
    
    # Save compressed image (strip metadata for smaller file)
    compressed_filename = f"compressed_{image_id}"
    compressed_path = os.path.join(app.config['COMPRESSED_FOLDER'], compressed_filename)
    
    compressed_img = Image.fromarray(result.compressed_image.astype(np.uint8))
    
    # Determine output format - save without metadata
    ext = image_id.rsplit('.', 1)[1].lower() if '.' in image_id else 'png'
    if ext in ['jpg', 'jpeg']:
        compressed_img.save(compressed_path, 'JPEG', quality=quality, 
                           optimize=True, progressive=True,
                           subsampling=2 if quality < 70 else 0)
    else:
        compressed_img.save(compressed_path, 'PNG', optimize=True)
    
    # Get file sizes for comparison
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    actual_compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
    
    logger.info(f"Compression complete: {original_size} -> {compressed_size} bytes")
    
    return jsonify({
        'success': True,
        'compressed_id': compressed_filename,
        'download_url': url_for('download_compressed', image_id=compressed_filename, _external=True),
        'preview_url': url_for('get_compressed_image', image_id=compressed_filename, _external=True),
        'stats': {
            'original_file_size': original_size,
            'compressed_file_size': compressed_size,
            'file_compression_ratio': round(actual_compression_ratio, 2),
            'theoretical_compression_ratio': round(result.compression_ratio, 2),
            'original_colors': result.original_colors,
            'reduced_colors': result.reduced_colors,
            'iterations': result.iterations,
            'processing_time_ms': round(result.processing_time * 1000, 2)
        }
    })


@app.route('/api/compress-direct', methods=['POST'])
@handle_errors
def compress_direct():
    """
    Upload and compress an image in a single request.
    
    Accepts multipart/form-data with:
        - image: The image file
        - n_colors: Number of colors (optional, default: 16)
        - max_iters: Maximum iterations (optional, default: 10)
    
    Returns the compressed image directly.
    """
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    n_colors = int(request.form.get('n_colors', 16))
    max_iters = int(request.form.get('max_iters', 10))
    
    # Load image directly into memory
    img = Image.open(file.stream)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    original_array = np.array(img)
    
    # Compress
    compressed_array, metadata = compress_image(
        original_array,
        n_colors=n_colors,
        max_iters=max_iters
    )
    
    # Convert back to image
    compressed_img = Image.fromarray(compressed_array.astype(np.uint8))
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    compressed_img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='image/png',
        as_attachment=True,
        download_name=f'compressed_{n_colors}colors.png'
    )


@app.route('/api/original/<image_id>')
@handle_errors
def get_original_image(image_id):
    """Serve the original uploaded image."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_id))
    
    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': 'Image not found'
        }), 404
    
    return send_file(filepath)


@app.route('/api/compressed/<image_id>')
@handle_errors
def get_compressed_image(image_id):
    """Serve the compressed image."""
    filepath = os.path.join(app.config['COMPRESSED_FOLDER'], secure_filename(image_id))
    
    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': 'Compressed image not found'
        }), 404
    
    return send_file(filepath)


@app.route('/api/download/<image_id>')
@handle_errors
def download_compressed(image_id):
    """Download the compressed image as attachment."""
    filepath = os.path.join(app.config['COMPRESSED_FOLDER'], secure_filename(image_id))
    
    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': 'Compressed image not found'
        }), 404
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=f'compressed_{image_id}'
    )


@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Get predefined compression presets."""
    return jsonify({
        'success': True,
        'presets': [
            {
                'name': 'Maximum Compression',
                'description': 'Smallest file size, reduced quality',
                'n_colors': 4,
                'max_iters': 15
            },
            {
                'name': 'High Compression',
                'description': 'Small file size with acceptable quality',
                'n_colors': 8,
                'max_iters': 12
            },
            {
                'name': 'Balanced',
                'description': 'Good balance of size and quality',
                'n_colors': 16,
                'max_iters': 10
            },
            {
                'name': 'High Quality',
                'description': 'Better quality with moderate compression',
                'n_colors': 32,
                'max_iters': 10
            },
            {
                'name': 'Maximum Quality',
                'description': 'Best quality, larger file size',
                'n_colors': 64,
                'max_iters': 8
            }
        ]
    })


@app.route('/api/compress-advanced', methods=['POST'])
@handle_errors
def compress_advanced():
    """
    Advanced compression with resize, crop, and file size targeting.
    
    Request JSON body:
        - image_id: The ID of the uploaded image
        - n_colors: Number of colors (2-256, default: 16)
        - max_iters: Maximum iterations (1-100, default: 10)
        - target_width: Target width in pixels (optional)
        - target_height: Target height in pixels (optional)
        - resize_mode: 'fit', 'fill', 'stretch', or 'crop' (default: 'fit')
        - target_size_kb: Target file size in KB (optional)
        - crop_data: {x, y, width, height} for custom crop (optional)
        - quality: Output quality for JPEG (1-100, default: 95)
    
    Returns compression results and download URL.
    """
    data = request.get_json()
    
    if not data or 'image_id' not in data:
        return jsonify({
            'success': False,
            'error': 'image_id is required'
        }), 400
    
    image_id = data['image_id']
    n_colors = min(max(int(data.get('n_colors', 16)), 2), 256)
    max_iters = min(max(int(data.get('max_iters', 10)), 1), 100)
    quality = min(max(int(data.get('quality', 95)), 1), 100)
    
    # Dimension settings
    target_width = data.get('target_width')
    target_height = data.get('target_height')
    resize_mode_str = data.get('resize_mode', 'fit').lower()
    target_size_kb = data.get('target_size_kb')
    crop_data = data.get('crop_data')
    
    # Map resize mode string to enum
    resize_mode_map = {
        'fit': ResizeMode.FIT,
        'fill': ResizeMode.FILL,
        'stretch': ResizeMode.STRETCH,
        'crop': ResizeMode.CROP
    }
    resize_mode = resize_mode_map.get(resize_mode_str, ResizeMode.FIT)
    
    # Load original image
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
    
    if not os.path.exists(original_path):
        return jsonify({
            'success': False,
            'error': 'Image not found. Please upload again.'
        }), 404
    
    # Open and convert image to numpy array
    with Image.open(original_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original_array = np.array(img)
    
    original_dimensions = (original_array.shape[1], original_array.shape[0])
    
    logger.info(f"Advanced compression: {image_id}, dimensions: {target_width}x{target_height}, mode: {resize_mode_str}")
    
    # Process image
    processed_array, process_metadata = ImageProcessor.process_image(
        original_array,
        target_width=int(target_width) if target_width else None,
        target_height=int(target_height) if target_height else None,
        resize_mode=resize_mode,
        n_colors=n_colors,
        max_iters=max_iters,
        target_size_kb=int(target_size_kb) if target_size_kb else None,
        crop_data=crop_data
    )
    
    # Save compressed image
    compressed_filename = f"compressed_{image_id}"
    compressed_path = os.path.join(app.config['COMPRESSED_FOLDER'], compressed_filename)
    
    compressed_img = Image.fromarray(processed_array.astype(np.uint8))
    
    # Determine output format and save (strip all metadata for smaller size)
    # Use quality from target size optimization if available, otherwise use request quality
    save_quality = process_metadata.get('quality_used', quality)
    
    ext = image_id.rsplit('.', 1)[1].lower() if '.' in image_id else 'png'
    if ext in ['jpg', 'jpeg']:
        # Save without EXIF/metadata, with optimization
        compressed_img.save(compressed_path, 'JPEG', quality=save_quality, 
                           optimize=True, progressive=True, 
                           subsampling=2 if save_quality < 70 else 0)
    else:
        compressed_img.save(compressed_path, 'PNG', optimize=True)
    
    # Get file sizes
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    
    return jsonify({
        'success': True,
        'compressed_id': compressed_filename,
        'download_url': url_for('download_compressed', image_id=compressed_filename, _external=True),
        'preview_url': url_for('get_compressed_image', image_id=compressed_filename, _external=True),
        'stats': {
            'original_file_size': original_size,
            'compressed_file_size': compressed_size,
            'file_compression_ratio': round(original_size / compressed_size, 2) if compressed_size > 0 else 1,
            'original_dimensions': original_dimensions,
            'final_dimensions': process_metadata.get('final_dimensions'),
            'operations': process_metadata.get('operations', []),
            'colors_used': process_metadata.get('colors_used', n_colors),
            'target_size_met': process_metadata.get('target_met', True) if target_size_kb else None,
            'achieved_size_kb': process_metadata.get('achieved_size_kb')
        }
    })


@app.route('/api/resize-modes', methods=['GET'])
def get_resize_modes():
    """Get available resize modes with descriptions."""
    return jsonify({
        'success': True,
        'modes': [
            {
                'value': 'fit',
                'name': 'Fit',
                'description': 'Fit within dimensions, maintain aspect ratio. May add padding.',
                'icon': 'fa-compress-arrows-alt'
            },
            {
                'value': 'fill',
                'name': 'Fill',
                'description': 'Fill dimensions, maintain aspect ratio. May crop edges.',
                'icon': 'fa-expand-arrows-alt'
            },
            {
                'value': 'stretch',
                'name': 'Stretch',
                'description': 'Stretch to exact dimensions. May distort image.',
                'icon': 'fa-arrows-alt-h'
            },
            {
                'value': 'crop',
                'name': 'Crop',
                'description': 'Crop from center to exact dimensions.',
                'icon': 'fa-crop-alt'
            }
        ]
    })


@app.route('/api/dimension-presets', methods=['GET'])
def get_dimension_presets():
    """Get common dimension presets."""
    return jsonify({
        'success': True,
        'presets': [
            {'name': 'Original', 'width': None, 'height': None},
            {'name': 'HD (1280x720)', 'width': 1280, 'height': 720},
            {'name': 'Full HD (1920x1080)', 'width': 1920, 'height': 1080},
            {'name': 'Square (1080x1080)', 'width': 1080, 'height': 1080},
            {'name': 'Instagram Post (1080x1350)', 'width': 1080, 'height': 1350},
            {'name': 'Instagram Story (1080x1920)', 'width': 1080, 'height': 1920},
            {'name': 'Twitter Post (1200x675)', 'width': 1200, 'height': 675},
            {'name': 'Facebook Cover (820x312)', 'width': 820, 'height': 312},
            {'name': 'LinkedIn Banner (1584x396)', 'width': 1584, 'height': 396},
            {'name': 'Thumbnail (320x180)', 'width': 320, 'height': 180},
            {'name': 'Icon (256x256)', 'width': 256, 'height': 256},
            {'name': 'Favicon (32x32)', 'width': 32, 'height': 32}
        ]
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Resource not found'
        }), 404
    return render_template('404.html'), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
    return render_template('500.html'), 500


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Image Compression App on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
