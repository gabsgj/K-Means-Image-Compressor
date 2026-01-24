"""
Unit Tests for K-Means Image Compressor
=======================================
Run with: pytest tests.py -v
"""

import pytest
import numpy as np
import os
import tempfile
from PIL import Image
from io import BytesIO

from kmeans_compressor import (
    KMeansCompressor, 
    InitMethod, 
    compress_image,
    CompressionResult
)
from app import app


# ============================================================================
# K-Means Algorithm Tests
# ============================================================================

class TestKMeansCompressor:
    """Tests for the K-Means compression algorithm."""
    
    def test_init_default_params(self):
        """Test default initialization parameters."""
        compressor = KMeansCompressor()
        assert compressor.n_colors == 16
        assert compressor.max_iters == 10
        assert compressor.init_method == InitMethod.KMEANS_PLUS_PLUS
    
    def test_init_custom_params(self):
        """Test custom initialization parameters."""
        compressor = KMeansCompressor(
            n_colors=32,
            max_iters=20,
            init_method=InitMethod.RANDOM,
            random_state=42
        )
        assert compressor.n_colors == 32
        assert compressor.max_iters == 20
        assert compressor.init_method == InitMethod.RANDOM
    
    def test_find_closest_centroids(self):
        """Test centroid assignment."""
        compressor = KMeansCompressor()
        
        # Simple 2D test case
        X = np.array([
            [0, 0],
            [1, 1],
            [10, 10],
            [11, 11]
        ])
        centroids = np.array([
            [0.5, 0.5],
            [10.5, 10.5]
        ])
        
        idx = compressor._find_closest_centroids(X, centroids)
        
        assert idx[0] == 0  # (0,0) closer to (0.5, 0.5)
        assert idx[1] == 0  # (1,1) closer to (0.5, 0.5)
        assert idx[2] == 1  # (10,10) closer to (10.5, 10.5)
        assert idx[3] == 1  # (11,11) closer to (10.5, 10.5)
    
    def test_compute_centroids(self):
        """Test centroid computation."""
        compressor = KMeansCompressor()
        
        X = np.array([
            [0, 0],
            [2, 2],
            [10, 10],
            [12, 12]
        ])
        idx = np.array([0, 0, 1, 1])
        K = 2
        
        centroids = compressor._compute_centroids(X, idx, K)
        
        np.testing.assert_array_almost_equal(centroids[0], [1, 1])
        np.testing.assert_array_almost_equal(centroids[1], [11, 11])
    
    def test_compress_simple_image(self):
        """Test compression on a simple synthetic image."""
        compressor = KMeansCompressor(n_colors=4, max_iters=10, random_state=42)
        
        # Create a simple 10x10 image with 4 distinct colors
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[0:5, 0:5] = [255, 0, 0]    # Red
        image[0:5, 5:10] = [0, 255, 0]   # Green
        image[5:10, 0:5] = [0, 0, 255]   # Blue
        image[5:10, 5:10] = [255, 255, 0] # Yellow
        
        result = compressor.compress(image)
        
        assert isinstance(result, CompressionResult)
        assert result.compressed_image.shape == image.shape
        assert result.reduced_colors == 4
        assert result.compression_ratio > 1
        assert result.processing_time > 0
    
    def test_compress_preserves_dimensions(self):
        """Test that compression preserves image dimensions."""
        compressor = KMeansCompressor(n_colors=8)
        
        # Test various image sizes
        for height, width in [(32, 32), (64, 48), (100, 200)]:
            image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            result = compressor.compress(image)
            assert result.compressed_image.shape == (height, width, 3)
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        compressor = KMeansCompressor(n_colors=16)
        
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = compressor.compress(image)
        
        # For K=16 colors: log2(16) = 4 bits per pixel
        # Original: 24 bits per pixel
        # Theoretical ratio should be close to 24/4 = 6
        assert 4 < result.compression_ratio < 8


class TestConvenienceFunction:
    """Tests for the compress_image convenience function."""
    
    def test_compress_image_function(self):
        """Test the convenience compression function."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        compressed, metadata = compress_image(image, n_colors=16, max_iters=5)
        
        assert compressed.shape == image.shape
        assert 'compression_ratio' in metadata
        assert 'processing_time' in metadata
        assert metadata['reduced_colors'] == 16


# ============================================================================
# Flask API Tests
# ============================================================================

class TestFlaskAPI:
    """Tests for Flask API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    def test_get_presets(self, client):
        """Test presets endpoint."""
        response = client.get('/api/presets')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert len(data['presets']) > 0
        
        # Check preset structure
        preset = data['presets'][0]
        assert 'name' in preset
        assert 'n_colors' in preset
        assert 'max_iters' in preset
    
    def test_upload_no_file(self, client):
        """Test upload without file."""
        response = client.post('/api/upload')
        assert response.status_code == 400
        
        data = response.get_json()
        assert data['success'] is False
    
    def test_upload_valid_image(self, client):
        """Test uploading a valid image."""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        response = client.post(
            '/api/upload',
            data={'image': (buffer, 'test.png')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'image_id' in data
        assert 'preview_url' in data
    
    def test_compress_invalid_image_id(self, client):
        """Test compression with invalid image ID."""
        response = client.post(
            '/api/compress',
            json={'image_id': 'nonexistent.jpg', 'n_colors': 16}
        )
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['success'] is False
    
    def test_full_workflow(self, client):
        """Test complete upload -> compress -> download workflow."""
        # 1. Upload
        img = Image.new('RGB', (50, 50), color='blue')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        upload_response = client.post(
            '/api/upload',
            data={'image': (buffer, 'test.png')},
            content_type='multipart/form-data'
        )
        
        assert upload_response.status_code == 200
        upload_data = upload_response.get_json()
        image_id = upload_data['image_id']
        
        # 2. Compress
        compress_response = client.post(
            '/api/compress',
            json={
                'image_id': image_id,
                'n_colors': 8,
                'max_iters': 5
            }
        )
        
        assert compress_response.status_code == 200
        compress_data = compress_response.get_json()
        assert compress_data['success'] is True
        assert 'stats' in compress_data
        
        # 3. Download
        compressed_id = compress_data['compressed_id']
        download_response = client.get(f'/api/download/{compressed_id}')
        
        assert download_response.status_code == 200


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_large_image_compression(self):
        """Test compression of larger images."""
        compressor = KMeansCompressor(n_colors=32, max_iters=5)
        
        # Create a 512x512 random image
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        result = compressor.compress(image)
        
        assert result.compressed_image.shape == image.shape
        assert result.processing_time < 30  # Should complete in reasonable time
    
    def test_grayscale_image_handling(self):
        """Test that grayscale images are handled properly."""
        compressor = KMeansCompressor(n_colors=8)
        
        # Create a grayscale-like RGB image
        gray_value = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        image = np.stack([gray_value] * 3, axis=-1)
        
        result = compressor.compress(image)
        
        assert result.compressed_image.shape == image.shape
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with fixed seed."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        compressor1 = KMeansCompressor(n_colors=8, random_state=42)
        result1 = compressor1.compress(image)
        
        compressor2 = KMeansCompressor(n_colors=8, random_state=42)
        result2 = compressor2.compress(image)
        
        np.testing.assert_array_equal(
            result1.compressed_image,
            result2.compressed_image
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
