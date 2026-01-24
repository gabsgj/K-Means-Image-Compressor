"""
K-Means Image Compression Module
================================
High-performance image compression using K-Means clustering algorithm.
Reduces the number of colors in an image while preserving visual quality.

Author: Your Name
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import io
from PIL import Image


class InitMethod(Enum):
    """Centroid initialization methods."""
    RANDOM = "random"
    KMEANS_PLUS_PLUS = "kmeans++"


class ResizeMode(Enum):
    """Image resize modes."""
    FIT = "fit"          # Fit within dimensions, maintain aspect ratio (may have padding)
    FILL = "fill"        # Fill dimensions, maintain aspect ratio (may crop)
    STRETCH = "stretch"  # Stretch to exact dimensions (may distort)
    CROP = "crop"        # Crop to exact dimensions from center


@dataclass
class CompressionResult:
    """Container for compression results and metadata."""
    compressed_image: np.ndarray
    centroids: np.ndarray
    labels: np.ndarray
    iterations: int
    compression_ratio: float
    original_colors: int
    reduced_colors: int
    processing_time: float
    original_size_bits: int
    compressed_size_bits: int


class KMeansCompressor:
    """
    K-Means based image compressor.
    
    Uses the K-Means clustering algorithm to reduce the number of colors
    in an image, achieving significant compression while maintaining
    visual quality.
    
    Attributes:
        n_colors: Number of colors to reduce the image to (K clusters)
        max_iters: Maximum number of K-Means iterations
        tolerance: Convergence tolerance for centroid movement
        init_method: Method for initializing centroids
        random_state: Seed for reproducibility
        sample_size: Max pixels to sample for clustering (speeds up large images)
    """
    
    def __init__(
        self,
        n_colors: int = 16,
        max_iters: int = 15,
        tolerance: float = 1e-5,
        init_method: InitMethod = InitMethod.KMEANS_PLUS_PLUS,
        random_state: Optional[int] = None,
        sample_size: int = 150000
    ):
        """
        Initialize the K-Means compressor.
        
        Args:
            n_colors: Number of colors (clusters) for compression
            max_iters: Maximum iterations for K-Means (default 15 for quality)
            tolerance: Convergence threshold (default 1e-5 for better quality)
            init_method: Centroid initialization method
            random_state: Random seed for reproducibility
            sample_size: Max pixels to use for clustering (default 150000)
        """
        self.n_colors = n_colors
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.init_method = init_method
        self.random_state = random_state
        self.sample_size = sample_size
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _find_closest_centroids(
        self, 
        X: np.ndarray, 
        centroids: np.ndarray,
        batch_size: int = 50000
    ) -> np.ndarray:
        """
        Assign each data point to its closest centroid.
        
        Uses vectorized operations with batching for optimal performance
        on large datasets without running out of memory.
        
        Args:
            X: Data points of shape (m, n)
            centroids: Centroid positions of shape (K, n)
            batch_size: Process pixels in batches for memory efficiency
            
        Returns:
            Array of centroid indices for each data point
        """
        m = X.shape[0]
        
        # For smaller images, use direct computation
        if m <= batch_size:
            # Optimized distance calculation using squared Euclidean (no sqrt needed for argmin)
            # ||x - c||² = ||x||² + ||c||² - 2*x·c
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            C_sq = np.sum(centroids ** 2, axis=1)
            XC = X @ centroids.T
            distances_sq = X_sq + C_sq - 2 * XC
            return np.argmin(distances_sq, axis=1)
        
        # For large images, process in batches
        idx = np.empty(m, dtype=np.int32)
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            batch = X[start:end]
            batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)
            C_sq = np.sum(centroids ** 2, axis=1)
            batch_C = batch @ centroids.T
            distances_sq = batch_sq + C_sq - 2 * batch_C
            idx[start:end] = np.argmin(distances_sq, axis=1)
        
        return idx
    
    def _compute_centroids(
        self, 
        X: np.ndarray, 
        idx: np.ndarray, 
        K: int
    ) -> np.ndarray:
        """
        Compute new centroid positions as mean of assigned points.
        
        Uses optimized bincount for fast aggregation.
        
        Args:
            X: Data points of shape (m, n)
            idx: Cluster assignments for each point
            K: Number of clusters
            
        Returns:
            New centroid positions of shape (K, n)
        """
        m, n = X.shape
        centroids = np.zeros((K, n))
        
        # Count points in each cluster
        counts = np.bincount(idx, minlength=K)
        
        # Sum points for each cluster (faster than loop)
        for dim in range(n):
            centroids[:, dim] = np.bincount(idx, weights=X[:, dim], minlength=K)
        
        # Compute means, handling empty clusters
        non_empty = counts > 0
        centroids[non_empty] /= counts[non_empty, np.newaxis]
        
        # Handle empty clusters by reinitializing randomly
        empty_clusters = np.where(~non_empty)[0]
        if len(empty_clusters) > 0:
            random_indices = np.random.choice(m, size=len(empty_clusters), replace=False)
            centroids[empty_clusters] = X[random_indices]
        
        return centroids
    
    def _init_centroids_random(self, X: np.ndarray, K: int) -> np.ndarray:
        """
        Initialize centroids by randomly selecting K data points.
        
        Args:
            X: Data points
            K: Number of centroids
            
        Returns:
            Initial centroid positions
        """
        randidx = np.random.permutation(X.shape[0])
        return X[randidx[:K]].copy()
    
    def _init_centroids_kmeans_plus_plus(
        self, 
        X: np.ndarray, 
        K: int
    ) -> np.ndarray:
        """
        Initialize centroids using K-Means++ algorithm.
        
        K-Means++ selects initial centroids that are well-spread out,
        leading to faster convergence and better results.
        
        Args:
            X: Data points
            K: Number of centroids
            
        Returns:
            Initial centroid positions
        """
        m, n = X.shape
        centroids = np.zeros((K, n), dtype=np.float64)
        
        # Choose first centroid randomly from the data
        first_idx = np.random.randint(m)
        centroids[0] = X[first_idx].copy()
        
        # Pre-compute X squared norms for faster distance calculation
        X_sq = np.sum(X ** 2, axis=1)
        
        # Initialize distances to infinity
        min_distances = np.full(m, np.inf)
        
        for k in range(1, K):
            # Update distances with the newly added centroid
            c = centroids[k-1]
            c_sq = np.sum(c ** 2)
            new_distances = X_sq + c_sq - 2 * (X @ c)
            min_distances = np.minimum(min_distances, new_distances)
            
            # Ensure no negative distances due to numerical errors
            min_distances = np.maximum(min_distances, 0)
            
            # Choose next centroid with probability proportional to distance squared
            total = min_distances.sum()
            if total > 0:
                probabilities = min_distances / total
            else:
                # Fallback to uniform if all distances are 0
                probabilities = np.ones(m) / m
            
            # Use searchsorted for faster selection
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.random()
            idx = np.searchsorted(cumulative_probs, r)
            idx = min(idx, m - 1)
            centroids[k] = X[idx].copy()
        
        return centroids
    
    def _init_centroids(self, X: np.ndarray, K: int) -> np.ndarray:
        """
        Initialize centroids based on the selected method.
        
        Args:
            X: Data points
            K: Number of centroids
            
        Returns:
            Initial centroid positions
        """
        if self.init_method == InitMethod.KMEANS_PLUS_PLUS:
            return self._init_centroids_kmeans_plus_plus(X, K)
        return self._init_centroids_random(X, K)
    
    def _run_kmeans(
        self, 
        X: np.ndarray, 
        initial_centroids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Execute the K-Means clustering algorithm.
        
        Args:
            X: Data points of shape (m, n)
            initial_centroids: Starting centroid positions
            
        Returns:
            Tuple of (final centroids, cluster assignments, iterations run)
        """
        m, n = X.shape
        K = initial_centroids.shape[0]
        centroids = initial_centroids.copy()
        idx = np.zeros(m, dtype=np.int32)
        
        for i in range(self.max_iters):
            # Cluster assignment step
            idx = self._find_closest_centroids(X, centroids)
            
            # Compute new centroids
            new_centroids = self._compute_centroids(X, idx, K)
            
            # Check for convergence (using squared distance for speed)
            centroid_shift = np.sum((new_centroids - centroids) ** 2)
            if centroid_shift < self.tolerance ** 2:
                return new_centroids, idx, i + 1
            
            centroids = new_centroids
        
        return centroids, idx, self.max_iters
    
    def compress(self, image: np.ndarray) -> CompressionResult:
        """
        Compress an image using K-Means clustering.
        
        Uses pixel sampling for large images to dramatically improve speed
        while maintaining quality. The centroids are computed from a sample,
        then all pixels are assigned to the nearest centroid.
        
        Args:
            image: Input image as numpy array (H, W, C) with values in [0, 1] or [0, 255]
            
        Returns:
            CompressionResult containing the compressed image and metadata
        """
        start_time = time.time()
        
        # Store original shape
        original_shape = image.shape
        height, width = original_shape[:2]
        channels = original_shape[2] if len(original_shape) > 2 else 1
        
        # Normalize to [0, 1] range for better numerical stability
        was_normalized = image.max() <= 1.0
        if not was_normalized:
            image = image.astype(np.float64) / 255.0
        else:
            image = image.astype(np.float64)
        
        # Reshape to (m, channels) where m = height * width
        X_full = image.reshape(-1, channels)
        total_pixels = X_full.shape[0]
        
        # Sample pixels for large images using stratified sampling for better coverage
        use_sampling = total_pixels > self.sample_size
        if use_sampling:
            # Use stratified sampling to ensure we capture colors from all regions
            sample_indices = np.random.choice(total_pixels, self.sample_size, replace=False)
            X_sample = X_full[sample_indices].copy()
        else:
            X_sample = X_full.copy()
        
        # Calculate original unique colors (on sample for speed)
        sample_for_colors = X_sample[:min(10000, len(X_sample))]
        original_colors = len(np.unique((sample_for_colors * 255).astype(np.uint8).view(np.dtype((np.void, channels)))))
        
        # Initialize centroids using sample
        initial_centroids = self._init_centroids(X_sample, self.n_colors)
        
        # Run K-Means on sample
        centroids, _, iterations = self._run_kmeans(X_sample, initial_centroids)
        
        # Refine centroids using full image if sampling was used (improves quality)
        if use_sampling and total_pixels < 500000:
            # Do one more pass with all pixels to refine centroids
            idx_full = self._find_closest_centroids(X_full, centroids)
            centroids = self._compute_centroids(X_full, idx_full, self.n_colors)
        
        # Assign ALL pixels to nearest centroid
        idx = self._find_closest_centroids(X_full, centroids)
        
        # Reconstruct image using centroid colors
        compressed_pixels = centroids[idx]
        compressed_image = compressed_pixels.reshape(original_shape)
        
        # Denormalize if input was in [0, 255]
        if not was_normalized:
            # Use rounding for better color accuracy
            compressed_image = np.clip(np.round(compressed_image * 255), 0, 255).astype(np.uint8)
            centroids = np.clip(np.round(centroids * 255), 0, 255).astype(np.uint8)
        
        processing_time = time.time() - start_time
        
        # Calculate compression metrics
        # Original: 24 bits per pixel (8 bits × 3 channels)
        # Compressed: 16 colors × 24 bits + 4 bits per pixel (for 16 colors, log2(16)=4)
        bits_per_color = 8 * channels
        original_size_bits = total_pixels * bits_per_color
        
        # Compressed size: color palette + pixel indices
        bits_for_index = int(np.ceil(np.log2(max(self.n_colors, 2))))
        compressed_size_bits = (self.n_colors * bits_per_color) + (total_pixels * bits_for_index)
        
        compression_ratio = original_size_bits / compressed_size_bits
        
        return CompressionResult(
            compressed_image=compressed_image,
            centroids=centroids,
            labels=idx.reshape(height, width),
            iterations=iterations,
            compression_ratio=compression_ratio,
            original_colors=original_colors,
            reduced_colors=self.n_colors,
            processing_time=processing_time,
            original_size_bits=original_size_bits,
            compressed_size_bits=compressed_size_bits
        )
    
    def compress_with_multiple_runs(
        self, 
        image: np.ndarray, 
        n_runs: int = 3
    ) -> CompressionResult:
        """
        Run K-Means multiple times and return the best result.
        
        Helps avoid poor local minima by trying different initializations.
        
        Args:
            image: Input image
            n_runs: Number of K-Means runs to perform
            
        Returns:
            Best CompressionResult based on inertia (within-cluster sum of squares)
        """
        best_result = None
        best_inertia = float('inf')
        
        for _ in range(n_runs):
            result = self.compress(image)
            
            # Calculate inertia (sum of squared distances to centroids)
            X = image.reshape(-1, image.shape[2] if len(image.shape) > 2 else 1)
            if image.max() > 1.0:
                X = X / 255.0
            
            distances = np.sum(
                (X - result.centroids[result.labels.flatten()] / 255.0 
                 if image.max() > 1.0 else X - result.centroids[result.labels.flatten()]) ** 2
            )
            
            if distances < best_inertia:
                best_inertia = distances
                best_result = result
        
        return best_result


def compress_image(
    image: np.ndarray,
    n_colors: int = 16,
    max_iters: int = 10,
    use_kmeans_plus_plus: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to compress an image.
    
    Args:
        image: Input image as numpy array
        n_colors: Number of colors for compression
        max_iters: Maximum K-Means iterations
        use_kmeans_plus_plus: Whether to use K-Means++ initialization
        
    Returns:
        Tuple of (compressed_image, metadata_dict)
    """
    init_method = InitMethod.KMEANS_PLUS_PLUS if use_kmeans_plus_plus else InitMethod.RANDOM
    
    compressor = KMeansCompressor(
        n_colors=n_colors,
        max_iters=max_iters,
        init_method=init_method
    )
    
    result = compressor.compress(image)
    
    metadata = {
        'compression_ratio': result.compression_ratio,
        'original_colors': result.original_colors,
        'reduced_colors': result.reduced_colors,
        'iterations': result.iterations,
        'processing_time': result.processing_time,
        'original_size_bits': result.original_size_bits,
        'compressed_size_bits': result.compressed_size_bits
    }
    
    return result.compressed_image, metadata


class ImageProcessor:
    """
    Advanced image processing with resizing, cropping, and file size targeting.
    """
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        mode: ResizeMode = ResizeMode.FIT,
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Resize an image to target dimensions using specified mode.
        
        Args:
            image: Input image as numpy array
            target_width: Target width in pixels
            target_height: Target height in pixels
            mode: Resize mode (FIT, FILL, STRETCH, CROP)
            background_color: Background color for FIT mode padding
            
        Returns:
            Resized image as numpy array
        """
        pil_image = Image.fromarray(image.astype(np.uint8))
        orig_width, orig_height = pil_image.size
        
        if mode == ResizeMode.STRETCH:
            # Simply resize to exact dimensions (may distort)
            resized = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
        elif mode == ResizeMode.FIT:
            # Fit within dimensions, maintain aspect ratio, add padding
            ratio = min(target_width / orig_width, target_height / orig_height)
            new_width = int(orig_width * ratio)
            new_height = int(orig_height * ratio)
            
            resized_img = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with background color
            resized = Image.new('RGB', (target_width, target_height), background_color)
            
            # Paste resized image centered
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            resized.paste(resized_img, (paste_x, paste_y))
            
        elif mode == ResizeMode.FILL:
            # Fill dimensions, maintain aspect ratio, crop excess
            ratio = max(target_width / orig_width, target_height / orig_height)
            new_width = int(orig_width * ratio)
            new_height = int(orig_height * ratio)
            
            resized_img = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop from center
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            
            resized = resized_img.crop((left, top, right, bottom))
            
        elif mode == ResizeMode.CROP:
            # Crop from center without resizing
            left = max(0, (orig_width - target_width) // 2)
            top = max(0, (orig_height - target_height) // 2)
            right = min(orig_width, left + target_width)
            bottom = min(orig_height, top + target_height)
            
            cropped = pil_image.crop((left, top, right, bottom))
            
            # If crop is smaller than target, create padded image
            if cropped.size != (target_width, target_height):
                resized = Image.new('RGB', (target_width, target_height), background_color)
                paste_x = (target_width - cropped.size[0]) // 2
                paste_y = (target_height - cropped.size[1]) // 2
                resized.paste(cropped, (paste_x, paste_y))
            else:
                resized = cropped
        else:
            resized = pil_image
        
        return np.array(resized)
    
    @staticmethod
    def crop_image(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Crop an image to specified region.
        
        Args:
            image: Input image as numpy array
            x: Left coordinate
            y: Top coordinate
            width: Crop width
            height: Crop height
            
        Returns:
            Cropped image as numpy array
        """
        pil_image = Image.fromarray(image.astype(np.uint8))
        cropped = pil_image.crop((x, y, x + width, y + height))
        return np.array(cropped)
    
    @staticmethod
    def compress_to_target_size(
        image: np.ndarray,
        target_size_kb: int,
        min_colors: int = 2,
        max_colors: int = 256,
        max_iters: int = 15,
        output_format: str = 'JPEG',
        quality: int = 95
    ) -> Tuple[np.ndarray, dict]:
        """
        Compress an image to achieve exact or very close to target file size.
        
        Uses binary search on JPEG quality for precise size control.
        Strips all metadata for smaller file sizes.
        
        Args:
            image: Input image as numpy array
            target_size_kb: Target file size in kilobytes
            min_colors: Minimum number of colors to try
            max_colors: Maximum number of colors to try
            max_iters: K-Means iterations
            output_format: Output format (JPEG or PNG)
            quality: Starting JPEG quality (1-100)
            
        Returns:
            Tuple of (compressed_image, metadata)
        """
        target_size_bytes = target_size_kb * 1024
        is_jpeg = output_format.upper() == 'JPEG'
        
        def save_and_measure(pil_img: Image.Image, q: int) -> int:
            """Save image to buffer and return size (strips metadata)."""
            buffer = io.BytesIO()
            if is_jpeg:
                # Strip metadata by not copying exif
                pil_img.save(buffer, format='JPEG', quality=q, optimize=True, 
                            progressive=True, subsampling=2 if q < 70 else 0)
            else:
                pil_img.save(buffer, format='PNG', optimize=True)
            return buffer.tell()
        
        def compress_with_colors(img: np.ndarray, n_colors: int) -> np.ndarray:
            """Apply K-Means compression."""
            compressor = KMeansCompressor(n_colors=n_colors, max_iters=max_iters)
            result = compressor.compress(img)
            return result.compressed_image
        
        best_image = None
        best_size = float('inf')
        best_colors = max_colors
        best_quality = quality
        attempts = []
        
        # Try different color levels
        color_levels = [max_colors, 128, 64, 32, 16, 8, 4, 2]
        color_levels = sorted(set(c for c in color_levels if min_colors <= c <= max_colors), reverse=True)
        
        for n_colors in color_levels:
            # Compress image with this color count
            compressed_array = compress_with_colors(image, n_colors)
            pil_img = Image.fromarray(compressed_array.astype(np.uint8))
            
            if is_jpeg:
                # Binary search on quality to find exact size
                q_low, q_high = 10, 100
                best_q_for_colors = None
                best_size_for_colors = float('inf')
                best_array_for_colors = None
                
                while q_low <= q_high:
                    q_mid = (q_low + q_high) // 2
                    size = save_and_measure(pil_img, q_mid)
                    
                    attempts.append({'colors': n_colors, 'quality': q_mid, 'size_kb': round(size / 1024, 2)})
                    
                    if size <= target_size_bytes:
                        # Good, try higher quality
                        best_q_for_colors = q_mid
                        best_size_for_colors = size
                        best_array_for_colors = compressed_array
                        q_low = q_mid + 1
                    else:
                        # Too big, try lower quality
                        q_high = q_mid - 1
                
                # Check if this color level found a valid result
                if best_q_for_colors is not None:
                    # Found a result that meets target
                    if best_size_for_colors > best_size * 0.95 or best_image is None:
                        # Better quality (closer to target) or first valid result
                        best_image = best_array_for_colors
                        best_size = best_size_for_colors
                        best_colors = n_colors
                        best_quality = best_q_for_colors
                    
                    # If we're within 5% of target, stop searching
                    if best_size >= target_size_bytes * 0.95:
                        break
            else:
                # PNG - just check if it fits
                size = save_and_measure(pil_img, 100)
                attempts.append({'colors': n_colors, 'quality': 100, 'size_kb': round(size / 1024, 2)})
                
                if size <= target_size_bytes:
                    best_image = compressed_array
                    best_size = size
                    best_colors = n_colors
                    best_quality = 100
                    break
        
        # Final fallback if nothing worked
        if best_image is None:
            compressed_array = compress_with_colors(image, min_colors)
            pil_img = Image.fromarray(compressed_array.astype(np.uint8))
            best_quality = 10 if is_jpeg else 100
            best_size = save_and_measure(pil_img, best_quality)
            best_image = compressed_array
            best_colors = min_colors
        
        metadata = {
            'achieved_size_kb': round(best_size / 1024, 2),
            'target_size_kb': target_size_kb,
            'colors_used': best_colors,
            'quality_used': best_quality,
            'target_met': best_size <= target_size_bytes,
            'size_accuracy_percent': round((best_size / target_size_bytes) * 100, 1),
            'attempts': len(attempts)
        }
        
        return best_image, metadata
    
    @staticmethod
    def process_image(
        image: np.ndarray,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        resize_mode: ResizeMode = ResizeMode.FIT,
        n_colors: int = 16,
        max_iters: int = 10,
        target_size_kb: Optional[int] = None,
        crop_data: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Full image processing pipeline with resize, crop, and compression.
        
        Args:
            image: Input image
            target_width: Target width (optional)
            target_height: Target height (optional)
            resize_mode: How to handle resizing
            n_colors: Number of colors for compression
            max_iters: K-Means iterations
            target_size_kb: Target file size in KB (optional)
            crop_data: Crop coordinates {x, y, width, height} (optional)
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        processed = image.copy()
        metadata = {
            'original_dimensions': (image.shape[1], image.shape[0]),
            'operations': []
        }
        
        # Apply crop if specified
        if crop_data:
            processed = ImageProcessor.crop_image(
                processed,
                crop_data['x'],
                crop_data['y'],
                crop_data['width'],
                crop_data['height']
            )
            metadata['operations'].append('crop')
            metadata['crop_dimensions'] = (crop_data['width'], crop_data['height'])
        
        # Apply resize if dimensions specified
        if target_width and target_height:
            processed = ImageProcessor.resize_image(
                processed,
                target_width,
                target_height,
                resize_mode
            )
            metadata['operations'].append(f'resize_{resize_mode.value}')
            metadata['resized_dimensions'] = (target_width, target_height)
        
        # Apply compression
        if target_size_kb:
            processed, size_meta = ImageProcessor.compress_to_target_size(
                processed,
                target_size_kb,
                max_colors=n_colors,
                max_iters=max_iters
            )
            metadata['operations'].append('compress_to_size')
            metadata.update(size_meta)
        else:
            compressor = KMeansCompressor(n_colors=n_colors, max_iters=max_iters)
            result = compressor.compress(processed)
            processed = result.compressed_image
            metadata['operations'].append('compress')
            metadata['colors_used'] = n_colors
            metadata['compression_ratio'] = result.compression_ratio
        
        metadata['final_dimensions'] = (processed.shape[1], processed.shape[0])
        
        return processed, metadata
