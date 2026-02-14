<div align="center">

# K-Means Image Compressor

**High-performance image compression using K-Means clustering algorithm**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NumPy](https://img.shields.io/badge/NumPy-Optimized-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github&logoColor=white)](https://github.com/gabsgj/K-Means-Image-Compressor)

[Live Demo](https://imagecompressor.gabrieljames.me/) · [Documentation](#api-documentation) · [Report Bug](https://github.com/gabsgj/K-Means-Image-Compressor/issues)

</div>

---
<img width="1897" height="1606" alt="Image" src="https://github.com/user-attachments/assets/98ae54f4-eea6-4838-8a2b-df78bee6e7df" />
<img width="1897" height="3981" alt="Image" src="https://github.com/user-attachments/assets/b8a9738f-c02b-438e-bfa8-63407b37a818" />
<img width="1897" height="1057" alt="Image" src="https://github.com/user-attachments/assets/28a0988c-2494-4e6c-a5c8-6b8e27329819" />

---
## Overview

A production-ready web application that leverages the K-Means clustering algorithm to intelligently compress images through color quantization. The application reduces file sizes by **up to 6x** while preserving visual quality, making it ideal for web optimization, storage reduction, and bandwidth conservation.

### Key Features

| Feature | Description |
|---------|-------------|
| **K-Means++ Initialization** | Smart centroid initialization for faster convergence and better results |
| **Pixel Sampling** | Processes large images efficiently by sampling up to 100,000 pixels |
| **Dimension Control** | Resize, crop, or fit images to target dimensions |
| **Target File Size** | Automatically adjusts compression to meet file size requirements |
| **Interactive Comparison** | Side-by-side slider to compare original vs. compressed images |
| **REST API** | Full programmatic access for integration into existing workflows |
| **Responsive UI** | Clean, minimal interface optimized for desktop and mobile |

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/gabsgj/K-Means-Image-Compressor.git
cd K-Means-Image-Compressor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the application
python app.py
```

The application will be available at `http://localhost:5000`

### Docker

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or build and run manually
docker build -t kmeans-compressor .
docker run -p 5000:5000 kmeans-compressor
```

---

## Project Structure

```
kmeans-image-compressor/
├── app.py                  # Flask application entry point
├── kmeans_compressor.py    # Core K-Means algorithm implementation
├── config.py               # Application configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-container orchestration
│
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Main application interface
│   ├── about.html          # Algorithm explanation
│   ├── api_docs.html       # API reference
│   └── error pages         # 404, 500 handlers
│
├── static/
│   ├── css/style.css       # Application stylesheet
│   └── js/app.js           # Frontend logic
│
├── uploads/                # Temporary upload storage
└── compressed/             # Processed image output
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Application
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key

# Upload Limits
MAX_CONTENT_LENGTH=16777216    # 16MB max file size

# Compression Defaults
DEFAULT_N_COLORS=16            # Color palette size
DEFAULT_MAX_ITERS=8            # K-Means iterations
DEFAULT_SAMPLE_SIZE=100000     # Pixels sampled for large images
```

---

## API Documentation

### Base URL

```
http://localhost:5000/api
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and system status |
| `POST` | `/upload` | Upload an image for processing |
| `POST` | `/compress` | Compress an uploaded image |
| `POST` | `/compress-advanced` | Compress with dimension and size targets |
| `POST` | `/compress-direct` | Upload and compress in single request |
| `GET` | `/presets` | Available compression presets |
| `GET` | `/resize-modes` | Available resize modes |
| `GET` | `/dimension-presets` | Preset dimension options |
| `GET` | `/download/<id>` | Download compressed image |

### Example Usage

**Python**
```python
import requests

# Upload and compress
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/compress-direct',
        files={'image': f},
        data={'n_colors': 16, 'max_iters': 8}
    )

result = response.json()
print(f"Compression ratio: {result['compression_ratio']:.2f}x")
print(f"Processing time: {result['processing_time']:.2f}s")

# Download compressed image
img_response = requests.get(result['download_url'])
with open('compressed.png', 'wb') as f:
    f.write(img_response.content)
```

**cURL**
```bash
curl -X POST http://localhost:5000/api/compress-direct \
  -F "image=@photo.jpg" \
  -F "n_colors=16"
```

---

## Algorithm

The K-Means clustering algorithm reduces the color palette of an image through iterative refinement:

```
┌─────────────────────────────────────────────────────────────────┐
│  Input Image          K-Means Clustering         Output Image   │
│  (millions of    →    (K centroids)         →    (K colors)     │
│   colors)                                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Process

1. **Initialize** — Select K colors using K-Means++ (weighted random selection favoring distant points)
2. **Assign** — Map each pixel to its nearest centroid using Euclidean distance
3. **Update** — Recalculate centroids as the mean of assigned pixels
4. **Converge** — Repeat steps 2-3 until centroids stabilize or max iterations reached
5. **Reconstruct** — Replace each pixel with its assigned centroid color

### Compression Analysis

| Representation | Bits per Pixel | Formula |
|----------------|----------------|---------|
| Original | 24 | 8 bits × 3 channels (RGB) |
| Compressed | log₂(K) | Index into color palette |

**Example** — 1024×1024 image with K=16 colors:
- Original: 1,048,576 × 24 = **25.2 MB**
- Compressed: (16 × 24) + (1,048,576 × 4) = **4.2 MB**
- **Compression ratio: ~6x**

---

## Performance

Optimizations implemented for production workloads:

| Optimization | Impact |
|--------------|--------|
| Pixel sampling (100K max) | 10-50x faster on large images |
| Squared Euclidean distance | 2-3x faster distance computation |
| NumPy `bincount` aggregation | 3-5x faster centroid updates |
| Batch processing (50K chunks) | Prevents memory overflow |
| Early convergence detection | Reduces unnecessary iterations |

### Benchmarks

| Image Size | Colors | Processing Time | Compression Ratio |
|------------|--------|-----------------|-------------------|
| 512×512 | 16 | ~150ms | 6.0x |
| 1024×1024 | 16 | ~300ms | 6.0x |
| 2048×2048 | 32 | ~500ms | 4.8x |
| 4096×4096 | 64 | ~800ms | 4.0x |

*Benchmarks measured on Intel i7-10700K, 32GB RAM*

---

## Deployment

### Heroku

```bash
heroku create your-app-name
heroku config:set SECRET_KEY=$(openssl rand -hex 32)
git push heroku main
```

### Zeabur

```bash
# Option 1: Deploy source code directly
# 1. Create a project in Zeabur dashboard
# 2. Import this GitHub repository
# 3. Zeabur auto-detects Dockerfile and deploys

# Option 2: Deploy from Docker Hub image
# Image: gabsgj/image-compressor-using-k-means:latest
```

### Google Cloud Run

```bash
gcloud run deploy kmeans-compressor \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS (Elastic Beanstalk)

```bash
eb init -p docker kmeans-compressor --region us-east-1
eb create production
eb deploy
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.9+, Flask 3.0, Gunicorn |
| **Algorithm** | NumPy (vectorized operations), K-Means++ |
| **Image Processing** | Pillow (PIL) |
| **Frontend** | Vanilla JavaScript, CSS3, Cropper.js |
| **Containerization** | Docker, Docker Compose |

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

Please ensure your code follows PEP 8 guidelines and includes appropriate tests.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- K-Means clustering concepts adapted from Andrew Ng's Machine Learning course
- K-Means++ initialization algorithm by Arthur & Vassilvitskii (2007)

---

<div align="center">

**[Back to Top](#k-means-image-compressor)**

</div>
