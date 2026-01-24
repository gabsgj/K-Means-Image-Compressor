"""
Application Configuration
=========================
Configuration settings for different environments.
"""

import os
from datetime import timedelta


class Config:
    """Base configuration class."""
    
    # Application
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    VERSION = '1.0.0'
    
    # File Upload Settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    COMPRESSED_FOLDER = os.environ.get('COMPRESSED_FOLDER', 'compressed')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # Compression Defaults
    DEFAULT_N_COLORS = int(os.environ.get('DEFAULT_N_COLORS', 16))
    DEFAULT_MAX_ITERS = int(os.environ.get('DEFAULT_MAX_ITERS', 10))
    
    # Security
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS Settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    
    def __init__(self):
        # In production, require a proper secret key
        secret_key = os.environ.get('SECRET_KEY')
        if secret_key:
            self.SECRET_KEY = secret_key


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = 'test_uploads'
    COMPRESSED_FOLDER = 'test_compressed'


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment."""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
