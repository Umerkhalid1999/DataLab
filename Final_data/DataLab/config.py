import os

class Config:
    """Application Configuration"""
    SECRET_KEY = os.urandom(32)
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'json', 'txt', 'xlsx', 'jpg', 'png'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

    # Use relative path from the current script directory
    FIREBASE_CONFIG_PATH = os.environ.get(
        'FIREBASE_CONFIG_PATH',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'data-storing123-firebase-adminsdk-fbsvc-2a77c2f29a.json')
    )
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')  # Default model

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 