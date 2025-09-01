# Configuration file for ML Recommender
import os

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or 'sk-your-openai-api-key-here'
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True
    
    # ML Configuration
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Model Configuration
    CV_FOLDS = 5
    RANDOM_STATE = 42
    N_JOBS = -1  # Use all available cores
    
    # Hyperparameter Tuning Configuration
    TUNING_CV_FOLDS = 3  # Reduced for faster tuning
    MAX_TUNING_TIME = 300  # 5 minutes max per model