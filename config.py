"""
Configuration settings for the travel planning system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# Amadeus API Configuration
AMADEUS_CONFIG = {
    'api_key': os.getenv('AMADEUS_API_KEY'),
    'api_secret': os.getenv('AMADEUS_API_SECRET'),
    'environment': os.getenv('AMADEUS_ENVIRONMENT', 'test')  # 'test' or 'production'
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': BASE_DIR / 'logs' / 'travel_planner.log',
            'mode': 'a',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'amadeus': {
            'level': 'WARNING',  # Reduce Amadeus SDK log noise
            'propagate': False
        }
    }
}

# Application Settings
APP_CONFIG = {
    'debug': os.getenv('DEBUG', 'false').lower() == 'true',
    'max_retries': int(os.getenv('MAX_RETRIES', '3')),
    'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '30'))
}

# Cache Settings
CACHE_CONFIG = {
    'enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
    'ttl': int(os.getenv('CACHE_TTL', '3600')),  # 1 hour
    'path': BASE_DIR / 'cache'
}

# Create necessary directories
for directory in [BASE_DIR / 'logs', CACHE_CONFIG['path']]:
    directory.mkdir(parents=True, exist_ok=True)
