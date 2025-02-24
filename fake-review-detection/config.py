class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'your-secret-key-here'
    CORS_HEADERS = 'Content-Type'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

class DevelopmentConfig(Config):
    DEBUG = True
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

class ProductionConfig(Config):
    # Update these with your actual production domain
    CORS_ORIGINS = ["https://verivoicecybron.netlify.app/"]
    # Add any production-specific settings
    PROPAGATE_EXCEPTIONS = True 