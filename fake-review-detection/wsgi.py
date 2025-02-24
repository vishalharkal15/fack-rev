import os
from api.app import app
from config import ProductionConfig, DevelopmentConfig

# Set the configuration based on environment
if os.environ.get('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

if __name__ == "__main__":
    # In production, use Gunicorn instead of app.run()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 