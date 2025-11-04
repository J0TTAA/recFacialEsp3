"""
Aplicación Flask principal para la API de reconocimiento facial.
"""
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import os

# Cargar variables de entorno
load_dotenv()

# Importar blueprints
from api.resources.verify import verify_bp
from api.resources.health import health_bp
from api.utils.model_loader import initialize_models

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """Factory function para crear la aplicación Flask."""
    app = Flask(__name__)
    
    # Configuración
    max_content_mb = int(os.environ.get("MAX_CONTENT_MB", 2))
    app.config['MAX_CONTENT_LENGTH'] = max_content_mb * 1024 * 1024
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    # Habilitar CORS (para permitir peticiones desde el frontend)
    CORS(app)
    
    # Cargar modelos al iniciar (una sola vez)
    logger.info("Iniciando la carga de modelos...")
    try:
        initialize_models()
        logger.info("--- ¡Todos los modelos cargados! Servidor listo. ---")
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        logger.warning("El servidor iniciará pero algunas funciones pueden no estar disponibles")
    
    # Registrar blueprints
    app.register_blueprint(verify_bp)
    app.register_blueprint(health_bp)
    
    # Ruta raíz
    @app.route('/', methods=['GET'])
    def root():
        """Endpoint raíz con información de la API."""
        return jsonify({
            "name": "Face Recognition API",
            "version": "1.0.0",
            "description": "API para reconocimiento facial que identifica si una cara pertenece al usuario (me) o no (not_me)",
            "endpoints": {
                "health": "/healthz",
                "verify": "/verify"
            },
            "usage": {
                "verify": "POST /verify con form-data campo 'image' (archivo de imagen)"
            }
        }), 200
    
    # Manejo de errores
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Endpoint no encontrado"
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "success": False,
            "error": "Error interno del servidor"
        }), 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "success": False,
            "error": "El archivo de imagen es demasiado grande. Máximo 16MB."
        }), 413
    
    # Crear directorio de uploads si no existe
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    logger.info("Aplicación Flask creada exitosamente")
    
    return app


if __name__ == '__main__':
    app = create_app()
    
    # Obtener configuración desde variables de entorno
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Iniciando servidor Flask en {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug)

