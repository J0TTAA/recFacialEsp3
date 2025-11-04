"""
Recurso de API para verificar el estado de la API.
"""
from flask import Blueprint, jsonify
import os
import logging

from api.utils.model_loader import (
    load_classifier_model,
    load_scaler,
    load_embedding_model,
    load_mtcnn
)

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)


@health_bp.route('/healthz', methods=['GET'])
def health():
    """
    Endpoint de "salud" para monitoreo.
    
    Response:
        {
            "status": "ok" o "unhealthy",
            "reason": string (si está unhealthy)
        }
    """
    try:
        # Intentar cargar modelos
        classifier_model = load_classifier_model()
        embedding_model = load_embedding_model()
        
        # Si los modelos no cargaron, el servicio no está saludable
        if classifier_model is None or embedding_model is None:
            return jsonify({"status": "unhealthy", "reason": "Models not loaded"}), 503
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({
            "status": "unhealthy",
            "reason": str(e)
        }), 503

