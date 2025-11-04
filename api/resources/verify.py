"""
Recurso de API para verificación de reconocimiento facial.
"""
from flask import Blueprint, request, jsonify
import logging
import time
import os
from dotenv import load_dotenv

from api.utils.model_loader import (
    load_classifier_model,
    load_scaler,
    load_embedding_model,
    load_mtcnn,
    get_device,
    get_preprocess
)
from api.utils.image_processor import verify_face

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

verify_bp = Blueprint('verify', __name__)

# Constantes
MODEL_VERSION = os.environ.get("MODEL_VERSION", "me-verifier-v1")
VERIFY_THRESHOLD = float(os.environ.get("VERIFY_THRESHOLD", 0.75))
MAX_CONTENT_MB = int(os.environ.get("MAX_CONTENT_MB", 2))


@verify_bp.route('/verify', methods=['POST'])
def verify():
    """
    Endpoint principal. Recibe una imagen y devuelve
    si es "YO" (is_me: true/false) y el score.
    
    Request:
        - Form-data con campo 'image' (archivo de imagen)
        
    Response:
        {
            "model_version": string,
            "is_me": bool,
            "score": float,
            "threshold": float,
            "timing_ms": float
        }
    """
    start_time = time.perf_counter()
    
    try:
        # --- 1. Validación de la Solicitud ---
        if 'image' not in request.files:
            return jsonify({"error": "No se encontró el archivo 'image' en la solicitud"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "Nombre de archivo vacío"}), 400
            
        # Validar tipo de archivo
        if file.mimetype not in ['image/jpeg', 'image/png']:
            return jsonify({"error": "Tipo de archivo no permitido. Solo image/jpeg o image/png"}), 400
            
        # Validar tamaño de archivo
        if file.content_length and file.content_length > MAX_CONTENT_MB * 1024 * 1024:
            return jsonify({"error": f"Archivo demasiado grande. Límite: {MAX_CONTENT_MB} MB"}), 413

        # Leer imagen como bytes
        image_bytes = file.read()
        
        # Cargar modelos (si no están cargados)
        classifier_model = load_classifier_model()
        scaler = load_scaler()
        embedding_model = load_embedding_model()
        mtcnn = load_mtcnn()
        preprocess = get_preprocess()
        device = get_device()
        
        # Realizar verificación
        result = verify_face(
            image_bytes,
            classifier_model,
            scaler,
            embedding_model,
            mtcnn,
            preprocess,
            device,
            VERIFY_THRESHOLD
        )
        
        # Si hay error, retornarlo
        if "error" in result:
            return jsonify(result), 400
        
        # Calcular tiempo de procesamiento
        end_time = time.perf_counter()
        timing_ms = (end_time - start_time) * 1000
        
        # Agregar información adicional
        result["model_version"] = MODEL_VERSION
        result["timing_ms"] = round(timing_ms, 2)
        
        logger.info(f"Petición /verify: OK. is_me={result['is_me']}, score={result['score']:.4f}, time={timing_ms:.2f}ms")
        
        return jsonify(result), 200
            
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return jsonify({
            "error": "Modelos no encontrados. Asegúrate de haber entrenado el modelo ejecutando 'train.py'."
        }), 503
        
    except Exception as e:
        logger.error(f"Error inesperado en verificación: {e}", exc_info=True)
        return jsonify({
            "error": "Error interno del servidor al procesar la imagen"
        }), 500

