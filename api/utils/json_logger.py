"""
Utilidades para logging JSON estructurado.
"""
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional


class JSONFormatter(logging.Formatter):
    """Formatter que convierte logs a formato JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatea un registro de log como JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Agregar campos extra si existen
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Agregar excepción si existe
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_json_logging(log_level: int = logging.INFO):
    """
    Configura el logging para usar formato JSON.
    
    Args:
        log_level: Nivel de logging (logging.INFO, logging.DEBUG, etc.)
    """
    # Obtener el logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remover handlers existentes
    root_logger.handlers = []
    
    # Crear handler para stdout
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(JSONFormatter())
    
    root_logger.addHandler(handler)


def log_api_request(
    logger: logging.Logger,
    endpoint: str,
    method: str,
    latency_ms: float,
    file_size_bytes: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
    error: Optional[str] = None
):
    """
    Registra una petición a la API en formato JSON estructurado.
    
    Args:
        logger: Logger a usar
        endpoint: Endpoint de la API
        method: Método HTTP (GET, POST, etc.)
        latency_ms: Latencia en milisegundos
        file_size_bytes: Tamaño del archivo en bytes (si aplica)
        result: Resultado de la operación
        status_code: Código de estado HTTP
        error: Mensaje de error (si aplica)
    """
    log_data = {
        'type': 'api_request',
        'endpoint': endpoint,
        'method': method,
        'latency_ms': round(latency_ms, 2),
        'status_code': status_code,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    if file_size_bytes is not None:
        log_data['file_size_bytes'] = file_size_bytes
        log_data['file_size_mb'] = round(file_size_bytes / (1024 * 1024), 4)
    
    if result is not None:
        log_data['result'] = result
    
    if error is not None:
        log_data['error'] = error
    
    # Log en formato JSON usando el formatter personalizado
    log_message = json.dumps(log_data, ensure_ascii=False)
    
    if status_code < 400:
        logger.info(log_message)
    else:
        logger.error(log_message)

