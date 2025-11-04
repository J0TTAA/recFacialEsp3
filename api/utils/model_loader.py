"""
Utilidades para cargar modelos de reconocimiento facial.
"""
import os
import joblib
import torch
import logging
from facenet_pytorch import InceptionResnetV1, MTCNN
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

# Rutas de los modelos (usando variables de entorno si están disponibles)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.joblib")
PRETRAINED_DATASET = 'vggface2'
TARGET_IMG_SIZE = 160

# Variables globales para almacenar los modelos cargados
_classifier_model = None
_scaler = None
_embedding_model = None
_mtcnn = None
_device = None


def get_device():
    """Obtiene el dispositivo (GPU/CPU) disponible."""
    global _device
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Dispositivo seleccionado: {_device}")
    return _device


def load_classifier_model():
    """Carga el modelo clasificador de LogisticRegression."""
    global _classifier_model
    if _classifier_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modelo no encontrado en {MODEL_PATH}. "
                "Ejecuta 'train.py' primero para entrenar el modelo."
            )
        logger.info(f"Cargando modelo clasificador desde {MODEL_PATH}")
        _classifier_model = joblib.load(MODEL_PATH)
        logger.info("Modelo clasificador cargado exitosamente")
    return _classifier_model


def load_scaler():
    """Carga el StandardScaler."""
    global _scaler
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler no encontrado en {SCALER_PATH}. "
                "Ejecuta 'train.py' primero para entrenar el modelo."
            )
        logger.info(f"Cargando scaler desde {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler cargado exitosamente")
    return _scaler


def load_embedding_model():
    """Carga el modelo InceptionResnetV1 para generar embeddings."""
    global _embedding_model
    if _embedding_model is None:
        device = get_device()
        logger.info(f"Cargando modelo InceptionResnetV1 ({PRETRAINED_DATASET})")
        _embedding_model = InceptionResnetV1(pretrained=PRETRAINED_DATASET).eval().to(device)
        logger.info("Modelo de embeddings cargado exitosamente")
    return _embedding_model


def load_mtcnn():
    """Carga el modelo MTCNN para detectar y recortar caras."""
    global _mtcnn
    if _mtcnn is None:
        device = get_device()
        logger.info("Cargando modelo MTCNN")
        _mtcnn = MTCNN(
            image_size=TARGET_IMG_SIZE,
            margin=0,  # Igual que en el original
            min_face_size=20,
            post_process=False,
            select_largest=False,
            keep_all=False,
            device=device
        )
        logger.info("Modelo MTCNN cargado exitosamente")
    return _mtcnn


def get_preprocess():
    """Retorna la transformación de preprocesamiento para embeddings."""
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return preprocess


def initialize_models():
    """Inicializa todos los modelos necesarios."""
    try:
        load_classifier_model()
        load_scaler()
        load_embedding_model()
        load_mtcnn()
        logger.info("Todos los modelos inicializados correctamente")
    except Exception as e:
        logger.error(f"Error inicializando modelos: {e}")
        raise

