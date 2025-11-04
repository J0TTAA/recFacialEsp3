"""
Utilidades para procesar imágenes para reconocimiento facial.
"""
import numpy as np
import torch
import io
from PIL import Image, ImageOps
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

TARGET_IMG_SIZE = 160


def preprocess_image_for_embedding(image):
    """
    Preprocesa una imagen PIL para generar embeddings.
    
    Args:
        image: Imagen PIL en RGB
        
    Returns:
        Tensor de PyTorch listo para el modelo de embeddings
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Asegurar que la imagen esté en RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar a 160x160 si es necesario
    if image.size != (TARGET_IMG_SIZE, TARGET_IMG_SIZE):
        image = image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE), Image.Resampling.LANCZOS)
    
    # Aplicar transformaciones
    tensor = preprocess(image)
    return tensor


def detect_and_crop_face(image, mtcnn):
    """
    Detecta y recorta una cara de una imagen.
    
    Args:
        image: Imagen PIL en RGB
        mtcnn: Modelo MTCNN cargado
        
    Returns:
        Imagen PIL recortada de la cara (160x160) o None si no se detecta cara
    """
    try:
        # Corregir orientación EXIF
        image = ImageOps.exif_transpose(image)
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detectar y recortar cara
        face_tensor = mtcnn(image, save_path=None)
        
        if face_tensor is None:
            logger.warning("No se detectó rostro en la imagen")
            return None
        
        # Convertir tensor a imagen PIL
        face_array = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
        face_image = Image.fromarray(face_array)
        
        # Asegurar tamaño exacto
        face_image = face_image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE), Image.Resampling.LANCZOS)
        
        return face_image
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        return None


def generate_embedding(face_image, embedding_model, device):
    """
    Genera el embedding de una cara detectada.
    
    Args:
        face_image: Imagen PIL de la cara (160x160)
        embedding_model: Modelo InceptionResnetV1 cargado
        device: Dispositivo (GPU/CPU)
        
    Returns:
        Array numpy con el embedding (512 dimensiones)
    """
    try:
        # Preprocesar imagen
        tensor = preprocess_image_for_embedding(face_image)
        
        # Agregar dimensión de batch [1, 3, 160, 160]
        tensor = tensor.unsqueeze(0).to(device)
        
        # Generar embedding
        with torch.no_grad():
            embedding = embedding_model(tensor)
        
        # Convertir a numpy
        embedding = embedding.cpu().numpy().flatten()
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error generando embedding: {e}")
        raise


def verify_face(image_bytes, classifier_model, scaler, embedding_model, mtcnn, preprocess, device, threshold=0.75):
    """
    Pipeline completo para verificar si una cara es "me" o "not_me".
    Similar al flujo original del app.py.
    
    Args:
        image_bytes: Bytes de la imagen
        classifier_model: Modelo clasificador cargado
        scaler: Scaler cargado
        embedding_model: Modelo de embeddings cargado
        mtcnn: Modelo MTCNN cargado
        preprocess: Transformación de preprocesamiento
        device: Dispositivo (GPU/CPU)
        threshold: Umbral para determinar si es "me" (default: 0.75)
        
    Returns:
        dict con is_me, score, threshold y timing
    """
    try:
        # 1. Cargar imagen desde bytes
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 2. Corregir orientación EXIF
        img = ImageOps.exif_transpose(img)
        
        # 3. Detectar y recortar cara con MTCNN
        face_tensor = mtcnn(img, save_path=None)
        if face_tensor is None:
            return {
                "error": "No se detectó un rostro en la imagen"
            }
        
        # 4. Procesar tensor de cara para generar embedding
        # Convertir tensor a PIL Image (como en el original)
        face_image = Image.fromarray(face_tensor.permute(1, 2, 0).byte().cpu().numpy())
        
        # Aplicar transformación de preprocesamiento
        face_tensor_norm = preprocess(face_image).to(device)
        
        # Agregar dimensión de batch [1, 3, 160, 160]
        with torch.no_grad():
            embedding = embedding_model(face_tensor_norm.unsqueeze(0))
        
        # Convertir a numpy
        embedding_np = embedding.cpu().numpy()  # Shape [1, 512]
        
        # 5. Escalar embedding
        embedding_scaled = scaler.transform(embedding_np)
        
        # 6. Obtener probabilidades
        probabilities = classifier_model.predict_proba(embedding_scaled)
        
        # Extraer score de la clase "1" (YO)
        score = float(probabilities[0, 1])
        
        # Aplicar umbral
        is_me = bool(score >= threshold)
        
        # 7. Preparar resultado
        result = {
            "is_me": is_me,
            "score": round(score, 4),
            "threshold": threshold
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error en verificación: {e}", exc_info=True)
        return {
            "error": "Error interno del servidor al procesar la imagen"
        }

