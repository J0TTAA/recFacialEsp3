import os
from glob import glob
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
import torch
import logging
from tqdm import tqdm

# --- Configuración del Logging ---
# Esto nos permite imprimir mensajes útiles (INFO, WARNING, ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Configuración ---
# Definimos las rutas en un solo lugar.
INPUT_DATA_DIR = "data"
OUTPUT_CROPPED_DIR = "data/cropped"
# El tamaño que espera el modelo InceptionResnetV1 (FaceNet)
TARGET_IMG_SIZE = 160
# Parámetros de MTCNN:
# image_size: El tamaño al que redimensiona la imagen ANTES de detectar.
#             Más pequeño (como 160) es más rápido pero menos preciso.
#             Más grande (como 512) es más lento pero detecta caras más pequeñas.
#             Dejémoslo en 160 por velocidad.
# margin: Cuánto "relleno" extra dejamos alrededor del rostro detectado.
#         Un poco de margen (ej. 20-30px) es bueno para capturar frente y mentón.
MTCNN_MARGIN = 30
# min_face_size: No intentes detectar caras más pequeñas que esto (en píxeles).
MTCNN_MIN_FACE_SIZE = 20


def setup_mtcnn():
    """
    Configura y devuelve el modelo MTCNN, seleccionando GPU (CUDA) si está disponible.
    """
    # El "Deep Dive" de 'torch.device':
    # PyTorch puede ejecutar cálculos en la CPU o en la GPU (si tienes una Nvidia, usará CUDA).
    # La detección de rostros es MUCHO más rápida en GPU.
    # 'cuda:0' es la primera GPU. 'cpu' es la CPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Dispositivo seleccionado para MTCNN: {device}")

    # 'select_largest=False': Si hay varias caras, no elijas solo la más grande.
    # 'post_process=False': No normaliza la imagen (0-255 a 0-1). Lo haremos nosotros.
    # 'keep_all=False': Quédate solo con la cara que tenga la probabilidad más alta.
    #                 Esto es crucial para nuestro caso de uso "selfie".
    mtcnn = MTCNN(
        image_size=TARGET_IMG_SIZE, 
        margin=MTCNN_MARGIN, 
        min_face_size=MTCNN_MIN_FACE_SIZE, 
        post_process=False, 
        select_largest=False,
        keep_all=False, 
        device=device
    )
    return mtcnn

def crop_and_save_image(image_path, mtcnn, output_path):
    """
    Procesa una sola imagen: la abre, detecta el rostro, la recorta,
    la redimensiona y la guarda en la ruta de salida.
    """
    try:
        # 1. Abrir la imagen usando Pillow (PIL)
        # Usamos .convert('RGB') para asegurar que no haya canal Alfa (RGBA)
        # o que no sea Blanco y Negro (L).
        img = Image.open(image_path).convert('RGB')

        # 2. Corregir orientación (problema común de móviles)
        # Algunas fotos de móviles se guardan "de lado" pero tienen
        # metadatos EXIF que dicen cómo rotarlas. 'ImageOps.exif_transpose'
        # lee esos metadatos y rota la imagen correctamente.
        img = ImageOps.exif_transpose(img)

        # 3. Detección del rostro
        # mtcnn(img) devuelve 3 cosas:
        #   boxes: Las coordenadas [x1, y1, x2, y2] de la cara.
        #   probs: La confianza (probabilidad) de que es una cara.
        #   landmarks: Puntos clave (ojos, nariz, boca).
        # Como usamos 'keep_all=False', solo devolverá 1 (el mejor) o None.
        
        # 'save_path=None' es un truco.
        # Si le pasas un save_path, mtcnn guarda la imagen él mismo.
        # Si le pones None, te devuelve un Tensor de PyTorch con la imagen recortada.
        # Esto nos da más control.
        face_tensor = mtcnn(img, save_path=None)

        if face_tensor is None:
            logging.warning(f"No se detectó rostro en: {image_path}")
            return False

        # 4. Convertir de Tensor a Imagen PIL
        # El tensor de salida de MTCNN está en formato PyTorch (C, H, W)
        # y normalizado (0 a 1). Debemos revertirlo para guardarlo.
        # .permute(1, 2, 0): Cambia de (Canales, Alto, Ancho) a (Alto, Ancho, Canales)
        # * 255: Vuelve al rango de píxeles 0-255
        # .byte(): Convierte a tipo de dato de 8 bits (entero)
        # .cpu().numpy(): Lo saca de la GPU y lo convierte a un array de Numpy
        face_array = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
        
        # Finalmente, crea una imagen PIL desde el array de Numpy
        face_image = Image.fromarray(face_array)
        
        # 5. Redimensión final (por si acaso)
        # Aunque pedimos 160, el 'margin' puede hacer que varíe.
        # Nos aseguramos de que sea EXACTAMENTE 160x160.
        # Usamos 'Resampling.LANCZOS' que es un filtro de alta calidad
        # para reducir el tamaño (antialiasing).
        face_image = face_image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE), Image.Resampling.LANCZOS)
        
        # 6. Guardar la imagen
        # Aseguramos que el directorio de salida exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        face_image.save(output_path)
        
        # logging.info(f"Guardado recorte de {image_path} en {output_path}")
        return True

    except Exception as e:
        # Capturamos cualquier error (ej. archivo corrupto)
        logging.error(f"Error procesando {image_path}: {e}")
        return False

def main():
    """
    Función principal: orquesta todo el proceso.
    """
    logging.info("Iniciando el script de recorte de rostros...")
    
    mtcnn = setup_mtcnn()
    
    # Buscamos en las subcarpetas de 'data' (ej. 'me', 'not_me')
    # "glob" es una librería para encontrar archivos usando "comodines"
    # INPUT_DATA_DIR + "/**/" -> busca en 'data/' y todos sus subdirectorios
    # "/*.jpg" -> cualquier archivo que termine en .jpg
    
    # Lista de tipos de imagen que aceptamos
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"]
    image_paths = []
    
    # Recorremos 'me' y 'not_me'
    for category in ["me", "not_me"]:
        category_path = os.path.join(INPUT_DATA_DIR, category)
        if not os.path.isdir(category_path):
            logging.warning(f"Directorio no encontrado, omitiendo: {category_path}")
            continue
            
        # Buscamos todos los tipos de imagen
        for ext in image_extensions:
            # os.path.join(category_path, ext) -> "data/me/*.jpg"
            # recursive=True permite buscar en subdirectorios (aunque no lo usamos mucho aquí)
            image_paths.extend(glob(os.path.join(category_path, ext), recursive=True))

    logging.info(f"Se encontraron {len(image_paths)} imágenes en total.")
    
    if not image_paths:
        logging.error(f"No se encontraron imágenes en {INPUT_DATA_DIR}. ¿Agregaste tus fotos a 'data/me' y 'data/not_me'?")
        return

    # Creamos el directorio de salida principal
    os.makedirs(OUTPUT_CROPPED_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0
    
    # --- La Barra de Progreso (tqdm) ---
    # 'tqdm' es una librería fantástica. Simplemente "envuelves"
    # cualquier lista o iterador (como 'image_paths') con 'tqdm()'
    # y te mostrará una barra de progreso inteligente.
    for img_path in tqdm(image_paths, desc="Procesando imágenes"):
        
        # 'os.path.relpath' calcula la ruta "relativa".
        # Ej: img_path = "data/me/foto1.jpg"
        #     INPUT_DATA_DIR = "data"
        #     -> rel_path = "me/foto1.jpg"
        rel_path = os.path.relpath(img_path, INPUT_DATA_DIR)
        
        # Armamos la ruta de salida
        # Ej: output_path = "data/cropped/me/foto1.jpg"
        output_path = os.path.join(OUTPUT_CROPPED_DIR, rel_path)
        
        # Nos aseguramos de que el directorio de salida (ej. "data/cropped/me") exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Procesamos la imagen
        if crop_and_save_image(img_path, mtcnn, output_path):
            success_count += 1
        else:
            fail_count += 1
            
    logging.info("--- Proceso de Recorte Finalizado ---")
    logging.info(f"Imágenes procesadas exitosamente: {success_count}")
    logging.info(f"Imágenes fallidas (sin rostro o corruptas): {fail_count}")

if __name__ == "__main__":
    main()