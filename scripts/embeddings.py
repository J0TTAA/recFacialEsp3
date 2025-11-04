import os
import torch
import numpy as np
from PIL import Image, ImageOps
from glob import glob
import logging
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Configuración ---
INPUT_CROPPED_DIR = "data/cropped"
OUTPUT_EMBEDDINGS_PATH = "data/embeddings.npy"
OUTPUT_LABELS_PATH = "data/labels.npy"
# Procesar imágenes en lotes es MUCHO más rápido en GPU
BATCH_SIZE = 32
# El dataset en el que fue entrenado el modelo. 'vggface2' es el estándar.
PRETRAINED_DATASET = 'vggface2'

def setup_environment():
    """Configura el dispositivo (GPU/CPU) y el modelo InceptionResnetV1."""
    
    # 1. Configurar Dispositivo (GPU o CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Dispositivo seleccionado para Embeddings: {device}")

    # 2. Cargar el Modelo Pre-entrenado
    # Cargamos InceptionResnetV1, entrenado en 'vggface2'.
    # .eval() es CRÍTICO: le dice al modelo que estamos en modo "inferencia"
    # y no "entrenamiento". Desactiva cosas como el Dropout.
    # .to(device) mueve el modelo a la GPU si está disponible.
    resnet = InceptionResnetV1(pretrained=PRETRAINED_DATASET).eval().to(device)
    logging.info(f"Modelo InceptionResnetV1 ({PRETRAINED_DATASET}) cargado.")
    
    return resnet, device

def get_image_paths_and_labels(input_dir):
    """
    Recorre los subdirectorios 'me' y 'not_me' y crea una lista de
    rutas de imagen y una lista paralela de etiquetas (1 y 0).
    """
    image_paths = []
    labels = []

    # 1. Categoría "YO" (Etiqueta = 1)
    me_dir = os.path.join(input_dir, "me")
    if os.path.isdir(me_dir):
        me_paths = glob(os.path.join(me_dir, "*.jpg"))
        image_paths.extend(me_paths)
        labels.extend([1] * len(me_paths))
        logging.info(f"Encontradas {len(me_paths)} imágenes en 'me'")
    else:
        logging.warning(f"Directorio no encontrado: {me_dir}")

    # 2. Categoría "NO-YO" (Etiqueta = 0)
    not_me_dir = os.path.join(input_dir, "not_me")
    if os.path.isdir(not_me_dir):
        not_me_paths = glob(os.path.join(not_me_dir, "*.jpg"))
        image_paths.extend(not_me_paths)
        labels.extend([0] * len(not_me_paths))
        logging.info(f"Encontradas {len(not_me_paths)} imágenes en 'not_me'")
    else:
        logging.warning(f"Directorio no encontrado: {not_me_dir}")
        
    return image_paths, labels

def process_images_in_batches(image_paths, resnet_model, device):
    """
    Procesa la lista de imágenes en lotes (batches) para generar
    los embeddings de 512 dimensiones.
    """
    
    all_embeddings = []
    
    # El "Deep Dive" de las Transformaciones (transforms):
    # Esto prepara la imagen de PIL (cargada del disco) para la red.
    # 1. transforms.ToTensor(): Convierte una imagen PIL (rango 0-255)
    #    a un Tensor de PyTorch (rango 0.0-1.0).
    # 2. transforms.Normalize: Ajusta el rango de [0.0, 1.0] a [-1.0, 1.0].
    #    Esto se hace restando la media (0.5) y dividiendo por la std (0.5).
    #    (valor - 0.5) / 0.5 = (valor * 2) - 1
    #    Es el preprocesamiento estándar de InceptionResnet.
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Usamos tqdm para una barra de progreso
    # Iteramos sobre la lista de imágenes en "trozos" del tamaño de BATCH_SIZE
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Generando Embeddings"):
        
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_tensors = []

        for img_path in batch_paths:
            try:
                # Abrimos la imagen (ya está 160x160 por el script H2)
                img = Image.open(img_path).convert('RGB')
                # Aplicamos la normalización
                tensor = preprocess(img)
                batch_tensors.append(tensor)
            except Exception as e:
                logging.warning(f"Error cargando {img_path}: {e}. Omitiendo.")
                
        if not batch_tensors:
            continue

        # 1. "Apilamos" los tensores individuales en un solo "lote" (batch)
        #    Ej. 32 tensores de [3, 160, 160] -> 1 tensor de [32, 3, 160, 160]
        # 2. .to(device) mueve todo el lote a la GPU de una sola vez.
        batch = torch.stack(batch_tensors).to(device)

        # 3. ¡La Magia! Aquí generamos los embeddings.
        #    torch.no_grad() es VITAL. Le dice a PyTorch: "no calcules
        #    gradientes, solo estamos prediciendo". Ahorra memoria y
        #    acelera el proceso enormemente.
        with torch.no_grad():
            embeddings = resnet_model(batch)

        # 4. .cpu().numpy() mueve los embeddings de vuelta de la GPU a la CPU
        #    y los convierte a un array de Numpy para guardarlos.
        all_embeddings.append(embeddings.cpu().numpy())

    if not all_embeddings:
        logging.error("No se generaron embeddings. ¿Está 'data/cropped' vacío?")
        return None
        
    # "Concatenamos" todos los lotes de embeddings en un solo gran array
    final_embeddings = np.concatenate(all_embeddings)
    return final_embeddings

def main():
    """Función principal: orquesta la generación de embeddings."""
    logging.info("Iniciando el script de generación de embeddings (H3)...")
    
    resnet, device = setup_environment()
    
    image_paths, labels = get_image_paths_and_labels(INPUT_CROPPED_DIR)
    
    if not image_paths:
        logging.error(f"No se encontraron imágenes en {INPUT_CROPPED_DIR}. "
                      f"¿Ejecutaste 'scripts/crop_faces.py' (H2) primero?")
        return

    embeddings = process_images_in_batches(image_paths, resnet, device)
    
    if embeddings is not None:
        # Nos aseguramos de que el número de embeddings coincida con el de etiquetas
        if len(embeddings) != len(labels):
            logging.error("¡Error! El número de embeddings no coincide con el de etiquetas."
                          " Esto puede pasar si algunas imágenes fallaron al cargar.")
            # Solución: Alinear las etiquetas con las imágenes que SÍ se procesaron.
            # (Simplificación por ahora: asumimos que todas funcionaron o falló el lote)
            # Para un script robusto, deberíamos manejar esto índice por índice.
            # Por ahora, si hay un desajuste, es mejor parar.
            logging.error(f"Embeddings: {len(embeddings)}, Labels: {len(labels)}")
            return

        # 5. Guardar los resultados
        # Creamos el directorio 'data/' si no existe (aunque ya debería)
        os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS_PATH), exist_ok=True)
        
        # Guardamos los arrays de Numpy
        np.save(OUTPUT_EMBEDDINGS_PATH, embeddings)
        np.save(OUTPUT_LABELS_PATH, np.array(labels))

        logging.info("--- Proceso de Embeddings Finalizado ---")
        logging.info(f"Embeddings guardados en: {OUTPUT_EMBEDDINGS_PATH}")
        logging.info(f"Etiquetas guardadas en: {OUTPUT_LABELS_PATH}")
        logging.info(f"Dimensiones de Embeddings: {embeddings.shape}")
        logging.info(f"Dimensiones de Etiquetas: {len(labels)}")

if __name__ == "__main__":
    main()