import os
import numpy as np
import joblib
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Configuración ---
EMBEDDINGS_PATH = "data/embeddings.npy"
LABELS_PATH = "data/labels.npy"
MODEL_SAVE_PATH = "models/model.joblib"
SCALER_SAVE_PATH = "models/scaler.joblib"
METRICS_SAVE_PATH = "reports/metrics.json"

# 'test_size=0.2' significa 20% para validación, 80% para entrenamiento
TEST_SPLIT_SIZE = 0.2
# 'random_state=42' es una "semilla". Asegura que la división
# sea siempre la misma. Esto hace que nuestros experimentos sean
# "reproducibles". 42 es solo una convención.
RANDOM_SEED = 42

def load_data(embeddings_path, labels_path):
    """Carga los embeddings y etiquetas desde los archivos .npy."""
    logging.info(f"Cargando embeddings desde {embeddings_path}")
    try:
        X = np.load(embeddings_path)
    except FileNotFoundError:
        logging.error(f"Error: Archivo no encontrado {embeddings_path}.")
        logging.error("Asegúrate de ejecutar 'scripts/embeddings.py' (H3) primero.")
        return None, None
        
    logging.info(f"Cargando etiquetas desde {labels_path}")
    try:
        y = np.load(labels_path)
    except FileNotFoundError:
        logging.error(f"Error: Archivo no encontrado {labels_path}.")
        logging.error("Asegúrate de ejecutar 'scripts/embeddings.py' (H3) primero.")
        return None, None
        
    logging.info(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} dimensiones.")
    
    # Comprobación de balanceo (muy importante)
    unique, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique, counts))
    logging.info(f"Conteo de etiquetas: {label_counts}")
    
    if 1 not in label_counts or label_counts[1] == 0:
        logging.error("Error: No se encontraron etiquetas '1' (YO).")
        logging.error("Asegúrate de que 'data/cropped/me' no esté vacío.")
        return None, None
    if 0 not in label_counts or label_counts[0] == 0:
         logging.error("Error: No se encontraron etiquetas '0' (NO-YO).")
         logging.error("Asegúrate de que 'data/cropped/not_me' no esté vacío.")
         return None, None
         
    return X, y

def preprocess_data(X_train, X_test):
    """Aplica StandardScaler a los datos."""
    
    # 1. Crear el Escalador
    scaler = StandardScaler()
    
    # 2. "Ajustar" (fit) el escalador SÓLO con los datos de entrenamiento
    #    El escalador aprende la media y la std dev del 80% (train)
    #    'fit_transform' hace dos cosas: aprende (fit) y transforma.
    logging.info("Ajustando StandardScaler a los datos de entrenamiento...")
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. "Transformar" (transform) los datos de prueba
    #    Usamos la media y std dev YA APRENDIDAS para escalar el 20% (test)
    #    ¡NUNCA hagas 'fit' en los datos de prueba! (sería "hacer trampa")
    logging.info("Transformando los datos de validación...")
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_classifier(X_train_scaled, y_train):
    """Entrena el clasificador de Regresión Logística."""
    
    # 'max_iter=200': Aumentamos las iteraciones por si no converge.
    # 'class_weight='balanced'': ¡MUY IMPORTANTE!
    #    Tenemos 40 "YO" (1) y 400 "NO-YO" (0). Los datos están
    #    desbalanceados 10:1. Si no hacemos esto, el modelo podría
    #    aprender a decir "siempre 0" y tener un 90% de accuracy.
    #    'balanced' ajusta los pesos para que "castigue" más
    #    los errores en la clase minoritaria (la clase "YO").
    logging.info("Entrenando el modelo LogisticRegression...")
    model = LogisticRegression(
        max_iter=200, 
        class_weight='balanced', 
        random_state=RANDOM_SEED,
        # Usamos 'liblinear' porque es bueno para datasets pequeños/medianos
        solver='liblinear' 
    )
    
    model.fit(X_train_scaled, y_train)
    logging.info("Entrenamiento completado.")
    return model

def evaluate_model(model, X_test_scaled, y_test):
    """Evalúa el modelo en el set de validación y devuelve las métricas."""
    
    logging.info("Evaluando el modelo en el set de validación...")
    
    # 1. Obtener predicciones (0 o 1)
    y_pred = model.predict(X_test_scaled)
    
    # 2. Obtener probabilidades (ej. 0.93 para la clase '1')
    #    'predict_proba' devuelve [prob_clase_0, prob_clase_1]
    #    Seleccionamos la columna [:, 1] que es la probabilidad de "YO".
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 3. Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    # AUC-ROC es una gran métrica para datos desbalanceados.
    # Mide qué tan bueno es el modelo para "rankear"
    # los positivos (YO) por encima de los negativos (NO-YO).
    # 1.0 es perfecto, 0.5 es aleatorio.
    auc = roc_auc_score(y_test, y_proba)
    
    logging.info(f"Accuracy en validación: {accuracy:.4f}")
    logging.info(f"AUC-ROC en validación: {auc:.4f}")
    
    # Reporte de clasificación (Precision, Recall, F1-score)
    report = classification_report(y_test, y_pred, target_names=["NO-YO (0)", "YO (1)"], output_dict=True)
    logging.info("Reporte de Clasificación:\n" + 
                 classification_report(y_test, y_pred, target_names=["NO-YO (0)", "YO (1)"]))

    metrics = {
        "accuracy": accuracy,
        "auc_roc": auc,
        "classification_report": report
    }
    return metrics

def save_artifacts(model, scaler, metrics):
    """Guarda el modelo, el escalador y las métricas en archivos."""
    
    # Asegurarse de que los directorios existan
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)
    
    # Guardar modelo y escalador con joblib (eficiente para numpy)
    logging.info(f"Guardando modelo en {MODEL_SAVE_PATH}")
    joblib.dump(model, MODEL_SAVE_PATH)
    
    logging.info(f"Guardando escalador en {SCALER_SAVE_PATH}")
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # Guardar métricas como un JSON legible
    logging.info(f"Guardando métricas en {METRICS_SAVE_PATH}")
    with open(METRICS_SAVE_PATH, 'w') as f:
        # 'indent=4' para que el JSON sea bonito
        json.dump(metrics, f, indent=4)

def main():
    """Función principal: orquesta todo el pipeline de entrenamiento."""
    logging.info("Iniciando el script de entrenamiento (H4)...")
    
    X, y = load_data(EMBEDDINGS_PATH, LABELS_PATH)
    if X is None or y is None:
        return # Error ya logueado en load_data

    # 1. Dividir los datos
    # 'stratify=y' es MUY importante para datos desbalanceados.
    # Asegura que la proporción de 1s y 0s sea la misma
    # en el set de 'train' y 'test'.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    logging.info(f"Datos divididos: {len(y_train)} para entrenamiento, {len(y_test)} para validación.")

    # 2. Preprocesar (Escalar)
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # 3. Entrenar
    model = train_classifier(X_train_scaled, y_train)

    # 4. Evaluar
    metrics = evaluate_model(model, X_test_scaled, y_test)

    # 5. Guardar
    save_artifacts(model, scaler, metrics)

    logging.info("--- Proceso de Entrenamiento Finalizado ---")
    logging.info(f"Modelo, escalador y métricas guardados en 'models/' y 'reports/'.")

if __name__ == "__main__":
    main()