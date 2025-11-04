import os
import numpy as np
import joblib
import json
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Configuración ---
EMBEDDINGS_PATH = "data/embeddings.npy"
LABELS_PATH = "data/labels.npy"
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
REPORTS_DIR = "reports"
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.2

def load_data_and_model():
    """Carga los datos y el modelo entrenado."""
    logging.info("Cargando datos y modelo...")
    
    # Cargar embeddings y etiquetas
    try:
        X = np.load(EMBEDDINGS_PATH)
        y = np.load(LABELS_PATH)
    except FileNotFoundError as e:
        logging.error(f"Error: Archivo no encontrado. {e}")
        logging.error("Asegúrate de ejecutar 'scripts/embeddings.py' primero.")
        return None, None, None, None
    
    # Cargar modelo y scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        logging.error(f"Error: Modelo no encontrado. {e}")
        logging.error("Asegúrate de ejecutar 'train.py' primero.")
        return None, None, None, None
    
    logging.info(f"Datos cargados: {X.shape[0]} muestras")
    return X, y, model, scaler

def prepare_data(X, y, scaler):
    """Prepara los datos usando el mismo split que en train.py."""
    # Usar el mismo random_state para reproducibilidad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    
    # Escalar usando el scaler ya entrenado
    X_test_scaled = scaler.transform(X_test)
    
    return X_test_scaled, y_test

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Genera y guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    # Etiquetas
    classes = ['NO-YO (0)', 'YO (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Agregar valores en la matriz
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight='bold')
    
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Matriz de confusión guardada en {save_path}")
    return cm

def plot_roc_curve(y_true, y_proba, save_path):
    """Genera y guarda la curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Curva ROC guardada en {save_path}")
    logging.info(f"AUC-ROC: {roc_auc:.4f}")
    return roc_auc, fpr, tpr, thresholds

def plot_pr_curve(y_true, y_proba, save_path):
    """Genera y guarda la curva Precision-Recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Curva Precision-Recall', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Curva PR guardada en {save_path}")
    logging.info(f"AUC-PR: {pr_auc:.4f}")
    return pr_auc, precision, recall, thresholds

def find_optimal_threshold(y_true, y_proba):
    """
    Busca el umbral óptimo τ que maximiza el F1-score.
    También puede considerar otros criterios como Youden's J statistic.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calcular F1-score para cada umbral
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    # Encontrar el umbral que maximiza F1
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # También calcular Youden's J (maximiza TPR - FPR)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    best_youden_idx = np.argmax(youden_j)
    optimal_threshold_youden = roc_thresholds[best_youden_idx]
    
    logging.info(f"Umbral óptimo (F1-max): {optimal_threshold:.4f} (F1={best_f1:.4f})")
    logging.info(f"Umbral óptimo (Youden's J): {optimal_threshold_youden:.4f}")
    
    return {
        'optimal_threshold_f1': float(optimal_threshold),
        'optimal_f1_score': float(best_f1),
        'optimal_threshold_youden': float(optimal_threshold_youden),
        'thresholds': thresholds.tolist(),
        'f1_scores': [float(f) for f in f1_scores]
    }

def evaluate_with_threshold(y_true, y_proba, threshold):
    """Evalúa el modelo con un umbral específico."""
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }

def main():
    """Función principal: realiza la evaluación completa."""
    logging.info("Iniciando evaluación del modelo (H5)...")
    
    # Asegurar que el directorio de reportes existe
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Cargar datos y modelo
    X, y, model, scaler = load_data_and_model()
    if X is None:
        return
    
    # Preparar datos de test
    X_test_scaled, y_test = prepare_data(X, y, scaler)
    logging.info(f"Evaluando en {len(y_test)} muestras de test")
    
    # Obtener probabilidades
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # 1. Matriz de confusión
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    cm = plot_confusion_matrix(y_test, y_pred, cm_path)
    
    # 2. Curva ROC
    roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    roc_auc, fpr, tpr, roc_thresholds = plot_roc_curve(y_test, y_proba, roc_path)
    
    # 3. Curva PR
    pr_path = os.path.join(REPORTS_DIR, "pr_curve.png")
    pr_auc, precision, recall, pr_thresholds = plot_pr_curve(y_test, y_proba, pr_path)
    
    # 4. Búsqueda de umbral óptimo τ
    threshold_info = find_optimal_threshold(y_test, y_proba)
    
    # 5. Evaluar con diferentes umbrales
    default_threshold = 0.75
    optimal_threshold_f1 = threshold_info['optimal_threshold_f1']
    
    metrics_default = evaluate_with_threshold(y_test, y_proba, default_threshold)
    metrics_optimal = evaluate_with_threshold(y_test, y_proba, optimal_threshold_f1)
    
    # 6. Reporte de clasificación
    report = classification_report(y_test, y_pred, target_names=["NO-YO (0)", "YO (1)"], output_dict=True)
    
    # 7. Compilar todas las métricas
    evaluation_results = {
        'model_metrics': {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'classification_report': report
        },
        'threshold_analysis': {
            'default_threshold': default_threshold,
            'optimal_threshold_f1': optimal_threshold_f1,
            'optimal_f1_score': threshold_info['optimal_f1_score'],
            'metrics_with_default': metrics_default,
            'metrics_with_optimal': metrics_optimal
        },
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        },
        'files_generated': {
            'confusion_matrix': 'confusion_matrix.png',
            'roc_curve': 'roc_curve.png',
            'pr_curve': 'pr_curve.png'
        }
    }
    
    # 8. Guardar resultados en JSON
    results_path = os.path.join(REPORTS_DIR, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Resultados de evaluación guardados en {results_path}")
    
    # 9. Resumen en consola
    logging.info("=" * 60)
    logging.info("RESUMEN DE EVALUACIÓN")
    logging.info("=" * 60)
    logging.info(f"AUC-ROC: {roc_auc:.4f}")
    logging.info(f"AUC-PR: {pr_auc:.4f}")
    logging.info(f"Umbral por defecto (0.75):")
    logging.info(f"  - Accuracy: {metrics_default['accuracy']:.4f}")
    logging.info(f"  - Precision: {metrics_default['precision']:.4f}")
    logging.info(f"  - Recall: {metrics_default['recall']:.4f}")
    logging.info(f"  - F1-score: {metrics_default['f1_score']:.4f}")
    logging.info(f"Umbral óptimo (F1-max, {optimal_threshold_f1:.4f}):")
    logging.info(f"  - Accuracy: {metrics_optimal['accuracy']:.4f}")
    logging.info(f"  - Precision: {metrics_optimal['precision']:.4f}")
    logging.info(f"  - Recall: {metrics_optimal['recall']:.4f}")
    logging.info(f"  - F1-score: {metrics_optimal['f1_score']:.4f}")
    logging.info("=" * 60)
    
    logging.info("--- Evaluación completada ---")
    logging.info(f"Gráficos guardados en: {REPORTS_DIR}/")
    logging.info(f"Resultados guardados en: {results_path}")

if __name__ == "__main__":
    main()

