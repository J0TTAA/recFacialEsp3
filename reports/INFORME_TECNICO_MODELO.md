# Informe Técnico del Modelo de Verificación Facial

## Resumen ejecutivo

- Objetivo: verificación binaria (`YO` vs `NO-YO`) a partir de embeddings faciales de 512 dimensiones generados con FaceNet.
- Última evaluación (`evaluate.py`, ejecutado el 2025-11-09) sobre un split de prueba estratificado del 20 % (`n=363`): `accuracy = 0.8485`, `AUC-ROC = 1.0`, `AUC-PR = 1.0`.
- Ajustando el umbral de decisión a `0.9749` (máximo F1) se obtiene `accuracy = 1.0`, `precision = 1.0`, `recall = 1.0`. Con el umbral operativo actual (`0.75`) las métricas son `accuracy = 0.9752`, `precision = 0.8043`, `recall = 1.0`.
- El modelo y el `StandardScaler` se almacenan en `models/model.joblib` y `models/scaler.joblib`. Los artefactos de evaluación se encuentran en `reports/`.

## Datos y preprocesamiento

- Conjunto original de imágenes etiquetadas:
  - `data/me`: fotos del usuario (`YO`, etiqueta `1`).
  - `data/not_me`: fotos de impostores (`NO-YO`, etiqueta `0`).
- **Recorte y alineamiento** (`scripts/crop_faces.py`):
  - Detector MTCNN (`facenet-pytorch`), `image_size=160`, `margin=30`, `keep_all=False`.
  - Salida en `data/cropped/me` y `data/cropped/not_me`.
- **Extracción de embeddings** (`scripts/embeddings.py`):
  - `InceptionResnetV1` pre-entrenado en `vggface2`.
  - Normalización `[-1, 1]` y procesamiento por lotes (`batch_size=32`).
  - Resultado: `data/embeddings.npy` (matriz `1815 x 512`) y `data/labels.npy`.
  - Distribución de clases en embeddings:
    - `YO (1)`: 184 muestras (10.1 %).
    - `NO-YO (0)`: 1631 muestras (89.9 %).

## Configuración de entrenamiento (`train.py`)

- División estratificada: 70 % entrenamiento, 15 % validación, 15 % test final (`random_state=42`).
- Escalado:
  - `StandardScaler` ajustado con `X_train` y aplicado a `X_val`, `X_test`.
- Clasificador:
  - `LogisticRegression` (`scikit-learn`), `solver='liblinear'`, `max_iter=2000`, `class_weight='balanced'`, `C=0.001`, `penalty='l2'`, `tol=1e-6`.
  - Validación cruzada 5-fold sobre `X_train` para control de sobreajuste.
- Métricas guardadas en `reports/metrics.json` (sobre el split de test final):
  - `accuracy = 0.8278`.
  - `AUC-ROC = 1.0`.
  - Informe de clasificación:

```0:15:reports/metrics.json
{
    "accuracy": 0.8278388278388278,
    "auc_roc": 1.0,
    "classification_report": {
        "NO-YO (0)": {
            "precision": 1.0,
            "recall": 0.8081632653061225,
            "f1-score": 0.8939051918735892,
            "support": 245.0
        },
        "YO (1)": {
            "precision": 0.37333333333333335,
            "recall": 1.0,
            "f1-score": 0.5436893203883495,
            "support": 28.0
        },
        "accuracy": 0.8278388278388278,
        "macro avg": {
            "precision": 0.6866666666666666,
            "recall": 0.9040816326530612,
            "f1-score": 0.7187972561309693,
            "support": 273.0
        },
        "weighted avg": {
            "precision": 0.9357264957264957,
            "recall": 0.8278388278388278,
            "f1-score": 0.8579856153110006,
            "support": 273.0
        }
    }
}
```

## Evaluación más reciente (`evaluate.py`)

- `evaluate.py` reproduce el split 80/20 (estratificado, `random_state=42`) y reutiliza el `StandardScaler` guardado.
- Resultados agregados (`reports/evaluation_results.json`):

```0:74:reports/evaluation_results.json
{
    "model_metrics": {
        "roc_auc": 1.0,
        "pr_auc": 0.9999999999999999,
        "classification_report": {
            "NO-YO (0)": {
                "precision": 1.0,
                "recall": 0.8312883435582822,
                "f1-score": 0.9078726968174204,
                "support": 326.0
            },
            "YO (1)": {
                "precision": 0.40217391304347827,
                "recall": 1.0,
                "f1-score": 0.5736434108527132,
                "support": 37.0
            },
            "accuracy": 0.8484848484848485,
            "macro avg": {
                "precision": 0.7010869565217391,
                "recall": 0.915644171779141,
                "f1-score": 0.7407580538350669,
                "support": 363.0
            },
            "weighted avg": {
                "precision": 0.939064558629776,
                "recall": 0.8484848484848485,
                "f1-score": 0.8738052489367203,
                "support": 363.0
            }
        }
    },
    "threshold_analysis": {
        "default_threshold": 0.75,
        "optimal_threshold_f1": 0.9748981636924129,
        "optimal_f1_score": 1.0,
        "metrics_with_default": {
            "threshold": 0.75,
            "accuracy": 0.9752066115702479,
            "precision": 0.8043478260869565,
            "recall": 1.0,
            "f1_score": 0.891566265060241,
            "confusion_matrix": {
                "tn": 317,
                "fp": 9,
                "fn": 0,
                "tp": 37
            }
        },
        "metrics_with_optimal": {
            "threshold": 0.9748981636924129,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "confusion_matrix": {
                "tn": 326,
                "fp": 0,
                "fn": 0,
                "tp": 37
            }
        }
    },
    "confusion_matrix": {
        "tn": 271,
        "fp": 55,
        "fn": 0,
        "tp": 37
    },
    "files_generated": {
        "confusion_matrix": "confusion_matrix.png",
        "roc_curve": "roc_curve.png",
        "pr_curve": "pr_curve.png"
    }
}
```

- Interpretación:
  - Con el umbral por defecto de `LogisticRegression` (0.5) el modelo incurre en 55 falsos positivos (`precision` limitada en la clase `YO`).
  - Ajustar el umbral operativo a `0.75` mejora significativamente la precisión manteniendo `recall = 1.0`.
  - El umbral óptimo calculado (`0.9749`) equilibra completamente la matriz de confusión en este split, aunque puede ser demasiado estricto en producción; conviene validarlo con datos externos.
  - El `AUC-ROC` y `AUC-PR` iguales a 1 indican una separación casi perfecta entre clases a nivel de probabilidades, por lo que el umbral es la principal palanca de rendimiento.

## Artefactos disponibles

- Modelos: `models/model.joblib`, `models/scaler.joblib`.
- Datos derivados: `data/embeddings.npy`, `data/labels.npy`.
- Reportes:
  - `reports/evaluation_results.json` (métricas completas, análisis de umbral).
  - `reports/metrics.json` (métricas del split de test al entrenar).
  - `reports/confusion_matrix.png`, `reports/roc_curve.png`, `reports/pr_curve.png`.
- Documentación relacionada: `INFORME_EVALUACION.md`, `MEJORAR_MODELO.md`, `SOLUCION_INMEDIATA.md`.

## Riesgos y recomendaciones

- **Desequilibrio severo de clases**: aunque se usa `class_weight='balanced'`, la precisión para la clase positiva depende en gran medida del umbral. Recolectar más imágenes propias (`YO`) ayudaría a estabilizar el modelo.
- **Posible sobreajuste**: métricas perfectas tras ajustar el umbral sugieren que el dataset de evaluación puede compartir condiciones con el entrenamiento. Validar con imágenes capturadas en entornos diferentes (iluminación, ángulos, accesorios).
- **Gestión de datos**: las imágenes crudas no están versionadas (correcto para repositorios Git). Documentar la procedencia y mantener un procedimiento reproducible para regenerar embeddings.
- **Monitoreo en producción**: registrar tasas de falsos positivos/negativos e introducir alertas si se degradan.
- **Automatización**: crear scripts para regenerar métricas tras nuevos entrenamientos y consolidar los reportes (p. ej., `make evaluate`).

## Reproducibilidad rápida

```bash
# 1. (Opcional) recrear entorno
python -m venv venv
venv/Scripts/pip install -r requirements.txt  # en Windows

# 2. Generar dataset procesado
python scripts/crop_faces.py
python scripts/embeddings.py

# 3. Entrenar
python train.py

# 4. Evaluar y generar reportes
python evaluate.py
```

> Nota: en Windows, reemplazar `python` por `py` si se usa el lanzador oficial.


