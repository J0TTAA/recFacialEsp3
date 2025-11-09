# H10 · Informe Técnico del Modelo de Verificación Facial

## 1. Introducción

Este documento sintetiza el estado del sistema de verificación facial desarrollado en el proyecto `recFacialEsp3`. El informe cubre el conjunto de datos disponible, el pipeline de procesamiento y entrenamiento, el análisis del umbral de decisión, resultados recientes de evaluación, tiempo de inferencia y consideraciones éticas/legales. También se proponen mejoras técnicas y operativas de corto plazo. El objetivo es dejar una base sólida de información para redactar el informe formal (2–3 páginas) en un procesador de textos.

## 2. Datos disponibles

- **Origen**: capturas propias organizadas en dos carpetas (`data/me`, `data/not_me`).
- **Dataset crudo**:
  - `YO (1)`: 184 imágenes originales.
  - `NO-YO (0)`: 400 imágenes originales.
- **Dataset procesado** (tras `crop_faces.py` y `embeddings.py`):
  - Embeddings (`data/embeddings.npy`): 1815 vectores de 512 dimensiones.
  - Etiquetas (`data/labels.npy`): 1815 etiquetas (184 positivas, 1631 negativas).
  - Proporción de clases: 10.1 % positivos vs. 89.9 % negativos (fuerte desbalance).
- **Formato y control de calidad**:
  - Recortes centrados en el rostro (160x160 px) generados con MTCNN.
  - Normalización de color y alineamiento aplicado automáticamente.
  - No se versionan las imágenes con Git; sólo se guardan embeddings y artefactos derivados.

## 3. Pipeline de procesamiento y entrenamiento

1. **Detección y recorte de caras (`scripts/crop_faces.py`)**
   - Detector: MTCNN (`facenet-pytorch`), `image_size=160`, `margin=30`.
   - Salida: `data/cropped/me`, `data/cropped/not_me`, manteniendo la estructura original.

2. **Extracción de embeddings (`scripts/embeddings.py`)**
   - Modelo: `InceptionResnetV1` pre-entrenado en `vggface2`.
  - Preprocesamiento: conversión a tensor, normalización a rango `[-1, 1]`.
   - Procesamiento por lotes de 32 imágenes; uso automático de GPU si está disponible.
   - Salidas: `data/embeddings.npy`, `data/labels.npy`.

3. **Entrenamiento (`train.py`)**
   - División estratificada reproducible (`random_state=42`):
     - 70 % entrenamiento, 15 % validación, 15 % test final.
   - Escalado: `StandardScaler` ajustado con `X_train`, reutilizado para `X_val`, `X_test`.
   - Clasificador: `LogisticRegression` (`liblinear`), `class_weight='balanced'`, `C=0.001`, `max_iter=2000`, `tol=1e-6`.
   - Validación cruzada 5-fold durante el entrenamiento para monitorear sobreajuste.
   - Artefactos guardados:
     - `models/model.joblib`, `models/scaler.joblib`.
     - `reports/metrics.json` con métricas del split de test final.

4. **Evaluación (`evaluate.py`)**
   - Repite un split 80/20 estratificado (seed 42) a partir del dataset completo de embeddings.
   - Genera métricas, reporte de clasificación, matriz de confusión, curvas ROC y PR.
   - Analiza el umbral óptimo basándose en F1-score y en la estadística de Youden.
   - Produce `reports/evaluation_results.json` y gráficos (`confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`).

## 4. Análisis de umbral de decisión

- Umbral por defecto en producción (`VERIFY_THRESHOLD` en `.env`): `0.75`.
- Evaluación automática (`evaluate.py`, 2025-11-09):
  - `threshold = 0.75`: `accuracy = 0.9752`, `precision = 0.8043`, `recall = 1.0`, `F1 = 0.8916`.
  - Umbral óptimo según F1 (`0.9749`): `accuracy = 1.0`, `precision = 1.0`, `recall = 1.0`.
  - Con un umbral neutro `0.5`, la matriz de confusión muestra 55 falsos positivos (precisión muy baja para la clase positiva).
- Conclusiones:
  - El modelo separa bien las clases a nivel de probabilidades (`AUC-ROC = 1.0`), por lo que escoger el umbral adecuado es esencial.
  - `0.75` ofrece un balance entre seguridad (recall = 1.0) y practicidad (minimiza falsos positivos sin ser extremo).
  - `0.97` podría usarse en contextos de máxima seguridad, pero requiere confirmar que no degrade experiencia en escenarios reales no vistos.

## 5. Resultados de evaluación recientes

### 5.1 Split de test (entrenamiento)

> Fuente: `reports/metrics.json`

- Accuracy: 0.828
- AUC-ROC: 1.000
- Clase `YO (1)`:
  - Precision: 0.373
  - Recall: 1.000
  - F1: 0.544
- Clase `NO-YO (0)`:
  - Precision: 1.000
  - Recall: 0.808
  - F1: 0.894

### 5.2 Evaluación 80/20 (2025-11-09)

> Fuente: `reports/evaluation_results.json`, re-generado antes de este informe.

- Accuracy (umbral 0.5 del modelo): 0.848
- AUC-ROC: 1.000
- AUC-PR: ≈1.000
- Reporte de clasificación (umbral 0.5):
  - `YO (1)`: Precision 0.402, Recall 1.000, F1 0.574 (37 positivos, 55 falsos positivos).
  - `NO-YO (0)`: Precision 1.000, Recall 0.831, F1 0.908.
- Matriz de confusión (umbral 0.5):
  - TN: 271, FP: 55, FN: 0, TP: 37.
- Tras recalibrar umbral a 0.75:
  - TN: 317, FP: 9, FN: 0, TP: 37.
  - Accuracy: 0.975, Precision: 0.804, Recall: 1.000, F1: 0.892.
- Curvas ROC y PR evidencian separación casi perfecta entre clases.

## 6. Latencia de inferencia

- Medición (2025-11-09) con `joblib` + `scikit-learn` en CPU (Windows 10, Intel Core i7, Python launcher `py`):
  - Tamaño del lote probado: 128 embeddings.
  - 100 iteraciones consecutivas para amortiguar costos de llamada.
  - Tiempo total: 0.0195 s.
  - Latencia promedio por lote (128 muestras): 0.20 ms.
  - Latencia promedio por muestra: 0.002 ms.
- Interpretación:
  - El clasificador es extremadamente ligero; el cuello de botella real de la API está en la detección de rostro (MTCNN) y la extracción de embeddings (FaceNet), no en la regresión logística.
  - Para uso interactivo, la latencia se percibe casi instantánea una vez que se dispone del embedding.

## 7. Consideraciones éticas, legales y de privacidad

- **Privacidad de datos**: las imágenes crudas contienen información biométrica sensible. Se recomienda:
  - Almacenar las fotos originales fuera de repositorios compartidos (actualmente se cumple).
  - Cifrar respaldos o unidades donde residan las imágenes.
  - Definir una política de retención y eliminación segura.
- **Consentimiento**: deben documentarse los consentimientos explícitos de cualquier persona incluida en `data/not_me`.
- **Uso indebido**: implementar controles de acceso a la API y auditoría de llamadas para evitar abusos.
- **Cumplimiento normativo**:
  - GDPR / LGPD / otras leyes locales pueden aplicar si el sistema se despliega en producción con datos de terceros.
  - Establecer procedimientos para atender solicitudes de acceso o eliminación de datos biométricos.
- **Sesgos y equidad**:
  - Dataset altamente desbalanceado; existe riesgo de sesgo hacia falsos negativos cuando se agreguen nuevas clases de impostores no representados.
  - Se recomienda monitorear métricas diferenciadas por grupos demográficos si el sistema se escala.
- **Transparencia**:
  - Documentar claramente al usuario final cómo se usan sus datos y con qué fines.
  - Incluir mecanismos de revocación del consentimiento.

## 8. Mejoras recomendadas

1. **Ampliar el dataset**:
   - Capturar más imágenes propias y de impostores, con variaciones de iluminación, ángulos y expresiones.
   - Aumentar la proporción de positivos (actualmente solo 10 %).

2. **Medición sistemática de latencia end-to-end**:
   - Incluir timers en la API (`image_processor` y `model_loader`) para reportar tiempos de detección + embedding + clasificación.
   - Guardar estadísticas en logs o en un dashboard.

3. **Gestión de umbral dinámica**:
   - Implementar una rutina que evalúe el rendimiento con datos nuevos y recomiende ajustes automáticos del umbral (p. ej., mantener FP < 2 %).

4. **Regularización y calibración**:
   - Probar técnicas de calibración de probabilidades (Platt scaling, isotonic regression) para mejorar correspondencia score/confianza.

5. **Hardening de la API**:
   - Añadir autenticación y rate limiting.
   - Incorporar almacenamiento temporal seguro (p. ej., borrar archivos en `uploads/` tras cada inferencia).

6. **Pruebas adicionales**:
   - Tests unitarios para el pipeline de preprocesamiento y para la función de inferencia.
   - Pruebas de integración que simulen llamadas reales al endpoint `/verify`.

7. **Monitoreo de calidad en producción**:
   - Registrar métricas de falsos positivos/negativos a lo largo del tiempo.
   - Configurar alertas cuando la precisión cae por debajo de un umbral.

8. **Evaluación con datos externos**:
   - Tomar un dataset externo (p. ej., LFW) con clases “pinchadas” para validar generalización.

## 9. Conclusiones

- El modelo actual ofrece una discriminación muy alta entre las dos clases cuando se utilizan embeddings de FaceNet, pero la decisión final depende críticamente de elegir un umbral conservador.
- La latencia de la etapa de clasificación es despreciable; optimizaciones deben enfocarse en detección/embedding.
- El pipeline es reproducible y los artefactos están organizados en `models/` y `reports/`, permitiendo auditorías y repeticiones.
- Es fundamental reforzar la gobernanza de los datos biométricos (consentimiento, cifrado, políticas) antes de desplegar el sistema con terceros.
- Las mejoras propuestas (más datos, calibración, métricas operativas) ayudarán a reducir los riesgos de sobreajuste y a robustecer el sistema para entornos reales.

---

**Fecha de actualización:** 2025-11-09  
**Scripts clave:** `scripts/crop_faces.py`, `scripts/embeddings.py`, `train.py`, `evaluate.py`  
**Responsable:** Equipo Técnico del proyecto `recFacialEsp3`


