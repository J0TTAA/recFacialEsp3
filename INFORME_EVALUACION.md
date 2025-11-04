# Informe de Evaluación del Modelo de Reconocimiento Facial

## Librerías Principales Utilizadas

### 1. **facenet-pytorch** ⭐ (Principal)
- **Propósito**: Reconocimiento facial
- **Componentes**:
  - **MTCNN**: Detección y alineamiento de caras
  - **InceptionResnetV1**: Extracción de características (embeddings de 512 dimensiones)
- **Pre-entrenado en**: VGGFace2 (millones de rostros)
- **Repositorio**: https://github.com/timesler/facenet-pytorch

### 2. **scikit-learn**
- **Propósito**: Machine Learning tradicional
- **Componentes usados**:
  - `LogisticRegression`: Clasificador binario
  - `StandardScaler`: Normalización de datos
  - Métricas: `confusion_matrix`, `roc_curve`, `precision_recall_curve`, `f1_score`, `classification_report`

### 3. **PyTorch (torch)**
- **Propósito**: Framework de deep learning
- **Uso**: Base para FaceNet y MTCNN

### 4. **matplotlib**
- **Propósito**: Visualización
- **Uso**: Generación de gráficos (matriz de confusión, curvas ROC y PR)

---

## Resumen Ejecutivo de la Evaluación

### Dataset
- **Total de muestras**: 584 imágenes
- **Clase "me" (YO)**: 184 imágenes (31.5%)
- **Clase "not_me" (NO-YO)**: 400 imágenes (68.5%)
- **División train/test**: 80% entrenamiento (467) / 20% validación (117)
- **Muestras de test**: 117 (80 "not_me", 37 "me")

### Métricas Generales

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 1.0000 (100%) |
| **AUC-ROC** | 1.0000 |
| **AUC-PR** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1-Score** | 1.0000 |

**Conclusión**: El modelo muestra un rendimiento perfecto en el conjunto de validación, clasificando correctamente todas las muestras.

---

## Matriz de Confusión

### Resultados

```
                Predicción
              NO-YO    YO
Real  NO-YO    80      0
      YO       0      37
```

### Interpretación

| Métrica | Valor | Significado |
|---------|-------|-------------|
| **True Negatives (TN)** | 80 | Correctamente identificados como "not_me" |
| **False Positives (FP)** | 0 | Incorrectamente identificados como "me" |
| **False Negatives (FN)** | 0 | Incorrectamente identificados como "not_me" |
| **True Positives (TP)** | 37 | Correctamente identificados como "me" |

### Análisis
- **0 Falsos Positivos**: No hay casos donde el modelo identifique incorrectamente a otra persona como "me"
- **0 Falsos Negativos**: No hay casos donde el modelo falle en identificar al usuario correcto
- **100% de precisión**: Todas las clasificaciones son correctas

**Archivo generado**: `reports/confusion_matrix.png`

---

## Curva ROC (Receiver Operating Characteristic)

### Métrica: AUC-ROC = 1.0000

### Interpretación

La curva ROC mide la capacidad del modelo para distinguir entre las dos clases:
- **AUC = 1.0**: Clasificación perfecta
- **AUC = 0.5**: Clasificación aleatoria (peor caso)
- **AUC > 0.9**: Excelente clasificación

### Componentes de la Curva

- **Eje X (FPR)**: Tasa de Falsos Positivos
  - FPR = FP / (FP + TN)
  - En nuestro caso: 0 / (0 + 80) = 0.0

- **Eje Y (TPR)**: Tasa de Verdaderos Positivos (Recall)
  - TPR = TP / (TP + FN)
  - En nuestro caso: 37 / (37 + 0) = 1.0

### Análisis
- El modelo alcanza **TPR = 1.0** (identifica todos los casos positivos)
- Mantiene **FPR = 0.0** (no genera falsos positivos)
- La curva está en el punto óptimo (0, 1) del espacio ROC

**Archivo generado**: `reports/roc_curve.png`

---

## Curva Precision-Recall (PR)

### Métrica: AUC-PR = 1.0000

### Interpretación

La curva PR es especialmente útil para datos desbalanceados:
- **AUC-PR = 1.0**: Clasificación perfecta
- **AUC-PR cercano a 1.0**: Excelente rendimiento

### Componentes de la Curva

- **Precision**: De todas las predicciones positivas, ¿cuántas son correctas?
  - Precision = TP / (TP + FP) = 37 / (37 + 0) = 1.0

- **Recall**: De todos los casos positivos reales, ¿cuántos identificamos?
  - Recall = TP / (TP + FN) = 37 / (37 + 0) = 1.0

### Análisis
- **Precision = 1.0**: Todas las predicciones de "me" son correctas
- **Recall = 1.0**: Identificamos todos los casos reales de "me"
- Balance perfecto entre precisión y cobertura

**Archivo generado**: `reports/pr_curve.png`

---

## Análisis de Umbral Óptimo (τ)

### Métodos de Búsqueda

#### 1. Maximización de F1-Score
- **Umbral óptimo (F1-max)**: 0.99999
- **F1-Score alcanzado**: 1.0000

**Método**: Evalúa todos los posibles umbrales y selecciona el que maximiza el F1-score.

#### 2. Youden's J Statistic
- **Umbral óptimo (Youden)**: 1.0000
- **Fórmula**: J = TPR - FPR

**Método**: Maximiza la diferencia entre TPR y FPR, encontrando el punto óptimo en la curva ROC.

### Comparación de Umbrales

| Umbral | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **0.75 (default)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **0.99999 (óptimo F1)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Análisis

- Ambos umbrales (default y óptimo) producen resultados idénticos
- Esto indica que el modelo es muy robusto y no depende críticamente del umbral
- El umbral óptimo es muy cercano a 1.0, sugiriendo alta confianza en las predicciones

### Recomendación

Para producción, se puede usar:
- **Umbral = 0.75**: Más permisivo, permite mayor flexibilidad
- **Umbral = 0.99999**: Más estricto, requiere mayor confianza

Dado que ambos producen resultados perfectos, se recomienda **0.75** para mayor robustez en condiciones reales.

---

## Reporte de Clasificación Detallado

### Por Clase

#### Clase "NO-YO (0)" - Otras Personas
- **Precision**: 1.0000 (100%)
- **Recall**: 1.0000 (100%)
- **F1-Score**: 1.0000
- **Support**: 80 muestras

**Interpretación**: El modelo identifica perfectamente a todas las personas que NO son el usuario.

#### Clase "YO (1)" - Usuario
- **Precision**: 1.0000 (100%)
- **Recall**: 1.0000 (100%)
- **F1-Score**: 1.0000
- **Support**: 37 muestras

**Interpretación**: El modelo identifica perfectamente al usuario en todas las situaciones.

### Promedios

| Métrica | Macro Avg | Weighted Avg |
|---------|-----------|--------------|
| **Precision** | 1.0000 | 1.0000 |
| **Recall** | 1.0000 | 1.0000 |
| **F1-Score** | 1.0000 | 1.0000 |
| **Support** | 117 | 117 |

**Nota**: 
- **Macro Avg**: Promedio simple de ambas clases
- **Weighted Avg**: Promedio ponderado por el número de muestras

---

## Archivos Generados

### Gráficos

1. **`reports/confusion_matrix.png`** (96 KB)
   - Matriz de confusión visual
   - Resolución: 300 DPI
   - Formato: PNG

2. **`reports/roc_curve.png`** (151 KB)
   - Curva ROC con AUC
   - Línea de referencia (clasificador aleatorio)
   - Resolución: 300 DPI

3. **`reports/pr_curve.png`** (80 KB)
   - Curva Precision-Recall con AUC
   - Resolución: 300 DPI

### Datos

4. **`reports/evaluation_results.json`** (2 KB)
   - Todas las métricas en formato JSON
   - Incluye análisis de umbrales
   - Matriz de confusión numérica

5. **`reports/metrics.json`** (729 bytes)
   - Métricas básicas del entrenamiento
   - Generado durante `train.py`

---

## Métricas Técnicas Detalladas

### Accuracy (Precisión General)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = (37 + 80) / (37 + 80 + 0 + 0)
         = 117 / 117
         = 1.0000 (100%)
```

### Precision (Precisión)
```
Precision = TP / (TP + FP)
          = 37 / (37 + 0)
          = 1.0000 (100%)
```

### Recall (Sensibilidad)
```
Recall = TP / (TP + FN)
       = 37 / (37 + 0)
       = 1.0000 (100%)
```

### F1-Score (Balance Precision-Recall)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (1.0 × 1.0) / (1.0 + 1.0)
   = 2 × 1.0 / 2.0
   = 1.0000
```

### Especificidad
```
Specificity = TN / (TN + FP)
            = 80 / (80 + 0)
            = 1.0000 (100%)
```

---

## Interpretación de Resultados

### Fortalezas del Modelo

1. **Clasificación Perfecta**: 100% de accuracy en el conjunto de validación
2. **Sin Errores**: 0 falsos positivos y 0 falsos negativos
3. **Balanceado**: Excelente rendimiento en ambas clases
4. **Robusto**: No depende críticamente del umbral seleccionado

### Posibles Consideraciones

1. **Overfitting Potencial**: Resultados perfectos pueden indicar sobreajuste
   - **Recomendación**: Probar con datos nuevos no vistos durante el entrenamiento

2. **Dataset Limitado**: 117 muestras de validación es relativamente pequeño
   - **Recomendación**: Validar con más datos en condiciones reales

3. **Condiciones Controladas**: El modelo fue entrenado con condiciones similares
   - **Recomendación**: Probar con diferentes:
     - Iluminaciones
     - Ángulos de cámara
     - Expresiones faciales
     - Accesorios (gafas, barba, etc.)

---

## Métodos de Evaluación Implementados

### 1. Matriz de Confusión
- **Función**: `plot_confusion_matrix()`
- **Librería**: `sklearn.metrics.confusion_matrix`
- **Visualización**: Heatmap con matplotlib

### 2. Curva ROC
- **Función**: `plot_roc_curve()`
- **Librería**: `sklearn.metrics.roc_curve`, `sklearn.metrics.auc`
- **Métrica calculada**: AUC-ROC

### 3. Curva Precision-Recall
- **Función**: `plot_pr_curve()`
- **Librería**: `sklearn.metrics.precision_recall_curve`, `sklearn.metrics.auc`
- **Métrica calculada**: AUC-PR

### 4. Búsqueda de Umbral Óptimo
- **Función**: `find_optimal_threshold()`
- **Métodos**:
  - Maximización de F1-Score
  - Youden's J Statistic
- **Librería**: `sklearn.metrics.f1_score`, `sklearn.metrics.roc_curve`

### 5. Evaluación con Umbrales Específicos
- **Función**: `evaluate_with_threshold()`
- **Umbrales evaluados**:
  - Default: 0.75
  - Óptimo F1: 0.99999

---

## Código de Evaluación

### Estructura del Script `evaluate.py`

```python
# 1. Carga de datos y modelo
X, y, model, scaler = load_data_and_model()

# 2. Preparación de datos de test
X_test_scaled, y_test = prepare_data(X, y, scaler)

# 3. Generación de predicciones y probabilidades
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# 4. Generación de gráficos
- Matriz de confusión
- Curva ROC
- Curva PR

# 5. Análisis de umbral óptimo
optimal_threshold = find_optimal_threshold(y_test, y_proba)

# 6. Evaluación con diferentes umbrales
metrics_default = evaluate_with_threshold(y_test, y_proba, 0.75)
metrics_optimal = evaluate_with_threshold(y_test, y_proba, optimal_threshold)

# 7. Guardado de resultados
- evaluation_results.json
- Gráficos PNG
```

---

## Conclusiones

### Rendimiento del Modelo

El modelo de reconocimiento facial muestra un **rendimiento excepcional** en el conjunto de validación, alcanzando:

- ✅ **100% de accuracy**
- ✅ **AUC-ROC perfecto (1.0)**
- ✅ **AUC-PR perfecto (1.0)**
- ✅ **0 errores de clasificación**

### Recomendaciones

1. **Validación Externa**: Probar con datos completamente nuevos no utilizados en entrenamiento
2. **Pruebas en Condiciones Reales**: Validar con diferentes:
   - Condiciones de iluminación
   - Ángulos de captura
   - Calidad de imagen
   - Expresiones faciales variadas
3. **Monitoreo Continuo**: Implementar logging para rastrear el rendimiento en producción
4. **Ajuste de Umbral**: Considerar ajustar el umbral según el caso de uso:
   - **Seguridad alta**: Umbral más estricto (0.9-0.95)
   - **Usabilidad**: Umbral más permisivo (0.7-0.75)

### Próximos Pasos

1. Desplegar el modelo en producción
2. Monitorear métricas en tiempo real
3. Recolectar datos de producción para validación continua
4. Considerar retraining periódico con nuevos datos

---

## Referencias Técnicas

### Librerías

- **facenet-pytorch**: https://github.com/timesler/facenet-pytorch
- **scikit-learn**: https://scikit-learn.org/
- **PyTorch**: https://pytorch.org/
- **matplotlib**: https://matplotlib.org/

### Papers Relacionados

- **FaceNet**: Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. CVPR 2015.
- **MTCNN**: Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters.

---

**Fecha de evaluación**: 2025-11-04  
**Versión del modelo**: me-verifier-v1  
**Script de evaluación**: `evaluate.py`

