# Librerías del Proyecto de Reconocimiento Facial

## Librería Principal: facenet-pytorch ⭐

### ¿Qué es facenet-pytorch?

**facenet-pytorch** es la librería principal que hace el reconocimiento facial. Es una implementación en PyTorch de FaceNet, un modelo de deep learning desarrollado por Google.

### Componentes Incluidos

#### 1. **MTCNN** (Multi-task Cascaded Convolutional Networks)
- **Propósito**: Detección y alineamiento de caras
- **Función en el proyecto**:
  - Detecta rostros en imágenes
  - Recorta y alinea las caras
  - Redimensiona a 160x160 píxeles
- **Uso**: `from facenet_pytorch import MTCNN`

#### 2. **InceptionResnetV1** (FaceNet)
- **Propósito**: Extracción de características faciales (embeddings)
- **Función en el proyecto**:
  - Convierte caras en vectores de 512 dimensiones
  - Pre-entrenado en VGGFace2 (millones de rostros)
- **Uso**: `from facenet_pytorch import InceptionResnetV1`

### Repositorio
- **GitHub**: https://github.com/timesler/facenet-pytorch
- **Documentación**: https://github.com/timesler/facenet-pytorch#usage

### Instalación
```bash
pip install facenet-pytorch
```

---

## Otras Librerías Importantes

### 2. scikit-learn
**Versión**: 1.7.2  
**Propósito**: Machine Learning tradicional

**Componentes usados**:
- `LogisticRegression`: Clasificador binario
- `StandardScaler`: Normalización de datos
- `train_test_split`: División de datos
- `confusion_matrix`: Matriz de confusión
- `roc_curve`, `auc`: Curvas ROC
- `precision_recall_curve`: Curvas Precision-Recall
- `f1_score`: Score F1
- `classification_report`: Reporte de clasificación

**Documentación**: https://scikit-learn.org/

---

### 3. PyTorch (torch)
**Versión**: 2.2.2  
**Propósito**: Framework de deep learning

**Uso en el proyecto**:
- Base para FaceNet y MTCNN
- Procesamiento de tensores
- GPU/CPU automático

**Documentación**: https://pytorch.org/

---

### 4. matplotlib
**Versión**: 3.10.7  
**Propósito**: Visualización de gráficos

**Uso en el proyecto**:
- Matriz de confusión
- Curvas ROC
- Curvas Precision-Recall

**Documentación**: https://matplotlib.org/

---

### 5. Flask
**Versión**: 3.1.2  
**Propósito**: Framework web para la API REST

**Componentes usados**:
- `Flask`: Aplicación principal
- `flask-cors`: Soporte CORS
- `Blueprint`: Organización modular

**Documentación**: https://flask.palletsprojects.com/

---

### 6. Pillow (PIL)
**Versión**: 10.2.0  
**Propósito**: Procesamiento de imágenes

**Uso en el proyecto**:
- Carga de imágenes
- Corrección de orientación EXIF
- Conversión de formatos
- Redimensionamiento

**Documentación**: https://pillow.readthedocs.io/

---

### 7. NumPy
**Versión**: 1.26.4  
**Propósito**: Operaciones numéricas

**Uso en el proyecto**:
- Manipulación de arrays
- Almacenamiento de embeddings (.npy)
- Operaciones matemáticas

**Documentación**: https://numpy.org/

---

### 8. joblib
**Versión**: 1.5.2  
**Propósito**: Serialización de modelos

**Uso en el proyecto**:
- Guardar/cargar modelo entrenado
- Guardar/cargar escalador

**Documentación**: https://joblib.readthedocs.io/

---

### 9. python-dotenv
**Versión**: 1.1.1  
**Propósito**: Manejo de variables de entorno

**Uso en el proyecto**:
- Cargar configuración desde `.env`
- Variables: `THRESHOLD`, `PORT`, `MAX_MB`, etc.

**Documentación**: https://python-dotenv.readthedocs.io/

---

### 10. tqdm
**Versión**: 4.67.1  
**Propósito**: Barras de progreso

**Uso en el proyecto**:
- Mostrar progreso en scripts de procesamiento
- `scripts/crop_faces.py`
- `scripts/embeddings.py`

**Documentación**: https://tqdm.github.io/

---

### 11. gunicorn
**Versión**: 23.0.0  
**Propósito**: Servidor WSGI para producción

**Uso en el proyecto**:
- Despliegue en producción
- Script `scripts/run_gunicorn.sh`

**Documentación**: https://gunicorn.org/

---

## Resumen de Dependencias

### Dependencias Principales (Core)
```
torch                    # Framework de deep learning
facenet-pytorch          # Reconocimiento facial ⭐
scikit-learn             # Machine Learning
numpy                    # Operaciones numéricas
```

### Dependencias de API
```
Flask                    # Framework web
flask-cors               # CORS support
gunicorn                 # Servidor WSGI
python-dotenv            # Variables de entorno
```

### Dependencias de Utilidades
```
Pillow                   # Procesamiento de imágenes
matplotlib               # Visualización
joblib                   # Serialización
tqdm                     # Barras de progreso
```

---

## Instalación Completa

```bash
pip install -r requirements.txt
```

### requirements.txt completo:
```
torch
facenet-pytorch
scikit-learn
numpy
Flask
flask-cors
gunicorn
python-dotenv
Pillow
joblib
tqdm
matplotlib
```

---

## ¿Cuál es la librería que hace el reconocimiento facial?

### Respuesta Directa:
**`facenet-pytorch`** es la librería principal que hace el reconocimiento facial.

### Explicación:
1. **MTCNN** (de facenet-pytorch) detecta las caras
2. **InceptionResnetV1** (de facenet-pytorch) extrae las características (embeddings)
3. **scikit-learn** solo clasifica esos embeddings como "me" o "not_me"

El "reconocimiento facial real" (detectar caras y extraer características) lo hace **facenet-pytorch**. El clasificador de scikit-learn solo decide si el embedding pertenece a "me" o "not_me" basado en los datos de entrenamiento.

---

## Versiones Usadas

| Librería | Versión | Estado |
|----------|---------|--------|
| torch | 2.2.2 | ✅ |
| facenet-pytorch | 2.6.0 | ✅ |
| scikit-learn | 1.7.2 | ✅ |
| numpy | 1.26.4 | ✅ |
| Flask | 3.1.2 | ✅ |
| matplotlib | 3.10.7 | ✅ |
| Pillow | 10.2.0 | ✅ |
| joblib | 1.5.2 | ✅ |
| python-dotenv | 1.1.1 | ✅ |
| tqdm | 4.67.1 | ✅ |
| gunicorn | 23.0.0 | ✅ |
| flask-cors | 6.0.1 | ✅ |

---

## Referencias

- **facenet-pytorch**: https://github.com/timesler/facenet-pytorch
- **FaceNet Paper**: https://arxiv.org/abs/1503.03832
- **MTCNN Paper**: https://arxiv.org/abs/1604.02878

