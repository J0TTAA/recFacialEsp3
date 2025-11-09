# Sistema de Reconocimiento Facial - API REST

Sistema de reconocimiento facial que identifica si una cara pertenece al usuario específico ("me") o no ("not_me"). El proyecto incluye un pipeline completo de entrenamiento y una API REST construida con Flask para realizar predicciones en tiempo real.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Pipeline de Entrenamiento](#pipeline-de-entrenamiento)
- [API REST](#api-rest)
- [Configuración](#configuración)
- [Uso](#uso)
- [Ejemplos](#ejemplos)
- [Troubleshooting](#troubleshooting)
- [Evaluación y Reportes](#evaluación-y-reportes)
- [Ética y Privacidad](#ética-y-privacidad)

## 📖 Descripción General

Este sistema utiliza técnicas de deep learning para el reconocimiento facial:

1. **Detección de Caras**: Usa MTCNN para detectar y recortar caras en imágenes
2. **Generación de Embeddings**: Utiliza InceptionResnetV1 (FaceNet) pre-entrenado en VGGFace2 para generar vectores de características de 512 dimensiones
3. **Clasificación**: Entrena un clasificador LogisticRegression para distinguir entre "me" (usuario) y "not_me" (otros)

La API REST permite realizar predicciones en tiempo real enviando imágenes que contengan caras.

## 🔧 Requisitos

### Software
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Hardware Recomendado
- **GPU NVIDIA** (opcional pero altamente recomendado para acelerar el procesamiento)
  - CUDA 11.0 o superior
  - CuDNN 8.0 o superior
- **CPU**: Procesador multi-core (funciona pero más lento)
- **RAM**: Mínimo 4GB, recomendado 8GB+

### Sistema Operativo
- Windows 10/11
- Linux (Ubuntu 18.04+)
- macOS (con limitaciones en GPU)

## 📦 Instalación

### 1. Clonar o Descargar el Proyecto

```bash
cd recFacialEsp3
```

### 2. Crear Entorno Virtual (Recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- `torch`: PyTorch para deep learning
- `facenet-pytorch`: Modelos pre-entrenados para reconocimiento facial
- `scikit-learn`: Machine learning (clasificación)
- `Flask`: Framework web para la API
- `flask-cors`: Soporte CORS
- `gunicorn`: Servidor WSGI para producción
- `python-dotenv`: Manejo de variables de entorno
- `Pillow`: Procesamiento de imágenes
- `joblib`: Serialización de modelos
- `numpy`: Operaciones numéricas
- `tqdm`: Barras de progreso

### 4. Verificar Instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## 📁 Estructura del Proyecto

```
recFacialEsp3/
├── api/                          # Módulo de la API
│   ├── __init__.py
│   ├── resources/                # Endpoints de la API
│   │   ├── __init__.py
│   │   ├── verify.py            # Endpoint /verify
│   │   └── health.py            # Endpoint /healthz
│   └── utils/                    # Utilidades
│       ├── __init__.py
│       ├── model_loader.py       # Carga de modelos
│       └── image_processor.py    # Procesamiento de imágenes
│
├── scripts/                      # Scripts de entrenamiento
│   ├── crop_faces.py            # Paso 1: Recorte de caras
│   ├── embeddings.py            # Paso 2: Generación de embeddings
│   └── run_gunicorn.sh          # Script para producción
│
├── data/                         # Datos del proyecto
│   ├── me/                       # Imágenes del usuario (YO)
│   ├── not_me/                   # Imágenes de otras personas (NO-YO)
│   ├── cropped/                  # Caras recortadas (generado)
│   │   ├── me/
│   │   └── not_me/
│   ├── embeddings.npy           # Embeddings generados (generado)
│   └── labels.npy               # Etiquetas (generado)
│
├── models/                       # Modelos entrenados (generado)
│   ├── model.joblib             # Clasificador entrenado
│   └── scaler.joblib            # Escalador de datos
│
├── reports/                      # Reportes y métricas (generado)
│   ├── metrics.json             # Métricas del modelo
│   └── confusion_matrix.png     # Matriz de confusión
│
├── app.py                        # Aplicación Flask principal
├── train.py                     # Script de entrenamiento
├── requirements.txt             # Dependencias
├── .env                         # Variables de entorno (crear)
└── README.md                    # Este archivo
```

## 🚀 Pipeline de Entrenamiento

El entrenamiento del modelo se realiza en **4 pasos secuenciales**. Asegúrate de ejecutarlos en orden.

### Paso 0: Preparar Datos

Antes de comenzar, organiza tus imágenes en la siguiente estructura:

```
data/
├── me/              # Imágenes del usuario (YO)
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ...
└── not_me/          # Imágenes de otras personas (NO-YO)
    ├── persona1.jpg
    ├── persona2.jpg
    └── ...
```

**Requisitos:**
- Formatos soportados: `.jpg`, `.jpeg`, `.png` (mayúsculas o minúsculas)
- Mínimo recomendado: 20-30 imágenes en cada categoría
- Ideal: 50+ imágenes en cada categoría para mejor precisión
- Las imágenes pueden tener diferentes tamaños y orientaciones

### Paso 1: Recorte de Caras (`scripts/crop_faces.py`)

Este script detecta y recorta caras de todas las imágenes usando MTCNN.

**¿Qué hace?**
- Detecta caras en las imágenes usando MTCNN
- Recorta cada cara detectada
- Redimensiona a 160x160 píxeles (tamaño requerido por FaceNet)
- Corrige la orientación EXIF automáticamente
- Guarda las caras recortadas en `data/cropped/`

**Ejecutar:**
```bash
python scripts/crop_faces.py
```

**Salida:**
- `data/cropped/me/`: Caras recortadas del usuario
- `data/cropped/not_me/`: Caras recortadas de otras personas

**Notas:**
- Si una imagen no tiene cara detectada, se omite y se registra un warning
- El proceso es más rápido con GPU (CUDA)

### Paso 2: Generación de Embeddings (`scripts/embeddings.py`)

Este script genera vectores de características (embeddings) de 512 dimensiones para cada cara recortada.

**¿Qué hace?**
- Carga el modelo InceptionResnetV1 pre-entrenado en VGGFace2
- Procesa las caras recortadas en lotes (batches) para eficiencia
- Genera embeddings de 512 dimensiones para cada cara
- Crea etiquetas paralelas (1 para "me", 0 para "not_me")

**Ejecutar:**
```bash
python scripts/embeddings.py
```

**Salida:**
- `data/embeddings.npy`: Array numpy con todos los embeddings [N, 512]
- `data/labels.npy`: Array numpy con todas las etiquetas [N]

**Notas:**
- Procesamiento por lotes (batch_size=32) acelera significativamente el proceso
- Mucho más rápido con GPU

### Paso 3: Entrenamiento del Clasificador (`train.py`)

Este script entrena un clasificador LogisticRegression para distinguir entre "me" y "not_me".

**¿Qué hace?**
- Carga los embeddings y etiquetas generados
- Divide los datos en entrenamiento (80%) y validación (20%)
- Aplica StandardScaler para normalizar los embeddings
- Entrena un LogisticRegression con `class_weight='balanced'` (importante para datos desbalanceados)
- Evalúa el modelo y calcula métricas (Accuracy, AUC-ROC, Precision, Recall, F1-score)
- Guarda el modelo entrenado y el escalador

**Ejecutar:**
```bash
python train.py
```

**Salida:**
- `models/model.joblib`: Clasificador entrenado
- `models/scaler.joblib`: Escalador ajustado
- `reports/metrics.json`: Métricas del modelo en formato JSON

**Métricas Generadas:**
- **Accuracy**: Precisión general del modelo
- **AUC-ROC**: Área bajo la curva ROC (mejor para datos desbalanceados)
- **Classification Report**: Precision, Recall y F1-score para cada clase

**Parámetros Importantes:**
- `test_size=0.2`: 20% para validación
- `random_state=42`: Semilla para reproducibilidad
- `stratify=y`: Mantiene la proporción de clases en train/test
- `class_weight='balanced'`: Ajusta pesos para datos desbalanceados

### Paso 4: Verificación (Opcional)

Puedes verificar que el entrenamiento fue exitoso revisando:

```bash
# Ver métricas
cat reports/metrics.json

# Verificar que los modelos existen
ls models/
```

### Paso 5: Evaluación completa

```bash
python evaluate.py
```

Genera métricas actualizadas, curvas ROC/PR y `reports/evaluation_results.json`. Ejecuta este paso cada vez que reentrenes el modelo para mantener trazabilidad.

## 🌐 API REST

La API REST permite realizar predicciones en tiempo real enviando imágenes que contengan caras.

### Iniciar el Servidor

**Desarrollo:**
```bash
python app.py
```

El servidor se iniciará en `http://0.0.0.0:5000` por defecto.

**Producción (con Gunicorn):**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

O usar el script:
```bash
bash scripts/run_gunicorn.sh
```

### Endpoints

#### 1. `GET /`
Información general de la API.

**Respuesta:**
```json
{
  "name": "Face Recognition API",
  "version": "1.0.0",
  "description": "API para reconocimiento facial...",
  "endpoints": {
    "health": "/healthz",
    "verify": "/verify"
  }
}
```

#### 2. `GET /healthz`
Verifica el estado de la API y los modelos.

**Respuesta Exitosa (200):**
```json
{
  "status": "ok"
}
```

**Respuesta de Error (503):**
```json
{
  "status": "unhealthy",
  "reason": "Models not loaded"
}
```

#### 3. `POST /verify`
Endpoint principal para verificar si una cara pertenece al usuario.

**Request:**
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - Campo `image`: Archivo de imagen (jpg, png)

**Ejemplo con cURL:**
```bash
curl -X POST http://localhost:5000/verify \
  -F "image=@ruta/a/tu/imagen.jpg"
```

**Ejemplo con Python:**
```python
import requests

url = "http://localhost:5000/verify"
files = {"image": open("imagen.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Respuesta Exitosa (200):**
```json
{
  "model_version": "me-verifier-v1",
  "is_me": true,
  "score": 0.9234,
  "threshold": 0.75,
  "timing_ms": 245.67
}
```

**Campos de Respuesta:**
- `model_version`: Versión del modelo usado
- `is_me`: `true` si la cara pertenece al usuario, `false` si no
- `score`: Probabilidad de que sea "me" (0.0 a 1.0)
- `threshold`: Umbral usado para determinar `is_me`
- `timing_ms`: Tiempo de procesamiento en milisegundos

**Respuestas de Error:**

**400 - No se detectó cara:**
```json
{
  "error": "No se detectó un rostro en la imagen"
}
```

**400 - Tipo de archivo inválido:**
```json
{
  "error": "Tipo de archivo no permitido. Solo image/jpeg o image/png"
}
```

**413 - Archivo demasiado grande:**
```json
{
  "error": "Archivo demasiado grande. Límite: 2 MB"
}
```

**503 - Modelos no cargados:**
```json
{
  "error": "Modelos no encontrados. Asegúrate de haber entrenado el modelo ejecutando 'train.py'."
}
```

## ⚙️ Configuración

El proyecto usa variables de entorno para configuración. Crea un archivo `.env` en la raíz del proyecto:

```env
# Rutas de modelos
MODEL_PATH=models/model.joblib
SCALER_PATH=models/scaler.joblib

# Umbral de verificación (0.0 a 1.0)
# Score >= VERIFY_THRESHOLD -> is_me = true
VERIFY_THRESHOLD=0.75

# Tamaño máximo de archivo (MB)
MAX_CONTENT_MB=2

# Versión del modelo
MODEL_VERSION=me-verifier-v1

# Configuración del servidor Flask
FLASK_HOST=0.0.0.0
PORT=5000
FLASK_DEBUG=False
```

**Variables Disponibles:**
- `MODEL_PATH`: Ruta al modelo clasificador (default: `models/model.joblib`)
- `SCALER_PATH`: Ruta al escalador (default: `models/scaler.joblib`)
- `VERIFY_THRESHOLD`: Umbral para determinar si es "me" (default: `0.75`)
- `MAX_CONTENT_MB`: Tamaño máximo de archivo en MB (default: `2`)
- `MODEL_VERSION`: Versión del modelo (default: `me-verifier-v1`)
- `FLASK_HOST`: Host del servidor (default: `0.0.0.0`)
- `PORT`: Puerto del servidor (default: `5000`)
- `FLASK_DEBUG`: Modo debug (default: `False`)

## 💡 Uso

### Flujo Completo

1. **Preparar datos:**
   ```bash
   # Organizar imágenes en data/me/ y data/not_me/
   ```

2. **Recortar caras:**
   ```bash
   python scripts/crop_faces.py
   ```

3. **Generar embeddings:**
   ```bash
   python scripts/embeddings.py
   ```

4. **Entrenar modelo:**
   ```bash
   python train.py
   ```

5. **Iniciar API:**
   ```bash
   python app.py
   ```

6. **Hacer predicciones:**
   ```bash
   curl -X POST http://localhost:5000/verify -F "image=@foto.jpg"
   ```

## 📝 Ejemplos

### Ejemplo 1: Verificación Básica

```python
import requests

# Subir imagen para verificación
url = "http://localhost:5000/verify"
with open("mi_foto.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)
    
result = response.json()
print(f"¿Es mi cara? {result['is_me']}")
print(f"Score: {result['score']:.2%}")
print(f"Tiempo: {result['timing_ms']}ms")
```

### Ejemplo 2: Verificación con Threshold Personalizado

```python
import requests
import os

# Configurar threshold en .env o directamente
os.environ["VERIFY_THRESHOLD"] = "0.8"

# Hacer verificación
url = "http://localhost:5000/verify"
with open("foto.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)
    
result = response.json()
if result["is_me"]:
    print("✅ Acceso autorizado")
else:
    print("❌ Acceso denegado")
```

### Ejemplo 3: Verificación de Salud del Servidor

```python
import requests

# Verificar estado
response = requests.get("http://localhost:5000/healthz")
status = response.json()

if status["status"] == "ok":
    print("✅ Servidor funcionando correctamente")
else:
    print(f"❌ Servidor con problemas: {status.get('reason', 'Desconocido')}")
```

### Ejemplo 4: Procesamiento por Lotes

```python
import requests
import os
from pathlib import Path

url = "http://localhost:5000/verify"
results = []

# Procesar múltiples imágenes
image_dir = Path("imagenes")
for image_path in image_dir.glob("*.jpg"):
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(url, files=files)
        result = response.json()
        results.append({
            "file": image_path.name,
            "is_me": result.get("is_me", False),
            "score": result.get("score", 0)
        })

# Mostrar resultados
for r in results:
    print(f"{r['file']}: {'✅' if r['is_me'] else '❌'} (score: {r['score']:.2%})")
```

## 🔍 Troubleshooting

### Problema: "No se detectó un rostro en la imagen"

**Causas posibles:**
- La imagen no contiene una cara visible
- La cara es muy pequeña o está muy oscura
- La imagen está borrosa o de baja calidad

**Soluciones:**
- Usa imágenes con buena iluminación y resolución
- Asegúrate de que la cara esté claramente visible
- Prueba con diferentes ángulos de la cara

### Problema: "Modelos no encontrados"

**Causas:**
- No se ejecutó el pipeline de entrenamiento completo
- Los modelos fueron eliminados o movidos

**Soluciones:**
```bash
# Verificar que los modelos existen
ls models/

# Si no existen, ejecutar el pipeline completo
python scripts/crop_faces.py
python scripts/embeddings.py
python train.py
```

### Problema: "CUDA out of memory"

**Causas:**
- GPU con poca memoria
- Batch size demasiado grande

**Soluciones:**
- Reducir el batch size en `scripts/embeddings.py` (línea 19)
- Usar CPU en lugar de GPU (automático si CUDA no está disponible)
- Procesar menos imágenes a la vez

### Problema: API muy lenta

**Causas:**
- Ejecutándose en CPU en lugar de GPU
- Imágenes muy grandes

**Soluciones:**
- Verificar que CUDA esté disponible: `python -c "import torch; print(torch.cuda.is_available())"`
- Reducir tamaño de imágenes antes de enviarlas
- Usar Gunicorn con múltiples workers en producción

### Problema: "Archivo demasiado grande"

**Causas:**
- Imagen excede el límite de `MAX_CONTENT_MB`

**Soluciones:**
- Reducir tamaño de la imagen antes de enviarla
- Aumentar `MAX_CONTENT_MB` en `.env`

### Problema: Accuracy baja o modelo malo

**Causas:**
- Pocas imágenes de entrenamiento
- Datos desbalanceados extremos
- Imágenes de mala calidad

**Soluciones:**
- Aumentar el número de imágenes en `data/me/` y `data/not_me/`
- Asegurar al menos 30-50 imágenes por categoría
- Usar imágenes de buena calidad y variadas (diferentes ángulos, iluminación, etc.)
- Verificar las métricas en `reports/metrics.json`

## 📊 Estructura de la API

### Arquitectura Modular

La API está organizada en módulos para facilitar el mantenimiento:

```
api/
├── resources/          # Endpoints (Rutas)
│   ├── verify.py       # Endpoint /verify
│   └── health.py       # Endpoint /healthz
└── utils/              # Utilidades
    ├── model_loader.py    # Carga de modelos (lazy loading)
    └── image_processor.py # Procesamiento de imágenes
```

### Flujo de Procesamiento en `/verify`

1. **Validación**: Verifica que se envió una imagen válida
2. **Detección**: MTCNN detecta y recorta la cara
3. **Embedding**: FaceNet genera vector de características (512D)
4. **Escalado**: Aplica StandardScaler
5. **Clasificación**: LogisticRegression predice probabilidad
6. **Umbral**: Compara con `VERIFY_THRESHOLD` para determinar `is_me`
7. **Respuesta**: Retorna resultado JSON

### Carga de Modelos

Los modelos se cargan **una sola vez al iniciar el servidor** (lazy loading):
- Primera petición: Carga los modelos
- Peticiones subsecuentes: Reutiliza modelos cargados en memoria
- Esto mejora significativamente el tiempo de respuesta

## 📈 Evaluación y Reportes

- **Métricas de entrenamiento**: `reports/metrics.json` (split 70/15/15, guardado por `train.py`).
- **Evaluación 80/20 reproducible**: `evaluate.py` genera `reports/evaluation_results.json` y gráficos (`confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`).
- **Informe técnico extendido**: `reports/INFORME_H10.md` resume dataset, pipeline, análisis de umbral, métricas, latencia y recomendaciones.
- **Umbral operativo**: configurable mediante `VERIFY_THRESHOLD` (default `0.75`). `evaluate.py` calcula el umbral óptimo según F1-score (`threshold_analysis.optimal_threshold_f1`).
- **Latencia de inferencia**: la regresión logística tarda ≈0.002 ms por muestra en CPU; los tiempos totales dependen de la detección y del embedding.

Recomendaciones:
- Re-ejecutar `evaluate.py` tras cualquier cambio en el dataset o en el modelo.
- Documentar en `reports/` cualquier evaluación manual adicional (p. ej., pruebas con datos externos).

## 🛡️ Ética y Privacidad

- **Datos sensibles**: las imágenes crudas contienen información biométrica. No se versionan; almacénalas cifradas y elimina copias temporales tras procesar.
- **Consentimiento**: asegúrate de contar con autorización explícita de cada persona en `data/not_me`.
- **Uso responsable**: limita el acceso a la API y registra auditorías de uso para detectar abuso.
- **Cumplimiento normativo**: considera GDPR/LGPD y leyes locales antes de desplegar en producción; ofrece mecanismos para revocar consentimiento y eliminar datos.
- **Sesgos**: el dataset actual está desbalanceado (≈10 % positivos). Amplía la cobertura con más imágenes propias y casos negativos diversos para reducir sesgos.

## 🚀 Despliegue en Producción

### Con Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
```

**Parámetros:**
- `-w 4`: 4 workers (ajustar según CPU)
- `-b 0.0.0.0:5000`: Host y puerto
- `--timeout 120`: Timeout de 120 segundos (para procesamiento de imágenes)

### Con Docker (Ejemplo)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Variables de Entorno en Producción

Asegúrate de configurar:
```env
FLASK_DEBUG=False
VERIFY_THRESHOLD=0.75
MAX_CONTENT_MB=2
```

## 📚 Referencias

- **FaceNet Paper**: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- **MTCNN**: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
- **facenet-pytorch**: [GitHub Repository](https://github.com/timesler/facenet-pytorch)
- **Flask**: [Documentación Oficial](https://flask.palletsprojects.com/)

## 📄 Licencia

Este proyecto es de uso educativo/académico.

## 👤 Autor

Sistema de Reconocimiento Facial - Universidad

---

**¿Problemas?** Revisa la sección [Troubleshooting](#troubleshooting) o abre un issue en el repositorio.

