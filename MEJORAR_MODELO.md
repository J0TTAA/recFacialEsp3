# Guía para Mejorar el Modelo de Reconocimiento Facial

## Problema Identificado

El modelo está dando **falsos positivos**: clasifica incorrectamente a otras personas como "me" con scores altos (ej: 0.92).

### Causas Posibles

1. **Umbral demasiado bajo**: El umbral actual (0.75) es muy permisivo
2. **Dataset desbalanceado**: 94 fotos "me" vs 200 "not_me" (31.97% vs 68.03%)
3. **Falta de diversidad**: Las fotos de "not_me" pueden no ser suficientemente variadas
4. **Overfitting**: El modelo puede estar sobreajustado a las condiciones específicas del entrenamiento

---

## Solución 1: Aumentar el Umbral (Solución Inmediata) ⚡

### Opción A: Cambiar en el código

Edita `api/resources/verify.py` línea 30:

```python
VERIFY_THRESHOLD = float(os.environ.get("VERIFY_THRESHOLD", 0.90))  # Cambiar de 0.75 a 0.90
```

### Opción B: Usar archivo .env

Crea un archivo `.env` en la raíz del proyecto:

```env
VERIFY_THRESHOLD=0.90
```

### Opción C: Usar el umbral óptimo de la evaluación

Según `evaluate.py`, el umbral óptimo es **0.99999**:

```env
VERIFY_THRESHOLD=0.95
```

**Recomendación**: Empieza con **0.90** y ajusta según resultados.

---

## Solución 2: Mejorar el Dataset de Entrenamiento

### 2.1 Aumentar variedad en "not_me"

**Problema**: Solo 200 fotos de "not_me" pueden no ser suficientes.

**Soluciones**:
- Agregar más fotos de personas diferentes en `data/not_me/`
- Variedad en:
  - Edades diferentes
  - Géneros diferentes
  - Etnias diferentes
  - Expresiones faciales variadas
  - Iluminaciones diferentes
  - Ángulos diferentes

**Recomendación**: Mínimo 300-400 fotos de "not_me"

### 2.2 Aumentar variedad en "me"

**Problema**: 94 fotos tuyas pueden no cubrir todas las variaciones.

**Soluciones**:
- Agregar más fotos tuyas en `data/me/`:
  - Diferentes ángulos (frontal, perfil, 3/4)
  - Diferentes expresiones (sonriendo, serio, etc.)
  - Diferentes iluminaciones
  - Con/sin gafas, barba, etc.
  - Diferentes edades (si tienes fotos antiguas)

**Recomendación**: Mínimo 150-200 fotos de "me"

### 2.3 Balancear el dataset

**Ratio ideal**: 1:1 o máximo 1:2 (me:not_me)

**Actual**: 94:200 (1:2.13) - Aceptable pero mejorable

**Objetivo**: 150:300 (1:2) o mejor aún 200:200 (1:1)

---

## Solución 3: Reentrenar el Modelo

Después de mejorar el dataset, reentrena:

```bash
# 1. Recortar nuevas caras
python scripts/crop_faces.py

# 2. Generar nuevos embeddings
python scripts/embeddings.py

# 3. Reentrenar el modelo
python train.py

# 4. Reevaluar
python evaluate.py
```

---

## Solución 4: Ajustar Hiperparámetros del Modelo

### Opción A: Modificar LogisticRegression

Edita `train.py` línea 95-101:

```python
model = LogisticRegression(
    max_iter=500,              # Aumentar iteraciones
    class_weight='balanced',   # Mantener balanceado
    random_state=RANDOM_SEED,
    solver='liblinear',
    C=0.1,                     # AÑADIR: Regularización más fuerte (default es 1.0)
    penalty='l2'               # AÑADIR: Regularización L2 explícita
)
```

**Parámetro C**:
- **C < 1.0**: Más regularización (más conservador, menos overfitting)
- **C = 1.0**: Default
- **C > 1.0**: Menos regularización (más flexible)

**Recomendación**: Probar con `C=0.1` o `C=0.5`

### Opción B: Usar SVM en lugar de LogisticRegression

SVM (Support Vector Machine) puede ser más robusto:

```python
from sklearn.svm import SVC

model = SVC(
    probability=True,          # Necesario para predict_proba
    class_weight='balanced',
    kernel='rbf',
    C=1.0,
    gamma='scale'
)
```

---

## Solución 5: Data Augmentation (Aumento de Datos)

Agregar variaciones de tus imágenes existentes:

### Script de aumento de datos

```python
# scripts/augment_data.py
from PIL import Image, ImageEnhance, ImageFilter
import os
from glob import glob

def augment_image(image_path, output_dir):
    """Crea variaciones de una imagen."""
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Rotación ligera
    img.rotate(5).save(os.path.join(output_dir, f"{base_name}_rot5.jpg"))
    img.rotate(-5).save(os.path.join(output_dir, f"{base_name}_rot-5.jpg"))
    
    # 2. Brillo
    enhancer = ImageEnhance.Brightness(img)
    enhancer.enhance(0.8).save(os.path.join(output_dir, f"{base_name}_dark.jpg"))
    enhancer.enhance(1.2).save(os.path.join(output_dir, f"{base_name}_bright.jpg"))
    
    # 3. Contraste
    enhancer = ImageEnhance.Contrast(img)
    enhancer.enhance(0.8).save(os.path.join(output_dir, f"{base_name}_lowcontrast.jpg"))
    enhancer.enhance(1.2).save(os.path.join(output_dir, f"{base_name}_highcontrast.jpg"))
```

---

## Plan de Acción Recomendado

### Paso 1: Solución Inmediata (5 minutos)
1. Aumentar umbral a **0.90** o **0.95**
2. Reiniciar la API
3. Probar de nuevo

### Paso 2: Mejora del Dataset (1-2 horas)
1. Agregar más fotos de "not_me" (mínimo 100-200 adicionales)
2. Agregar más fotos tuyas variadas (mínimo 50-100 adicionales)
3. Asegurar variedad en condiciones, ángulos, expresiones

### Paso 3: Reentrenamiento (10-15 minutos)
1. Recortar caras: `python scripts/crop_faces.py`
2. Generar embeddings: `python scripts/embeddings.py`
3. Entrenar: `python train.py`
4. Evaluar: `python evaluate.py`

### Paso 4: Ajuste Fino (Opcional)
1. Si persisten problemas, ajustar hiperparámetros
2. Probar con regularización más fuerte (C=0.1)
3. Considerar SVM como alternativa

---

## Umbrales Recomendados

| Nivel de Seguridad | Umbral | Descripción |
|-------------------|--------|-------------|
| **Bajo** | 0.75 | Permisivo, más falsos positivos |
| **Medio** | 0.85 | Balanceado |
| **Alto** | 0.90 | Estricto, menos falsos positivos |
| **Muy Alto** | 0.95 | Muy estricto, máxima seguridad |
| **Extremo** | 0.99 | Extremadamente estricto |

**Recomendación inicial**: **0.90**

---

## Verificación de Mejoras

Después de aplicar cambios, verifica:

1. **Probar con fotos tuyas**: Debería dar `is_me: true` con score > umbral
2. **Probar con fotos de otras personas**: Debería dar `is_me: false` con score < umbral
3. **Probar casos límite**: Fotos que se parezcan a ti pero no sean tuyas
4. **Revisar métricas**: Ejecutar `evaluate.py` y verificar métricas

---

## Monitoreo Continuo

Después de desplegar:

1. **Logging**: Revisar logs JSON para identificar patrones
2. **Falsos positivos**: Registrar casos donde el modelo falla
3. **Falsos negativos**: Registrar casos donde no te reconoce
4. **Retraining**: Reentrenar periódicamente con nuevos datos

---

## Notas Importantes

⚠️ **El modelo perfecto (100% accuracy) puede ser signo de overfitting**
- Si el modelo tiene 100% en validación pero falla en producción, está sobreajustado
- Esto es común cuando el dataset es pequeño o poco variado

✅ **Es normal que el modelo no sea 100% perfecto**
- 95-98% de accuracy es excelente para reconocimiento facial
- Lo importante es minimizar falsos positivos (seguridad)

📊 **Balance entre seguridad y usabilidad**
- Umbral alto = Más seguro pero puede rechazar casos válidos
- Umbral bajo = Más permisivo pero puede aceptar casos inválidos
- Encuentra el balance según tu caso de uso

