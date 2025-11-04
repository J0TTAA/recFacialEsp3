# Guía para Probar la API con Postman

## Paso 1: Verificar que la API esté corriendo

Asegúrate de que la API esté ejecutándose. Si no está corriendo, ejecuta:

```bash
python app.py
```

Deberías ver algo como:
```
Iniciando servidor Flask en 0.0.0.0:5000
```

---

## Paso 2: Configurar Postman

### 2.1 Crear una nueva petición

1. Abre **Postman**
2. Haz clic en **"New"** → **"HTTP Request"**
3. O simplemente haz clic en el botón **"+"** para crear una nueva petición

### 2.2 Configurar el método y URL

1. **Método**: Selecciona **POST** (en el dropdown a la izquierda)
2. **URL**: Escribe `http://localhost:5000/verify`

```
POST http://localhost:5000/verify
```

### 2.3 Configurar el Body (Cuerpo de la petición)

1. Ve a la pestaña **"Body"**
2. Selecciona **"form-data"** (NO "raw" ni "x-www-form-urlencoded")
3. En la primera fila:
   - **Key**: Escribe `image`
   - **Tipo**: Cambia de "Text" a **"File"** (hay un dropdown al lado derecho)
   - **Value**: Haz clic en **"Select Files"** y elige una imagen de tu computadora

### 2.4 Configuración final

Tu configuración debería verse así:

```
Method: POST
URL: http://localhost:5000/verify
Body: form-data
  Key: image (tipo: File)
  Value: [Tu archivo de imagen]
```

---

## Paso 3: Enviar la petición

1. Haz clic en el botón **"Send"** (azul)
2. Espera la respuesta (puede tardar unos segundos mientras procesa la imagen)

---

## Paso 4: Ver la respuesta

### Respuesta exitosa (200 OK)

```json
{
  "model_version": "me-verifier-v1",
  "is_me": true,
  "score": 0.9234,
  "threshold": 0.75,
  "timing_ms": 245.67
}
```

**Campos**:
- `is_me`: `true` si es tu cara, `false` si no
- `score`: Probabilidad (0.0 a 1.0) de que sea "me"
- `threshold`: Umbral usado para la decisión
- `timing_ms`: Tiempo de procesamiento en milisegundos

### Respuesta de error (400)

```json
{
  "error": "No se detectó un rostro en la imagen"
}
```

**Posibles errores**:
- `"No se detectó un rostro en la imagen"`: La imagen no tiene una cara visible
- `"Tipo de archivo no permitido"`: Solo acepta JPEG/PNG
- `"Archivo demasiado grande"`: Límite de 2 MB por defecto

---

## Capturas de pantalla de referencia

### Configuración en Postman:

```
┌─────────────────────────────────────────┐
│ POST  │  http://localhost:5000/verify   │  [Send]
├─────────────────────────────────────────┤
│ Params │ Authorization │ Headers │ Body  │
├─────────────────────────────────────────┤
│ Body:  ○ none  ○ form-data  ○ x-www... │
│        ○ raw    ○ binary    ○ GraphQL   │
│                                        │
│ Key         │ Value      │ Description │
│ image (File)│ [Browse...]│             │
└─────────────────────────────────────────┘
```

---

## Ejemplo paso a paso (con imágenes)

### 1. Seleccionar método POST
   - En el dropdown izquierdo, selecciona **POST**

### 2. Escribir URL
   - En el campo de URL, escribe: `http://localhost:5000/verify`

### 3. Ir a la pestaña Body
   - Haz clic en **"Body"** debajo de la URL

### 4. Seleccionar form-data
   - Marca la opción **"form-data"**

### 5. Agregar campo image
   - En la fila de campos, escribe `image` en la columna **Key**
   - En la columna **Value**, haz clic en el dropdown y selecciona **"File"**
   - Haz clic en **"Select Files"** y elige una imagen

### 6. Enviar
   - Haz clic en **"Send"**

---

## Probar diferentes endpoints

### 1. Health Check (GET /healthz)

```
Method: GET
URL: http://localhost:5000/healthz
Body: (vacío)
```

**Respuesta esperada**:
```json
{
  "status": "ok"
}
```

### 2. Información de la API (GET /)

```
Method: GET
URL: http://localhost:5000/
Body: (vacío)
```

**Respuesta esperada**:
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

---

## Troubleshooting

### Error: "Could not get response"

**Causa**: La API no está corriendo

**Solución**:
1. Abre una terminal en la carpeta del proyecto
2. Ejecuta: `python app.py`
3. Espera a ver: "Running on http://0.0.0.0:5000"
4. Vuelve a intentar en Postman

### Error: "Connection refused"

**Causa**: La API no está escuchando en el puerto 5000

**Solución**:
- Verifica que la API esté corriendo
- Verifica que el puerto 5000 no esté bloqueado por firewall

### Error: "No se detectó un rostro en la imagen"

**Causa**: La imagen no tiene una cara claramente visible

**Solución**:
- Usa una imagen con una cara claramente visible
- Asegúrate de que la cara esté bien iluminada
- La cara debe estar frontal o de perfil claramente visible

### Error: "Tipo de archivo no permitido"

**Causa**: El archivo no es JPEG o PNG

**Solución**:
- Convierte la imagen a formato .jpg o .png
- Asegúrate de que el archivo tenga la extensión correcta

### Error: "Archivo demasiado grande"

**Causa**: La imagen excede el límite de tamaño (2 MB por defecto)

**Solución**:
- Redimensiona o comprime la imagen
- O modifica `MAX_CONTENT_MB` en el archivo `.env`

---

## Colección de Postman (Opcional)

Puedes crear una colección en Postman para guardar estas peticiones:

1. Haz clic en **"New"** → **"Collection"**
2. Nombra la colección: "Face Recognition API"
3. Guarda las peticiones en esta colección para uso futuro

### Peticiones recomendadas para la colección:

1. **Health Check** - GET /healthz
2. **API Info** - GET /
3. **Verify Face (Me)** - POST /verify (con tu foto)
4. **Verify Face (Not Me)** - POST /verify (con foto de otra persona)

---

## Ejemplos de imágenes para probar

### Para obtener `is_me: true`
- Usa una foto tuya que esté en `data/me/`
- O cualquier foto tuya que no esté en el dataset

### Para obtener `is_me: false`
- Usa una foto de otra persona
- O una foto de `data/not_me/`

### Para probar errores
- Imagen sin cara: Paisaje, objeto, etc.
- Imagen muy grande: > 2 MB
- Formato incorrecto: .gif, .bmp, etc.

---

## Configuración avanzada (Headers)

Por defecto, no necesitas configurar headers. Pero si quieres agregar algunos:

### Headers opcionales:

```
Content-Type: multipart/form-data
```

**Nota**: Postman configura esto automáticamente cuando usas `form-data`, no necesitas agregarlo manualmente.

---

## Verificación rápida

### Checklist antes de enviar:

- ✅ Método: **POST**
- ✅ URL: `http://localhost:5000/verify`
- ✅ Body: **form-data** seleccionado
- ✅ Key: `image` (tipo: File)
- ✅ Value: Archivo seleccionado
- ✅ API corriendo en el puerto 5000

---

## Resultados esperados

### Caso 1: Tu foto (debería ser `is_me: true`)
```json
{
  "is_me": true,
  "score": 0.95,
  "threshold": 0.75,
  "timing_ms": 250.5
}
```

### Caso 2: Foto de otra persona (debería ser `is_me: false`)
```json
{
  "is_me": false,
  "score": 0.15,
  "threshold": 0.75,
  "timing_ms": 245.3
}
```

---

**¡Listo para probar!** 🚀

Si tienes algún problema, revisa la sección de Troubleshooting arriba.

