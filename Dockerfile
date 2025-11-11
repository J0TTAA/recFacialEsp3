# Imagen base ligera de Python 3.10
FROM python:3.10-slim

# Variables de entorno para un comportamiento consistente de Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Instalar dependencias del sistema necesarias para PyTorch, Pillow y procesamiento de imágenes
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libgtk-3-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias de Python e instalarlas
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Exponer el puerto interno usado por Gunicorn
EXPOSE 5000

# Ejecutar la aplicación con Gunicorn usando la factory de Flask
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:create_app()", "--factory"]

