#!/bin/bash

# Script para ejecutar la API con Gunicorn en producción
# H8 - Producción

# Configuración
HOST="0.0.0.0"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-2}"
TIMEOUT="${TIMEOUT:-120}"

echo "Iniciando servidor Gunicorn..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Timeout: $TIMEOUT"

# Ejecutar Gunicorn
# Nota: app:app se refiere a app.py:app (la variable app en app.py)
gunicorn -w $WORKERS -b $HOST:$PORT --timeout $TIMEOUT app:app

