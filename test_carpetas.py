"""
Script de prueba para verificar que las carpetas funcionen correctamente.
"""
import os
import sys
from pathlib import Path

def test_carpetas():
    """Verifica que las carpetas existan y sean accesibles."""
    print("=" * 60)
    print("VERIFICACIÓN DE CARPETAS Y ESTRUCTURA")
    print("=" * 60)
    
    # Verificar carpetas principales
    carpetas_requeridas = [
        "data/me",
        "data/not_me",
        "data/cropped/me",
        "data/cropped/not_me",
        "models",
        "reports"
    ]
    
    print("\n1. Verificando existencia de carpetas:")
    todas_existen = True
    for carpeta in carpetas_requeridas:
        existe = os.path.isdir(carpeta)
        estado = "✅" if existe else "❌"
        print(f"   {estado} {carpeta}")
        if not existe:
            todas_existen = False
            print(f"      ⚠️  Creando carpeta: {carpeta}")
            os.makedirs(carpeta, exist_ok=True)
    
    if todas_existen:
        print("   ✅ Todas las carpetas existen")
    else:
        print("   ⚠️  Algunas carpetas se crearon automáticamente")
    
    # Verificar permisos de escritura
    print("\n2. Verificando permisos de escritura:")
    permisos_ok = True
    for carpeta in carpetas_requeridas:
        try:
            test_file = os.path.join(carpeta, ".test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"   ✅ {carpeta} - Escritura OK")
        except Exception as e:
            print(f"   ❌ {carpeta} - Error de escritura: {e}")
            permisos_ok = False
    
    # Verificar que el script crop_faces.py puede encontrar las carpetas
    print("\n3. Verificando compatibilidad con scripts:")
    try:
        from glob import glob
        INPUT_DATA_DIR = "data"
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"]
        
        for category in ["me", "not_me"]:
            category_path = os.path.join(INPUT_DATA_DIR, category)
            if os.path.isdir(category_path):
                # Buscar imágenes
                image_paths = []
                for ext in image_extensions:
                    image_paths.extend(glob(os.path.join(category_path, ext), recursive=True))
                print(f"   ✅ data/{category}/ - Encontradas {len(image_paths)} imágenes")
                if len(image_paths) == 0:
                    print(f"      ℹ️  Carpeta vacía (esto es normal si aún no has agregado imágenes)")
            else:
                print(f"   ❌ data/{category}/ - No existe")
    except Exception as e:
        print(f"   ❌ Error verificando scripts: {e}")
    
    # Verificar rutas absolutas
    print("\n4. Rutas absolutas:")
    print(f"   📁 Directorio actual: {os.path.abspath('.')}")
    print(f"   📁 data/me: {os.path.abspath('data/me')}")
    print(f"   📁 data/not_me: {os.path.abspath('data/not_me')}")
    
    # Resumen final
    print("\n" + "=" * 60)
    if todas_existen and permisos_ok:
        print("✅ VERIFICACIÓN COMPLETA: Todo está listo para usar")
        print("\n📝 Próximos pasos:")
        print("   1. Coloca tus fotos en: data/me/")
        print("   2. Coloca fotos de otras personas en: data/not_me/")
        print("   3. Ejecuta: python scripts/crop_faces.py")
        print("   4. Ejecuta: python scripts/embeddings.py")
        print("   5. Ejecuta: python train.py")
    else:
        print("⚠️  VERIFICACIÓN COMPLETA: Hay algunos problemas")
        print("   Revisa los errores arriba")
    print("=" * 60)
    
    return todas_existen and permisos_ok

if __name__ == "__main__":
    success = test_carpetas()
    sys.exit(0 if success else 1)

