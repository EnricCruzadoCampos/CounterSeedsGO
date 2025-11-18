import cv2
import os
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
# Ajusta estas rutas según tu estructura de carpetas
RUTA_MODELO = 'models/best.pt'        # Tu modelo entrenado
RUTA_IMAGEN = 'pruebas/test.jpg'      # La foto que quieres analizar
UMBRAL_CONFIANZA = 0.5                # Confianza mínima (0.5 = 50%)

def procesar_granos(ruta_img, ruta_modelo):
    # 1. Verificar que existen los archivos
    if not os.path.exists(ruta_modelo):
        print(f"ERROR: No se encuentra el modelo en {ruta_modelo}")
        print("Asegúrate de descargar 'best.pt' de Colab y ponerlo en la carpeta models/")
        return

    if not os.path.exists(ruta_img):
        print(f"ERROR: No se encuentra la imagen en {ruta_img}")
        return

    # 2. Cargar el modelo y la imagen
    print("Cargando modelo...")
    model = YOLO(ruta_modelo)
    img = cv2.imread(ruta_img)
    
    # 3. Inferencia (Detección)
    print("Analizando imagen...")
    # conf=UMBRAL_CONFIANZA: Filtra detecciones dudosas
    # iou=0.45: Ayuda a evitar duplicados si las cajas se solapan mucho
    results = model.predict(img, conf=UMBRAL_CONFIANZA, iou=0.45)

    # Contadores
    cnt_trigo = 0
    cnt_extraños = 0

    # 4. Dibujar resultados
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Coordenadas (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Clase (0, 1, etc.) y Confianza
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # --- LÓGICA DE CLASES ---
            # IMPORTANTE: Verifica en tu data.yaml qué número es cada cosa.
            # Aquí asumo: 0 = Trigo, 1 = No Grano (Piedra/Impureza)
            
            if cls == 0: # ES TRIGO
                cnt_trigo += 1
                label = f"#{cnt_trigo}" # Solo el número para que no ocupe mucho
                color = (0, 255, 0)     # Verde (B, G, R)
                
                # Caja y Texto
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # Fondo negro para el texto para que se lea bien
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
            else: # ES IMPUREZA (Cualquier otra cosa)
                cnt_extraños += 1
                label = "NO GRANO"
                color = (0, 0, 255)     # Rojo
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 5. Panel de Resumen (Esquina superior izquierda)
    # Creamos un recuadro semitransparente para el texto
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (350, 80), (0, 0, 0), -1) # Caja negra fondo
    alpha = 0.6 # Transparencia
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    texto_trigo = f"Granos Trigo: {cnt_trigo}"
    texto_suci = f"Impurezas:   {cnt_extraños}"
    
    cv2.putText(img, texto_trigo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, texto_suci, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    print(f"--- RESULTADOS ---")
    print(f"Granos detectados: {cnt_trigo}")
    print(f"Objetos extraños: {cnt_extraños}")

    # 6. Guardar y Mostrar
    nombre_salida = 'resultado_analisis.jpg'
    cv2.imwrite(nombre_salida, img)
    print(f"Imagen guardada como: {nombre_salida}")
    
    # Mostrar en ventana (Solo funciona en PC local, NO en Colab)
    cv2.imshow('Detector de Trigo v1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    procesar_granos(RUTA_IMAGEN, RUTA_MODELO)
