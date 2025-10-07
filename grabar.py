import cv2
import os
import time

# --- --- --- CONFIGURACIÓN --- --- ---
# Directorio base para guardar los datos de entrenamiento.
BASE_DIR = "./dataset/train"
# Duración de cada videoclip en segundos.
CLIP_DURATION = 15
# Fotogramas por segundo (FPS) para la grabación.
FPS = 14
# Resolución del video (ancho, alto).
RESOLUTION = (640, 480)
# --- --- ------------------- --- --- ---

def get_person_name():
    """Solicita y valida el nombre de la persona para usarlo como nombre de carpeta."""
    while True:
        person_name = input("Introduce el identificador para la persona (ej. 'persona_01'): ").strip()
        if person_name:
            # Reemplaza espacios y convierte a minúsculas para un nombre de carpeta válido.
            return person_name.replace(" ", "_").lower()
        else:
            print("El nombre no puede estar vacío. Inténtalo de nuevo.")

def get_next_clip_number(person_path):
    """Calcula el número del siguiente clip para evitar sobrescribir archivos."""
    if not os.path.exists(person_path):
        return 1
    
    existing_clips = [f for f in os.listdir(person_path) if f.endswith('.mp4')]
    if not existing_clips:
        return 1
        
    # Extrae los números de los nombres de archivo y encuentra el máximo.
    max_num = 0
    for clip in existing_clips:
        try:
            # Asume el formato 'clip_XXX.mp4'
            num = int(clip.replace('clip_', '').replace('.mp4', ''))
            if num > max_num:
                max_num = num
        except ValueError:
            continue # Ignora archivos que no sigan el patrón.
            
    return max_num + 1

def record_clip(video_path):
    """Abre la cámara y graba un clip de video durante el tiempo especificado."""
    # Inicializa la captura de video desde la cámara web (el '0' es usualmente la cámara por defecto).
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Define el códec y crea el objeto VideoWriter.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Códec para .mp4
    out = cv2.VideoWriter(video_path, fourcc, FPS, RESOLUTION)

    print(f"\n🎥 Iniciando grabación de {CLIP_DURATION} segundos...")
    print("Realiza las acciones frente a la cámara: caminar, sentarte, levantarte, etc.")
    
    start_time = time.time()
    
    while (time.time() - start_time) < CLIP_DURATION:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        # Redimensiona el frame a la resolución deseada.
        frame_resized = cv2.resize(frame, RESOLUTION)
        
        # Escribe el frame en el archivo de video.
        out.write(frame_resized)
        
        # Muestra la cuenta regresiva en la pantalla.
        remaining_time = int(CLIP_DURATION - (time.time() - start_time))
        display_text = f"Grabando: {remaining_time}s"
        cv2.putText(frame_resized, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Muestra la vista en vivo.
        cv2.imshow('Grabacion en Vivo', frame_resized)

        # Permite salir de la grabación presionando 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"✅ Grabación finalizada. Video guardado en: {video_path}")

    # Libera los recursos.
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    """Función principal que orquesta el proceso de captura."""
    print("--- Programa de Captura de Datos para Entrenamiento de YOLO ---")
    
    person_name = get_person_name()
    person_path = os.path.join(BASE_DIR, person_name)

    # Crea los directorios necesarios si no existen.
    os.makedirs(person_path, exist_ok=True)
    print(f"Los datos para '{person_name}' se guardarán en: {person_path}")

    while True:
        # Pregunta al usuario si desea grabar un nuevo clip.
        action = input("\nPresiona 's' para grabar un nuevo clip o 'q' para salir: ").lower()
        
        if action == 's':
            clip_number = get_next_clip_number(person_path)
            # Formatea el nombre del archivo con ceros a la izquierda (ej. clip_001.mp4).
            clip_filename = f"clip_{clip_number:03d}.mp4"
            video_path = os.path.join(person_path, clip_filename)
            
            record_clip(video_path)
            
        elif action == 'q':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")

if __name__ == "__main__":
    main()