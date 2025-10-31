from ultralytics import YOLO

# --- CONFIGURACIÓN ---
MODEL_TO_EXPORT = './yolo11m-pose.pt'
EXPORT_NAME = './yolo11m-pose.onnx'
# ---------------------

print(f"Cargando el modelo de pose '{MODEL_TO_EXPORT}'...")
# Carga el modelo de pose de YOLOv8
model = YOLO(MODEL_TO_EXPORT)

# Exporta el modelo a formato ONNX
# El argumento 'opset' es importante para la compatibilidad. 12 es un buen valor general.
model.export(format='onnx', opset=12)

print(f"✅ Modelo exportado exitosamente como '{EXPORT_NAME}'")