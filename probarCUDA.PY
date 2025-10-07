import torch

# Verifica si PyTorch puede acceder a CUDA
if torch.cuda.is_available():
    print("¡Éxito! CUDA está activado y disponible para PyTorch.")
    # Muestra el nombre de tu GPU
    print(f"Dispositivo detectado: {torch.cuda.get_device_name(0)}")
else:
    print("Error: CUDA no está disponible. Revisa los pasos de instalación.")