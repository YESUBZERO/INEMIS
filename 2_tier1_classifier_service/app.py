import os
import cv2
import shutil
import queue
import threading
import pandas as pd
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
import pathlib
# Solución para PosixPath en Windows
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

TIER1_MODEL = os.getenv("MODEL_TIER1_PATH", "models/best_cl_t1.pt")
TIER1_MODEL = os.getenv("MODEL_PC_PATH", "models/best_PC.pt")
results_dir = os.getenv("RESULTS_DIR", "results")
os.makedirs(results_dir, exist_ok=True)

# Cargar modelos
print("Cargando modelos...")
tier1_model = torch.hub.load('ultralytics/yolov5', 'custom', path="models/best_cl_t1.pt", force_reload=True)
pc_model = torch.hub.load('ultralytics/yolov5', 'custom', path="models/best_PC.pt", force_reload=True)
print("Modelos cargados.")

tier1_model.conf = 0.05  # Cambia la confianza al 50%
pc_model.conf = 0.10

# Cargar constantes
constants = pd.read_csv("2_tier1_classifier_service/constantes.csv", sep=";").set_index("Type")

# Crear una cola de procesamiento
task_queue = queue.Queue()

def process_image(file_path):
    """Procesa una imagen, clasifica en Tier1 y PC si aplica y calcula consumo/emisiones."""
    vehicle_name = os.path.basename(file_path)
    tier1_class = "No detection"
    pc_class = "No PC subclass"

    # Clasificación Tier1
    results_tier1 = tier1_model(file_path)
    if len(results_tier1.xyxy[0]) > 0:
        tier1_class = results_tier1.names[int(results_tier1.xyxy[0][0, -1])]

    dest_dir_tier1 = f"{results_dir}/{tier1_class}"
    os.makedirs(dest_dir_tier1, exist_ok=True)
    new_path_tier1 = os.path.join(dest_dir_tier1, vehicle_name)
    shutil.move(file_path, new_path_tier1)

    # Si es PC, realizar subclasificación
    if tier1_class == "PC":
        results_pc = pc_model(new_path_tier1)
        if len(results_pc.xyxy[0]) > 0:
            pc_class = results_pc.names[int(results_pc.xyxy[0][0, -1])]

        dest_dir_pc = f"{dest_dir_tier1}/{pc_class}"
        os.makedirs(dest_dir_pc, exist_ok=True)
        new_path_pc = os.path.join(dest_dir_pc, f"{vehicle_name.split('.')[0]}_{pc_class}.jpg")
        shutil.move(new_path_tier1, new_path_pc)

    # Actualizar CSV
    vehicle_data_csv = os.path.join(results_dir, "vehicle_data.csv")
    df = pd.read_csv(vehicle_data_csv)
    if vehicle_name in df["vehicle"].values:
        df.loc[df["vehicle"] == vehicle_name, "classifier-tier1"] = tier1_class
        df.loc[df["vehicle"] == vehicle_name, "classifier-PC"] = pc_class if tier1_class == "PC" else None

        # Calcular consumo de combustible y emisiones
        if tier1_class in constants.index:
            df.loc[df["vehicle"] == vehicle_name, "Fuel"] = constants.at[tier1_class, "Fuel"]
            df.loc[df["vehicle"] == vehicle_name, "Fuel_Consumption"] = (float(constants.at[tier1_class, "CONSUMO (g/km)"]) * 1) / 1000
            
            # Calcular emisiones solo si los valores existen
            emission_columns = ["CO", "NMVOC", "NOx", "PM", "N2O", "NH3", "IDP", "BKF"]
            for col in emission_columns:
                col_name = f"{col} (g/kg fuel)"
                if col_name in constants.columns:
                    df.loc[df["vehicle"] == vehicle_name, col] = float(constants.at[tier1_class, col_name]) * df.loc[df["vehicle"] == vehicle_name, "Fuel_Consumption"]

        df.to_csv(vehicle_data_csv, index=False)

class ImageEventHandler(FileSystemEventHandler):
    """Detecta nuevas imágenes en la carpeta y las añade a la cola de procesamiento."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".jpg", ".jpeg", ".png")):
            print(f"Nueva imagen detectada: {event.src_path}")
            task_queue.put(event.src_path)

def worker():
    """Ejecuta tareas en la cola."""
    while True:
        file_path = task_queue.get()
        if file_path is None:
            break
        process_image(file_path)
        task_queue.task_done()

# Iniciar hilos de procesamiento
num_threads = 5
threads = []
for _ in range(num_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# Monitorear la carpeta de imágenes
input_dir = os.path.join(results_dir, "images")
os.makedirs(input_dir, exist_ok=True)
observer = Observer()
event_handler = ImageEventHandler()
observer.schedule(event_handler, path=input_dir, recursive=False)
observer.start()

print(f"Monitoreando la carpeta: {input_dir} en tiempo real...")

try:
    while True:
        task_queue.join()
except KeyboardInterrupt:
    print("Deteniendo el servicio...")
    observer.stop()
    observer.join()
    
    # Detener los hilos
    for _ in range(num_threads):
        task_queue.put(None)
    for t in threads:
        t.join()
    print("Servicio detenido correctamente.")
