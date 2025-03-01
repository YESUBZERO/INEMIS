import os
import cv2
import csv
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from report import report

def normalize_bbox(frame, x1, y1, x2, y2, target_size=224):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    half_size = target_size // 2
    
    x1_new = max(center_x - half_size, 0)
    x2_new = min(center_x + half_size, frame.shape[1])
    y1_new = max(center_y - half_size, 0)
    y2_new = min(center_y + half_size, frame.shape[0])
    
    return x1_new, y1_new, x2_new, y2_new

def extract_images(video_path, output_dir, coords, tol, csv_file_dir, target_size=224, truck_size=320):
    os.makedirs(output_dir, exist_ok=True)
    csv_file = csv_file_dir
    
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vehicle", "classifier-tier1", "classifier-PC"])

    model_path = os.getenv("MODEL_PATH", "models/best.pt")
    model = YOLO(model_path)
    model.to('cuda')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    car_data = {}
    img_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_copy = frame.copy()
        result = model.track(source=frame, persist=True, conf=0.05, iou=0.2, show=False, classes=[2,3,5,7], max_det=1000)
        result = result[0]
        boxes = result.boxes.xyxy
        ids = result.boxes.id
        class_ids = result.boxes.cls  # Obtener las clases detectadas

        if boxes is None or ids is None:
            continue

        for box, track_id, class_id in zip(boxes, ids, class_ids):
            track_id = int(track_id.item())
            class_id = int(class_id.item())
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Determinar tamaño según tipo de vehículo
            bbox_size = truck_size if class_id == 7 or class_id == 5 else target_size
            x1_norm, y1_norm, x2_norm, y2_norm = normalize_bbox(frame, x1, y1, x2, y2, bbox_size)
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            x, y = midpoint

            for coord in coords:
                if abs(y - coord[0][1]) < tol and coord[0][0] < x < coord[2][0]:
                    if not car_data.get(track_id, {}).get("saved", False):
                        img_name = f"vehicle_{img_counter}.jpg"
                        img_path = os.path.join(output_dir, img_name)
                        cv2.imwrite(img_path, frame_copy[y1_norm:y2_norm, x1_norm:x2_norm])

                        with open(csv_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([img_name, "", ""])

                        img_counter += 1
                        car_data[track_id] = {"saved": True}
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Procesamiento de Video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Extracción completada. Se guardaron {img_counter} imágenes en {output_dir}")

if __name__ == "__main__":
    load_dotenv()
    video_path = os.getenv("VIDEO_PATH", "shared/input/video.MP4")
    output_dir = os.getenv("OUTPUT_DIR", "shared/images")
    vehicle_data_csv = os.getenv("OUTPUT_CSV", "shared/vehicle_data.csv")
    coords = [[[13, 322], [221, 231], [853, 566], [932, 278]]]
    tol = int(os.getenv("TOL", 10))
    target_size = int(os.getenv("TARGET_SIZE", 224))
    truck_size = int(os.getenv("TRUCK_SIZE", 448))

    extract_images(video_path, output_dir, coords, tol, vehicle_data_csv, target_size, truck_size)
    # Uso del script
    report.generar_reporte_html(vehicle_data_csv, "reporte.html")
