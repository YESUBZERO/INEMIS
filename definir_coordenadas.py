import cv2
import json

coordinates = []
video_path = "F:/IDDS/TUMBA_MUERTO/MAH00031.MP4"  # Cambia esto si es necesario

def draw_rectangle(event, x, y, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(coordinates) < 4:
            coordinates.append((x, y))
            print(f"Punto seleccionado: {x}, {y}")

cv2.namedWindow("Seleccionar Coordenadas")
cv2.setMouseCallback("Seleccionar Coordenadas", draw_rectangle)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    for point in coordinates:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
    
    if len(coordinates) == 4:
        cv2.line(frame, coordinates[0], coordinates[1], (255, 0, 0), 2)
        cv2.line(frame, coordinates[1], coordinates[3], (255, 0, 0), 2)
        cv2.line(frame, coordinates[3], coordinates[2], (255, 0, 0), 2)
        cv2.line(frame, coordinates[2], coordinates[0], (255, 0, 0), 2)
    
    cv2.imshow("Seleccionar Coordenadas", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') and len(coordinates) == 4:
        break

cap.release()
cv2.destroyAllWindows()

if len(coordinates) == 4:
    with open("selected_coordinates.json", "w") as f:
        json.dump(coordinates, f)
    print("Coordenadas guardadas en 'selected_coordinates.json'")
else:
    print("No se seleccionaron suficientes puntos.")
