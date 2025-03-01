# Proyecto de Detección y Clasificación de Vehículos 🚗

## Descripción General
Este proyecto utiliza modelos de inteligencia artificial para la detección, clasificación y análisis de vehículos a partir de videos o imágenes. Se implementa un flujo de trabajo que permite extraer imágenes de vehículos, clasificarlos en diferentes categorías y generar un informe con métricas y gráficos de emisiones y consumo de combustible.

## Estructura del Proyecto 🧱
El proyecto consta de los siguientes archivos principales:

- `app.py`: Se encarga de la extracción de imágenes desde un video, detectando vehículos mediante YOLO y almacenando la información en un archivo CSV.
- `report.py`: Procesa los datos recogidos en el CSV y genera un informe en HTML con estadísticas y gráficos.

## Requisitos Previos ✅
Antes de ejecutar el proyecto, asegúrese de tener instaladas las siguientes dependencias:

```bash
pip install opencv-python pandas matplotlib seaborn watchdog torch ultralytics python-dotenv
```

Además, se requiere tener los modelos de YOLO para detección y clasificación en la carpeta `models/`.

## Uso

### 1. Configurar Variables de Entorno
Crear un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
VIDEO_PATH=ruta/al/video.mp4
RESULTS_DIR=results
MODEL_EXTRACTION_PATH=models/best.pt
MODEL_TIER1_PATH=models/best_cl_t1.pt
MODEL_PC_PATH=models/best_PC.pt
```

### 2. Extracción de Imágenes 📷
Ejecute en el mismo orden los siguientes comandos para procesar el video, extraer y clasificar imágenes de los vehículos detectados:

Este servicio monitorea la carpeta results/images y clasifica las imagenes.
```bash
python app.py #2_tier1_classifier_service
```

Este servicio inicia el proceso de extracion de imagenes (cars, bus, trucks, motorcycle) y las guarda en results/images, al finalizar la extracion realiza el reporte HTML.
```bash
python app.py #1_extraction_service
```

Las imágenes extraídas se almacenarán en la carpeta `results/images` y los datos en `vehicle_data.csv`.

### 3. Generación de Reporte 📄
Una vez extraídas las imágenes y clasificados los vehículos, ejecute el siguiente comando para generar el informe en HTML:

```bash
python report.py
```

El informe estará disponible en `results/reporte.html` e incluirá análisis visuales de emisiones y tipos de vehículos detectados.

## Funcionamiento del Código

### `app.py`
1. Carga un modelo YOLO para detección de vehículos.
2. Procesa un video y extrae imágenes de vehículos detectados.
3. Clasifica los vehículos en categorías y almacena los datos en un CSV.
4. Muestra la detección en tiempo real.

### Algoritmo de Clasificación de Vehículos
El proceso de clasificación se realiza en varias etapas:
1. **Detección de vehículos**: Se usa un modelo YOLO preentrenado para identificar vehículos en las imágenes extraídas del video.
2. **Clasificación Tier1**: Una vez detectados, los vehículos son clasificados en una categoría general (como automóvil, camión, etc.) usando un modelo YOLO específico para esta tarea (`best_cl_t1.pt`).
3. **Subclasificación en PC**: Si el vehículo es clasificado como "PC" (Pasajeros), se utiliza un segundo modelo (`best_PC.pt`) para subclasificarlo en una categoría más específica.
4. **Almacenamiento de resultados**: La información de clasificación se guarda en el archivo `vehicle_data.csv`, incluyendo:
   - Nombre del vehículo
   - Categoría Tier1
   - Subcategoría PC (si aplica)
   - Tipo de combustible y consumo estimado
   - Emisiones estimadas basadas en datos del archivo `constantes.csv`.
5. **Organización de archivos**: Las imágenes se mueven a carpetas específicas según la categoría de clasificación para facilitar su análisis.

### `report.py`
1. Carga los datos del CSV.
2. Genera estadísticas y gráficos de emisiones por tipo de vehículo y combustible.
3. Crea un informe en HTML con los resultados obtenidos.

## Análisis de Resultados
El reporte generado incluye múltiples gráficos que muestran información detallada sobre las emisiones y la clasificación de vehículos:

- **Emisiones por tipo de vehículo**: Gráficos como `BKF_by_vehicle.png`, `CO_by_vehicle.png`, `IDP_by_vehicle.png`, `N2O_by_vehicle.png`, `NH3_by_vehicle.png`, `NOx_by_vehicle.png`, `PM_by_vehicle.png` muestran las emisiones totales de cada contaminante para cada tipo de vehículo.
- **Comparación de emisiones por tipo de combustible**: `fuel_emissions.png` muestra el impacto de diferentes combustibles en las emisiones generales.
- **Cantidad de vehículos detectados por categoría**: `tier1_vehicle_count.png` y `pc_vehicle_count.png` presentan el número de vehículos identificados en cada categoría Tier1 y PC respectivamente.

Estos análisis ayudan a evaluar la contaminación vehicular y proporcionar datos para futuras optimizaciones en el tráfico y la reducción de emisiones.

## Contribuciones
Si desea contribuir a este proyecto, puede realizar un `fork`, implementar mejoras y crear un `pull request` en el repositorio correspondiente.

## Licencia
Este proyecto se distribuye bajo la licencia MIT.

