# Proyecto de Detección y Clasificación de Vehículos

## Descripción General
Este proyecto utiliza modelos de inteligencia artificial para la detección, clasificación y análisis de vehículos a partir de videos o imágenes. Se implementa un flujo de trabajo que permite extraer imágenes de vehículos, clasificarlos en diferentes categorías y generar un informe con métricas y gráficos de emisiones y consumo de combustible.

## Estructura del Proyecto
El proyecto consta de los siguientes archivos principales:

- `app.py`: Se encarga de la extracción de imágenes desde un video, detectando vehículos mediante YOLO y almacenando la información en un archivo CSV.
- `report.py`: Procesa los datos recogidos en el CSV y genera un informe en HTML con estadísticas y gráficos.
- `constantes.csv`: Contiene los valores de consumo y emisiones de diferentes tipos de vehículos.

## Requisitos Previos
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

### 2. Extracción de Imágenes
Ejecute el siguiente comando para procesar el video y extraer imágenes de los vehículos detectados:

```bash
python app.py
```

Las imágenes extraídas se almacenarán en la carpeta `results/images` y los datos en `vehicle_data.csv`.

### 3. Generación de Reporte
Una vez extraídas las imágenes y clasificados los vehículos, ejecute el siguiente comando para generar el informe en HTML:

```bash
python report.py
```

El informe estará disponible en `results/reporte.html` e incluirá análisis visuales de emisiones y tipos de vehículos detectados.

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
