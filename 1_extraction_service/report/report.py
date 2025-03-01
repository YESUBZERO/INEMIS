import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generar_reporte_html(csv_file, output_dir, output_html):
    # Cargar datos
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['classifier-tier1'])

    # Crear carpeta para imágenes
    img_folder = os.path.join(output_dir, "graficas")
    os.makedirs(img_folder, exist_ok=True)

    # Análisis de datos
    resumen = df.describe().to_html(classes='table table-striped', border=0)

    # Lista de gráficos
    plots = []

    # Gráficos de cantidad de emisión por tipo de vehículo
    emission_types = ['CO', 'NOx', 'PM', 'N2O', 'NH3', 'IDP', 'BKF']
    for emission in emission_types:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=df['classifier-tier1'], y=df[emission], estimator=sum, errorbar=None, hue=df['classifier-tier1'], legend=False, palette='viridis')
        plt.xlabel('Tipo de Vehículo')
        plt.ylabel(f'{emission} Total (g/km)')
        plt.title(f'Total de {emission} por Tipo de Vehículo')

        # Corregir la ruta y guardarla con formato accesible
        emission_plot = os.path.join(img_folder, f'{emission}_by_vehicle.png')
        plt.savefig(emission_plot, bbox_inches='tight')
        plt.close()

        # Convertir la ruta en relativa para que funcione en HTML
        plots.append(os.path.relpath(emission_plot, start=os.path.dirname(output_html)).replace("\\", "/"))

    # Comparación de emisiones por tipo de combustible
    plt.figure(figsize=(8, 6))
    emissions_fuel = df.groupby('Fuel')[['CO', 'NOx', 'PM', 'N2O', 'NH3', 'IDP', 'BKF']].sum()
    emissions_fuel.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.xlabel('Tipo de Combustible')
    plt.ylabel('Total de Emisiones (g/km)')
    plt.title('Comparación de Emisiones por Tipo de Combustible')

    fuel_emission_plot = os.path.join(img_folder, 'fuel_emissions.png')
    plt.savefig(fuel_emission_plot, bbox_inches='tight')
    plt.close()
    plots.append(os.path.relpath(fuel_emission_plot, start=os.path.dirname(output_html)).replace("\\", "/"))

    # Gráfico de cantidad de vehículos detectados por classifier-tier1
    plt.figure(figsize=(8, 6))
    tier1_counts = df['classifier-tier1'].value_counts()
    tier1_counts.plot(kind='bar', color='blue', alpha=0.7)
    plt.xlabel('Categoría de Vehículo (Tier1)')
    plt.ylabel('Cantidad de Vehículos Detectados')
    plt.title('Cantidad de Vehículos Detectados por Categoría (Tier1)')

    tier1_count_plot = os.path.join(img_folder, 'tier1_vehicle_count.png')
    plt.savefig(tier1_count_plot, bbox_inches='tight')
    plt.close()
    plots.append(os.path.relpath(tier1_count_plot, start=os.path.dirname(output_html)).replace("\\", "/"))

    # Gráfico de cantidad de vehículos detectados por classifier-PC
    plt.figure(figsize=(8, 6))
    pc_counts = df['classifier-PC'].value_counts()
    pc_counts.plot(kind='bar', color='red', alpha=0.7)
    plt.xlabel('Categoría de Vehículo (PC)')
    plt.ylabel('Cantidad de Vehículos Detectados')
    plt.title('Cantidad de Vehículos Detectados por Categoría (PC)')

    pc_count_plot = os.path.join(img_folder, 'pc_vehicle_count.png')
    plt.savefig(pc_count_plot, bbox_inches='tight')
    plt.close()
    plots.append(os.path.relpath(pc_count_plot, start=os.path.dirname(output_html)).replace("\\", "/"))

    # Generar el contenido del reporte en HTML con las rutas corregidas
    html_content = f"""
    <html>
    <head>
        <title>Reporte de Análisis de Datos</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; margin: 10px; background-color: #ffffff; color: #333; }}
            h1, h2 {{ color: #2C3E50; text-align: center; }}
            .table-container {{ width: 90%; margin: 0 auto; overflow-x: auto; }}
            .table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 5px; text-align: center; }}
            .table th {{ background-color: #2C3E50; color: white; }}
            .image-grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }}
            .image-grid img {{ width: 25%; border: 1px solid #2C3E50; border-radius: 5px; }}
            .description {{ text-align: center; font-size: 14px; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Análisis de Datos del CSV</h1>
        
        <div class="image-grid">
            {''.join([f"<img src='{plot}' alt='{plot}'>" for plot in plots])}
        </div>
        
        <h2>Resumen de Datos</h2>
        <div class="table-container">
            {resumen}
        </div>
        <p class="description">Esta tabla proporciona un resumen estadístico de los datos, incluyendo valores como la media, desviación estándar, valores mínimos y máximos. Es útil para entender la variabilidad y distribución de los datos.</p>
    </body>
    </html>
    """

    # Guardar el HTML
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Reporte HTML generado: {output_html}")
