import pandas as pd
import numpy as np

def calculate_descriptive_stats(data):
    """
    Calcula estadísticas descriptivas para campos numéricos relevantes
    """
    numeric_columns = ['edad', 'temperatura', 'tas', 'tad', 'fc', 'fr', 'glasgow']
    stats = {}
    
    for col in numeric_columns:
        if col in data.columns:
            column_stats = {
                'media': round(data[col].mean(), 2),
                'mediana': round(data[col].median(), 2),
                'moda': round(data[col].mode().iloc[0], 2) if not data[col].mode().empty else None,
                'desviacion_tipica': round(data[col].std(), 2),
                'varianza': round(data[col].var(), 2),
                'min': round(data[col].min(), 2),
                'max': round(data[col].max(), 2)
            }
            stats[col] = column_stats
    
    return stats

def format_stats_table(stats):
    """
    Formatea las estadísticas para mostrarlas en una tabla HTML
    """
    html = """
    <div class="table-responsive">
        <table class="table table-striped table-hover">
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Media</th>
                    <th>Mediana</th>
                    <th>Moda</th>
                    <th>Desv. Típica</th>
                    <th>Varianza</th>
                    <th>Mínimo</th>
                    <th>Máximo</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for var_name, var_stats in stats.items():
        html += f"""
            <tr>
                <td>{var_name}</td>
                <td>{var_stats['media']}</td>
                <td>{var_stats['mediana']}</td>
                <td>{var_stats['moda']}</td>
                <td>{var_stats['desviacion_tipica']}</td>
                <td>{var_stats['varianza']}</td>
                <td>{var_stats['min']}</td>
                <td>{var_stats['max']}</td>
            </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    return html

def calculate_kpis(data):
    """
    Calcula los KPIs principales
    """
    kpis = {
        'total_casos': len(data),
        'casos_confirmados': len(data[data['clasfinal'] > 0]),
        'casos_semanales_promedio': round(len(data) / data['semana'].nunique(), 2),
        'casos_pediatricos': len(data[data['edad'] < 18])
    }
    return kpis

def generate_plots(data):
    """
    Genera todas las gráficas necesarias para el dashboard
    """
    plots = {}
    
    # Casos por semana
    casos_semana = data.groupby('semana').size().reset_index(name='casos')
    plots['casos_semana'] = {
        'data': [{'x': casos_semana['semana'], 'y': casos_semana['casos'], 'type': 'line'}],
        'layout': {'title': 'Casos por Semana'}
    }
    
    # Distribución por género
    genero = data['sexo'].value_counts()
    plots['genero'] = {
        'data': [{'labels': genero.index, 'values': genero.values, 'type': 'pie'}],
        'layout': {'title': 'Distribución por Género'}
    }
    
    # ... Agregar más gráficas según sea necesario ...
    
    return {k: plot_to_json(v) for k, v in plots.items()}

def plot_to_json(plot_dict):
    """
    Convierte el diccionario de la gráfica a JSON
    """
    import json
    return json.dumps(plot_dict)