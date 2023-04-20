"""
Modulo que generainformes de métricas para los dos mejores modelos los cuales
alimentan las visualizaciones para evaluación
"""

import os
import sys

home_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(home_path)

from src.model_management import ClasificadorMora

home_path = os.path.dirname(os.path.dirname(__file__))

path_etiquetas = os.path.join(home_path, 'data', 'etiquetas.csv')
path_info_clientes = os.path.join(home_path, 'data', 'informacion_clientes.csv')
path_hist_transacciones = os.path.join(home_path, 'data', 'historial_transacciones.csv')

if __name__ == '__main__':

    clasificador1 = ClasificadorMora(
        path_etiquetas,
        path_info_clientes,
        path_hist_transacciones
    )
    df_informe1 = clasificador1.obtener_informe_modelo()
    df_informe1.to_csv(os.path.join(home_path,'resultados','informe_clasificador_1.csv'), sep=";", decimal=",")

    clasificador2 = ClasificadorMora(
        path_etiquetas,
        path_info_clientes,
        path_hist_transacciones,
        model_type='hgb'
    )
    df_informe2 = clasificador2.obtener_informe_modelo()
    df_informe2.to_csv(os.path.join(home_path,'resultados','informe_clasificador_2.csv'), sep=";", decimal=",")