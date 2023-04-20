"""
Modulo para clasificar la mora en el siguiente mes a partir de los archivos
de la información del cliente y el historial de transacciones
"""
import os
import sys

import numpy as np

from dotenv import load_dotenv

home_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(home_path)

from src.model_management import ClasificadorMora, preprocess

load_dotenv()

path_etiquetas = os.path.join(home_path, 'data', 'etiquetas.csv')
path_info_clientes = os.path.join(home_path, 'data', 'informacion_clientes.csv')
path_hist_transacciones = os.path.join(home_path, 'data', 'historial_transacciones.csv')
path_info_clientes_futuro = os.path.join(home_path, 'data', 'informacion_clientes.csv')
path_hist_transacciones_futuro = os.path.join(home_path, 'data', 'historial_transacciones.csv')

if __name__ == '__main__':

    threshold = float(os.getenv('THRESHOLD'))
    model = os.getenv('MODEL')

    X = preprocess(path_info_clientes_futuro,path_hist_transacciones_futuro)

    clasificador1 = ClasificadorMora(
        path_etiquetas,
        path_info_clientes,
        path_hist_transacciones
    )
    clasificador1.fit()
    path_clasificacion = os.path.join(home_path,'resultados',"clasificacion.csv")

    np.savetxt(path_clasificacion,clasificador1.predict(X,threshold=0.2).astype(int),fmt="%d")

