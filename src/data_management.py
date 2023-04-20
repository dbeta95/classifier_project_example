"""
Modulo con las funciones definidas para le manejo de datos del proyecto.
"""
import pandas as pd

def obtener_datos(path_etiquetas:str, path_info_clientes:str, path_hist_transacciones:str):
    """
    Función para obtener un DataFrame único con la información a nivel de cada cliente y la etiqueta
    de si entró o no en mora el mes siguiente a partir de los archivos con la información de la mora,
    la información del cliente y el historial transaccional.

    Args:
    ----------
    path_etiquetas:str
        Archivo (con su ruta) en que se encuentran los clientes y su respectiva etiqueta de si entra o no en mora        
    path_info_clientes:str
        Archivo (con su ruta) en que se encuentra la información de los clientes
    path_hist_transacciones:str
        Archivo (con su ruta) en que se encuentra el historial transaccional de los últimos 6 meses del cliente

    Results:
    ---------
        pd.DataFrame
            DataFrame único con la información a nivel de cada cliente y la etiqueta de si entró o no en mora el 
            mes siguiente
    """
    # Se obtienen los datos de las etiquetas de mora
    df_etiquetas = pd.read_csv(path_etiquetas, sep='|')
    df_etiquetas = df_etiquetas.set_index('CLIENT_ID')

    # Se obtienen los datos de la información de los clientes
    df_info_clientes = pd.read_csv(path_info_clientes, sep='|')
    df_info_clientes = df_info_clientes.set_index('CLIENT_ID')

    # Se obtienen los datos dle historial transaccional
    df_hist_transacciones = pd.read_csv(path_hist_transacciones, sep='|')
    # Se lleva la información a registro únicos por cliente
    df_ht = df_hist_transacciones.pivot(
        index='CLIENT_ID', columns='MONTH', values=['RETRASO_PAGO_ESTADO', 'DEUDA_MES', 'PAGO_MES']
    )
    df_ht.columns = [column[0]+'_'+str(column[1]) for column in df_ht.columns]
    
    # Se unen los tres conjuntos de datos usando el ID del cliente
    df = df_info_clientes.merge(df_ht, left_index=True, right_index=True).merge(df_etiquetas, left_index=True, right_index=True)

    return df

def obtener_datos_futuros(path_info_clientes:str, path_hist_transacciones:str):
    """
    Función para obtener un DataFrame único con la información a nivel de cada cliente a partir de 
    los archivos con la información del cliente y el historial transaccional.

    Args:
    ----------
    path_info_clientes:str
        Archivo (con su ruta) en que se encuentra la información de los clientes
    path_hist_transacciones:str
        Archivo (con su ruta) en que se encuentra el historial transaccional de los últimos 6 meses del cliente

    Results:
    ---------
        pd.DataFrame
            DataFrame único con la información a nivel de cada cliente
    """
    # Se obtienen los datos de la información de los clientes
    df_info_clientes = pd.read_csv(path_info_clientes, sep='|')
    df_info_clientes = df_info_clientes.set_index('CLIENT_ID')

    # Se obtienen los datos dle historial transaccional
    df_hist_transacciones = pd.read_csv(path_hist_transacciones, sep='|')
    # Se lleva la información a registro únicos por cliente
    df_ht = df_hist_transacciones.pivot(
        index='CLIENT_ID', columns='MONTH', values=['RETRASO_PAGO_ESTADO', 'DEUDA_MES', 'PAGO_MES']
    )
    df_ht.columns = [column[0]+'_'+str(column[1]) for column in df_ht.columns]
    
    # Se unen los tres conjuntos de datos usando el ID del cliente
    df = df_info_clientes.merge(df_ht, left_index=True, right_index=True)

    return df