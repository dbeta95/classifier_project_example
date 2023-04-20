"""
Modulo con de clases y funciones para el entrenamiento y pronóstico.
"""
import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Optional, Any
from sklearn.metrics import (
    mean_squared_error, 
    accuracy_score, 
    recall_score
)
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from src.data_management import obtener_datos, obtener_datos_futuros

src_path = os.path.dirname(__file__)

def obtener_threshold(
        y_true:np.ndarray, 
        y_proba:np.ndarray,
        metric:Optional[Any]=accuracy_score,
        greater_is_better:Optional[bool]=False,
        **kwargs
    ):
    
    """
    Función que permite hallar el threshold óptimo para maximizar una métrica de clasificación.

    Args:
    ----------
    y_true:np.ndarray
        Arreglo unidimensional con los valores reales de las etiquetas
    y_proba:np.ndarray
        Arreglo unidimensional con los valores predichos para la probabilidad de la etiqueta 1.
    metric:Optional[Any]=accuracy_score
        Métrica a optimizar con la selección del threshold
    greater_is_better:Optional[bool]=False
        Indicador de si la metrica debe maximizarse o minimizarse
    """

    grid = np.arange(0.0,1.01,0.01)

    if greater_is_better:
        t_index = np.argmax(list(map(
            lambda threshold: metric(y_true,np.where(y_proba >= threshold, 1.0, 0.0), **kwargs), grid
        )))
    else:
        t_index = np.argmin(list(map(
            lambda threshold: metric(y_true,np.where(y_proba >= threshold, 1.0, 0.0), **kwargs), grid
        )))

    threshold = grid[t_index]

    return threshold

def evaluar_clasificador(y_true:np.ndarray,y_pred:np.ndarray,verbose:Optional[bool]=True):
    """
    Función que arroja métricas y gráficas de evaluación de la clasificación

    Args:
    ----------
    y_true:np.ndarray
        Arreglo unidimensional con los valores reales de las etiquetas
    y_pred:np.ndarray
        Arreglo unidimensional con los valores predichos para la probabilidad de la etiqueta 1.
    verbose:bool
        Define si deben imprimirse las metricas en pantalla

    Results:
        (float,float,float)
        Valores de las tres métricas
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if verbose:
        print(f"Se obtienen las metricas:\n   - RMSE: {rmse}\n   - Accuracy: {accuracy}\n   - recall: {recall}")
    return rmse,accuracy,recall
   

def obtener_simulacion_perdidas(
    df_probabilidades:pd.DataFrame, 
    umbrales:Optional[np.ndarray] = np.arange(0.1,0.6,0.1)
):
    """
    Función que permite simular las perdidas totales de un portafolio y las perdidas
    por grupos condicinados por superar umbrales de probabilidad de default.

    Args:
    ----------
    df_probabilidades:pd.DataFrame
        DataFrame con las probabilidades de mora para todos los clientes en la base 
        de datos con sus respectivos ID
    umbrales:Optional[np.ndarray] = np.arange(0.1,0.6,0.1)
        Serie de umbrales que separan los grupos.

    Results:
    ---------
    list:
        Lista con las perdidas totales en primera posición y las demás para los
        subgrupos, teniendo tantos elementos como la longitud de umbrales más uno.
    """
    datos = df_probabilidades.to_numpy()
    N = datos.shape[0]
    lj = datos[:,0]
    pj = datos[:,1]
    rs_sim = np.random.random(N)
    Dj = np.where(rs_sim<=pj,1,0)
    xj = lj*Dj
    L = np.sum(xj)

    perdidas_grupos = {
        f"L{np.round(umbral,2)}":np.sum(np.where(pj>umbral, xj, 0))
        for umbral in umbrales
    }

    return [L] + [perdida for perdida in perdidas_grupos.values()]

def obtener_exposicion_riesgo(
    df_probabilidades:pd.DataFrame, 
    umbrales:Optional[np.ndarray] = np.arange(0.1,0.6,0.1),
    n:int=10000
):
    """
    Función que permite obtener las métricas de exposición al riesgopor grupos determinados
    por distintos umbrales.

    Args:
    ----------
    df_probabilidades:pd.DataFrame
        DataFrame con las probabilidades de mora para todos los clientes en la base 
        de datos con sus respectivos ID
    umbrales:Optional[np.ndarray] = np.arange(0.1,0.6,0.1)
        Serie de umbrales que separan los grupos.
    n:int=10000
        Número de simulaciones a realizar

    Results:
    ---------
    pd:DataFrame
        DataFrame con la información del riesgo, sus columnas son:
        - p: El Umbral que se utilzia para determinar el grupo
        - VaR: Valor en riesgo del portafolio. Se usa percentil 95
        - deuda_total: Monto total de la deuda del portafolio
        - VaR_grupo: Valor en riesgo marginal del grupo
        - deuda_grupo: Valor total de la deudaen el grupo
        - relacion_VaR: Porcentaje del VaR total que aporta el grupo
        - relacion_deuda: Porcentaje de la deuda totalque aporta el grupo
        - relacion_r_d_grupo: Porcentaje del capital en deuda del grupo que se consedera valor en riesgo
    """

    sim_res = np.array(list(
        map(lambda x: obtener_simulacion_perdidas(df_probabilidades, umbrales),range(n))
    ))

    df_VaRs = pd.DataFrame({
        "p":[],
        "VaR":[],
        "deuda_total":[],
        "VaR_grupo":[]
    })
    df_row = 0

    VaR = np.quantile(sim_res[:,0],0.95)
    deuda_total = df_probabilidades['deuda'].sum()
    for grupo in np.arange(1,sim_res.shape[1]):
        VaR_grupo = np.sum(
            np.where(sim_res[:,0]>VaR,sim_res[:,grupo],0.0)    
        )/np.sum(
            np.where(sim_res[:,0]>VaR,1.0,0.0)
        )
        df_VaRs.loc[df_row] = [umbrales[grupo-1], VaR, deuda_total, VaR_grupo]
        df_row+= 1

    df_VaRs['deuda_grupo'] = [
        df_probabilidades['deuda'][df_probabilidades['probabilidad'] > umbral].sum()
        for umbral in umbrales
    ]

    df_VaRs['relacion_VaR'] = df_VaRs['VaR_grupo']/df_VaRs['VaR']
    df_VaRs['relacion_deuda'] = df_VaRs['deuda_grupo']/df_VaRs['deuda_total']

    df_VaRs['relacion_r_d_grupo'] = df_VaRs['VaR_grupo']/df_VaRs['deuda_grupo']

    return df_VaRs

class CategoricalEncoder():
    """
    Clase que define un calsificador que realiza una transformación de una variable categorica a multiples
    variables dummy (One Hot enconding) y devuelve el codigicador y la codificación

    """

    def __init__(self)->None:
        """
        Método de instanciación de clase
        """
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False)

    def fit(self, x:np.ndarray) -> None:
        """
        Método de ajuste del encoder.
        
        Args:
        ----------
        x:np.ndarray
            Arreglo unidimensional con las observaciones de la variable categorica
        """
        label_encoded = self.label_encoder.fit_transform(x).reshape(-1,1)
        self.onehot_encoder.fit(label_encoded)

    def fit_transform(self, x:np.ndarray) -> np.ndarray:
        """
        Método de ajuste del encoder y transformación de las observaciones de la variable
        categorica.
        
        Args:
        ----------
        x:np.ndarray
            Arreglo unidimensional con las observaciones de la variable categorica

        Results:
        ---------
        np.ndarray
            Arreglo multidimensional con el resultado de la transformación
        """
        label_encoded = self.label_encoder.fit_transform(x).reshape(-1,1)
        onehot_encoded = self.onehot_encoder.fit_transform(label_encoded)

        return onehot_encoded
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        """
        Método de transformación de las observaciones de la variable categorica.
        
        Args:
        ----------
        x:np.ndarray
            Arreglo unidimensional con las observaciones de la variable categorica

        Results:
        ---------
        np.ndarray
            Arreglo multidimensional con el resultado de la transformación
        """
        label_encoder = self.label_encoder.transform(x).reshape(-1,1)
        onehot_encoded = self.onehot_encoder.transform(label_encoder)

        return onehot_encoded
    
    def inverse_transform(self, X:np.ndarray) -> np.ndarray:   
        """
        Método de transformación inversa de las codificaciones a la variable original.
        
        Args:
        ----------
        X:np.ndarray
            Arreglo multidimensional con el resultado de la transformación           

        Results:
        ---------
        np.ndarray
            Arreglo unidimensional con las observaciones de la variable categorica
        """

        return self.label_encoder.inverse_transform(self.onehot_encoder.inverse_transform(X).reshape(-1,))
    
def codificar_categoricas(
            X:np.ndarray
        ):
        """
        Método que codifica las variables categoricas del conjunto de datos
        en columnas de variables indicadoras

        Args:
        ----------
            X:np.ndarray
                Arreglo multidimensional con las observaciones de las variables independientes
        """
        genero = X[:,1]
        genero_encoder = CategoricalEncoder()
        genero_encoded = genero_encoder.fit_transform(genero)

        educacion = X[:,2]
        educacion_encoder = CategoricalEncoder()
        educacion_encoded = educacion_encoder.fit_transform(educacion)

        estado_civil = X[:,3]
        estado_civil_encoder = CategoricalEncoder()
        estado_civil_encoded = estado_civil_encoder.fit_transform(estado_civil)

        X = np.delete(X,[1,2,3],1)
        X = np.concatenate([X,genero_encoded], axis=1)
        X = np.concatenate([X,educacion_encoded], axis=1)
        X = np.concatenate([X,estado_civil_encoded], axis=1)
        
        return X

def preprocess(
        path_info_clientes:str, 
        path_hist_transacciones:str,):
    """
    Método que preprocesa los datos y devuelve los conjuntos de entrenamiento y prueba
    """

    df = obtener_datos_futuros(path_info_clientes, path_hist_transacciones)
    X = df.to_numpy()
    
    X = codificar_categoricas(X)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X
    
class ClasificadorMora():
    """
    Clase que define el clasificador de mora y que arrojará las probabilidades
    de mora y las etiquetas, así como métricas de evaluación
    """

    def __init__(self,
        path_etiquetas:str, 
        path_info_clientes:str, 
        path_hist_transacciones:str,
        test_size:Optional[float]=0.2,
        balance_strategy:Optional[str]=None,
        model_type:Optional[str]='xgb'
    ):
        """
        Método de instanciación de clase.

        Args:
        ----------
        path_etiquetas:str
            Archivo (con su ruta) en que se encuentran los clientes y su respectiva etiqueta de si entra o no en mora        
        path_info_clientes:str
            Archivo (con su ruta) en que se encuentra la información de los clientes
        path_hist_transacciones:str
            Archivo (con su ruta) en que se encuentra el historial transaccional de los últimos 6 meses del cliente
        test_size:float
            Proporción de elementos a incluir en el conjunto de evaluación
        balance_srategy:Optional[str]=None
            Estrategia de balanceo si se desea usar una. Las opciones son "over_sample" y "under_sample".
        model_type:Optional[str]='xgb'
            Modelo a utilizar para la clasificación. las opciones son "xgb" y "hgb"
        """

        self.path_etiquetas = path_etiquetas
        self.path_info_clientes = path_info_clientes
        self.path_hist_transacciones = path_hist_transacciones
        self.test_size = test_size

        if balance_strategy is not None:
            assert balance_strategy.lower() in ["over_sample", "under_sample"], "Las opciones válidas para el balance strategy son 'over_sample' y 'under_sample'."
        self.balance_strategy = balance_strategy

        assert model_type in ["xgb", "hgb"], "El modelo elegido no es valido, las opciones son 'xgb' y 'hgb'"
        self.model_type = model_type

    def codificar_categoricas(self,
            X:np.ndarray
        ):
        """
        Método que codifica las variables categoricas del conjunto de datos
        en columnas de variables indicadoras

        Args:
        ----------
            X:np.ndarray
                Arreglo multidimensional con las observaciones de las variables independientes
        """
        genero = X[:,1]
        self.genero_encoder = CategoricalEncoder()
        genero_encoded = self.genero_encoder.fit_transform(genero)

        educacion = X[:,2]
        self.educacion_encoder = CategoricalEncoder()
        educacion_encoded = self.educacion_encoder.fit_transform(educacion)

        estado_civil = X[:,3]
        self.estado_civil_encoder = CategoricalEncoder()
        estado_civil_encoded = self.estado_civil_encoder.fit_transform(estado_civil)

        X = np.delete(X,[1,2,3],1)
        X = np.concatenate([X,genero_encoded], axis=1)
        X = np.concatenate([X,educacion_encoded], axis=1)
        X = np.concatenate([X,estado_civil_encoded], axis=1)
        
        return X
    
    def preprocess(self):
        """
        Método que preprocesa los datos y devuelve los conjuntos de entrenamiento y prueba
        """

        df = obtener_datos(self.path_etiquetas, self.path_info_clientes, self.path_hist_transacciones)
        self.deudas = df['DEUDA_MES_9']
        data = df.to_numpy()
        self.ids = df.index
        X,y = data[:,:-1], data[:,-1].astype('int')

        X = self.codificar_categoricas(X)
        
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        self.X = X

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=self.test_size, random_state=1)

        if self.balance_strategy is not None:
            if self.balance_strategy.lower() == 'over_sample':
                over_sampler = SMOTE(random_state=1)
                X_train, y_train = over_sampler.fit_resample(X_train, y_train)
            
            if self.balance_strategy.lower() == 'under_sample':
                under_sampler = RandomUnderSampler(random_state=1)
                X_train, y_train = under_sampler.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test
    
    def entrenar_xgb(self):
        """
        Método que realiza el entrenamiento de un clasificador XGB
        """

        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()

        with open(os.path.join(src_path,'parameters','xgb_parameters.json')) as json_file:
            parameters = json.load(json_file)

        self.model = XGBClassifier(objective="binary:logistic",random_state=1,**parameters)
        self.model.fit(self.X_train, self.y_train)

        y_proba = self.model.predict_proba(self.X_test)[:,1].reshape(-1,)
        self.threshold = obtener_threshold(self.y_test, y_proba,mean_squared_error, greater_is_better=False,**{"squared":False})

    def entrenar_hgb(self):
        """
        Método que realiza el entrenamiento de un clasificador HGB
        """

        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()

        with open(os.path.join(src_path,'parameters','hb_parameters.json')) as json_file:
            parameters = json.load(json_file)

        self.model = HistGradientBoostingClassifier(random_state=1,**parameters)
        self.model.fit(self.X_train, self.y_train)

        y_proba = self.model.predict_proba(self.X_test)[:,1].reshape(-1,)
        self.threshold = obtener_threshold(self.y_test, y_proba,mean_squared_error, greater_is_better=False,**{"squared":False})

    def fit(self):
        """
        Método de ajuste del modelo seleccionado
        """
        if self.model_type == "xgb":
            self.entrenar_xgb()
        if self.model_type == "hgb":
            self.entrenar_hgb()

    def predict_proba(self,X:np.ndarray):
        """
        Método que predice la probabilidad de mora para un conjunto de datos

        Args:
        ----------
        X:np.ndarray
            Conjunto de variables independientes pre procesadas

        Results:
        ----------
        np.ndarray
            Arreglo bidimensional con las probabilidades pronosticadas

        """
        return self.model.predict_proba(X)
    
    def predict(self,X:np.ndarray,threshold:Optional[float]=None):
        """
        Método que predice la etiqueta de mora para un conjunto de datos

        Args:
        ----------
        X:np.ndarray
            Conjunto de variables independientes pre procesadas
        threshold:Optional[float]=None
            Trheshold a utilizar para definir cuandola etiqueta se determina 1.

        Results:
        ----------
        np.ndarray
            Arreglo unidimensional con las etiquetas

        """
        if threshold is None:
            threshold = self.threshold
        y_proba = self.model.predict_proba(X)[:,1].reshape(-1,)
        y_pred = np.where(y_proba > threshold, 1., 0.)
        return y_pred
    
    def evaluar(self, verbose:Optional[bool]=True):
        """
        Método que arroja métricas de evaluación del modelo

        Args:
        ---------
        verbose:Optional[bool]=True
            Define si deben imprimirse las metricas en pantalla

        Results:
        ----------
        float
            RMSE del conjunto de entrenamiento con threshold óptimo
        float
            Accuracy del conjunto de entrenamiento con threshold óptimo
        float
            Recall del conjunto de entrenamiento con threshold óptimo
        float
            RMSE del conjunto de prueba con threshold óptimo
        float
            Accuracy del conjunto de prueba con threshold óptimo
        float
            Recall del conjunto de prueba con threshold óptimo
        """

        self.fit()
        y_pred_train = self.predict(self.X_train)
        y_pred_test = self.predict(self.X_test)

        rmse_train,accuracy_train,recall_train = evaluar_clasificador(self.y_train, y_pred_train, verbose=False)
        rmse_test,accuracy_test,recall_test = evaluar_clasificador(self.y_test, y_pred_test, verbose=False)

        if verbose:
            print(
                f"Se obtienen las metricas de entrenamiento:\n   - RMSE: {rmse_train}\n   - Accuracy: {accuracy_train}\n   - recall: {recall_train}\n\n",
                f"Se obtienen las metricas de prueba:\n   - RMSE: {rmse_test}\n   - Accuracy: {accuracy_test}\n   - recall: {recall_test}"
                )

        return rmse_train,accuracy_train,recall_train,rmse_test,accuracy_test,recall_test
    
    def evaluar_threshold(self,threshold:float, verbose:Optional[bool]=True):
        """
        Método paraobtener métricas de evaluación utilizando un threshold arbitrario

        Args:
        ----------
        threshold:float
            Trheshold a utilizar para definir cuandola etiqueta se determina 1.
        verbose:Optional[bool]=True
            Define si deben imprimirse las metricas en pantalla

        Results:
        ----------
        float
            RMSE del conjunto de entrenamiento con threshold arbitrario
        float
            Accuracy del conjunto de entrenamiento con threshold arbitrario
        float
            Recall del conjunto de entrenamiento con threshold arbitrario
        float
            RMSE del conjunto de prueba con threshold arbitrario
        float
            Accuracy del conjunto de prueba con threshold arbitrario
        float
            Recall del conjunto de prueba con threshold arbitrario
        """

        y_proba_train = self.predict_proba(self.X_train)[:,1].reshape(-1,)
        y_proba_test = self.predict_proba(self.X_test)[:,1].reshape(-1,)

        y_pred_train = np.where(y_proba_train > threshold, 1., 0.)
        y_pred_test = np.where(y_proba_test > threshold, 1., 0.)

        rmse_train,accuracy_train,recall_train = evaluar_clasificador(self.y_train, y_pred_train, verbose=False)
        rmse_test,accuracy_test,recall_test = evaluar_clasificador(self.y_test, y_pred_test, verbose=False)

        if verbose:
            print(
                f"Se utiliza un threshold de {threshold}\n\n"
                f"   Se obtienen las metricas de entrenamiento:\n   - RMSE: {rmse_train}\n   - Accuracy: {accuracy_train}\n   - recall: {recall_train}\n\n",
                f"   Se obtienen las metricas de prueba:\n   - RMSE: {rmse_test}\n   - Accuracy: {accuracy_test}\n   - recall: {recall_test}"
                )

        return rmse_train,accuracy_train,recall_train,rmse_test,accuracy_test,recall_test
    
    def obtener_probabilidades_clientes(self):
        """
        Método que permite obtener las probabilidades de mora para todos los clientes en la base de datos
        con sus respectivos ID
        """
        probabilidades = self.predict_proba(self.X)[:,1].reshape(-1,)

        df = pd.DataFrame({            
            'deuda':self.deudas,
            'probabilidad':probabilidades
        }, index=self.ids)

        return df
    
    def obtener_metricas_deciles(self,deciles:Optional[np.ndarray] = np.arange(0.1,0.6,0.1)):
        """
        Método que permite obtener las métricas del modelo clasificando con
        diferentes deciles

        Args:
        ----------
        deciles:Optional[np.ndarray] = np.arange(0.1,0.6,0.1)
            Deciles a utilziar para evaluar además del threshold óptimo

        Results:
        ----------
        pd.Dataframe
            Dataframe con la información de las métricas para cada decil
        """
        df_metricas_deciles = pd.DataFrame({
            "p":[],
            "rmse_train":[],
            "accuracy_train":[],
            "recall_train":[],
            "rmse_test":[],
            "accuracy_test":[],
            "recall_test":[]
        })

        rmse_train,accuracy_train,recall_train,rmse_test,accuracy_test,recall_test = self.evaluar()

        df_metricas_deciles.loc[0] = [
            self.threshold,rmse_train,accuracy_train,recall_train,rmse_test,accuracy_test,recall_test
        ]
        df_row = 1

        for threshold in deciles:
            rmse_train,accuracy_train,recall_train,rmse_test,accuracy_test,recall_test = self.evaluar_threshold(threshold, verbose=False)
            df_metricas_deciles.loc[df_row] = [
                threshold,rmse_train,accuracy_train,recall_train,rmse_test,accuracy_test,recall_test
            ]
            df_row += 1

        return df_metricas_deciles
    
    def obtener_df_deuda_acumulada(self):
        """
        Métodoque permite obtene run dataframe con la deuda acumulada a cada
        punto de corte de probabilidad de mora, así como la deuda superior a 
        dicho punto de corte

        Results:
        ----------
        pd.Dataframe
            Dataframe con la información
        """
        df_probabilidades = self.obtener_probabilidades_clientes()

        df_deuda_acumulada = df_probabilidades.sort_values('probabilidad')
        df_deuda_acumulada['deuda_acumulada'] = df_deuda_acumulada['deuda'].cumsum()
        df_deuda_acumulada['deuda_superior'] = np.max(df_deuda_acumulada['deuda_acumulada']) - df_deuda_acumulada['deuda_acumulada']
        df_deuda_acumulada.reset_index(inplace=True)
        df_deuda_acumulada.drop(columns=["CLIENT_ID", "deuda"], inplace=True)

        return df_deuda_acumulada

    def obtener_informe_modelo(self):
        """
        Método que entrena el modelo y obtiene un informe final de métricas
        para distintos umbrales de decisión en la clasificación.

        Results:
        ----------
        pd.Dataframe
            DataFrame con la información final de métricas para distintos 
            umbrales de decisión en la clasificación.
        """
        self.fit()

        df_metricas_deciles = self.obtener_metricas_deciles()
        df_probabilidades = self.obtener_probabilidades_clientes()
        umbrales = df_metricas_deciles['p'].to_numpy()

        df_VaRs = obtener_exposicion_riesgo(df_probabilidades, umbrales)

        df_informe = df_metricas_deciles.merge(df_VaRs,on='p')

        return df_informe
    