# Proyecto de datos: 

El presente proeycto busca la implementación de todas las étapas de un proyecto de datos a la vez que se resuelve el problema técnico planteado. Los archivos dentro del poryecto obedecen a la lógica de pertenecer a dichas étapas.

## 1. Entendimiento de datos

En primera instancia se procede a leer, entender la estructura original de los datos y la información que estos brindan y se consolida informe de estos. Los archivos relacionados con esta étapa son:

* exploracion_datosipynb: Notebook del analisis exploratorio en insghts.
* reporte_datos.html: Se encuentra en la carpeta de documentos y se trata de un reporte de la información disponible consolidada

## 2. Procesamiento de datos

En este étapa se procesa el conjunto de datos según las manipulaciones requeridas por su naturaleza para prepararlo para la étapa del modelamiento. El archivo enq ue se realiza es:

* preprocesamiento.ipynb

## 3. Modelamiento

En este étapa se ajustan los modelos de clasificación, buscando los parámetros óptimos. Corresponde a los archivos

* entrenamiento_ml.ipynb
* entrenamiento_ml_balanced.ipynb
* parameters: la carpeta contenida en src donde se almacenan todos los parámetros óptimos de los diferentes modelos

## 4. Evaluación

Finalmente se evalúan métricas de los mejores modelos, las cuales se llevan también a un tablero de visualización para evaluación bajo una lógica de despliegue continuo. Los archivos son:

* evaluacion.ipynb

## 5. Despliegue - Entendimiento de negocio

Finalmente se preparan archivos resultantes de todos los procesos anteriores para ejecutar cada tarea con facilidad y permiter el despliegue y evaluación continuos. Los archivos son:

* data_management.py
* model_management.py
* generar_informes.py
* clasificar.py
* carpeta de resultados

