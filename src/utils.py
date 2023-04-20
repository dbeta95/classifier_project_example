
import numpy as np

from typing import Any, List
from sklearn.model_selection import ParameterGrid

class NNGridSearchCV():
    """
    Class that defines a method to search a parameters grid with cross-validation
    """

    def __init__(self, 
        model_class:any,
        input_shape:tuple,
        loss:str, 
        metrics:List[str], 
        target:str, 
        param_distribution:dict, 
        cv:Any, 
        epochs:int=100,
        verbose:int=1
    ):
        """
        Class instantiation method.

        Parameters:
        ----------
            model_class:any
                Class containing the neural network base model
            input_shape:tuple
                Tuple defining the input's shape
            loss:str
                Loss function
            metrics:List[str]
                List of metrics to get during the training process 
            target:str
                Metric to use to determine the best parameters
            param_distribution:dict
                Dictionary containing the lists of values for the parameters
            cv:Any
                Iterable element of indexes for the cross validation
            epochs:int=100
                Number of epochs to fit
            verbose:int=1
                Either information about the training process should be printed
        """

        self.model_class = model_class
        self.input_shape = input_shape
        self.param_distribution = param_distribution
        self.loss = loss
        assert target in metrics,"The target metric must be one of the metrics to be evaluated"
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = metrics
        self.target = target
        self.epochs = epochs
        self.cv = [(train_index, val_index) for train_index, val_index in cv]
        self.verbose = verbose

    def initialize_model(self, **kwargs):
        """
        Method that initializes the model given the parameters
        """
        
        kwargs_model = {
            key:value for key,value in kwargs.items() if key in self.model_class.__init__.__code__.co_varnames
        }

        self.kwargs_fit = {
            key:value for key,value in kwargs.items() if key not in self.model_class.__init__.__code__.co_varnames
        }

        self.model = self.model_class(input_shape = self.input_shape, **kwargs_model)

        optimizer=kwargs.get("optimizer")
        if not optimizer:
            optimizer = 'adam'

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        

    def fit(self,X:np.ndarray,y:np.ndarray):
        """
        Method to fit the model for all posible combinations of the parameters and
        for all fold in the cross validation process. The best parameters are saved as 
        an attribute of the main object, as are all the best metrics and the history
        for the result in each iteration.

        Parameters:
        ----------
            X:np.ndarray
                Array of independent variables in the training set
            y: np.ndarray
                Array with the model's response in the training set
        """

        self.history = {
            'parameters':[],
            'history':[],
            'loss':[],
            'epochs':[]
        }
        metrics_dict = {metric:[] for metric in self.metrics}
        self.history.update(metrics_dict)
        
        for parameters in ParameterGrid(param_grid=self.param_distribution):
            
            if self.verbose:
                print(f"Training with parameters:\n{parameters}")
            
            self.history['parameters'].append(parameters)
            params_history = {
                'loss':[]
            }
            params_history.update({metric:[] for metric in self.metrics})

            fold = 0

            for train_index, val_index in self.cv:

                fold += 1
                if self.verbose:
                    print(f"Fold: {fold}")
                
                X_train = X[train_index]
                y_train = y[train_index]

                X_val = X[val_index]
                y_val = y[val_index]
                self.initialize_model(**parameters)
                
                fit_results = self.model.fit(
                    X_train,
                    y_train,
                    validation_data = (X_val, y_val),
                    epochs = self.epochs,
                    **self.kwargs_fit,
                    verbose = 0
                )

                params_history['loss'].append(fit_results.history.get('val_loss'))
                for metric in self.metrics:
                    metric_name = 'val_'+metric
                    params_history[metric].append(fit_results.history.get(metric_name))     
            
            history_mean = {
                key : np.mean(values, axis=0) for key, values in params_history.items()
            }
            
            self.history['history'].append(history_mean)

            index_best = np.argmax(history_mean.get(self.target))

            for key,values in history_mean.items():
                self.history[key].append(values[index_best])

            self.history['epochs'].append(index_best+1)

        best_params_index = np.argmax(self.history.get(self.target))
        self.best_parameters = self.history.get('parameters')[best_params_index]

        for key, values in self.history.items():
            if key in self.metrics:
                exec("self.best_" + key + "= values[best_params_index]")

        self.best_epochs = best_params_index + 1


## NEXT STEP: https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
## https://paperswithcode.com/method/resnet