import os
print(os.getcwd())

import sys
sys.path.append('/home/saraiva/Documentos/GitHub/data-science-analytics/utils')
from my_tools import custom_inspect_class
from my_tools import unpack_list

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import tqdm
import time
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

#Regression Algorithms
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor as XGBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor

#Classifier Algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

#Clustering Algorithms
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering

#Regression Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

#Classification Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import auc

#Clustering Metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

from pprint import pprint

method_dict_regression ={
         
           'algo': 
                   {
                    AdaBoostRegressor.__name__ : AdaBoostRegressor,
                    BaggingRegressor.__name__ : BaggingRegressor,
                    GradientBoostingRegressor.__name__ : GradientBoostingRegressor,
                    RandomForestRegressor.__name__ : RandomForestRegressor ,
                    XGBoostRegressor.__name__ : XGBoostRegressor,
                   },                           
          
          'eval_metric' : 
                                 [mean_absolute_error,
                                  mean_squared_error,                                  
                                  mean_squared_log_error,],
                        
          'space' : 
                      {
                       AdaBoostRegressor.__name__ : {},
                       BaggingRegressor.__name__ : {},
                       GradientBoostingRegressor.__name__ : {}, 
                       RandomForestRegressor.__name__ :
                                                    { 
                                                      'bootstrap' :[True,False], 
                                                      'max_depth' : list(range(2, 10)),
                                                      'max_features' : ['auto','sqrt','log2'],
                                                      'min_samples_leaf': list(range(5,20)), 
                                                      'n_estimators': list(range(30, 300 ,5))
                                                    },
                       
                      XGBoostRegressor.__name__ : {},                                            
                      }           
                    
        }

class AutomatedRegression():
    
    def __init__(self,method_dict):
        self.method_dict = method_dict
        self.metric_list = method_dict['eval_metric']
        
    def pretty_print(self):
        '''Method to pprint some information
        First - Information about the method_dict whict dictionary possess'''
        pprint(self.method_dict)
        
    def fit_predict_tune(self,X,y,
                         choosen_algo,
                         number_splits = 5,
                         iterations = 20):
        ''' 
        This method is used to get the best hyperparameters for a given algorithm(choosen_algo) for regression 
        using Random Search.
        The space of search is given by the dictionary used for inicialization and the result dataframe contains 
        columns about the parameters used in the round and the mean obtained for the evaluation metrics.
        
        - Input :
        X : DataFrame - Exogenous Variables
        y : DataFrame - Endogenous Variables
        choosen_algo : Str - Algorithm's name, it's necessary to have the algorithm class specified in method_dict
        number_splits : Int - Number of splits in KFold
        iterations : Number of iterations of Random Search
        '''
        
        self.choosen_algo = choosen_algo
        param_grid = self.method_dict['space'][choosen_algo]
        
        param_col = [key for key in param_grid.keys()] + [metrics.__name__ for metrics in self.metric_list]
        
        lots_vals = list()
        print(choosen_algo)
        for rounds in tqdm.tqdm_notebook(range(iterations)):
            #Aleatoriza os parametros
            params = { key : random.sample(value,1)[0] for key,value in param_grid.items()}

            kf = KFold(n_splits = number_splits,shuffle = True)
            matrix_metrics = []
            counter = 1
            
            for train_idx,test_idx in kf.split(X):

                X_train,X_test = X.loc[train_idx,:],X.loc[test_idx,:]
                y_train,y_test = y.loc[train_idx,:],y.loc[test_idx,:]

                new_algo = method_dict_regression['algo'][choosen_algo](**params)

                new_algo.fit(X_train,np.ravel(y_train))
                y_pred = new_algo.predict(X_test)
                
                evaluation_list = []
                for evaluations in self.metric_list:
                    result = evaluations(y_test,y_pred)
                    evaluation_list.append(result)
                matrix_metrics.append(evaluation_list)
                
            #Dataframe criado com os resultados das metricas em cada fold
            fold_dataframe = pd.DataFrame(matrix_metrics,
                                          columns = [metrics.__name__ for metrics in self.metric_list])

            # Return list of results
            metrics_results = fold_dataframe.mean().tolist()
            vals = list(params.values()) + metrics_results
            lots_vals.append(vals)
            
            self.result_dataframe = pd.DataFrame(lots_vals,columns = param_col)
            self.result_dataframe['choosen_algo'] = choosen_algo
        
        #if(report == True):
        
        return self.result_dataframe
    
    def reports(self,main_metric):
        '''
        To run this method,it's necessary to pre-run fit_predict_tune.
        It displays the 5 Top Iterations in RandomSearch, display the describe about all runs and plot a
        histplot priorizing the main metric choosen by the user'''
        display(self.result_dataframe.sort_values(by = main_metric).head(5))
        self.desc_results = self.result_dataframe.describe()
        display(self.result_dataframe.describe())
        
        print("The mean of the main metric is : {}".format(self.desc_results.loc['mean'][main_metric]))
        print("The median of the main metric is : {}".format(self.desc_results.loc['50%'][main_metric]))
        print("The standard deviation of the main metric is : {}".format(self.desc_results.loc['std'][main_metric]))
        print('Mean + 3*sigma : {}'.
              format(self.desc_results.loc['mean'][main_metric] + 3*self.desc_results.loc['std'][main_metric]))
        print('Mean - 3*sigma : {}'.
              format(self.desc_results.loc['mean'][main_metric] - 3*self.desc_results.loc['std'][main_metric]))
        print("The amplitude of the main metric is : {}".
              format(self.desc_results.loc['max'][main_metric] - self.desc_results.loc['min'][main_metric]))

        plt.figure(figsize=(10,7))
        
        kwargs = dict(histtype='stepfilled', alpha=0.8, ec="k")

        plt.hist(self.result_dataframe[str(main_metric)],
                 align='mid',
                 density = True,
                 stacked = True,
                 **kwargs)
        
        plt.xlabel(main_metric)
        plt.ylabel('Probability')
        plt.title('Histogram - {}'.format(self.choosen_algo))
        plt.grid(True)
        plt.show()
    
    def info_full_reports(self, X, y, iter_report, number_splits):
        
        self.X_EXOGENOUS = X
        self.Y_ENDOGENOUS = y
        self.ITER_REPORT = iter_report
        self.NUM_SPLITS = number_splits
        
    def auto_full_reports(self,main_metric):
        
        for alg in method_dict_regression['algo'].keys():
            temp_ar = AutomatedRegression(method_dict=method_dict_regression)
            
            temp_ar.fit_predict_tune(X = pd.DataFrame(self.X_EXOGENOUS),
                                     y = pd.DataFrame(self.Y_ENDOGENOUS),
                                     choosen_algo = alg,
                                     number_splits = self.NUM_SPLITS,
                                     iterations = self.ITER_REPORT)
            temp_ar.reports(main_metric)