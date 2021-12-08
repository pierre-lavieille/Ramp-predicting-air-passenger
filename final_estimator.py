# Projet_final
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder   

def _encode_dates(X):
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    
    #ADD Holidays
    we = [4, 5, 6]
    ho_we = [51, 35, 21, 26, 46]
    ho = [1, 52, 27, 47]
    ho_0 = [36, 22]
    summer = range(22, 35)
    X.loc[:, 'holiday'] = np.where((X['week'].isin(ho)) |
                                    ((X['week'].isin(ho_we)) & (X['weekday'].isin(we))) |
                                    ((X['week'].isin(ho_0)) & (X['weekday'] == 0)) |
                                   ((X['week'].isin(summer)) & (X['weekday'] == 5))
                                   , 1, 0)
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture", "std_wtd"])   
  
  
def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    
    categorical_encoder = OneHotEncoder(sparse=False)
    categorical_cols = ["Arrival", "Departure"]
    
    numerical_cols = ['WeeksToDeparture']
    numerical_encoder = KBinsDiscretizer(n_bins=9,
                                           encode='ordinal',
                                         strategy='kmeans')
    
    preprocessor1 = make_column_transformer(
        (categorical_encoder, categorical_cols),
        (numerical_encoder, numerical_cols),
        remainder='passthrough'
        )

    
    regressor = XGBRegressor(colsample_bytree= 0.7, 
                  learning_rate = 0.03, 
                  max_depth = 10, 
                  min_child_weight = 3, n_estimators = 2000, 
                  early_stopping_rounds = 10,          
                  nthread = 4, objective = 'reg:squarederror', subsample= 0.7)
    
    return make_pipeline(date_encoder, preprocessor1, regressor)