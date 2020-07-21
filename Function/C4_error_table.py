import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from A3_select_data import DataRange
from C4_error import Nor_Mean_Square_Error
from C4_error import Root_Mean_Square_Error
from C4_error import Mean_Absolute_Percentage_Error
from C4_error import Mean_Absolute_Error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def Step_Error(df,Start_date,Stop_date,i):
    df.dropna(how='any',inplace=True)
    df.reset_index(inplace=True,drop=True)
    df['V1'].replace(0,0.001,inplace=True)
    data = DataRange(df,Start_date,Stop_date)
    WS_v = data['V1'].copy()
    WS_f = data['F{}'.format(i+1)]
    MAE  = Mean_Absolute_Error(WS_v,WS_f)
    RMSE = Root_Mean_Square_Error(WS_v,WS_f)
    NMSE = Nor_Mean_Square_Error(WS_v,WS_f)
    print("{},{},{},{}".format(i+1,MAE,RMSE,NMSE))
    return MAE,RMSE,NMSE

def Step_Error_ndim(df,Start_date,Stop_date,i,variable='WS'):
    df.fillna(method='ffill',inplace=True)
    df.reset_index(inplace=True,drop=True)
    df['V1_{}'.format(variable)].replace(0,0.001,inplace=True)
    data = DataRange(df,Start_date,Stop_date)
    WS_v = data['V1_{}'.format(variable)].copy()
    WS_f = data['F{}_{}'.format(i+1,variable)]
    MAE  = Mean_Absolute_Error(WS_v,WS_f)
    RMSE = Root_Mean_Square_Error(WS_v,WS_f)
    NMSE = Nor_Mean_Square_Error(WS_v,WS_f)
    print("{},{},{},{}".format(i+1,MAE,RMSE,NMSE))
    return MAE,RMSE,NMSE
