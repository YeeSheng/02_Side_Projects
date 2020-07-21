import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.metrics import r2_score
#Validation

# Linear
def Mean_Square_Error(y_real,y_pred):
    delta = np.subtract(y_real,y_pred)
    MSE = np.square(delta).mean()
    return MSE
def Nor_Mean_Square_Error(y_real,y_pred):
    delta = np.subtract(y_real,y_pred)
    MSE = np.square(delta).mean()
    var = np.var(y_real)
    NMSE= MSE/var
    return NMSE
def Root_Mean_Square_Error(y_real,y_pred):
    delta = np.subtract(y_real,y_pred)
    MSE = np.square(delta).mean()
    RMSE = np.sqrt(MSE)
    return RMSE
def Nor_Root_Mean_Square_Error(y_real,y_pred):
    delta = np.subtract(y_real,y_pred)
    MSE = np.square(delta).mean()
    std = np.std(y_real)
    NRMSE= np.sqrt(MSE/std)
    return NRMSE
def Mean_Absolute_Error(y_real,y_pred):
    delta = np.subtract(y_real,y_pred)
    MAE = np.abs(delta).mean()
    return MAE
def Mean_Absolute_Percentage_Error(y_real,y_pred):
    delta = np.subtract(y_real,y_pred)    
    MAPE = (np.abs(delta)/y_real).mean()
    return MAPE
def Relative_Error(y_real,y_pred,circular=0):
    delta = np.subtract(y_pred,y_real)
    RE = delta/y_real
    return RE


# Circular
def MSE_Circular(y_real,y_pred):
    pi = np.pi
    delta = pi - np.abs(pi-np.abs(np.subtract(y_real,y_pred)))
    MSE = (np.square(delta)).mean()
    return MSE
def MAE_Circular(y_real,y_pred):
    pi = np.pi
    delta = pi - np.abs(pi-np.abs(np.subtract(y_real,y_pred)))
    MAE = np.abs(delta).mean()
    return MAE
def NMAE_Circular(y_real,y_pred):
    delta = np.subtract(y_pred,y_real)    
    NMAE = (np.abs(delta)/np.pi).mean()
    NMAE = NMAE*100
    return NMAE