import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pywt import wavedec, waverec
import pywt
from A3_select_data import DataRange

def Wavelet_Dec2(Train_data,x_lags,y_lags,wav='db1'):
    Train_data.reset_index(inplace=True,drop=True)
    print(x_lags,y_lags)
    df       = Train_data[['X{}'.format(x_lags-1-i) for i in range(x_lags-1)]]
    print(df.columns)
    A1 = []
    for i in range(len(df)):
        dfx = np.array(df.iloc[i])
        coeffs = pywt.dwt(dfx,wav,mode='smooth')
        A1.append(pywt.idwt(coeffs[0],None,wav,'smooth'))
    A1 = np.array(A1)
    dfA = pd.DataFrame()
    for i in range(x_lags):
        dfA['X{}'.format(i+1)] = A1[:,x_lags-1-i]
    for i in range(y_lags):
        dfA['Y{}'.format(i+1)] = Train_data['Y{}'.format(i+1)]
    dfA['DateTime'] = Train_data['DateTime'].copy()
    return dfA

def Wavelet_Dec(Train_data,x_lags,y_lags,wav='db1'):
    Train_data.reset_index(inplace=True,drop=True)
    print(x_lags,y_lags)
    df       = Train_data[['X{}'.format(x_lags-i) for i in range(x_lags)]]
    print(df.columns)
    A1 = []
    for i in range(len(df)):
        dfx = np.array(df.iloc[i])
        coeffs = pywt.dwt(dfx,wav,mode='smooth')
        A1.append(pywt.idwt(coeffs[0],None,wav,'smooth'))
    A1 = np.array(A1)
    dfA = pd.DataFrame()
    for i in range(x_lags):
        dfA['X{}'.format(i+1)] = A1[:,x_lags-1-i]
    for i in range(y_lags):
        dfA['Y{}'.format(i+1)] = Train_data['Y{}'.format(i+1)]
    dfA['DateTime'] = Train_data['DateTime'].copy()
    return dfA
