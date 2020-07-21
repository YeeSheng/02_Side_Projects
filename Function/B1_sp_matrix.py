import numpy as np
import pandas as pd

def LookBack(dataframe,x_lags,y_lags,variable='WS95'):
    time = dataframe['DateTime'].copy()
    data = dataframe[['{}'.format(variable)]].copy()
    t = time[x_lags:].copy()
    df = pd.DataFrame()
    df['DateTime']= t
    # Make X
    for i in range(x_lags):
        x = data.shift(i+1)
        x.reset_index(inplace=True,drop=True)
        df['X{}'.format(i+1)]= x
    #Make Y
    y = data[x_lags:].copy()
    for i in range(y_lags):
        df['Y{}'.format(i+1)]= y.shift(-i)
    df.reset_index(inplace=True,drop=True)
    return df

def Lookback_WSWD(dataframe,x_lags,y_lags):
    time = dataframe['DateTime'].copy()
    dat  = dataframe[['WS95S','sinS','cosS']][x_lags:].copy()
    # time setting
    t    = time[x_lags:].copy()
    df   = pd.DataFrame()
    df['DateTime']= t
    # X
    for i in range(x_lags):
        x = dat.shift(i+1)
        x.reset_index(inplace=True,drop=True)
        Arr_X = np.array(x)
        df['X{}'.format(i+1)]= (np.ndarray.tolist(Arr_X))
    # Y
    y = dat.copy()
    y.reset_index(inplace=True,drop=True)
    for i in range(y_lags):
        Arr_Y = np.array(y.shift(-i))
        df['Y{}'.format(i+1)]=  (np.ndarray.tolist(Arr_Y))
    df.reset_index(inplace=True,drop=True)
    return df

def Lookback_WS_sc(dataframe,x_lags,y_lags):
    time = dataframe['DateTime'].copy()
    dat  = dataframe[['WSsinS','WScosS']][x_lags:].copy()
    # time setting
    t    = time[x_lags:].copy()
    df   = pd.DataFrame()
    df['DateTime']= t
    # X
    for i in range(x_lags):
        x = dat.shift(i+1)
        x.reset_index(inplace=True,drop=True)
        Arr_X = np.array(x)
        df['X{}'.format(i+1)]= (np.ndarray.tolist(Arr_X))
    # Y
    y = dat.copy()
    y.reset_index(inplace=True,drop=True)
    for i in range(y_lags):
        Arr_Y = np.array(y.shift(-i))
        df['Y{}'.format(i+1)]=  (np.ndarray.tolist(Arr_Y))
    df.reset_index(inplace=True,drop=True)
    return df

def shaping_one(data,LB,LF):
    #X_part
    X_shape = []
    for i in range(len(data)):
        p0 = []
        for j in LB:
            p0.append(data['X{}'.format(int(j))][i])
        p0 = np.array(p0).transpose()
        X_shape.append(p0)
    X_shape = np.array(X_shape)
    X_shape = np.reshape(X_shape,(X_shape.shape[0],1,X_shape.shape[1]))
    #Y_part
    Y_shape = []
    for i in range(len(data)):
        p0 = []
        for j in LF:
            p0.append(data['Y{}'.format(int(j))][i])
        p0 = np.array(p0).transpose()
        Y_shape.append(p0)
    Y_shape = np.array(Y_shape)
    Y_shape = np.reshape(Y_shape,(Y_shape.shape[0],1,Y_shape.shape[1]))
    print("X_shape = {}".format(X_shape.shape))
    print("Y_shape = {}".format(Y_shape.shape))
    return X_shape,Y_shape

def shaping_ndim(data,LB,LF):
    #X_part
    X_shape = []
    for i in range(len(data)):
        p0 = []
        for j in LB:
            p0.append(data['X{}'.format(int(j))][i])
        p0 = np.array(p0).transpose()
        X_shape.append(p0)
    X_shape = np.array(X_shape)
    X_shape = np.reshape(X_shape,(X_shape.shape[0],X_shape.shape[1],X_shape.shape[2]))
    #Y_part
    Y_shape = []
    for i in range(len(data)):
        p0 = []
        for j in LF:
            p0.append(data['Y{}'.format(int(j))][i])
        p0 = np.array(p0).transpose()
        Y_shape.append(p0)
    Y_shape = np.array(Y_shape)
    Y_shape = np.reshape(Y_shape,(Y_shape.shape[0],Y_shape.shape[1],Y_shape.shape[2]))
    print("X_shape = {}".format(X_shape.shape))
    print("Y_shape = {}".format(Y_shape.shape))
    return X_shape,Y_shape

def shaping_2dim(data,LB):
    #X_part
    X_shape = []
    for i in range(len(data)):
        p0 = []
        for j in LB:
            p0.append(data['A{}'.format(int(j))][i])
        p0 = np.array(p0).transpose()
        X_shape.append(p0)
    X_shape = np.array(X_shape)
    #Y_part
    Y_shape = data['Y'].copy()
    Y_shape = np.array(Y_shape)
    Y_shape = np.reshape(Y_shape,(Y_shape.shape[0],1))
    print("X_train_shape = {}".format(X_shape.shape))
    print("Y_train_shape = {}".format(Y_shape.shape))
    return X_shape,Y_shape
