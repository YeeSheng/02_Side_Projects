import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from A3_select_data import DataRange
from B1_sp_matrix import shaping_one,shaping_ndim
from C4_error import (Mean_Absolute_Error, Nor_Mean_Square_Error,
                      Root_Mean_Square_Error)

# Recursive Method
def First_Step_Prediction(df,df01,model,LB,LF):
    #df = df_val01
    x_val,y_val = shaping_one(df,LB,LF)
    f_val = model.predict(x_val) 
    df_Valid = pd.DataFrame()
    df_Valid[['DateTime','WS95']] = df01[['DateTime','WS95']].copy()
    df_Valid['V1']  = y_val.flatten()
    df_Valid['F1']  = f_val.flatten()
    return df_Valid

def Multi_Step_Prediction(df,df_Valid,model,LB,LF,steps):
    df_new = df.copy()
    x_lags = len(LB)
    # Shift the x1-> x2, x2-> x3
    df_new.rename(columns={'X{}'.format(i+steps): 'X{}'.format(i+steps+1) for i in range(x_lags)},inplace=True)
    for i in range(steps):
        df_new['X{}'.format(steps)] = df_Valid['F{}'.format(steps-i)]
    df_new.dropna(how='any',inplace=True)
    df_new.reset_index(inplace=True,drop=True)
    x_val,y_val = shaping_one(df_new,LB,LF)
    f_val = model.predict(x_val)
    print(f_val.shape)
    df_Valid['F{}'.format(steps+1)] = f_val.flatten()
    df_Valid.replace(0,np.nan,inplace=True)
    return df_new,df_Valid

def First_Step_ndim(df,df01,model,LB,LF):
    x_val,y_val = shaping_ndim(df,LB,LF)
    f_val = model.predict(x_val) 
    df_Valid = pd.DataFrame()
    df_Valid[['DateTime','WS95']] = df01[['DateTime','WS95']].copy()
    df_Valid['V1']  = np.ndarray.tolist(y_val.reshape(y_val.shape[0],y_val.shape[1]))
    df_Valid['F1']  = np.ndarray.tolist(f_val.reshape(f_val.shape[0],f_val.shape[1]))
    return df_Valid

def Multi_Step_ndim(df,df_Valid,model,LB,LF,steps):
    df_new = df.copy()
    x_lags = len(LB)
    # Shift the x1-> x2, x2-> x3
    df_new.rename(columns={'X{}'.format(i+steps): 'X{}'.format(i+steps+1) for i in range(x_lags)},inplace=True)
    for i in range(steps):
        df_new['X{}'.format(steps)] = df_Valid['F{}'.format(steps-i)]
    df_new.dropna(how='any',inplace=True)
    df_new.reset_index(inplace=True,drop=True)
    x_val,y_val = shaping_ndim(df_new,LB,LF)
    f_val = model.predict(x_val)
    print(f_val.shape)
    df_Valid['F{}'.format(steps+1)] = np.ndarray.tolist(f_val.reshape(f_val.shape[0],f_val.shape[1]))
    return df_new,df_Valid


def MIMO_Step_Prediction(df,df01,model,LB,LF,step):
    x_val,y_val = shaping_one(df,LB,LF)
    f_val = model.predict(x_val) 
    df_Valid = pd.DataFrame()
    df_Valid['DateTime'] = df01['DateTime'].copy()
    df_Valid['WS95'] = df01['WS95'].copy()
    df_Valid['V1'] = y_val[:,:,0].flatten()
    for i in range(step):
        df_Valid['F{}'.format(i+1)] = f_val[:,:,i].flatten()
    return df_Valid

def MIMO_Step_Prediction_dS(df,df01,model,LB,LF):
    #df = df_val01/ Train_data
    x_val,y_val = shaping_one(df,LB,LF)
    f_val = model.predict(x_val) 
    df_Valid = pd.DataFrame()
    df_Valid['DateTime'] = df01['DateTime'].copy()
    df_Valid['WS95'] = df01['WS95'].copy()
    df_Valid['WS95S'] = df01['WS95S'].copy()
    df_Valid['V1'] = y_val[:,:,0].flatten() + df_Valid['WS95S'].shift(1)
    df_Valid['F1'] = f_val[:,:,0].flatten() + df_Valid['WS95S'].shift(1)
    df_Valid['F2'] = f_val[:,:,1].flatten() + df_Valid['F1']
    df_Valid['F3'] = f_val[:,:,2].flatten() + df_Valid['F2']
    df_Valid['F2'] = df_Valid['F2'].shift(1)
    df_Valid['F3'] = df_Valid['F3'].shift(2)
    return df_Valid

def Inverse_dS(df_Valid,step):
    df = df_Valid.copy()
    df['V1'] = df['V1'] + df['WS95'].shift(1)
    df['F1'] = df['F1'] + df['WS95'].shift(1)
    for i in range(step-1):
        df['F{}'.format(i+2)] = df['F{}'.format(i+2)] + df['F{}'.format(i+1)]
    return df

def Inverse_STD(df_Valid,std,miu,step):
    df = df_Valid.copy()
    for i in range(step):
        df['F{}'.format(i+1)] = (df_Valid['F{}'.format(i+1)]*std+miu)
    df['V1'] = (df_Valid['V1']*std+miu)
    return df

def Inverse_STD_ndim(df_Valid,scaler,step):
    df = df_Valid.copy()
    # validation part
    df['V1']     = df['V1'].apply(lambda x: scaler.inverse_transform(x))
    df['V1_WS']  = df['V1'].apply(lambda x: x[0])
    df['V1_sin'] = df['V1'].apply(lambda x: x[1])
    df['V1_cos'] = df['V1'].apply(lambda x: x[2])
    # predicion part
    for i in range(step):
        df['F{}'.format(i+1)] = df['F{}'.format(i+1)].apply(lambda x: scaler.inverse_transform(x))
        df['F{}_WS'.format(i+1)]  = df['F{}'.format(i+1)].apply(lambda x: x[0])
        df['F{}_sin'.format(i+1)] = df['F{}'.format(i+1)].apply(lambda x: x[1])
        df['F{}_cos'.format(i+1)] = df['F{}'.format(i+1)].apply(lambda x: x[2])
    return df

def Inverse_STD_sc(df_Valid,scaler,step):
    df = df_Valid.copy()
    # validation part
    df['V1']       = df['V1'].apply(lambda x: scaler.inverse_transform(x))
    df['V1_WSsin'] = df['V1'].apply(lambda x: x[0])
    df['V1_WScos'] = df['V1'].apply(lambda x: x[1])
    df['V1_WS'] = np.sqrt(df['V1_WSsin']**2 + df['V1_WScos']**2  )
    # predicion part
    for i in range(step):
        df['F{}'.format(i+1)] = df['F{}'.format(i+1)].apply(lambda x: scaler.inverse_transform(x))
        df['F{}_WSsin'.format(i+1)] = df['F{}'.format(i+1)].apply(lambda x: x[0])
        df['F{}_WScos'.format(i+1)] = df['F{}'.format(i+1)].apply(lambda x: x[1])
        df['F{}_WS'.format(i+1)] = np.sqrt(df['F{}_WSsin'.format(i+1)]**2 + df['F{}_WSsin'.format(i+1)]**2)
    return df


def Step_Error(df,Start_date,Stop_date,i):
    df.fillna(method='ffill',inplace=True)
    df.reset_index(inplace=True,drop=True)
    df['V1'].replace(0,0.001,inplace=True)
    data = DataRange(df,Start_date,Stop_date)
    WS_v = data['V1'].copy()
    WS_f = data['F{}'.format(i+1)]
    MAE  = Mean_Absolute_Error(WS_v,WS_f)
    RMSE = Root_Mean_Square_Error(WS_v,WS_f)
    NMSE= Nor_Mean_Square_Error(WS_v,WS_f)
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
    NMSE= Nor_Mean_Square_Error(WS_v,WS_f)
    print("{},{},{},{}".format(i+1,MAE,RMSE,NMSE))
    return MAE,RMSE,NMSE

def Plot_Step_Prediction(data,steps):
    date01 = '2019-02-07'
    date02 = '2019-02-08'
    df = DataRange(data,"{} 18:00".format(date01),"{} 03:01".format(date02))
    WS_v  = df['V1'].copy()
    time  = df['DateTime'].copy()
    u1 = time.iloc[0]
    u2 = time.iloc[-1]
    #naming = {1:'1st',2:'2nd',3:'3rd'}
    fig, (ax) = plt.subplots(steps,1,sharex=True,figsize=(12,9), gridspec_kw = {'wspace':0, 'hspace':0.05})
    for step in range(steps):
        WS_f = df['F{}'.format(step+1)].copy()
        ax[step].plot(time,WS_v, label='Real', marker='h',color='black',linewidth=2)
        ax[step].plot(time,WS_f,label='Predict',linewidth=1, linestyle=':', marker='D' ,color='blue')    
        ax[step].legend(loc=2,fontsize=12)
        ax[step].tick_params(axis="x", labelsize=10,rotation=0)
        ax[step].tick_params(axis="y", labelsize=15)
        ax[step].set_xlim([u1,u2])
        ax[step].set_ylim([0,26])
        ax[step].set_yticks([0,5,10,15,20,25])
        ax[step].grid(True)
    ax[0].set_title("Wind Speed Prediction ({})".format(date01),fontsize=20)
    ax[step-1].set_ylabel("Wind Speed (m/s)",fontsize=20)
    ax[step].set_xlabel("Time Series (Time Step=10min)",fontsize=20)
    ax[step].xaxis.set_major_locator(mdates.HourLocator())
    ax[step].xaxis.set_major_formatter(mdates.DateFormatter('%H00'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.80, 0.656, '1st-Step Prediction', ha='center',size=16,bbox=props)
    fig.text(0.80, 0.400, '2nd-Step Prediction', ha='center',size=16,bbox=props)
    fig.text(0.80, 0.145, '3rd-Step Prediction', ha='center',size=16,bbox=props)

def Shifting_Step(df_Valid,step):
    df = df_Valid.copy()
    for i in range(step):
        df['F{}'.format(i+1)] = df['F{}'.format(i+1)].shift(i)
    return df

def Shifting_Step_ndim(df_Valid,step):
    df = df_Valid.copy()
    for i in range(step):
        df['F{}_WS'.format(i+1)]  = df['F{}_WS'.format(i+1)].shift(i)
        df['F{}_sin'.format(i+1)] = df['F{}_sin'.format(i+1)].shift(i)
        df['F{}_cos'.format(i+1)] = df['F{}_cos'.format(i+1)].shift(i)
    return df

def Shifting_Step_sc(df_Valid,step):
    df = df_Valid.copy()
    for i in range(step):
        df['F{}_WSsin'.format(i+1)] = df['F{}_WSsin'.format(i+1)].shift(i)
        df['F{}_WScos'.format(i+1)] = df['F{}_WSsin'.format(i+1)].shift(i)
        df['F{}_WS'.format(i+1)]    = df['F{}_WS'.format(i+1)].shift(i)
    return df

def Shrink(data,scale):
    dataA = data.copy()
    scale = scale
    ShrinkData = []
    for i in range(len(dataA)):
        replace = int(np.floor((dataA['DateTime'][i].minute)/scale))*scale
        ShrinkData.append(dataA['DateTime'][i].replace(minute=replace))
        if i%10000 ==0:
            print("Progress {}".format(i))
    print("Shrink Time")
    dataA['DateTime'] = ShrinkData
    dataA = dataA.groupby('DateTime').mean()
    dataA.reset_index(inplace=True,drop=False)
    return dataA

def five_to_ten(data):
    df = data.copy()
    # combine to 10-min average
    df['F1A'] = (df['F1']+df['F2'])/2
    df['F2A'] = (df['F3']+df['F4'])/2
    df['F3A'] = (df['F5']+df['F6'])/2
    df01 = df[['DateTime','WS95','V1','F1A','F2A','F3A']].copy()
    df01.columns = ['DateTime','WS95','V1','F1','F2','F3']
    df02 = Shifting_Step(df01,3)
    df02.reset_index(inplace=True)
    # Get F1, F2, F3
    logic = df02['index'].apply(lambda x: x%2==0)
    df03  = df02[logic].copy()
    df03.reset_index(inplace=True,drop=True)
    # Get V1
    df0B =Shrink(df02,10)
    
    # Combine DataFrame
    dfA = pd.DataFrame()
    dfA = df0B[['DateTime','V1']].copy()
    dfA[['F1','F2','F3']] = df03[['F1','F2','F3']].copy()
    return dfA