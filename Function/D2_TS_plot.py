import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def PLOT_WholeYear(DATA,year):
    logicY= (DATA["DateTime"].apply(lambda x: x.year))==(year)
    df    = DATA[logicY].copy()
    df.reset_index(inplace=True,drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.4)

    ax1.plot(df['DateTime'], df['WS95'],c='black')
    ax2.scatter(df['DateTime'],df['WD95'],s=1,c='grey')
    ax2.scatter(df['DateTime'],df['WD95r'],s=1,c='grey')
    ax2.scatter(df['DateTime'],df['WD95n'],s=1,c='black')
    
    ax1.set_title('Wind Speed (m/s)',fontweight='bold',size=25)
    ax1.tick_params(axis="x", labelsize=20,rotation=10)
    ax1.tick_params(axis="y", labelsize=20)
    
    u1 = df['DateTime'].iloc[0]
    u2 = df['DateTime'].iloc[-1]
    ax1.set_xlim([u1,u2])
    ax1.set_ylim([0,30])
    ax1.set_yticks([0,5,10,15,20,25,30])

    ax2.set_title('Wind Direction',fontweight='bold',size=25)
    ax2.tick_params(axis="x", labelsize=20,rotation=10)
    ax2.tick_params(axis="y", labelsize=20)

    ax2.set_xlim([u1,u2])
    ax2.set_ylim([-360,360])
    ax2.set_yticks([-360,-270,-180,-90,0,90,180,270,360])
    
    plt.show()

def WindRosePLOT(DATA,month,year):
    #Initiate
    plt.ion()
    DATA = DATA[['DateTime','WS95','WD95']].copy()
    DATA.dropna(how='any',inplace=True)
    mon = ['Jan' ,'Feb','Mar','Apr',
           'May' ,'Jun','Jul','Aug',
           'Sep' ,'Oct','Nov','Dec']
    logicY= (DATA["DateTime"].apply(lambda x: x.year))==(year)
    data    = DATA[logicY].copy()
    data.reset_index(inplace=True,drop=True)
    fig = plt.figure(figsize=(20,32),facecolor='w', edgecolor='r')
    #Plotting 12 graph
    for i in range(month):
        ax = plt.subplot2grid((4,3),(int(np.floor(i/3)),int(i%3)),projection='windrose')
        logic = (data["DateTime"].apply(lambda x: x.month))==(i+1)
        wd = data['WD95'][logic]
        ws = data['WS95'][logic]
        if (np.sum(logic)!=0):
            bins = np.array([0,5,15,25])
            ax.contourf(wd, ws,nsector=16,bins=bins,
                        normed=True,cmap=cm.get_cmap('seismic', 20), lw=1) 
        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=22)
        ax.set_title('{}'.format(mon[i]),fontweight='bold',size=30, pad=25)
    plt.subplots_adjust(hspace=0.1,top=0.83)
    x0, y0 =-0.70 , -0.2
    plt.legend(loc='upper center', bbox_to_anchor=(x0, y0),
          ncol=7, fancybox=True, shadow=True, fontsize='20')
    plt.show()

#Autocorrelation
def plot_acf_pacf(signal, lags, alpha,scale):
    fig, axes = plt.subplots(1,2,figsize=(18,5))
    sm.graphics.tsa.plot_acf(signal, lags= lags ,fft=True, alpha=alpha, ax=axes[0])
    sm.graphics.tsa.plot_pacf(signal, lags=lags, alpha=alpha, ax=axes[1])
    axes[0].set_title('Autocorrelation',fontweight='bold',size=20)
    axes[0].set_xlabel('Lags',size=20)
    axes[0].tick_params(axis="x", labelsize=20)
    axes[0].tick_params(axis="y", labelsize=20)

    axes[1].set_title('Partial Autocorrelation',fontweight='bold',size=20)
    axes[1].set_xlabel('Lags',size=20)
    axes[1].tick_params(axis="x", labelsize=20)
    axes[1].tick_params(axis="y", labelsize=20)
    plt.suptitle(("ACF & PACF delay={} mins".format(scale)),fontsize=22, fontweight='bold')

    plt.show()
# Histogram



#ML_code--------------------------------------------------------------------------------------------------------------------
#plot
def plot_ts_RMSE(df,WS_v,WS_f):
    fig, (ax) = plt.subplots(2, 1,sharex=True, figsize=(20, 10), gridspec_kw = {'wspace':0, 'hspace':0})
    ax[0].plot(df['DateTime'],WS_v,label="Real")
    ax[0].plot(df['DateTime'],WS_f,label="Prediction")
    u1 = df['DateTime'].iloc[0]
    u2 = df['DateTime'].iloc[-1]
    ax[0].legend()
    ax[0].tick_params(axis="x", labelsize=15,rotation=0)
    ax[0].tick_params(axis="y", labelsize=15)
    ax[0].set_xlim([u1,u2])
    ax[0].set_ylabel("Wind Speed (m/s)",fontsize=22)
    ax[0].set_title("Wind Speed One-Step Prediction", fontsize=25)
    ax[0].set_ylim([0,25])
    RMSE = np.sqrt(np.square(np.subtract(WS_v,WS_f)))
    ax[1].plot(df['DateTime'],RMSE)
    ax[1].tick_params(axis="x", labelsize=15,rotation=0)
    ax[1].tick_params(axis="y", labelsize=15)
    ax[1].set_xlim([u1,u2])
    ax[1].set_ylim([0,10])
    ax[1].set_xlabel("Time Series",fontsize=22)
    ax[1].set_ylabel("RMSE (m/s)",fontsize=22)
    plt.show()



def plot_MAPE_bar(data,year=2019):
    # plot data['Error']
    data['Year']   = data['DateTime'].apply(lambda x: x.year)
    data['Month']  = data['DateTime'].apply(lambda x: x.month)
    m_name = ['J','F','M','A','M','J','J','A','S','O','N','D']
    fig, ax = plt.subplots(1,1,figsize=(16, 10),)
    box_plot = data[data['Year'] ==2019]
    sns.boxplot(x="Month", y="Error",data=box_plot, width=0.8,linewidth=1, ax=ax)
    ax.tick_params(axis="x", labelsize=20)
    xk = data['Month'].unique() -1
    ax.set_xticks(xk)
    ax.set_xticklabels(m_name)
    ax.set_xlabel("Year {}".format(year), fontsize=20)
    ax.set_ylabel("MAPE (%) ", fontsize=24)
    ax.tick_params(axis="y", labelsize=20)
    #ax.legend(title="Northwind",title_fontsize=20,fontsize=20)
    fig.text(0.5, 0.02, 'Time', ha='center',size=24)
    plt.show()


#Validation
def Plot02_V2(data,steps):
    date01 = '2019-02-07'
    date02 = '2019-02-08'
    df = DataRange(data,"{} 16:00".format(date01),"{} 03:01".format(date02))
    WS_v  = df['V1'].copy()
    time  = df['DateTime'].copy()
    u1 = time.iloc[0]
    u2 = time.iloc[-1]
    naming = {1:'1st',2:'2nd',3:'3rd'}
    fig, (ax) = plt.subplots(steps,1,sharex=True,figsize=(12,9), gridspec_kw = {'wspace':0, 'hspace':0.05})
    for step in range(steps):
        WS_fF = df['F{}'.format(step+1)].copy()
        ax[step].plot(time,WS_v, label='Real', marker='h',color='black',linewidth=2)
        ax[step].plot(time,WS_fF,label='Filtered_Predict',linewidth=1, linestyle=':', marker='D' ,color='blue')    
        ax[step].legend(loc=2,fontsize=12)
        #ax.set_title("{}-step Prediction ({})".format(naming[step],date01),fontsize=20)
        ax[step].tick_params(axis="x", labelsize=10,rotation=0)
        ax[step].tick_params(axis="y", labelsize=15)
        ax[step].set_xlim([u1,u2])
        ax[step].set_ylim([0,21])
        ax[step].set_yticks([0,5,10,15,20])
        ax[step].grid(True)
    ax[0].set_title("Wind Speed Prediction ({})".format(date01),fontsize=20)
    ax[step-1].set_ylabel("Wind Speed (m/s)",fontsize=20)
    ax[step].set_xlabel("Time Series (Sampling Time=10min)",fontsize=20)
    ax[step].xaxis.set_major_locator(mdates.HourLocator())
    ax[step].xaxis.set_major_formatter(mdates.DateFormatter('%H00'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.80, 0.656, '1st-Step Prediction', ha='center',size=16,bbox=props)
    fig.text(0.80, 0.40, '2nd-Step Prediction', ha='center',size=16,bbox=props)
    fig.text(0.80, 0.145, '3rd-Step Prediction', ha='center',size=16,bbox=props)