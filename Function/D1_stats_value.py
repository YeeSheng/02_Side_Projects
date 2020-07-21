from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus as pydot
import seaborn as sns
import statsmodels.api as sm
from sklearn import mixture

from A2_add_variable import Convert_Angle
from A3_select_data import DataRange2
from reliability.Distributions import (Gamma_Distribution,
                                       Lognormal_Distribution,
                                       Weibull_Distribution)
from reliability.Fitters import (Fit_Gamma_2P, Fit_Gamma_3P, Fit_Lognormal_2P,
                                 Fit_Weibull_2P, Fit_Weibull_3P,
                                 Fit_Weibull_Mixture)
from X9_read_listed import m_day, mon, mon_name

d_name = ['N','N-E-N','N-E','N-E-E',
          'E','S-E-E','S-E','S-E-S',
          'S','S-W-S','S-W','S-W-W',
          'W','N-W-W','N-W','N-W-N']
#----------------------------------------------------------------------------------------------------------------#
#Wind Speed
def STAT_WS(data):
    x = data['WS95'].copy()
    table = []
    table.append([x.count(),x.mean(),x.std(),
                  x.min(),x.quantile(0.25),
                  x.quantile(0.5),x.quantile(0.75),
                  x.max(),x.skew(),x.kurt()])
    df_table = pd.DataFrame(table)
    df_table.columns = ['count','mean','std','min','Q1','Q2','Q3','max','skew','kurt']
    return df_table
#----------------------------------------------------------------------------------------------------------------#
#Wind Direction
def STAT_WD(data):
    x = data['WD95'].copy()
    theta_p = x*np.pi/180
    S  = (np.sin(theta_p)).sum()
    C  = (np.cos(theta_p)).sum()
    S2 = (np.sin(2*theta_p)).sum()
    C2 = (np.cos(2*theta_p)).sum()

    R  = np.sqrt((S**2)+(C**2))
    count  = len(theta_p)
    R_bar = R/count
    std  = (-2*np.log(R_bar))**0.5
    
    nu1  = np.arctan2(S,C)
    nu2  = np.arctan2(S2,C2)
    rho2 = (np.cos(2*(nu1-theta_p))).mean()
    
    dispersion = 0.5*(1-rho2)/(R_bar**2)
    skewness   = rho2*np.sin(nu2-(2*nu1))/((1-R_bar)**1.5)
    kurtosis   = (rho2*np.cos(nu2-(2*nu1))-R_bar**4)/((1-R_bar)**2)
    mean       = nu1*180/np.pi
    table      = []
    table.append([count,mean,R_bar,std,dispersion,skewness,kurtosis])
    df_table = pd.DataFrame(table)
    df_table.columns = ['count','mean','R_bar','std','dispersion','skewness','kurtosis']
    return df_table
#----------------------------------------------------------------------------------------------------------------#
#Year
def Stats_Year(data,year,variable ='WS95'):    
    data01 = data[['DateTime','{}'.format(variable)]].copy()
    df = data01.dropna(how='any')
    logicY = (df['DateTime'].apply(lambda x:x.year) ==year)
    dataY = df[logicY].copy()
    dataY.reset_index(inplace=True,drop=True)
    print("Select year={}, {}".format(year,variable))
    if variable == 'WS95':
        table = STAT_WS(dataY)
    elif variable== 'WD95':
        table = STAT_WD(dataY)
    table['Year'] = year
    return table

#Month
def Stats_Month(data,year,month,variable='WS95'):
    data01 = data[['DateTime','{}'.format(variable)]].copy()
    df = data01.dropna(how='any')
    logicY = (df['DateTime'].apply(lambda x:x.year) ==year)
    dataY = df[logicY].copy()
    dataY.reset_index(inplace=True,drop=True)
    logicM = (dataY['DateTime'].apply(lambda x:x.month) ==month)
    dataM = dataY[logicM].copy()
    dataM.reset_index(inplace=True,drop=True)
    print("Select {}-{}".format(year,month))
    if variable == 'WS95':
        table = STAT_WS(dataM)
    elif variable== 'WD95':
        table = STAT_WD(dataM)
    table['Year'] = year
    table['Month']=month
    return table
# Monthly
def Stats_12month(data,year,variable='WS95'):
    df_mon = pd.DataFrame()
    for i in range(12):
        table = Stats_Month(data,year,i+1,variable)
        df_mon= pd.concat([df_mon,table],axis=0)
    return df_mon

# Season
def Stats_Season(data,year,season,variable='WS95'):
    data01 = data[['DateTime','{}'.format(variable)]].copy()
    #data01['Year']   = data['DateTime'].apply(lambda x: x.year)
    data01['Month']  = data['DateTime'].apply(lambda x: x.month)
    seasons = [(month%12 + 3)//3 for month in range(1, 13)]
    month_to_season = dict(zip(range(1,13), seasons))
    data01['Season']= data01['Month'].apply(lambda x:month_to_season[x])
    df = data01.dropna(how='any')
    logicY = (df['DateTime'].apply(lambda x:x.year) ==year)
    dataY = df[logicY].copy()
    dataY.reset_index(inplace=True,drop=True)
    # Season
    logicS = (dataY['Season'] == season)
    dataS = dataY[logicS].copy()
    dataS.reset_index(inplace=True,drop=True)
    print("Select {}-{}".format(year,season))
    if variable == 'WS95':
        table = STAT_WS(dataS)
    elif variable== 'WD95':
        table = STAT_WD(dataS)
    table['Year'] = year
    table['Season']=season
    Season_dict = {1:'Winter',2:'Spring',3:'Summer',4:'Autumn'}
    table['Seasons']=table['Season'].apply(lambda x:Season_dict[x])
    return table


# Seasonly
def Stats_4seasons(data,year,variable='WS95'):
    df_sea = pd.DataFrame()
    for i in range(4):
        table = Stats_Season(data,year,i+1,variable)
        df_sea= pd.concat([df_sea,table],axis=0)
    return df_sea
#----------------------------------------------------------------------------------------------------------------#
# bubble_plot
def Stat_Direction_Probability(data,year):
    data = Convert_Angle(data,95,-11.25,348.75,'h')
    d_name = ['N','N-E-N','N-E','N-E-E',
              'E','S-E-E','S-E','S-E-S',
              'S','S-W-S','S-W','S-W-W',
              'W','N-W-W','N-W','N-W-N']
    #Year
    logicY = (data["DateTime"].apply(lambda x: x.year))==year
    df_Y = data[logicY].copy()
    
    wd_d = pd.DataFrame()
    wd_p = pd.DataFrame()
    wd_d['Direction'] = d_name
    for i in range(12):
        #month
        
        logicM = (df_Y["DateTime"].apply(lambda x: x.month))==i+1
        df_M = df_Y[logicM].copy()
        wd = df_M['WD95h']
        a,b = np.histogram(wd,bins = np.arange(-11.25,348.76,22.5))
        wd_d['{}'.format(mon_name[i])] = a

        N = wd.count()
        print(N)
        wd_p['{}'.format(mon_name[i])]  = wd_d['{}'.format(mon_name[i])] /N*100

    mp_tab = []
    for i in range(12):
        a_new = wd_d.sort_values(by=['{}'.format(mon_name[i])],ascending=False)
        a_new.reset_index(inplace=True,drop=True)
        Mp1 = a_new['Direction'].iloc[0]
        Mp2 = a_new['Direction'].iloc[1]
        Mp3 = a_new['Direction'].iloc[2]
        mp_tab.append([Mp1,Mp2,Mp3])
    mp_tab = np.array(mp_tab)
    return wd_d,wd_p,mp_tab

def GetVariable(data):
    V_mean = data['WS95'].mean()
    S = data['sin95'].sum()
    C = data['cos95'].sum()
    R = np.sqrt(S**2+C**2)
    R_bar = R/len(data)
    D_mean = np.arctan2(S,C)
    wd = data['WD95h'].copy()
    a,b = np.histogram(wd,bins = np.arange(-11.25,348.76,22.5))
    a_norm = np.round(a/a.sum()*100,2)
    P_North = a_norm[0]+a_norm[1]+a_norm[-1]
    P_South = a_norm[8]+a_norm[9]+a_norm[10]
    dataS = ([V_mean,D_mean,R_bar,P_North,P_South])
    return dataS

def Stat_Variable(data,moving,duration):
    date0 = "2017-01-01 00:00"
    start_date = datetime.strptime(date0, '%Y-%m-%d %H:%M')
    S_var =[]
    S_date=[]
    N =int(np.floor(len(data)/1440))
    steps = int(np.floor((N-duration)/moving) +1)
    for i in range(steps):
        #print(start_date)
        dataA = DataRange2(data,start_date,duration)
        dataAS = dataA[['DateTime','WS95','WD95h','sin95','cos95']].copy()
        Stat_var =  GetVariable(dataAS)
        S_var.append(Stat_var)
        S_date.append(start_date)
        start_date  = start_date+timedelta(days=moving)
    Stat = np.array(S_var)
    Stat =pd.DataFrame(S_var)
    Stat.columns = ['V_mean','D_mean','R_bar','P_North','P_South']
    Stat['Date'] = S_date
    return Stat

# Histogram BIW ------------------------------------------------------------------------------------------------------------------

def HistogramPLOT(data,month,year):
    #Initiate
    Situation = []
    mon = ['January','Febuary','March','April','May','June','July','August','September','October','November','December']
    data01 = data[['DateTime','WS95']].copy()
    data01.dropna(how='any',inplace=True)
    logicY = (data01["DateTime"].apply(lambda x: x.year)==(year))
    data01 = data01[logicY].copy()
    fig = plt.figure(figsize=(20,32), facecolor='w', edgecolor='r')
    #Plotting 12 graph
    xvals = np.linspace(0,30,1000)
    for i in range(month):
        ax = plt.subplot2grid((4,3),(int(np.floor(i/3)),int(i%3)))
        logic = (data01["DateTime"].apply(lambda x: x.month))==(i+1)
        ws = data01['WS95'][logic]
        ws = ws+0.0001
        failures = []
        censored = []
        threshold = 30
        for item in ws:
            if item>threshold:
                censored.append(threshold)
            else:
                failures.append(item)
        xvals = np.linspace(0,30,1000)
        if (np.sum(logic)!=0):
            ax.hist(ws,bins=30,density=True)
            hist,edge = np.histogram(np.array(ws),bins=1000,range=(0,30)  ,density=True)
        ax.set_ylim(0,0.18)
        ax.set_xlim(0,30)
        ax.set_xticks([0,5,10,15,20,25,30])
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=26)
        ax.set_title('{}'.format(mon[i]),fontweight='bold',size=30)
    plt.tight_layout()
    plt.show()

def Histogram_Season(data,year,season):
    #Initiate
    Seasons= {1:'Winter',2:'Spring',3:'Summer',4:'Autumn',}
    logicY = (data["DateTime"].apply(lambda x: x.year)==(year))
    data01 = data[logicY].copy() 
    fig, (ax) = plt.subplots(1, 1,figsize=(10, 10), gridspec_kw = {'wspace':0, 'hspace':0})
    #Plotting 12 graph
    logic = (data01["Season"]==(season))
    ws = data01['WS95'][logic]
    if (np.sum(logic)!=0):
        ax.hist(ws,bins=30,density=True)
    ax.set_ylim(0,0.14)
    ax.set_xlim(0,30)
    ax.set_xticks([0,5,10,15,20,25,30])
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=26)
    ax.set_title('{}'.format(Seasons[season]),fontweight='bold',size=30)
    plt.tight_layout()
    plt.show()


def HistogramPLOT_wbm(data,month,year):
    #Initiate
    Situation = []
    mon = ['January','Febuary','March','April','May','June','July','August','September','October','November','December']
    data01 = data[['DateTime','WS95']].copy()
    data01.dropna(how='any',inplace=True)
    logicY = (data01["DateTime"].apply(lambda x: x.year)==(year))
    data01 = data01[logicY].copy()
    fig = plt.figure(figsize=(20,32), facecolor='w', edgecolor='r')
    #Plotting 12 graph
    xvals = np.linspace(0,30,1000)
    for i in range(month):
        ax = plt.subplot2grid((4,3),(int(np.floor(i/3)),int(i%3)))
        logic = (data01["DateTime"].apply(lambda x: x.month))==(i+1)
        ws = data01['WS95'][logic]
        ws = ws+0.0001
        failures = []
        censored = []
        threshold = 30
        for item in ws:
            if item>threshold:
                censored.append(threshold)
            else:
                failures.append(item)
        xvals = np.linspace(0,30,1000)
        if (np.sum(logic)!=0):
            ax.hist(ws,bins=30,density=True)
            hist,edge = np.histogram(np.array(ws),bins=1000,range=(0,30)  ,density=True)
            wbm = Fit_Weibull_Mixture(failures=failures,right_censored=censored,show_plot=False,print_results=False)
            part1_pdf = Weibull_Distribution(alpha=wbm.alpha_1,beta=wbm.beta_1).PDF(xvals=xvals,show_plot=False)
            part2_pdf = Weibull_Distribution(alpha=wbm.alpha_2,beta=wbm.beta_2).PDF(xvals=xvals,show_plot=False)
            Mixture_PDF = part1_pdf*wbm.proportion_1+part2_pdf*wbm.proportion_2
            ax.plot(xvals,Mixture_PDF,label='Weibull_Mixture')
        ax.legend()
        ax.set_ylim(0,0.18)
        ax.set_xlim(0,30)
        ax.set_xticks([0,5,10,15,20,25,30])
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=26)
        ax.set_title('{}'.format(mon[i]),fontweight='bold',size=30)
    plt.tight_layout()
    plt.show()


def HistogramPLOT_all(data,month,year):
    #Initiate
    Situation = []
    mon = ['January','Febuary','March','April','May','June','July','August','September','October','November','December']
    #Get just Full day data
    logicF = (data["isFULL"].apply(lambda x: x)==(1))
    data01 = data[logicF].copy()
    data01.fillna(method='ffill',inplace=True)
    
    logicY = (data01["DateTime"].apply(lambda x: x.year)==(year))
    data01 = data01[logicY].copy()

    fig = plt.figure(figsize=(24,18), dpi= 80, facecolor='w', edgecolor='r')
    #Plotting 12 graph
    xvals = np.linspace(0,30,1000)
    for i in range(month):
        ax = plt.subplot2grid((4,3),(int(np.floor(i/3)),int(i%3)))
        logic = (data01["DateTime"].apply(lambda x: x.month))==(i+1)
        ws = data01['WS95'][logic]
        ws = ws+0.0001
        failures = []
        censored = []
        threshold = 30
        for item in ws:
            if item>threshold:
                censored.append(threshold)
            else:
                failures.append(item)
        xvals = np.linspace(0,30,1000)
        print(ws.shape)
        if (np.sum(logic)!=0):
            ax.hist(ws,bins=30,normed=True)
            hist,edge = np.histogram(np.array(ws),bins=1000,range=(0,30)  ,normed=True)
            wb2 = Fit_Weibull_2P(failures=failures,show_probability_plot=False,print_results=False)
            wb3 = Fit_Weibull_3P(failures=failures,show_probability_plot=False,print_results=False)
            gm2 = Fit_Gamma_2P(failures=failures,show_probability_plot=False,print_results=False)
            gm3 = Fit_Gamma_3P(failures=failures,show_probability_plot=False,print_results=False)
            ln2 = Fit_Lognormal_2P(failures=failures,show_probability_plot=False,print_results=False)
            wbm = Fit_Weibull_Mixture(failures=failures,right_censored=censored,show_plot=False,print_results=False)
            
            wb2_pdf   = Weibull_Distribution(alpha=wb2.alpha, beta=wb2.beta).PDF(xvals=xvals, show_plot=True,label='Weibull_2P' )
            wb3_pdf   = Weibull_Distribution(alpha=wb3.alpha, beta=wb3.beta,gamma=wb3.gamma).PDF(xvals=xvals, show_plot=True,label='Weibull_3P')
            gm2_pdf   = Gamma_Distribution(alpha=gm2.alpha, beta=gm2.beta).PDF(xvals=xvals, show_plot=True,label='Gamma_2P' )
            gm3_pdf   = Gamma_Distribution(alpha=gm3.alpha, beta=gm3.beta, gamma=gm3.gamma).PDF(xvals=xvals, show_plot=True,label='Gamma_3P')
            ln2_pdf   = Lognormal_Distribution(mu=ln2.mu, sigma=ln2.sigma).PDF(xvals=xvals, show_plot=True,label='Lognormal_2P' )
            
            part1_pdf = Weibull_Distribution(alpha=wbm.alpha_1,beta=wbm.beta_1).PDF(xvals=xvals,show_plot=False)
            part2_pdf = Weibull_Distribution(alpha=wbm.alpha_2,beta=wbm.beta_2).PDF(xvals=xvals,show_plot=False)
            Mixture_PDF = part1_pdf*wbm.proportion_1+part2_pdf*wbm.proportion_2
            ax.plot(xvals,Mixture_PDF,label='Weibull_Mixture')
        ax.legend()
        ax.set_ylim(0,0.16)
        ax.set_xlim(0,30)
        ax.set_xticks([0,5,10,15,20,25,30])
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_title('{}'.format(mon[i]),fontweight='bold',size=20)
    plt.tight_layout()
    plt.show()
