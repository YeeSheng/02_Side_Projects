import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
from datetime import timedelta

m_day = [31,28,31,30,31,30,31,31,30,31,30,31]

def DataRange(data,start_date,stop_date):
    log1   = (data['DateTime'] >=start_date)
    log2   = (data['DateTime'] < stop_date)   
    rangeT = np.logical_and(log1,log2)    
    df_out = pd.DataFrame(data[rangeT])
    df_out.reset_index(inplace=True,drop=True)
    return df_out

def DataRange2(data,start_date,duration):
    # strptime already for start_date
    # start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
    stop_date  = start_date+timedelta(days=duration)
    log1   = (data['DateTime'] >=start_date)
    log2   = (data['DateTime'] < stop_date)   
    rangeT = np.logical_and(log1,log2)    
    df_out = pd.DataFrame(data[rangeT])
    df_out.reset_index(inplace=True,drop=True)
    return df_out


def DETECT_FULL_DATA_DAY(raw,Count_month):
    #drop other data
    data = raw[['DateTime','WS95']].copy()
    print(len(data))
    #find which day is full, 1 = full, 0 = percent<95%
    FULL = []
    scale= 1 #1-min
    total = 24*60/scale
    data.dropna(how='any',inplace=True)
    for i in range(Count_month):
        start_m= data["DateTime"][1].month
        logicM = (data["DateTime"].apply(lambda x: x.month))==(i+start_m)
        MONTH = data[logicM]
        for j in range(m_day[i+start_m-1]):
            logicD =(MONTH["DateTime"].apply(lambda x: x.day))==(j+1)
            DAY = MONTH[logicD]
            Percent = len(DAY)/total*100
            Percent = np.round(Percent,2)
            if Percent >= 95:
                k = (np.ones(int(total)))*1
            else:
                k = (np.ones(int(total)))*0
            FULL = np.concatenate((FULL,k), axis=None)
        print(len(FULL))
    print(len(FULL))
    return FULL

def Year_Monthly(data):
    df = data.copy()
    df['Year'] = data["DateTime"].apply(lambda x:x.year)
    mon = {1:'January'  ,2:'Febuary' ,3:'March'    ,4:'April',
           5:'May'      ,6:'June'    ,7:'July'     ,8:'August',
           9:'September',10:'October',11:'November',12:'December'}
    months = data["DateTime"].apply(lambda x: x.month)
    df['month'] = months
    df.replace({'month': mon },inplace=True)
    return df
