import numpy as np
import pandas as pd


def Convert_Angle01(data,Height):
    # from -pi~pi to 0~2pi
    pi = np.pi
    wd = data['WD{}n'.format(Height)].copy()
    logic    = wd.apply(lambda x: x>0)
    wd_new   = np.where(logic,wd, wd+360 )
    data['WD{}'.format(Height)] = wd_new
    return data

def Convert_Angle(data,Height,low,upp,sig):
    # for 0~360
    wd_A = 360 + data['WD{}'.format(Height)]
    wd_B = data['WD{}'.format(Height)]
    logic1 = (wd_A.apply(lambda x: x))>  upp
    logic2 = (wd_B.apply(lambda x: x))<= low
    wd1 = np.where(logic1, np.nan, wd_A)
    wd2 = np.where(logic2, np.nan, wd_B)
    wd = pd.DataFrame()
    wd['X'] = wd1
    wd['Y'] = wd2
    dfs = wd[['X','Y']]
    wd['C1']= dfs.mean(axis=1)
    data['WD{}{}'.format(Height,sig)] = wd['C1'].copy() 
    return  data

def Choose_Variable_xy(data):
    data['cos'] = np.cos(data['WD95']*np.pi/180)
    data['sin'] = np.sin(data['WD95']*np.pi/180)
    data['WS_y'] = data['cos']*data['WS95']
    data['WS_x'] = data['sin']*data['WS95']

    df_train = data[['DateTime','WS_x','WS_y']].copy()
    df_train.reset_index(inplace=True)
    return df_train

def Choose_Variable_Vxy(data):
    data['cos'] = np.cos(data['WD95']*np.pi/180)
    data['sin'] = np.sin(data['WD95']*np.pi/180)
    df_train = data[['DateTime','WS95','sin','cos']].copy()
    df_train.reset_index(inplace=True)
    return df_train

def To_Speed_Direction_Vxy(y_data,df):
    WS = y_data[:,0].flatten()
    sin = y_data[:,1].flatten()
    cos = y_data[:,2].flatten()
    R_bar = np.sqrt(sin**2+cos**2)
    WD = np.arctan2(sin,cos)*180/np.pi
    return WS,WD,R_bar
    
def Add_Season_Var(data):
    df = data.copy()
    seasons = [(month%12 + 3)//3 for month in range(1, 13)]
    month_to_season = dict(zip(range(1,13), seasons))
    df['Season']= df['Month'].apply(lambda x:month_to_season[x])
    Season_dict = {1:'Winter',2:'Spring',3:'Summer',4:'Autumn'}
    df['Seasons'] = df['Season'].apply(lambda x:Season_dict[x])
    return df
#data01['Monsoon']= data01['Month'].isin([1,2,3,10,11,12]) 