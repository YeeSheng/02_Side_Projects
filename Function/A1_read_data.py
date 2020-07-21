import pandas as pd
import numpy as np

#Read Excel
#------------------------------#
# folder = /1.1 ALL windspeed/ or /Thermo/
# file_name list:
    # ALL_wind_{}-1min.xlsx
    # 
#------------------------------#
def read_excel_data(folder,file_name,sht=0):
    #year = 2016,2017,2018,2019
    data_path = 'D:/01_Academy/VScode/01_Data/'
    df   = pd.read_excel(data_path + folder +  file_name,sheet_name=sht)
    print("Import Done")
    observations = df.copy()
    return observations

#Read Csv
#------------------------------#
# folder = 1.1 ALL windspeed/ or /Thermo/
# file_name list:
    # 'ALL_wind_{}-1min.xlsx'
    # 'ALL-wind_1605-1911.csv'
    # 'WIND_95m_1605-1912.csv'
    # 'Wind_For_Spectrum.csv'
    # 'WS95_{}min_avg.csv'      
    # 'data1719_Full_Day.csv'
    #  'ALL_Thermo_{}-1min.csv'
#------------------------------#
def read_csv_data(folder,file_name):
    #year = 2016,2017,2018,2019
    data_path = 'D:/01_Academy/VScode/01_Data/'
    df   = pd.read_csv(data_path + folder + file_name)
    observations = df.copy()
    observations['DateTime'] = pd.to_datetime(observations['DateTime'])
    print("Import '{}'Done".format(file_name))
    return observations


