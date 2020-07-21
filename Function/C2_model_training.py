from A3_select_data import DataRange
from B1_sp_matrix import shaping_one
#from C1_Model import my_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout,GRU,TimeDistributed
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

def Train_Model(Train_data,LB,LF):
    start_date1, stop_date1 = "2017-01-01 00:00", "2018-01-01 00:01"
    df_train = DataRange(Train_data,start_date1,stop_date1)
    df_train.reset_index(inplace=True,drop=True)
    x_train,y_train = shaping_one(df_train,LB,LF)
    start_date2, stop_date2 = "2018-01-01 00:00", "2019-01-01 00:01"
    df_test = DataRange(Train_data,start_date2,stop_date2)
    df_test.reset_index(inplace=True,drop=True)
    x_test,y_test = shaping_one(df_test,LB,LF)
    dimension = x_train.shape[1]
    timesteps = x_train.shape[2]
    model = my_model(dimension,timesteps,output=int(len(LF)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model_ckpt = ModelCheckpoint(filepath="./tmp2.h5",monitor="val_loss", save_best_only=True)
    history = model.fit(x_train, y_train,validation_data=(x_test, y_test), batch_size=96, epochs=50, callbacks=[model_ckpt],shuffle=True)
    model = tf.keras.models.load_model("./tmp2.h5")
    return model,history


def Persistence_Method(data):
    per = data[['DateTime','WS95']].copy()
    per['V1'] = per['WS95'].copy()
    per['F1'] = per['WS95'].shift(1)
    per['F2'] = per['WS95'].shift(2)
    per['F3'] = per['WS95'].shift(3)
    per2 = DataRange(per,"2019-01-01 00:00","2020-01-01 00:00")
    per2.to_csv('WS_PER_Result.csv')


def Training_History(history,epoch_num,name):
    print(np.argmin(history.history['val_loss']))
    # Plot history: MSE
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    epoc = np.arange(1,epoch_num+1,1)
    print(epoc)
    ax.plot(epoc,history.history['loss'], label='MSE (training data)')
    ax.plot(epoc,history.history['val_loss'], label='MSE (testing data)')
    # model choose
    v_min = np.min(history.history['val_loss'])
    y = np.ones(epoch_num)*v_min
    ax.plot(epoc,y,color='r',linestyle='--' ,label='highest accuracy')

    ax.set_title('MSE for model training ({})'.format(name),fontsize=22)   
    ax.set_ylabel('MSE value',fontsize=22)
    ax.set_xlabel('No. epoch',fontsize=22)
    ax.tick_params(axis="x", labelsize=15,rotation=0)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_xlim([0,epoch_num])
    #ax.set_ylim([0,4])
    plt.legend(loc=1,fontsize=16)
    plt.show()
