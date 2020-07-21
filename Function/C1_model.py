import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout,GRU,TimeDistributed
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import seed
from sklearn.model_selection import GridSearchCV

def GRU_model(dimension,timesteps,output,ACTIVE='elu',DROP=0.3,NEURON=256,OPTI='adam',STACKED_LAYER=2):
    model = Sequential()
    for i in range(STACKED_LAYER):
        model.add(GRU(NEURON, return_sequences=True,input_shape=(dimension,timesteps)))
        model.add(Dropout(DROP))
    for i in range(3-STACKED_LAYER):
        model.add(Dense(units=NEURON,kernel_initializer='normal', activation=ACTIVE))
        model.add(Dropout(DROP))
    model.add(TimeDistributed(Dense(units=int(output))))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer=OPTI)
    return model

def LSTM_model(dimension,timesteps,output,ACTIVE='elu',DROP=0.3,NEURON=256,OPTI='adam',STACKED_LAYER=2):
    model = Sequential()
    for i in range(STACKED_LAYER):
        model.add(GRU(NEURON, return_sequences=True,input_shape=(dimension,timesteps)))
        model.add(Dropout(DROP))
    for i in range(3-STACKED_LAYER):
        model.add(Dense(units=NEURON,kernel_initializer='normal', activation=ACTIVE))
        model.add(Dropout(DROP))
    model.add(TimeDistributed(Dense(units=int(output))))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer=OPTI)
    return model

