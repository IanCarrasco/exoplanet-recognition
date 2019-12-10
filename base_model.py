import numpy as np
import pandas as pd
from numpy.random import seed
from tensorflow import set_random_seed

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score

from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Conv1D, GRU, Flatten, MaxPooling1D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

def model(input_shape)
   
    #Define Input Layer
    X_input = Input(shape = input_shape)
    
    #Perform a 1D Convolution To Detect Features and Reduce Dimensionality
    X = Conv1D(32, kernel_size=10, strides=4)(X_input)
    X = MaxPooling1D(pool_size=4, strides=2)(X)
    X = Activation('relu')(X)
    
    #Pass Input Through GRU with Hidden Size 192
    X = GRU(192,return_sequences=True)(X)
    X = Flatten()(X)
    
    # Regularize Model With Dropout
    X = Dropout(0.5)(X)            

    #Additional Regularization with BatchNorm                     
    X = BatchNormalization()(X)   

    #Calculate Class Probability Score 
    X = Dense(1, activation="sigmoid")(X)

    #Initialize Model Object
    model = Model(inputs=X_input, outputs = X)
    
    return model  


def get_training_data():
    #Perform Train Data Processing
    train = pd.read_csv("./data/exoTrain.csv")

    #Get the number of observations and features in train set
    N, M = train.shape

    #Reduce Number of Features by 1 | To account for Label Col
    M -= 1

    #Store Training Labels
    y_train = train['LABEL'].values

    #Turn Labels from (1,2) to (0,1)
    y_train -=1

    #Store Training Data
    X_train = train.drop('LABEL', axis=1).values.reshape(N, M, 1)

    return X_train, y_train

def get_testing_data():
    test = pd.read_csv("../input/exoTest.csv")
    N, M = test.shape
    M -= 1

    y_test = test['LABEL'].values
    y_test -= 1

    X_test = test.drop('LABEL',axis=1).values.reshape(N, M, 1)

    return X_test, y_test

if __name__ == "__main__":

    seed(42)
    set_random_seed(42)

    gru_model = model(input_shape = (3197, 1))
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    gru_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    gru_model.summary()

    class_weight = {0: 1., 1: 10}
    
    history = gru_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=4, shuffle=True, class_weight=class_weight)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])