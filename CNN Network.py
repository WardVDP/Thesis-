from numpy.core.fromnumeric import shape
import tensorflow as tf
import keras 
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.layers import Conv1D, MaxPooling1D
#from keras.layers.normalization import Batchnormalization
from keras.layers.advanced_activations import LeakyReLU, ReLU, Softmax
import pandas as pd 
import numpy as np
import openpyxl
from openpyxl import load_workbook
import pickle

Train_X = pickle.load(open("trainx.p","rb"))
Train_Y = pickle.load(open("trainy.p","rb"))
Validate_X = pickle.load(open("validatex.p","rb"))
Validate_Y = pickle.load(open("validatey.p","rb"))

Train_X = tf.expand_dims(Train_X, axis=0)
Train_Y = tf.expand_dims(Train_Y, axis=0)
Validate_X = tf.expand_dims(Validate_X, axis=0)
Validate_Y = tf.expand_dims(Validate_Y, axis=0)
print(Train_X)


class Model(tf.keras.Model):
    def __init__(self,use_dp=False, num_output=1):
        super(Model,self).__init__()
        self.use_dp = use_dp
        self.conv1 = Conv1D(32,21,activation = ReLU, input_shape = (2500,1))
        self.pool1 = MaxPooling1D(pool_size=4)
        self.conv2 = Conv1D(32,21,activation = ReLU)
        self.pool2 = MaxPooling1D(pool_size=4)
        self.conv3 = Conv1D(32,11,activation=ReLU)
        self.pool3 = MaxPooling1D(pool_size=4)
        self.dense = Dense( 1,activation=Softmax)

        if self.use_dp:
            use_dp = Dropout(0.5)

    def call(self,x):
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.conv3(x)
        x=self.pool3(x)
        return self.dense(x)

model = Model()

model.compile(loss=tf.losses.binary_crossentropy, optimizer='adam', metrics = ['accuracy'])

model.fit(Train_X,Train_Y,validation_data = (Validate_X,Validate_Y),epochs = 3)