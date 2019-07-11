# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:12:38 2019


After an example of Jason Brownlee:
https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

@author: timvmou
"""

# Larger CNN for the MNIST Dataset
import numpy
import git
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard

from GIRAFFE.code.neural_net import NeuralNet

K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# build the model
model = NeuralNet(y_test.shape[1])
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Save logs in this folder
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
logdir="logs/" + sha
# Fit the model
model.fit(
        X_train, 
        y_train, 
        validation_data=(X_test, y_test), 
        epochs=10, 
        batch_size=200,
        callbacks=[TensorBoard(log_dir=logdir)]
 )
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))