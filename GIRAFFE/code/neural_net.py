'''
Created by the GiraffeTools Tensorflow generator.
Warning, here be dragons.

'''

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

# Model
def NeuralNet(shape):
    model = Sequential()

    model.add(Conv2D(
      #mandatory argument,  # filters
      #mandatory argument,  # kernel_size,
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='valid'
    ))

    model.add(Conv2D(
      #mandatory argument,  # filters
      #mandatory argument,  # kernel_size,
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='valid'
    ))

    model.add(Dropout(
      #mandatory argument,  # rate
    ))

    model.add(Flatten(

    ))

    model.add(Dense(
      #mandatory argument,  # units,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(Dense(
      #mandatory argument,  # units,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(Dense(
      #mandatory argument,  # units,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    # Returning model
    return model
