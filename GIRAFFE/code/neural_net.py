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
      30,  # filters
      (5,5),  # kernel_size,
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      activation='relu',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='valid'
    ))

    model.add(Conv2D(
      15,  # filters
      (3,3),  # kernel_size,
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      activation='relu',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(MaxPooling2D(
      pool_size=(2, 2),
      padding='valid'
    ))

    model.add(Dropout(
      0.2,  # rate
    ))

    model.add(Flatten(

    ))

    model.add(Dense(
      128,  # units,
      activation='relu',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(Dense(
      50,  # units,
      activation='relu',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    model.add(Dense(
      10,  # units,
      activation='softmax',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    ))

    # Returning model
    return model
