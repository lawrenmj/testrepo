# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:49:37 2022

@author: Matt
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
model.fit(xs, ys, epochs = 500)
print(model.predict([7.0]))