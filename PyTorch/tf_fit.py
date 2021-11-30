# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:11:01 2021

@author: Tech
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


x = np.linspace(-10,10,400)
y = np.sin(x) + 0.05*x*np.random.normal(size=400) + 0.1*x*np.exp(0.025*x)

x_plot=np.linspace(-10,10,2000)

model = tf.keras.Sequential()

model.add (tf.keras.Input (shape=(1,)))
model.add (tf.keras.layers.Dense(16,activation="relu"))
model.add (tf.keras.layers.Dense(32,activation="softmax"))
model.add (tf.keras.layers.Dense(16,activation="relu"))
model.add (tf.keras.layers.Dense(1,activation="linear"))
opt_adam=tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile (loss='mse',optimizer=opt_adam)


model.fit (x,y, epochs=200, verbose=0)

y_pred=model.predict (x_plot)

plt.plot (x,y,'o')
plt.plot(x_plot,y_pred,color='blue')
plt.show()