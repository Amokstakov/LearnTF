#import 
import tensorflow as tf

from helpers import breaker
from model import build_network, compile_model

mnist = tf.keras.datasets.mnist

# dataset
"""
We gather our training and testin data
the x_train/test represents the data and what were going to feed into the model

the y_train/test represents the label of the data above
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize 
"""
What is normalization?
It is the process of standardising our data so that our model has no biases or variances
"""

breaker()

x_train, x_test = x_train/255.0, x_test/255.0

model = build_network()
compile_model(model)

'''
We use model.fit to essentially train our compiled model with our training data!
'''

model.fit(x_train, y_train, epochs=5)

"""
We need to evaluate our model because training only focuses on training data - we need to know how our model does on data its never seen
that is why we use model.evaluate on testing data to analyse how well our model does generalising.

"""

model.evaluate(x_test, y_test, verbose=2)























