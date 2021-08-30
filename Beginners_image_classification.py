
import numpy as np
import tensorflow as  tf
import matplotlib.pyplot as plt

from helpers import breaker
from model import build_network, compile_model 


def show_data(imgs, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


def preprocess_data(train, test):
    return train / 255.0, test / 255.0
    

fashion_mnist = tf.keras.datasets.fashion_mnist
# deconstruct trainng and testing data

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data
print(x_train.shape)
print(len(x_train))
breaker()
print(x_test.shape)
print(len(x_test))

show_data(x_train, y_train)
x_train, x_test = preprocess_data(x_train, x_test)

model = build_network()
compile_model(model)

model.fit(x_train, y_train, epochs=10)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(test_loss, test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model(x_test)
predictions[0]















