import tensorflow as tf

"""
We are building our network
Essentially it is a simple dense network
We we use Flatten as the first layer to flatten our input to 1D
We have a dense layer to process all of our inputs 
Dropout to reduce the probability of overfitting
and our final layer is an output layer with 10
The output layer should equal the amount of labels we are tring to predict


Example:
    0,1,2,3,4,5,6,7,8,9
    = 10 labels
"""

def build_network():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    return model

"""
What is compiling?
Compiling is basically giving our network fuel.
It provides the context of what the network is actually going to do
the optimizer is basically what dictates how much to penalise for correct or wrong predictions

"""
def compile_model(model):
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])












