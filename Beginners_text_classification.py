import os
import re
import shutil
import string
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

SEED = 42
EPOCHS = 10
SEQ_LEN = 250
BATCH_SIZE = 32
MAX_FEATURES = 10000
AUTOTUNE = tf.data.AUTOTUNE


def __structure_data(url):
    dataset = tf.keras.utils.get_file("aclImdb_v1",
                                      url,
                                      untar=True,
                                      cache_dir='.',
                                      cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    return dataset_dir


def remove_html_from_text(input_data):
    lower_case = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lower_case, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


def text_vectorize(max_features, seq_len, remove_html_from_text):
    vectorize_layer = TextVectorization(standardize=remove_html_from_text,
                                        max_tokens=max_features,
                                        output_mode='int',
                                        output_sequence_length=seq_len)

    return vectorize_layer


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def build_model(embedding_dim, max_features):
    model = tf.keras.Sequential([
        layers.Embedding(max_features+1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.summary()

    return model


def plot_training_history(history):
    acc = history['binary_accuracy']
    val_acc = history['val_binary_accuracy'] 
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# dataset_dir = __structure_data(url)

# train_dir = os.path.join(dataset_dir, 'train')
# remove_dir = os.path.join(train_dir, 'unsup')
# shutil.rmtree(remove_dir)

raw_train_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=SEED)

raw_val_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=SEED)

raw_test_dataset = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test', batch_size=BATCH_SIZE)

vectorize_layer = text_vectorize(MAX_FEATURES, SEQ_LEN, remove_html_from_text)
train_text = raw_train_dataset.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

text_batch, label_batch = next(iter(raw_train_dataset))
first_review, first_label = text_batch[0], label_batch[0]
print('Review', first_review)
print('Label', first_label)
print('Vectorize_text',
      vectorize_text(first_review, first_label))

train_dataset = raw_train_dataset.map(vectorize_text)
val_dataset = raw_val_dataset.map(vectorize_text)
test_dataset = raw_test_dataset.map(vectorize_text)

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

model = build_model(16, MAX_FEATURES)

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
             optimizer='adam',
             metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

loss, accuracy = model.evaluate(test_dataset)
print("Loss:", loss)
print("Accuracy:", accuracy)

plot_training_history(history.history)
