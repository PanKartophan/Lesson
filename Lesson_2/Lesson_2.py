from IPython.display import display
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing


df_sonar = pd.read_json('./Lesson_2/data.json')
print(df_sonar.head())
print(df_sonar.columns)

def train(neurons_num=100, activation='relu', batch_size=128, epochs=20, lr=0.001, n_train=50000):
    model = Sequential()
    model.add(Dense(neurons_num, input_dim=28 * 28, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    history = model.fit(x_train_val[:n_train], y_train_val[:n_train], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_train_val[n_train:], y_train_val[n_train:]))
    return model, history


def plot_loss_acc(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


# (x_train_val, y_train_val),(x_test, y_test) = mnist.load_data()
# x_train_val, x_test = x_train_val.reshape(60000, 28 * 28), x_test.reshape(10000, 28 * 28)
# x_train_val, x_test = x_train_val.astype('float32') / 255., x_test.astype('float32') / 255.
# y_train_val, y_test = utils.to_categorical(y_train_val, 10), utils.to_categorical(y_test, 10)
#
# model, history = train(n_train=50000, epochs=2)
# plot_loss_acc(history)
# display(pd.DataFrame(history.history).T)
# print(model.evaluate(x_test, y_test, verbose=False))