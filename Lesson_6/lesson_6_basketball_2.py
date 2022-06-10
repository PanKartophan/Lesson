import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

random.seed(13)
np.random.seed(13)
set_seed(13)
np.set_printoptions(threshold=np.inf)


# PRO (Вариант 4).
#######################################################################################################################
def plot_metric(history, metric='maeUnscaled_metric_fn', title_mae='MAE'):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(title_mae)
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def cutting_window(data, window=30):
    data_chunk = []
    result = []
    start_idx = 0
    for i in range(1, len(data)):
        if (data[i, 0] >= data[i-1, 0]) and (i != len(data) - 1):
            i += 1
        else:
            fin_idx = i - 1
            fcount = data[fin_idx, 0]
            for j in range(start_idx, fin_idx - window + 2):
                data_chunk.append(data[j:j+window, :])
                result.append(fcount)
            start_idx = i
    return np.array(data_chunk), np.array(result)


def to_vec(data_chunk):
    data_vec = []
    for sample_idx in range(len(data_chunk)):
        vec = np.zeros(3300)
        for time_idx in range(len(data_chunk[sample_idx]) - 1):
            left = int(data_chunk[sample_idx, time_idx, 1])
            right = int(data_chunk[sample_idx, time_idx + 1, 1])
            vec[left:right] = data_chunk[sample_idx, time_idx, 0]
        vec[right:] = data_chunk[sample_idx, time_idx + 1, 0]
        data_vec.append(vec)
    return np.array(data_vec)




df = pd.read_csv('./basketball.csv', encoding='cp1251', sep=';', header=0, index_col=0)
df['total'] = df['Ком. 1'] + df['Ком. 2']
data = df[['total', 'ftime']].values
# y_std = data[:, 0].std()
y_max = data[:, 0].max()
y_min = data[:, 0].min()
# data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()
data[:, 0] = (data[:, 0] - y_min) / (y_max - y_min)
X, Y = cutting_window(data, window=30)
X = to_vec(X)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)


# def maeUnscaledNorm_metric_fn(y_true, y_pred):
#     return tf.reduce_mean(tf.abs(y_true - y_pred)) * y_std


def maeUnscaledMinMax_metric_fn(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred)) * (y_max - y_min)


def train(lr=1.e-3, epochs=10, batch_size=128):
    model = Sequential()
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))

    #model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr, amsgrad=True), loss='mse', metrics=[maeUnscaledMinMax_metric_fn])

    history = model.fit(X_train, Y_train,
                        epochs=epochs, validation_data=(X_val, Y_val),
                        verbose=1, shuffle=True, batch_size=batch_size)
    return history


history = train(lr=1.e-3, batch_size=512, epochs=500)
plot_metric(history, metric='maeUnscaledMinMax_metric_fn', title_mae='MAE')


