import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, LSTM, \
                                    GlobalMaxPooling1D, Bidirectional, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.random import set_seed
from sklearn.preprocessing import MinMaxScaler

random.seed(13)
np.random.seed(13)
set_seed(13)


def plot_metric(history, metric='maeUnscaled_metric_fn', title='MAE'):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()


def create_model_x(shape=(100, 1)):
    dataInput = Input(shape=shape)

    convWay = Conv1D(64, 3, activation="relu")(dataInput)
    convWay = Conv1D(64, 3, activation="relu")(convWay)
    convWay = GlobalMaxPooling1D()(convWay)
    convWay = Dropout(0.5)(convWay)

    denseWay = Dense(64, activation="relu")(dataInput)
    denseWay = Dense(64, activation="relu")(denseWay)

    convWay = Flatten()(convWay)
    denseWay = Flatten()(denseWay)

    finWay = concatenate([convWay, denseWay])
    finWay = Dense(1, activation="sigmoid")(finWay)

    model = Model(dataInput, finWay)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1.e-3))
    return model


def create_buy_sell_vec(prices, step):
    buy_sell = np.zeros_like(prices).astype(np.uint8)
    for i in range(len(buy_sell) - step):
        if prices[i+step] >= prices[i]:
            buy_sell[i] = 1.
    return buy_sell.reshape(-1, 1)


data16_17_df = pd.read_csv('./16_17.csv', sep=';')
data18_19_df = pd.read_csv('./18_19.csv', sep=';')
data = pd.concat([data16_17_df.iloc[:, 2:], data18_19_df.iloc[:, 2:]]).values
buy_cell_vec = create_buy_sell_vec(prices=data[:, 3], step=3)
print(buy_cell_vec.sum() / len(buy_cell_vec))

n_features = data.shape[-1]
# add columns with pair-wise column subtractions and absolute subtractions
for i in range(n_features - 1):
    for j in range(i + 1, n_features):
        data = np.hstack([data, (data[:, i] - data[:, j]).reshape(-1, 1),
                          np.fabs(data[:, i] - data[:, j]).reshape(-1, 1)])
# add columns with pair-wise column products
for i in range(n_features):
    for j in range(i, n_features):
        data = np.hstack([data, np.multiply(data[:, i], data[:, j]).reshape(-1, 1)])
# add columns with 1st, 2nd derivatives and reciprocal columns
for i in range(n_features):
    derivative_1 = np.hstack([np.zeros(1), data[1:, i] - data[:-1, i]]).reshape(-1, 1)
    derivative_2 = np.hstack([np.zeros(2), data[2:, i] - 2. * data[1:len(data) - 1, i] + data[:-2, i]]).reshape(-1, 1)
    data = np.hstack([data, np.reciprocal(data[:, i] + 1.e-3).reshape(-1, 1),
                      derivative_1, derivative_2])

x_len = 300
val_len = 30000
train_len = data.shape[0] - val_len

x_train, x_val = data[1:train_len], data[train_len+x_len+2:]
x_scaler = MinMaxScaler()
x_scaler.fit(x_train)
x_train = x_scaler.transform(x_train)
x_val = x_scaler.transform(x_val)

y_train, y_val = np.roll(buy_cell_vec, 1, axis=0)[1:train_len], np.roll(buy_cell_vec, 1, axis=0)[train_len+x_len+2:]

train_datagen = TimeseriesGenerator(x_train, y_train, length=x_len, stride=10, batch_size=300)
val_datagen = TimeseriesGenerator(x_val, y_val, length=x_len, stride=10, batch_size=300)
val_datagen_full = TimeseriesGenerator(x_val, y_val, length=x_len, stride=1, batch_size=len(x_val))

model_x = create_model_x(shape=train_datagen[0][0].shape[1:])
save_model_callback = ModelCheckpoint(filepath='./model.hdf5', save_weights_only=True, monitor='val_accuracy',
                                      mode='max', save_best_only=True)
history_x = model_x.fit_generator(train_datagen, epochs=10, verbose=1, validation_data=val_datagen,
                                  callbacks=[save_model_callback])
plot_metric(history_x, metric='loss', title='Loss')
plot_metric(history_x, metric='accuracy', title='Accuracy')






