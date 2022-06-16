import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, LSTM, \
                                    GlobalMaxPooling1D, Bidirectional, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

random.seed(13)
np.random.seed(13)
set_seed(13)


# LIGHT (Вариант 1).
#######################################################################################################################
def plot_metric(history, metric='maeUnscaled_metric_fn', title='MAE'):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()


def create_model_x(shape=(100, 1), step=1):
    dataInput = Input(shape=shape)

    lstmWay = LSTM(50, return_sequences=True)(dataInput)
    lstmWay = Dropout(0.3)(lstmWay)
    lstmWay = LSTM(50, return_sequences=True)(lstmWay)
    lstmWay = Dropout(0.3)(lstmWay)
    lstmWay = LSTM(50, return_sequences=True)(lstmWay)
    lstmWay = Dropout(0.3)(lstmWay)

    convWay = Conv1D(100, 5, activation="relu")(dataInput)
    convWay = Dropout(0.3)(convWay)
    convWay = Conv1D(100, 5, activation="relu")(convWay)

    denseWay = Dense(10, activation="linear")(dataInput)
    denseWay = Dense(10, activation="linear")(denseWay)

    lstmWay = Flatten()(lstmWay)
    convWay = Flatten()(convWay)
    denseWay = Flatten()(denseWay)

    finWay = concatenate([lstmWay, convWay, denseWay])
    finWay = Dense(30, activation="relu")(finWay)
    finWay = Dense(step, activation="relu")(finWay)

    model = Model(dataInput, finWay)
    model.compile(loss='mse', optimizer=Adam(learning_rate=1.e-5))
    print(model.summary())
    return model


def get_pred(model, x_val, y_val, y_scaler):
    # Предсказываем ответ сети по проверочной выборке
    # И возвращаем исходный масштаб данных, до нормализации
    pred_val = y_scaler.inverse_transform(model.predict(x_val))
    y_val_unscaled = y_scaler.inverse_transform(y_val)
    return pred_val, y_val_unscaled


# Функция визуализирует графики, что предсказала сеть и какие были правильные ответы
# start - точка с которой начинаем отрисовку графика
# step - длина графика, которую отрисовываем
# channel - какой канал отрисовываем
def show_predict(start, step, channel, pred_val, y_val_unscaled):
    plt.plot(pred_val[start:start+step, channel], label='Прогноз')
    plt.plot(y_val_unscaled[start:start+step, channel], label='Базовый ряд')
    plt.xlabel('Время')
    plt.ylabel('Значение Close')
    plt.title('На ' + str(channel + 1) + ' шаг вперёд')
    plt.legend()
    plt.show()


# Функция расёта корреляции дух одномерных векторов
def correlate(a, b):
    ma = a.mean()
    mb = b.mean()
    mab = (a * b).mean()
    sa = a.std()
    sb = b.std()
    corr = 1
    if (sa > 0) & (sb > 0):
        corr = (mab - ma * mb) / (sa * sb)
    return corr


# Функция рисуем корреляцию прогнозированного сигнала с правильным
# Смещая на различное количество шагов назад
# Для проверки появления эффекта автокорреляции
# channels - по каким каналам отображать корреляцию
# corrSteps - на какое количество шагов смещать сигнал назад для рассчёта корреляции
def show_corr(channels, corr_steps, pred_val, y_val_unscaled):
    for ch in channels:
        corr = []  # Создаём пустой лист, в нём будут корреляции при смещении на i шагов обратно
        y_len = y_val_unscaled.shape[0]
        # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
        for i in range(corr_steps):
            corr.append(correlate(y_val_unscaled[:y_len-i, ch], pred_val[i:, ch]))
        own_corr = []  # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно
        # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
        for i in range(corr_steps):
            own_corr.append(correlate(y_val_unscaled[:y_len-i, ch], y_val_unscaled[i:, ch]))

        plt.plot(corr, label='Предсказание на ' + str(ch + 1) + ' шаг')
        plt.plot(own_corr, label='Эталон')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()


data16_17_df = pd.read_csv('./16_17.csv', sep=';')
data18_19_df = pd.read_csv('./18_19.csv', sep=';')
data = pd.concat([data16_17_df.iloc[:, 2:], data18_19_df.iloc[:, 2:]]).values[:300000]
print(data.shape)

x_len = 300
val_len = 30000
train_len = data.shape[0] - val_len

# Обучаем полносвязную сеть для прогнозирования на 1 шаг вперёд и визуализируем результаты.
x_train, x_val = data[:train_len], data[train_len+x_len+2:]
x_scaler = MinMaxScaler()
x_scaler.fit(x_train)
x_train = x_scaler.transform(x_train)
x_val = x_scaler.transform(x_val)

y_train, y_val = data[:train_len, 3].reshape(-1, 1), data[train_len+x_len+2:, 3].reshape(-1, 1)
y_scaler = MinMaxScaler()
y_scaler.fit(y_train)
y_train = y_scaler.transform(y_train)
y_val = y_scaler.transform(y_val)

train_datagen = TimeseriesGenerator(x_train, y_train, length=x_len, stride=10, batch_size=20)
val_datagen = TimeseriesGenerator(x_val, y_val, length=x_len, stride=10, batch_size=20)
val_datagen_full = TimeseriesGenerator(x_val, y_val, length=x_len, stride=1, batch_size=len(x_val))

model_x = create_model_x(shape=train_datagen[0][0].shape[1:],  step=1)
history_dense = model_x.fit_generator(train_datagen, epochs=4, verbose=1, validation_data=val_datagen)
plot_metric(history_dense, metric='loss', title='MSE')
pred_val, y_val_unscaled = get_pred(model_x, val_datagen_full[0][0], val_datagen_full[0][1], y_scaler)
show_predict(0, 1000, 0, pred_val, y_val_unscaled)
show_corr([0, ], 300, pred_val, y_val_unscaled)



