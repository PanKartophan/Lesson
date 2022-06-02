import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

np.random.seed(13)
set_random_seed(13)

(x_train_val, y_train_val),(x_test, y_test) = mnist.load_data()
x_train_val, x_test = x_train_val.reshape(60000, 28 * 28), x_test.reshape(10000, 28 * 28)
x_train_val, x_test = x_train_val.astype('float32') / 255., x_test.astype('float32') / 255.
y_train_val, y_test = utils.to_categorical(y_train_val, 10), utils.to_categorical(y_test, 10)


def train(neurons_num=100, activation='relu', batch_size=128, epochs=20, lr=0.001, n_train=50000):
    model = Sequential()
    model.add(Dense(neurons_num, input_dim=28 * 28, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    history = model.fit(x_train_val[:n_train], y_train_val[:n_train], batch_size=batch_size, epochs=epochs, verbose=0,
                        validation_data=(x_train_val[n_train:], y_train_val[n_train:]))
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


def plot_mse(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MSE')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


# LIGHT
"""
Пункт 1.
Варьируем размер тренировочной выборки: 50000, 10000, 500 примеров.
"""
#####################################################################################################################
accuracy_df = pd.DataFrame(columns=['model_name', 'n_train', 'val_acc, %', 'test_acc, %'])

n_train = 50000
model, history = train(n_train=n_train)
plot_loss_acc(history)
test_metrics = model.evaluate(x_test, y_test, verbose=False)
accuracy_df.loc[0] = ['model', n_train, history.history['val_accuracy'][-1] * 100., test_metrics[-1] * 100.]

n_train = 10000
model, history = train(n_train=n_train)
plot_loss_acc(history)
accuracy_df.loc[1] = ['model', n_train, history.history['val_accuracy'][-1] * 100., '-']

n_train = 500
model, history = train(n_train=n_train)
plot_loss_acc(history)
accuracy_df.loc[2] = ['model', n_train, history.history['val_accuracy'][-1] * 100., '-']

"""
Пункт 2.
Создадим ещё две сети и сравним значения точности предсказания на валидационной и тестовой выборке.
"""
#####################################################################################################################
def train_net2(neurons_num=100, activation='relu', batch_size=128, epochs=20, lr=0.001, n_train=50000):
    model = Sequential()
    model.add(Dense(neurons_num, input_dim=28 * 28, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    history = model.fit(x_train_val[:n_train], y_train_val[:n_train], batch_size=batch_size, epochs=epochs, verbose=0,
                      validation_data=(x_train_val[n_train:], y_train_val[n_train:]))
    return model, history


model_net2, history_net2 = train_net2()
plot_loss_acc(history)
test_metrics = model_net2.evaluate(x_test, y_test, verbose=False)
accuracy_df.loc[3] = ['model_net2', 50000, history_net2.history['val_accuracy'][-1] * 100., test_metrics[-1] * 100.]


def train_net3(neurons_num=100, activation='relu', batch_size=128, epochs=20, lr=0.001, n_train=50000):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(28 * 28,)))
    model.add(Dense(neurons_num, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    history = model.fit(x_train_val[:n_train], y_train_val[:n_train], batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_train_val[n_train:], y_train_val[n_train:]))
    return model, history


model_net3, history_net3 = train_net3(n_train=50000)
plot_loss_acc(history)
test_metrics = model_net3.evaluate(x_test, y_test, verbose=False)
accuracy_df.loc[4] = ['model_net3', 50000, history_net3.history['val_accuracy'][-1] * 100., test_metrics[-1] * 100.]

print(accuracy_df)

"""
model: Dense(ReLU) -> Dense(SoftMax).
model_2: Dense(ReLU) -> Dropout(0.2) -> Dense(SoftMax).
model_3: BatchNorm -> Dense(ReLU) -> BatchNorm -> Dense(SoftMax).
################################################################################
model, 50000 обучение, 10000 валидация: val_acc: 97.7 %, test_acc: 97.6 %.
model, 10000 обучение, 50000 валидация: val_acc: 94.3 %.
model, 500 обучение, 59500 валидация: val_acc: 82.6 %.
model_2, 50000 обучение, 10000 валидация: val_acc: 97.8 %, test_acc: 97.6 %.
model_3, 50000 обучение, 10000 валидация: val_acc: 97.5 %, test_acc: 97.5 %.
################################################################################
"""
print('1) С уменьшением тренировочной выборки снижается точность предсказания модели.')
print('2) Добавление слоёв BatchNormalization привело к более быстрому появлению переобучения сети.')


# Пункт 3.
#####################################################################################################################
def train_net_hard(neurons_num=100, activation='relu', batch_size=128, epochs=20, lr=0.001, n_train=50000):
    model = Sequential()
    model.add(Dense(neurons_num, input_dim=28 * 28, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(neurons_num, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(neurons_num, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    history = model.fit(x_train_val[:n_train], y_train_val[:n_train], batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_train_val[n_train:], y_train_val[n_train:]))
    return model, history


model_net3, history_net3 = train_net3(n_train=50000)
plot_loss_acc(history)
test_metrics = model_net3.evaluate(x_test, y_test, verbose=False)
print('Точность модели из пункта 3:', test_metrics[1] * 100.)

"""
Точность предсказания модели на валидационном и тестовом наборах совпадает с ранее обучаемыми моделями.
Усложнение архитектуры сети не дало никакого улучшения.
"""

# PRO
# Вариант 1.
#####################################################################################################################
df = pd.read_csv("./Lesson_2/sonar.csv", header=None)

dataset = df.values
X = dataset[:, :-1].astype('float32')
Y = dataset[:, -1]
Y[Y == 'R'] = '0'
Y[Y == 'M'] = '1'
Y = Y.astype('uint8')

x_train_val, x_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1, shuffle=True)


def train_mine(neurons_num=10, activation='relu', batch_size=8, epochs=200, lr=0.0005):
  model = Sequential()
  model.add(BatchNormalization())
  model.add(Dense(neurons_num, input_dim=60, activation=activation))
  #model.add(BatchNormalization())
  model.add(Dropout(0.2))
  # model.add(Dense(40, activation=activation))
  # model.add(BatchNormalization())
  #model.add(Dropout(0.1))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_val, y_val))
  return model, history


model_mine, history_mine = train_mine()
plot_loss_acc(history_mine)
print('Test Accuracy (Mine Dataset):', round(model_mine.evaluate(x_test, y_test, verbose=False)[-1] * 100., 1), '%')

# Вариант 2.
#####################################################################################################################
cars_df = pd.read_csv('./Lesson_2/cars_new.csv', sep=',')


def create_dict(s):
    ret = {}
    for _id, name in enumerate(s):
        ret.update({name: _id})
    return ret


def to_ohe(value, d):
    arr = [0] * len(d)
    arr[d[value]] = 1
    return arr


marks_dict = create_dict(set(cars_df['mark']))
models_dict = create_dict(set(cars_df['model']))
bodies_dict = create_dict(set(cars_df['body']))
kpps_dict = create_dict(set(cars_df['kpp']))
fuels_dict = create_dict(set(cars_df['fuel']))
prices = np.array(cars_df['price'], dtype=np.float32)
years = preprocessing.scale(cars_df['year'])
mileages = preprocessing.scale(cars_df['mileage'])
volumes = preprocessing.scale(cars_df['volume'])
powers = preprocessing.scale(cars_df['power'])

x = []
y = []

for _id, car in enumerate(np.array(cars_df)):
    y.append(prices[_id])

    x_tr = to_ohe(car[0], marks_dict) + \
           to_ohe(car[1], models_dict) + \
           to_ohe(car[5], bodies_dict) + \
           to_ohe(car[6], kpps_dict) + \
           to_ohe(car[7], fuels_dict) + \
           [years[_id]] + \
           [mileages[_id]] + \
           [volumes[_id]] + \
           [powers[_id]]
    x.append(x_tr)

x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1,1)).flatten()

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y_scaled, test_size=0.2, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, shuffle=True)


def train_cars(batch_size=4096, epochs=50, lr=0.001, activation='relu', size_1 = 3000, size_2 = 300, size_3 = 30):
  model = Sequential()
  model.add(Dense(size_1, input_dim=3208, activation=activation))
  model.add(Dense(size_2, activation=activation))
  model.add(Dense(size_3, activation=activation))
  model.add(Dense(10, activation=activation))
  model.add(Dense(1, activation='linear'))
  model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_val, y_val))
  return model, history


model_cars, history_cars = train_cars(activation='relu',
                                      size_1=1000, size_2=1000, size_3=150, batch_size=16384, epochs=30)
plot_mse(history_cars)
y_val_unscaled = y_scaler.inverse_transform(y_val.reshape(-1, 1))
y_predict = y_scaler.inverse_transform(model_cars.predict(x_val, verbose=False))
error = np.abs((y_val_unscaled - y_predict) / y_val_unscaled).mean() * 100.
print('Error (Cars Dataset:', round(error, 1), '%')
