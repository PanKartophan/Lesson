import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras import utils

np.random.seed(13)
utils.set_random_seed(13)


def train_mnist(batch_size=128, epochs=20, lr=0.001, n_filters=32, activation='relu'):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(28, 28, 1)))
    model.add(Conv2D(n_filters, 3, padding='same', activation=activation))
    model.add(Conv2D(n_filters, 3, padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                        validation_data=(x_test, y_test))
    return model, history


def plot_loss_acc(history, title_loss='Loss', title_acc='Accuracy'):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title_acc)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


# LIGHT
# Вариант 1.
#######################################################################################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
x_train = x_train[..., np.newaxis].astype('float32') / 255.
x_test = x_test[..., np.newaxis].astype('float32') / 255.

# # Создадим датафрейм для сведения в одну таблицу точности предсказания модели, обученной с различными гиперпараметрами
# accuracy_df = pd.DataFrame(columns=['model_name', 'n_filters', 'activation', 'batch_size', 'val_acc, %'])
#
# n_filters = 32; activation = 'relu'; batch_size = 128
# model_mnist, history_mnist = train_mnist(n_filters=n_filters, activation=activation, batch_size=batch_size)
# plot_loss_acc(history_mnist, title_loss='Loss (Default Model)', title_acc='Accuracy (Default Model)')
# accuracy_df.loc[0] = ['mnist', n_filters, activation, batch_size, history_mnist.history['val_accuracy'][-1] * 100.]
#
# # Вариант 2.
# #######################################################################################################################
# # Варьируем количество фильтров в свёрточных слоях
# df_row = 1
# for n_filters in [2, 4, 16]:
#     model_mnist, history_mnist = train_mnist(n_filters=n_filters, activation=activation, batch_size=batch_size)
#     plot_loss_acc(history_mnist, title_loss=('Loss (' + str(n_filters) + ' filters)'),
#                   title_acc=('Accuracy (' + str(n_filters) + ' filters)'))
#     accuracy_df.loc[df_row] = ['mnist', n_filters, activation, batch_size,
#                               history_mnist.history['val_accuracy'][-1] * 100.]
#     df_row += 1
#
# # Меняем функции активации в скрытых слоях на linear
# n_filters = 32; activation = 'linear'
# model_mnist, history_mnist = train_mnist(n_filters=n_filters, activation=activation, batch_size=batch_size)
# plot_loss_acc(history_mnist, title_loss=('Loss (' + activation + ' activation)'),
#                   title_acc=('Accuracy (' + activation + ' activation)'))
# accuracy_df.loc[df_row] = ['mnist', n_filters, activation, batch_size,
#                           history_mnist.history['val_accuracy'][-1] * 100.]
# df_row += 1
#
# # Варьируем размер батча
# activation = 'relu'
# for batch_size in [10, 100, 40000]:
#     model_mnist, history_mnist = train_mnist(n_filters=n_filters, activation=activation, batch_size=batch_size)
#     plot_loss_acc(history_mnist, title_loss=('Loss (' + str(batch_size) + ' batch_size)'),
#                   title_acc=('Accuracy (' + str(batch_size) + ' batch_size)'))
#     accuracy_df.loc[df_row] = ['mnist', n_filters, activation, batch_size,
#                               history_mnist.history['val_accuracy'][-1] * 100.]
#     df_row += 1
# print(accuracy_df)
# print('ВЫВОДЫ:',
#       '1) Увеличение числа фильтров в свёрточных слоях сети приводит к росту точности предсказания модели.',
#       '2) Функция активации Linear ухудшает точность предсказания модели по сравнению с ReLU.',
#       '3) Очень большой размер батча 40000 требует увеличения числа эпох тренировки (или увеличения скорости обучения).',
#       '4) Увеличение размера батча приводит к более "гладкому" (в отношении loss-функции) обучению модели.', sep='\n')

# PRO
# Вариант 1.
#######################################################################################################################
def train_mnist_research(batch_size=128, epochs=15, lr=0.001, n_filters=32, activation='relu', n_conv2d=2, n_neurons=256,
                       n_maxpooling2d=1, dropout=0.25):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(28, 28, 1)))
    for _ in range(n_conv2d):
        model.add(Conv2D(n_filters, 3, padding='same', activation=activation))
    for _ in range(n_maxpooling2d):
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(n_neurons, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                        validation_data=(x_test, y_test))
    return model, history


# Создадим датафрейм для сведения в одну таблицу точности предсказания модели, обученной с различными гиперпараметрами
accuracy_research_df = pd.DataFrame(columns=['model_name', 'n_conv2d', 'n_neurons',
                                             'n_maxpooling2d', 'dropout', 'val_acc, %'])
df_row = 0

# Варьируем количество слоёв Conv2D: 1, 2, 5, 10
n_neurons = 256; n_maxpooling2d = 1; dropout = 0.25
for n_conv2d in [1, 2, 5, 10]:
    model_mnist_research, history_mnist_research = train_mnist_research(n_conv2d=n_conv2d)
    plot_loss_acc(history_mnist_research, title_loss=('Loss (' + str(n_conv2d) + ' Conv2D)'),
                  title_acc=('Accuracy (' + str(n_conv2d) + ' Conv2D)'))
    accuracy_research_df.loc[df_row] = ['mnist_research', n_conv2d, n_neurons, n_maxpooling2d, dropout,
                               history_mnist_research.history['val_accuracy'][-1] * 100.]
    df_row += 1

# Варьируем количество нейронов в Dense-слое: 5, 50, 100, 256, 512, 1024
n_conv2d = 2; n_maxpooling2d = 1; dropout = 0.25
for n_neurons in [5, 50, 100, 256, 512, 1024]:
    model_mnist_research, history_mnist_research = train_mnist_research(n_neurons=n_neurons)
    plot_loss_acc(history_mnist_research, title_loss=('Loss (' + str(n_neurons) + ' neurons)'),
                  title_acc=('Accuracy (' + str(n_neurons) + ' neurons)'))
    accuracy_research_df.loc[df_row] = ['mnist_research', n_conv2d, n_neurons, n_maxpooling2d, dropout,
                               history_mnist_research.history['val_accuracy'][-1] * 100.]
    df_row += 1

# Варьируем количество слоёв MaxPooling2D: 0, 1, 2, 3
n_conv2d = 2; n_neurons = 256; dropout = 0.25
for n_maxpooling2d in [0, 1, 2, 3]:
    model_mnist_research, history_mnist_research = train_mnist_research(n_maxpooling2d=n_maxpooling2d)
    plot_loss_acc(history_mnist_research, title_loss=('Loss (' + str(n_maxpooling2d) + ' MaxPooling2D)'),
                  title_acc=('Accuracy (' + str(n_maxpooling2d) + ' MaxPooling2D)'))
    accuracy_research_df.loc[df_row] = ['mnist_research', n_conv2d, n_neurons, n_maxpooling2d, dropout,
                               history_mnist_research.history['val_accuracy'][-1] * 100.]
    df_row += 1

# Варьируем dropout: 0, 0.1, 0.25, 0.4, 0.5, 0,8
n_conv2d = 2; n_neurons = 256; n_maxpooling2d = 1
for dropout in [0, 0.1, 0.25, 0.4, 0.5, 0.8]:
    model_mnist_research, history_mnist_research = train_mnist_research(dropout=dropout)
    plot_loss_acc(history_mnist_research, title_loss=('Loss (' + str(dropout) + ' dropout)'),
                  title_acc=('Accuracy (' + str(dropout) + ' dropout)'))
    accuracy_research_df.loc[df_row] = ['mnist_research', n_conv2d, n_neurons, n_maxpooling2d, dropout,
                                        history_mnist_research.history['val_accuracy'][-1] * 100.]
    df_row += 1

print(accuracy_research_df)
print('ВЫВОДЫ:',
      '1) Увеличение числа свёрточных слоёв приводит к увеличению точности предсказания модели.',
      'Однако существует предел: с некоторого момента обучение модели не приводит к росту её качества.',
      '2) Увеличение числа нейронов в Dense-слое приводит к увеличению точности предсказания модели.',
      'Однако существует предел: с некоторого момента обучение модели не приводит к росту её качества.',
      'Более того, при дальнейшем уввличении числа нейронов начиается переобучение модели.'
      '3) Увеличение числа MaxPooling2D-слоёв не приводит к увеличению точности предсказания модели.',
      'Известно, однако, что их нужно ставить между свёрточными слоями, а не подряд.',
      '4) Увеличение dropout до 0.5 приводит к увеличению точности предсказания модели и исчезновению переобучения.',
      'Однако дальнейшее увеличение dropout ухудшает качество модели, требует большего времени обучения.', sep='\n')
