import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Input, AveragePooling2D
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras import utils
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

np.random.seed(13)
utils.set_random_seed(13)


def train_cifar10(batch_size=256, epochs=30, lr=0.001):
    model = Sequential()

    model.add(BatchNormalization(input_shape=(32, 32, 3)))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr, amsgrad=True), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                        validation_data=(x_test, y_test))
    return model, history


# def train_cifar10_resnet(batch_size=512, epochs=10, lr=0.001):
#     basemodel = ResNet50(input_tensor=Input(shape=(32, 32, 3)), include_top=False,
#                        weights='imagenet')
#     headmodel = basemodel.output
#     headmodel = Flatten()(headmodel)
#     headmodel = Dense(256, activation='relu')(headmodel)
#     #headmodel = Dropout(0.25)(headmodel)
#     headmodel = Dense(10, activation='softmax')(headmodel)
#     model = Model(inputs=basemodel.input, outputs=headmodel)
#     #for layer in basemodel.layers:
#     #    layer.trainable = False
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
#     history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
#                         validation_data=(x_test, y_test))
#     return model, history


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


# PRO
# Вариант 2.
#######################################################################################################################
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model_cifar10, history_cifar10 = train_cifar10(lr=0.001, epochs=120, batch_size=256)
plot_loss_acc(history_cifar10, title_loss='Loss (Default Model)', title_acc='Accuracy (Default Model)')
print('Last 10 accuracies on validation dataset:', history_cifar10.history['val_accuracy'][-10:])

