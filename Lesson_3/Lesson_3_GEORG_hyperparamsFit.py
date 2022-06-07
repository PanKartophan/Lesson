import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D,  \
                                    GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.random import set_seed

random.seed(13)
np.random.seed(13)
set_seed(13)


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
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


# ULTRA PRO
#######################################################################################################################
train_path = './georges/train'
val_path = './georges/val'
test_path = './georges/test'
classes = ['georges', 'non_georges']
img_width = 224
img_height = 224


def train_georg(dropout=0.4, n_dense_neurons=512, lr=0.001, batch_size=32, epochs=10):
    batch_size = batch_size

    train_datagen = ImageDataGenerator(
        rescale=1. / 255.,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0.
    )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    val_generator = ImageDataGenerator(rescale=1. / 255.).flow_from_directory(
        val_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    steps_per_epoch = (train_generator.samples + batch_size - 1) // batch_size
    validation_steps = (val_generator.samples + batch_size - 1) // batch_size

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    x = Dense(n_dense_neurons, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, amsgrad=True), metrics=['accuracy'])

    train_generator.reset()
    val_generator.reset()
    history = model.fit_generator(train_generator, validation_data=val_generator, epochs=epochs,
                                  steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                  verbose=0)
    return model, history


hyper_params = {'dropout': [0., 0.25, 0.5], 'n_dense_neurons': [32, 128, 512],
                'lr': [1.e-4, 1.e-3, 1.e-2], 'batch_size': [32, 64, 128], 'val_accuracy': []}
df = pd.DataFrame(columns=[*hyper_params])

n_row = 0
for dropout in hyper_params['dropout']:
    for n_dense_neurons in hyper_params['n_dense_neurons']:
        for lr in hyper_params['lr']:
            for batch_size in hyper_params['batch_size']:
                model_georg, history_georg = train_georg(dropout=dropout, n_dense_neurons=n_dense_neurons,
                                                         lr=lr, batch_size=batch_size)
                df.loc[n_row] = [dropout, n_dense_neurons, lr,
                                 batch_size, round(max(history_georg.history['val_accuracy']) * 100., 2)]
                print(dropout, n_dense_neurons, lr,
                                 batch_size, round(max(history_georg.history['val_accuracy']) * 100., 2))
                n_row += 1

df.to_csv('./GEORG_hyperparamsFit.csv', index=False)
