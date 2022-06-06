import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, Input, \
                                    GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
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

batch_size = 32
img_width = 224
img_height = 224

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
test_generator = ImageDataGenerator(rescale=1. / 255.).flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

steps_per_epoch = (train_generator.samples + batch_size - 1) // batch_size
validation_steps = (val_generator.samples + batch_size - 1) // batch_size


def train_georg(epochs=10, lr=0.001):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, amsgrad=True), metrics=['accuracy'])

    train_generator.reset()
    val_generator.reset()
    history = model.fit_generator(train_generator, validation_data=val_generator, epochs=epochs,
                                  steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                  verbose=1)
    return model, history


model_georg, history_georg = train_georg(lr=0.0003, epochs=50)
plot_loss_acc(history_georg, title_loss='Loss (Default Model)', title_acc='Accuracy (Default Model)')
model_georg.save_weights('./weights_georg_2.hdf5')

test_generator.reset()
score = model_georg.evaluate_generator(test_generator, (test_generator.samples + batch_size - 1) // batch_size)
print('Test Dataset:', 'Loss:', round(score[0], 3), 'Accuracy:', round(score[1], 3))
