import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.random import set_seed

import random
import re
import matplotlib.pyplot as plt

random.seed(13)
np.random.seed(13)
set_seed(13)


def plot_loss_mae(history, title_mae='MAE'):
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title(title_mae)
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

# LIGHT, PRO (Вариаент 1).
#######################################################################################################################
fixed_df = pd.read_csv('./hh_fixed.csv')

# Убираем первый столбец с продублированными индексами. Он нам не нужен
fixed_df = fixed_df.drop(fixed_df.columns[0], axis=1)

i = 0
# Заменяем фрагменты кодировки ASCII на отсутствие символов
for j in range(12):
    for i in range(fixed_df.shape[0]):
        if type(fixed_df.values[i][j]) != float:
            fixed_df.values[i][j] = fixed_df.values[i][j].replace("\xa0", "")
            fixed_df.values[i][j] = fixed_df.values[i][j].replace("\n", " ")


# Данные о поле и возрасте
def getParameterSexAge(arg):
    out = [0, 0]
    # Если М, то 1. По умолчанию 0 - Ж
    if "М" in arg:
        out[0] = 1
    # текущий год - год рождения
    year_tec = 2020
    if len(arg) > 7:
        out[1] = year_tec - int(re.findall(r'\d{4}', arg)[0])
    return out


# Полученный возраст превращаем в класс возрастной категории
def getParameterAgeVect(arg):
    outClass = int((arg - 13) / 5)
    outClass = max(0, min(10, outClass))
    # На выходе получаем вектор с нужной категорией возраста
    return list(utils.to_categorical(outClass, 11).astype('int'))


# Зарплата
def getParameterSalary(arg):
    num = arg
    # Сначала получаем чистое число, убираем лишние знаки
    if (type(num) == str):
        num = re.sub(' ', '', num)
        num = re.sub('[а-яА-ЯёЁ]', '', num)
        num = re.sub('[a-zA-Z]', '', num)
        num = num.replace('.', '')

        # Получаем чисто валюту, убираем цифры
        curr = re.sub('[0-9]', '', arg)
        curr = curr.replace('.', '').strip()

        # Конвертируем в рубли, если валюта
        if curr == 'USD':
            num = float(num) * 65
        elif curr == 'KZT':
            num = float(num) * 0.17
        elif curr == 'грн':
            num = float(num) * 2.6
        elif curr == 'белруб':
            num = float(num) * 30.5
        elif curr == 'EUR':
            num = float(num) * 70
        elif curr == 'KGS':
            num = float(num) * 0.9
        elif curr == 'сум':
            num = float(num) * 0.007
        elif curr == 'AZN':
            num = float(num) * 37.5

    salaryStr = int(num)

    return salaryStr


# Данные о городе
def getParameterCity(arg):
    millionCities = "Новосибирск Екатеринбург Нижний Новгород Казань Челябинск Омск Самара Ростов-на-Дону Уфа " \
                    "Красноярск Пермь Воронеж Волгоград"
    sarg = arg.split(',')
    for item in sarg:
        item = item.strip()
        if item == "Москва":
            return [1, 0, 0, 0]
        if item == "Санкт-Петербург":
            return [0, 1, 0, 0]
        if item in millionCities:
            return [0, 0, 1, 0]

    return [0, 0, 0, 1]


# Данные о желаемой занятости
def getParameterEmployment(arg):
    out = [0, 0, 0, 0]
    if "стажировка" in arg:
        out[0] = 1
    if "частичная занятость" in arg:
        out[1] = 1
    if "проектная работа" in arg:
        out[2] = 1
    if "полная занятость" in arg:
        out[3] = 1

    return out


# Данные о желаемом графике работы
def getParameterSchedule(arg):
    out = [0, 0, 0, 0]
    if "гибкий график" in arg:
        out[0] = 1
    if "полный день" in arg:
        out[1] = 1
    if "сменный график" in arg:
        out[2] = 1
    if "удаленная работа" in arg:
        out[3] = 1

    return out


# Данные об образовании
def getParameterEducation(arg):
    out = [0, 0, 0, 0]  # По умолчанию не указано
    if arg in "Высшее Higher education":
        out[0] = 1
    if arg in "Среднее специальное":
        out[1] = 1
    if arg in "Неоконченное высшее":
        out[2] = 1
    if arg in "Среднее образование":
        out[3] = 1

    return out


# Данные об опыте работы
def getParameterExperience(arg):
    arg = str(arg)
    # Проверяем, если не пустая строка
    symbols = 0
    years = 0
    months = 0
    for s in arg:
        if s != " ":
            symbols += 1

    # Находим индексы пробелов около фразы "опыт работы"
    if symbols > 10:
        spacesIndexes = []
        index = 0
        while ((len(spacesIndexes) < 5) & (index < len(arg))):
            if arg[index] == " ":
                spacesIndexes.append(index)
            index += 1

        years = 0
        months = 0
        if arg[spacesIndexes[2] + 1] != "м":
            if len(spacesIndexes) >= 3:
                yearsStr = arg[spacesIndexes[1]:spacesIndexes[2]]  # Записываем в строку значение лет
                years = int(yearsStr)

            if len(spacesIndexes) >= 5:
                monthsStr = arg[spacesIndexes[3]:spacesIndexes[4]]  # Записываем в строку значение месяцев
                if arg[spacesIndexes[2] + 1] == "м":
                    months = int(monthsStr)
        else:
            if len(spacesIndexes) >= 3:
                monthsStr = arg[spacesIndexes[1]:spacesIndexes[2]]
                months = int(monthsStr)

    return 12 * years + months


# Категориальное представление опыта работы
def getParameterExperienceVector(arg):
    out = getParameterExperience(arg)
    outClass = 0
    if out > 6:  # если больше 6 месяцев
        outClass = 1
    if out > 12:  # если больше 12 месяцев
        outClass = 2
    if out > 24:  # если больше 24 месяцев
        outClass = 3
    if out > 36:  # если больше 36 месяцев
        outClass = 4
    if out > 60:  # если больше 60 месяцев
        outClass = 5
    if out > 96:  # если больше 96 месяцев
        outClass = 6
    if out > 120:  # если больше 120 месяцев
        outClass = 7
    if out > 156:  # если больше 156 месяцев
        outClass = 8
    if out > 192:  # если больше 192 месяцев
        outClass = 9
    if out > 240:  # если больше 240 месяцев
        outClass = 10

    return list(utils.to_categorical(outClass, 11).astype('int'))


# Извлекаем все параметры
def getAllParameters(val):
    result = getParameterSexAge(val[0])
    sex = result[0]  # getParameterSex() #параметры о поле
    age = getParameterAgeVect(result[1])  # параметры о возрасте
    city = getParameterCity(val[3])  # параметры о городе
    employment = getParameterEmployment(val[4])  # параметры о желаемой занятости
    shedule = getParameterSchedule(val[5])  # параметры о желаемом графике
    education = getParameterEducation(val[9])  # параметры об образовании
    experience = getParameterExperienceVector(val[6])  # параметры об опыте
    out = []

    # Склеиваем все параметры в вектор
    out.append(sex)
    out += age
    out += city
    out += employment
    out += shedule
    out += education
    out += experience

    return out


# Создаем тренировочную выборку
def get01Data(values):
    xTrain = []
    yTrain = []

    # Предсказывать будем зарплату
    for val in values:
        y = getParameterSalary(val[1])

        # Все, что не зарплата - обучающая выборка
        if y != -1:
            x = getAllParameters(val)
            xTrain.append(x)
            yTrain.append(y / 1000.)

    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    return (xTrain, yTrain)


# Выкачиваем данные по профессиям
def getXTrainTProf(values):
    xTrainTProf = []

    for val in values:
        currText = ""
        if type(val[3]) != float:
            currText += val[2]
        if type(val[7]) != float:
            currText += " " + val[7]

        if getParameterSalary(val[1]) != -1:  # Проверяем, если есть данные о зарплате
            xTrainTProf.append(currText)

    xTrainTProf = np.array(xTrainTProf)

    return xTrainTProf


# Выкачиваем данные по резюме
def getXTrainTRez(values):
    xTrainTRez = []

    for val in values:
        currText = ""
        if (type(val[6]) != float):
            currText += val[6]

        if (getParameterSalary(val[1]) != -1):
            xTrainTRez.append(currText)

    xTrainTRez = np.array(xTrainTRez)

    return xTrainTRez


# Извлекаем значения загруженного набора данных
(xTrain01, yTrain) = get01Data(fixed_df.values)
xTrainTProf = getXTrainTProf(fixed_df.values)

# Преобразовываем текстовые данные в числовые/векторные для обучения нейросетью
# определяем макс.кол-во слов/индексов, учитываемое при обучении текстов
maxWordsCount = 2000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0',
                      lower=True, split=' ', oov_token='unknown', char_level=False)
# "скармливаем" наши тексты, т.е даём в обработку методу, который соберет словарь частотности
tokenizer.fit_on_texts(xTrainTProf)
# преобразовываем текст в последовательность индексов согласно частотному словарю
xTrainProfIndexes = tokenizer.texts_to_sequences(xTrainTProf)
# Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
xTrainProf01 = tokenizer.sequences_to_matrix(xTrainProfIndexes)

# Вытаскиваем резюме для выборки
xTrainTRez = getXTrainTRez(fixed_df.values)
maxWordsCount = 2000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0',
                      lower=True, split=' ', oov_token='unknown', char_level=False)
tokenizer.fit_on_texts(xTrainTRez)
xTrainRezIndexes = tokenizer.texts_to_sequences(xTrainTRez)
xTrainRez01 = tokenizer.sequences_to_matrix(xTrainRezIndexes)

# np.save('./xTrain01.npy', xTrain01)
# np.save('./xTrainProf01.npy', xTrainProf01)
# np.save('./xTrainRez01.npy', xTrainRez01)
# np.save('./yTrain.npy', yTrain)

# xTrain01 = np.load('./xTrain01.npy')
# xTrainProf01 = np.load('./xTrainProf01.npy')
# xTrainRez01 = np.load('./xTrainRez01.npy')
# yTrain = np.load('./yTrain.npy')


def train(n_neurons_1=2048, n_neurons_2=2048, n_neurons_3=2048, dropout=0., lr=1.e-3, epochs=30):
    input1 = Input((xTrain01.shape[1],))
    input2 = Input((xTrainProf01.shape[1],))
    input3 = Input((xTrainRez01.shape[1],))

    x = concatenate([input1, input2, input3])
    x = Dense(n_neurons_1, activation='tanh')(x)
    x = Dense(n_neurons_2, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(n_neurons_3, activation='tanh')(x)
    x = Dense(1, activation='relu')(x)

    model = Model((input1, input2, input3), x)
    model.compile(optimizer=Adam(learning_rate=lr, amsgrad=True), loss='mse', metrics=['mae'])
    history = model.fit([xTrain01[:50000], xTrainProf01[:50000], xTrainRez01[:50000]], yTrain[:50000],
                        epochs=epochs, validation_data=([xTrain01[50000:], xTrainProf01[50000:], xTrainRez01[50000:]],
                                                    yTrain[50000:]), verbose=1, shuffle=True, batch_size=512)
    return history


# hyper_params = {'n_neurons_1': [6144, 4096], 'n_neurons_2': [6144, 4096, 1024], 'n_neurons_3': [6144, 4096, 1024],
#                 'dropout': [0., ],
#                 'lr': [2.e-3, ], 'val_mae': []}
# df = pd.DataFrame(columns=[*hyper_params])
#
# n_row = 0
# for n_neurons_1 in hyper_params['n_neurons_1']:
#     for n_neurons_2 in hyper_params['n_neurons_2']:
#         for n_neurons_3 in hyper_params['n_neurons_3']:
#             for dropout in hyper_params['dropout']:
#                 for lr in hyper_params['lr']:
#                     history = train(n_neurons_1=n_neurons_1, n_neurons_2=n_neurons_2, n_neurons_3=n_neurons_3,
#                                     dropout=dropout, lr=lr)
#                     df.loc[n_row] = [n_neurons_1, n_neurons_2, n_neurons_3,
#                                     dropout, lr,
#                                     round(min(history.history['val_mae']), 2)]
#                     print(n_neurons_1, n_neurons_2, n_neurons_3,
#                           dropout, lr,
#                           round(min(history.history['val_mae']), 2))
#                     n_row += 1
#
# df.to_csv('./hh_hyperparamsFit.csv', index=False)

history = train(n_neurons_1=512, n_neurons_2=256, n_neurons_3=256,
                dropout=0.2, lr=1.e-3, epochs=150)
plot_loss_mae(history, title_mae='MAE')

# Точность MAE = 13.4 %.