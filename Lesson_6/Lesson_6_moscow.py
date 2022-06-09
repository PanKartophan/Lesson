import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.random import set_seed

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import random
import statistics
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


# PRO (Вариаент 2 и 3).
#######################################################################################################################


df = pd.read_csv('./moscow.csv', sep=";")
df = df.iloc[::2, :]
data = df.values


# Вычисляем количество комнат
# maxRoomCount - максимальное число комнат в квартире
def getRoomsCount(d, maxRoomCount):
    roomsCountStr = d[0]  # Получаем строку с числом комнат
    roomsCount = 0
    try:
        roomsCount = int(roomsCountStr)  # Пробуем превратить строку в число
        if roomsCount > maxRoomCount:
            roomsCount = maxRoomCount  # Если число комнат больше максимального, то присваиваем максимальное
    except:  # Если не получается превратить строку в число
        if roomsCountStr == roomsCountStr:  # Проверяем строку на nan (сравнение с самим собой)
            if "Ст" in roomsCountStr:  # Еcть строка = "Ст", значит это Студия
                roomsCount = maxRoomCount + 1

    return roomsCount


# Превращаем число комнат в категорию
def getRoomsCountCategory(d, maxRoomCount):
    roomsCount = getRoomsCount(d, maxRoomCount)  # Получаем число комнат
    roomsCount = utils.to_categorical(roomsCount, maxRoomCount + 2)  # Превращаем в категорию
    # maxRoomCount+2 потому что 0 зарезервирован на неопознаное число комнат, а maxRoomCount+1 на "Студию"
    return roomsCount


# Получаем индекс станции метро
# allMetroNames - все уникальные названия метро в базе
def getMetro(d, allMetroNames):
    metroStr = d[1]  # Получаем строку метро
    metro = 0
    if metroStr in allMetroNames:  # Если находим метро во всех названиях
        metro = allMetroNames.index(metroStr) + 1  # Присваиваем индекс
        # +1 так как 0 зарезервирован на неопознанное метро
    return metro


# Получаем тип метро
# 0 - внутри кольца
# 1 - кольцо
# 2 - 1-3 станции от конца
# 3 - 4-8 станций от кольца
# 4 - больше 8 станций от кольца
def getMetroType(d):
    metroTypeStr = d[1]  # Получаем строку метро
    metroTypeClasses = 5  # Число классов метро
    metroType = metroTypeClasses - 1  # Изначально считаем последний класс
    # Метро внутри кольца
    metroNamesInsideCircle = ["Площадь Революции", "Арбатская", "Смоленская", "Красные Ворота", "Чистые пруды",
                              "Лубянка", "Охотный Ряд", "Библиотека имени Ленина", "Кропоткинская", "Сухаревская",
                              "Тургеневская", "Китай-город", "Третьяковская", "Трубная", "Сретенский бульвар",
                              "Цветной бульвар", "Чеховская", "Боровицкая", "Полянка", "Маяковская", "Тверская",
                              "Театральная", "Новокузнецкая", "Пушкинская", "Кузнецкий Мост", "Китай-город",
                              "Александровский сад"]
    # Метро на кольце
    metroNamesCircle = ["Киевская", "Парк Культуры", "Октябрьская", "Добрынинская", "Павелецкая", "Таганская",
                        "Курская", "Комсомольская", "Проспект Мира", "Новослободская", "Белорусская",
                        "Краснопресненская"]
    # Метро 1-3 станции от кольца
    metroNames13FromCircle = ["Бауманская", "Электрозаводская", "Семёновская", "Площадь Ильича", "Авиамоторная",
                              "Шоссе Энтузиастов", "Римская", "Крестьянская Застава", "Дубровка", "Пролетарская",
                              "Волгоградский проспект", "Текстильщики", "Автозаводская", "Технопарк", "Коломенская",
                              "Тульская", "Нагатинская", "Нагорная", "Шаболовская", "Ленинский проспект",
                              "Академическая", "Фрунзенская", "Спортивная", "Воробьёвы горы", "Студенческая",
                              "Кутузовская", "Фили", "Парк Победы", "Выставочная", "Международная", "Улица 1905 года",
                              "Беговая", "Полежаевская", "Динамо", "Аэропорт", "Сокол", "Деловой центр", "Шелепиха",
                              "Хорошёвская", "ЦСКА", "Петровский парк", "Савёловская", "Дмитровская", "Тимирязевская",
                              "Достоевская", "Марьина Роща", "Бутырская", "Фонвизинская", "Рижская", "Алексеевская",
                              "ВДНХ", "Красносельская", "Сокольники", "Преображенская площадь"]
    # Метро 4-8 станций от кольа
    metroNames48FromCircle = ["Партизанская", "Измайловская", "Первомайская", "Щёлковская", "Новокосино", "Новогиреево",
                              "Перово", "Кузьминки", "Рязанский проспект", "Выхино", "Лермонтовский проспект",
                              "Жулебино", "Партизанская", "Измайловская", "Первомайская", "Щёлковская", "Новокосино",
                              "Новогиреево", "Перово", "Кузьминки", "Рязанский проспект", "Выхино",
                              "Лермонтовский проспект", "Жулебино", "Улица Дмитриевского", "Кожуховская", "Печатники",
                              "Волжская", "Люблино", "Братиславская", "Коломенская", "Каширская", "Кантемировская",
                              "Царицыно", "Орехово", "Севастопольская", "Чертановская", "Южная", "Пражская",
                              "Варшавская", "Профсоюзная", "Новые Черёмушки", "Калужская", "Беляево", "Коньково",
                              "Университет", "Багратионовская", "Филёвский парк", "Пионерская", "Кунцевская",
                              "Молодёжная", "Октябрьское Поле", "Щукинская", "Спартак", "Тушинская", "Сходненская",
                              "Войковская", "Водный стадион", "Речной вокзал", "Беломорская", "Ховрино",
                              "Петровско-Разумовская", "Владыкино", "Отрадное", "Бибирево", "Алтуфьево", "Фонвизинская",
                              "Окружная", "Верхние Лихоборы", "Селигерская", "ВДНХ", "Ботанический сад", "Свиблово",
                              "Бабушкинская", "Медведково", "Преображенская площадь", "Черкизовская",
                              "Бульвар Рокоссовского"]
    # Проверяем, в какую категорию попадает наша станция
    if metroTypeStr in metroNamesInsideCircle:
        metroType = 0
    if metroTypeStr in metroNamesCircle:
        metroType = 1
    if metroTypeStr in metroNames13FromCircle:
        metroType = 2
    if metroTypeStr in metroNames48FromCircle:
        metroType = 3

    # Превращаем результат в категорию
    metroType = utils.to_categorical(metroType, metroTypeClasses)
    return metroType


# Вычисляем растояние до метро
def getMetroDistance(d):
    metroDistanceStr = d[2]  # Получаем строку
    metroDistance = 0  # Расстояние до метро
    metroDistanceType = 0  # Тип расстояния - пешком или на транспорте

    # ЕСли строка не равна nan
    if metroDistanceStr == metroDistanceStr:
        if len(metroDistanceStr) > 0:
            # Определяем тип расстояния
            if metroDistanceStr[-1] == "п":
                metroDistanceType = 1  # Пешком
            elif metroDistanceStr[-1] == "т":
                metroDistanceType = 2  # На транспорте

            # Выбрасываем последний символ, чтобы осталось только число
            metroDistanceStr = metroDistanceStr[:-1]
            try:
                # Разделяем дистанции на категории
                metroDistance = int(metroDistanceStr)
                if metroDistance < 3:
                    metroDistance = 1
                elif metroDistance < 6:
                    metroDistance = 2
                elif metroDistance < 10:
                    metroDistance = 3
                elif metroDistance < 15:
                    metroDistance = 4
                elif metroDistance < 20:
                    metroDistance = 5
                else:
                    metroDistance = 6
            except:  # Если в строке не число, то категория 0
                metroDistance = 0

    # Число классов дистанции
    metroDistanceClasses = 7
    # У нас 7 категорий дистанции по расстоянию
    # И 3 типа дистанции - неопознанный, пешком и транспортом
    # Мы создадим вектор длины 3*7 = 21
    # Будем преобразовывать индекс расстояния 0-6 в 0-20
    # Для типа "Пешком" - ничего не меняем
    if metroDistanceType == 2:
        metroDistance += metroDistanceClasses  # Для типа "Транспортом" добавляем 7
    if metroDistanceType == 0:
        metroDistance += 2 * metroDistanceClasses  # Для неопознанного типа добавляем 14

    # Превращаем в категории
    metroDistance = utils.to_categorical(metroDistance, 3 * metroDistanceClasses)
    return metroDistance


# Получаем набор из 4 данных
# - этаж квартиры
# - этажность дома
# - индикатор, что последний этаж
# - тип дома
def getHouseTypeAndFloor(d):
    try:
        houseStr = d[3]  # Получаем строку типа дома и этажей
    except:
        houseStr = ""

    houseType = 0  # Тип дома
    floor = 0  # Этаж квартиры
    floors = 0  # Этажность дома
    isLastFloor = 0  # Индикатор последнего этажа

    # Проверяем строку на nan
    if houseStr == houseStr:
        if len(houseStr) > 1:

            try:
                slashIndex = houseStr.index("/")  # Ищем разделитель /
            except:
                print(houseStr)

            try:
                spaceIndex = houseStr.index(" ")  # Ищем разделитель " "
            except:
                print(houseStr)

            # Вытаскиваем строки
            floorStr = houseStr[:slashIndex]  # Строка этажа
            floorsStr = houseStr[slashIndex + 1:spaceIndex]  # Строка этажнгости дома
            houseTypeStr = houseStr[spaceIndex + 1:]  # Строка типа дома

            # Выбираем категорию этажа
            try:
                floor = int(floorStr)  # Превращаем строку в число
                floorSave = floor
                if floorSave < 5:
                    floor = 2
                if floorSave < 10:
                    floor = 3
                if floorSave < 20:
                    floor = 4
                if floorSave >= 20:
                    floor = 5
                if floorSave == 1:  # Первый этаж выделяем в отдельную категорию
                    floor = 1

                if floor == floors:  # Если этаж последний, включаем индикатор последнего этажа
                    isLastFloor = 1
            except:
                floor = 0  # Если строка не парсится в число, то категория этажа = 0 (отдельная)

            # Выбираем категорию этажности дома
            try:
                floors = int(floorsStr)  # Превращаем строку в число
                floorsSave = floors
                if floorsSave < 5:
                    floors = 1
                if floorsSave < 10:
                    floors = 2
                if floorsSave < 20:
                    floors = 3
                if floorsSave >= 20:
                    floors = 4
            except:
                floors = 0  # Если строка не парсится в число, то категория этажности = 0 (отдельная)

            # Определяем категорию типа дома
            if len(houseTypeStr) > 0:
                if "М" in houseTypeStr:
                    houseType = 1
                if "К" in houseTypeStr:
                    houseType = 2
                if "П" in houseTypeStr:
                    houseType = 3
                if "Б" in houseTypeStr:
                    houseType = 4
                if "?" in houseTypeStr:
                    houseType = 5
                if "-" in houseTypeStr:
                    houseType = 6

        # Превращаем все категории в one hot encoding
        floor = utils.to_categorical(floor, 6)
        floors = utils.to_categorical(floors, 5)
        houseType = utils.to_categorical(houseType, 7)

    return floor, floors, isLastFloor, houseType


# Вычисляем тип балкона
def getBalcony(d):
    balconyStr = d[4]  # Полуаем строку
    # Выписываем все варианты балконов в базе
    balconyVariants = ['Л', 'Б', '2Б', '-', '2Б2Л', 'БЛ', '3Б', '2Л', 'Эрк', 'Б2Л', 'ЭркЛ', '3Л', '4Л', '*Л', '*Б']
    # Проверяем на nan
    if balconyStr == balconyStr:
        balcony = balconyVariants.index(balconyStr) + 1  # Находим индекс строки балкона во всех строках
    else:
        balcony = 0  # Индекс 0 выделяем на строку nan

    # Превращаем в one hot encoding
    balcony = utils.to_categorical(balcony, 16)

    return balcony


# Определяем тип санузла
def getWC(d):
    wcStr = d[5]  # Получаем строку
    # Выписываем все варианты санузлов в базе
    wcVariants = ['2', 'Р', 'С', '-', '2С', '+', '4Р', '2Р', '3С', '4С', '4', '3', '3Р']
    # Проверяем на nan
    if wcStr == wcStr:
        wc = wcVariants.index(wcStr) + 1  # Находим индекс строки санузла во всех строках
    else:
        wc = 0  # Индекс 0 выделяем на строку nan

    # Превращаем в one hot encoding
    wc = utils.to_categorical(wc, 14)

    return wc


# Определяем площадь
def getArea(d):
    areaStr = d[6]  # Поулачем строку площади

    if "/" in areaStr:
        slashIndex = areaStr.index("/")  # Находим разделитель /
        try:
            area = float(areaStr[:slashIndex])  # Берём число до разделителя и превращаем в число
        except:
            area = 0  # Если не получается, возвращаем 0
    else:
        area = 0  # Или если нет разделителя, возвращаем 0

    return area


# Полуаем цену
def getCost(d):
    costStr = d[7]  # Загружаем строку

    try:
        cost = float(costStr)  # Пробуем превратить в число
    except:
        cost = 0  # Если не получается, возвращаем 0

    return cost


# Получаем комментарий
def getComment(d):
    commentStr = d[-1]  # Возвращаем данные из последней колонки

    return commentStr


# Объединяем все числовые параметры вместе
def getAllParameters(d, allMetroNames):
    # Загружаем все данные по отдельности
    roomsCountType = getRoomsCountCategory(d, 30)
    metro = getMetro(d, allMetroNames)
    metroType = getMetroType(d)
    metroDistance = getMetroDistance(d)
    floor, floors, isLastFloor, houseType = getHouseTypeAndFloor(d)
    balcony = getBalcony(d)
    wc = getWC(d)
    area = getArea(d)

    # Объединяем в один лист
    out = list(roomsCountType)
    out.append(metro)
    out.extend(metroType)
    out.extend(metroDistance)
    out.extend(floor)
    out.extend(floors)
    out.append(isLastFloor)
    out.extend(houseType)
    out.extend(balcony)
    out.extend(wc)
    out.append(area)

    return out


# Генерируем обучающаюу выборку - xTrain
def getXTrain(data):
    # Получаем строку во всеми вариантами метро
    allMertroNames = list(df["Метро / ЖД станции"].unique())

    # Всевращаем все строки в data1 в векторы параметров и записываем в xTrain
    xTrain = [getAllParameters(d, allMertroNames) for d in data]
    xTrain = np.array(xTrain)

    return xTrain


# Генерируем обучающую выборку - yTrain
def getYTrain(data):
    # Зашружаем лист всех цен квартир по всем строкам data1
    costList = [getCost(d) for d in data]
    yTrain = np.array(costList)

    return yTrain


###########################
# Очистка текста и превращение в набор слов
##########################
def text2Words(text):
    # Удаляем лишние символы
    text = text.replace(".", "")  # удаляем лишние символы
    text = text.replace("—", "")
    text = text.replace(",", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace("…", "")
    text = text.lower()  # Переводим в нижний регистр

    words = []  # Тут будут все слов
    currWord = ""  # Тут будет накапливаться текущее слово, между двумя пробелами

    # идём по всем символам
    for symbol in text:

        if (symbol != "\ufeff"):  # Игнорируем системынй символ в начале строки
            if (symbol != " "):  # Если символ не пробел
                currWord += symbol  # То добавляем вимвол в текущее слово
            else:  # Если символ пробел
                if (currWord != ""):
                    words.append(currWord)  # Добавляем тккущее слово в список слов
                    currWord = ""  # И обнуляем текущее слово

    # Добавляем финальное слово, если оно не пустое
    # Если не сделать, то потеряем финальное слово, потому что текст чаще всего заканчивается на не пробел
    if (currWord != ""):
        words.append(currWord)

    return words


###########################
# Создание словаря - все слова, упорядоченные по частоте появления
##########################
def createVocabulary(allWords):
    # Создаём словарь, в котором будут слова и количество их поялвений во всём текста
    # Ключи - все наши слова
    # Количество появлений пока везде 0
    wCount = dict.fromkeys(allWords, 0)

    # Проходим по всем словам
    for word in allWords:
        wCount[word] += 1  # И увеличиаем количество появлений текущего слова на 1

    # Выцепляем лист из словаря
    wordsList = list(wCount.items())
    # И сортируем по частоте появления
    wordsList.sort(key=lambda i: i[1], reverse=1)
    # key = lambda i:i[1] - говорит, что сортировать надо по частоте появления
    # В i[0] у нас слово, в i[1] - частота появления
    # reverse=1 говорить сортироваться по убыванию

    sortedWords = []  # Тут будет лист всех отсортированных слов

    # Проходим по всем словам в отсортированном списке
    for word in wordsList:
        sortedWords.append(word[0])  # Докидываем слово в лист отсортированных слов

    # Это словарь слово - индекс
    # Изначально заполнен всеми словами
    # У всех индекс 0
    wordIndexes = dict.fromkeys(allWords, 0)
    # Проходим по всем словам
    for word in wordIndexes.keys():
        wordIndexes[word] = sortedWords.index(word) + 1  # Ставим индекс = индекс слова в отсортированном листе слов + 1
        # +1 потому, что индекс 0 резервируем под неопознанные слова

    return wordIndexes


###########################
# Преобразования листа слов в лист индексов
##########################
def words2Indexes(words, vocabulary, maxWordsCount):
    wordsIndexes = []

    # Идём по всем словая
    for word in words:

        wordIndex = 0  # Тут будет индекс слова, изначально 0 - слово неопознано
        wordInVocabulary = word in vocabulary  # Проверяем, есть ли слово в словаре

        # Если слово есть в словаре
        if (wordInVocabulary):
            index = vocabulary[word]  # Индекс = индексу слова в словаре
            if (index < maxWordsCount):  # Если индекс ниже maxWordsCount - черты отсечения слов
                wordIndex = index  # То записываем индекс
            # Иначе останется значение 0

        wordsIndexes.append(wordIndex)

    return wordsIndexes


###########################
# Преобразование одного короткого вектора в вектор из 0 и 1
# По принципу words bag
##########################
def changeXTo01(trainVector, wordsCount):
    # Создаём вектор длины wordsCount с нулями
    out = np.zeros(wordsCount)

    # Идём по всем индексам в строке
    for x in trainVector:
        out[x] = 1  # В позицию нужного индекса ставим 1

    return out


###########################
# Преобразование выборки (обучающей или проверочной) к виду 0 и 1
# По принципу words bag
##########################
def changeSetTo01(trainSet, wordsCount):
    out = []

    # Проходим по всем векторам в наборе
    for x in trainSet:
        out.append(
            changeXTo01(x, wordsCount))  # Добавляем в итоговый набор текущий вектор, преобразованный в bag of words

    return np.array(out)


###########################
# Формируем обучающую выборку из примечаний к квартирам
# Пока в виде слов
##########################
def getXTrainComments(data):
    xTrainComments = []  # Тут будет обучающся выборка
    allTextComments = ""  # Тут будуте все тексты вместе для словаря

    # Идём по всем строкам квартир в базе
    for d in data:
        currText = getComment(d)  # Вытаскиваем примечание к квартире
        try:
            if (currText == currText):  # Проверяем на nan
                allTextComments += currText + " "  # Добавляем текст в общий текст для словаря
        except:
            currText = "Нет комментария"  # Если не получается, то делаем стандартный текст "Нет комментария"
        xTrainComments.append(currText)  # Добавляем примечание новой строкой в обучающую выборку

    xTrainComments = np.array(xTrainComments)

    return (xTrainComments, allTextComments)


###########################
# Формируем обучающую выборку из примечаний к квартирам
# Теперь в виде индексов
##########################
def changeSetToIndexes(xTrainComments, vocabulary, maxWordsCount):
    xTrainCommentsIndexes = []  # Тут будет итоговый xTrain примечаний в виде индексов

    # Идём по всем текстам
    for text in xTrainComments:
        currWords = text2Words(text)  # Разбиваем текст на слова
        currIndexes = words2Indexes(currWords, vocabulary, maxWordsCount)  # Превращаем в лист индексов
        currIndexes = np.array(currIndexes)
        xTrainCommentsIndexes.append(currIndexes)  # Добавляем в xTrain

    xTrainCommentsIndexes = np.array(xTrainCommentsIndexes)
    xTrainCommentsIndexes = changeSetTo01(xTrainCommentsIndexes, maxWordsCount)  # Превращаем в формат bag of words
    return xTrainCommentsIndexes


###########################
# Формируем обучающую выборку из примечаний к квартирам
# Теперь в виде индексов
# И с приведением к стандартной длине всех векторов - cropLen
##########################
def changeSetToIndexesCrop(xTrainComments, vocabulary, maxWordsCount, cropLen):
    xTrainCommentsIndexes = []  # Тут будет итоговый xTrain примечаний в виде индексов

    # Идём по всем текстам
    for text in xTrainComments:
        currWords = text2Words(text)  # Разбиваем текст на слова
        currIndexes = words2Indexes(currWords, vocabulary, maxWordsCount)  # Превращаем в лист индексов
        currIndexes = np.array(currIndexes)
        xTrainCommentsIndexes.append(currIndexes)  # Добавляем в xTrain

    xTrainCommentsIndexes = np.array(xTrainCommentsIndexes)
    xTrainCommentsIndexes = pad_sequences(xTrainCommentsIndexes,
                                          maxlen=cropLen)  # Приводим все вектора к стандартной длине
    return xTrainCommentsIndexes


oneRoomMask = [getRoomsCount(d, 30) == 1 for d in data]
data1 = data[oneRoomMask] #В data1 оставляем только однокомнатные квартиры
xTrain = getXTrain(data1)
yTrain = getYTrain(data1)
print(data.shape)
print(data1.shape)
print(xTrain.shape)
xScaler = StandardScaler() #Создаём нормировщик нормальным распределением
xScaler.fit(xTrain[:, -1].reshape(-1, 1)) #Обучаем его на площадях квартир (последня колонка в xTrain)
xTrainScaled = xTrain.copy()
xTrainScaled[:, -1] = xScaler.transform(xTrain[:, -1].reshape(-1, 1)).flatten() #Нормируем данные нормировщиком

xTrainC, allTextComments = getXTrainComments(data1) #Создаём обучающую выборку по текстам и большо текст для словаря
allWords = text2Words(allTextComments) #Собираем полный текст в слова
allWords = allWords[::10] #Берём 10% слов (иначе словарь слишком долго формируется)
vocabulary = createVocabulary(allWords) #Создаём словарь
xTrainC01 = changeSetToIndexes(xTrainC, vocabulary, 2000) #Преобразеум xTrain в bag of words
print(xTrain.shape)
print(xTrainC01.shape)
print(yTrain.shape)
splitVal = 0.2
valMask = np.random.sample(xTrainScaled.shape[0]) < splitVal

yScaler = StandardScaler() #Делаемнормальный нормировщик
yScaler.fit(yTrain.reshape(-1, 1)) #Обучаем на ценах квартир
yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)) #Нормируем цены квартир
#
# np.save('./xTrainScaled_moscow.npy', xTrainScaled)
# np.save('./xTrainC01_moscow.npy', xTrainC01)
# np.save('./yTrain_moscow.npy', yTrain)
# np.save('./yTrainScaled_moscow.npy', yTrainScaled)

# xTrainScaled = np.load('./xTrainScaled_moscow.npy')
# xTrainC01 = np.load('./xTrainC01_moscow.npy')
# yTrain = np.load('./yTrain_moscow.npy')
# yTrainScaled = np.load('./yTrainScaled_moscow.npy')
# yTrain = yTrain / 1.e7

splitVal = 0.2
valMask = np.random.sample(xTrainScaled.shape[0]) < splitVal


def train(n_neurons_1=1024, n_neurons_2=128, lr=1.e-3, epochs=10, batch_size=128):
    input1 = Input((xTrainScaled.shape[1],))
    input2 = Input((xTrainC01.shape[1],))

    x = concatenate([input1, input2])
    x = Dense(n_neurons_1, activation='tanh')(x)
    x = Dense(n_neurons_2, activation='tanh')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    model = Model((input1, input2), x)
    model.compile(optimizer=Adam(learning_rate=lr, amsgrad=True), loss='mse', metrics=['mae'])

    mae_train = []
    mae_val = []
    def callback_plt_mae_fn(epoch, logs):
        pred_train = model.predict([xTrainScaled[~valMask], xTrainC01[~valMask]], verbose=0)
        pred_val = model.predict([xTrainScaled[valMask], xTrainC01[valMask]], verbose=0)
        mae_train.append(round(sum(abs(pred_train.flatten() - yTrain[~valMask])) / sum(~valMask), 4))
        mae_val.append(round(sum(abs(pred_val.flatten() - yTrain[valMask])) / sum(valMask), 4))
        print("Эпоха:", epoch, "MAE:", round(sum(abs(pred_val.flatten() - yTrain[valMask])) / sum(valMask), 4))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
        ax1.scatter(yTrain[valMask], pred_val)
        ax1.set_xlabel('Правильные значение')
        ax1.set_ylabel('Предсказания')
        ax1.set_xlim(yTrain[valMask].min(), yTrain[valMask].max())
        ax1.set_ylim(yTrain[valMask].min(), yTrain[valMask].max())
        ax1.grid(True)
        ax1.plot([-1000, 1000], [-1000, 1000])

        ax2.hist(pred_val.flatten() - yTrain[valMask], bins=np.linspace(-3., 3., 201))
        ax2.set_xlim(-3, 3)
        ax2.grid(True)
        plt.show()

        if epoch == epochs - 1:
            plt.plot(mae_train)
            plt.plot(mae_val)
            plt.title('MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Val'], loc='upper right')
            plt.grid(True)
            plt.show()

    times = []
    cum_time = [0., ]
    def callback_begin_fn(epoch, logs):
        times.append([time.time(), 0.])

    def callback_fin_fn(epoch, logs):
        times[-1][1] = time.time()
        epoch_duration = times[-1][1] - times[-1][0]
        cum_time[0] += epoch_duration
        print('Epoch takes', round(epoch_duration, 1), 'sec.', 'Training takes', round(cum_time[0], 1), 'sec.',
              round((epochs - epoch - 1) * epoch_duration, 1), 'sec remain.')

    def callback_lr_fn(epoch, logs):
        if epoch >= 30:
            if statistics.stdev(mae_val[-5:]) / statistics.mean(mae_val[-5:]) > 0.2:
                model.optimizer.learning_rate.set_value(lr / 5.)
            if statistics.stdev(mae_val[-5:]) / statistics.mean(mae_val[-5:]) < 0.02:
                model.optimizer.learning_rate.set_value(lr * 5.)

    def callback_save_fn(epoch, logs):
        if epoch > 0:
            if mae_val[-1] < mae_val[-2]:
                model.save_weights('./weights_moscow.hdf5', overwrite=True)

    callback_plt_mae = LambdaCallback(on_epoch_end=callback_plt_mae_fn)
    callback_time = LambdaCallback(on_epoch_begin=callback_begin_fn, on_epoch_end=callback_fin_fn)
    callback_lr = LambdaCallback(on_epoch_end=callback_lr_fn)
    callback_save = LambdaCallback(on_epoch_end=callback_save_fn)

    history = model.fit([xTrainScaled[~valMask], xTrainC01[~valMask]], yTrain[~valMask],
                        epochs=epochs, validation_data=([xTrainScaled[valMask], xTrainC01[valMask]], yTrain[valMask]),
                        verbose=0, shuffle=True, batch_size=batch_size,
                        callbacks=[callback_plt_mae, callback_time, callback_lr, callback_save])
    return history


history = train(n_neurons_1=512, n_neurons_2=256, lr=1.e-3, batch_size=512, epochs=200)
plot_loss_mae(history, title_mae='MAE')

# # Точность MAE = 0.0800 * 1E7.



