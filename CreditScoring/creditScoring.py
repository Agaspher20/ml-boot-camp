""" Тренировочная задача 'Кредитный скоринг'"""
#%%
# Загрузим библиотеки
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline

sys.path.append("ml-boot-camp\\CreditScoring")
import helperFunctions as hlp
#%%
def read_and_prepare_data(path):
    """ Функция читает данные в frame и удаляет пропуски """
    # Загрузим данные
    frame = pd.read_csv(
        path,
        ",",
        dtype={
            10: "float64",
            14: "float64"
        },
        header=None)
    
    # В колонке 1 (287 значений train, 161 значение test)
    # и 13 (145 значений train, 68 значений test) есть пропуски.
    # Пропуски обозначены символом "?".
    # # Из-за пропусков значения читаются как строки.
    # # Преобразуем значения в числа и заполним пропуски средними.
    frame[1] = hlp.fill_series_na(frame[1])
    frame[13] = hlp.fill_series_na(frame[13])
    return frame

def transform_to_bool_columns(frame, column_index, column_values, false_value=-1., init_true_value=1.):
    """ Преобразует категориальную колонку в несколько колонок, принимающих бинарные значения. """
    indices = list(frame.columns)
    next_index = indices[len(indices)-1] + 1
    true_value = init_true_value
    for col_value in column_values:
        frame[next_index] = false_value
        frame.loc[frame[column_index] == col_value, next_index] = true_value
        next_index += 1
        true_value += 1.

    return (frame.drop([column_index], axis=1), true_value)

def transform_columns_to_bool(train_frame, test_frame, indices):
    """ Преобразует список категориальных колонок в колонке, принимающие бинарные значения."""
    false_val = -1.
    true_val = 1.
    for col_index in indices:
        col_values = list(train_frame[col_index]) + list(test_frame[col_index])
        col_values = set(col_values)
        col_values.pop()
        train_frame, next_true_val_train = transform_to_bool_columns(
            train_frame,
            col_index,
            col_values,
            false_value=false_val,
            init_true_value=true_val)
            
        test_frame, next_true_val_test = transform_to_bool_columns(
            test_frame,
            col_index,
            col_values,
            false_value=false_val,
            init_true_value=true_val)
        false_val -= 1.
        true_val = next_true_val_train if next_true_val_train > next_true_val_test else next_true_val_test
    return (train_frame, test_frame)
#%%
# Загрузим тренировочные данные
train_x = read_and_prepare_data("ml-boot-camp\\Data\\crx_data_train_x.csv")
train_x.head()
#%%
train_y = pd.read_csv("ml-boot-camp\\Data\\crx_data_train_y.csv",",",header=None)
train_y.shape
#%%
hlp.frame_report(train_x)
#%%
test_x = read_and_prepare_data("ml-boot-camp\\Data\\crx_data_test_x.csv")
test_x.head()
#%%
hlp.frame_report(test_x)
#%%
# Колонки с количеством значений 15 и меньше похожи на категориальные.
# Выпишем категориальные индексы колонок
categorial_columns = [0,3,4,5,6,8,9,11,12]
train_x,test_x = transform_columns_to_bool(train_x, test_x, categorial_columns)

# Поскольку одни и те же колонки являются категориальными и числовыми,
# можно применить одни и те же трансформации, чтобы убрать пропуски в
# данных    

# На самом деле колонку с 22 значениями можно попробовать проинтерпретировать как категориальную и
# посмотреть, что получится.
#%%
train_x_y = train_x.copy()
train_x_y[15] = train_y[0]
sns.pairplot(train_x_y)
# По pairplot можно сделать следующие выводы
# 1. Не наблюдается сильно связанных признаков
# 2. Есть признаки неплохо описывающие ответ
# 3. Классы в выборке распределены равномерно
