""" Тренировочная задача 'Кредитный скоринг'"""
#%%
# Загрузим библиотеки
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

sys.path.append("ml-boot-camp\\CreditScoring")
import helperFunctions as hlp
#%%
def readAndPrepareData(path):
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

def transformToBoolColumns(frame, column_index, false_value=-1., init_true_value=1.):
    """ Преобразует категориальную колонку в несколько колонок, принимающих бинарные значения. """
    indices = list(frame.columns)
    next_index = indices[len(indices)-1] + 1
    column_values = list(set(frame[column_index]))
    column_values.pop()
    true_value = init_true_value
    for col_value in column_values:
        frame[next_index] = false_value
        frame.loc[frame[column_index] == col_value, next_index] = true_value
        next_index +=1
        true_value += 1.

    return (frame.drop([column_index], axis=1),true_value)

def transform_columns_to_bool(frame, indices):
    """ Преобразует список категориальных колонок в колонке, принимающие бинарные значения."""
    false_val = -1.
    true_val = 1.
    for col_index in indices:
        frame,next_true_val = transformToBoolColumns(
            frame,
            col_index,
            false_value = false_val,
            init_true_value = true_val)
        false_val-=1.
        true_val=next_true_val
    return frame
#%%
# Загрузим тренировочные данные
train_x = readAndPrepareData("ml-boot-camp\\Data\\crx_data_train_x.csv")
# Выпишем категориальные индексы колонок
categorial_columns = [0,3,4,5,6,8,9,11,12]
train_x = transform_columns_to_bool(train_x, categorial_columns)
train_x.head()
#%%
hlp.frame_report(train_x)
#%%
test_x = readAndPrepareData("ml-boot-camp\\Data\\crx_data_test_x.csv")
test_x = transform_columns_to_bool(test_x, categorial_columns)
test_x.head()
#%%
hlp.frame_report(test_x)

# Колонки с количеством значений 15 и меньше похожи на категориальные.

# Поскольку одни и те же колонки являются категориальными и числовыми,
# можно применить одни и те же трансформации, чтобы убрать пропуски в
# данных    

# На самом деле колонку с 22 значениями можно попробовать проинтерпретировать как категориальную и
# посмотреть, что получится.
