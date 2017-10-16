""" Тренировочная задача 'Кредитный скоринг'"""
#%%
# Загрузим библиотеки
import sys
import pandas as pd
import numpy as np

sys.path.append("ml-boot-camp\\CreditScoring")
import helperFunctions as hlp
#%%
# Загрузим тренировочные данные
train_x = pd.read_csv(
    "ml-boot-camp\\Data\\crx_data_train_x.csv",
    ",",
    dtype={
        10: "float64", # Можно попробовать использовать 10ю колонку как категориальную
        14: "float64",
    },
    header=None)
train_x.head()
#%%
hlp.frame_report(train_x)
#%%
# В колонке 1 (287 значений) и 13 (145 значений) есть пропуски. Пропуски обозначены символом "?".
# Из-за пропусков значения читаются как строки.
# Преобразуем значения в числа и заполним пропуски средними.
train_x[1] = hlp.fill_series_na(train_x[1])
train_x[13] = hlp.fill_series_na(train_x[13])
#%%
train_x[1]
#%%
test_x = pd.read_csv(
    "ml-boot-camp\\Data\\crx_data_test_x.csv",
    ",",
    header=None)
test_x.head()
#%%
hlp.frame_report(test_x)

# Колонки с количеством значений 15 и меньше похожи на категориальные.

# На самом деле колонку с 22 значениями можно попробовать проинтерпретировать как категориальную и
# посмотреть, что получится.

# Сравнить значения в колонках train и test
