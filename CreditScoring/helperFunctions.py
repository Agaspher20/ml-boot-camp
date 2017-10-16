""" Различные вспомогательные функции """

import pandas as pd
import numpy as np
import numbers

def fill_series_na(series):
    """ Функция принимает на вход Pandas series преобразует значения в numeric и заполняет nan средними """
    numerics = pd.to_numeric(series, errors="coerce")
    return numerics.fillna(numerics.mean())

def print_array_by_rows(array, row_len=10):
    """ Функция выводит массив по строкам. Длина строки может быть задана в параметре """
    row = []
    for i,val in enumerate(array):
        row.append(val)
        if i>0 and (i%10) == 0:
            print ("\t\t", row)
            row = []
    if(len(row) > 0):
        print ("\t\t", row)

def list_contains_nulls(list):
    """ Функция проверяет содержит ли список пустые значения """
    contains = False
    for el in list:
        if el == "?" or (isinstance(el, numbers.Number) and np.isnan(el)):
            contains = True
            break
    return contains

def frame_report(frame):
    """ Функция выводит отчет по каждой колонке в pandas dataframe """

    print ("Количество колонок: %i" % len(frame.columns.values))
    print ("Количество строк: %i" % len(frame))
    
    for column_name in frame.columns.values:
        all_column_values = list(frame[column_name])
        column_values = set(all_column_values)

        print ("Количество значений в \"%i\" колонке: %i" % (column_name, len(column_values)))
        
        if list_contains_nulls(all_column_values):
            print ("Колонка содержит пропуски")
        else:
            print ("\tПропусков нет")
                
        print ("\tТип значений: ", type(all_column_values[0]))
        print ("\tЗначения в колонке:")
        print_array_by_rows(column_values)