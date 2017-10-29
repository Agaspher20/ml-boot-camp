""" Различные вспомогательные функции """

import pandas as pd
import numpy as np
import numbers
from matplotlib import pyplot as plt

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

def build_pairplot(
        frame,
        y_col,
        frame_columns,
        plot_columns_count=2):
    """ Bulds pair plots for given columns """
    frame_columns_count = len(frame_columns)
    if frame_columns_count == 0:
        return
    plot_rows_count = int(frame_columns_count/plot_columns_count)

    if plot_rows_count < frame_columns_count/plot_columns_count:
        plot_rows_count = plot_rows_count + 1
    if frame_columns_count < plot_columns_count:
        plot_rows_count = 1
        plot_columns_count = frame_columns_count
    fig, axes = plt.subplots(
        nrows=plot_rows_count,
        ncols=plot_columns_count,
        figsize=(10, 10))
    for idx, feature in enumerate(frame_columns):
        frame.plot(
            feature,
            y_col,
            subplots=True,
            kind="hist",
            ax=axes[int(idx / plot_columns_count), int(idx % plot_columns_count)])

def write_answer(file_name, answer):
    """ Writes result into answer file """
    with open("ml-boot-camp\\Results\\credit_scoring_" + file_name + ".csv", "w") as fout:
        for val in answer:
            fout.write(str(val)+"\n")

def pairwise_correlations(frame, threshold):
    feature_correlations = set([])
    correlations = frame.corr()
    for col in correlations.columns:
        for idx in correlations.index:
            correlation = correlations[col][idx]
            if (col != idx
                and correlation > threshold
                and not feature_correlations.intersection([(col, idx, correlation)])
                and not feature_correlations.intersection([(idx, col, correlation)])):
                feature_correlations.add((col, idx, correlation))
    return feature_correlations
