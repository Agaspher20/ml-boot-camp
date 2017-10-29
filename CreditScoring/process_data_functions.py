""" Functions to process credit scoring data """

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.append("ml-boot-camp\\CreditScoring")
import helperFunctions as hlp

def read_and_prepare_data(
    path,
    numeric_columns,
    categorial_columns,
    na_numeric_strategy,
    na_categorial_strategy):
    """ Функция читает данные в frame и удаляет пропуски """
    # Загрузим данные
    frame = pd.read_csv(
        path,
        ",",
        dtype={
            10: "float64",
            14: "float64"
        },
        na_values="?",
        header=None)

    # В колонке 1 (287 значений train, 161 значение test)
    # и 13 (145 значений train, 68 значений test) есть пропуски.
    # Пропуски обозначены символом "?".
    # # Из-за пропусков значения читаются как строки.
    # # Преобразуем значения в числа и заполним пропуски средними.
    for col in numeric_columns:
        frame[col] = frame[col].fillna(na_numeric_strategy(frame[col]))
    for col in categorial_columns:
        frame[col] = frame[col].fillna(na_categorial_strategy(frame[col]))
    return frame

def transform_to_bool_columns(frame, column_index, column_values):
    """ Преобразует категориальную колонку в несколько колонок, принимающих бинарные значения. """
    indices = list(frame.columns)
    next_index = indices[len(indices)-1] + 1
    true_value = 1
    false_value = 0
    for col_value in column_values:
        frame[next_index] = false_value
        frame.loc[frame[column_index] == col_value, next_index] = true_value
        next_index += 1

    return frame.drop([column_index], axis=1)

def transform_columns_to_bool(train_frame, test_frame, indices):
    """ Преобразует список категориальных колонок в колонке, принимающие бинарные значения."""
    train_frame_res = train_frame.copy()
    test_frame_res = test_frame.copy()
    for col_index in indices:
        col_values = list(train_frame_res[col_index]) + list(test_frame_res[col_index])
        col_values = set(col_values)
        col_values.pop()
        train_frame_res = transform_to_bool_columns(
            train_frame_res,
            col_index,
            col_values)
        test_frame_res = transform_to_bool_columns(
            test_frame_res,
            col_index,
            col_values)
    return (train_frame_res, test_frame_res)

def scale_columns(train_frame, test_frame, columns):
    """ Scales columns """
    train_frame_c = train_frame[columns].as_matrix()
    test_frame_c = test_frame[columns].as_matrix()

    scaler = StandardScaler()
    scaler.fit(train_frame_c)
    train_frame_s = scaler.transform(train_frame_c)
    test_frame_s = scaler.transform(test_frame_c)

    rest_columns = list(set(train_frame.columns.values.tolist()) - set(columns))
    train_frame_r = train_frame[rest_columns].as_matrix()
    test_frame_r = test_frame[rest_columns].as_matrix()

    train_result = np.hstack((train_frame_s, train_frame_r))
    test_result = np.hstack((test_frame_s, test_frame_r))

    return (pd.DataFrame(train_result), pd.DataFrame(test_result))
