""" Тренировочная задача 'Кредитный скоринг'"""
#%%
# Загрузим библиотеки
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

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

def write_answer(file_name, answer):
    """ Writes result into answer file """
    with open("ml-boot-camp\\Results\\credit_scoring_" + file_name + ".csv", "w") as fout:
        for val in answer:
            fout.write(str(val)+"\n")

def solve_task(model_name, model_factory, x_train_frame, y_train_frame, x_test_frame):
    """ Solves task with model, given via model_factory """
    x_train = x_train_frame.as_matrix()
    y_train = y_train_frame.as_matrix().flatten()
    scorer_model = model_factory()
    scorer = cross_val_score(scorer_model, x_train, y_train, cv=3)
    print(scorer.mean())
    model = model_factory()
    x_test = x_test_frame.as_matrix()
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    print(result)
    write_answer(model_name, result)

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
#%%
# Загрузим тренировочные данные
train_x = read_and_prepare_data("ml-boot-camp\\Data\\crx_data_train_x.csv")
train_x.head()
#%%
train_y = pd.read_csv("ml-boot-camp\\Data\\crx_data_train_y.csv", ",", header=None)
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
categorial_columns = [0, 3, 4, 5, 6, 8, 9, 11, 12]
train_x_base, test_x_base = transform_columns_to_bool(train_x, test_x, categorial_columns)
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
# 1. Не наблюдается коллинеарных признаков
# 2. Есть признаки неплохо описывающие ответ
# 3. Классы в выборке распределены равномерно
#%%
# После препроцессинга попробуем обучить логистическую регрессию
solve_task(
    "baseLine",
    LogisticRegression,
    train_x_base,
    train_y,
    test_x_base)
#%%
solve_task(
    "l1Reg",
    lambda: LogisticRegression(penalty='l1'),
    train_x_base,
    train_y,
    test_x_base)
#%%
solve_task(
    "decisionTree",
    DecisionTreeClassifier,
    train_x_base,
    train_y,
    test_x_base)
#%%
def decision_forest_factory():
    """ Factory build decision forest based on BaggingClassifier """
    tree_classifier = DecisionTreeClassifier()
    trees_count = 100
    return BaggingClassifier(base_estimator=tree_classifier, n_estimators=trees_count)
solve_task(
    "decisionForest",
    decision_forest_factory,
    train_x_base,
    train_y,
    test_x_base)
#%%
def grid_search_logistic_factory(x_train, y_train):
    """ regularization strength grid search """
    optimizer = LogisticRegression("l2")
    param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
    estimator = GridSearchCV(optimizer, param_grid, cv=3)
    estimator.fit(x_train, y_train)
    return estimator.best_estimator_

solve_task(
    "gridSearchLogisticRegression",
    lambda: grid_search_logistic_factory(train_x_base.as_matrix(), train_y.as_matrix().flatten()),
    train_x_base,
    train_y,
    test_x_base)

#%%
numeric_columns = list(set(train_x.columns.values.tolist()) - set(categorial_columns))
train_x_scaled, test_x_scaled = scale_columns(
    train_x_base,
    test_x_base,
    numeric_columns)

#%%
solve_task(
    "baseLineScaled",
    LogisticRegression,
    train_x_scaled,
    train_y,
    test_x_scaled)
# Попробовать найти скореллированные колонки и удалить часть признаков
