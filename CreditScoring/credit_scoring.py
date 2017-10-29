""" Тренировочная задача 'Кредитный скоринг'"""
#%%
# Загрузим библиотеки
import sys
import random
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.append("ml-boot-camp\\CreditScoring")
import helperFunctions as hlp
import process_data_functions as process
#%%
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
    hlp.write_answer(model_name, result)
    return model

def largest_category(values):
    """ Finds largest value in categorial column """
    groups = values.dropna().groupby(lambda vl: values[vl])
    return max(
        [(g[0], len(g[1])) for g in groups],
        key=lambda g: g[1])[0]
#%%
random.seed(42)
categorial_columns = [0, 3, 4, 5, 6, 8, 9, 11, 12]
numeric_columns = [1, 2, 7, 10, 13, 14]
# Загрузим тренировочные данные
train_x = process.read_and_prepare_data(
    "ml-boot-camp\\Data\\crx_data_train_x.csv",
    numeric_columns,
    categorial_columns,
    lambda numerics: numerics.median(),
    lambda values: "nan")
train_x.head()
#%%
train_y = pd.read_csv("ml-boot-camp\\Data\\crx_data_train_y.csv", ",", header=None)
train_y.shape
#%%
hlp.frame_report(train_x)
#%%
test_x = process.read_and_prepare_data(
    "ml-boot-camp\\Data\\crx_data_test_x.csv",
    numeric_columns,
    categorial_columns,
    lambda numerics: numerics.median(),
    lambda values: "nan")
test_x.head()
#%%
hlp.frame_report(test_x)
#%%
# Колонки с количеством значений 15 и меньше похожи на категориальные.
# Выпишем категориальные индексы колонок
train_x_base, test_x_base = process.transform_columns_to_bool(train_x, test_x, categorial_columns)
# Поскольку одни и те же колонки являются категориальными и числовыми,
# можно применить одни и те же трансформации, чтобы убрать пропуски в
# данных

# На самом деле колонку с 22 значениями можно попробовать проинтерпретировать как категориальную и
# посмотреть, что получится.
#%%
train_x_y = train_x.copy()
train_x_y[15] = train_y[0]
sns.pairplot(train_x_y)
#%%
y_column = max(train_x_base.columns) + 1
train_x_y = train_x_base.copy()
train_x_y[y_column] = train_y[0]
feature_columns = train_x_y.columns[:-1]
start = 0
step = 6
end = step
plot_columns = feature_columns[start:end]
while len(plot_columns) > 0:
    hlp.build_pairplot(
        train_x_y,
        y_column,
        plot_columns)
    start = end
    end = end + step
    plot_columns = feature_columns[start:end]
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
train_x_scaled, test_x_scaled = process.scale_columns(
    train_x_base,
    test_x_base,
    numeric_columns)
#%%
solve_task(
    "decisionForest_scaled",
    decision_forest_factory,
    train_x_scaled,
    train_y,
    test_x_scaled)
# Попробовать найти скореллированные колонки и удалить часть признаков
#%%
feature_correlations = hlp.pairwise_correlations(train_x_scaled, 0.3)
#%%
for col, idx, corr in sorted(feature_correlations, key=lambda x: -x[2]):
    print ("Correlation between features [%i;%i] is %f" % (col, idx, corr))
#%%
columns_to_remove = [13, 36, 11]
train_x_corr = train_x_scaled.drop(columns_to_remove, axis=1)
test_x_corr = test_x_scaled.drop(columns_to_remove, axis=1)
#%%
feature_correlations = hlp.pairwise_correlations(train_x_corr, 0.3)
#%%
solve_task(
    "decisionForest_corr",
    decision_forest_factory,
    train_x_corr,
    train_y,
    test_x_corr)
#%%
solve_task(
    "gradient_boosting_corr",
    GradientBoostingClassifier,
    train_x_corr,
    train_y,
    test_x_corr)
#%%
def gradient_boosting_optimized():
    optimizer = GradientBoostingClassifier()
    param_grid = {
        "loss": ["deviance", "exponential"],
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "n_estimators": [50, 75, 100],
        "max_depth": [5, 6, 7, 8]
    }
    estimator = GridSearchCV(optimizer, param_grid, cv=3)
    x_train = train_x_corr.as_matrix()
    y_train = train_y.as_matrix().flatten()
    estimator.fit(x_train, y_train)
    return estimator.best_estimator_
best_model = solve_task(
    "gradient_boosting_optimized_corr",
    gradient_boosting_optimized,
    train_x_corr,
    train_y,
    test_x_corr)
#%%
solve_task(
    "random_forest_corr",
    RandomForestClassifier,
    train_x_corr,
    train_y,
    test_x_corr)
#%%
def random_forest_optimized():
    optimizer = RandomForestClassifier()
    param_grid = {
        "n_estimators": [10, 20, 30, 40],
        "criterion": ["gini", "entropy"],
        "max_features": [10, 20, 30, 39],
        "max_depth": [None, 5, 10, 15, 20]
    }
    estimator = GridSearchCV(optimizer, param_grid, cv=3)
    x_train = train_x_corr.as_matrix()
    y_train = train_y.as_matrix().flatten()
    estimator.fit(x_train, y_train)
    return estimator.best_estimator_

best_rforest_model = solve_task(
    "random_forest_optimized_corr",
    random_forest_optimized,
    train_x_corr,
    train_y,
    test_x_corr)
