#%%
import sys
import random
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("ml-boot-camp\\CreditScoring")
import helperFunctions as hlp
import process_data_functions as process
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
hlp.frame_report(train_x, hide_values=True)
#%%
test_x = process.read_and_prepare_data(
    "ml-boot-camp\\Data\\crx_data_test_x.csv",
    numeric_columns,
    categorial_columns,
    lambda numerics: numerics.median(),
    lambda values: "nan")
#%%
train_x_numeric = train_x[numeric_columns]
train_x_y = train_x_numeric.copy()
y_column = max(train_x_y.columns) + 1
train_x_y[y_column] = train_y[0]
#%%
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
#%%
train_x_y[train_x_y[y_column] == 1][1]
#%%
for column1 in train_x_y.columns:
    if column1 == y_column:
        continue
    for column2 in train_x_y.columns:
        if column1 < column2:
            print("features [%i;%i]" % (column1, column2))
            plt.scatter(
                    train_x_y[train_x_y[y_column] == 1][column1],
                    train_x_y[train_x_y[y_column] == 1][column2],
                    color="red")
            plt.scatter(
                    train_x_y[train_x_y[y_column] == 0][column1],
                    train_x_y[train_x_y[y_column] == 0][column2],
                    color="blue")
            plt.show()
# По pairplot можно сделать следующие выводы
# 1. Не наблюдается коллинеарных признаков
# 2. Есть признаки неплохо описывающие ответ
# 3. Классы в выборке распределены равномерно