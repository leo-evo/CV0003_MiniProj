# %% codecell
# xfu0008@ntu.edu.sg
# Fu Xian Xu
# %% codecell
# class dataframe(object):
#     def __init__(self, name, var):
#         self.name = name
#         self.var = var
#     def dataset(name):
#         x = pd.read_csv(self.name + '.csv')
#         return x
#     def variable(var):
#         y = pd.DataFrame(x[self.var])
#         return y

# %% codecell
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set()
# %% codecell
def olympics_var(var):
    x = pd.DataFrame(abalone_data[var])
    f, axes = plt.subplots(1, 3, figsize=(18, 6))
    sb.boxplot(x, orient = "h", color = 'r', ax = axes[0])
    sb.distplot(x, color = 'g', ax = axes[1])
    sb.violinplot(x, color = 'b', ax = axes[2])
    return x
# %% codecell
def bmi(weight, height):
    bmi = pd.DataFrame(data = weight.values / (height.values * height.values), columns = ['BMI'])
    print(bmi.head(15))
    return bmi
