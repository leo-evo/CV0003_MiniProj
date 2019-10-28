import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set()
# %% codecell
# Sets dataframe of single variables.
# Inputs: dataset = variable of main datasets; var = variable name of variable to be explored.
def olympics_var(dataset, var):
    x = pd.DataFrame(dataset[var])
    f, axes = plt.subplots(1, 3, figsize=(18, 6))
    sb.boxplot(x, orient = "h", color = 'r', ax = axes[0])
    sb.distplot(x, color = 'g', ax = axes[1])
    sb.violinplot(x, color = 'b', ax = axes[2])
    print(dataset.head())
    print(type(dataset))
    print(dataset.dtypes)
    print(len(dataset[var]).unique())
    f, axes = plt.subplots(1, 1, figsize=(18, 8))
    sb.catplot(var, dataset, kind = 'Count', height = 8)
    return x
# %% codecell
# Calculates BMI from 2 single variables
def bmi(weight, height):
    bmi = pd.DataFrame(data = weight.values / (height.values * height.values), columns = ['BMI'])
    print(bmi.head(15))
    return bmi

# %% codecell
# Only for single variables. data1 is the argument for the dataset variable
## Testing code block. To be included in olympic_var().
def exploredata(data1, var1):
    dataset = data1
    var = var1
    print(dataset.head())
    print(type(dataset))
    print(dataset.dtypes)
    print(len(dataset[var]).unique())
    sb.catplot(y = var, data = dataset, kind = 'Count', height = 8)
