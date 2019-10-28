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

# %% codecell
def exploredata(var):
    x = str(var) + ".head()"
    print(x)
