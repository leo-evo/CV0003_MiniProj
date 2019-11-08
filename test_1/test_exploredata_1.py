# %% codecell
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set()
# %% codecell
olympicsdata = pd.read_csv('athlete_events.csv')
olympicsdata.head()

# %% codecell
print("Data type : ", type(olympicsdata))
print("Data dims : ", olympicsdata.shape)
# %% codecell
print(olympicsdata.dtypes)
# %% codecell
olympicsdata.info()
# %% codecell
# Age types in the Dataset
print("Number of Age Types :", len(olympicsdata["Age"].unique()))

# Athletes of each Age Type
print(olympicsdata["Age"].value_counts())
sb.catplot(y = "Age", data = olympicsdata, kind = "count", height = 8)
# %% codecell
# Height types in the Dataset
print("Number of Height Types :", len(olympicsdata["Height"].unique()))

# Athletes of each Height Type
print(olympicsdata["Height"].value_counts())
sb.catplot(y = "Height", data = olympicsdata, kind = "count", height = 8)
# %% codecell
# Season types in the Dataset
print("Number of Medal Types :", len(olympicsdata["Medal"].unique()))

# Athletes of each Season Type
print(olympicsdata["Medal"].value_counts())
sb.catplot(y = "Medal", data = olympicsdata, kind = "count", height = 8)
