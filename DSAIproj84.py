#!/usr/bin/env python
# coding: utf-8

# In[88]:


global np, pd, sb, plt
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set(style="ticks", color_codes = True) # set the default Seaborn style for graphics


# # DataAcquisition and Cleaning of DataSet

# In[2]:


xls_data = pd.read_excel('WHR2018Chapter2OnlineData.xls', header = None)
xls_data.head(20)
xls_data.columns = xls_data.iloc[0]
xls_data.drop(0)


# In[3]:


xls_data.describe()


# In[4]:


xls_data.info()


# In[5]:


#working_df
#working_df_agg2


# In[6]:


working_df = xls_data.copy()
working_df=working_df.drop(0)
working_df.info()


# In[7]:


working_df['Life Ladder']=working_df['Life Ladder'].astype('float64')
working_df['Log GDP per capita']=working_df['Log GDP per capita'].astype('float64')
working_df['country']=working_df['country'].astype('category')
working_df['Social support']=working_df['Social support'].astype('float64')
working_df['Healthy life expectancy at birth']=working_df['Healthy life expectancy at birth'].astype('float64')
working_df['Freedom to make life choices']=working_df['Freedom to make life choices'].astype('float64')
working_df['Generosity']=working_df['Generosity'].astype('float64')
working_df['Perceptions of corruption']=working_df['Perceptions of corruption'].astype('float64')
working_df['Positive affect']=working_df['Positive affect'].astype('float64')
working_df['Negative affect']=working_df['Negative affect'].astype('float64')
working_df['Confidence in national government']=working_df['Confidence in national government'].astype('float64')
working_df['Democratic Quality']=working_df['Democratic Quality'].astype('float64')
working_df['Delivery Quality']=working_df['Delivery Quality'].astype('float64')
working_df['Standard deviation of ladder by country-year']=working_df['Standard deviation of ladder by country-year'].astype('float64')
working_df['Standard deviation/Mean of ladder by country-year']=working_df['Standard deviation/Mean of ladder by country-year'].astype('float64')
working_df['GINI index (World Bank estimate)']=working_df['GINI index (World Bank estimate)'].astype('float64')
working_df['GINI index (World Bank estimate), average 2000-15']=working_df['GINI index (World Bank estimate), average 2000-15'].astype('float64')
working_df['gini of household income reported in Gallup, by wp5-year']=working_df['gini of household income reported in Gallup, by wp5-year'].astype('float64')


# In[8]:


working_df.info()


# In[9]:


working_df_agg2 = working_df.groupby('country').mean().reset_index()


# In[10]:


working_df_agg2.set_index('country', inplace = True)


# In[11]:


working_df_agg2.head()


# In[12]:


working_df_agg2.index


# In[13]:


data2017=xls_data[xls_data['year']==2017]
data2016=xls_data[xls_data['year']==2016]
data2015=xls_data[xls_data['year']==2015]
datayear=pd.concat([data2015,data2016,data2017])
df= datayear.sort_values(by=['country'])


# In[14]:


df = df.reset_index().drop('index', axis = 1)


# In[15]:


df.describe()


# In[16]:


df['year']=df['year'].astype('int64')
df['Life Ladder']=df['Life Ladder'].astype('float64')
df['Log GDP per capita']=df['Log GDP per capita'].astype('float64')
df['country']=df['country'].astype('category')
df['Social support']=df['Social support'].astype('float64')
df['Healthy life expectancy at birth']=df['Healthy life expectancy at birth'].astype('float64')
df['Freedom to make life choices']=df['Freedom to make life choices'].astype('float64')
df['Generosity']=df['Generosity'].astype('float64')
df['Perceptions of corruption']=df['Perceptions of corruption'].astype('float64')
df['Positive affect']=df['Positive affect'].astype('float64')
df['Negative affect']=df['Negative affect'].astype('float64')
df['Confidence in national government']=df['Confidence in national government'].astype('float64')
df['Democratic Quality']=df['Democratic Quality'].astype('float64')
df['Delivery Quality']=df['Delivery Quality'].astype('float64')
df['Standard deviation of ladder by country-year']=df['Standard deviation of ladder by country-year'].astype('float64')
df['Standard deviation/Mean of ladder by country-year']=df['Standard deviation/Mean of ladder by country-year'].astype('float64')
df['GINI index (World Bank estimate)']=df['GINI index (World Bank estimate)'].astype('float64')
df['GINI index (World Bank estimate), average 2000-15']=df['GINI index (World Bank estimate), average 2000-15'].astype('float64')
df['gini of household income reported in Gallup, by wp5-year']=df['gini of household income reported in Gallup, by wp5-year'].astype('float64')


# In[17]:


df.info()


# In[18]:


cleandata = df.merge(working_df_agg2, on='country')


# In[19]:


cleandata.head()


# In[20]:


xls_data.head()


# In[21]:


cleandata.shape


# In[22]:


cleandata.info()


# In[23]:


for i in range(426):
    for j in range(2,19):
         if np.isnan(cleandata.iloc[i,j]):
            cleandata.iloc[i,j]=(cleandata.iloc[i,j+17])
            


# In[24]:


cleandata.info()


# In[25]:


cleandata.isnull().sum()


# In[26]:


cleandata=cleandata.groupby('country').mean().reset_index()


# In[27]:


cleandata.isnull().sum()


# In[28]:


data20157=cleandata.fillna(cleandata.mean())
data2018=data20157.drop('year',axis=1)
data2018


# In[29]:


xls_score = pd.read_excel('WHR2018Chapter2OnlineData.xls', sheet_name = 'Figure2.2', header= None)


# In[30]:


xls_score.head()
xls_score.columns=xls_score.iloc[0]


# In[31]:


xls_score=xls_score[['Country','Happiness score']]


# In[32]:


xls_score=xls_score.drop(0)


# In[33]:


xls_score.head()


# In[34]:


xls_score=xls_score.rename(columns={'Country':'country'})


# In[35]:


whrdata2018=data2018.merge(xls_score, on='country')


# In[36]:


whrdata2018.head()


# In[37]:


whrdata2018.drop(whrdata2018.iloc[:, 18:35], inplace=True, axis = 1)
whrdata2018.head()


# In[91]:


whrdata2018.head()


# In[93]:


def correlation(df, response, *predictors):
    x = pd.DataFrame(df[response])
    y_list = []
    for j in predictors:
        y_list.append(j)
    y = pd.DataFrame(df[y_list])
    jointdf = pd.concat([y, x,], axis = 1, join_axes = [y.index])
    print(jointdf.head())
    sb.heatmap(jointdf.corr(), vmin = -1, vmax = 1, annot = True, fmt = ".2f")
    sb.pairplot(data = jointdf, palette = None)
    return x, y


# # Conduct a correlation matrix for all given variables

# In[39]:


whrdata2018_corr = whrdata2018.copy()


# In[40]:


whrdata2018_corr['Happiness score'] = whrdata2018_corr['Happiness score'].apply(np.float)


# In[41]:


whrdata2018_corr.dtypes


# In[42]:


f,axes = plt.subplots(1,1,figsize = (20,20))
sb.heatmap(whrdata2018_corr.corr(), vmin = -1, vmax = 1, linewidths = 1, annot = True, fmt = ".2f", annot_kws={"size":18}, cmap = "RdBu")


# #### Analysis of Correlation Matrix
#  **Top 6 highly correlated variables with Happiness**
# > **GDP Per Capita** : 0.82
# 
# > **Healthy Life Expectancy** : 0.78
# 
# > **Social Support** : 0.77
# 
# > **Delievery Quality** : 0.75
# 
# > **Democratic Quality** : 0.65
# 
# > **Freedom to make choice** : 0. 56
# 
# - for interest: we would also like to check on the variable generoisty

# In[43]:


whrdata2018_graph = whrdata2018_corr.copy()
whrdata2018_graph = whrdata2018_graph.drop(columns = "country")


# In[44]:


f, axes = plt.subplots(18, 1, figsize = (30,70))
colors = ["aqua", "azure", "b", "m", "c", "y","r", "gold", "indigo", "lavender", "lime", "maroon","navy", "orangered", "plum", "purple", "teal", "violet"]

count = 0
for var in whrdata2018_graph:
    sb.boxplot(whrdata2018_graph[var], orient = "h", color = colors[count], ax = axes[count])
    count += 1
plt.show()


# In[45]:


f, axes = plt.subplots(18, 1, figsize = (30,70))
colors = ["aqua", "azure", "b", "m", "c", "y","r", "gold", "indigo", "lavender", "lime", "maroon","navy", "orangered", "plum", "purple", "teal", "violet"]

count = 0
for var in whrdata2018_graph:
    sb.distplot(whrdata2018_graph[var], color = colors[count], ax = axes[count])
    count += 1
plt.show()


# In[46]:


f, axes = plt.subplots(18, 1, figsize = (30,70))
colors = ["aqua", "azure", "b", "m", "c", "y","r", "gold", "indigo", "lavender", "lime", "maroon","navy", "orangered", "plum", "purple", "teal", "violet"]

count = 0
for var in whrdata2018_graph:
    sb.violinplot(whrdata2018_graph[var], color = colors[count], ax = axes[count])
    count += 1
plt.show()


# In[ ]:





# # Looking for important features using Random Forest Algorithm

# In[47]:


#make a copy of data20157 which has the cleaned data set
working_df.head()


# In[48]:


impt_df = working_df.copy()


# In[49]:


impt_df.isnull().sum()


# In[50]:


impt_df.shape


# In[51]:


#for quick analysis, will drop, GINI index(world Banks estimate, and gini fo household income)
impt_df = impt_df.drop(columns =['GINI index (World Bank estimate)','gini of household income reported in Gallup, by wp5-year'])


# In[52]:


impt_df = impt_df.fillna(impt_df.mean())


# In[53]:


impt_df.isnull().sum()


# In[54]:


impt_df.head()


# #### Custom Binary Encoding for country using sklearn LabelEncoder

# In[55]:


#LabelEncoder for each country for preprocessing
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
impt_df['country_no'] = lb_make.fit_transform(impt_df['country'])


# In[56]:


impt_df.isnull().sum()


# #### Looking for important features using random forest

# In[57]:


#import train_test_split function
from sklearn.model_selection import train_test_split
x = impt_df[['year','Life Ladder', 'Log GDP per capita','Social support','Healthy life expectancy at birth','Freedom to make life choices','Generosity','Perceptions of corruption','Positive affect','Negative affect','Confidence in national government','Democratic Quality','Delivery Quality','Standard deviation of ladder by country-year','Standard deviation/Mean of ladder by country-year','GINI index (World Bank estimate), average 2000-15']]
y = impt_df['country_no'] #labels

#split dataset into training set and test set
#split 70% training and 30% testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)


# In[58]:


#Import random forest model
from sklearn.ensemble import RandomForestClassifier

#create a gaussian classifier#import train_test_split function
from sklearn.model_selection import train_test_split
x = impt_df[['year','Life Ladder', 'Log GDP per capita','Social support','Healthy life expectancy at birth','Freedom to make life choices','Generosity','Perceptions of corruption','Positive affect','Negative affect','Confidence in national government','Democratic Quality','Delivery Quality','Standard deviation of ladder by country-year','Standard deviation/Mean of ladder by country-year','GINI index (World Bank estimate), average 2000-15']]
y = impt_df['country_no'] #labels

#split dataset into training set and test set
#split 70% training and 30% testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
clf = RandomForestClassifier(n_estimators = 100)

#train th emodel using training set
clf.fit(x_train,y_train)


# In[59]:


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# In[60]:


from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[61]:


impt_df = impt_df.drop(columns = 'country')


# In[62]:


temp_list = []
for i in impt_df:
    if i != 'country_no':
        temp_list.append(i)


# In[63]:


print(len(temp_list))


# In[64]:


feature_imp = pd.Series(clf.feature_importances_, index = temp_list).sort_values(ascending = False)
feature_imp


# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')
#creating a bar plot
sb.barplot(x=feature_imp, y=feature_imp.index)
#add labels to graph
plt.xlabel("Feature Importance Score")
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.legend()
plt.show()


# #### Analysis from important features
# **Top Few Features**
# > **Log GDP**
# 
# > **Healthy life expectancy at birth**
# 
# > **Delivery Quality**
# 
# > **Democratic Quality**
# 
# > **Generosity**
# 
# > **Perception of corruption**
# 
# > **Social Support**

# #### Analysis of Correlation Matrix
#  **Top 6 highly correlated variables with Happiness**
# > **GDP Per Capita** : 0.82
# 
# > **Healthy Life Expectancy** : 0.78
# 
# > **Social Support** : 0.77
# 
# > **Delievery Quality** : 0.75
# 
# > **Democratic Quality** : 0.65
# 
# > **Freedom to make choice** : 0. 56
# 
# - for interest: we would also like to check on the variable generoisty

# # Supervised Learning Linear Regression

# In[66]:


Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
Lifeladder= pd.DataFrame(whrdata2018['Life Ladder_x'])    # Predictor


# In[67]:


# Import essential models and functions from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Recall the Happiness-Lifeladder Dataset
Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
LogGDP= pd.DataFrame(whrdata2018['Log GDP per capita_x'])    # Predictor

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(LogGDP, Happiness, test_size = 0.25)

# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()

# Predict Total values corresponding to HP
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()


# In[68]:


# Recall the Happiness-Lifeladder Dataset
Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
Freedom= pd.DataFrame(whrdata2018['Freedom to make life choices_x'])    # Predictor

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(Freedom, Happiness, test_size = 0.25)

# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()

# Predict Total values corresponding to HP
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()


# In[69]:


# Recall the Happiness-Lifeladder Dataset
Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
Support= pd.DataFrame(whrdata2018['Social support_x'])    # Predictor

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(Support, Happiness, test_size = 0.25)

# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()

# Predict Total values corresponding to HP
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()


# In[70]:


# Recall the Happiness-Lifeladder Dataset
Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
Democracy= pd.DataFrame(whrdata2018['Democratic Quality_x'])    # Predictor

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(Democracy, Happiness, test_size = 0.25)

# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()

# Predict Total values corresponding to HP
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()


# In[71]:


# Recall the Happiness-Lifeladder Dataset
Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
Corruption= pd.DataFrame(whrdata2018['Perceptions of corruption_x'])    # Predictor

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(Corruption, Happiness, test_size = 0.25)

# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()

# Predict Total values corresponding to HP
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()


# In[72]:


# Recall the Happiness-Lifeladder Dataset
Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
Healthy= pd.DataFrame(whrdata2018['Healthy life expectancy at birth_x'])    # Predictor

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(Healthy, Happiness, test_size = 0.25)

# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()

# Predict Total values corresponding to HP
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()


# In[73]:


# Recall the Happiness-Lifeladder Dataset
Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
Delivery= pd.DataFrame(whrdata2018['Delivery Quality_x'])    # Predictor

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(Delivery, Happiness, test_size = 0.25)

# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()

# Predict Total values corresponding to HP
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()


# In[74]:


# Recall the Happiness-Lifeladder Dataset
Happiness = pd.DataFrame(whrdata2018['Happiness score'])  # Response
Generosity= pd.DataFrame(whrdata2018['Generosity_x'])    # Predictor

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(Generosity, Happiness, test_size = 0.25)

# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()

# Predict Total values corresponding to HP
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()


# ### Analysis for Linear Regression Model
# - GDP per capita vs Happiness: R^2: **0.65(Train); 0.70(Test)** **Top Feature & Correlation**
# - Freedom to make life choices vs Happiness: R^2: 0.25(Train), 0.27(Test) 
# - Social support vs Happiness: R^2: **0.56(Train); 0.63(Test)** **Top Feature & Correlation**
# - Democratic Quality vs Happiness: R^2: 0.39(Train); 0.47(Test) *Top Feature & Correlation*
# - Healthy Life Expectancy vs Happiness: R^2: **0.58(Train); 0.64 (Test)** **Top Feature & Correlation**
# - Delivery vs Happiness: R^2: **0.58(Train); 0.49(Test)** **Top Feature & Correlation**
# - Generosity vs Happiness: R^2: 0.019(Train); 0.013(Test) *Top Feature only*
# - Corruption vs Happiness: R^2: 0.16(Train); 0.22(Test) *Top Feature only*
# 
# > **Good Predictors of Happiness**
# - **GDP per capita**
# - **Social Support**
# - **Healthy Life Expectancy**
# - **Delivery Quality**
# 

# #### Analysis from important features
# **Top Few Features**
# > **Log GDP**
# 
# > **Healthy life expectancy at birth**
# 
# > **Delivery Quality**
# 
# > **Democratic Quality**
# 
# > **Generosity**
# 
# > **Perception of corruption**
# 
# > **Social Support**

# #### Analysis of Correlation Matrix
#  **Top 6 highly correlated variables with Happiness**
# > **GDP Per Capita** : 0.82
# 
# > **Healthy Life Expectancy** : 0.78
# 
# > **Social Support** : 0.77
# 
# > **Delievery Quality** : 0.75
# 
# > **Democratic Quality** : 0.65
# 
# > **Freedom to make choice** : 0. 56
# 
# - for interest: we would also like to check on the variable generoisty

# # K-mean++ clustering for the selected 3 variables

# **Questions**
# 
# - How true is the linear relationship between Higher GDP and Happiness Score?
# - Which countries in particular have both high GDP and high happiness score?
# - Which countries have both low GDP and low happiness score?
# - Are there countries with low GDP and very high happiness score?
# 
# **Visualize using plotly**

# In[80]:


import chart_studio.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import cufflinks as cf
tls.set_credentials_file(username = 'rollsicy' , api_key= 'TbEA82DdG3eo7i86N0XX')


# #### KMeans++ for Log GDP per capita_x & Happiness score

# In[ ]:


from sklearn.cluster import KMeans

#extract variabels
x= pd.DataFrame(whrdata2018[['Log GDP per capita_x','Happiness score']])
init_algo = 'k-means++'

min_clust = 1
max_clust = 40

within_ss = []
for num_clust in range(min_clust, max_clust+1):
    kmeans = KMeans(n_clusters = num_clust, init = init_algo, n_init =5)
    kmeans.fit(x)
    within_ss.append(kmeans.inertia_)
f, axes = plt.subplots(1, 1, figsize=(16,4))
plt.plot(range(min_clust, max_clust+1), within_ss)
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.xticks(np.arange(min_clust, max_clust+1, 1.0))
plt.grid(which='major', axis='y')
plt.show()


# #### Clustering for GDP and Happiness

# In[ ]:


num_clust = 4

init_algo = 'k-means++'

kmeans = KMeans(n_clusters = num_clust, init = init_algo, n_init = 20)

kmeans.fit(x)
# Print the Cluster Centers
print("Features", "\tGDP", "\tHappiness")
print()

for i, center in enumerate(kmeans.cluster_centers_):
    print("Cluster", i, end=":\t")
    for coord in center:
        print(round(coord, 2), end="\t")
    print()
print()

# Print the Within Cluster Sum of Squares
print("Within Cluster Sum of Squares :", kmeans.inertia_)
print()

# Predict the Cluster Labels
labels = kmeans.predict(x)

x_labeled = x.copy()
x_labeled["Cluster"] = pd.Categorical(labels)

# Summary of the Cluster Labels
sb.countplot(x_labeled["Cluster"])


# In[ ]:


whrdata2018.head()


# In[ ]:


data = [dict(type = 'choropleth', locations = whrdata2018['country'],locationmode = 'country names',z=x_labeled['Cluster'], text = whrdata2018['country'],colorbar = {'title':'Cluster Group'},colorscale = 'Viridis')]
layout = dict(title = 'Clustering of Countries based on K-Means(GDP & Happiness)', geo = dict(showframe = False, projection = {'type':'eckert4'}))
map_1 = go.Figure(data = data, layout = layout)
py.iplot(map_1)


# In[ ]:


f,axes =plt.subplots(1,1, figsize = (16,10))
axes.set_title('K-Means Clustering')
axes.set_xlabel('Log GDP per capita_x')
axes.set_ylabel('Happiness score')
scatter = axes.scatter(x= 'Log GDP per capita_x', y='Happiness score', c='Cluster', cmap = 'viridis', data = x_labeled, s= 70)
plt.colorbar(scatter)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(16,8))
sb.boxplot(x = 'Log GDP per capita_x', y = 'Cluster', data = x_labeled, ax = axes)


# In[ ]:


whrdata2018.head()


# In[ ]:


whrdata2018['GDP Cluster']=x_labeled['Cluster']


# In[ ]:


whrdata2018['GDP Cluster'].unique()


# In[ ]:


whrdata2018[whrdata2018['GDP Cluster'] == 3]


# In[ ]:


whrdata2018[whrdata2018['GDP Cluster'] == 0].describe()


# In[ ]:


whrdata2018[whrdata2018['GDP Cluster'] == 3]


# In[ ]:


whrdata2018[whrdata2018['GDP Cluster'] == 3].describe()


# #### Analysis of Data after Computing the 4 Clusters
# >**Group 0** Countries such as United Sates, Singapore are in the higher spectrum of the GDP and Happiness
# >**Group 3** Countries such as Malaysia, 

# In[ ]:


whrdata2018.head()


# # Detecting for Anomaly

# In[ ]:


#Extracting only GDP and Happiness
anom = pd.DataFrame(whrdata2018[['Log GDP per capita_x','Happiness score']])

f, axes = plt.subplots(1,1, figsize = (16,8))
plt.scatter( x = 'Log GDP per capita_x', y = 'Happiness score', data = anom)


# In[ ]:


# Using anomly detection
from sklearn.neighbors import LocalOutlierFactor

num_neighbors = 20
cont_fraction = 0.01

lof = LocalOutlierFactor(n_neighbors = num_neighbors, contamination = cont_fraction)

lof.fit(anom)


# In[ ]:


#Predicting the anomalies
labels = lof.fit_predict(anom)

anom_labeled = anom.copy()
anom_labeled['Anomaly'] = pd.Categorical(labels)

sb.countplot(anom_labeled['Anomaly'])


# In[ ]:


f, axes = plt.subplots(1,1, figsize=(16,8))
plt.scatter(x= "Log GDP per capita_x", y = "Happiness score", c= 'Anomaly', cmap = 'winter', data = anom_labeled)


# In[ ]:


whrdata2018['Anomaly GDP Happiness'] = anom_labeled['Anomaly']


# In[ ]:


whrdata2018[whrdata2018['Anomaly GDP Happiness'] == -1]


# In[ ]:


# Working with Finance (BSE Ticker)
# Finding movement between GDP and BSE closing price


# In[ ]:


#Next objective is to extract relevant data from Botswana; look for year unique as well
# next extract data from finance market ( lookk out for the tiker from Botswana)
# do a comapre plot between the two countries 
finance_df = xls_data.copy()


# In[ ]:


#Extracting only Botswana 
bots_finance_df = finance_df[finance_df['country'] == 'Botswana']


# In[ ]:


bots_finance_df.head(15)


# In[ ]:


bots_finance_gdp = pd.DataFrame(bots_finance_df['Log GDP per capita'])


# In[ ]:


bots_finance_gdp.head()


# In[ ]:


# Extracting stock ticker of botswana from website
import datetime as dt
from matplotlib import style
import pandas_datareader.data as web
style.use('ggplot')


# In[ ]:


start = dt.datetime(2010, 1,4)
end = dt.datetime(2017, 12, 31)
BSE_ticker = web.DataReader('BSE', 'yahoo', start, end)
BSE_ticker.to_csv('BSE.csv')


# In[ ]:


BSE_df = pd.read_csv('BSE.csv', parse_dates = True, index_col = 0)


# In[ ]:


# now that we have the extracted data
# plot 2 different scatters
BSE_df['Close'].plot()


# In[ ]:


BSE_df_date = BSE_df.reset_index(inplace = False)


# In[ ]:


BSE_df_date.head(15)


# In[ ]:


BSE_df_date['year'],BSE_df_date['month'] = BSE_df_date['Date'].dt.year, BSE_df_date['Date'].dt.month


# In[ ]:


BSE_df_date.head()


# In[ ]:


BSE_df_date_backup = BSE_df_date.copy()


# In[ ]:


BSE_df_date_backup = BSE_df_date_backup.groupby('year').mean()


# In[ ]:


BSE_df_date_backup.head()


# In[ ]:


bots_finance_df_backup = bots_finance_df.copy()


# In[ ]:


# take note only 2 import dfs BSE_df_date_backup & bots_finance_df_backup
bots_finance_df_backup = bots_finance_df_backup.drop(index = [168, 169])


# In[ ]:


bots_finance_df_backup = bots_finance_df_backup.iloc[:,:4]


# In[ ]:


bots_finance_df_backup = bots_finance_df_backup.drop('Life Ladder', axis = 1)


# In[ ]:


bots_finance_df_backup.head()


# In[ ]:


BSE_df_date_backup.head(1)


# In[ ]:


BSE_df_date_backup.info()


# In[ ]:


bots_finance_df_backup.info()


# In[ ]:


bots_finance_df_backup = bots_finance_df_backup.reset_index()


# In[ ]:


bots_finance_df_backup.head()


# In[ ]:


BSE_df_date_backup = BSE_df_date_backup.reset_index()


# In[ ]:


BSE_df_date_backup.head(1)


# In[ ]:


bots_finance_df_backup['year'] = bots_finance_df_backup['year'].apply(np.int)


# In[ ]:


result_df = bots_finance_df_backup.merge(BSE_df_date_backup, on = 'year')


# In[ ]:


result_df.head()


# In[ ]:


result_df_2 = result_df.copy()


# In[ ]:


#plotting line chart
sb.set()
DIMS = (20,8)
ax = result_df.plot(x='year', y = 'Log GDP per capita', figsize = DIMS, legend = False)
ax2 = ax.twinx()
result_df.plot(x='year', y='Close', ax=ax2, legend= False,color = 'g')
ax.figure.legend()
ax.set_ylabel('Log GDP per capita')
ax2.set_ylabel('Average Closing price')
ax.set_xlabel('Date')
plt.title('Log GDP per capita vs Average BSE closing price')
plt.show()


# In[ ]:


close_df = pd.DataFrame(result_df['Close'])
GDP_df = pd.DataFrame(result_df['Log GDP per capita'])


# In[ ]:


result_df['Log GDP per capita'] = result_df['Log GDP per capita'].apply(np.float)


# In[ ]:


result_df_2 = result_df[['Log GDP per capita', 'Close']]


# In[ ]:


sb.heatmap(result_df_2.corr(), vmin = -1, vmax =1 , annot = True, fmt = ".2f", color = "skyblue")


# In[ ]:


#Extracting only Healthy life expectancy at birth and Happiness
anom1 = pd.DataFrame(whrdata2018[['country','Happiness score','Healthy life expectancy at birth_x']])

f, axes = plt.subplots(1,1, figsize = (16,8))
plt.scatter( x = 'Healthy life expectancy at birth_x', y = 'Happiness score', data = anom1)


# In[ ]:


#Printing botswana from another variable

anom_labeled1 = anom1.copy()
anom_labeled1['botswana']=pd.Categorical(anom_labeled1.country=='Botswana')
anom_labeled1.head()


# In[ ]:


f, axes = plt.subplots(1,1, figsize=(16,8))
plt.scatter(x= "Healthy life expectancy at birth_x", y = "Happiness score", c= 'botswana', cmap = 'winter', data = anom_labeled1)


# In[ ]:


#Extracting only Social Support and Happiness
anom2 = pd.DataFrame(whrdata2018[['country','Happiness score','Social support_x']])

f, axes = plt.subplots(1,1, figsize = (16,8))
plt.scatter( x = 'Social support_x', y = 'Happiness score', data = anom2)


# In[ ]:


#Printing botswana from another variable

anom_labeled2 = anom2.copy()
anom_labeled2['botswana']=pd.Categorical(anom_labeled2.country=='Botswana')
anom_labeled2.head()


# In[ ]:


f, axes = plt.subplots(1,1, figsize=(16,8))
plt.scatter(x= "Social support_x", y = "Happiness score", c= 'botswana', cmap = 'winter', data = anom_labeled2)


# In[ ]:


#Extracting only Delivery Quality and Happiness
anom3 = pd.DataFrame(whrdata2018[['country','Happiness score','Delivery Quality_x']])

f, axes = plt.subplots(1,1, figsize = (16,8))
plt.scatter( x = 'Delivery Quality_x', y = 'Happiness score', data = anom3)


# In[ ]:


#Printing botswana from another variable

anom_labeled3 = anom3.copy()
anom_labeled3['botswana']=pd.Categorical(anom_labeled3.country=='Botswana')
anom_labeled3.head()


# In[ ]:


f, axes = plt.subplots(1,1, figsize=(16,8))
plt.scatter(x= "Delivery Quality_x", y = "Happiness score", c= 'botswana', cmap = 'winter', data = anom_labeled3)


# In[ ]:


whrdatacompare=whrdata2018[whrdata2018["Log GDP per capita_x"].between(9.5,9.8,inclusive=True)]


# In[ ]:


whrdatacompare.shape


# In[ ]:


whrdatacompare.head(20)


# In[ ]:


whrdatacomparebots=whrdatacompare[whrdatacompare['country']=='Botswana']


# In[ ]:


whrdatacomparebots.head()


# In[ ]:


whrdatacomparebots = whrdatacomparebots.drop(columns = 'country')


# In[ ]:


bots_series = whrdatacomparebots.loc[16]


# In[ ]:


bots_series.head(20)


# In[ ]:


whrdatacompare.drop(16,inplace=True)


# In[ ]:


whrdatacompare_mean = whrdatacompare.mean()


# In[ ]:


whrdatacompare_mean.head(20)


# In[ ]:


bots_list1=[]
for i in bots_series:
    bots_list1.append(i)
bots_list1.pop(19) 
bots_list1.pop(18)
bots_list1.pop(17)
print(bots_list1)    


# In[ ]:


bots_list2 = []
for i in whrdatacompare_mean:
    bots_list2.append(i)    
print(bots_list2)


# In[ ]:


botratio=[]
for i in range(17):
    botratio.append(abs(bots_list1[i]/bots_list2[i]))
print(botratio)    


# In[ ]:


index = ['Life Ladder','Log GDP per capita','Social support',
         'Freedom to make life choices', 'Healthy life expectancy at birth',
         'Generosity','Perceptions of corruption','Positive affect',
         'Negative affect','Confidence in national government',
         'Democratic Quality', 'Delivery Quality',
         'Standard deviation of ladder by country-year',
         'Standard deviation/Mean of ladder by country-year','GINI index (World Bank estimate)',
         'GINI index (World Bank estimate), average 2000-15','gini of household income reported in Gallup, by wp5-year']


# In[ ]:


dfcompare=pd.DataFrame({'Ratio of Botswana against Mean':botratio},index=index)


# In[ ]:


dfcompare.head()


# In[ ]:


# Create a color if the group is what i want to highlight
my_color=np.where(np.logical_or(dfcompare.index=='Standard deviation/Mean of ladder by country-year',dfcompare.index =='GINI index (World Bank estimate), average 2000-15'), 'orange', 'darkblue')
my_size=np.where(np.logical_or(dfcompare.index=='Standard deviation/Mean of ladder by country-year',dfcompare.index =='GINI index (World Bank estimate), average 2000-15') , 70, 30)

plt.hlines(y=dfcompare.index, xmin=0, xmax=dfcompare['Ratio of Botswana against Mean'], color=my_color, alpha=0.4)
plt.scatter(dfcompare['Ratio of Botswana against Mean'], dfcompare.index, color=my_color, s=my_size, alpha=1)
plt.title("Comparison of Botswana against Mean of countries with similar Log GDP")
plt.xlabel('Ratio of Botswana against Mean')
plt.ylabel('Variables of Happiness')


# Difference between GINI index of botswana and other countries with similar Log GDP is extremely high alongside standard deviation/mean of ladder. Income Inequality is the main factor as to why Botswana has low Happiness score compared to other countries with similar Log GDP.

# In[ ]:


data = [dict(type = 'choropleth', locations = whrdata2018['country'],locationmode = 'country names',z=whrdata2018['Happiness score'], text = whrdata2018['country'],colorbar = {'title':'Cluster Group'}, colorscale = 'Viridis')]
layout = dict(title = 'Happiness Index 2018', geo = dict(showframe = False, projection = {'type':'eckert4'}))
map_1 = go.Figure(data = data, layout = layout)
py.iplot(map_1)


# In[ ]:




