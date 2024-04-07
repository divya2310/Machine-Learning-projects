#!/usr/bin/env python
# coding: utf-8

# ## Laptop Price Predictor

# In[1]:


import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir("C:\\Users\Dell\\Downloads\\Machine Learning projects")


# In[4]:


os.getcwd()


# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv("laptop_data.csv")


# In[7]:


df.head()


# In[8]:


df.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)


# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


df.describe()


# In[14]:


df.duplicated().sum()## to check whether our data has duplicacy or not


# In[15]:


df.drop_duplicates(inplace = True) ## to delete duplicate values


# In[16]:


df.duplicated().sum()


# In[17]:


df['Ram'] = df['Ram'].str.replace('GB','').astype(int)


# In[18]:


df.head()


# In[19]:


df['Weight'] = df['Weight'].str.replace('kg','')


# In[20]:


df.head()


# In[21]:


df.dtypes


# In[22]:


df['Weight']= df['Weight'].astype(float)


# In[23]:


df['Weight'] = df['Weight'].astype(int)


# In[24]:


df.dtypes


# In[25]:


df.head()


# ###  EDA - Analysing Data

# In[26]:


sns.displot(df['Price'])  ## as we can see from the graph that our data is not normally distributed it is right skewed
plt.show()


# In[27]:


df['Company'].value_counts().plot(kind = 'bar')
plt.show()


# In[28]:


plt.figure(figsize=(20,20))
sns.barplot(x = df['Company'], y = df['Price'])
plt.show()


# In[29]:


df['TypeName'].value_counts().plot(kind = 'bar')
plt.show()


# In[30]:


plt.figure(figsize=(10,10))
sns.barplot(x = df['TypeName'], y = df['Price'])
plt.show()


# In[31]:


sns.scatterplot(x = df['Inches'], y = df['Price'])
plt.show()


# In[32]:


df['ScreenResolution'].value_counts()


# In[33]:


## feature engineering
df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False)

## i used str.contains function to check whether the given filed has tochscreen or not 


# In[34]:


# Convert boolean values to 'Yes' and 'No'
df['Touchscreen'] = df['Touchscreen'].map({True: 'Yes', False: 'No'})


# In[35]:


df.head()


# In[36]:


df['Touchscreen'] = df['Touchscreen'].map({'Yes': 1, 'No': 0})


# In[37]:


df.head()


# In[38]:


sns.barplot(x= df['Touchscreen'], y = df['Price'])
plt.show()


# In[39]:


df['IPS'] = df['ScreenResolution'].str.contains('IPS Panel', case=False)


# In[40]:


df['IPS'] = df['IPS'].map({True: 'Yes', False:'No'})


# In[41]:


df['IPS'] = df['IPS'].map({'Yes': 1, 'No':0})


# In[42]:


df.sample()


# In[43]:


sns.barplot(x= df['IPS'], y = df['Price'])
plt.show()


# In[44]:


df[['X_resolution', 'Y_resolution']] = df['ScreenResolution'].str.split('x', expand=True)


# In[45]:


df.drop(columns = ['X_resolution', 'Y_resolution'], inplace = True, axis = 1)


# In[46]:


df.head()


# In[47]:


df[['X_resolution', 'Y_resolution']] = df['ScreenResolution'].str.split('x', expand=True)


# In[48]:


df.head()


# In[49]:


df['X_resolution'] = df['X_resolution'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[50]:


df.head()


# In[51]:


df.dtypes


# In[52]:


df['X_resolution'] = df['X_resolution'].astype(int)


# In[53]:


df['Y_resolution'] = df['Y_resolution'].astype(int)


# In[54]:


df.dtypes


# In[55]:


## we calculate PPI which given as :
df['PPI'] = (df['X_resolution'] ** 2 + df['Y_resolution'] ** 2) ** 0.5 / df['Inches']


# In[56]:


df.head()


# In[57]:


df['PPI'] = df['PPI'].astype(float)


# In[58]:


df.head()


# In[59]:


df.corr()


# In[60]:


## so as we can see the corre of price woth ppi , x , y resolution and inches are strong so instead of all these we canm use ppi


# In[61]:


df.drop(columns = ['ScreenResolution'], axis = 1, inplace = True)


# In[62]:


df.head()


# In[63]:


df.drop(columns = ['X_resolution', 'Y_resolution','Inches'], axis = 1, inplace = True)


# In[64]:


df.head()


# In[65]:


df['Cpu'].value_counts()


# In[66]:


df['Generation'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))


# In[67]:


df['Generation']


# In[68]:


df.head()


# In[69]:


def fetch_processor(text):
    if text == 'Intel Core i5' or text == 'Intel Core i7' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD processor'


# In[70]:


df['Cpu Brand'] = df['Generation'].apply(fetch_processor)


# In[71]:


df.sample()


# In[72]:


sns.barplot(x = df['Cpu Brand'], y = df['Price'])


# In[73]:


df.drop(columns = ['Cpu', 'Generation'], inplace = True)


# In[74]:


df.head()


# In[75]:


df['Memory'].value_counts()


# In[76]:


df['Memory'] = df['Memory'].astype(str).replace('\.0' , '', regex= True)
df['Memory'] = df['Memory'].str.replace('GB' , '')
df['Memory'] = df['Memory'].str.replace('TB' , '000')
new = df['Memory'].str.split("+" , n=1 , expand = True)

df['first'] = new[0]
df['first'] = df['first'].str.strip()

df['second'] = new[1]


df["Layer1HDD"] = df['first'].apply(lambda x : 1 if "HDD" in x else 0)
df["Layer1SSD"] = df['first'].apply(lambda x : 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df['first'].apply(lambda x : 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df['first'].apply(lambda x : 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')
df['second'].fillna("0" , inplace = True)

df["Layer2HDD"] = df['second'].apply(lambda x : 1 if "HDD" in x else 0)
df["Layer2SSD"] = df['second'].apply(lambda x : 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df['second'].apply(lambda x : 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df['second'].apply(lambda x : 1 if "Flash Storage" in x else 0)


df['second'] = df['second'].str.replace(r'\D', '')

df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)


df['HDD'] = (df['first']*df["Layer1HDD"]+df['second']*df["Layer2HDD"])
df['SSD'] = (df['first']*df["Layer1SSD"]+df['second']*df["Layer2SSD"])
df['Hybrid'] = (df['first']*df["Layer1Hybrid"]+df['second']*df["Layer2Hybrid"])
df['Flash Storage'] = (df['first']*df["Layer1Flash_Storage"]+df['second']*df["Layer2Flash_Storage"])


# In[77]:


df.drop(columns = ['Flash_Storage'],inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.sample()


# In[ ]:


df.drop(columns = ['Memory'], inplace = True)


# In[ ]:


df.drop(columns = ['Hybrid', 'Flash Storage'], inplace = True)


# In[ ]:


df.head()


# In[ ]:


df['Brand Name'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[ ]:


df.head()


# In[ ]:


df['Brand Name'].value_counts()


# In[ ]:


df = df[df['Brand Name']!= 'ARM']


# In[ ]:


df.drop(columns = ['Gpu'], inplace = True)


# In[ ]:


df.head()


# In[ ]:


df['OpSys'].value_counts()


# In[ ]:


def fetch_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[ ]:


df['OS'] = df['OpSys'].apply(fetch_os)


# In[ ]:


df.head()


# In[ ]:


df.drop(columns = ['OpSys'],inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


## we will convert skwede into normal transformation by using log normal


# In[ ]:


## Train test split
X = df.drop(columns = ['Price'])


# In[ ]:


y = np.log(df['Price'])


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 2)


# In[ ]:


## as we know ml models does not perform well with categorical data so we will do one hot encoding using pipelines and will 
## convert them into numerical values


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# In[ ]:


pip install xgboost


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score


# In[ ]:


## Linear Regression


# In[ ]:


step1 = ColumnTransformer = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse = False, drop='first'),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Ridge regresion

# In[ ]:


step1 = ColumnTransformer = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse = False, drop='first'),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = Ridge(alpha = 10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Lasso Regreesion

# In[ ]:


step1 = ColumnTransformer = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse = False, drop='first'),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = Lasso(alpha = 0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## KNN

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

step1 = ColumnTransformer = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse = False, drop='first'),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = KNeighborsRegressor(n_neighbors= 5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## Decision Tree

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

step1 = ColumnTransformer = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse = False, drop='first'),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = DecisionTreeRegressor(max_depth= 10, max_leaf_nodes=20,min_samples_split=2)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

step1 = ColumnTransformer = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse = False, drop='first'),[0,1,7,10,11])
],remainder = 'passthrough')

step2 = RandomForestRegressor(n_estimators= 100, random_state=10,max_samples=0.25,
                             max_features = 0.005, max_depth= 20)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Voting Regressor, stacking Regressor

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor, StackingRegressor

step1 = ColumnTransformer = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse = False, drop='first'),[0,1,7,10,11])
],remainder = 'passthrough')


rf = RandomForestRegressor(n_estimators= 100, random_state=10,max_samples=0.25,max_features = 0.005, max_depth= 20)
gbdt = GradientBoostingRegressor(n_estimators=100, max_features=0.5)
xgb = XGBRegressor(n_estimators=23, learning_rate= 0.3, max_depth= 5)
et = ExtraTreesRegressor(n_estimators= 100, random_state=10,max_features = 0.005, max_depth= 20)

step2 = VotingRegressor([('rf',rf),('gbdt',gbdt),('xgb',xgb),('et',et)],weights=[5,1,1,1])

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# #### We will create website

# In[ ]:


import pickle


# In[ ]:


pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:


import sklearn
print(sklearn.__version__)


# In[ ]:


pip install --upgrade scikit-learn


# In[79]:


import sklearn
print(sklearn.__version__)


# In[82]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




