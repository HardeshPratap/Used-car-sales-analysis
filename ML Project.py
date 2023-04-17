#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv(r"C:\Users\lenovo\Downloads\CarDataSet.csv")


# In[4]:


df


# In[5]:


df=df.drop('New_Price',axis=1)


# In[6]:


df=df.dropna()


# In[7]:


df


# In[8]:


df['Mileage']=df['Mileage'].astype(str).str.rstrip('kmpl''km/kg').astype(float)


# In[9]:


df['Mileage']


# In[10]:


df['Engine']=df['Engine'].astype(str).str.rstrip('CC').astype(float)


# In[11]:


df['Engine']


# In[12]:


df['Power']=df['Power'].astype(str).str.rstrip('bhp')


# In[13]:


df['Power']=df['Power'].str.replace('null','0')
df['Power']=df['Power'].astype(float)


# In[14]:


df['Power']


# In[15]:


df.info()


# In[16]:


df['Brand']=df['Name'].str.split(' ',expand=True)[0]


# In[17]:


df


# In[18]:


Maruti = df[df['Brand'] == 'Maruti']
Hyundai = df[df['Brand'] == 'Hyundai']
Honda = df[df['Brand'] == 'Honda']
Audi = df[df['Brand'] == 'Audi']
Nissan  = df[df['Brand'] == 'Nissan']
Toyota  = df[df['Brand'] == 'Toyota']
Volkswagen  = df[df['Brand'] == 'Volkswagen']
Tata = df[df['Brand'] == 'Tata']
Land = df[df['Brand'] == 'Land']
Mitsubishi  = df[df['Brand'] == 'Mitsubishi']
MercedesBenz  = df[df['Brand'] == 'Mercedes-Benz']
BMW = df[df['Brand'] == 'BMW']
Mahindra  = df[df['Brand'] == 'Mahindra']
Ford = df[df['Brand'] == 'Ford']
Porsche  = df[df['Brand'] == 'Porsche']
Datsun = df[df['Brand'] == 'Datsun']
Jaguar  = df[df['Brand'] == 'Jaguar']
Volvo  = df[df['Brand'] == 'Volvo ']
Chevrolet  = df[df['Brand'] == 'Chevrolet']
Skoda = df[df['Brand'] == 'Skoda']
Mini = df[df['Brand'] == 'Mini']
Fiat = df[df['Brand'] == 'Fiat']
Jeep  = df[df['Brand'] == 'Jeep']
Smart = df[df['Brand'] == 'Smart']
Ambassador  = df[df['Brand'] == 'Ambassador']
Isuzu = df[df['Brand'] == 'Isuzu']
ISUZU = df[df['Brand'] == 'ISUZU']
Force = df[df['Brand'] == 'Force']
Bentley = df[df['Brand'] == 'Bentley']
Lamborghini  = df[df['Brand'] == 'Lamborghini']


# In[19]:


plt.figure(figsize=(10,10))
plt.bar(Maruti['Mileage'],Maruti['Price'],color='red')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Maruti Mileage vs Price Analysis')


# In[20]:


for brand in df['Brand'].unique():
    plt.figure(figsize=(20,15))
    plt.bar(df[df['Brand'] == brand]['Mileage'], df[df['Brand'] == brand]['Price'], label=brand,color='orange',width=0.5)
    plt.title('Mileage vs Price analysis')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[22]:


dfnew = df[df['Brand'] != 'Lamborghini']
dfnew = dfnew[dfnew['Brand'] != 'Bentley']
dfnew = dfnew[dfnew['Brand'] != 'Porsche']


# In[23]:


x= df[['Power','Year','Mileage']].values
y= df['Price'].values.reshape(-1,1)


# In[24]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=100)


# In[29]:


regression=LinearRegression()


# In[30]:


regression.fit(xtrain,ytrain)


# In[31]:


regression.intercept_


# In[32]:


regression.coef_


# In[33]:


prediction = regression.predict(xtest)


# In[34]:


mse= mean_squared_error(ytest,prediction)


# In[35]:


mse


# In[36]:


rmse= np.sqrt(mse)


# In[37]:


rmse


# In[38]:


mae= mean_absolute_error(ytest,prediction)


# In[39]:


mae


# In[40]:


score = r2_score(ytest,prediction)


# In[41]:


score


# In[42]:


df.corr()


# In[43]:


import seaborn as sns


# In[44]:


sns.pairplot(df)


# In[45]:


plt.figure(figsize=(10,5))
sns.histplot(df['Fuel_Type'])
plt.xticks(rotation=90)
plt.show()


# In[46]:


df.corr()


# In[47]:


df['Price'].min()


# In[48]:


df['Price'].max()


# In[49]:


plt.figure(figsize=(10,5))
sns.histplot(df['Owner_Type'])
plt.xticks(rotation=90)
plt.show()


# In[50]:


plt.figure(figsize=(10,5))
sns.histplot(df['Seats'])
plt.xticks(rotation=90)
plt.show()


# In[51]:


plt.figure(figsize=(10,5))
sns.histplot(df['Location'])
plt.xticks(rotation=90)
plt.show()


# In[52]:


plt.figure(figsize=(10,5))
sns.histplot(df['Price'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:




