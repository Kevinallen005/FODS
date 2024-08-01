#!/usr/bin/env python
# coding: utf-8

# # Import Dataset

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv("C:/Users/Kevin Allen/Desktop/orders.csv")
df


# # Adding New Column

# In[12]:


df['tot']=df['Quantity']*df['TotalPrice']


# # First 5 rows

# In[11]:


df.head()


# # Last 5 rows

# In[13]:


df.tail()


# # Bar Plot

# In[16]:


import matplotlib.pylab as plt
plt.bar(df.OrderID,df.TotalPrice)
plt.title("TOTAL PRICES")
plt.xlabel("COST")
plt.ylabel("ORDER ID")
plt.show()


# # Histogram

# In[17]:


import matplotlib.pylab as plt
plt.hist(df.TotalPrice)
plt.title("TOTAL PRICES")
plt.ylabel("ORDER ID")
plt.show()


# # Scatter Plot

# In[18]:


import matplotlib.pylab as plt
plt.scatter(df.OrderID,df.TotalPrice)
plt.title("TOTAL PRICES")
plt.xlabel("COST")
plt.ylabel("ORDER ID")
plt.show()


# # Line plot

# In[19]:


import matplotlib.pylab as plt
plt.plot(df.OrderID,df.TotalPrice)
plt.title("TOTAL PRICES")
plt.xlabel("COST")
plt.ylabel("ORDER ID")
plt.show()


# In[20]:


import scipy.stats as stats
stats.probplot(df['TotalPrice'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Total Prices")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()


# In[22]:


mean_price = df['TotalPrice'].mean()
median_price = df['TotalPrice'].median()
mode_price = df['TotalPrice'].mode()[0]  
std_price = df['TotalPrice'].std()
variance_price = df['TotalPrice'].var()

print(f"Mean: {mean_price}")
print(f"Median: {median_price}")
print(f"Mode: {mode_price}")
print(f"Standard Deviation: {std_price}")
print(f"Variance: {variance_price}")


# In[24]:


corelation=df.corr()
print(corelation)


# In[25]:


orderid_freq = df['OrderID'].value_counts()
print("OrderID Frequency Distribution:")
print(orderid_freq)

customerid_freq = df['CustomerID'].value_counts()
print("\nCustomerID Frequency Distribution:")
print(customerid_freq)

productid_freq = df['ProductID'].value_counts()
print("\nProductID Frequency Distribution:")
print(productid_freq)


quantity_freq = df['Quantity'].value_counts()
print("\nQuantity Frequency Distribution:")
print(quantity_freq)


totalprice_freq = df['TotalPrice'].value_counts()
print("\nTotalPrice Frequency Distribution:")
print(totalprice_freq)


# In[33]:


percentile_25 = np.percentile(df['TotalPrice'], 25)
percentile_50 = np.percentile(df['TotalPrice'], 50) 
percentile_75 = np.percentile(df['TotalPrice'], 75)

print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile: {percentile_50}")
print(f"75th Percentile: {percentile_75}")


# In[31]:


import pandas as pd

# Lists or arrays
x = [1, 2, 3, 4, 5]
y = [1, 6, 7, 8, 9]
correlation_coefficient = x_series.corr(y_series)
print(correlation_coefficient)


# In[ ]:




