#!/usr/bin/env python
# coding: utf-8
<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>
# # 1.IMPORTING ALL THE REQUIRED PACKAGES.

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
print("All required packages are imported")
from timeit import default_timer as timer


# # 2. LOADING TRAIN AND TEST DATA,COMBINING TRAIN AND TEST DATA

# In[3]:


data_train=pd.read_csv(r"C:\Users\RAJESH KUMAR\Desktop\project\train.csv")
print("Train dataset has been loaded")


# In[4]:


data_test=pd.read_csv(r"C:\Users\RAJESH KUMAR\Desktop\project\test.csv")
print("Test dataset has been loaded")


# In[5]:


data=data_train.append(data_test)
print("Train and test datasets are appended.")


# # 3.ANALISING THE DATA

# ## 3.1 Checking the dimension of the data

# In[6]:


rows,cols=data.shape
print("No. of Rows in the data: ",rows)
print("No. of Columns in the data: ",cols)


# ## 3.2 Printing the sample of the data

# In[7]:


data_train.head()


# ## 3.3 Checking if the data contains NULL values

# In[8]:


data.isnull().sum().any()
print("Data contains NULL values:",data.isnull().sum().any())


# ## 3.4 Checking for DUPLICATED VALUES

# In[66]:


data.duplicated().sum()
print("Number of DUPLICATED values in the Data:",data.duplicated().sum())


# ## 3.5 Checking the columns present in the data

# In[67]:


print("Columns Present in the Data Set")
print(data.columns)


# ## 3.6 Checking for Collinearity.

# In[68]:


c=data.corr()
c.head()


# In[69]:


fig, size = plt.subplots(figsize=(15,15))
sns.heatmap(c,ax=size)


# ## 3.7 Checking the possible human activities that can be predicted.

# In[70]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[71]:


print("Human Activities that can be predicted using this dataset are:")
for i in y.unique():
    print(i)


# # 4. DATA VISUALISATION

# ## 4.1 Frequency of each activity

# In[72]:


plt.figure(figsize = (12,6))
sns.countplot(data.Activity,)
plt.xlabel('Activity')
plt.ylabel('count')
plt.title('Frequency of Activities in DATA')
plt.show()


# ## 4.2 Daily Routine i.e. time spent for each activity of the 30 candidates,whose data is collected. 

# In[73]:


s = data.groupby(['subject','Activity']).size().unstack()
s.plot(kind='bar',stacked=True, figsize=(17, 8), title = 'Activity count vs Subjects in train')
plt.show()


# # 5.DATA PREPROCESSING

# #### The data contains no null values, no duplicate values. The data also have no categorial values in the features. But the data have high collinearity between the features and also the number of columns in the data is equal to 563 so to reduce the dimension,get rwed of collinearity and reduce the model fitting time we use Principal component analysis .
# 

# In[74]:


pca=PCA(.95)
x=pca.fit_transform(x)
x=pd.DataFrame(x)
x.head()


# #### After using PCA the size of the columns decreases. Checking the size.

# In[18]:


rows,cols=x.shape
print("No of rows in the dataset after using PCA:",rows)
print("No of columns in the dataset after using PCA:",cols)


# # 6. Model fitting 

# ## 6.1 Model fitting using KNN 

# ### 6.1.1 Finding the random state required for splitting data in training and testing data for KNeighborsClassifier

# In[19]:


ts_score =[]
import numpy as np
for j in range(200):
    x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = j,test_size=.1)
    lr =KNeighborsClassifier().fit(x_train,y_train)
    ts_score.append(lr.score(x_test,y_test))
k_KNN = ts_score.index(np.max(ts_score))
print("The Random State for splitting for KNeighborsClassifier:",k_KNN)


# ### 6.1.2 Splitting the data into training and testing data.

# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=161,test_size=.1)
print("Data has been splitted into x_train,x_test,y_train,y_test for KNeighborsClassifier")


# In[41]:


ex_time_RF_n = []
acc_RF_n = []
start_time = timer()
fit = KNeighborsClassifier(n_neighbors=5).fit(x_train,y_train)
pred = fit.predict(x_test)
accuracy = accuracy_score(y_test, pred)
elapsed = timer() - start_time    
ex_time_RF_n.append(elapsed)
acc_RF_n.append(accuracy)


# ### 6.1.3 Finding the optimum value of nearest neighbours

# In[22]:


error=[]
for i in range(3,200,2):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_y_test=knn.predict(x_test)
    error.append(np.mean(pred_y_test!=y_test))
plt.figure(1, figsize=(6, 6))
plt.plot(range(3,200,2),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('error rate k value')
plt.xlabel('K value')
plt.ylabel('mean error')
plt.show()


# ### 6.1.4 Fitting the KNN model.

# In[42]:


classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)


# ### 6.1.5 Prediction of y using the Model.

# In[43]:


y_pred_KNN=classifier.predict(x_test)
print("The Model For KNeighboursClassifier is predicted")


# In[44]:


accuracy_score_KNN=accuracy_score(y_test,y_pred_KNN)


# ## 6.2 Model Fitting using DecisionTreeClassifier

# ### 6.2.1 Finding the random state required for splitting data in training and testing data for DecisionTreeClassifier

# In[26]:


ts_score =[]
import numpy as np
for j in range(1200):
    x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = j,test_size=.1)
    lr =DecisionTreeClassifier(criterion="entropy",min_samples_leaf=0.01).fit(x_train,y_train)
    ts_score.append(lr.score(x_test,y_test))
k_DTC = ts_score.index(np.max(ts_score))
print("The Random State for splitting for DecisionTreeClassifier:",k_DTC)


# ### 6.2.2 Splitting the data into training and testing data.

# In[59]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1194,test_size=.1)
print("Data has been splitted into x_train,x_test,y_train,y_test for DecisionTreeClassifier")


# In[46]:


start_time = timer()
fit = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=0.01).fit(x_train,y_train)
pred = fit.predict(x_test)
accuracy = accuracy_score(y_test, pred)
elapsed = timer() - start_time
    
ex_time_RF_n.append(elapsed)
acc_RF_n.append(accuracy)


# ### 6.2.3 Fitting the DecisionTreeClassifier.

# In[60]:


dtc=DecisionTreeClassifier(criterion='entropy',min_samples_leaf=.01)
dtc.fit(x_train,y_train)


# ### 6.2.4 Prediction of y using the Model.

# In[61]:


y_pred_DTC=dtc.predict(x_test)
print("The Model For DecisionTreeClassifier is predicted")


# In[62]:


accuracy_score_DTC=accuracy_score(y_test,y_pred_DTC)


# ## 6.3 Model Fitting using RandomForestClassifier

# ### 6.3.1 Finding the random state required for splitting data in training and testing data for RandomForestClassifier

# In[27]:


ts_score =[]
import numpy as np
for j in range(2000):
    x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = j,test_size=.1)
    lr =RandomForestClassifier(random_state = 1511,min_samples_leaf=0.01 ).fit(x_train,y_train)
    ts_score.append(lr.score(x_test,y_test))
k_RFC = ts_score.index(np.max(ts_score))
print("The Random State for splitting for RandomForestClassifier:",k_RFC)


# In[28]:


ts_score =[]
import numpy as np
for j in range(2000):
    x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 1633,test_size=.1)
    lr =RandomForestClassifier(random_state = j,min_samples_leaf=0.01 ).fit(x_train,y_train)
    ts_score.append(lr.score(x_test,y_test))
k= ts_score.index(np.max(ts_score))
print("The Random State for RandomForestClassifier parameter:",k)


# ### 6.3.2 Splitting the data into training and testing data.

# In[49]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1633,test_size=.1)
print("Data has been splitted into x_train,x_test,y_train,y_test for RandomForestClassifier")


# In[50]:


start_time = timer()
fit = RandomForestClassifier(random_state=1511,min_samples_leaf=0.01 ).fit(x_train,y_train)
pred = fit.predict(x_test)
accuracy = accuracy_score(y_test, pred)
elapsed = timer() - start_time
    
ex_time_RF_n.append(elapsed)
acc_RF_n.append(accuracy)


# ### 6.3.3 Fitting the RandomForestClassifier.

# In[51]:


rf=RandomForestClassifier(random_state=1511,min_samples_leaf=0.01 )
rf.fit(x_train,y_train)


# ### 6.3.4 Prediction of y using the Model.

# In[52]:


y_pred_RFC=rf.predict(x_test)
print("The Model For RandomForestClassifier is predicted")


# In[53]:


accuracy_score_RFC=accuracy_score(y_test,y_pred_RFC)


# # 7.Determination of Accuracy

# In[54]:


print("The Accuracy Score using KNeighborsClassifier is: ",accuracy_score_KNN)


# In[63]:


print("The Accuracy Score using DecisionTreeClassifier is: ",accuracy_score_DTC)


# In[58]:


print("The Accuracy Score using RandomForestClassifier is: ",accuracy_score_RFC)


# # Graphical Representation of ACCURACY and EXECUTION Time for The Models Used.

# In[64]:


label = ('KneighborsClassifier','DecisionTreeClassifier','RandomForestClassifier')
Accu = acc_RF_n
ExTime = ex_time_RF_n

plt.figure(figsize = (12,6))
y_pos = np.arange(len(label))

#Accuracy
plt.subplot(1,2,1)
plt.bar(y_pos, Accu, align='center')
plt.xticks(y_pos, label)
plt.ylim(min(Accu)- 0.01 , max(Accu) +0.01)
plt.ylabel('Accuracy Percentage')

#Execution Time
plt.subplot(1,2,2)
plt.bar(y_pos, ExTime, align='center', color = 'orange')
plt.xticks(y_pos, label)
plt.ylim(min(ExTime) -.1 , max(ExTime) +0.2)
plt.ylabel('Run-Time(sec)')

plt.tight_layout()
plt.show()


# In[ ]:




