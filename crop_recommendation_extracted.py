!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# # Load and visualize the Dataset

# In[2]:


df = pd.read_csv('Crop Recommendation.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.size


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df['label'].unique()


# In[9]:


df.dtypes


# In[10]:


df['label'].value_counts()


# In[11]:


print(df.dtypes)


# In[12]:


numeric_df = df.select_dtypes(include=np.number)


# In[13]:


sns.heatmap(numeric_df.corr(), annot=True)


# # Seperating features and target label

# In[14]:


features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']


# In[15]:


features


# In[16]:


target


# In[17]:


# Initialzing empty lists to append all Accuracy score and corresponding model names

acc = []
model = []


# # Train-Test Split

# In[18]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# # Decision Tree

# In[19]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)


# In[20]:


# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)
score


# Saving the trained Decision Tree Model

# In[21]:


DT_pkl_filename = 'DecisionTree.pkl'
DT_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(DecisionTree, DT_Model_pkl)
DT_Model_pkl.close()


# # Guassian Naive Bayes

# In[22]:


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x*100)


# In[23]:


# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)
score


# Saving the trained Gaussian Naive Bayes Model

# In[24]:


NB_pkl_filename = 'NBClassifier.pkl'
NB_Model_pkl = open(NB_pkl_filename, 'wb')
pickle.dump(NaiveBayes, NB_Model_pkl)
NB_Model_pkl.close()


# # Support Vector Machine

# In[25]:


from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(Xtrain)
X_train_norm = norm.transform(Xtrain)
X_test_norm = norm.transform(Xtest)
SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(X_train_norm,Ytrain)
predicted_values = SVM.predict(X_test_norm)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x*100)


# In[26]:


# Cross validation score (SVM)
score = cross_val_score(SVM,features,target,cv=5)
score


# Saving the trained SVM Model

# In[27]:


SVM_pkl_filename = 'SVMClassifier.pkl'
SVM_Model_pkl = open(SVM_pkl_filename, 'wb')
pickle.dump(SVM, SVM_Model_pkl)
SVM_Model_pkl.close()


# # Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x*100)


# In[29]:


# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv=5)
score


# Saving the trained Logistic Regression Model

# In[30]:


LR_pkl_filename = 'LogisticRegression.pkl'
LR_Model_pkl = open(LR_pkl_filename, 'wb')
pickle.dump(LogReg, LR_Model_pkl)
LR_Model_pkl.close()


# # Random Forest

# In[31]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Random Forest')
print("Random Forest's Accuracy is: ", x*100)


# In[32]:


# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score


# Saving the trained Random Forest Model

# In[33]:


RF_pkl_filename = 'RandomForest.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
RF_Model_pkl.close()


# # Accuracy Comparision

# In[34]:


plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[35]:


accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)


# # Making a Prediction

# In[38]:


data = np.array([[20,	43,	43,	25.95263,	61.89082199,	8.325235159	,99.57981207]])
prediction = RF.predict(data)
print(prediction)


# In[ ]:




