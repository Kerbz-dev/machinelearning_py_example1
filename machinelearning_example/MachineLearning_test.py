#!/usr/bin/env python
# coding: utf-8

# In[29]:


#prediction based on user input

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x, y)
genderinput1 = input("What is your gender (M)ale or (F)emale? ")
ageinput1 = int(input("How old are you? "))
genderinput2 = input("What is your gender (M)ale or (F)emale? ")
ageinput2 = int(input("How old are you? "))

if genderinput1 == "M":
    genderinput1 = 1
elif genderinput1 == "F":
    genderinput1 = 0
else: 
    print("incorrect input, please select (M)ale or (F)emale")
    
if genderinput2 == "M":
    genderinput2 = 1
elif genderinput2 == "F":
    genderinput2 = 0
else: 
    print("incorrect input, please select (M)ale or (F)emale")

predictions = model.predict([ [ageinput1, genderinput1], [ageinput2, genderinput2] ])
predictions


# In[83]:


#testing predictions without user input
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #allocating 20% of data for testing

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

score = accuracy_score(y_test, predictions)
score


# In[88]:


#testing predictions without user input
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

#music_data = pd.read_csv('music.csv')
#x = music_data.drop(columns=['genre'])
#y = music_data['genre']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #allocating 20% of data for testing

#model = DecisionTreeClassifier()
#model.fit(x_train, y_train)

model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
predictions


# In[89]:


#visualizing data as decision tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x,y)

tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names=['age', 'gender'], 
                    class_names=sorted(y.unique()),
                    label='all',
                    rounded=True,
                    filled=True)


# In[ ]:




