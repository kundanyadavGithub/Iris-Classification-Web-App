import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Read the dataset
df = pd.read_csv("/home/kundan/Datasets_/iris.csv")

# Top 5 Rows
print(df.head())

# Columns name ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
print(df.columns)

# Shape of the data set
print(df.shape) # Rows = 149 , cols = 5ata 

# Data Preprocessing
# 1) Handling the Null values
print(df.isnull().sum().any()) # No null values

# 2) Handle the duplicates
print(df.duplicated().sum()) # Total number of duplicates = 3

# 3) Drop the duplicates
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

# 4) Check the data type
print(df.dtypes)

# 5) Checking the target variable
print(df['label'].value_counts())
# Iris-versicolor    50
# Iris-virginica     49
# Iris-setosa        47

#Select x(Independent Feature) and y (dependent feature)
x=df.drop('label',axis=1)
y=df['label']
print(type(x))  # Dataframe
print(type(y))  # Series
#print(x.shape)  # (146,4)
#print(y.shape)  # (146,)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
print(x_train.shape)  #(102, 4)
print(x_test.shape)   #(102, 4)
print(y_train.shape)  #(102,)
print(y_test.shape)   #(44,)

lr  = LogisticRegression()
rf  = RandomForestClassifier(n_estimators=80,criterion='gini',max_depth=4,min_samples_split=15)
knn = KNeighborsClassifier(n_neighbors=11)

lr.fit(x_train,y_train)
rf.fit(x_train,y_train)
knn.fit(x_train,y_train)

print('Test Score LR',lr.score(x_test,y_test))
print('Test Score RF',rf.score(x_test,y_test))
print('Test Score KNN',knn.score(x_test,y_test))

pickle.dump(lr,open('lr_model.pkl','wb'))
pickle.dump(rf,open('rf_model.pkl','wb'))
pickle.dump(knn,open('knn_model.pkl','wb'))
