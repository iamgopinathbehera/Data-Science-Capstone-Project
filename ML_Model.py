import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle    # Save the model

# Read the datset
df = pd.read_csv('encoded_car_df.csv')
# Top 5 rows
print(df.head())

# Column Names
print(df.columns)   # 

# Shape
print(df.shape)   # rows=149,cols=5

# Data Preprocessing
# 1) Handling Null Values
print(df.isnull().sum())      # There are no null values

# 2) Handle the duplicates
print(df.duplicated().sum())  # Total number of dupliactes =  3

# Drop the duplicates
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

# 4) Check the data types
print(df.dtypes) 

# 5) Checking the target variable
print(df['selling_price'].value_counts())

# Select x (independenat feature) and y (dependent feature)
x = df.drop(['selling_price','brand'],axis=1)
y = df['selling_price']

print(type(x))    # Dataframe
print(type(y))    # Series
print(x.shape)     
print(y.shape)    


# Split the data into train and test data
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
print(x_train.shape)  
print(x_test.shape)       
print(y_train.shape)       
print(y_test.shape)        

# ML Model Building

lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=80,criterion='gini',
                            max_depth=4,min_samples_split=15)
knn = KNeighborsClassifier(n_neighbors=11)

lr.fit(x_train,y_train)
rf.fit(x_train,y_train)
knn.fit(x_train,y_train)


print('Test Score LR',lr.score(x_test,y_test))
print('Test Score RF',rf.score(x_test,y_test))
print('Test Score KNN',knn.score(x_test,y_test))

# Saving the  model
pickle.dump(rf,open('rf_model.pkl','wb'))
pickle.dump(lr,open('lr_model.pkl','wb'))
pickle.dump(knn,open('knn_model.pkl','wb'))
