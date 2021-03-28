import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('C:/Users/zydyx/Desktop/351 project/data_featureselection.csv')
dataset.shape
x = dataset.drop('team_one_win',axis = 1)
y = dataset['team_one_win']

#split the dataset into training set and testing set 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

#build svc using linear kernal
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)

#predict values using the SVM model
y_pred = svclassifier.predict(x_test)

print (confusion_matrix(y_test, y_pred))
print (classification_report(y_test,y_pred))