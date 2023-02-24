import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# Reading data from given dataset breast-cancer-wisconsin.csv file #
df = pd.read_csv("project1data.csv")
print(df)#to show us the values#


for index,category in enumerate(df.diagnosis):
    if df.diagnosis[index]== 'B' :
        df.diagnosis[index] = 'C1'
    else:
        df.diagnosis[index] = 'C2'

            
print(df.diagnosis)

plt.scatter(df.id,df.diagnosis, color='red',marker='+')
plt.show()

np_array=df.to_numpy()
print(np_array)


print(df['diagnosis'].value_counts())
sns.countplot(df['diagnosis'],label="Count")
plt.show()

p=df.iloc[0:569, 1:31]  
print(p)
T=df.loc[0:569, 'diagnosis']
print(T)


x_train, x_test, y_train, y_test = train_test_split(p, T, test_size = 0.3, random_state=10)
print(x_test)
print(x_train)
print(y_train)



#feature scaling for LDA:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print(x_train.shape)

from sklearn import svm, datasets


## rbf kernel with varying c value as 0.01,5,10 and 15:
svc_classifier = svm.SVC(kernel='rbf', C=0.01).fit(x_train,y_train)
print(svc_classifier.predict(x_test))
print(svc_classifier.score(x_test,y_test))

#ROC curve for training data:
from sklearn.metrics import  plot_roc_curve
roc= plot_roc_curve(svc_classifier, x_train, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc= plot_roc_curve(svc_classifier, x_test, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = svc_classifier.predict(x_train)
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)

matrix_confusion = confusion_matrix(y_train, y_pred_train)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_train, y_pred_train,cmap='PuRd')
plt.show()

#Confusion matrix for Testing Data:
from sklearn.metrics import confusion_matrix
y_pred_test = svc_classifier.predict(x_test)
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

from sklearn.metrics import plot_confusion_matrix
matrix_confusion = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_test, y_pred_test,cmap='PuRd')
plt.show()


## Linear kernel with varying c value as 0.01,5,10 and 15:
svc_classifier = svm.SVC(kernel='linear', C=30).fit(x_train,y_train)
print(svc_classifier.predict(x_test))
print(svc_classifier.score(x_test,y_test))

#ROC curve for training data:
from sklearn.metrics import  plot_roc_curve
roc= plot_roc_curve(svc_classifier, x_train, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc= plot_roc_curve(svc_classifier, x_test, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = svc_classifier.predict(x_train)
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)

matrix_confusion = confusion_matrix(y_train, y_pred_train)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_train, y_pred_train,cmap='PuRd')
plt.show()

#Confusion matrix for Testing Data:
from sklearn.metrics import confusion_matrix
y_pred_test = svc_classifier.predict(x_test)
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

from sklearn.metrics import plot_confusion_matrix
matrix_confusion = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_test, y_pred_test,cmap='PuRd')
plt.show()


## polynomial kernel with degree 2 with varying c value as 0.01,5,10 and 15:
svc_classifier = svm.SVC(kernel='poly', C=0.01,degree=2).fit(x_train,y_train)
print(svc_classifier.predict(x_test))
print(svc_classifier.score(x_test,y_test))

#ROC curve for training data:
from sklearn.metrics import  plot_roc_curve
roc = plot_roc_curve(svc_classifier, x_train, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc = plot_roc_curve(svc_classifier, x_test, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = svc_classifier.predict(x_train)
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)

matrix_confusion = confusion_matrix(y_train, y_pred_train)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_train, y_pred_train,cmap='PuRd')
plt.show()

#Confusion matrix for Testing Data:
from sklearn.metrics import confusion_matrix
y_pred_test = svc_classifier.predict(x_test)
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

from sklearn.metrics import plot_confusion_matrix
matrix_confusion = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_test, y_pred_test,cmap='PuRd')
plt.show()


## polynomial kernel with degree 3 with varying c value as 0.01,5,10 and 15:
svc_classifier = svm.SVC(kernel='poly', C=0.01, degree=3).fit(x_train,y_train)
print(svc_classifier.predict(x_test))
print(svc_classifier.score(x_test,y_test))

#ROC curve for training data:
from sklearn.metrics import  plot_roc_curve
roc = plot_roc_curve(svc_classifier, x_train, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc = plot_roc_curve(svc_classifier, x_test, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = svc_classifier.predict(x_train)
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)

matrix_confusion = confusion_matrix(y_train, y_pred_train)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_train, y_pred_train,cmap='PuRd')
plt.show()

#Confusion matrix for Testing Data:
from sklearn.metrics import confusion_matrix
y_pred_test = svc_classifier.predict(x_test)
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

from sklearn.metrics import plot_confusion_matrix
matrix_confusion = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_test, y_pred_test,cmap='PuRd')
plt.show()

