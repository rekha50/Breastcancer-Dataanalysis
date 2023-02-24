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

p=df.iloc[0:569, 0:31]  
print(p)
T=df.loc[0:569, 'diagnosis']
print(T)


x_train, x_test, y_train, y_test = train_test_split(p, T, test_size = 0.6, random_state=10)
print(x_test)
print(x_train)
print(y_train)


#feature scaling for LDA:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)

#Applying LDA on data:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
model.fit(x_train,y_train)
print(model.predict(x_test))
print(model.score(x_test,y_test))


#ROC curve for training data:
from sklearn.metrics import plot_roc_curve
roc_lda = plot_roc_curve(model, x_train, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_lda = plot_roc_curve(model, x_test, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = model.predict(x_train)
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)

matrix_confusion = confusion_matrix(y_train, y_pred_train)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_pred_train,y_train,cmap='PuRd')
plt.show()

#Confusion matrix for Testing Data:
y_pred_test = model.predict(x_test)
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
pp_matrix_from_data(y_pred_test,y_test,cmap= 'PuRd')
plt.show()
