import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
from sklearn.metrics import plot_roc_curve

from sklearn.metrics import classification_report, confusion_matrix
from statistics import mode
from sklearn import datasets




# Reading data from given dataset breast-cancer-wisconsin.csv file #
df = pd.read_csv("project1data.csv")
print(df)#to show us the values#


for index,category in enumerate(df.diagnosis):
    if df.diagnosis[index]== 'B' :
        df.diagnosis[index] = '0'
    else:
        df.diagnosis[index] = '1'

            
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


model1 = LinearDiscriminantAnalysis()
model2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=1)
model3 = QuadraticDiscriminantAnalysis()
model4 = GaussianNB()
model5 = svm.SVC(kernel='rbf', C=1.0)
model6 = svm.SVC(kernel='linear', C=1.0)
model7 = svm.SVC(kernel='poly', C=1.0, degree=2)
model8 = svm.SVC(kernel='poly', C=1.0, degree=3)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


clf = [model1,model2,model3,model4,model5,model6,model7,model8]
for algo in clf:
    score = cross_val_score( algo,p,T,cv = 5,scoring = 'accuracy')
    print("The accuracy score of {} is:".format(algo),score.mean())


clf1 = [('model1',model1),('model2',model2),('model3',model3),('model4',model4),('model5',model5),('model6',model6),('model7',model7),('model8',model8)] #list of (str, estimator)
lr = LogisticRegression()
stack_model = StackingClassifier( estimators = clf1,final_estimator = lr)
#score = cross_val_score(stack_model,p,T,cv = 5,scoring = 'accuracy')
#print("The accuracy score of stack model is:",score.mean())

print(stack_model.fit(x_train, y_train).score(x_test, y_test))

#Roc for Training:
from sklearn.metrics import  plot_roc_curve
roc = plot_roc_curve(stack_model, x_train, y_train)
plt.show()


#Roc for Testing:
from sklearn.metrics import  plot_roc_curve
roc = plot_roc_curve(stack_model, x_test, y_test)
plt.show()



