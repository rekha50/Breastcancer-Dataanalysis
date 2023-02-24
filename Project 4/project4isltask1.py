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


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from numpy import asarray


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

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)
model6.fit(x_train,y_train)
model7.fit(x_train,y_train)
model8.fit(x_train,y_train)


y_pred1=model1.predict(x_test)
y_pred2=model2.predict(x_test)
y_pred3=model3.predict(x_test)
y_pred4=model4.predict(x_test)
y_pred5=model5.predict(x_test)
y_pred6=model6.predict(x_test)
y_pred7=model7.predict(x_test)
y_pred8=model8.predict(x_test)

#Fetching outputs of classifiers:
clf = [model1,model2,model3,model4,model5,model6,model7,model8]
for algo in clf:
    score = cross_val_score( algo,p,T,cv = 10,scoring = 'accuracy')
    data = score.mean()
    print("The accuracy score of {} is:".format(algo),score.mean())

#Norrmalisation
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = asarray([0.9595870206489675,0.8910572892408011,0.9578171091445427,0.9385188635305075,0.9121720229777983,0.9455364073901569,0.9121720229777985,0.9086632510479739])
print(data)
scaled_data = scaler.fit_transform(data.reshape(-1, 1))
print(scaled_data)

def fetch_fp(y_test,final_pred):
    num_fp=0
    for idx, (t,pred) in enumerate(zip(y_test,final_pred)):
        if pred == 1 and t == 0 :
            print('hello')
            num_fp = num_fp+1
    return num_fp    


def combine_model(model1,model2,model3):
    model=model1+model2+model3
    model=model/3
    if model>=0.5:
        return 1
    else:
        return 0


# final prediction after product on the prediction of all 3 models
ensem3 = 0.35 * scaled_data[2] + 0.45*scaled_data[0] + 0.2*scaled_data[5] 
print(ensem3)

print('************************************')


# final prediction after averaging on the prediction of all 3 models
final_pred1 = []

for i in range(0,len(x_test)):
    
    final_pred1.append([list(y_pred3)[i]+list(y_pred4)[i]+list(y_pred5)[i]])
print(final_pred1) 
print(len(final_pred1) )
print(len(list(y_test)))

print(list(y_test))
true_pred=0
for idx, (true,pred) in enumerate(zip(y_test,final_pred1)):
    if true == pred :
        true_pred=true_pred+1
print(true_pred/len(y_test))    
num_fp = fetch_fp(y_test,final_pred1)   
print(num_fp) 

#Another way:
num=(scaled_data[0]+scaled_data[2]+scaled_data[5])
ensem1= num/3
print(ensem1)

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


#Max:

final_pred = []
for i in range(0,len(x_test)):
    
    final_pred.append(mode([list(y_pred3)[i],list(y_pred4)[i],list(y_pred5)[i]]))
print(final_pred) 
print(len(final_pred) )
print(len(list(y_test)))

print(list(y_test))
true_pred=0
for idx, (true,pred) in enumerate(zip(y_test,final_pred)):
    if true == pred :
        true_pred=true_pred+1
print(true_pred/len(y_test))    
num_fp = fetch_fp(y_test,final_pred)   
print(num_fp) 
