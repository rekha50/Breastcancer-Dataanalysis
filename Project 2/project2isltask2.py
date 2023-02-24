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


from sklearn.decomposition import PCA
pca = PCA(0.99)
y=pca.fit(x_train)
print(pca.n_components_)
pca = PCA(n_components= pca.n_components_)
x_train_pca=pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
print(x_train_pca.shape)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))



#Applying LDA on data:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


model = LinearDiscriminantAnalysis()
model.fit(x_train_pca,y_train)
print(model.predict(x_test_pca))
print(model.score(x_test_pca,y_test))

#ROC curve for training data:
from sklearn.metrics import  plot_roc_curve
roc_lda = plot_roc_curve(model, x_train_pca, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_lda = plot_roc_curve(model, x_test_pca, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = model.predict(x_train_pca)
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
y_pred_test = model.predict(x_test_pca)
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


clf2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=1)
clf2.fit(x_train_pca,y_train)
print(clf2.predict(x_test_pca))
print(clf2.score(x_test_pca,y_test))
y_pred_test = model.predict(x_test_pca)

#ROC curve for training data:
from sklearn.metrics import plot_roc_curve
roc_dlda = plot_roc_curve(clf2, x_train_pca, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_dlda = plot_roc_curve(clf2, x_test_pca, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = clf2.predict(x_train_pca)
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
y_pred_test = clf2.predict(x_test_pca)
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


qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train_pca,y_train)
y_pred_train = qda.predict(x_train_pca)
y_pred_test = qda.predict(x_test_pca)
print(y_pred_train)
print(qda.score(x_test_pca,y_test))

#ROC curve for training data:
from sklearn.metrics import plot_roc_curve
roc_qda = plot_roc_curve(qda, x_train_pca, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_qda = plot_roc_curve(qda, x_test_pca, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = qda.predict(x_train_pca)
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
y_pred_test = qda.predict(x_test_pca)
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

matrix_confusion = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_test, y_pred_test,cmap='PuRd')
plt.show()


#covariance of X under classes C1 and C2 are different, but the off-diagonal elements are 0 ie DQDA
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train_pca,y_train)
y_pred_train = gnb.predict(x_train_pca)
y_pred_test = gnb.predict(x_test_pca)
print(y_pred_train)
print(gnb.score(x_test_pca,y_test))

#ROC curve for training data:
from sklearn.metrics import plot_roc_curve
roc_dqda = plot_roc_curve(gnb, x_train_pca, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_dqda = plot_roc_curve(gnb, x_test_pca, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = gnb.predict(x_train_pca)
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

y_pred_test = gnb.predict(x_test_pca)
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

matrix_confusion = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_test, y_pred_test,cmap='PuRd')
plt.show()


from sklearn.decomposition import PCA
pca = PCA(0.999)
y=pca.fit(x_train)
print(pca.n_components_)
pca = PCA(n_components= pca.n_components_)
x_train_pca=pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
print(x_train_pca.shape)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))



#Applying LDA on data:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


model = LinearDiscriminantAnalysis()
model.fit(x_train_pca,y_train)
print(model.predict(x_test_pca))
print(model.score(x_test_pca,y_test))

#ROC curve for training data:
from sklearn.metrics import  plot_roc_curve
roc_lda = plot_roc_curve(model, x_train_pca, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_lda = plot_roc_curve(model, x_test_pca, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = model.predict(x_train_pca)
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
y_pred_test = model.predict(x_test_pca)
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


clf2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=1)
clf2.fit(x_train_pca,y_train)
print(clf2.predict(x_test_pca))
print(clf2.score(x_test_pca,y_test))
y_pred_test = model.predict(x_test_pca)

#ROC curve for training data:
from sklearn.metrics import plot_roc_curve
roc_dlda = plot_roc_curve(clf2, x_train_pca, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_dlda = plot_roc_curve(clf2, x_test_pca, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = clf2.predict(x_train_pca)
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
y_pred_test = clf2.predict(x_test_pca)
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


qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train_pca,y_train)
y_pred_train = qda.predict(x_train_pca)
y_pred_test = qda.predict(x_test_pca)
print(y_pred_train)
print(qda.score(x_test_pca,y_test))

#ROC curve for training data:
from sklearn.metrics import plot_roc_curve
roc_qda = plot_roc_curve(qda, x_train_pca, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_qda = plot_roc_curve(qda, x_test_pca, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = qda.predict(x_train_pca)
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
y_pred_test = qda.predict(x_test_pca)
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

matrix_confusion = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_test, y_pred_test,cmap='PuRd')
plt.show()


#covariance of X under classes C1 and C2 are different, but the off-diagonal elements are 0 ie DQDA
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train_pca,y_train)
y_pred_train = gnb.predict(x_train_pca)
y_pred_test = gnb.predict(x_test_pca)
print(y_pred_train)
print(gnb.score(x_test_pca,y_test))

#ROC curve for training data:
from sklearn.metrics import plot_roc_curve
roc_dqda = plot_roc_curve(gnb, x_train_pca, y_train)
plt.show()


#ROC curve for testing data:
from sklearn.metrics import plot_roc_curve
roc_dqda = plot_roc_curve(gnb, x_test_pca, y_test)
plt.show()

#Confusion matrix for Training Data:
from sklearn.metrics import confusion_matrix
y_pred_train = gnb.predict(x_train_pca)
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

y_pred_test = gnb.predict(x_test_pca)
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

matrix_confusion = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()

from pretty_confusion_matrix import pp_matrix_from_data
pp_matrix_from_data(y_test, y_pred_test,cmap='PuRd')
plt.show()

