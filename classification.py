import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
train = pd.read_csv('dataset.csv')
X = train
y = train['Type']


Y=np.zeros(len(y))
X.drop('Type',axis=1,inplace=True)


for i in range(len(y)):
   if(y[i]=="BIKE"):
       Y[i]=0
   elif (y[i]=="CAR"):	
       Y[i]=1
   elif (y[i]=="BUS"):
       Y[i]=2

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)



print("\n\n********************************GAUSSIANNB************************************")
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("\nGaussianNB Accuracy:",accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_pred,y_test)
print('\nconfusion_matrix')
print(cm)
print("\n")
print(classification_report(y_test, y_pred))
###############################################################################
print("\n\n********************************MULTINOMIALNB************************************")
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("\nMultinomialNB Accuracy:",accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_pred,y_test)
print('\nconfusion_matrix')
print(cm)
print("\n")
print("\n")
print(classification_report(y_test, y_pred))
###############################################################################
print("\n\n********************************LOGISTICREGRESSION************************************")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("\nLogisticRegression Accuracy:",accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_pred,y_test)
print('\nconfusion_matrix')
print(cm)
print("\n")
print(classification_report(y_test, y_pred))
###############################################################################
print("\n\n********************************RANDOMFOREST************************************")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("\nRandomForestClassifier Accuracy:",accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_pred,y_test)
print('\nconfusion_matrix')
print(cm)
print("\n")
print(classification_report(y_test, y_pred))


# ###############################################################################
print("\n\n********************************SVC************************************")
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("SVC Accuracy:",accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_pred,y_test)
print('\nconfusion_matrix')
print(cm)
print("\n")
print(classification_report(y_test, y_pred))
