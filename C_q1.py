import numpy as np
import pandas as pd
import io
import joblib

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier

def find(Y_test, y_predicted,st,X_test):
    acccuracy = accuracy_score(Y_test, y_predicted)
    recall = recall_score(Y_test, y_predicted, average="weighted")
    precision = precision_score(Y_test, y_predicted, average="weighted")

    print("------------- Decision Tree Results -------------")
    print(st)
    print("Precision: ", precision)
    print("Recall   : ", recall)
    print("Accuracy : ", acccuracy)

    #------AUC ROC curve-----

    y_predicted_probability=ht.predict_proba(X_test)

    y_predicted_probability=np.delete(y_predicted_probability,0,1)
    # print(y_predicted_probability)

    fpr = {}
    tpr = {}
    thresh ={}

    for i in range(1,4):
        fpr[i-1], tpr[i-1], thresh[i-1] = roc_curve(Y_test, y_predicted_probability[:,i-1], pos_label=i)

    plt.plot(fpr[0], tpr[0], linestyle="dotted",color='blue', label='Class 1')
    plt.plot(fpr[1], tpr[1],linestyle="dotted",color='red', label='Class 2 ')
    plt.plot(fpr[2], tpr[2],linestyle="dotted",color='black', label='Class 3')

    plt.xlabel('Rate of false +ve')
    plt.ylabel('Rate of true +ve')
    plt.legend(loc='lower right')
    plt.title("ROC curve")
    plt.show()
    # plt.savefig('AUC-ROC '+st,dpi=300);
    plt.close()

    auc_score=roc_auc_score(Y_test,y_predicted_probability,multi_class="ovr",average='weighted')
    print("auc score: ",auc_score)

dataset = pd.read_csv('data_1.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

y_testd1=Y_test
x_testd1=X_test

ht = HoeffdingTreeClassifier()
y_pred = ht.predict(X_train)
ht = ht.partial_fit(X_train, Y_train)

y_predicted=ht.predict(X_test)
# print(type(y_predicted))
find(Y_test,y_predicted,"performance of testset of data1 on dc-1 ",X_test)

#---------------------------------

dataset = pd.read_csv('data_2.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
y_predicted=ht.predict(X_test)
find(Y_test,y_predicted,"Performance of testset of data2 on dc-1",X_test)

correct_cnt = 0
# print(np.reshape(X_train[0], (1, 21)))

for i in range(len(X_train)):

    y_pred = ht.predict(np.reshape(X_train[i], (1, 21)))
    # print(type(Y_train[i]))
    ht = ht.partial_fit(np.reshape(X_train[i], (1, 21)), np.array([Y_train[i]]))


prob=[]
for i in range(len(X_test)):

    y_pred = ht.predict(np.reshape(X_test[i], (1, 21)))
    prob.append(y_pred[0])

# print(prob)
# print(type(prob))
prob=np.array(prob)
# print(type(prob))

find(Y_test,prob,"Performance of testset of data2 on dc-2",X_test)

prob=[]
for i in range(len(x_testd1)):

    y_pred = ht.predict(np.reshape(x_testd1[i], (1, 21)))
    prob.append(y_pred[0])

# print(prob)
# print(type(prob))
prob=np.array(prob)
# print(type(prob))
find(y_testd1,prob,"Performance of testset of data1 on dc-2",x_testd1)

joblib.dump(ht,'DT_C_1.pkl')
