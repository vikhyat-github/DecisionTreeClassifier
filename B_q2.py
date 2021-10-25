import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

dataset = pd.read_csv('data_1.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# print(X)
# print(Y)



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,max_leaf_nodes=8)
model = classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_predicted = classifier.predict(X_test)

path = classifier.cost_complexity_pruning_path(X_train, Y_train)
alpha_value = path['ccp_alphas']
# print(alpha_value)

# print(accuracy_score(Y_test,y_predicted))

def find_Alpha_train(training):
    i=0
    while i<len(alpha_value):
        classifier1=DecisionTreeClassifier(ccp_alpha=alpha_value[i],random_state=0,max_leaf_nodes=8)
        classifier1.fit(X_train,Y_train)
        training.append(accuracy_score(Y_train,classifier1.predict(X_train)))
        i+=1
    return training

def find_Alpha_test(testing):
    i = 0
    while i < len(alpha_value):
        classifier1 = DecisionTreeClassifier(ccp_alpha=alpha_value[i],random_state=0,max_leaf_nodes=8)
        classifier1.fit(X_train, Y_train)
        testing.append(accuracy_score(Y_test, classifier1.predict(X_test)))
        i += 1
    return testing

test_accuracy=[]
train_accuracy=[]
find_Alpha_test(test_accuracy)
find_Alpha_train(train_accuracy)
# print(test_accuracy)
# print(train_accuracy)


plt.figure(figsize=(20,20))
plt.plot(alpha_value ,train_accuracy,color='blue',label='Train Accuracy')
plt.plot( alpha_value,test_accuracy,color='red' ,label='Test Accuracy')
plt.xticks(ticks=np.arange(0.00,0.40,0.01))
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.title("Alpha vs Accuracy")
plt.show()
# plt.savefig('AccuracyVsAplha.png',dpi=300)


def find(Y_test, y_predicted,st):
    acccuracy = accuracy_score(Y_test, y_predicted)
    recall = recall_score(Y_test, y_predicted, average="weighted")
    precision = precision_score(Y_test, y_predicted, average="weighted")
    # f1_score = f1_score(Y_test, y_predicted, average="micro")

    print("------------- Decision Tree Results -------------")
    print(st)
    print("Precision: ", precision)
    print("Recall   : ", recall)
    print("Accuracy : ", acccuracy)



find(Y_test,y_predicted,"DT-A")

alpha=0.00
classifier_ccp=DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=8,random_state=0,ccp_alpha=alpha)
classifier_ccp.fit(X_train,Y_train)
y_predicted_ccp=classifier_ccp.predict(X_test)
find(Y_test,y_predicted_ccp,"Cost Complexity pruning")



parameter={"criterion":['gini','entropy'], 'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,None],
           'splitter':['random','best'], 'max_features':['auto','sqrt','log2',None],'max_leaf_nodes':[8]}


grid_searchCV=GridSearchCV(estimator=classifier,param_grid=parameter)
grid_searchCV.fit(X_train,Y_train)



classifier_prep=DecisionTreeClassifier(criterion='gini',max_depth=5,max_features=None,splitter='best',max_leaf_nodes=8)
classifier_prep.fit(X_train,Y_train)
y_predicted_prep=classifier_prep.predict(X_test)
find(Y_test,y_predicted_prep,"Pre Pruning")
print("Pre Pruning Parameters",grid_searchCV.best_params_)



#------Visualization

feature_name=list(dataset.columns)
feature_name.remove("fetal_health")
# print(feature_name)



fig = plt.figure(figsize=(50,50))
_ =tree.plot_tree(classifier,feature_names=feature_name,filled=True,class_names=['1','2','3'],proportion=True)
# fig.savefig("DT_A.png")

fig = plt.figure(figsize=(50,50))
_ =tree.plot_tree(classifier_ccp,feature_names=feature_name,filled=True,class_names=['1','2','3'],proportion=True)
# fig.savefig("DT_B_CC.pdf")

fig = plt.figure(figsize=(50,50))
_ =tree.plot_tree(classifier_prep,feature_names=feature_name,filled=True,class_names=['1','2','3'],proportion=True)
# fig.savefig("DT_B_2_PP.pdf")