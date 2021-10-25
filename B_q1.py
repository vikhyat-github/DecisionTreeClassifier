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
from sklearn.tree._tree import TREE_LEAF
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score

dataset = pd.read_csv('data_1.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
# print(X)
# print(Y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# DT-A

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, max_leaf_nodes=8)
model = classifier.fit(X_train, Y_train)
y_predicted = classifier.predict(X_test)

count = 0


def delete_random(inner_tree, index, value):
    global count
    if inner_tree.children_left[index] == TREE_LEAF and inner_tree.children_right[index] == TREE_LEAF:
        count += 1

    if inner_tree.children_left[index] == TREE_LEAF:
        pass
    else:
        delete_random(inner_tree, inner_tree.children_left[index], value)
        if count == value:
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            count += 1
    if inner_tree.children_right[index] == TREE_LEAF:
        pass
    else:
        delete_random(inner_tree, inner_tree.children_right[index], value)
        if count == value:
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            count += 1


delete_random(classifier.tree_, 0, 6)

acccuracy = accuracy_score(Y_test, y_predicted)
# acccuracy = accuracy_score(Y_train, classifier.predict(X_train))
recall = recall_score(Y_test, y_predicted, average="weighted")
precision = precision_score(Y_test, y_predicted, average="weighted")
# f1_score = f1_score(Y_test, y_predicted, average="micro")

print("------------- Decision Tree Results -------------")

print("Precision: ", precision)
print("Recall   : ", recall)
print("Accuracy : ", acccuracy)

# ------AUC ROC curve-----


y_predicted_probability = classifier.predict_proba(X_test)
# print(y_predicted_probability)


fpr = {}
tpr = {}
thresh = {}

for i in range(1, 4):
    fpr[i - 1], tpr[i - 1], thresh[i - 1] = roc_curve(Y_test, y_predicted_probability[:, i - 1], pos_label=i)

# print(fpr)
# print(tpr)
plt.plot(fpr[0], tpr[0], linestyle="dotted", color='blue', label='Class 1')
plt.plot(fpr[1], tpr[1], linestyle="dotted", color='red', label='Class 2 ')
plt.plot(fpr[2], tpr[2], linestyle="dotted", color='black', label='Class 3')

plt.xlabel('Rate of false +ve')
plt.ylabel('Rate of true +ve')
plt.legend(loc='lower right')
plt.title("ROC curve")
plt.show()
plt.close()
# plt.savefig('AUC-ROC Curve ',dpi=300);

auc_score = roc_auc_score(Y_test, y_predicted_probability, multi_class="ovr", average='weighted')
print("auc score: ", auc_score)

feature_name = list(dataset.columns)
feature_name.remove("fetal_health")
# print(feature_name)

#
fig = plt.figure(figsize=(50, 50))
_ = tree.plot_tree(classifier, feature_names=feature_name, filled=True, class_names=['1', '2', '3'], proportion=True,
                   fontsize=10)


# fig.savefig("DT_A_1.pdf")

