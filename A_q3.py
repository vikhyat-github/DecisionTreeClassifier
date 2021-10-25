import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('data_1.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# print(y_predicted)
def find(Y_test, y_predicted,st,index):
    acccuracy = accuracy_score(Y_test, y_predicted)
    recall = recall_score(Y_test, y_predicted, average="weighted")
    precision = precision_score(Y_test, y_predicted, average="weighted")
    # f1_score = f1_score(Y_test, y_predicted, average="micro")


    print("------------- Decision Tree Results -------------")
    print(st)
    print("Precision: ", precision)
    print("Recall   : ", recall)
    print("Accuracy : ", acccuracy)
    y_predicted_probability = classifier.predict_proba(X_test)
    # print(y_predicted_probability)

    fpr = {}
    tpr = {}
    hold = {}

    for i in range(1, 4):
        fpr[i - 1], tpr[i - 1], hold[i - 1] = roc_curve(Y_test, y_predicted_probability[:, i - 1], pos_label=i)

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
    # plt.savefig('curve'+str(index)+'', dpi=300)

    auc_score = roc_auc_score(Y_test, y_predicted_probability, multi_class="ovr", average='weighted')
    print("auc score: ", auc_score)
    plt.close();

#hyperparameter 1
classifier = DecisionTreeClassifier(criterion='gini', random_state=0)
model_data = classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_predicted1 = classifier.predict(X_test)
find(Y_test,y_predicted1,"Criterion changed to gini",1)

#hyperparameter 2
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,splitter='random')
model_data = classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_predicted1 = classifier.predict(X_test)
find(Y_test,y_predicted1,"splitter changed to random",2)

#hyperparameter 3
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,max_depth=10)
model_data = classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_predicted1 = classifier.predict(X_test)
find(Y_test,y_predicted1,"maxdepth changed to 10",3)

#hyperparameter 4
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,min_samples_split=5)
model_data = classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_predicted1 = classifier.predict(X_test)
find(Y_test,y_predicted1,"min Sample Split changed to 5",4)

#hyperparameter 5
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,min_samples_leaf=2)
model_data = classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_predicted1 = classifier.predict(X_test)
find(Y_test,y_predicted1,"min Sample leaf changed to 2",5)

#hyperparameter 6
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,max_features='sqrt')
model_data = classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_predicted1 = classifier.predict(X_test)
find(Y_test,y_predicted1,"max features changed to sqrt",6)

#hyperparameter 7
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,class_weight='balanced')
model_data = classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_predicted1 = classifier.predict(X_test)
find(Y_test,y_predicted1,"class weight changed to balanced",7)

#hyperparameter 8
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,max_leaf_nodes=8)
model_data = classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_predicted1 = classifier.predict(X_test)
find(Y_test,y_predicted1,"max leaf node changed to 8",8)

