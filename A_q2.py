import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score
import pandas as pd


dataset = pd.read_csv('data_1.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# print(X)
# print(Y)

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier


accuracy=[]
depth=[2,4,6,8,10]
for i in range(5):
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0,max_depth=depth[i])
    model = classifier.fit(X_train, Y_train)
    # Predicting the Test set results
    y_predicted = classifier.predict(X_test)
    # print(y_predicted)
    accuracy.append( accuracy_score(Y_test, y_predicted))
    # print(accuracy)

plt.plot(depth,accuracy)
plt.xlabel("max depths")
plt.ylabel("accuracy")
plt.savefig('AccuracyVsDepth.png')
plt.show()