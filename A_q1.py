import graphviz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import export_graphviz
import pydotplus
import graphviz

from sklearn.model_selection import train_test_split



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

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
model = classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_predicted = classifier.predict(X_test)
# print(y_predicted)

from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score

# accuracy = accuracy_score(Y_test, y_predicted)
# print("accuracy",accuracy)

acccuracy = accuracy_score(Y_test, y_predicted)
# acccuracy = accuracy_score(Y_train, classifier.predict(X_train))
recall = recall_score(Y_test, y_predicted, average="weighted")
precision = precision_score(Y_test, y_predicted, average="weighted")
# f1_score = f1_score(Y_test, y_predicted, average="micro")

print("------------- Decision Tree Results -------------")

print("Precision: ", precision)
print("Recall   : ", recall)
print("Accuracy : ", acccuracy)


#------AUC ROC curve-----


y_predicted_probability=classifier.predict_proba(X_test)
# print(y_predicted_probability)


fpr = {}
tpr = {}
thresh ={}

for i in range(1,4):
    fpr[i-1], tpr[i-1], thresh[i-1] = roc_curve(Y_test, y_predicted_probability[:,i-1], pos_label=i)

# print(fpr)
# print(tpr)
plt.plot(fpr[0], tpr[0], linestyle="dotted",color='blue', label='Class 1')
plt.plot(fpr[1], tpr[1],linestyle="dotted",color='red', label='Class 2 ')
plt.plot(fpr[2], tpr[2],linestyle="dotted",color='black', label='Class 3')

plt.xlabel('Rate of false +ve')
plt.ylabel('Rate of true +ve')
plt.legend(loc='lower right')
plt.title("ROC curve")
plt.savefig('AUC-ROC Curve ',dpi=300);

auc_score=roc_auc_score(Y_test,y_predicted_probability,multi_class="ovr",average='weighted')
print("auc score: ",auc_score)



#---------------------------------------------------

# feature_name=[ "baseline_value","accelerations","fetal_movement,uterine_contractions,light_decelerations,severe_decelerations,prolongued_decelerations,abnormal_short_term_variability,mean_value_of_short_term_variability,percentage_of_time_with_abnormal_long_term_variability,mean_value_of_long_term_variability,histogram_width,histogram_min,histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency,fetal_health]
feature_name=list(dataset.columns)
feature_name.remove("fetal_health")
# print(feature_name)

#
fig = plt.figure(figsize=(50,50))
_ =tree.plot_tree(classifier,feature_names=feature_name,filled=True,class_names=['1','2','3'],proportion=True,fontsize=10)
# fig.savefig("DT_A_1.pdf")






