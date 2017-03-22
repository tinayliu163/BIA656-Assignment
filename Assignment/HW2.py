# BIA656 HW2

# Predictive modeling
# classification model

import pandas as pd
import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# load dataset
DM = pd.read_csv("./directMarketing.csv")
freq_saleSizeCode = DM['saleSizeCode'].value_counts()
freq_starCustomer = DM['starCustomer'].value_counts()


# Data pre-processing

# replace values in 'starCustomer' and 'saleSizeCode' columns
DM['starCustomer'] = DM['starCustomer'].replace('X', 1)
DM['saleSizeCode'] = DM['saleSizeCode'].replace(['F','E','G','D'], [1,2,3,4])

train, test = train_test_split(DM, test_size = 0.3)

null = train.isnull().sum()
# no missing observations in trainning set
column = train.columns.tolist()
col = column[:-1]

# fit a logistic regression model
logreg = LogisticRegression()
X = train[col]
Y = train['class']
logreg.fit(X, Y)


# evaluate model in testing data
feature_data = test[col]
test['class_pred'] = logreg.predict(feature_data)
y_true = test['class']
y_pred = test['class_pred']


print(" ******************* Evaluation of Logistic regression *********************")

# test error rate

accuracy = metrics.accuracy_score(y_true, y_pred)
error_rate = 1 - accuracy
print("test error rate : %0.3f" % error_rate)

# confusion matrix
matrix = metrics.confusion_matrix(y_true, y_pred)
print(matrix)

# the area under the ROC curve 
fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred)
AUC = metrics.auc(fpr, tpr)
print("AUC : %0.3f " % AUC)

# Visualization

# ROC curve

plt.plot(fpr, tpr, color = 'r', label = 'ROC curve (area = %0.4f)' % AUC,)
plt.plot([0, 1], [0, 1],  color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic regression : ROC Curve ')

# precison/recall curve
precision, recall, thresholds2 = metrics.precision_recall_curve(y_true, y_pred)
plt.plot(recall, precision, color='g')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Logistic regression : precision/recall Curve')


# fit Decision tree model

decs_tree = tree.DecisionTreeClassifier()
decs_tree.fit(X, Y)
test['class_decstree_pred'] = decs_tree.predict(feature_data)
y_decs_pred = test['class_decstree_pred']


print(" ******************* Evaluation of Decision tree *********************")

# test error rate

accuracy_decs = metrics.accuracy_score(y_true, y_decs_pred)
error_rate_decs = 1 - accuracy_decs
print("test error rate : %0.3f" % error_rate_decs)

# confusion matrix
matrix_decs = metrics.confusion_matrix(y_true, y_decs_pred)
print(matrix_decs)

# the area under the ROC curve 
fpr_decs, tpr_decs, thresholds_decs = metrics.roc_curve(y_true,y_decs_pred)
AUC_decs = metrics.auc(fpr_decs, tpr_decs)
print("AUC : %0.3f" % AUC_decs)

# Visualization

# ROC curve

plt.plot(fpr_decs, tpr_decs, color = 'r', label = 'ROC curve (area = %0.4f)' % AUC_decs,)
plt.plot([0, 1], [0, 1],  color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision tree : ROC Curve')

# precison/recall curve
precision_decs, recall_decs, thresholds2_decs = metrics.precision_recall_curve(y_true, y_decs_pred)
plt.plot(recall_decs, precision_decs, color='g')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Decision tree : precision/recall Curve')


# fit Naive Bayes model
gnb = GaussianNB()
gnb.fit(X,Y)
test['class_gnb_pred'] = gnb.predict(feature_data)
y_gnb_pred = test['class_gnb_pred']


print(" ******************* Evaluation of Naive Bayes *********************")

# test error rate

accuracy_gnb = metrics.accuracy_score(y_true, y_gnb_pred)
error_rate_gnb = 1 - accuracy_gnb
print("test error rate : %0.3f" % error_rate_gnb)

# confusion matrix
matrix_gnb = metrics.confusion_matrix(y_true, y_gnb_pred)
print(matrix_gnb)

# the area under the ROC curve 
fpr_gnb, tpr_gnb, thresholds_gnb = metrics.roc_curve(y_true,y_gnb_pred)
AUC_gnb = metrics.auc(fpr_gnb, tpr_gnb)
print("AUC : %0.3f" % AUC_gnb)

# Visualization

# ROC curve

plt.plot(fpr_gnb, tpr_gnb, color = 'r', label = 'ROC curve (area = %0.4f)' % AUC_gnb,)
plt.plot([0, 1], [0, 1],  color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes : ROC Curve')

# precison/recall curve
precision_gnb, recall_gnb, thresholds2_gnb = metrics.precision_recall_curve(y_true, y_gnb_pred)
plt.plot(recall_gnb, precision_gnb, color='g')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Naive Bayes : precision/recall Curve')

# fit SVM model

svm = SVC()
svm.fit(X,Y)
test['class_svm_pred'] = svm.predict(feature_data)
y_svm_pred = test['class_svm_pred']


print(" ******************* Evaluation of SVM *********************")

# test error rate

accuracy_svm = metrics.accuracy_score(y_true, y_svm_pred)
error_rate_svm = 1 - accuracy_svm
print("test error rate : %0.3f" % error_rate_svm)

# confusion matrix
matrix_svm = metrics.confusion_matrix(y_true, y_svm_pred)
print(matrix_svm)

# the area under the ROC curve 
fpr_svm, tpr_svm, thresholds_svm = metrics.roc_curve(y_true, y_svm_pred)
AUC_svm = metrics.auc(fpr_svm, tpr_svm)
print("AUC : %0.3f" % AUC_svm)

# Visualization

# ROC curve

plt.plot(fpr_svm, tpr_svm, color = 'r', label = 'ROC curve (area = %0.4f)' % AUC_svm,)
plt.plot([0, 1], [0, 1],  color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM : ROC Curve')

# precison/recall curve
precision_svm, recall_svm, thresholds2_svm = metrics.precision_recall_curve(y_true, y_svm_pred)
plt.plot(recall_svm, precision_svm, color='g')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('SVM : precision/recall Curve')















