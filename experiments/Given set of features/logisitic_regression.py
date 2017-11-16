import sys
import pickle
import os
sys.path.append("../../tools/")
#from sklearn.metrics import classification_report
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
#from tester import dump_classifier_and_data 
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import sklearn.pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
import sklearn.grid_search
from sklearn.decomposition import PCA

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

os.chdir('../../final_submission/')

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)    

my_dataset = data_dict
features_all = set()

for x in my_dataset : 
    for features in my_dataset[x]:
        features_all.add(features)
try :
    features_all.remove('poi')
except KeyError:
    pass

del my_dataset['TOTAL']
del my_dataset['THE TRAVEL AGENCY IN THE PARK']
del my_dataset['LOCKHART EUGENE E']

# del my_dataset['GRAMM WENDY L']
# del my_dataset['WHALEY DAVID A']
# del my_dataset['WROBEL BRUCE'] -- brings down precision and accuracy 


features_list = ['poi']
for i in features_all : 
    if(i != 'email_address'):
        features_list.append(i)    

print('\nTotal number of features are  : ' , len(features_list))


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest

select = SelectKBest()

#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

scaler = preprocessing.MinMaxScaler()

pca = PCA()

steps = [('feature_selection' , select) , ('scaler' , scaler), ('classifier' , clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

parameters = parameters = dict(feature_selection__k=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,'all'] ,
                               classifier__penalty = ['l2'] ,
                               classifier__max_iter = [100,200,500,1000 , 10000 , 100000] ,
                               classifier__solver = ['newton-cg' , 'sag' , 'lbfgs'] ,
                      #         classifier__dual = [True] ,
                               classifier__fit_intercept = [True , False] ,
                               PCA__random_state = [42] ,
                               PCA__n_components = [1],
                               PCA__whiten = [True , False] )

sss = StratifiedShuffleSplit(labels, 5, test_size=0.3, random_state=60)

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid = parameters , scoring = 'f1' ,cv = sss)
32

cv.fit(features, labels)

# X_new = cv.best_estimator_.named_steps['PCA']

# # Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
# feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
# # Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
# feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
# # Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
# # create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
# features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in X_new.get_support(indices=True)]

# Sort the tuple by score, in reverse order
# features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# # Print
# print ' '
# print 'Not Scaled - Selected Features, Scores, P-Values'
# print features_selected_tuple

# y = cv.predict(features_test)

#print(str('\n Best estimators : \n')+ str(cv.best_estimator_))

clf  = cv.best_estimator_

test_classifier(clf , my_dataset , features_list)

#_______________________________________________________________________________________
# #Lib-linear - wthout dual!
# Pipeline(steps=[('feature_selection', SelectKBest(k=14, score_func=<function f_c
# lassif at 0x0000000016829D68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Fal
# se, fit_intercept=False,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.87160       Precision: 0.56655      Recall: 0.15750 F1: 0.24
# 648     F2: 0.18408
#         Total predictions: 15000        True positives:  315    False positives:
#   241   False negatives: 1685   True negatives: 12759


#lib-linear - dual!

# Pipeline(steps=[('feature_selection', SelectKBest(k=15, score_func=<function f_c
# lassif at 0x00000000167D4D68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=7, random
# _state=42,
#   svd_solver='auto', tol=0.0, whiten=False)), ('classifier...ty='l2', random_sta
# te=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.70873       Precision: 0.28553      Recall: 0.78850 F1: 0.41
# 925     F2: 0.58308
#         Total predictions: 15000        True positives: 1577    False positives:
#  3946   False negatives:  423   True negatives: 9054


#PCA : 
#Liblinear - with dual

# Pipeline(steps=[('feature_selection', SelectKBest(k=14, score_func=<function f_c
# lassif at 0x0000000016729D68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=5, random
# _state=42,
#   svd_solver='auto', tol=0.0, whiten=False)), ('classifier...ty='l2', random_sta
# te=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.71133       Precision: 0.29182      Recall: 0.81650 F1: 0.42
# 996     F2: 0.60054
#         Total predictions: 15000        True positives: 1633    False positives:
#  3963   False negatives:  367   True negatives: 9037

#Without duel!
# Pipeline(steps=[('feature_selection', SelectKBest(k=18, score_func=<function f_c
# lassif at 0x0000000016759D68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=18, rando
# m_state=42,
#   svd_solver='auto', tol=0.0, whiten=False)), ('classifie...ty='l1', random_stat
# e=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.70160       Precision: 0.29165      Recall: 0.86650 F1: 0.43
# 641     F2: 0.62150
#         Total predictions: 15000        True positives: 1733    False positives:
#  4209   False negatives:  267   True negatives: 8791

#__________________________________________________________________________________
#Not - lib - linear

# Pipeline(steps=[('feature_selection', SelectKBest(k=15, score_func=<function f_c
# lassif at 0x00000000166FED68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Fal
# se, fit_intercept=False,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.86600       Precision: 0.49035      Recall: 0.12700 F1: 0.20

# 175     F2: 0.14910
#         Total predictions: 15000        True positives:  254    False positives:
#   264   False negatives: 1746   True negatives: 12736

#PCA - Non - lib - linear
# Pipeline(steps=[('feature_selection', SelectKBest(k=11, score_func=<function f_c
# lassif at 0x0000000016758D68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=11, rando
# m_state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier...ty='l2', random_stat
# e=None, solver='newton-cg', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.86367       Precision: 0.47399      Recall: 0.20500 F1: 0.28
# 621     F2: 0.23125
#         Total predictions: 15000        True positives:  410    False positives:
# #   455   False negatives: 1590   True negatives: 12545

