import sys
import pickle
import os
sys.path.append("../../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
#from tester import dump_classifier_and_data 
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import sklearn.pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
import sklearn.grid_search
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

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

from sklearn.ensemble import ExtraTreesClassifier

clf =   LinearSVC()

scaler = preprocessing.MinMaxScaler()

pca = PCA()

steps = [('feature_selection' , select) ,('scaler' , scaler),  ('classifier' , clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

parameters = parameters = dict(feature_selection__k = [4] , 
                               classifier__loss = ['hinge' , 'squared_hinge'] ,
                               classifier__max_iter = [1000,500,2000] ,
                               classifier__tol = [1e-2 , 1e-4 , 1e-6 , 1e-8 , 1e-10] ,
                               classifier__multi_class = ['ovr' , 'crammer_singer'] )
                               # PCA__n_components =  [] ,
                               # PCA__random_state = [42] ,
                               # PCA__whiten = [True , False])

sss = StratifiedShuffleSplit(labels, 6, test_size=0.3, random_state=60)

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid = parameters , scoring = 'f1' ,cv = sss)

cv.fit(features, labels)

#X_new = cv.best_estimator_.named_steps['PCA']



# explained_ratio_components = ['%s' % i for i in X_new.explained_variance_ratio_]

# print('\n\n' + str('Ratio'))

# for i in explained_ratio_components :
#     print(i)





#y = cv.predict(features_test)

#print(str('\n Best estimators : \n')+ str(cv.best_estimator_))

clf  = cv.best_estimator_

test_classifier(clf , my_dataset , features_list)

# Pipeline(steps=[('feature_selection', SelectKBest(k=12, score_func=<function f_c
# lassif at 0x00000000166E7D68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LinearSVC(C=1.0, class_weight=None, dual=True, fit_in
# tercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.01,
#      verbose=0))])
#         Accuracy: 0.86420       Precision: 0.46726      Recall: 0.13200 F1: 0.20
# 585     F2: 0.15412
#         Total predictions: 15000        True positives:  264    False positives:
#   301   False negatives: 1736   True negatives: 12699

# #With - PCA :

# Pipeline(steps=[('feature_selection', SelectKBest(k=19, score_func=<function f_c
# lassif at 0x000000001669DD68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=13, rando
# m_state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier... max_iter=500,
#      multi_class='ovr', penalty='l2', random_state=None, tol=1e-10,
#      verbose=0))])
#         Accuracy: 0.86240       Precision: 0.46364      Recall: 0.20400 F1: 0.28
# 333     F2: 0.22973
#         Total predictions: 15000        True positives:  408    False positives:
#   472   False negatives: 1592   True negatives: 12528

