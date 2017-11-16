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
# del my_dataset['WROBEL BRUCE'] 

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

clf = ExtraTreesClassifier()

scaler = preprocessing.MinMaxScaler()

pca = PCA()

steps = [('feature_selection' , select) ,('scaler' , scaler), ('PCA' , pca) ,  ('classifier' , clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

parameters = parameters = dict(feature_selection__k=[16,18,19,'all'], 
                               classifier__n_estimators = [10 , 15 , 5] ,
                               classifier__criterion = ['gini' , 'entropy']  ,
                               classifier__min_samples_split = [2 , 3 , 4 , 5 , 10] ,
                               PCA__n_components =  [16] ,
                               PCA__random_state = [42] ,
                               PCA__whiten = [True , False] )

sss = StratifiedShuffleSplit(labels, 4 , test_size=0.3, random_state=60)

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid = parameters , scoring = 'f1' ,cv = sss)

cv.fit(features, labels)

#X_new = cv.best_estimator_.named_steps['PCA']

# explained_ratio_components = ['%s' % i for i in X_new.explained_variance_ratio_]

# print('\n\n' + str('Ratio'))

# for i in explained_ratio_components :
#     print(i)

y = cv.predict(features_test)

#print(str('\n Best estimators : \n')+ str(cv.best_estimator_))

clf  = cv.best_estimator_

test_classifier(clf , my_dataset , features_list)

# Pipeline(steps=[('feature_selection', SelectKBest(k=19, score_func=<function f_c
# lassif at 0x00000000167C0D68>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', ExtraTreesClassifier(bootstrap=False, class_weight=No
# ne, criterion='gini',
#            max_depth=None, max_features='auto',...stimators=5, n_jobs=1, oob_sco
# re=False, random_state=None,
#            verbose=0, warm_start=False))])
#         Accuracy: 0.84493       Precision: 0.36551      Recall: 0.22150 F1: 0.27
# 584     F2: 0.24045
#         Total predictions: 15000        True positives:  443    False positives:
#   769   False negatives: 1557   True negatives: 12231


# # With - PCA
# Pipeline(steps=[('feature_selection', SelectKBest(k='all', score_func=<function
# f_classif at 0x000000001680ED68>)), ('scaler', MinMaxScaler(copy=True, feature_r
# ange=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=14, ra
# ndom_state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classif...stimators=5, n_jobs=1,
# oob_score=False, random_state=None,
#            verbose=0, warm_start=False))])
#         Accuracy: 0.84860       Precision: 0.33850      Recall: 0.14200 F1: 0.20
# 007     F2: 0.16065
#         Total predictions: 15000        True positives:  284    False positives:
#   555   False negatives: 1716   True negatives: 12445

