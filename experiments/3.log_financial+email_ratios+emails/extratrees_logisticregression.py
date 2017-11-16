import sys
import pickle
import os
sys.path.append("../../tools/")
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
import sklearn.pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
import sklearn.grid_search
from sklearn.linear_model import LogisticRegression
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

finance_features = ['deferal_payments' , 'expenses' , 'deferred_income' , 'restricted_stock_deferred','loan_advances' ,'other' , 'director_fees', 'bonus' ,'restricted_stock' ,'total_stock_value' ,'long_term_incentive','salary','total_payments','exercised_stock_options']

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


del my_dataset['TOTAL']
del my_dataset['THE TRAVEL AGENCY IN THE PARK']
del my_dataset['LOCKHART EUGENE E']

def add_new_features(x):

	if(x['from_messages']!='NaN' and x['from_this_person_to_poi']!='NaN'):
		ratio_from_x_to_poi = x['from_this_person_to_poi'] / float(x['from_messages'])
	else:
		ratio_from_x_to_poi = 0.

	if(x['to_messages']!='NaN' or x['from_poi_to_this_person']!='NaN'):
		ratio_from_poi_to_x = x['from_poi_to_this_person'] / float(x['to_messages'])
	else:
		ratio_from_poi_to_x = 0.		

	if(x['to_messages']!='NaN' and x['shared_receipt_with_poi']!='NaN'):
		ratio_shared_receipt_with_poi = x['shared_receipt_with_poi'] / float(x['to_messages'])
	else:
		ratio_shared_receipt_with_poi = 0.

	return ratio_from_x_to_poi , ratio_from_poi_to_x , ratio_shared_receipt_with_poi

for i in my_dataset:
	my_dataset[i]['ratio_to_poi'] = 0.
	my_dataset[i]['ratio_from_poi'] = 0.
	my_dataset[i]['ratio_shared_receipt'] = 0.

for i in my_dataset : 
	ratio_to_poi , ratio_from_poi , ratio_shared_receipt = add_new_features(my_dataset[i])
	my_dataset[i]['ratio_to_poi'] = ratio_to_poi
	my_dataset[i]['ratio_from_poi'] = ratio_from_poi
	my_dataset[i]['ratio_shared_receipt'] = ratio_shared_receipt

### Verification - print
# ______________________________________________

# for i in my_dataset:
# 	print('\n')
# 	for features in my_dataset[i] :
# 		print(features , my_dataset[i][features])
# 	print('\n\n')
# 	break

#______________________________
# end of verification


def log_features(x) :

    d = defaultdict(lambda : 0.)
    d.clear()
    for features in x:
        if(features in finance_features) : 
            if(x[features] != 'NaN'):
                if(x[features]!=0) :
                    d['log_' + str(features)] = math.log(abs(x[features]) , 10)
            else:
                d['log_' + str(features)] = 0
    return d

for i in my_dataset : 
    d = log_features(my_dataset[i])
    for features in d:
        my_dataset[i][features] = d[features]

features_list = ['poi']

my_dataset = data_dict
features_all = set()

for x in my_dataset : 
    for features in my_dataset[x]:
        features_all.add(features)
try :
    features_all.remove('poi')
except KeyError:
    pass


features_list = ['poi']

for i in features_all : 
    if i!='email_address':    
        features_list.append(i)


for i in features_list :
    print(i)


print('\n Total feature size : ' , len(features_list))

# # features_list.remove('ratio_from_poi')
# # features_list.remove('ratio_shared_receipt')

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest

select = SelectKBest()

from sklearn.ensemble import ExtraTreesClassifier

clf = LogisticRegression()

scaler = preprocessing.MinMaxScaler()

pca = PCA()

steps = [('feature_selection' , select) , ('scaler' , scaler) , ('PCA' , pca) , ('classifier' , clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

parameters = parameters = dict(feature_selection__k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,'all'], 
                               # classifier__n_estimators = [10 , 15 , 5] ,
                               # classifier__criterion = ['gini' , 'entropy']  ,
                               # classifier__min_samples_split = [2 , 3 , 4 , 5 , 10] ,
                               classifier__penalty = ['l2'] ,
                               classifier__max_iter = [100,200,500,1000 , 10000 , 100000] ,
                               classifier__solver = ['newton-cg','sag','lbfgs'] ,
                               classifier__dual = [True] ,
                               classifier__fit_intercept = [True , False] ,
                               PCA__random_state = [42] ,
                               PCA__n_components = [35],
                               PCA__whiten = [True , False] )


sss = StratifiedShuffleSplit(labels, 6, test_size=0.3, random_state=60)

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid = parameters , scoring = 'f1' ,cv = sss)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

cv.fit(features, labels)

X_new = cv.best_estimator_.named_steps['feature_selection']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in X_new.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print
print ' '
print 'Not Scaled - Selected Features, Scores, P-Values'
for i in features_selected_tuple :
	print(i)



#y = cv.predict(features_test)

#print(str('\n Best estimators : \n')+ str(cv.best_estimator_))

clf  = cv.best_estimator_

print('\nBest features : \n' + str(clf) + '\n\n')

test_classifier(clf , my_dataset , features_list)

#Extratrees --->
# Pipeline(steps=[('feature_selection', SelectKBest(k=17, score_func=<function f_c
# lassif at 0x000000001635BC18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', ExtraTreesClassifier(bootstrap=False, class_weight=No
# ne, criterion='entropy',
#            max_depth=None, max_features='aut...stimators=5, n_jobs=1, oob_score=
# False, random_state=None,
#            verbose=0, warm_start=False))])
#         Accuracy: 0.85347       Precision: 0.41524      Recall: 0.24250 F1: 0.30
# 619     F2: 0.26451
#         Total predictions: 15000        True positives:  485    False positives:
#   683   False negatives: 1515   True negatives: 12317

#Libinear - with dual
# Pipeline(steps=[('feature_selection', SelectKBest(k=24, score_func=<function f_c
# lassif at 0x0000000016760C18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Tru
# e, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.88020       Precision: 0.71828      Recall: 0.16700 F1: 0.27
# 099     F2: 0.19728
#         Total predictions: 15000        True positives:  334    False positives:
#   131   False negatives: 1666   True negatives: 12869

#Liblinear - without dual
# Pipeline(steps=[('feature_selection', SelectKBest(k=34, score_func=<function f_c
# lassif at 0x00000000166A0C18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Fal
# se, fit_intercept=False,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.87720       Precision: 0.63958      Recall: 0.18100 F1: 0.28
# 215     F2: 0.21130
#         Total predictions: 15000        True positives:  362    False positives:
#   204   False negatives: 1638   True negatives: 12796

#newton - cg - 26
# Pipeline(steps=[('feature_selection', SelectKBest(k=24, score_func=<function f_c
# lassif at 0x000000001631EC18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Fal
# se, fit_intercept=False,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.87767       Precision: 0.64007      Recall: 0.18850 F1: 0.29
# 123     F2: 0.21947
#         Total predictions: 15000        True positives:  377    False positives:
#   212   False negatives: 1623   True negatives: 12788

#POI
# Pipeline(steps=[('feature_selection', SelectKBest(k=28, score_func=<function f_c
# lassif at 0x000000001664CC18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=2, random
# _state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier'...timators=15, n_jobs
# =1, oob_score=False, random_state=None,
#            verbose=0, warm_start=False))])
#         Accuracy: 0.86640       Precision: 0.49844      Recall: 0.32050 F1: 0.39
# 014     F2: 0.34514
#         Total predictions: 15000        True positives:  641    False positives:
#   645   False negatives: 1359   True negatives: 12355




#Liblinear -wthout dual

# Pipeline(steps=[('feature_selection', SelectKBest(k=23, score_func=<function f_c
# lassif at 0x0000000016789C18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=23, rando
# m_state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier...ty='l1', random_stat
# e=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.74920       Precision: 0.32195      Recall: 0.79650 F1: 0.45
# 855     F2: 0.61515
#         Total predictions: 15000        True positives: 1593    False positives:
#  3355   False negatives:  407   True negatives: 9645


# Pipeline(steps=[('feature_selection', SelectKBest(k=33, score_func=<function f_c
# lassif at 0x00000000167B5C18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=31, rando
# m_state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier...ty='l2', random_stat
# e=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.84853       Precision: 0.41574      Recall: 0.33550 F1: 0.37
# 133     F2: 0.34897
#         Total predictions: 15000        True positives:  671    False positives:
#   943   False negatives: 1329   True negatives: 12057


#liblinaer - with dual

# Pipeline(steps=[('feature_selection', SelectKBest(k=22, score_func=<function f_c
# lassif at 0x000000001674FC18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=22, rando
# m_state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier...ty='l2', random_stat
# e=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.75333       Precision: 0.32336      Recall: 0.77800 F1: 0.45
# 684     F2: 0.60724
#         Total predictions: 15000        True positives: 1556    False positives:
#  3256   False negatives:  444   True negatives: 9744

# Pipeline(steps=[('feature_selection', SelectKBest(k=35, score_func=<function f_c
# lassif at 0x00000000166BAC18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=35, rando
# m_state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier...ty='l2', random_stat
# e=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.84827       Precision: 0.42005      Recall: 0.36250 F1: 0.38
# 916     F2: 0.37271
#         Total predictions: 15000        True positives:  725    False positives:
#  1001   False negatives: 1275   True negatives: 11999

#Newton-cg

# Pipeline(steps=[('feature_selection', SelectKBest(k=17, score_func=<function f_c
# lassif at 0x000000001640EC18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Fal
# se, fit_intercept=False,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.86713       Precision: 0.50657      Recall: 0.13500 F1: 0.21
# 319     F2: 0.15821
#         Total predictions: 15000        True positives:  270    False positives:
#   263   False negatives: 1730   True negatives: 12737
# Pipeline(steps=[('feature_selection', SelectKBest(k=27, score_func=<function f_c
# lassif at 0x000000001677AC18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=8, random
# _state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier'...ty='l2', random_sta
# te=None, solver='newton-cg', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.86880       Precision: 0.52649      Recall: 0.15900 F1: 0.24
# 424     F2: 0.18480
#         Total predictions: 15000        True positives:  318    False positives:
#   286   False negatives: 1682   True negatives: 12714

# Pipeline(steps=[('feature_selection', SelectKBest(k=29, score_func=<function f_c
# lassif at 0x0000000016747C18>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=2, random
# _state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier'...timators=15, n_jobs
# =1, oob_score=False, random_state=None,
#            verbose=0, warm_start=False))])
#         Accuracy: 0.86233       Precision: 0.47513      Recall: 0.31050 F1: 0.37
# 557     F2: 0.33362
#         Total predictions: 15000        True positives:  621    False positives:
#   686   False negatives: 1379   True negatives: 12314