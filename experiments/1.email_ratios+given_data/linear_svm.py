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
from sklearn.svm import LinearSVC
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

### Verification
# ______________________________________________

# for i in my_dataset:
# 	print('\n')
# 	for features in my_dataset[i] :
# 		print(features , my_dataset[i][features])
# 	print('\n\n')
# 	break

#______________________________
# end of verification

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
    if(i != 'email_address'):
        features_list.append(i)    

print('\nTotal length finally is : ' , len(features_list))

# features_list.remove('ratio_from_poi')
# features_list.remove('ratio_shared_receipt')

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest

select = SelectKBest()

from sklearn.ensemble import ExtraTreesClassifier

clf = LinearSVC()

scaler = preprocessing.MinMaxScaler()

pca = PCA()

steps = [('feature_selection' , select) ,('scaler' , scaler), ('PCA' , pca) ,  ('classifier' , clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

parameters = parameters = dict(feature_selection__k =  ['all'],  	#10 , 15 , 18 , 20 , 22 , 'all'
                               classifier__loss = ['hinge' , 'squared_hinge'] ,
                               classifier__max_iter = [1000,500,2000] ,
                               classifier__multi_class = ['ovr' , 'crammer_singer'] ,
                               PCA__n_components =  [1],
                               PCA__random_state = [42] ,
                               PCA__whiten = [True , False])

sss = StratifiedShuffleSplit(labels, 5, test_size=0.3, random_state=60)

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


#Linear SVC
# Pipeline(steps=[('feature_selection', SelectKBest(k=15, score_func=<function f_c
# lassif at 0x00000000167220B8>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LinearSVC(C=1.0, class_weight=None, dual=True, fit_in
# tercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0))])
#         Accuracy: 0.86707       Precision: 0.50405      Recall: 0.18650 F1: 0.27
# 226     F2: 0.21339
#         Total predictions: 15000        True positives:  373    False positives:
#   367   False negatives: 1627   True negatives: 12633


#Linear SVC - PCA
# Pipeline(steps=[('feature_selection', SelectKBest(k=12, score_func=<function f_c
# lassif at 0x00000000166E50B8>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=12, rando
# m_state=42,
#   svd_solver='auto', tol=0.0, whiten=True)), ('classifier...ge', max_iter=500, m
# ulti_class='ovr',
#      penalty='l2', random_state=None, tol=0.0001, verbose=0))])
#         Accuracy: 0.87887       Precision: 0.66310      Recall: 0.18600 F1: 0.29
# 051     F2: 0.21726
#         Total predictions: 15000        True positives:  372    False positives:
#   189   False negatives: 1628   True negatives: 12811