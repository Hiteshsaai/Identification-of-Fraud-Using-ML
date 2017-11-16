import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from collections import defaultdict
import sklearn.grid_search
import sklearn.pipeline
import math 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

### Task 1: Select what features you'll use.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
my_dataset = data_dict

### Task 2: Remove outliers

del my_dataset['TOTAL'] 							#Deleting outlier total
del my_dataset['THE TRAVEL AGENCY IN THE PARK'] 	#not corresponding to a person
del my_dataset['LOCKHART EUGENE E']					#has all features missing

### Task 3: Create new feature(s)

finance_features = ['deferral_payments' , 'expenses' , 'deferred_income' , 'restricted_stock_deferred','loan_advances' ,'other' , 'director_fees', 'bonus' ,'restricted_stock' ,'total_stock_value' ,'long_term_incentive','salary','total_payments','exercised_stock_options']
normalized = ['bonus' , 'deferred_income']

def add_new_features(x):	
	"""
	Created three ratios based on email : to,from, and shared with POIs
	"""
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

#Initialize feature with 0.0
for i in my_dataset:
    my_dataset[i]['ratio_to_poi'] = 0.
    my_dataset[i]['ratio_from_poi'] = 0.
    my_dataset[i]['ratio_shared_receipt'] = 0.

#Set value
for i in my_dataset : 
    ratio_to_poi , ratio_from_poi , ratio_shared_receipt = add_new_features(my_dataset[i])
    my_dataset[i]['ratio_to_poi'] = ratio_to_poi
    my_dataset[i]['ratio_from_poi'] = ratio_from_poi
    my_dataset[i]['ratio_shared_receipt'] = ratio_shared_receipt

def log_features(x):
    """	
    Created log of financial features
    """
    d = defaultdict(lambda: 0.)
    d.clear()
    for features in x:
        if(features in finance_features): 
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


features_all = set()
for x in my_dataset : 
    for features in my_dataset[x]:
        features_all.add(features)   
try :
    features_all.remove('poi')
except KeyError:
    pass

features_list = ['poi']

for i in features_all : 		#Removing e-mail address field from classification
    if(i != 'email_address'):
        features_list.append(i)    

### Extract features and labels from dataset for local testing


data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

clf = LinearSVC()
pca = PCA()
scaler = preprocessing.MinMaxScaler()
select = SelectKBest()

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42) 			#extract features

min_max_scaler = preprocessing.MinMaxScaler()

features_train = min_max_scaler.fit_transform(features_train)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

steps = [('feature_selection' , select) ,('scaler' , scaler) , ('PCA' , pca) , ('classifier' , clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

scaler = preprocessing.MinMaxScaler()



parameters = parameters = dict(feature_selection__k= [35],      #18,20,22,25,26,27,29 , 30 , 31 ,33 , 35,
                               classifier__loss = ['hinge' , 'squared_hinge'] ,
                               classifier__max_iter = [1000,500,2000] ,
                               classifier__multi_class = ['ovr' , 'crammer_singer'] ,
                               PCA__n_components =  [19] ,
                               PCA__random_state = [42] ,
                               PCA__whiten = [True , False])

sss = StratifiedShuffleSplit(labels, 10, test_size=0.3, random_state=60)

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid = parameters , scoring = 'f1' ,cv = sss)

cv.fit(features, labels)

clf  = cv.best_estimator_
print(cv.best_estimator_)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

