# Identification of fraud using ML - Enron dataset


### Brief : 
The goal of this project is to use ML algorithms from the python-sk-learn library and identify Persons of Interest (POIs) in the given enron dataset.

The dataset has **146 datapoint with 21 features** extracted from emails. To pass this particular project, I need a precision and recall greater than or equal to **0.3**.

### Steps Involved  :
1. Understanding and cleaning dataset.
2. Removing outliers
3. Optimizing features and feature selection
4. Feature engineering
5. Scaling the features
6. Selecting algorithm
7. Parameter tuning.

### Algorithms used :
Since this is a binary classification problem i.e is a given person a *Person of interest* or not, I decided to use:
1. [Extra Trees Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
2. [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
3. [Linear-SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

For **Feature selection** I used both Tree-based feature selection and Select-K-Best. I ended up using Select-K-Best since that gave the best precision and accuracy for given number of features.

### Parameters tuned :
1. Extra Trees classifier :

* n_estimators - [5 , 10 , 15]
* criterion = ['gini' , 'entropy']
* min_sample_split = [2 , 3 , 4 , 5 , 10 ]

2. Logistic Regression :

* max_iter = [100, 200 , 300 , 500 , 1000 , 10000]
* penalty = ['l1' , 'l2'] (liblinear uses both l1 and l2 , liblinear with dual uses only l2 and newton-cg , sags , lbfgs use only l2).
* solver = ['liblinear' , 'newton-cg' , 'sag', 'lbfgs'].
* fit_intercept : [True , False]

3. Linear-SVM :

* loss = ['hinge','squared_hinge']
* max_iter = [100,200,500,1000,10000]
* tolerance = [1e-2 , 1e-4 , 1e-6 , 1e-8 , 1e-10],
* multi_class = ['ovr' , 'crammer_singer'] 

All of the above algorithms and parameters were tried once with and without - [Principle Component analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

For parameter tuning I used : 
1. [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
2. [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

### Validating metrics : 
For validating the results , I used many methods from [Cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html). In the end I decided to use [Stratified shuffle split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) to split the data, Since the POIs and Non-POIs are unevenly distributed, and I wanted a good classification with the training and testing dataset split(30% split for testing). Few others I tried were : Shuffle split, k-1 split, and simple test-train split.


---------------
**Full report  : report.pdf**   
**Results of various tests : Enron Dataset - ML results.pdf**
------

### Additional mini-project : Text based mining.

I downloaded the body of the emails from the Enron dataset and performed text-based classification on the emails using [Count-Vectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) as well as [TfIdf transformer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html). I got an accuracy of 50% when the dataset had equal amount of POIs and Non-POIs. When the ratio of POIs to Non-POIs was 1:3 or close , I got 25% accuracy. Anything beyond 30 points in dataset failed to converge and produced 0% accuracy. This is because the sampling of data I used has very skewed distribution of POIs and Non-POIs.
Refer : *get_poi_names.py*
