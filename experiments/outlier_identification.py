import sys
import pickle
import os
os.chdir('C:/Users/Anirudh/Desktop/ud120-projects/final_project/')
sys.path.append("../tools/")
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict


with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)    
my_dataset = data_dict

del my_dataset['TOTAL']

outlier = defaultdict()

total_nans_percent = 0.0

for i in my_dataset:
	for features in my_dataset[i]:
		if(features != 'poi'):
			if(my_dataset[i][features] == 'NaN'):
				total_nans_percent += 1
	outlier[i] = ((total_nans_percent / 20.) * 100)
	total_nans_percent = 0

for i in outlier :
	if(outlier[i] > 80.0):
		#print(i , outlier[i])
		pass

for i in my_dataset: 
	for j in my_dataset[i]:
		print(j , my_dataset[i][j])
	break


# ('WODRASKA JOHN', 85.0)
# ('WHALEY DAVID A', 90.0)
# ('CLINE KENNETH W', 85.0)
# ('WAKEHAM JOHN', 85.0)
# ('WROBEL BRUCE', 90.0)
# ('GILLIS JOHN', 85.0)
# ('LOCKHART EUGENE E', 100.0)
# ('THE TRAVEL AGENCY IN THE PARK', 90.0)
# ('SCRIMSHAW MATTHEW', 85.0)
# ('SAVAGE FRANK', 85.0)
# ('GRAMM WENDY L', 90.0)