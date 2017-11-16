import os
import sys
from pandas import DataFrame
import numpy as np
import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import tree

os.path.realpath(__file__)

y = 'yes'
n = 'no'

SOURCES = {
	# ('albert_meyers' , n) , 
	# ('andrea_ring', n) , 
	# ('andrew_lewis' , n) ,
	# ('andy_zipper' , n) ,
	# ('barry_tycholiz' , n),
	# ('benjamin_rogers' , n),
	# ('bill_rapp' , n) ,
	# ('bradley_mckay' , n) ,
	# ('cara_semperger'  , n) ,
	# ('carol_stclair' , n) , 
	# ('charles_weldon'  , n),
	# ('chris_dorland' , n),
	# ('chris_germany' , n) ,
	# ('cooper_richey' , n) ,
	# ('craig_dean' , n) ,
	# ('dan_hyvl' , n) ,
	# ('dana_davis' , n) ,
	# ('danny_mccarty' , n) ,
	('darell_schoolcraft' , n) ,
	('daren_farmer' , n), 
	('daron_giron' , n),
	('david_delainey' , y),
	# ('debra_perlingiere' , n), 
	# ('diana_scholtes' , n) ,
	# ('don_baughman' , n ) ,
	# ('douglas_gilberth-smith' , n),
	# ('drew_fossum' , n) ,
	# ('dutch_quigley' , n),
	# # ('elizabeth_sager' , n),
	# ('eric_bass' , n),
	# ('eric_linder' , n) ,
	# ('eric_saibi' , n),
	# ('errol_lclaughlin' , n) ,
	# ('fletcher_sturm' , n) ,
	# ('frank_ermis' , n),
	# ('geir_solberg' , n) ,
	# ('geoffery_storey' , n),
	# ('gerald_nemec' , n),
	# ('harpreet_arora' , n) ,
	# ('holden_salisbury' , n),
	# ('hunter_shivley' , n),
	# ('james_derrick' , n),
	# ('james_steffes' , n),
	# ('jane_tholt' , n) ,
	# ('jason_williams' , n),
	# ('jason_wolfe' , n),
	# ('jay_reitmeyer' , n),
	# ('jeff_dasovich' , n) ,
	# ('jeff_king' , n),
	('jeffery_skilling' ,y ),
	# ('jeffrey_shankman',  n) ,
	# ('jim_schwieger' , n),
	# ('joe_parks' , n),
	# ('joe_quenet' , n) ,
	# ('joe_stepenovitch' , n) ,
	# ('john_arnold' , n),
	# ('john_forney' , n),
	# ('john_giffith' , n),
	# ('john_hodge' , n),
	# ('john_lavorato' , n),
	# ('john_zufferli' , n),
	# ('jonathan_mckay' , n) ,
	# ('juan_hernandez' , n),
	# ('judy_townsend' , n),
	# ('kam_kaiser' , n),
	('keith_holst' , n),
	 ('kenneth_lay' , y),
	#  ('kevin_hyatt' , n) ,
	# ('kevin_presto' , n),
	#  ('kevin_ruscitti',n),
	# ('kim_ward' , n),
	# ('kimberely_watson',  n),
	# ('larry_campbell' , n) ,
	# ('lawrence_greg-whalley' , n) ,
	# ('lawrence_may' , n),
	# ('lindy_donoho' , n),
	# ('lisa_gang' , n),
	# ('louise_kitchen' , n),
	# ('lynn_blair' , n),
	# ('marie_heard' , n),
	# ('mark_haedicke' , n),
	# ('mark_taylor' , n),
	# ('mark_whitt' , n),
	# ('martin_cuilla' , n),
	# ('mary_fischer' , n),
	# ('matt_smith' , n),
	# ('matthew_lenhart', n),
	# ('matthew_motley' , n),
	# ('michael_grigsby' , n),
	# ('michael_maggi',n),
	# ('michelle_cash' , n),
	# ('michelle_lokay' , n),
	# ('mike_carson',n),
	# ('mike_mcconnell', n),
	# ('mike_swerzbin',  n),
	# ('monika_causholi', n),
	# ('monique_sanchez',n) ,
	# ('partice_mims-thurston',n),
	# ('paul_lucci', n),
	# ('paul_thomas' , n),
	# ('paul_ybarbo' , n),
	# ('peter_keavey' , n) ,
	# ('philip_allen',  n),
	# ('phillip_love' , n),
	# ('phillip_platter' , n),
	# ('randall_gay' , n),
	# ('richard_ring' , n),
	# ('richard_sanders' , n),
	# ('richard_shapiro' , n),
	# ('rick_buy' , n),
	# ('robert_badeer' , n),
	# ('robert_benson' , n),
	# ('rod_hayslett' , n),
	# ('ryan_slinger' , n),
	# ('sally_beck' , n),
	# ('sandra_brawner' , n),
	# ('sara_shackleton' , n),
	# ('scott_hendrickson' , n),
	# ('scott_neal' , n),
	# ('sean_crandall' , n),
	# ('shelley_corman' , n),
	# ('stacey_white' , n),
	# ('stanley_horton', n),
	# ('stephanie_panus' , n),
	# ('steven_kean' , n),
	# ('susan_bailey' , n),
	# ('susan_pereira',  n),
	# ('susan_scott' , n),
	# ('tana_jones' , n),
	# ('teb_lokey' , n),
	# ('theresa_staab' , n),
	# ('thomas_martin' , n),
	# ('tory_kuykendall' , n),
	# ('tracy_geaccone' , n),
	# ('vince_kaminsi' , n),
	# ('vladi_pimenov' , n),
	('william_williams-III', n)
}

def remove_blanks(text):
	text = re.sub(' +\s\n+' , ' ' , text)
	text = re.sub(r'\n\s*\n', ' ', text)
	#text = re.sub('[^a-zA-Z0-9-_*.]', '', text)
	return str(text.split('--')[0]).strip()

def return_text(x):

	total_text_content = ''
	with open(x , 'rb') as csvfile:
		reader = csv.DictReader(csvfile)
		try:
			for row in reader : 
				subject , body = row['Subject'] , remove_blanks(row['Body'])
				total_text_content += subject + str(' ') + body
		except csv.Error:
			print('error in file - ' , x)
		
	return str(total_text_content)

def read_files(path):
	file_path =  'enron/' + str(path) + '/'
	inbox = file_path + 'inbox.CSV'
	deleted = file_path + 'deleted.CSV'
	sent = file_path + 'sent.CSV'
	
	text = ''

	if(not os.path.isfile(inbox)):
		print('inbox not in ' , inbox)
	else:
		text += return_text(inbox)

	if(not os.path.isfile(deleted)):
		print('deleted not in ' , path)
	else:
		text += return_text(deleted)

	if(not os.path.isfile(sent)):
		print('sent not in ' , path)
	else:
		text += return_text(sent)


	return path , text

def build_data_frame(path, classification):
    rows = []
    index = []
    file_name, text =  read_files(path)
    rows.append({'text': text, 'class': classification})
    index.append(file_name)

    data_frame = DataFrame(rows, index=index)    
    return data_frame

d = {'text': [], 'class': []}
data = DataFrame(d)

for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(np.random.permutation(data.index))

#print(data)

# tfidf_vectorizer = TfidfVectorizer(stop_words = 'english' , decode_error = 'ignore' )
# counts = tfidf_vectorizer.fit_transform(data['text'].values)

# #print(counts)

# classifier = MultinomialNB()
# targets = data['class'].values
# classifier.fit(counts, targets)
	
# test = ['Steve Kean and I believe quite strongly, within a four peer group structure,  that we need to allocate somewhere between 30 and 50 percent of Govt./ public  Affairs out of commercial support into specialized technical- there is a clear divide between individuals in the Govt/public  Affairs groups  who are managing significant risk for the company and having a significant and  direct impact on commercial activities/ net income( these individuals should be moved to specialized technical)....and others who are performing a less strategic function( who should remain in the commercial support category). With that said, it is probably not necessary to undertake this effort for  mid-year PRC purposes since we are probably moving to a three peer stucture where that split could be achieved (for year-end purposes where the compensation impact would be felt). If we stay with the four group stucture, I intend to push hard for the remapping of Govt Affairs by year-end.    On a completely seperate issue, I continue to be  very concerned about the impact of the "preferred" distribution  relative to the bottom category for " commercial support" groups(Im not implying this isnt an issue for commercial groups- I just dont have the same ability to assess the impact there)....the concern, simply stated,  is that  it may be entirely  counterproductive to force individuals who are not performance problems  into a category that has significant implications for their careers here at Enron, particularly after going thru the past year where many of the non-performers in my group and others  were moved out']

# print(classifier.predict(tfidf_vectorizer.transform(test)))

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2) , decode_error = 'ignore' , stop_words = 'english')),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',         MultinomialNB())
])


#X = []

# for x in range(1,138):
# 	X.append(x)

# y = data['class'].values.astype(str)

# print('Length of x : ' + str(len(X)) ,'Length of y : ' + str(len(y)))

k_fold = KFold(n=len(data), n_folds=4)
scores = []
confusion = np.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label= y)
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)


# http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html