import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime

content = pd.read_csv('./input/promoted_content.csv', usecols = ['ad_id', 'document_id', 'campaign_id', 'advertiser_id'])
events = pd.read_csv('./input/events.csv', usecols = ['display_id', 'document_id', 'timestamp', 'platform', 'geo_location'], dtype = str)
meta = pd.read_csv('./input/documents_meta.csv', usecols = ['document_id', 'source_id', 'publisher_id'])
topics = pd.read_csv('./data/selected/topics.csv')
categories = pd.read_csv('./data/selected/categories.csv')
entities = pd.read_csv('./data/selected/entities.csv')
events.fillna('other', inplace = True)

delay = 4 * 60 * 60 * 1e3
my_timestamp = 1465876799998
events['real_time'] = [long(time) + my_timestamp for time in events.timestamp]
events['weekday'] = [datetime.datetime.utcfromtimestamp(time / 1e3).weekday() for time in events.real_time]
events['hour'] = [datetime.datetime.utcfromtimestamp(time / 1e3).hour for time in events.real_time]

events['country'] =[location.split('>')[0] for location in events.geo_location]
events['city'] = [location.split('>')[1] if len(location.split('>')) > 1 else 'other' for location in events.geo_location]
events['dma'] = [location.split('>')[2] if len(location.split('>')) > 2 else 'other' for location in events.geo_location]

remove = ['timestamp', 'geo_location', 'real_time']
events.drop(remove, axis = 1, inplace = True)

def getOneHotEncoding(column):
	enc = LabelEncoder()
	return enc.fit_transform(column), enc.classes_.shape[0] + 1  #enable other

def transfer(num, encoding):  #feautre index starts from 0
	number = 0
	while num > 0:	
		number += count[num - 1]
		num -= 1
	return long(encoding) + number
	
count = []
print 'get one hot encoding and n_classes'
for column in events.columns:
	if column not in ['display_id', 'document_id']:
		temp_column, n_classes = getOneHotEncoding(events[column]) 
		count.append(n_classes)
		events[column] = temp_column
for column in content.columns:
	if column not in ['ad_id', 'document_id']:
		temp_column, n_classes = getOneHotEncoding(content[column]) 
		count.append(n_classes)
		content[column] = temp_column
for column in meta.columns:
	if column not in ['document_id']:
		temp_column, n_classes = getOneHotEncoding(meta[column]) 
		count.append(n_classes)
		meta[column] = temp_column
for column in topics.columns:
	if column not in ['document_id', 'confidence_level_topic']:
		temp_column, n_classes = getOneHotEncoding(topics[column]) 
		count.append(n_classes)
		topics[column] = temp_column
for column in categories.columns:
	if column not in ['document_id', 'confidence_level_category']:
		temp_column, n_classes = getOneHotEncoding(categories[column]) 
		count.append(n_classes)
		categories[column] = temp_column
for column in entities.columns:
	if column not in ['document_id', 'confidence_level_entity']:
		temp_column, n_classes = getOneHotEncoding(entities[column]) 
		count.append(n_classes)
		entities[column] = temp_column
		
print count
print events['platform'].value_counts()
column_list = ['platform', 'weekday', 'hour', 'country', 'city', 'dma', 'campaign_id', 'advertiser_id', 'source_id', 'publisher_id', \
 'topic_id', 'category_id', 'entity_id']
#count = [4L, 7L, 24L, 231L, 400L, 212L, 34675L, 4385L]

print 'transfering'
for (num, column) in enumerate(column_list[0:6]):
	events[column] = ['%d : %d' % (transfer(num, item) , 1) for item in events[column]]	
for (num, column) in enumerate(column_list[6:8]):
	content[column] = ['%d : %d' % (transfer(num + 6, item), 1) for item in content[column]]	
for (num, column) in enumerate(column_list[8:10]):
	meta[column] = ['%d : %d' % (transfer(num + 8, item), 1) for item in meta[column]]
topics['topic_id'] = ['%d : %f' % (transfer(10, item), topics['confidence_level_topic'][index]) for index, item in enumerate(topics['topic'])]	
categories['category_id'] = ['%d: %f' % (transfer(11, item), categories['confidence_level_category'][index]) for index, item in enumerate(categories['category'])]	
entities['entity_id'] = ['%d: %f' % (transfer(12, item), entities['confidence_level_entity'][index]) for index, item in enumerate(entities['entity'])]		
# weird results   but good to use
print events.head()
print content.head()
print meta.head()
print topics.head()
print categories.head()
print entities.head()
events.to_csv('./data/one_hot_input/transfered_one_hot_events.csv', index = False)
content.to_csv('./data/one_hot_input/transfered_one_hot_content.csv', index = False)
meta.to_csv('./data/one_hot_input/transfered_one_hot_meta.csv', index = False)
topics.to_csv('./data/one_hot_input/transfered_one_hot_topics.csv', index = False)
categories.to_csv('./data/one_hot_input/transfered_one_hot_categories.csv', index = False)
entities.to_csv('./data/one_hot_input/transfered_one_hot_entities.csv', index = False)
