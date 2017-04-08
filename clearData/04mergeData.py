import dask.dataframe as dd
import numpy as np
import pandas as pd
import os
control = 'merge_test'
chunksize = 50000
if control == 'merge_train':
	train = pd.read_csv('./data/feature_engineering/train.csv', iterator = True, chunksize = chunksize)
	events = pd.read_csv('./data/one_hot_input/transfered_one_hot_events.csv')
	count = 0
	for chunk in train:
		df = chunk.merge(events, on = ['display_id'])
		filename = './data/merge_train1/train%d.csv' % count
		df.to_csv(filename, index = False)
		count += 1
elif control == 'merge_doc':
	meta = pd.read_csv('./data/one_hot_input/transfered_one_hot_meta.csv', usecols = ['document_id', 'source_id', 'publisher_id'])
	topics = pd.read_csv('./data/one_hot_input/transfered_one_hot_topics.csv', usecols = ['document_id', 'topic_id'])
	categories = pd.read_csv('./data/one_hot_input/transfered_one_hot_categories.csv', usecols = ['document_id', 'category_id'])
	entities = pd.read_csv('./data/one_hot_input/transfered_one_hot_entities.csv', usecols = ['document_id', 'entity_id'])
	df = meta.merge(topics, on = ['document_id']).merge(categories, on = ['document_id']).merge(entities, on = ['document_id'])
	df.to_csv('./data/merge_doc/doc.csv', index = False)
elif control == 'merge_train_ad':
	top = './data/merge_train1'
	count = 0
	df = pd.read_csv('./data/one_hot_input/transfered_one_hot_content.csv', usecols = ['ad_id', 'campaign_id', 'advertiser_id'])
	for dirpath, dir, files in os.walk(top):
		for file in files:
			#print file
			df_t = pd.read_csv(top + '/' + file).merge(df, on = ['ad_id'])
			filename = './data/merge_train2/train%d.csv' % count
			df_t.to_csv(filename, index = False)
			count += 1
elif control == 'merge_train_ad_doc':
	top = './data/merge_train2'
	count = 0
	df = pd.read_csv('./data/merge_doc/doc.csv')
	for dirpath, dir, files in os.walk(top):
		for file in files:
			#print file
			df_t = pd.read_csv(top + '/' + file).merge(df, on = ['document_id'])
			filename = './data/merge_train3/train%d.csv' % count
			df_t.to_csv(filename, index = False)
			count += 1
elif control == 'merge_test':
	test = pd.read_csv('./data/feature_engineering/test.csv', iterator = True, chunksize = chunksize)
	events = pd.read_csv('./data/one_hot_input/transfered_one_hot_events.csv')
	count = 0
	for chunk in test:
		df = chunk.merge(events, on = ['display_id'])
		filename = './data/merge_test1/test%d.csv' % count
		df.to_csv(filename, index = False)
		count += 1
elif control == 'merge_test_ad_doc':
	top = './data/merge_test'
	count = 0
	df = pd.read_csv('./data/merge_doc/doc.csv')
	for dirpath, dir, files in os.walk(top):
		for file in files:
			print file
			df_t = df.merge(pd.read_csv(file))
			filename = './data/merge_test2/test%d.csv' % count
			df_t.to_csv(filename, index = False)
			count += 1			
elif control == 'merge_test_doc_ad':
	top = './data/merge_test'
	count = 0
	df = pd.read_csv('./data/merge_doc/doc.csv')
	for dirpath, dir, files in os.walk(top):
		for file in files:
			print file
			df_t = df.merge(pd.read_csv(file))
			filename = './data/merge_test2/test%d.csv' % count
			df_t.to_csv(filename, index = False)
			count += 1	