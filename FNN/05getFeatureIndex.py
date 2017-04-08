import pandas as pd
content = pd.read_csv('./data/one_hot_input/transfered_one_hot_content.csv')
events = pd.read_csv('./data/one_hot_input/transfered_one_hot_events.csv')
meta = pd.read_csv('./data/one_hot_input/transfered_one_hot_meta.csv', usecols = ['document_id', 'source_id', 'publisher_id'])
topics = pd.read_csv('./data/one_hot_input/transfered_one_hot_topics.csv', usecols = ['document_id', 'topic_id'])
categories = pd.read_csv('./data/one_hot_input/transfered_one_hot_categories.csv', usecols = ['document_id', 'category_id'])
entities = pd.read_csv('./data/one_hot_input/transfered_one_hot_entities.csv', usecols = ['document_id', 'entity_id'])
feature_index = []

print events.head()
print content.head()
print meta.head()
print topics.head()
print categories.head()
print entities.head()

for column in events:
	if column not in ['display_id', 'document_id']:
		temp_column = events[column].unique()
		#print temp_column
		for item in temp_column:
			feature_index.append([column, item.split(':')[0].strip()])
		num_column = [int(item.split(':')[0].strip()) for item in temp_column]
		#print feature_index
		#print num_column
		feature_index.append([column, str(max(num_column) + 1)])
		
for column in content:
	if column not in ['ad_id', 'document_id']:
		temp_column = content[column].unique()
		#print temp_column
		for item in temp_column:
			feature_index.append([column, item.split(':')[0].strip()])
		num_column = [int(item.split(':')[0].strip()) for item in temp_column]
		#print feature_index
		#print num_column
		feature_index.append([column, str(max(num_column) + 1)])
			
for column in meta:
	if column not in ['document_id']:
		temp_column = meta[column].unique()
		#print temp_column
		for item in temp_column:
			feature_index.append([column, item.split(':')[0].strip()])
		num_column = [int(item.split(':')[0].strip()) for item in temp_column]
		#print feature_index
		#print num_column
		feature_index.append([column, str(max(num_column) + 1)])
			
for column in topics:
	if column not in ['document_id']:
		temp_column = topics[column].unique()
		#print temp_column
		for item in temp_column:
			feature_index.append([column, item.split(':')[0].strip()])
		num_column = [int(item.split(':')[0].strip()) for item in temp_column]
		#print feature_index
		#print num_column
		feature_index.append([column, str(max(num_column) + 1)])
			
for column in categories:
	if column not in ['document_id']:
		temp_column = categories[column].unique()
		#print temp_column
		for item in temp_column:
			feature_index.append([column, item.split(':')[0].strip()])
		num_column = [int(item.split(':')[0].strip()) for item in temp_column]
		#print feature_index
		#print num_column
		feature_index.append([column, str(max(num_column) + 1)])

for column in entities:
	if column not in ['document_id']:
		temp_column = entities[column].unique()
		#print temp_column
		for item in temp_column:
			feature_index.append([column, item.split(':')[0].strip()])	
		num_column = [int(item.split(':')[0].strip()) for item in temp_column]
		#print feature_index
		#print num_column
		feature_index.append([column, str(max(num_column) + 1)])
			
#print feature_index
df_feature_index = pd.DataFrame(feature_index)
df_feature_index.to_csv('./feature_index/feature_index.csv', index = False, header = None)