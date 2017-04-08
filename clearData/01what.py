import pandas as pd
to_be_processed = ['./input/documents_topics.csv', './input/documents_categories.csv', './input/documents_entities.csv']
filename = ['./data/selected/topics.csv', './data/selected/categories.csv', './data/selected/entities.csv']
column_names = ['topic_id', 'category_id', 'entity_id']
new_column_names = ['topic', 'category', 'entity']
i = 2    # change i to select different files to process
my_df = pd.read_csv(to_be_processed[i])

grouped = my_df.groupby(my_df['document_id'])
new_column = []
new_confidence_level = []
#print grouped.groups
for group in grouped.groups.values():
	new_column.append(my_df[column_names[i]][group[0]])
	new_confidence_level.append(my_df['confidence_level'][group[0]])
df = pd.DataFrame()
df['document_id'] = grouped.groups.keys()
df[new_column_names[i]] = new_column
df['confidence_level' + new_column_names[i]] = new_confidence_level
df.to_csv(filename[i], index = False)