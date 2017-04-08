import pandas as pd
import csv
#file_name = 'test36.csv'
test_list = ['test33.csv', 'test34.csv', 'test35.csv', 'test36.csv', 'test37.csv', 'test38.csv', \
  'test39.csv', 'test40.csv', 'test41.csv', 'test42.csv', 'test43.csv', 'test44.csv', 'test45.csv', 'test46.csv']

def get_prob(display, ad):
    return click_prob[(display, ad)]

def decode(x):
    return map(int, x.split())

def sort_ad(id, ad_list):
	ad_prob = {ad: get_prob(subm['display_id'][id], ad) for ad in ad_list}
	print ad_prob
	sorted_ad = sorted(ad_list, key = lambda x: ad_prob[x], reverse = True)
	print sorted_ad

def change(x):
	return " ".join(map(str, x.translate(None, '[]').split(',')))
	
click_prob = {}	
top1 = './input/data/test/'
top2 = './output/submission1/'

for file_name in test_list:
	file_name1 = top1 + file_name
	file_name2 = top2 + file_name
	data1 = pd.read_csv(file_name1, usecols = ['display_id', 'ad_id'])
	data2 = pd.read_csv(file_name2, header = None)
	data1['clicked'] = data2
	for i in xrange(data1.shape[0]):
		click_prob[(data1['display_id'][i], data1['ad_id'][i])] = data1['clicked'][i]
#print result	
print 'done'

subm = pd.read_csv("./input/sample_submission.csv")
ad_ids = subm['ad_id'].apply(decode)
sorted_ads = []
#print type(ad_ids)
for i,ad_list in enumerate(ad_ids):
	ad_prob = {ad: click_prob[(subm['display_id'][i], ad)] for ad in ad_list}
	#print ad_prob
	sorted_ad = sorted(ad_list, key = lambda x: ad_prob[x], reverse = True)
	#print sorted_ad
	sorted_ads.append(sorted_ad[:])
subm['ad_id'] = sorted_ads
#subm['ad_id'] = subm['ad_id'].apply(change)
subm.to_csv("./output/to_kaggle/subm_fnn.csv", index=False)
'''
output_file = './output/my/submission%s' % file_name		
with open(output_file, 'wb') as csvfile:
	fieldnames = ['display_id', 'ad_id']
	writer = csv.DictWriter(csvfile, quotechar = '\'', fieldnames = fieldnames)
	writer.writeheader()
	for row in submission_list:
		writer.writerow(row)
'''
