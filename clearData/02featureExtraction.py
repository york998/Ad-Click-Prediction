import pandas as pd
import numpy as np 

train = pd.read_csv("./input/clicks_train.csv")
test = pd.read_csv('./input/clicks_test.csv')
clicked_cnt = train[train.clicked==1].ad_id.value_counts()
cntall_train = train.ad_id.value_counts()
cntall_test = test.ad_id.value_counts()
def get_prob(k):
	return clicked_cnt[k]/(float(cntall_train[k])) if k in clicked_cnt else 0
#train['ad_clicked_prob'] = train.ad_id.apply(lambda x: get_prob(x))
#train['ad_prob'] = train.ad_id.apply(lambda x: cntall_train[x]/(float(train.shape[0])))
#print train.head()
#train.to_csv('./data/feature_engineering/train.csv')
test['ad_clicked_prob'] = test.ad_id.apply(lambda x: get_prob(x))
test['ad_prob'] = test.ad_id.apply(lambda x: test.ad_id.value_counts()[x]/(float(test.shape[0])))
print test.head()
test.to_csv('./data/feature_engineering/test.csv')