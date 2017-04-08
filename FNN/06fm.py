import sys
import time
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import os

def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))

def pred_lr(x):
    p = w_0
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return p

def pred(x):
    p = w_0
    sum_1 = 0
    sum_2 = 0
    for (feat, val) in x:
        tmp = v[feat] * val
        sum_1 += tmp
        sum_2 += tmp * tmp
    p = np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return (p, sum_1)

def one_data_y_x(line):
    s = line.strip().replace(':', ' ').split(' ')
    y = int(s[0])
    x = []
    for i in range(1, len(s), 2):
        val = 1
        if not one_value:
            val = float(s[i+1])
        x.append((int(s[i]), val))
    return (y, x)

def getX(line):
	x = []
	for item in line:
		s = item.split(':')
		x.append((float(s[0]), float(s[1])))
	return x
	
def output_model(model_file):
    print 'output model to ' + model_file
    foo = open(model_file, 'w')
    foo.write('%.5f %d %d\n' % (w_0, feature_num, k))
    for i in range(feature_num):
        foo.write('%d %.5f' % (i, w[i]))
        for j in range(k):
            foo.write(' %.5f' % v[i][j])
        foo.write(' %s\n' % index_feature[i])
    foo.close()

def load_model(model_file):
    global feature_num, k, w_0, w, v, index_feature, feature_index
    print 'loading model from ' + model_file
    fi = open(model_file, 'r')
    line_num = 0
    for line in fi:
        line_num += 1
        s = line.strip().split()
        if line_num == 1:
            w_0 = float(s[0])
            feature_num = int(s[1])
            k = int(s[2])
            v = np.zeros((feature_num, k))
            w = np.zeros(feature_num)
            index_feature = {}
            feature_index = {}
        else:
            i = int(s[0])
            w[i] = float(s[1])
            for j in range(2, 2 + k):
                v[i][j] = float(s[j])
            feature = s[2 + k]
            index_feature[i] = feature
            feature_index[feature] = i
    fi.close()

# start here
# global setting
np.random.seed(10)
one_value = True
k = 10
learning_rate = 0.01
weight_decay = 1E-6
v_weight_decay = 1E-6
nrows = 100000

# initialise
feature_index = {}
index_feature = {}
max_feature_index = 0
feature_num = 0
print 'reading feature index'
feature_index_file_name = './feature_index/feature_index.csv'
feature_index_data = pd.read_csv(feature_index_file_name, header = None)
#print feature_index_data.head()
for row in xrange(feature_index_data.shape[0]):
    index = int(feature_index_data.iloc[row, 1])
    feature_index[feature_index_data.iloc[row, 0]] = index
    index_feature[index] = feature_index_data.iloc[row, 0]
    max_feature_index = max(max_feature_index, index)
feature_num = max_feature_index 
#print index_feature
print 'feature number: ' + str(feature_num)
print 'initialising'
init_weight = 0.05
v = (np.random.rand(feature_num, k) - 0.5) * init_weight
w = np.zeros(feature_num)
w_0 = 0
top = './data/merge_train3'
best_auc = 0.
overfitting = False
model_file_name = './output/fm/fm.txt'
train_log_file_name = './output/fm/train_log.txt'
fo = open(train_log_file_name, 'w')
round = 0
time_reduce_number = 3
time_reduce = 0
for dirpath, dirnames, files in os.walk(top, topdown = True):
	test_list = [files[i] for i in np.random.choice(len(files), int(0.2 * len(files)), replace = False)]
	train_list = [train_file_name for train_file_name in files if train_file_name not in test_list]
	# train
	#print dirpath   #'./input/data/train'
	#print dirnames  # []
	print test_list
	print train_list	
	for file_name in train_list:
		train_file_name = os.path.join(dirpath, file_name)
		train_data = pd.read_csv(train_file_name, usecols = ['clicked', 'platform', 'weekday', 'hour', 'country', 'city', 'dma', 'campaign_id', \
			'advertiser_id', 'source_id', 'publisher_id', 'topic_id', 'category_id', 'entity_id'])
		print 'training: %s' % train_file_name
		x_train = train_data.iloc[:,1:]
		y_train = train_data.iloc[:, 0]			
		start_time = time.time()
		round += 1
		for i in xrange(y_train.shape[0]):
		# train one data
			y = y_train[i]
			x = getX(x_train.iloc[i,:])
			(p, vsum) = pred(x)
			d = y - p
			w_0 = w_0 * (1 - weight_decay) + learning_rate * d
			for (feat, val) in x:
				w[feat] = w[feat] * (1 - weight_decay) + learning_rate * d * val
			for (feat, val) in x:
				v[feat] = v[feat] * (1 - v_weight_decay) + learning_rate * d * (val * vsum - v[feat] * val * val)
					
		train_time = time.time() - start_time
		train_min = int(train_time / 60)
		train_sec = int(train_time % 60)
		#print w
		#print v
		# test for this round
		y = []
		yp = []
		random_number = np.random.randint(len(test_list))
		test_file_name = './data/merge_train3/%s' % test_list[random_number]
		print 'using %s to do the test' % test_file_name
		test_data = pd.read_csv(test_file_name, usecols = ['clicked', 'platform', 'weekday', 'hour', 'country', 'city', 'dma', 'campaign_id', \
			'advertiser_id', 'source_id', 'publisher_id', 'topic_id', 'category_id', 'entity_id'])
		for row in xrange(test_data.shape[0]):
			clk = test_data.iloc[row, 0]
			x = test_data.iloc[row, 1:]
			x = getX(x)
			pclk = pred(x)[0]
			y.append(clk)
			yp.append(pclk)
		auc = roc_auc_score(y, yp)
		rmse = math.sqrt(mean_squared_error(y, yp))
		print '%d\t%.8f\t%.8f\t%dm%ds' % (round, auc, rmse, train_min, train_sec)
		fo.write('%d\t%.8f\t%.8f\t%dm%ds\n' % (round, auc, rmse, train_min, train_sec))
		fo.flush()
		if auc > best_auc:
			best_auc = auc
			if time_reduce < time_reduce_number:
				time_reduce += 1
		else:
			time_reduce -= 1
		if time_reduce < 0:
			print 'output model into ' + model_file_name
			output_model(model_file_name)
			break
fo.close()