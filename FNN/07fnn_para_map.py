import sys
import numpy as np
import time
import theano
import theano.tensor as T
import linecache
import math
from theano.tensor.shared_randomstreams import RandomStreams
import os
import pandas as pd
from average_precision import mapk
import cPickle

def feat_layer_one_index(feat, l):
    return 1 + feat_field[feat] * k + l

def feats_to_layer_one_array(feats, weights_for_feats):
    x = np.zeros(xdim - 2)
    x[0] = w_0
    for i, feat in enumerate(feats):
		#feat = int(feat.split(':')[0])
		x[feat_layer_one_index(feat, 0):feat_layer_one_index(feat, k)] = np.multiply(feat_weights[feat], weights_for_feats[i])
    return x

class parameters:
	def __init__(self, batch_size = 100, lr = 0.002, lambda1 = 0.1, lambda_fm = 0.1, hidden1 = 300, hidden2 = 100, \
		acti_type = 'tanh', epoch = 100, x_drop = 1, dropout = 0.5, times_reduce_number = 1, nrows = 10000):
			self.batch_size = batch_size
			self.lr = lr
			self.lambda1 = lambda1
			self.lambda_fm = lambda_fm
			self.hidden1 = hidden1
			self.hidden2 = hidden2
			self.acti_type =acti_type
			self.epoch = epoch
			self.x_drop = x_drop
			self.dropout = dropout
			self.times_reduce_number = times_reduce_number
			self.nrows = nrows
	def getpara(self):
		return ' batch_size: ' + str(self.batch_size) + ' lr: ' + str(self.lr) + ' lambda1: ' + str(self.lambda1) + ' lambda_fm: ' + str(self.lambda_fm) + \
			' hidden1: ' + str(self.hidden1) + ' hidden2: ' + str(self.hidden2) + ' acti_type' + str(self.acti_type) + ' epoch: ' + str(self.epoch) + \
			' x_drop: ' + str(self.x_drop) + ' dropout: ' + str(self.dropout) + ' times_reduce_number' + str(self.times_reduce_number)
parameter = parameters(nrows = 10000)
srng = RandomStreams(seed=234)
rng = np.random
rng.seed(1234)
batch_size = parameter.batch_size                                                         #batch size
lr = parameter.lr                                                               #learning rate
lambda1 = parameter.lambda1 # .01                                                        #regularisation rate
lambda_fm = parameter.lambda_fm
hidden1 = parameter.hidden1 															#hidden layer 1
hidden2 = parameter.hidden2 															#hidden layer 2
acti_type = parameter.acti_type                                                   #activation type
epoch = parameter.epoch    
nrows = parameter.nrows

name_list = ['platform', 'weekday', 'hour', 'country', 'city', 'dma', 'campaign_id', 'advertiser_id', 'source_id', 'publisher_id', \
 'topic_id', 'category_id', 'entity_id']
name_field = dict(zip(name_list, range(len(name_list))))
print name_field
#exit(0)
feat_field = {}
feat_weights = {}
w_0 = 0
feat_num = 0
k = 0
xdim = 0
print 'parameters initialised by fm.txt'
fm_model_file = './output/fm/fm.txt'                 #fm model file
fi = open(fm_model_file, 'r')
first = True
for line in fi:
    s = line.strip().split()
    if first:
        first = False
        w_0 = float(s[0])
        feat_num = int(s[1])
        k = int(s[2]) + 1 # w and v   11
        xdim = 1 + len(name_field) * k   #  1+8*11
    else:
		feat = int(s[0])
		weights = [float(s[1 + i]) for i in range(k)]
		feat_weights[feat] = weights  #feat_weights[0] = 0.00000 0.01357 -0.02396 0.00668 ...
		name = s[1 + k]            # weekday 
		field = name_field[name]   #0
		feat_field[feat] = field    #feat_field[0] = 0

top = './data/merge_train3'                             
x_drop = parameter.x_drop
dropout = parameter.dropout
xdim += 2
       
# initialise parameters
w = rng.uniform(low = -np.sqrt(6. / (xdim + hidden1)),
                high = np.sqrt(6. / (xdim + hidden1)),
                size = (xdim, hidden1))
if acti_type == 'sigmoid':
    ww1 = np.asarray((w))
elif acti_type == 'tanh':
    ww1 = np.asarray((w*4))
else:
    ww1 = np.asarray(rng.uniform(-1, 1, size =( xdim, hidden1)))
bb1 = np.zeros(hidden1)

v = rng.uniform(low = -np.sqrt(6. / (hidden1 + hidden2)),
                high = np.sqrt(6. / (hidden1 + hidden2)),
                size = (hidden1, hidden2))
if acti_type == 'sigmoid':
    ww2 = np.asarray((v))
elif acti_type == 'tanh':
    ww2 = np.asarray((v*4))
else:
    ww2 = np.asarray(rng.uniform(-1, 1, size = (hidden1, hidden2)))
bb2 = np.zeros(hidden2)
ww3 = np.zeros(hidden2)

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name = "w1")
w2 = theano.shared(ww2, name = "w2")
w3 = theano.shared(ww3, name = "w3")
b1 = theano.shared(bb1, name = "b1")
b2 = theano.shared(bb2, name = "b2")
b3 = theano.shared(0. , name = "b3")

# Construct Theano expression graph

r0 = srng.binomial(size = (1, xdim), n = 1, p = x_drop)  # to do dropout
x = x * r0[0]

z1 = T.dot(x, w1) + b1
if acti_type == 'sigmoid':
    h1 = 1 / (1 + T.exp(-z1))              # hidden layer 1
elif acti_type == 'linear':
    h1 = z1
elif acti_type == 'tanh':
    h1 = T.tanh(z1)
r1 = srng.binomial(size = (1, hidden1), n = 1, p = dropout)
d1 = h1 * r1[0]

z2 = T.dot(h1, w2) + b2
if acti_type == 'sigmoid':
    h2 = 1 / (1 + T.exp(-z2))              # hidden layer 2
elif acti_type == 'linear':
    h2 = z2
elif acti_type == 'tanh':
    h2 = T.tanh(z2)
    
d2 = T.tanh(T.dot(d1, w2) + b2)
r2 = srng.binomial(size = (1, hidden2), n = 1, p = dropout)
d2 = d2 * r2[0]

p_drop = (1 / (1 + T.exp(-T.dot(d2, w3) - b3)))
p_1 = 1 / (1 + T.exp(-T.dot(h2, w3) - b3))               # Probability that target = 1
prediction = p_1 #> 0.5                                   # The prediction thresholded
xent = - y * T.log(p_drop) - (1 - y) * T.log(1 - p_drop)             # Cross-entropy loss function
cost = xent.sum() + lambda1 * ((w3 ** 2).sum() + (b3 ** 2))    # The cost to minimize
gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w3, b3, w2, b2, w1, b1, x])        # Compute the gradient of the cost

# Compile
train = theano.function(
          inputs = [x, y],
          outputs = [gx, w1, w2, w3, b1, b2, b3], updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)))
predict = theano.function(inputs = [x], outputs = prediction)

log_file = './output/fnn/logfile.txt'
def log_p(msg):
    print msg
    with open(log_file, "a+") as myfile:
        myfile.write(msg + "\n")
#print error
def print_err(dataframe, msg = ''):
    map_12 = get_map(dataframe)
    log_p(msg + '\t' + str(map_12))    

def get_map(dataframe, cols, num):
    grouped = dataframe.groupby(dataframe['display_id'])
    count = 0
    predicted = []
    actual = []
    for group in grouped.groups.values():
		xx_bat = []
		ad_prob = {}
		temp = []
		# if num > 0:
			# print group
		for loc in group:
			# if num > 0:
				# print loc - num * nrows
				#print dataframe[cols].iloc[140]
			xx, yy = get_xy(dataframe[cols].iloc[loc - num * nrows,:])
			#print xx.shape
			#print dataframe[old_cols].iloc[loc - num * nrows,1]
			xx = np.append(xx, dataframe['ad_clicked_prob'][loc - num * nrows])
			xx = np.append(xx, dataframe['ad_prob'][loc - num * nrows])
			xx_bat.append(np.asarray(xx))
			#print len(xx_bat)
			if dataframe['clicked'][loc] == 1:
				temp.append(dataframe['ad_id'][loc])
		#print temp
		pred = predict(xx_bat)
		#print pred
		for i, loc in enumerate(group):
			ad_prob[dataframe['ad_id'][loc]] = pred[i]
		sorted_pred = sorted(dataframe['ad_id'][group], key = lambda x: ad_prob[x], reverse = True)
		#print dataframe['ad_id'][group]
		#print sorted_pred
		# count += 1
		# if count > 5:
			# break
		predicted.append(sorted_pred[:])
		actual.append(temp[:])	
		#print predicted
		#print actual
    #print mapk(actual, predicted, 12)
    return mapk(actual, predicted, 12)
	
def get_batch_data(dataframe, index, size):#1,5->1,2,3,4,5
    xarray = []
    yarray = []
    farray = []
    fxy_list = range(dataframe.shape[1])
    fxy_list.remove(1)
    fxy_list.remove(2)
    #print dataframe
    for i in range(index, index + size):
        f, x, y = get_fxy(dataframe.iloc[i, fxy_list])
        x = np.append(x, dataframe['ad_clicked_prob'][i])
        x = np.append(x, dataframe['ad_prob'][i])
        #print dataframe.iloc[i,1]
        #print dataframe.iloc[i,2]
        #exit(0)
        xarray.append(x)
        yarray.append(y)
        farray.append(f)
    xarray = np.array(xarray, dtype = theano.config.floatX)
    yarray = np.array(yarray, dtype = np.int32)
    #print xarray.shape
    #print yarray.shape
    return farray, xarray, yarray

def get_xy(line):
    y = int(line[0])
    feats = [int(line[i].split(':')[0]) for i in range(1, len(line))]
    weights_for_feats = [float(line[i].split(':')[1]) for i in range(1, len(line))]
    x = feats_to_layer_one_array(feats, weights_for_feats)
    return x, y
	
def get_x(line):
    feats = [int(line[i].split(':')[0]) for i in range(0, len(line))]
    weights_for_feats = [float(line[i].split(':')[1]) for i in range(1, len(line))]
    x = feats_to_layer_one_array(feats, weights_for_feats)
    return x

def get_fxy(line):
    y = int(line[0])
    feats = [int(line[i].split(':')[0]) for i in range(1, len(line))]
    weights_for_feats = [float(line[i].split(':')[1]) for i in range(1, len(line))]
    #print weights_for_feats
    x = feats_to_layer_one_array(feats, weights_for_feats)
    return feats, x, y

def list2str(alist):
	str1 = ''
	for i in alist:
		str1 += str(i)
		str1 += ' '
	return str1
	
round = 0
min_map_epoch = 0
times_reduce = 0
min_err = 0
times_reduce_number = parameter.times_reduce_number
times_reduce = 0
train_map_round = []
test_map_round = []
train_map_epoch = []
test_map_epoch = []
for dirpath, dirnames, files in os.walk(top, topdown = True):
	whole_test_list = [files[random_num] for random_num in np.random.choice(len(files), int(0.2 * len(files)), replace = False)]
	whole_train_list = [train_file_name for train_file_name in files if train_file_name not in whole_test_list]
#print whole_test_list
#print whole_train_list
for i in range(epoch):
	#test_list = [whole_test_list[random_num] for random_num in np.random.choice(len(whole_test_list), int(0.1 * len(whole_test_list)), replace = False)]
	#train_list = [whole_train_list[random_num] for random_num in np.random.choice(len(whole_train_list), int(0.1 * len(whole_train_list)), replace = False)]
	train_list = whole_train_list
	test_list = whole_test_list
	columns_to_use = ['clicked', 'ad_clicked_prob', 'ad_prob', 'platform', 'weekday', 'hour', 'country', 'city', 'dma', \
		'campaign_id', 'advertiser_id', 'source_id', 'publisher_id', 'topic_id', 'category_id', 'entity_id']
	columns_map = ['clicked', 'platform', 'weekday', 'hour', 'country', 'city', 'dma', \
		'campaign_id', 'advertiser_id', 'source_id', 'publisher_id', 'topic_id', 'category_id', 'entity_id']
		# train
		#print dirpath   #'./input/data/train'
		#print dirnames  # []
	for file_name in train_list:
		#print columns_to_use
		file_name = os.path.join(dirpath, file_name)
		round += 1
		train_data = pd.read_csv(file_name, usecols = ['display_id', 'ad_id', 'clicked', 'ad_clicked_prob', 'ad_prob', 'platform', 'weekday', 'hour', 'country', 'city', 'dma', \
			'campaign_id', 'advertiser_id', 'source_id', 'publisher_id', 'topic_id', 'category_id', 'entity_id'])
			#start_time = time.time()
			#train_data = train_data.iloc[i * nrows: (i + 1)* nrows, :]
		train_size = train_data.shape[0]
			#print train_size
		n_batch = train_size/batch_size       #number of batches
			#print n_batch
			# Train
		print "Training model: %s" % file_name
		best_w1 = w1.get_value()
		best_w2 = w2.get_value()
		best_w3 = w1.get_value()
		best_b1 = b1.get_value()
		best_b2 = b2.get_value()
		best_b3 = b3.get_value()
		index = 0
		for j in range(n_batch):
			if index > train_size:
				break
			f, x, y = get_batch_data(train_data[columns_to_use], index, batch_size)
			index += batch_size
				#print index
				#print y
			#print x.shape, y.shape
			gx, w1t, w2t, w3t, b1t, b2t, b3t = train(x, y)
			b_size = len(f)
			for t in range(b_size):
				ft = f[t]
				gxt = gx[t]
				for feat in ft:
					for l in range(k):
						feat_weights[feat][l] = feat_weights[feat][l] * (1 - 2. * lambda_fm * lr / b_size) \
                                           - lr * gxt[feat_layer_one_index(feat, l)] * 1
			#print train_data
		train_map = get_map(train_data, columns_map, 0)# train error
		train_map_round.append(train_map)
			
	log_p('Train Err:' + str(i) + '\tmap:' + str(sum(train_map_round)/len(train_map_round)))
	train_map_epoch.append(sum(train_map_round)/len(train_map_round))
	train_map_round = []		
		
	for test_file_name in test_list:
		test_file_name = os.path.join(dirpath, test_file_name)
		test_data = pd.read_csv(test_file_name, usecols = ['display_id', 'ad_id', 'clicked', 'ad_clicked_prob', 'ad_prob', 'platform', 'weekday', 'hour', 'country', 'city', 'dma', \
				'campaign_id', 'advertiser_id', 'source_id', 'publisher_id', 'topic_id', 'category_id', 'entity_id'])
		test_map = get_map(test_data, columns_map, 0)
		test_map_round.append(test_map)
			
	map_epoch = sum(test_map_round)/len(test_map_round)
	log_p('Test Err:' + str(round) + '\t' + str(map_epoch))
	test_map_epoch.append(map_epoch)
	test_map_round = []
	#stop training when no improvement for a while 
	if map_epoch > min_err:
		best_w1 = w1t
		best_w2 = w2t
		best_w3 = w3t
		best_b1 = b1t
		best_b2 = b2t
		best_b3 = b3t
		min_err = map_epoch
		min_map_epoch = i
		if times_reduce < times_reduce_number:
			times_reduce += 1
	else:
		times_reduce -= 1
	if times_reduce < 0:
		break
	print times_reduce

log_p('Minimal test error is '+ str(min_err)+' , at epoch ' + str(min_map_epoch) + '. parameters: ' + parameter.getpara())
log_p('train_error_epoch ' + list2str(train_map_epoch))
log_p('test_error_epoch ' + list2str(test_map_epoch))
save_file = open('./output/model_paras/paras', 'wb')
cPickle.dump(best_w1.get_value(borrow=True), save_file, -1)
cPickle.dump(best_w2.get_value(borrow=True), save_file, -1) 
cPickle.dump(best_w3.get_value(borrow=True), save_file, -1)
cPickle.dump(best_b1.get_value(borrow=True), save_file, -1)
cPickle.dump(best_b2.get_value(borrow=True), save_file, -1) 
cPickle.dump(best_b3.get_value(borrow=True), save_file, -1)
save_file.close()
'''
save_file = open('path')
w.set_value(cPickle.load(save_file), borrow=True)
v.set_value(cPickle.load(save_file), borrow=True)
u.set_value(cPickle.load(save_file), borrow=True)
'''
'''
def get_prediction(dataframe):
    xx_bat = []
    for i in xrange(dataframe.shape[0]):
		xx = get_x(dataframe.iloc[i,:])
		xx_bat.append(np.asarray(xx))
    pred = predict(xx_bat)
    return pred
	
top = './input/data/test'
for dirpath, dirnames, files in os.walk(top):
	test_list = [file_name for file_name in files]
	print test_list
	for file_name in files:
		test_data = pd.read_csv(os.path.join(dirpath, file_name), usecols = ['campaign_id', 'advertiser_id', 'platform', 'country', 'city', 'dma', 'weekday', 'hour'])
		pred = get_prediction(test_data)
		df_pred = pd.DataFrame(pred)
		df_pred.to_csv('./output/submission1/%s' % file_name, index = False, header = None)
'''