
import tensorflow as tf;
import numpy as np;

import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

from tensorflow import keras

from PIL import Image
from scipy.io import loadmat




class hyperfanin_for_kernel(keras.initializers.Initializer):
	def __init__(self,fanin,varin=1.0,relu=True,bias=True):
		self.fanin = fanin
		self.varin = varin
		self.relu = relu
		self.bias = bias

	def __call__(self, shape, dtype=None, **kwargs):
		hfanin,_ = shape;
		variance = (1/self.varin)*(1/self.fanin)*(1/hfanin)

		if self.relu:
			variance *= 2.0;
		if self.bias:
			variance /= 2.0;
		
		variance = np.sqrt(3*variance);
		
		return tf.random.uniform(shape, minval=-variance, maxval=variance)
		#return tf.random.normal(shape)*variance
		
	def get_config(self):  # To support serialization
		return {"fanin": self.fanin, "varin": self.varin, "relu": self.relu, "bias": self.bias}
		
		
		

class hyperfanin_for_bias(keras.initializers.Initializer):
	def __init__(self,varin=1.0,relu=True):
		self.varin = varin
		self.relu = relu

	def __call__(self, shape, dtype=None, **kwargs):
		hfanin,_ = shape;
		variance = (1/2)*(1/self.varin)*(1/hfanin)
		
		if self.relu:
			variance *= 2.0;
		
		variance = np.sqrt(3*variance);
		
		return tf.random.uniform(shape, minval=-variance, maxval=variance)
		#return tf.random.normal(shape)*variance

	def get_config(self):  # To support serialization
		return {"relu": self.relu, "varin": self.varin}




	
def load_affnist(CFG):
	dat_val_train = loadmat('./Data/affnist/training_and_validation_batches/1.mat')['affNISTdata'][0][0]
	dat_test = loadmat('./Data/affnist/test_batches/1.mat')['affNISTdata'][0][0]

	x_val_train = np.transpose(dat_val_train[2])
	y_val_train = np.transpose(dat_val_train[5]).reshape([-1,])
	
	
	x_test = np.transpose(dat_test[2])
	y_test = np.transpose(dat_test[5]).reshape([-1,])
	
	x_val_train, x_test = x_val_train / 255.0, x_test / 255.0

	x_val_train = x_val_train.reshape([-1,40,40,1]).astype(np.float32)
	x_test = x_test.reshape([-1,40,40,1]).astype(np.float32)
	
	y_val_train = y_val_train.astype(np.int32)
	y_test = y_test.astype(np.int32)
	
	
	H = W = 40
	
	x_train = x_val_train[:50000]
	y_train = y_val_train[:50000]
	
	x_valid = x_val_train[50000:]
	y_valid = y_val_train[50000:]
	
	
	
	assert x_train.shape[0] == 50000
	assert y_train.shape[0] == 50000
	assert x_valid.shape[0] == 10000
	assert y_valid.shape[0] == 10000
	assert x_test.shape[0] == 10000
	assert y_test.shape[0] == 10000
	

	tr_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	vd_ds = tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).shuffle(10000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	ts_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(10000).batch(CFG.BATCH_SIZE, drop_remainder=True)

	

	return tr_ds, vd_ds, ts_ds;



	
def load_omni(CFG):
	x_train = np.load('./Data/omni/omni_train.npy')
	x_valid = np.load('./Data/omni/omni_valid.npy')
	x_test = np.load('./Data/omni/omni_test.npy')
	x_transfer = np.load('./Data/omni/omni_transfer.npy')

	
	x_orig = np.concatenate([x_train,x_valid,x_test],axis=0)
	
	x_train = x_orig[:,:17]
	x_test = x_orig[:,17:]
	
	
	H,W = x_train.shape[-2:]
	C = 1

	x_train = x_train.reshape([-1,H,W,C]).astype(np.float32)
	x_test = x_test.reshape([-1,H,W,C]).astype(np.float32)
	x_transfer = x_transfer.reshape([-1,H,W,C]).astype(np.float32)
	
	y_train = np.ones((x_train.shape[0],))
	y_test = np.ones((x_test.shape[0],))
	y_transfer = np.ones((x_transfer.shape[0],))
	
	
	tr_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	ts_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	tf_ds = tf.data.Dataset.from_tensor_slices((x_transfer,y_transfer)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	
	return tr_ds, ts_ds, tf_ds;
	
	




def load_fashion(CFG):


	data = tf.keras.datasets.fashion_mnist
	(x_train, y_train), (x_test, y_test) = data.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	x_train = x_train[..., tf.newaxis].astype(np.float32)
	x_test = x_test[..., tf.newaxis].astype(np.float32)
	
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)
	
	
	data = np.concatenate([x_train,x_test],axis=0)
	labs = np.concatenate([y_train,y_test],axis=0)
	
	
	label_set = CFG.label_set
	
	if label_set is not None:
		set_idx = np.where(np.isin(labs,label_set))[0]
		data = data[set_idx]
		labs = labs[set_idx]
		
		for lab in range(len(label_set)):
			labs[np.where(labs==label_set[lab])[0]] = lab


	N = data.shape[0]
	
	
	num_split = np.int32(np.floor((1-0.9)*N));
	N_ts = num_split
	N_vd = num_split
	N_tr = N - 2*num_split
	
	np.random.seed(1337) 
	
	shuffle_idx = np.arange(N);
	shuffle_idx = np.random.permutation(shuffle_idx)
	
	
	x_test = data[shuffle_idx[:num_split]]
	x_valid = data[shuffle_idx[num_split:(2*num_split)]]
	x_train = data[shuffle_idx[(2*num_split):]]
	
	
	y_test = labs[shuffle_idx[:num_split]]
	y_valid = labs[shuffle_idx[num_split:(2*num_split)]]
	y_train = labs[shuffle_idx[(2*num_split):]]
	

	tr_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	vd_ds = tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	ts_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)

	return tr_ds, vd_ds, ts_ds;






def load_mnist(CFG):


	data = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = data.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	x_train = x_train[..., tf.newaxis].astype(np.float32)
	x_test = x_test[..., tf.newaxis].astype(np.float32)
	
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)
	
	
	data = np.concatenate([x_train,x_test],axis=0)
	labs = np.concatenate([y_train,y_test],axis=0)
	
	
	label_set = CFG.label_set
	
	if label_set is not None:
		set_idx = np.where(np.isin(labs,label_set))[0]
		data = data[set_idx]
		labs = labs[set_idx]
		
		for lab in range(len(label_set)):
			labs[np.where(labs==label_set[lab])[0]] = lab


	N = data.shape[0]
	
	
	num_split = np.int32(np.floor((1-0.9)*N));
	N_ts = num_split
	N_vd = num_split
	N_tr = N - 2*num_split
	
	np.random.seed(1337) 
	
	shuffle_idx = np.arange(N);
	shuffle_idx = np.random.permutation(shuffle_idx)
	
	
	x_test = data[shuffle_idx[:num_split]]
	x_valid = data[shuffle_idx[num_split:(2*num_split)]]
	x_train = data[shuffle_idx[(2*num_split):]]
	
	
	y_test = labs[shuffle_idx[:num_split]]
	y_valid = labs[shuffle_idx[num_split:(2*num_split)]]
	y_train = labs[shuffle_idx[(2*num_split):]]
	

	tr_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	vd_ds = tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)
	ts_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(1000).batch(CFG.BATCH_SIZE, drop_remainder=True)

	return tr_ds, vd_ds, ts_ds;




	




def convert_node(x):
	for t in range(len(x)):
		x[t] = x[t].numpy()
	return x;



def generate_grid(scale,resolution,dims):

	x_C = np.linspace(-scale,scale,resolution);

	axes = [];
	
	for dim in range(dims):
		axes += [x_C];

	packed_grid = np.meshgrid(*axes);
	
	np_grid = packed_grid[0].reshape([-1,1]);

	for dim in range(dims-1):
		np_grid = np.concatenate([np_grid,packed_grid[dim+1].reshape([-1,1])],axis=1);
	
	return np.float32(np_grid);
	
	
def extract_box(tgrid):
	
	Ax = tgrid[0,0,0].reshape([-1,1])
	Ay = tgrid[0,0,1].reshape([-1,1])
	Bx = tgrid[0,-1,0].reshape([-1,1])
	By = tgrid[0,-1,1].reshape([-1,1])
	Cx = tgrid[-1,0,0].reshape([-1,1])
	Cy = tgrid[-1,0,1].reshape([-1,1])
	Dx = tgrid[-1,-1,0].reshape([-1,1])
	Dy = tgrid[-1,-1,1].reshape([-1,1])

	x_vals = np.concatenate([Ax,Bx,Cx,Dx],axis=-1)
	y_vals = np.concatenate([Ay,By,Cy,Dy],axis=-1)

	return x_vals,y_vals;
