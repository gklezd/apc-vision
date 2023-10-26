
import os
import numpy as np
import sys

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow import keras
import tensorflow as tf
import argparse
import pickle

import importlib

from two_layer_apc import APC
from utils import load_mnist, load_fashion, load_omni, load_affnist






parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="mnist or coil100",default='mnist', type=str)
parser.add_argument("--outdir", help="Directory for results and log files", default='.', type=str)
parser.add_argument("--load", help="load model",default=0, type=int)
parser.add_argument("--epochs", help="# of training epochs",default=100, type=int)
parser.add_argument("--eval", help="evaluate",default=0, type=int)
args = parser.parse_args()
DATASET = args.dataset
OUTDIR = args.outdir
LOAD = args.load
EPOCHS = args.epochs
EVAL = args.eval



if not os.path.exists('./models/{}/'.format(OUTDIR)):
	os.mkdir('./models/{}/'.format(OUTDIR))


if LOAD:
	fh = open('./models/{}/'.format(OUTDIR)+'config.pkl','rb')
	CFG = pickle.load(fh)
	fh.close()
else:
	CFG = importlib.import_module('.'+DATASET+'_config','configs').cfg


if DATASET=='mnist':
	tr_ds, vd_ds, ts_ds = load_mnist(CFG)
elif DATASET=='fashion':
	tr_ds, vd_ds, ts_ds = load_fashion(CFG)
elif DATASET=='omni':
	tr_ds, vd_ds, ts_ds = load_omni(CFG)
elif DATASET=='affnist':
	tr_ds, vd_ds, ts_ds = load_affnist(CFG)
	


fh = open('./models/{}/'.format(OUTDIR)+'config.pkl','wb')
pickle.dump(CFG,fh)
fh.close()

BATCH_SIZE = CFG.BATCH_SIZE
H,W,C = CFG.dims



model = APC(CFG)

model.build([(CFG.BATCH_SIZE, H, W, C),(CFG.BATCH_SIZE,)])
print(model.summary())

if LOAD:
	model.load_weights('./models/{}/model/weights'.format(OUTDIR)).expect_partial()

loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanSquaredError()]

model.compile(
	optimizer=keras.optimizers.Adam(CFG.lr),
	loss=loss,
	metrics=metrics#, run_eagerly=True
)


if EVAL:
	
	model.std1 = 0.0
	model.std2 = 0.0
	
	x = 0.0
	for i in range(10):
		x += model.evaluate(ts_ds)/10
	print(x)
	exit()
	
else:

	model.fit(
		tr_ds,
		validation_data = vd_ds,
		validation_freq = 50,
		epochs = EPOCHS,
		verbose = 1
	)
	
	model.save_weights('./models/{}/model/weights'.format(OUTDIR))
	model.evaluate(ts_ds)











