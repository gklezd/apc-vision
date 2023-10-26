
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

import imageio as img

from two_layer_apc import APC
from utils import load_mnist, load_fashion, load_omni, load_affnist, convert_node, extract_box


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="mnist or coil100",default='mnist', type=str)
parser.add_argument("--outdir", help="Directory for results and log files", default='.', type=str)
args = parser.parse_args()
DATASET = args.dataset
OUTDIR = args.outdir


fh = open('./models/{}/'.format(OUTDIR)+'config.pkl','rb')
CFG = pickle.load(fh)
fh.close()


if not os.path.exists('./viz/{}/'.format(OUTDIR)):
	os.mkdir('./viz/{}/'.format(OUTDIR))

if not os.path.exists('./viz/{}/steps/'.format(OUTDIR)):
	os.mkdir('./viz/{}/steps/'.format(OUTDIR))


CFG.BATCH_SIZE = N = np.minimum(CFG.BATCH_SIZE,10)


if DATASET=='mnist':
	tr_ds, vd_ds, ts_ds = load_mnist(CFG)
elif DATASET=='fashion':
	tr_ds, vd_ds, ts_ds = load_fashion(CFG)
elif DATASET=='omni':
	tr_ds, vd_ds, ts_ds = load_omni(CFG)
elif DATASET=='affnist':
	tr_ds, vd_ds, ts_ds = load_affnist(CFG)



model = APC(CFG)

model.std2 = 0.0
model.std1 = 0.0

H,W,C = CFG.dims
model.build([(CFG.BATCH_SIZE, H, W, C),(CFG.BATCH_SIZE,)])
print(model.summary())

model.load_weights('./models/{}/model/weights'.format(OUTDIR)).expect_partial()



ts_iter = iter(ts_ds)
model.forward_pass(next(ts_iter))



g1_sz = CFG.g1_sz
g2_sz = CFG.g2_sz
g2_sub_sz = CFG.g2_sub_sz


ACTS = model.activations;

I = ACTS['I'].numpy();

_, img_H, img_W, _ = I.shape
print( I.shape)


GRIDS1 = convert_node(ACTS['grids1'])
GRIDS2 = convert_node(ACTS['grids2'])

G1 = convert_node(ACTS['g1'])
G1_HAT = convert_node(ACTS['g1_hat'])
G2_HAT = convert_node(ACTS['g2_hat'])

G2 = convert_node(ACTS['g2'])
G2_SUB = convert_node(ACTS['g2_sub'])

G1_INIT = ACTS['init_glimpse']
GRID_INIT1 = ACTS['init_grid1']

MSE = convert_node(ACTS['mse'])

GLIMPSE_INIT = ACTS['init_glimpse'].numpy()
			



I_HAT = ACTS['I_hat'][1:]

	
T1 = CFG.T1
T2 = CFG.T2
T = T1*T2

PLOT_WIDTH = 8

for n in range(N):
	images = []
	
	print(n)
	for t2 in range(T2):
		for t1 in range(T1):
		
			t = t2*T1+t1;
			
			pidx = 1;
		
			plt.subplot(1,PLOT_WIDTH,pidx)
			plt.axis('off')
			
			if C==1:
				plt.imshow(I_HAT[t][n].numpy(),cmap='gray',vmin=0.0,vmax=1.0)
			else:					
				plt.imshow(I_HAT[t][n].numpy(),vmin=0.0,vmax=1.0)
			
			
			
			pidx += 1
			plt.subplot(1,PLOT_WIDTH,pidx)
			plt.axis('off')
			
			if C==1:
				plt.imshow(I[n].reshape([H,W]),cmap='gray',vmin=0.0,vmax=1.0)
			else:
				plt.imshow(I[n].reshape([H,W,C]),vmin=0.0,vmax=1.0)
			plt.xlim(0,img_W)
			plt.ylim(img_H,0)
				
				
			GRID_INIT = ACTS['init_grid1'].numpy()
			GRID_INIT = (GRID_INIT[n]+1)*H/2
			GRID_INIT = GRID_INIT.reshape([2,-1])
			GRID_INIT = np.transpose(GRID_INIT)
			GRID_INIT = GRID_INIT.reshape([g1_sz,g1_sz,2])
			x0_vals,y0_vals = extract_box(GRID_INIT);
			
			plt.plot(x0_vals[0,[0,1]],y0_vals[0,[0,1]],color='r')
			plt.plot(x0_vals[0,[0,2]],y0_vals[0,[0,2]],color='r')
			plt.plot(x0_vals[0,[1,3]],y0_vals[0,[1,3]],color='r')
			plt.plot(x0_vals[0,[2,3]],y0_vals[0,[2,3]],color='r')
			
			
			GRID2 = (GRIDS2[t2][n]+1)*H/2
			GRID2 = GRID2.reshape([2,-1])
			GRID2 = np.transpose(GRID2)
			GRID2 = GRID2.reshape([g2_sz,g2_sz,2])
			x2_vals,y2_vals = extract_box(GRID2);
			
			plt.plot(x2_vals[0,[0,1]],y2_vals[0,[0,1]])
			plt.plot(x2_vals[0,[0,2]],y2_vals[0,[0,2]])
			plt.plot(x2_vals[0,[1,3]],y2_vals[0,[1,3]])
			plt.plot(x2_vals[0,[2,3]],y2_vals[0,[2,3]])
			
			
			
			
			pidx += 1
			plt.subplot(1,PLOT_WIDTH,pidx)
			plt.axis('off')
			
			if C==1:
				plt.imshow(G2[t2][n].reshape([g2_sz,g2_sz]),cmap='gray',vmin=0.0,vmax=1.0)
			else:					
				plt.imshow(G2[t2][n].reshape([g2_sz,g2_sz,C]),vmin=0.0,vmax=1.0)
			plt.xlim(0,g2_sz)
			plt.ylim(g2_sz,0)
			
			GRID1 = (GRIDS1[t][n]+1)*CFG.g2_sz/2
			GRID1 = GRID1.reshape([2,-1])
			GRID1 = np.transpose(GRID1)
			GRID1 = GRID1.reshape([g1_sz,g1_sz,2])
			x1_vals,y1_vals = extract_box(GRID1);
			
			plt.plot(x1_vals[0,[0,1]],y1_vals[0,[0,1]])
			plt.plot(x1_vals[0,[0,2]],y1_vals[0,[0,2]])
			plt.plot(x1_vals[0,[1,3]],y1_vals[0,[1,3]])
			plt.plot(x1_vals[0,[2,3]],y1_vals[0,[2,3]])
			
			
			
			pidx += 1
			plt.subplot(1,PLOT_WIDTH,pidx)
			plt.axis('off')
			
			if C==1:
				plt.imshow(G1[t][n].reshape([g1_sz,g1_sz]),cmap='gray',vmin=0.0,vmax=1.0)
			else:					
				plt.imshow(G1[t][n].reshape([g1_sz,g1_sz,C]),vmin=0.0,vmax=1.0)
			
			
			
			pidx += 1
			plt.subplot(1,PLOT_WIDTH,pidx)
			plt.axis('off')
			
			if C==1:
				plt.imshow(G1_HAT[t][n].reshape([g1_sz,g1_sz]),cmap='gray',vmin=0.0,vmax=1.0)
			else:					
				plt.imshow(G1_HAT[t][n].reshape([g1_sz,g1_sz,C]),vmin=0.0,vmax=1.0)
			
			
			
			pidx += 1
			plt.subplot(1,PLOT_WIDTH,pidx)
			plt.axis('off')
			
			if C==1:
				plt.imshow(G2_SUB[t2][n].reshape([g2_sub_sz,g2_sub_sz]),cmap='gray',vmin=0.0,vmax=1.0)
			else:					
			
				plt.imshow(G2_SUB[t2][n].reshape([g2_sub_sz,g2_sub_sz,C]),vmin=0.0,vmax=1.0)
				
				
				
				
				
			pidx += 1
			plt.subplot(1,PLOT_WIDTH,pidx)
			plt.axis('off')
			
			if C==1:
				plt.imshow(G2_HAT[t2][n].reshape([g2_sub_sz,g2_sub_sz]),cmap='gray',vmin=0.0,vmax=1.0)
			else:					
				plt.imshow(G2_HAT[t2][n].reshape([g2_sub_sz,g2_sub_sz,C]),vmin=0.0,vmax=1.0)
			
			
			
			pidx += 1
			plt.subplot(1,PLOT_WIDTH,pidx)
			plt.axis('off')
			
			if C==1:
				plt.imshow(GLIMPSE_INIT[n].reshape([g1_sz,g1_sz]),cmap='gray',vmin=0.0,vmax=1.0)
			else:					
				plt.imshow(GLIMPSE_INIT[n].reshape([g1_sz,g1_sz,C]),vmin=0.0,vmax=1.0)
				
			
				

				
			plt.subplots_adjust(wspace=0.05, hspace=0.05)
			fnm = './viz/{}/steps/'.format(OUTDIR)+'step'+str(n)+'_'+str(t)+'.png'
			plt.savefig(fnm,bbox_inches="tight")
			
			plt.clf()				
			
			images.append(img.imread(fnm))
			
	img.mimsave('./viz/{}/'.format(OUTDIR)+'ex'+str(n)+'.gif', images, duration=1000)

		
	
