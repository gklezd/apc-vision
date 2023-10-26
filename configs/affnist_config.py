
import numpy as np

class CFG:
	def __init__(self):
		self.var_z = False
		self.disable_loc1 = False
		self.num_classes = 10
		self.label_set = None
		self.dataset = 'omni'
		self.use_peripheral = False
		self.use_low_res = False
		self.a2_sz = 256;
		self.a1_sz = 64;
		pass;

cfg = CFG()


cfg.BATCH_SIZE = 128;
cfg.lr = 1e-4

cfg.hyper = True


cfg.factor1 = 14/7
cfg.factor2 = 40/14


cfg.g1_sz = 7
cfg.g2_sz = 14
cfg.g2_sub_sz = 7



cfg.T1 = 3;
cfg.T2 = 4;


cfg.dims = [40,40,1]

cfg.r2_sz = 256;
cfg.r1_sz = 64;
cfg.z_sz = 8;

cfg.std1 = 1e-1
cfg.std2 = 1e-1



	
	
	
	
