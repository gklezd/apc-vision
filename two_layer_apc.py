

import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformer import transformer
from utils import hyperfanin_for_kernel, hyperfanin_for_bias


import tensorflow_probability as tfp

class APC(keras.Model):

	def __init__(self,cfg):
		super(APC, self).__init__()
		
		self.dataset = cfg.dataset
		self.epsilon = 1e-1
		
		
		self.T1 = cfg.T1;
		self.T2 = cfg.T2;
		self.T = self.T1*self.T2
		
		self.THR2 = 1.0 - 1/cfg.factor2
		self.THR1 = 1.0 - 1/cfg.factor1
		
		
		self.cfg = cfg
		self.hyper = cfg.hyper
		self.a2_sz = cfg.a2_sz;
		self.a1_sz = cfg.a1_sz;
		

		self.BATCH_SIZE = cfg.BATCH_SIZE
		

		self.lr = cfg.lr;
		
		self.use_baseline = True
		
		self.H = cfg.dims[0];
		self.W = cfg.dims[1];
		self.C = cfg.dims[2];
		
		self.g1_sz = cfg.g1_sz
		self.g2_sz = cfg.g2_sz
		self.g2_sub_sz = cfg.g2_sub_sz
		self.bottleneck = self.g1_sz*self.g1_sz*self.C
		
		
		self.num_classes = cfg.num_classes
		
		self.r1_sz = cfg.r1_sz;
		self.r2_sz = cfg.r2_sz;
		self.z_sz = cfg.z_sz;
		
		self.e1_sz = cfg.r1_sz;
		self.e2_sz = cfg.r2_sz;
		
		
		self.in_fs1_sz = self.e1_sz+self.r1_sz+2
		self.in_fa1_sz = self.a1_sz
		
		self.fa1_sz = fa_sz = [2];
		self.fs1_sz = fs_sz = [self.r1_sz];
		
		
		self.action_layer_collection = []
		self.state_layer_collection = []
		
		
		self._create_action2()
		self._create_action_hypernet()
		self._create_loc_net2()
		self._create_action_feedback()
		
		self._create_rec()
		
		self._create_state2()
		self._create_state_hypernet()
		self._create_state1_init()
		self._create_state_feedback()
		self._create_r2_init()
		self._create_decoder()
		self._create_decoder2()
		
		self._create_baseline()
		
		self.flatten_e1 = tf.keras.layers.Flatten()
		
		
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr);
		
		self.std1 = cfg.std1;
		self.std2 = cfg.std2;
		
		self.activations = {};
		
		
		
		
		
	def _MSE(self,y,y_hat):
		N = self.cfg.BATCH_SIZE
		y_flat = tf.reshape(y,[N,-1])
		y_hat_flat = tf.reshape(y_hat,[N,-1])
		return tf.reduce_mean(tf.square(y_flat-y_hat_flat),axis=-1)

		
	def pred_error(self):
		error = 0.0
		for t in range(self.T):
			error += tf.reduce_mean(self.activations['pred_errors'][t])
		for t2 in range(self.T2):
			error += 1e-1*tf.reduce_mean(self.activations['pred_errors2'][t2])
		for error_token in self.activations['mse']:
			error += tf.reduce_mean(error_token)
		return error
		
	
		
		
	def dense_mse_reward(self):
		
		T1 = self.T1
		T2 = self.T2
		T = self.T
		N = self.cfg.BATCH_SIZE
		I = self.activations['I']
		
		R = tf.zeros((N,0));
		
		for t2 in range(self.T2):
			for t1 in range(self.T1):
				t = t2*self.T1+t1
				hat0 = self.activations['I_hat'][t]
				hat1 = self.activations['I_hat'][t+1]
				mse0 = self._MSE(I,hat0)
				mse1 = self._MSE(I,hat1)
				
				Rt = tf.expand_dims(mse0-mse1,axis=-1)
				R = tf.concat([R,Rt],axis=-1)

		cumulative_reward2 = tf.zeros((N,0))
		cumulative_reward1 = tf.zeros((N,0))
		
		for t2 in range(self.T2):
			cumulative_reward2_t = tf.slice(R,[0,t2*T1],[-1,-1])
			cumulative_reward2_t = tf.reduce_sum(cumulative_reward2_t,axis=-1,keepdims=True)
			cumulative_reward2 = tf.concat([cumulative_reward2,cumulative_reward2_t],axis=-1)
			
			for t1 in range(self.T1):
				t = t2*T1+t1
				cumulative_reward1_t = tf.slice(R,[0,t],[-1,-1])
				cumulative_reward1_t = tf.reduce_sum(cumulative_reward1_t,axis=-1,keepdims=True)
				cumulative_reward1 = tf.concat([cumulative_reward1,cumulative_reward1_t],axis=-1)

		return cumulative_reward2,cumulative_reward1;


	def penalty_func(self,x,c):
		pos = tf.square(tf.nn.leaky_relu(x-c))
		neg = tf.square(tf.nn.leaky_relu(-x-c))
		const = -2*tf.square(0.2*c)
		
		penalty = pos + neg - const;
		
		penalty = tf.reduce_sum(penalty,axis=-1)
		return tf.reduce_mean(penalty);
	
		
	def REINFORCE(self):
	
		
		where1_log_prob = tf.zeros((self.cfg.BATCH_SIZE,0))
		
		for t2 in range(self.T2):
			for t1 in range(self.T1):
				t = t2*self.T1 + t1
				loc_mu_t = self.activations['loc1_mu'][t]
				loc_t = self.activations['loc1'][t]
				
				locs_dist = tfp.distributions.Normal(loc=loc_t,scale=self.std1)
				where1_log_prob_t = tf.reduce_sum(locs_dist.log_prob(loc_mu_t),axis=-1,keepdims=True)
				where1_log_prob = tf.concat([where1_log_prob,where1_log_prob_t],axis=-1)
				
		
		where2_log_prob = tf.zeros((self.cfg.BATCH_SIZE,0))
		
		for t2 in range(self.T2):
			loc2_mu_t = self.activations['loc2_mu'][t2]
			loc2_t = self.activations['loc2'][t2]
			
			loc2_dist = tfp.distributions.Normal(loc=loc2_t,scale=self.std2)
			where2_log_prob_t = tf.reduce_sum(loc2_dist.log_prob(loc2_mu_t),axis=-1,keepdims=True)
			where2_log_prob = tf.concat([where2_log_prob,where2_log_prob_t],axis=-1)
			
			
		
		THR1 = self.THR1
		THR2 = self.THR2
		
		penalty = 0.0
		
		for t2 in range(self.T2):
			loc2 = self.activations['loc2_mu'][t2]
			penalty += self.penalty_func(loc2,THR2)
			
			for t1 in range(self.T1):
				t = t2*self.T1 + t1
				
				loc1 = self.activations['loc1_mu'][t]
				penalty += self.penalty_func(loc1,THR1)
		
		cumulative_reward2,cumulative_reward1 = self.dense_mse_reward()
		
		baseline2 = self.activations['baseline2']
		baseline1 = self.activations['baseline1']
		
		baseline_mse = self._MSE(tf.stop_gradient(cumulative_reward2),baseline2)
		baseline_mse += self._MSE(tf.stop_gradient(cumulative_reward1),baseline1)
		baseline_mse = tf.reduce_mean(baseline_mse)
		
	
		advantage2 = cumulative_reward2
		advantage1 = cumulative_reward1
	
		
		if self.use_baseline:
			advantage2 -= baseline2
			advantage1 -= baseline1
			
			
		REINFORCE_loss = -tf.reduce_mean(where2_log_prob*tf.stop_gradient(advantage2),axis=-1)
		REINFORCE_loss -= tf.reduce_mean(where1_log_prob*tf.stop_gradient(advantage1),axis=-1)
		REINFORCE_loss += 1e-5*penalty
		REINFORCE_loss = tf.reduce_mean(REINFORCE_loss)
			
		return REINFORCE_loss, baseline_mse
	
		
		
		
	def attempt_task(self,a2,a1):
		a2_a1 = tf.concat([a2,a1],axis=-1)
		
		I_hat = self.reconstruct(a2_a1)
		self.activations['I_hat'].append(I_hat);
		self.activations['mse'].append(self._MSE(self.activations['I'],I_hat));
		

		
	def forward_pass(self, inputs):
		[I,y] = inputs;
		
		
		
		self.activations['pred_errors'] = [];
		self.activations['pred_errors2'] = [];
		self.activations['loc1'] = [];
		self.activations['loc2'] = [];
		self.activations['loc1_mu'] = [];
		self.activations['loc2_mu'] = [];
		self.activations['loc1_sum'] = [];
		self.activations['g1'] = [];
		self.activations['g2'] = [];
		self.activations['g1_hat'] = [];
		self.activations['g2_hat'] = [];
		self.activations['g2_sub'] = [];
		self.activations['g2_sub_hat'] = [];
		self.activations['g1_params'] = [];
		self.activations['g2_params'] = [];
		self.activations['grids1'] = [];
		self.activations['grids2'] = [];
		
		self.activations['I'] = I;
		self.activations['I_hat'] = [];
		self.activations['mse'] = [];
		self.activations['y_hat'] = [];
		self.activations['baseline1'] = [];
		self.activations['baseline2'] = [];
		
		
		N = self.cfg.BATCH_SIZE
	
		init_loc = (tf.random.uniform((N,2))-0.5)*2*0.5
		init_loc = tf.stop_gradient(init_loc)
		self.activations['init_loc'] = init_loc;
		
		init_glimpse,init_params,init_grid = self.glimpse_network(I,[self.g1_sz,self.g1_sz],init_loc,layer=0)
		init_glimpse = tf.stop_gradient(init_glimpse)
		
		self.activations['init_glimpse'] = init_glimpse;
		self.activations['init_grid1'] = init_grid;
		init_glimpse = tf.reshape(init_glimpse,[N,-1])
		r2_in = tf.concat([init_glimpse,init_loc],axis=-1)
		
		r2 = self.init_state2(r2_in)
		a2 = tf.zeros((N,self.a2_sz))
		a2 += self.update_action2(a2,tf.stop_gradient(r2),tf.zeros((N,self.a2_sz)))
		
			
		res2 = tf.zeros((N,self.g2_sub_sz*self.g2_sub_sz*self.C))
		###############################
		###### MACRO CYCLE START ######
		###############################
		
			
		for t2 in range(self.T2):
		
		
			loc2_mu,z2 = self.loc_net2(tf.stop_gradient(a2));
			loc2 = loc2_mu + self.std2*tf.random.normal(tf.shape(loc2_mu));
			loc2 = tf.stop_gradient(loc2);

			self.activations['loc2_mu'].append(loc2_mu)
			self.activations['loc2'].append(loc2)
			
			
			fs_hyp_in = tf.concat([r2,tf.stop_gradient(z2),tf.stop_gradient(loc2)],axis=-1)
			
			if self.hyper:
				fa1_weights,ln1_weights = self.generate_fa1(z2)
				fa1 = lambda x : tf.nn.leaky_relu(self.f_theta(x,fa1_weights,[self.a1_sz],self.a1_sz+self.r1_sz))
				ln1 = lambda x : self.f_theta(x,ln1_weights,[2],self.a1_sz)
				
				
				fs1_weights = self.generate_fs1(fs_hyp_in)
				fs1 = lambda x : tf.nn.leaky_relu(self.f_theta(x,fs1_weights,[self.r1_sz],self.r1_sz+self.bottleneck+2))
				
			else:
				fa1,ln1 = self.act1(z2)
				fs1 = self.state1(fs_hyp_in);
			
			
			
			g2,g2_params,grids2 = self.glimpse_network(I,[self.g2_sz,self.g2_sz],tf.stop_gradient(loc2),layer=2)
			g2 = tf.stop_gradient(g2)
			g2_sub = self.sample_g2(g2)

			self.activations['g2'].append(g2)
			self.activations['g2_sub'].append(g2_sub)
			self.activations['g2_params'].append(g2_params)
			self.activations['grids2'].append(grids2)
			
			
			###############################
			###### MICRO CYCLE START ######
			###############################
			
			
			for t1 in range(self.T1):
			
				if t1==0:
					r1 = self.init_state1(fs_hyp_in);
					
					a1 = tf.zeros((N,self.a1_sz))
					fa1_in = tf.concat([tf.stop_gradient(r1),a1],axis=-1)
					a1 += fa1(fa1_in)	
									
					if t2==0:
						self.attempt_task(a2,a1)
						
				loc1_mu = ln1(tf.stop_gradient(a1))
				loc1 = loc1_mu + self.std1*tf.random.normal(tf.shape(loc1_mu));
				loc1 = tf.stop_gradient(loc1)
				
				self.activations['loc1_mu'].append(loc1_mu)
				self.activations['loc1'].append(loc1)
				
				
				
				g1,g1_params,grids1 = self.glimpse_network(g2,[self.g1_sz,self.g1_sz],loc1,layer=1,extra_loc=loc2)
				g1 = tf.stop_gradient(g1)
				
				self.activations['g1'].append(g1)
				self.activations['g1_params'].append(g1_params)
				self.activations['grids1'].append(grids1)	
				
				
				dec_input = tf.concat([r1,tf.stop_gradient(loc1),tf.stop_gradient(loc2)],axis=-1)
				g1_hat = self.decode(dec_input)
				self.activations['g1_hat'].append(g1_hat)
				
				res1 = self.flatten_e1(g1 - g1_hat)
				self.activations['pred_errors'].append(self._MSE(g1,g1_hat))
				
				
				fs1_in = tf.concat([tf.stop_gradient(loc1),r1,res1],axis=-1)
				r1 += fs1(fs1_in)
				
				fa1_in = tf.concat([tf.stop_gradient(r1),a1],axis=-1)
				a1 += fa1(fa1_in)
				
				r2 += self.update_state2(r2,loc2,self.state_feedback(r1))
				a2 += self.update_action2(a2,tf.stop_gradient(r2),self.action_feedback(a1))
			
				self.attempt_task(a2,a1)
			
			dec2_input = tf.concat([r2,tf.stop_gradient(loc2)],axis=-1)
			g2_hat = self.decode2(dec2_input)
			
			self.activations['pred_errors2'].append(self._MSE(g2_sub,g2_hat))
			self.activations['g2_hat'].append(g2_hat)
			
		self.activations['baseline1'] = self.baseline1(tf.ones((N,64)))
		self.activations['baseline2'] = self.baseline2(tf.ones((N,64)))
		
	###########################################################################################
	############################## SENSOR #####################################################
	###########################################################################################
				
	
	
	def glimpse_network(self,I,dims,loc,layer=1,extra_loc=None):
		N = I.get_shape().as_list()[0]
		gH,gW = dims;
		
		if layer==1:
			rf = -1.0 + 1/self.cfg.factor1
		elif layer==2:
			rf = -1.0 + 1/self.cfg.factor2
		elif layer==0:
			rf = -1.0 + 1/self.cfg.factor2*1/self.cfg.factor1
		
		rot = tf.zeros((N,1))
		shear = tf.zeros((N,1))
		scale = rf + tf.zeros((N,2))
		theta = tf.concat([loc,rot,scale,shear],axis=-1)

		glimpse,grids = transformer(I,theta,gH,gW,grids=True)
		
		return glimpse,theta,grids
			
			
		
		
	###########################################################################################
	############################## DEC NET ####################################################
	###########################################################################################
	
	
	def _create_rec(self):
		self.rec1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='rec1')
		self.rec2 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='rec2')
		self.rec3 = tf.keras.layers.Dense(self.H*self.W*self.C,activation=tf.nn.leaky_relu,name='rec3')
		
		self.action_layer_collection.append(self.rec1)
		self.action_layer_collection.append(self.rec2)
		self.action_layer_collection.append(self.rec3)
		
		
		
	def reconstruct(self,x):
		h = self.rec1(x)
		h = self.rec2(h)
		y = self.rec3(h)
		return tf.reshape(y,[-1,self.H,self.W,self.C])
		
		
	
	
	def _create_decoder(self):
		self.dec11 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='dec11')
		self.dec12 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='dec12')
		self.dec13 = tf.keras.layers.Dense(self.g1_sz*self.g1_sz*self.C,name='dec13')
		
		self.state_layer_collection.append(self.dec11)
		self.state_layer_collection.append(self.dec12)
		self.state_layer_collection.append(self.dec13)
		
		
	def decode(self,r):
		y = self.dec13(self.dec12(self.dec11(r)))
		return tf.reshape(y,[-1,self.g1_sz,self.g1_sz,self.C])
		
		
	def _create_decoder2(self):
		self.dec21 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='dec21')
		self.dec22 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='dec22')
		self.dec23 = tf.keras.layers.Dense(self.g2_sub_sz*self.g2_sub_sz*self.C,name='dec23')
		
		self.state_layer_collection.append(self.dec21)
		self.state_layer_collection.append(self.dec22)
		self.state_layer_collection.append(self.dec23)
		
		
	def decode2(self,r):
		y = self.dec23(self.dec22(self.dec21(r)))
		return tf.reshape(y,[-1,self.g2_sub_sz,self.g2_sub_sz,self.C])
	
	
			
	###########################################################################################
	############################## ACT2 NET ###################################################
	###########################################################################################
	
	
	def _create_action2(self):
		self.fa_prev = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fa_prev')
		self.fa_feed_r = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fa_feed_r')
		self.fa_feed_a1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fa_feed_a1')
		self.fa_feed_merge = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fa_feed_merge')
		self.fa_merge = tf.keras.layers.Dense(self.a2_sz,activation=tf.nn.leaky_relu,name='fa_merge')
		
		
		self.action_layer_collection.append(self.fa_prev)
		self.action_layer_collection.append(self.fa_feed_r)
		self.action_layer_collection.append(self.fa_feed_a1)
		self.action_layer_collection.append(self.fa_feed_merge)
		self.action_layer_collection.append(self.fa_merge)
		
		
	
	def update_action2(self,a2,r2,a1_feed):
		h_prev = self.fa_prev(a2)
		h_feed_r = self.fa_feed_r(r2)
		h_feed_a1 = self.fa_feed_a1(a1_feed)
		h_feed = self.fa_feed_merge(tf.concat([h_feed_r,h_feed_a1],axis=-1))

		h = tf.concat([h_prev,h_feed],axis=-1)
		return self.fa_merge(h)
		
		
		
	def _create_loc_net2(self):
		self.fa_loc1 = tf.keras.layers.Dense(128,name='fa_loc1',activation=tf.nn.leaky_relu)
		self.fa_loc2 = tf.keras.layers.Dense(2,name='fa_loc2',kernel_initializer='zeros')
		self.fa_z1 = tf.keras.layers.Dense(256,name='fa_z1',activation=tf.nn.leaky_relu)
		self.fa_z2 = tf.keras.layers.Dense(256,name='fa_z2',activation=tf.nn.leaky_relu)
		self.fa_z3 = tf.keras.layers.Dense(self.z_sz,name='fa_z3',kernel_initializer='zeros')
		
		self.action_layer_collection.append(self.fa_loc1)
		self.action_layer_collection.append(self.fa_loc2)
		self.action_layer_collection.append(self.fa_z1)
		self.action_layer_collection.append(self.fa_z2)
		self.action_layer_collection.append(self.fa_z3)
		
	def loc_net2(self,a2):
		return self.fa_loc2(self.fa_loc1(a2)),self.fa_z3(self.fa_z2(self.fa_z1(a2)))
		
	
			
	###########################################################################################
	############################## STATE2 NET #################################################
	###########################################################################################
		
	
	
	def _create_state2(self):
		self.fs_prev = tf.keras.layers.Dense(256,name='fs_prev',activation=tf.nn.leaky_relu)
		self.fs_feed_loc = tf.keras.layers.Dense(256,name='fs_feed_loc',activation=tf.nn.leaky_relu)
		self.fs_feed_r1 = tf.keras.layers.Dense(256,name='fs_feed_r1',activation=tf.nn.leaky_relu)
		self.fs_feed_merge = tf.keras.layers.Dense(256,name='fs_feed_merge',activation=tf.nn.leaky_relu)
		self.fs_merge = tf.keras.layers.Dense(self.r2_sz,name='fs_merge',activation=tf.nn.leaky_relu)
			
		self.state_layer_collection.append(self.fs_prev)
		self.state_layer_collection.append(self.fs_feed_loc)
		self.state_layer_collection.append(self.fs_feed_r1)
		self.state_layer_collection.append(self.fs_feed_merge)
		self.state_layer_collection.append(self.fs_merge)
		
		
	def update_state2(self,r2,loc2,r1_feed):
		h_prev = self.fs_prev(r2)
		h_feed_loc2 = self.fs_feed_loc(loc2)
		h_feed_r1 = self.fs_feed_r1(r1_feed)
		h_feed_merge = self.fs_feed_merge(tf.concat([h_feed_loc2,h_feed_r1],axis=-1))
		h = tf.concat([h_prev,h_feed_merge],axis=-1)
		
		return self.fs_merge(h)
		
		
		
		
	###########################################################################################
	############################## HYPERNETS ##################################################
	###########################################################################################
		
		
		
	##########################################
	################# ACTION #################
	##########################################
	
	def _create_action_hypernet(self):
		
		if self.hyper:
			self.fa_W1 = tf.keras.layers.Dense((self.a1_sz+self.r1_sz)*self.a1_sz,kernel_initializer=hyperfanin_for_kernel(fanin=(self.a1_sz+self.r1_sz)),name='HaW')
			self.fa_b1 = tf.keras.layers.Dense(self.a1_sz,kernel_initializer=hyperfanin_for_bias(),name='Hab')
			
			self.action_layer_collection.append(self.fa_W1)
			self.action_layer_collection.append(self.fa_b1)
			
			
			self.ln_W1 = tf.keras.layers.Dense(self.a1_sz*2,kernel_initializer=hyperfanin_for_kernel(fanin=self.in_fa1_sz),name='HlW')
			self.ln_b1 = tf.keras.layers.Dense(2,kernel_initializer=hyperfanin_for_bias(),name='Hlb')

			self.action_layer_collection.append(self.ln_W1)
			self.action_layer_collection.append(self.ln_b1)
			
		else:
			self.act_base_theta1 = tf.keras.layers.Dense(512,name='act_base_theta1',activation=tf.nn.leaky_relu)
			self.act11 = tf.keras.layers.Dense(self.a1_sz,name='act11',activation=tf.nn.leaky_relu)
			
			self.action_layer_collection.append(self.act_base_theta1)
			self.action_layer_collection.append(self.act11)
			
			
			self.act_loc_theta1 = tf.keras.layers.Dense(256,name='act_loc_theta1',activation=tf.nn.leaky_relu)
			self.loc11 = tf.keras.layers.Dense(2,name='loc11',activation=None)
			
			self.action_layer_collection.append(self.act_loc_theta1)
			self.action_layer_collection.append(self.loc11)
		
		
	def act1(self,z):
		theta_act = self.act_base_theta1(z)
		theta_loc = self.act_loc_theta1(z)
		return lambda x : self.act11(tf.concat([x,theta_act],axis=-1)), lambda x : self.loc11(tf.concat([x,theta_loc],axis=-1))
		
		
		
		
	def generate_fa1(self, z):
		
		fa_W1 = self.fa_W1(z)
		fa_b1 = self.fa_b1(z)
		fa_theta = tf.concat([fa_W1,fa_b1],axis=-1)
		
		ln_W1 = self.ln_W1(z)
		ln_b1 = self.ln_b1(z)
		ln_theta = tf.concat([ln_W1,ln_b1],axis=-1)

		return fa_theta, ln_theta;
		
		
	def _create_baseline(self):
		self.base1 = tf.keras.layers.Dense(self.T,name='base1',activation=None)
		self.base2 = tf.keras.layers.Dense(self.T2,name='base2',activation=None)
		
		self.action_layer_collection.append(self.base1)
		self.action_layer_collection.append(self.base2)
		
	def baseline1(self,x):
		return self.base1(x);
		
	def baseline2(self,x):
		return self.base2(x);
		
	
	##########################################
	################# STATE ##################
	##########################################
	
	def _create_state_hypernet(self):
		
		self.st_hyp1 = tf.keras.layers.Dense(256,name='hs1',activation=tf.nn.leaky_relu)
		self.st_hyp2 = tf.keras.layers.Dense(256,name='hs2',activation=tf.nn.leaky_relu)
		
		
		self.state_layer_collection.append(self.st_hyp1)
		self.state_layer_collection.append(self.st_hyp2)
		
		if self.hyper:
			self.st_hyp3 = tf.keras.layers.Dense(self.z_sz,name='hs3',activation=None)
			self.st_W1 = tf.keras.layers.Dense((self.r1_sz+self.bottleneck+2)*self.r1_sz,name='hsW',kernel_initializer=hyperfanin_for_kernel(fanin=(self.r1_sz+self.bottleneck+2)))
			self.st_b1 = tf.keras.layers.Dense(self.r1_sz,name='hsb',kernel_initializer=hyperfanin_for_bias())

			self.state_layer_collection.append(self.st_W1)
			self.state_layer_collection.append(self.st_b1)
			self.state_layer_collection.append(self.st_hyp3)
		else:
			self.st_base_theta1 = tf.keras.layers.Dense(512,name='st_base_theta1',activation=tf.nn.leaky_relu)
			self.state11 = tf.keras.layers.Dense(self.r1_sz,name='state11',activation=tf.nn.leaky_relu)
			
			self.state_layer_collection.append(self.st_base_theta1)
			self.state_layer_collection.append(self.state11)
		
		
		
	def state1(self,z):
		theta_st = self.st_base_theta1(z)
		return lambda x : self.state11(tf.concat([x,theta_st],axis=-1))
		
		
	def generate_fs1(self, r2):
	
		z = self.st_hyp3(self.st_hyp2(self.st_hyp1(r2)))
	
		W1 = self.st_W1(z)
		b1 = self.st_b1(z)
		theta = tf.concat([W1,b1],axis=-1)
		return theta;
		
		
		
	###########################################################################################
	############################## AUX NETS ###################################################
	###########################################################################################
	
	
	# R2 initialization
	def _create_r2_init(self):
		self.r2_init_flatten = tf.keras.layers.Flatten()
		self.r2_init1 = tf.keras.layers.Dense(self.r2_sz,activation=tf.nn.leaky_relu,name='r2_init1')
		self.state_layer_collection.append(self.r2_init_flatten)
		self.state_layer_collection.append(self.r2_init1)
		
	def init_state2(self,x):
		return self.r2_init1(self.r2_init_flatten(x))
	
	# R2 -> R1 initialization
	def _create_state1_init(self):
		self.state_init1 = tf.keras.layers.Dense(self.r1_sz,activation=tf.nn.leaky_relu,name='state_init1')
		self.state_layer_collection.append(self.state_init1)
		
	def init_state1(self,x):
		return self.state_init1(x);
		
	
	
	# A2 -> A1 initialization
	def _create_action1_init(self):
		self.action_init1 = tf.keras.layers.Dense(self.a1_sz,activation=tf.nn.leaky_relu,name='action_init1')
		self.action_layer_collection.append(self.action_init1)
		
	def init_action1(self,x):
		return self.action_init1(x);
		
		
		
	# R1 -> R2 feedback
	def _create_state_feedback(self):
		self.fs_feed1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='state_feedback1')
		self.fs_feed2 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='state_feedback2')
		self.fs_feed3 = tf.keras.layers.Dense(self.r2_sz,activation=tf.nn.leaky_relu,name='state_feedback3')
		self.state_layer_collection.append(self.fs_feed1)
		self.state_layer_collection.append(self.fs_feed2)
		self.state_layer_collection.append(self.fs_feed3)
		
	def state_feedback(self,x):
		return self.fs_feed3(self.fs_feed2(self.fs_feed1(x)));
		
		
		
	# A1 -> A2 feedback
	def _create_action_feedback(self):
		self.fa_feed1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='action_feedback1')
		self.fa_feed2 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='action_feedback2')
		self.fa_feed3 = tf.keras.layers.Dense(self.a2_sz,activation=tf.nn.leaky_relu,name='action_feedback3')
		self.action_layer_collection.append(self.fa_feed1)
		self.action_layer_collection.append(self.fa_feed2)
		self.action_layer_collection.append(self.fa_feed3)
		
	def action_feedback(self,x):
		return self.fa_feed3(self.fa_feed2(self.fa_feed1(x)));
		
	def sample_g2(self,g2):
		return tf.image.resize(g2,[self.g2_sub_sz,self.g2_sub_sz])

		
	###########################################################################################
	############################## VAR ACCOUNTING #############################################
	###########################################################################################
	
		
	def get_state_vars(self):
		var_list = []
	
		for layer in self.state_layer_collection:
			var_list += layer.trainable_variables;
		return var_list;
		
		
	def get_action_vars(self):
		var_list = []
	
		for layer in self.action_layer_collection:
			var_list += layer.trainable_variables;
		return var_list;
		
		
		
		
	###########################################################################################
	############################## TRAIN OPS ##################################################
	###########################################################################################
	
		
		

	def train_step(self, inputs):
	
		with tf.GradientTape(persistent=True) as tape:

			state_vars = self.get_state_vars()
			action_vars = self.get_action_vars()
			all_vars = action_vars + state_vars;
			
			tape.watch(all_vars)
			
			self.forward_pass(inputs)
				
			fp_loss = tf.reduce_mean(self.pred_error());
			fa_loss, baseline_mse = self.REINFORCE();
			

			### Sanity check
			total_size = 0.0
			
			for vr in all_vars:
				vr_sz = np.prod(vr.get_shape().as_list())
				total_size += vr_sz
				
			print(total_size)
			
			loss = fp_loss + baseline_mse + fa_loss
			
			
		gradients = tape.gradient(loss, all_vars)
		gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		self.optimizer.apply_gradients(zip(gradients, all_vars))
		
		pred = self.activations['I_hat'][-1]
		self.compiled_metrics.update_state(inputs[0],pred)

		return {m.name: m.result() for m in self.metrics}

	
	
	def test_step(self, inputs):
		self.forward_pass(inputs)
			
		pred = self.activations['I_hat'][-1]
		self.compiled_metrics.update_state(inputs[0],pred)
		
		return {m.name: m.result() for m in self.metrics}


	def call(self, inputs):
		return self.forward_pass(inputs)
		
	
		
		
	###########################################################################################
	############################## FUNCTIONAL OPS #############################################
	###########################################################################################
	
	def f_theta(self,x,theta,sz,in_sz):
	
		num_layers = len(sz)
		
		offset = 0;
		
		y = tf.reshape(x,[-1,1,in_sz])
		
		for i in range(num_layers):
			out_sz = sz[i]
			
			W_sz = in_sz*out_sz
			b_sz = out_sz
			
			W = tf.slice(theta,[0,offset],[-1,W_sz])
			offset += W_sz
			
			b = tf.slice(theta,[0,offset],[-1,b_sz])
			offset += b_sz
			

			W = tf.reshape(W,[-1,in_sz,out_sz])
			b = tf.reshape(b,[-1,1,out_sz])

			y = tf.matmul(y,W)+b
			
			if i<num_layers-1:
				y = tf.nn.leaky_relu(y)

			in_sz = out_sz

		y = tf.squeeze(y,axis=1)
		
		return y;
			
	
