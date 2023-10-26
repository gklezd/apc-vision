

import pickle
import argparse
import numpy as np




parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model name.",default='', type=str, required=True)
parser.add_argument("--var", help="Which variable to config.",default='', type=str, required=True)
parser.add_argument("--val", help="Which variable to set it to.", default='.', type=str, required=True)
parser.add_argument("--type", help="Type of variable.", default='', type=str, required=True)


args = parser.parse_args()
MODEL = args.model
VAR = args.var
VAL = args.val
TYPE = args.type




	
FNM = './models/' + MODEL + '/config.pkl'


	


fp = open(FNM,'rb')
cfg = pickle.load(fp)
fp.close()



	
if TYPE == 'int':
	VAL = np.int32(VAL)
elif TYPE == 'bool':
	VAL = (VAL=='True')
elif TYPE == 'float':
	VAL = np.float32(VAL)

setattr(cfg,VAR,VAL)



fp = open(FNM,'wb')
pickle.dump(cfg,fp)
fp.close()


for key in cfg.__dict__.keys():
	print(key + ' : ' + str(cfg.__dict__[key]))






