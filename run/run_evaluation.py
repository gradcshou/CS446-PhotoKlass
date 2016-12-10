from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys

import numpy as np
import tensorflow as tf

import photo_klass

import sys,re,os
import photo_klass_eval

# usage: python run_evaluation.py [n1] [n2] [increment]

(n1, n2) = (int(sys.argv[1]), int(sys.argv[2])) 
increment = int(sys.argv[3])

print("Evaluation Process Begins")
index_list = list(range(n1,n2,increment))+[n2-1]
for i in index_list:
	ckptFile = open("train/checkpoint",'w')
	str_to_be_inserted = "model_checkpoint_path: "+str('"')+ "model.ckpt-"+str(i)+str('"')
	ckptFile.write(str_to_be_inserted)
	ckptFile.close()
	#os.system("python photo_klass_eval.py")	
	photo_klass_eval.evaluate()

print("Evaluation Process Ended")
	




















