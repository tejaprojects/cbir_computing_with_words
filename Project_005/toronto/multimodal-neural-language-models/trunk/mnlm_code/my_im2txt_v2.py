#!/usr/bin/python
# Python 2 Code

import expr
import proc
from utils import stop

net = stop.load_model('/home/teja/Project_005/toronto/models/mlbl.pkl')
(z, zt) = proc.process()

TOTAL = 2688

for img_indx in range(0, TOTAL):
	captions = expr.im2txt(net, z, zt['IM'][img_indx], k=1, shortlist=25)
	for c in captions:
		desc = ' '.join(c)
	filename = 'descriptions_2_5k/description_' + str(img_indx) + '.txt'
	with open(filename, 'w') as f:
		f.write(desc)
