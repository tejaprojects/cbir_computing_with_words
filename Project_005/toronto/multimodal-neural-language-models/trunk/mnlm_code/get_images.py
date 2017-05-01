#!/usr/bin/python
# Python 2 Code

import os, sys

img_dir = '/home/teja/Project_005/toronto/iaprtc12/images/'
list_dir = '/home/teja/Project_005/toronto/iaprtc12_2/'
out_dir = '/home/teja/Project_005/toronto/multimodal-neural-language-models/trunk/mnlm_code/test_images/'

list_file = list_dir + 'iaprtc12_test_list.txt'
with open(list_file, 'r') as f:
	for line in f.readlines():
		img_file = img_dir + str(line.strip()) + '.jpg'
		command = 'cp ' + img_file + ' ' + out_dir
		os.system(command)
