#!/usr/bin/python
# Python 2 Code

import os, sys

parser_dir = './stanford_parser/'
parser_script = parser_dir + 'lexparser.sh'
descriptions_dir = './descriptions_2_5k/'
parsed_out_dir = './parsed_outputs_2_5k/'

TOTAL = 2688

for indx in range(0, TOTAL):
	os.system(parser_script + ' ' +
		descriptions_dir + 'description_' + str(indx) +'.txt' + ' > ' +
		parsed_out_dir + 'parsed_output_' + str(indx) + '.txt')

for indx in range(0, TOTAL):
	filename = parsed_out_dir + 'parsed_output_' + str(indx) + '.txt'
	parsed_tree = ''
	with open(filename, 'r') as f:
		count = 0
		parsed_tree = ''
		for line in f.readlines():
			parsed_tree += line
		end_indx = parsed_tree.rfind(';')
		for i in range(end_indx + 1, len(parsed_tree)):
			if parsed_tree[i] != ')':
				end_indx = i
				break
		parsed_tree = parsed_tree[:end_indx + 1]
	with open(filename, 'w') as f:
		f.write(parsed_tree)
