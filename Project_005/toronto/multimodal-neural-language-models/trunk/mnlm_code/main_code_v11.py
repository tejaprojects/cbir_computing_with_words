#!/usr/bin/python
# Python 2 Code

# Final Code For Image Retrieval
# Using Dataset with 10 Classes

import os, sys, cv2, time, math

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

from operator import itemgetter
from utils import stop

import expr, proc

TOTAL = 1000
TOP = 20

img_indx = input('Enter a query image [0 to 999]: ')

desc = ''
desc_file = 'descriptions/description_' + str(img_indx) + '.txt'
with open('description.txt', 'w') as ff:
	with open(desc_file, 'r') as f:
		desc = f.readlines()[0]
		for line in f.readlines():
			ff.write(line)

# print 'Description: ' + desc

parse_file = 'parsed_outputs/parsed_output_' + str(img_indx) + '.txt'
with open('parsed_output.txt', 'w') as ff:
	with open(parse_file, 'r') as f:
		for line in f.readlines():
			ff.write(line)

filename = 'parsed_output.txt'
parsed_tree = ''
with open(filename, 'r') as f:
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

sentence = ''
with open('parsed_output.txt', 'r') as file:
	for line in file.readlines():
		sentence += line
sentence = sentence.strip()
parts = sentence.split(' ')

count = len(parts)
nouns_attribs = dict()
for i in range(count):
	if parts[i] == '(NN' or parts[i] == '(NNS':
		noun = parts[i + 1].strip(')')
		nouns_attribs[noun] = ['none']

attribs = list()
for i in range(count):
	if parts[i] == '(JJ':
		for j in range(i + 1, count):
			if parts[j] == '(PP':
				break
			elif parts[j] == '(NN' or parts[j] == '(NNS':
				noun = parts[j + 1].strip(')')
				nouns_attribs[noun] = nouns_attribs[noun] + [parts[i + 1].strip(')')]
				break

for i in nouns_attribs.keys():
	nouns_attribs[i].remove('none')

parsed_out_dir = './parsed_outputs/'
nns_atts = list(range(0, TOTAL))
for indx in range(0, TOTAL):
	sentence = ''
	filename = parsed_out_dir + 'parsed_output_' + str(indx) + '.txt'
	with open(filename, 'r') as file:
		for line in file.readlines():
			sentence += line
	sentence = sentence.strip()
	parts = sentence.split(' ')
	count = len(parts)
	nns_atts[indx] = dict()
	for i in range(count):
		if parts[i] == '(NN' or parts[i] == '(NNS':
			noun = parts[i + 1].strip(')')
			nns_atts[indx][noun] = ['none']
	atts = list()
	for i in range(count):
		if parts[i] == '(JJ':
			for j in range(i + 1, count):
				if parts[j] == '(PP':
					break
				elif parts[j] == '(NN' or parts[j] == '(NNS':
					noun = parts[j + 1].strip(')')
					nns_atts[indx][noun] = nns_atts[indx][noun] + [parts[i + 1].strip(')')]
					break

fuzzy_sets_file = open('fuzzy_sets.txt', 'r')
fuzzy_vars = list()
fuzzy_sets = dict()
for line in fuzzy_sets_file.readlines():
	tmp = line.strip().split(',')
	if len(tmp) >= 4:
		fuzzy_vars.append(tmp[0])
		tmp_0 = tmp[0]
		tmp = tmp[1:]
		tmp_lst = np.array([float(i) for i in tmp])
		range_tmp_lst = np.arange(tmp_lst[0], tmp_lst[-1] + 1)
		mf_tmp_lst = fuzz.trapmf(range_tmp_lst, tmp_lst)
		fuzzy_sets[tmp_0] = [tmp_lst[0], tmp_lst[-1]] + list(mf_tmp_lst)

fuzzy_sets_file.close()

def fuzz_and(x, y):
	xx = fuzzy_sets[x]
	yy = fuzzy_sets[y]
	range_x = np.arange(xx[0], xx[1] + 1)
	range_y = np.arange(yy[0], yy[1] + 1)
	mf_x = np.array(xx[2:])
	mf_y = np.array(yy[2:])
	(range_and, mf_and) = fuzz.fuzzy_and(range_x, mf_x, range_y, mf_y)
	return [range_and[0], range_and[-1]] + list(mf_and)

def fuzzy_area(x):
	range_x = np.arange(x[0], x[1] + 1)
	mf_x = np.array(x[2:])
	corners = [0,0,0,0]
	area = 0.0
	count = 0
	max_mf = max(mf_x)
	min_values = list()
	max_values = list()
	for i in range(0, len(mf_x) - 1):
		if mf_x[i] == 0.0:
			min_values.append(i)
		if mf_x[i] == max_mf:
			max_values.append(i)
	corners[1] = max_values[0]
	corners[2] = max_values[-1]
	for m in min_values:
		if m <= corners[1]:
			corners[0] = m
		if m >= corners[2]:
			corners[3] = m;
			break;
	if len(corners) == 0:
		return 0.0
	else:
		area += 0.5*(corners[1]-corners[0]) + 0.5*(corners[3]-corners[2])
		area += corners[2]-corners[1]
		return area

keys = list(nouns_attribs.keys())
num_keys = len(keys)
probability = list(range(0, TOTAL))
for indx in range(0, TOTAL):
	probability[indx] = 0.0
	kys = list(nns_atts[indx].keys())
	for key in keys:
		if key in kys:
			similarity = 0.0
			if len(nouns_attribs[key]) > 0 and len(nns_atts[indx][key]) > 0:
				for value in nouns_attribs[key]:
					if value in fuzzy_vars:
						for v in nns_atts[indx][key]:
							if v in fuzzy_vars:
								result = fuzz_and(value, v)
								similarity += 2 * fuzzy_area(result)/(
									fuzzy_area(fuzzy_sets[value]) + fuzzy_area(fuzzy_sets[v])
									)
								break
						else:
							break
			probability[indx] += (1.0 + similarity)/(2.0 * num_keys)

indx_sorted, probs_sorted = zip(*sorted(
	[(indx,prob) for indx,prob in enumerate(probability)],
	key=itemgetter(1), reverse=True))

indx_sorted_backup = indx_sorted
indx_sorted = indx_sorted[0:TOP]
# print(indx_sorted)

classes = list()
tmp = 0
with open('classes_list.txt', 'r') as f:
	for line in f.readlines():
		classes.append(int(line.strip()) + tmp)
		tmp += int(line.strip())

def which_class(indx):
	for i in range(0, len(classes)):
		if indx < classes[i]:
			class_num = i
			break
	return class_num

selection = list()
for i in range(0, len(probs_sorted)):
	if probs_sorted[i] >= 0.5:
		selection.append(indx_sorted_backup[i])

tp_set = list()
fp_set = list()
query_class = which_class(img_indx)
for s in selection:
	if query_class == which_class(s):
		tp_set.append(s)
	else:
		fp_set.append(s)

tp = len(tp_set)
fp = len(fp_set)
precision = tp/float(tp+fp)

recall_denom = (classes[query_class] - classes[query_class-1]) if query_class > 0 else classes[query_class]
recall = tp/float(recall_denom)

# print('Precision: ' + str(precision))
# print('Recall: ' + str(recall))

plt.figure(1)
if TOP == 20:
	x,y,z = 5,5,1
elif TOP == 50:
	x,y,z = 6,10,1

expr.vis('test', [img_indx], save=True)
filename = 'r'+str(img_indx)+'.jpg'
image = cv2.imread(filename)
plt.subplot(x,y,z)
plt.axis('off')
plt.imshow(image)

count = y+1
for i in range(len(indx_sorted)):
	expr.vis('test', [indx_sorted[i]], save=True)
	filename = 'r'+str(indx_sorted[i])+'.jpg'
	image = cv2.imread(filename)
	plt.subplot(x,y,count)
	plt.axis('off')
	count += 1
	plt.imshow(image)
plt.show()


# CBIR (PAPER)

dataset = list()
with open('dataset_double_2.txt', 'r') as f:
	for line in f.readlines():
		line = line.strip().split(',')
		data = [float(i) for i in line[:-1]]
		dataset.append(data)

query_vec = dataset[img_indx]

dist = list()
for i in range(TOTAL):
	# Euclidean Distance
	vec = dataset[i]
	tmp_dist = 0.0
	for i in range(42):
		tmp = query_vec[i] - vec[i]
		tmp = tmp*tmp
		tmp_dist += tmp
	dist.append(math.sqrt(tmp_dist))

indx_sorted, dist_sorted = zip(*sorted(
		[(i,r) for i,r in enumerate(dist)],
		key=itemgetter(1)))

dist_sorted = dist_sorted[0:TOP]
indx_sorted = indx_sorted[0:TOP]

plt.figure(2)
if TOP == 20:
	x,y,z = 5,5,1
elif TOP == 50:
	x,y,z = 6,10,1

expr.vis('test', [img_indx], save=True)
filename = 'r'+str(img_indx)+'.jpg'
image = cv2.imread(filename)
plt.subplot(x,y,z)
plt.axis('off')
plt.imshow(image)

count = y+1
for i in range(len(indx_sorted)):
	expr.vis('test', [indx_sorted[i]], save=True)
	filename = 'r'+str(indx_sorted[i])+'.jpg'
	image = cv2.imread(filename)
	plt.subplot(x,y,count)
	plt.axis('off')
	count += 1
	plt.imshow(image)
plt.show()


# COMBINATION

final_measure = list()
for i in range(TOTAL):
	if dist[i] == 0.0:
		tmp = dist[0:i] + dist[i+1:]
		tmp = min(tmp)/2
	else:
		tmp = dist[i]
	final_measure.append(probability[i] + (1/float(tmp)))

final_indx_sorted, final_measure_sorted = zip(*sorted(
		[(i,r) for i,r in enumerate(final_measure)],
		key=itemgetter(1), reverse=True))

final_measure_sorted = final_measure_sorted[0:TOP]
final_indx_sorted = final_indx_sorted[0:TOP]

plt.figure(3)
if TOP == 20:
	x,y,z = 5,5,1
elif TOP == 50:
	x,y,z = 6,10,1

expr.vis('test', [img_indx], save=True)
filename = 'r'+str(img_indx)+'.jpg'
image = cv2.imread(filename)
plt.subplot(x,y,z)
plt.axis('off')
plt.imshow(image)

count = y+1
for i in range(len(final_indx_sorted)):
	expr.vis('test', [final_indx_sorted[i]], save=True)
	filename = 'r'+str(final_indx_sorted[i])+'.jpg'
	image = cv2.imread(filename)
	plt.subplot(x,y,count)
	plt.axis('off')
	count += 1
	plt.imshow(image)
plt.show()
