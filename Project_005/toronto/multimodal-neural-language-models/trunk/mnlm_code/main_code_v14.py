#!/usr/bin/python
# Python 2 Code

# Final Code For Precision, Recall & F-Measure Calculations
# Using Dataset with 15 Classes (1500 Images)

import os, sys, cv2, math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

from operator import itemgetter

import expr, proc

TOTAL = 1500

classes = list()
tmp = 0
with open('classes_list_2k.txt', 'r') as f:
	for line in f.readlines():
		classes.append(int(line.strip()) + tmp)
		tmp += int(line.strip())

dataset = list()
with open('dataset_double_2k.txt', 'r') as f:
	for line in f.readlines():
		line = line.strip().split(',')
		data = [float(i) for i in line[:-1]]
		dataset.append(data)

def which_class(indx):
	for i in range(0, len(classes)):
		if indx < classes[i]:
			class_num = i
			break
	return class_num

num_results = list(range(10,101,10))

precision_list_cww = [0 for _ in range(len(num_results))]
recall_list_cww = [0 for _ in range(len(num_results))]
f_measure_cww = [0 for _ in range(len(num_results))]

precision_list_ct = [0 for _ in range(len(num_results))]
recall_list_ct = [0 for _ in range(len(num_results))]
f_measure_ct = [0 for _ in range(len(num_results))]

precision_list_comb = [0 for _ in range(len(num_results))]
recall_list_comb = [0 for _ in range(len(num_results))]
f_measure_comb = [0 for _ in range(len(num_results))]

for img_indx in range(TOTAL):
	query_class = which_class(img_indx)

	for N in range(len(num_results)):
		TOP = num_results[N]

		# CBIR (CWW)
		desc = ''
		desc_file = 'descriptions_2k/description_' + str(img_indx) + '.txt'
		with open('description.txt', 'w') as ff:
			with open(desc_file, 'r') as f:
				desc = f.readlines()[0]
				for line in f.readlines():
					ff.write(line)

		parse_file = 'parsed_outputs_2k/parsed_output_' + str(img_indx) + '.txt'
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

		parsed_out_dir = './parsed_outputs_2k/'
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
										if fuzzy_area(fuzzy_sets[value]) + fuzzy_area(fuzzy_sets[v]) == 0.0:
											similarity += 0.0
										else:
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

		'''
		max_prob = max(probability)
		if max_prob > 0.0:
			probability = [(p/max_prob) for p in probability]
		'''
		
		probs_sorted = probs_sorted[0:TOP]
		indx_sorted = indx_sorted[0:TOP]

		tp_set = list()
		fp_set = list()
		for i in indx_sorted:
			if query_class == which_class(i):
				tp_set.append(i)
			else:
				fp_set.append(i)

		precision = 0.0
		recall = 0.0

		tp = len(tp_set)
		fp = len(fp_set)
		precision_denom = float(tp+fp)
		precision = tp/precision_denom

		recall_denom = (classes[query_class] - classes[query_class-1]) if query_class > 0 else classes[query_class]
		recall_denom = float(recall_denom)
		recall = tp/recall_denom

		precision_list_cww[N] += precision
		recall_list_cww[N] += recall

		if float(precision+recall) > 0.0:
			f_measure_cww[N] += 2*precision*recall/float(precision+recall)

		# CBIR (PAPER)
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
			[(i,d) for i,d in enumerate(dist)],
			key=itemgetter(1), reverse=False))

		'''	
		max_dist = max(dist)
		if max_dist > 0.0:
			dist = [(d/max_dist) for d in dist]
		'''

		dist_sorted = dist_sorted[0:TOP]
		indx_sorted = indx_sorted[0:TOP]

		tp_set = list()
		fp_set = list()
		for i in indx_sorted:
			if query_class == which_class(i):
				tp_set.append(i)
			else:
				fp_set.append(i)

		precision = 0.0
		recall = 0.0

		tp = len(tp_set)
		fp = len(fp_set)
		precision_denom = float(tp+fp)
		precision = tp/precision_denom

		recall_denom = (classes[query_class] - classes[query_class-1]) if query_class > 0 else classes[query_class]
		recall_denom = float(recall_denom)
		recall = tp/recall_denom

		precision_list_ct[N] += precision
		recall_list_ct[N] += recall

		if float(precision+recall) > 0.0:
			f_measure_ct[N] += 2*precision*recall/float(precision+recall)

		# COMBINATION
		final_measure = list()
		for i in range(TOTAL):
			if dist[i] == 0.0:
				tmp = dist[0:i] + dist[i+1:]
				tmp = min(tmp)/2
			else:
				tmp = dist[i]
			if float(tmp) == 0.0:
				tmp = 1e-12
			final_measure.append(probability[i] + (1/float(tmp)))

		final_indx_sorted, final_measure_sorted = zip(*sorted(
			[(i,r) for i,r in enumerate(final_measure)],
			key=itemgetter(1), reverse=True))

		final_measure_sorted = final_measure_sorted[0:TOP]
		final_indx_sorted = final_indx_sorted[0:TOP]

		tp_set = list()
		fp_set = list()
		for i in final_indx_sorted:
			if query_class == which_class(i):
				tp_set.append(i)
			else:
				fp_set.append(i)

		precision = 0.0
		recall = 0.0

		tp = len(tp_set)
		fp = len(fp_set)
		precision_denom = float(tp+fp)
		precision = tp/precision_denom

		recall_denom = (classes[query_class] - classes[query_class-1]) if query_class > 0 else classes[query_class]
		recall_denom = float(recall_denom)
		recall = tp/recall_denom

		precision_list_comb[N] += precision
		recall_list_comb[N] += recall

		if float(precision+recall) > 0.0:
			f_measure_comb[N] += 2*precision*recall/float(precision+recall)


precision_list_cww = [(x/TOTAL) for x in precision_list_cww]
precision_list_ct = [(x/TOTAL) for x in precision_list_ct]
precision_list_comb = [(x/TOTAL) for x in precision_list_comb]

recall_list_cww = [(x/TOTAL) for x in recall_list_cww]
recall_list_ct = [(x/TOTAL) for x in recall_list_ct]
recall_list_comb = [(x/TOTAL) for x in recall_list_comb]

f_measure_cww = [(x/TOTAL) for x in f_measure_cww]
f_measure_ct = [(x/TOTAL) for x in f_measure_ct]
f_measure_comb = [(x/TOTAL) for x in f_measure_comb]

print('Precision-Recall Calculations Done')


plt.figure(1)
graph_1, = plt.plot(recall_list_cww, precision_list_cww, 'ro--', label='CBIR using CWW')
graph_2, = plt.plot(recall_list_ct, precision_list_ct, 'bs--', label='CBIR using Color & Texture Features')
graph_3, = plt.plot(recall_list_comb, precision_list_comb, 'g^--', label='CBIR using Both')
plt.axis([0,1.2,0,1.2])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall')
plt.legend(handles=[graph_1, graph_2, graph_3], loc=2, prop={'size':10})
plt.show()

plt.figure(2)
graph_1, = plt.plot(num_results, f_measure_cww, 'ro--', label='F-Measure using CWW')
graph_2, = plt.plot(num_results, f_measure_ct, 'bs--', label='F-Measure using Color & Texture Features')
graph_3, = plt.plot(num_results, f_measure_comb, 'g^--', label='F-Measure using Both')
plt.xlabel('Number of Retrieved Images')
plt.ylabel('F-Measure')
plt.title('F-Measure')
plt.legend(handles=[graph_1, graph_2, graph_3], loc=2, prop={'size':10})
plt.show()
