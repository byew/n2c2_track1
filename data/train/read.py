import ipdb

with open('./clinicalSTS2019.train.txt') as f:
	data = [line.strip().split('\t') for line in f.readlines()]
	data = [[line[0], line[1] , float(line[2])] for line in data]
ipdb.set_trace()