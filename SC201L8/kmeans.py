import random
import math
import sys
import time
from collections import defaultdict


def k_means(points, centroids, num_iters):
	"""
	:param points: List[Tuple], containing all the data points to be segmented
	:param centroids: List[Tuple], containing K number of centroids
	:param num_iters: int, the number of iterations to run k-means algorithm
	:return: centroids: List[Tuple], containing K number of new centroids after learning
	"""
	raise Exception('NOT IMPLEMENTED YET...')



############## DO NOT EDIT THE CODES BELOW THIS LINE ##############


def problem2_a():
	points0 = generate0()
	centroids= k_means(points0, [(2, 3), (2, -1)], 3)
	print('Centroids:', centroids)


def problem2_b():
	points0 = generate0()
	centroids = k_means(points0, [(0, 1), (3, 2)], 3)
	print('Centroids:', centroids)


def problem2_c():
	points1 = generate1()
	tic = time.time()
	centroids = k_means(points1, [(0.27, 0.27), (0.43, 0.43)], 5000)
	print('Centroids:', centroids)


def problem2_d():
	points1 = generate1()
	tic = time.time()
	centroids = k_means(points1, [(0.27, 0.27), (0.35, 0.35) ,(0.43, 0.43)], 5000)
	print('Centroids:', centroids)
	

def generate0():
	return [(1, 0), (1, 2), (3, 0), (2, 2)]


def generate1():
	random.seed(42)
	x1 = [random.randint(30, 50)/100 for i in range(100)]
	y1 = [random.randint(28, 48)/100 for i in range(100)]
	x2 = [random.randint(18, 38)/100 for i in range(100)]
	y2 = [random.randint(18, 38)/100 for i in range(100)]
	data1 = []
	for i in range(len(x1)):
		data1.append((x1[i], y1[i]))
	data2 = []
	for i in range(len(x2)):
		data1.append((x2[i], y2[i]))
	all_data = data1 + data2
	random.shuffle(all_data)
	return all_data


def main():
	args = sys.argv[1:]
	if len(args) == 0:
		print('Please Enter the test you would like to run.')
		print('For example:')
		print('Python3 kmeans.py 2a')
		print('Python3 kmeans.py 2b')
		print('Python3 kmeans.py 2c')
		print('Python3 kmeans.py 2d')
	elif len(args) == 1:
		if args[0] == '2a':
			problem2_a()
		elif args[0] == '2b':
			problem2_b()
		elif args[0] == '2c':
			problem2_c()
		elif args[0] == '2d':
			problem2_d()
		else:
			print('Illegal...')
	else:
		print('Illegal...')


if __name__ == '__main__':
	main()
