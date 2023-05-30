"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
import pandas as pd
import sklearn 
import util

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""

	d = pd.read_csv(filename)
	if mode == 'Train':	
		feature_name = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		d = d[feature_name]
		d.dropna(subset=['Age', 'Embarked'],inplace=True)

	elif mode == 'Test':
		feature_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		d = d[feature_name]

		train_age_avg = round(sum(training_data.get('Age')) / len(training_data.get('Age')),3)
		train_fare_avg = round(sum(training_data.get('Fare')) / len(training_data.get('Fare')),3)
		
		d['Age'].fillna(train_age_avg, inplace = True)	
		d['Fare'].fillna(train_fare_avg, inplace = True)	

	# Changing 'male' to 1, 'female' to 0
	d.loc[d.Sex == 'male', 'Sex'] = 1
	d.loc[d.Sex == 'female', 'Sex'] = 0

	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	d.loc[d.Embarked == 'S', 'Embarked'] = 0
	d.loc[d.Embarked == 'C', 'Embarked'] = 1
	d.loc[d.Embarked == 'Q', 'Embarked'] = 2


	data = d.to_dict(orient='list')

	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	
	"""
	unique_list = []
	
	for x in data.get(feature):
		if x not in unique_list:
			unique_list.append(x)
	
	print(unique_list)

	data = pd.DataFrame(data)

	for i in unique_list:
		data[feature + '_' + str(i)] = 0	
		data.loc[data[feature] == i, feature + '_' + str(i)] = 1
	
	data.pop(feature)
	data = data.to_dict(orient='list')
	
	"""
	data = pd.DataFrame(data)
	if feature == 'Sex':
		data['Sex_0'] = 0
		data['Sex_1'] = 0

		data.loc[data.Sex == 1, 'Sex_1'] = 1
		data.loc[data.Sex == 0, 'Sex_0'] = 1
		data.pop('Sex')
	
	elif feature == 'Pclass':
		data['Pclass_0'] = 0
		data['Pclass_1'] = 0
		data['Pclass_2'] = 0

		data.loc[data.Pclass == 1, 'Pclass_0'] = 1
		data.loc[data.Pclass == 2, 'Pclass_1'] = 1
		data.loc[data.Pclass == 3, 'Pclass_2'] = 1
		
		data.pop('Pclass')
	
	elif  feature == 'Embarked':
		data['Embarked_0'] = 0
		data['Embarked_1'] = 0
		data['Embarked_2'] = 0

		data.loc[data.Embarked == 0, 'Embarked_0'] = 1
		data.loc[data.Embarked == 1, 'Embarked_1'] = 1
		data.loc[data.Embarked == 2, 'Embarked_2'] = 1
		
		data.pop('Embarked')
	
	data = data.to_dict(orient='list')
	
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	data = pd.DataFrame(data)

	data = (data - data.min()) / (data.max() - data.min())

	data = data.to_dict(orient='list')

	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0


	def sigmoid(k):
    		return 1 / (1 + math.exp(-k))
	# Step 2 : Start training
	# TODO:
	# Step 3 : Feature Extract
	# TODO:
	# Step 4 : Update weights
	# TODO:

	length = len(inputs[keys[0]])

	for epoch in range(num_epochs):
		for r in range(length):
			if degree == 1:
				x = {k:v[r] for k, v in inputs.items()}
			elif degree == 2:
				x = {k:v[r] for k, v in inputs.items()}
				for i in range(len(keys)):
					for j in range(i, len(keys)):
						x[keys[i] + keys[j]] = inputs[keys[i]][r] * inputs[keys[j]][r]

			h = sigmoid(util.dotProduct(weights, x))
			scale = -alpha*(h-labels[r])
			util.increment(weights, scale, x)

	return weights
