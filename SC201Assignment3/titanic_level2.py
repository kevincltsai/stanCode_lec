"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None

	# Changing 'male' to 1, 'female' to 0
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0

	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	################################
	#                              #
	#             TODO:            #
	#                              #
	################################
	if mode == 'Train':	
		feature_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		data.dropna(subset=['Age', 'Embarked'],inplace=True)
		
		labels = data['Survived']
		data = data[feature_name]
		return data, labels

	elif mode == 'Test':
		feature_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		data = data[feature_name]
		train_age_avg = round(sum(training_data.get('Age')) / len(training_data.get('Age')),3)
		train_fare_avg = round(sum(training_data.get('Fare')) / len(training_data.get('Fare')),3)
		
		data['Age'].fillna(train_age_avg, inplace = True)	
		data['Fare'].fillna(train_fare_avg, inplace = True)	
		data.dropna(subset=['Age', 'Embarked'],inplace=True)
		
		return data

def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
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


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	############################
	#                          #
	#          TODO:           #
	#                          #
	############################
	scaler = preprocessing.StandardScaler()
	data = scaler.fit_transform(data)
	
	return data

def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.8019662921348315
	TODO: real accuracy on degree2 -> 0.8370786516853933
	TODO: real accuracy on degree3 -> 0.8764044943820225
	"""
	train, labels = data_preprocess(TRAIN_FILE)
	train = one_hot_encoding(train,'Sex')
	train = one_hot_encoding(train,'Pclass')
	train = one_hot_encoding(train,'Embarked')
	
	train = pd.DataFrame(train)
	scaler = preprocessing.StandardScaler()
	train = scaler.fit_transform(train)

	poly = preprocessing.PolynomialFeatures(degree=3)
	train = poly.fit_transform(train)
	
	model = linear_model.LogisticRegression(max_iter=10000)
	model.fit(train,labels)

	print(model.score(train,labels))

if __name__ == '__main__':
	main()
