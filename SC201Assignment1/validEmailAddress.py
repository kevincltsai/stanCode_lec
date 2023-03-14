"""
File: validEmailAddress.py
Name: 
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: TODO:
"""
import os

WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.7]
]

#DATA_FILE = 'SC201Assignment1/is_valid_email.txt'     # This is the file name to be processed
DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	#current_directory()
	maybe_email_list = read_in_data()

	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		print("maybe_email:", maybe_email, " feature_vector :", feature_vector)
		# TODO:


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0: #'@' in the str
			feature_vector[i] = 1 if '@' in maybe_email else 0
		elif i == 1: #No '.' before '@'
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0
		elif i == 2: #Some strings before '@'
			if feature_vector[0]:
   				feature_vector[i] = 1 if bool(maybe_email.split('@')[0]) else 0
		elif i == 3: #Some strings after '@'
			if feature_vector[0]:
   				feature_vector[i] = 1 if bool(maybe_email.split('@')[1]) else 0
		elif i == 4: #There is '.' after '@'
			if feature_vector[0]:
   				feature_vector[i] = 1 if any('.' in x for x in maybe_email.split('@')[1:]) else 0
		elif i == 5: # There is no white space
			if feature_vector[0]:
   				feature_vector[i] = 1 if any(char.isspace() for char in maybe_email) else 0
		elif i == 6: # Ends with '.com'
			if feature_vector[0]:
   				feature_vector[i] = 1 if bool(maybe_email.endswith('.com')) else 0
		elif i == 7: # Ends with '.edu'
			if feature_vector[0]:
   				feature_vector[i] = 1 if bool(maybe_email.endswith('.edu')) else 0
		elif i == 8: # ends with '.tw'
			if feature_vector[0]:
   				feature_vector[i] = 1 if bool(maybe_email.endswith('.tw')) else 0
		elif i == 9: # Length > 10
			if feature_vector[0]:
   				feature_vector[i] = 1 if len(maybe_email) > 10 else 0
				
		
			
		###################################
		#                                 #
		#              TODO:              #
		#                                 #
		###################################
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	
	with open(DATA_FILE, "r") as f:
		lines = f.read().splitlines()
	return lines

if __name__ == '__main__':
	main()


