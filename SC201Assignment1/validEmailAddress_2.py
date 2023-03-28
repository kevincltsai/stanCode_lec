"""
File: validEmailAddress_2.py
Name: 
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  TODO: '@' in the str
feature2:  TODO: length of local or domain is 0
feature3:  TODO: “..” in email  
feature4:  TODO: local contains double quotation marks with text in between
feature5:  TODO: domain has more than 2 elements splited by "."
feature6:  TODO: contains not_allowed_char
feature7:  TODO: contains at least 2 special chars
feature8:  TODO: contains at least 4 special chars
feature9:  TODO: local starts or ends with "."
feature10: TODO: domain starts or ends with "."

Accuracy of your model: TODO:
"""
import numpy as np
import string


WEIGHT = [                           # The weight vector selected by Jerry
	[1.5],     # 0 '@' in the str
	[-5],      # 1 length of local or domain is 0
	[-0.8],    # 2 “..” in email  
	[-0.6],    # 3 local contains double quotation marks with text in between
	[0.3],     # 4 domain has more than 2 elements splited by "."
	[-5],      # 5 contains not_allowed_char 
	[-0.2],    # 6 contains at least 2 special chars
	[-0.5],    # 7 contains at least 4 special chars
	[-5],      # 8 local starts or ends with "."
	[-5]       # 9 domain starts or ends with "."
]

#DATA_FILE = 'SC201Assignment1/is_valid_email.txt'     # This is the file name to be processed
DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	#current_directory()
	maybe_email_list = read_in_data()

	i = 0
	answer = 0 
	for maybe_email in maybe_email_list:
		feature_vector = [[f] for f in feature_extractor(maybe_email)]
		
		cp = np.array(WEIGHT).T.dot(np.array(feature_vector))

		
		if i <= 12:
			if cp <= 0:
				answer += 1
		elif i > 12:
			if cp > 0:
				answer += 1
		i += 1		
		
		#print('email :', maybe_email)
		#print(' feature :', feature_vector)		
		#print('i:', i, 'cp:', cp)
	#print("correct :", answer)
	print("accurary :", answer / len(maybe_email_list))	
		
		# TODO:


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	
	local_domain_spliter_loc = maybe_email.rfind('@') #assuming last '@' splits apart local and domain

	
	local = maybe_email[0:local_domain_spliter_loc]
	domain = maybe_email[local_domain_spliter_loc+1:]
	
	not_allowed_char = ['(', ')', ',', ']', ':', ';', '<', '>', '[', '\\',  ']']
	special_char = ['!','#','$','%','&',"'",'*','+','-','/','=','?','^','_','`','{','|','}','~']
	
	#print(sum(maybe_email.count(char) for char in not_allowed_char))
	#print("email :", maybe_email)
	#print("local :", local)
	#print("domain :", domain)
	
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0: #'@' in the str
			feature_vector[i] = 1 if '@' in maybe_email else 0
		elif i == 1: #length of local and domain is 0
			feature_vector[i] = 1 if len(local) <= 0 or len(domain) <=0 else 0
		elif i == 2: # “..” in email  
			feature_vector[i] = 1 if '..' in maybe_email else 0
		elif i == 3: #local contains double quotation marks with text in between
			feature_vector[i] = 1 if local.find('"') > 0 and local[local.find('"')+1:].find('"') > 0 else 0
		elif i == 4: #domain has more than 2 elements splited by "."
			feature_vector[i] = 1 if len(domain.split('@')) >=2 else 0
		elif i == 5: # contains not_allowed_char 
   			feature_vector[i] = 1 if sum(maybe_email.count(char) for char in not_allowed_char) > 0 else 0
		elif i == 6: # contains special chars at least 2
   			feature_vector[i] = 1 if sum(maybe_email.count(char) for char in special_char) >= 2 else 1
		elif i == 7: # contains special chars at least 4
   			feature_vector[i] = 1 if sum(maybe_email.count(char) for char in special_char) >= 4 else 1
		elif i == 8: # 
			if len(local) >0:
				feature_vector[i] = 1 if local[0] == '.' or local[-1] == '.' else 0
		elif i == 9: # Length > 10
			if len(domain) >0:
   				feature_vector[i] = 1 if domain[0] == '.' or domain[-1] == '.' else 0
				
		
			
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

