"""
File: interactive.py
Name: 
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

import graderUtil
import util
import time
from util import *

grader = graderUtil.Grader()
submission = grader.load('submission')
util = grader.load('util')

def main():
	
	trainExamples = readExamples('polarity.train')
	validationExamples = readExamples('polarity.dev')
	featureExtractor = submission.extractWordFeatures
	#featureExtractor = submission.extractCharacterFeatures(4) 
    
	weights = submission.learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=40, alpha=0.01)
	interactivePrompt(featureExtractor, weights)


if __name__ == '__main__':
	main()