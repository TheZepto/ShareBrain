#This program was designed by Dane Laban to analyse share data and predict
#whether the share price would continue to increase in the foreseeable future.
#A neural network performs the analysis and neurolab is being used to provide
#this functionality.
print("#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@")
print("@         Welcome to ShareBrain          #")
print("# Designed and implemented by Dane Laban @")
print("@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#")

# import pdb; pdb.set_trace()

# Common libraries
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# My libraries
import sharescraper

# Boolean prediction of whether the close_price tomorrow will be greater than today.

# Get training input and targets from sharescraper
(price_input, boolean_target) = sharescraper.get_share_data_boolean_target(
	share_name='ANZ.AX',
	start_date='2006-01-01',
	end_date='2016-01-01',
	days_of_data=15,
	use_existing_data=True)

# Separate data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
	price_input, boolean_target, test_size=0.2, random_state=0)

# Feature scale the training data and apply the scaling to the training and test datasets.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Set up the MLPClassifier
clf = LogisticRegression(
	C = 1000,
	fit_intercept = True,
	penalty = 'l1',
	solver = 'liblinear',
	tol = 1E-10,
	max_iter = 1000,
	verbose = 1 )

accuracy_check = True
while accuracy_check:
	# Train the neural network on the dataset
	clf.fit(X_train, y_train)

	# Use the cross validation set to calculate the accuracy of the network
	test_accuracy = clf.score(X_test, y_test)
	train_accuracy = clf.score(X_train, y_train)
	print("The network fitted the test data {:.3f}% and the training data {:.3f}%."
		.format(test_accuracy*100, train_accuracy*100))
	accuracy_check = test_accuracy < 0.1

# import pdb; pdb.set_trace()