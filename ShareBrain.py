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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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
random_number = 0
test_split = 0.2
X_train, X_test, y_train, y_test = train_test_split(
	price_input, boolean_target, test_size=test_split, random_state=random_number)

# Feature scale the training data and apply the scaling to the training and test datasets.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Set up the MLPClassifier
clf = MLPClassifier(
	activation = 'logistic',
	solver ='lbfgs',
	hidden_layer_sizes=(2),
	alpha = 1E-7,
	tol = 1E-5,
	verbose = False )

# Train the network on the dataset until the test_accuracy raches a threshold
accuracy_check = True
while accuracy_check:
	# Train the neural network on the dataset
	clf.fit(X_train, y_train)

	# Use the cross validation set to calculate the accuracy of the network
	test_accuracy = clf.score(X_test, y_test)
	train_accuracy = clf.score(X_train, y_train)
	print("The network fitted the test data {:.3f}% and the training data {:.3f}%."
		.format(test_accuracy*100, train_accuracy*100))
	accuracy_check = test_accuracy < 0.8

# Build an array of indices to order the train and test data back in the plot
indices = np.arange(len(boolean_target))
i_train, i_test, iy_train, iy_test = train_test_split(
	indices, indices, test_size=test_split, random_state=random_number)

# Extract closing price data for test and training data
# rescaled back to original price
close_price_test = scaler.inverse_transform(X_test)[:,-1]
close_price_train = scaler.inverse_transform(X_train)[:,-1]

# Generate the predictions for the test data
predictions = clf.predict(X_test)
predictions_prob = clf.predict_proba(X_test)

# Separate the correct and incorrect predictions into their own arrays
correct_predictions = np.array([])
correct_predictions_prob = np.array([])
i_correct_predictions = np.array([])
incorrect_predictions = np.array([])
incorrect_predictions_prob = np.array([])
i_incorrect_predictions = np.array([])

for n in range(0,len(predictions)):
	if predictions[n] == y_test[n]:
		correct_predictions = np.append(correct_predictions, close_price_test[n])
		correct_predictions_prob = np.append(correct_predictions, predictions_prob[n])
		i_correct_predictions = np.append(i_correct_predictions, i_test[n])
	else :
		incorrect_predictions = np.append(incorrect_predictions, close_price_test[n])
		incorrect_predictions_prob = np.append(incorrect_predictions, predictions_prob[n])
		i_incorrect_predictions = np.append(i_incorrect_predictions, i_test[n])

# Plot the test and training data
fig, ax = plt.subplots()
ax.plot(i_train, close_price_train,'b.')
ax.plot(i_test, close_price_test,'m.')

# Shade the regions where the predictions are made
# This is set up to colour the next price data point from when the prediction is made
for n in range(0,len(correct_predictions)):
	ax.fill_between(i_correct_predictions[n]+[0.5, 1.5], 0, 100, facecolor='g', linewidth=0)
for n in range(0,len(incorrect_predictions)):
	ax.fill_between(i_incorrect_predictions[n]+[0.5, 1.5], 0, 100, facecolor='r', linewidth=0)

# Display the plot
ax.set_ylim(bottom= np.min(np.append(close_price_test,close_price_train))-2, top=np.max(np.append(close_price_test,close_price_train))+2)
ax.set_xlim(left=0, right= len(boolean_target))
plt.show()