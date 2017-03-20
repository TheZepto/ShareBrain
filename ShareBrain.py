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
	start_date='1990-01-01',
	end_date='2016-01-01',
	days_of_data=10,
	use_existing_data=True)

# Separate data into training set and test set
random_number = 0
test_split = 0.3
X_train, X_test, y_train, y_test = train_test_split(
	price_input, boolean_target, test_size=test_split, random_state=random_number)

# Feature scale the training data and apply the scaling to the training and test datasets.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Set up the MLPClassifier
clf = MLPClassifier(
	activation = 'tanh',
	learning_rate = 'adaptive',
	solver ='adam',
	hidden_layer_sizes=(10),
	alpha = 0.01,
	max_iter = 10000,
	tol = 1E-8,
	warm_start = False,
	verbose = True )

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
	accuracy_check = test_accuracy < 0.1

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
true_pos = []
true_pos_prob = []
i_true_pos = []
true_neg = []
true_neg_prob = []
i_true_neg = []
false_pos = []
false_pos_prob = []
i_false_pos = []
false_neg = []
false_neg_prob = []
i_false_neg = []

for n in range(0,len(predictions)):
	if predictions[n] == True and y_test[n] == True:
		true_pos = np.append(true_pos, close_price_test[n])
		true_pos_prob = np.append(true_pos_prob, predictions_prob[n,1])
		i_true_pos = np.append(i_true_pos, i_test[n])
	if predictions[n] == False and y_test[n] == False:
		true_neg = np.append(true_neg, close_price_test[n])
		true_neg_prob = np.append(true_neg_prob, predictions_prob[n,0])
		i_true_neg = np.append(i_true_neg, i_test[n])
	if predictions[n] == True and y_test[n] == False:
		false_pos = np.append(false_pos, close_price_test[n])
		false_pos_prob = np.append(false_pos_prob, predictions_prob[n,1])
		i_false_pos = np.append(i_false_pos, i_test[n])
	if predictions[n] == False and y_test[n] == True:
		false_neg = np.append(false_neg, close_price_test[n])
		false_neg_prob = np.append(false_neg_prob, predictions_prob[n,0])
		i_false_neg = np.append(i_false_neg, i_test[n])

#Calculate precision, recall and F1 scores and display them
try:	
	precision = len(true_pos) / (len(true_pos) + len(false_pos))
	recall = len(true_pos) / (len(true_pos) + len(false_neg))
	F1_score = 2 * precision * recall / (precision + recall)
except:
	precision = 0
	recall = 0
	F1_score = 0

print("The precision is {:.3f}, the recall is {:.3f}, and the F1 score is {:.3f}."
	.format(precision, recall, F1_score))

# Plot the test and training data
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(i_train, close_price_train,'b.')
ax1.plot(i_test, close_price_test,'m.')

# Shade the regions where the predictions are made
# This is set up to colour the next price data point from when the prediction is made
# The shaded area starts from the middle of the graph and extends upwards when
# the price actually increased and extends downwards when the true price decreased.
# Red indicates the prediction was incorrect and green for correct.
graph_top = np.max(np.append(close_price_test,close_price_train))+2
graph_bottom = np.min(np.append(close_price_test,close_price_train))-2
graph_middle = 0.5*(graph_top+graph_bottom)

for n in range(0,len(true_pos)):
	ax1.fill_between(i_true_pos[n]+[0.5, 1.5], graph_middle, 100, facecolor='g', linewidth=0)

for n in range(0,len(true_neg)):
	ax1.fill_between(i_true_neg[n]+[0.5, 1.5], 0, graph_middle, facecolor='g', linewidth=0)

for n in range(0,len(false_pos)):
	ax1.fill_between(i_false_pos[n]+[0.5, 1.5], graph_middle, 100, facecolor='r', linewidth=0)

for n in range(0,len(false_neg)):
	ax1.fill_between(i_false_neg[n]+[0.5, 1.5], 0, graph_middle, facecolor='r', linewidth=0)

# Plot the prediction uncertainty on the second axis
ax2.plot(i_true_pos, true_pos_prob,'g+')
ax2.plot(i_true_neg, true_neg_prob, 'g.')
ax2.plot(i_false_pos, false_pos_prob, 'r+')
ax2.plot(i_false_neg, false_neg_prob, 'r.')

# Display the plot
ax1.set_ylim(bottom=graph_bottom , top=graph_top)
ax1.set_xlim(left=0, right= len(boolean_target))
ax2.set_xlim(left=0, right= len(boolean_target))
plt.show()

# # Use the network to predict earning
# (price_input_sim, boolean_target_sim) = sharescraper.get_share_data_boolean_target(
# 	share_name='ANZ.AX',
# 	start_date='2016-01-01',
# 	end_date='2016-03-01',
# 	days_of_data=15,
# 	use_existing_data=False)

# # Build the input array for the first day of prediction
# X_today = price_input[-1,:]
# X_today = np.reshape(X_today,(1,-1))

# #Initialise the simulation variables
# shares_owned = False
# profit = 0
# buy_prob = 0.99
# sell_prob = 0.99

# #Begin the market simulation
# for n in range(0, len(boolean_target_sim)):
# 	X_today = np.append(X_today[:,-28:], price_input_sim[n,-2:])
# 	X_today = np.reshape(X_today,(1,-1))

# 	X_scaled = scaler.transform(X_today)
# 	import pdb; pdb.set_trace()
# 	pred_today = clf.predict(X_scaled)
# 	if pred_today:
# 		pred_today_err = clf.predict_proba(X_scaled)[0,1]
# 	else:
# 		pred_today_err = clf.predict_proba(X_scaled)[0,0]

# 	# Buy 1 share is the price is predicted to increase, we don't already
# 	# own a share, and the prediction probability is greater than the limit
# 	if pred_today == True and shares_owned == False and pred_today_err >= buy_prob :
# 		profit = profit - X_today[:,-1]
# 		shares_owned = True

# 	if pred_today == False and shares_owned == True and pred_today_err >= sell_prob :
# 		profit = profit + X_today[:,-1]
# 		shares_owned = False

# 	print(X_today[:,-1], profit, pred_today, pred_today_err)