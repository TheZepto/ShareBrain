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
from sklearn.neural_network import MLPRegressor

# My libraries
import sharescraper

# Boolean prediction of whether the close_price tomorrow will be greater than today.

# Get training input and targets from sharescraper
historical_data = sharescraper.get_share_data(
	share_name='ANZ.AX',
	start_date='2000-01-01',
	end_date='2016-01-01',
	use_existing_data=True)

(price_input, price_target) = sharescraper.proc_share_real_target(
	historical_data,
	days_of_data=50)

# Separate data into training set and test set
random_number = 0
test_split = 0.3
X_train, X_test, y_train, y_test = train_test_split(
	price_input, price_target, test_size=test_split, random_state=random_number)

# Feature scale the training data and apply the scaling to the training and test datasets.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Set up the MLPClassifier
clf = MLPRegressor(
	activation = 'identity',
	solver ='adam',
	hidden_layer_sizes=(3000,1000,500,100),
	alpha = 1,
	max_iter = 10000,
	tol = 1E-5,
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
	print("The network fitted the test data {:.3f}% and the training data {:.3f}% (R2 score)."
		.format(test_accuracy*100, train_accuracy*100))
	accuracy_check = test_accuracy < 0.1

# Build an array of indices to order the train and test data back in the plot
indices = np.arange(len(price_target))
i_train, i_test, iy_train, iy_test = train_test_split(
	indices, indices, test_size=test_split, random_state=random_number)

# Extract closing price data for test and training data
# rescaled back to original price
close_price_test = scaler.inverse_transform(X_test)[:,-1]
close_price_train = scaler.inverse_transform(X_train)[:,-1]

# Generate the predictions for the test data
predictions = clf.predict(X_test)

# Plot the test and training data
fig, ax1, = plt.subplots(1,1)
ax1.plot(i_train, close_price_train,'b.')
ax1.plot(i_test, close_price_test,'m.')
ax1.plot(i_test+1, predictions,'k+')

# Draw lines between predicted price and actual close price
# Red indicates that the network failed to predict the correct trend
# from one day to the next and green is vice-versa.
# Also use this opportunity to count correct trend predictions.

correct_trend_prediction = 0

for n in range(0,len(predictions)):
	if predictions[n] >= close_price_test[n] and y_test[n] >= close_price_test[n]:
		ax1.plot([i_test[n]+1, i_test[n]+1], [y_test[n], predictions[n]], 'g-') 
		correct_trend_prediction = correct_trend_prediction + 1
	if predictions[n] < close_price_test[n] and y_test[n] < close_price_test[n]:
		ax1.plot([i_test[n]+1, i_test[n]+1], [y_test[n], predictions[n]], 'g-')  
		correct_trend_prediction = correct_trend_prediction + 1
	if predictions[n] >= close_price_test[n] and y_test[n] < close_price_test[n]:
		ax1.plot([i_test[n]+1, i_test[n]+1], [y_test[n], predictions[n]], 'r-') 
	if predictions[n] < close_price_test[n] and y_test[n] >= close_price_test[n]:
		ax1.plot([i_test[n]+1, i_test[n]+1], [y_test[n], predictions[n]], 'r-') 

print("The network correctly predicted the price trend {:.3f}%."
	.format(correct_trend_prediction/len(predictions)*100) )

# Display the plot
ax1.set_xlim(left=0, right= len(price_target))
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