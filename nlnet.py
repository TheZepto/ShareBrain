import sys
import numpy as np
import neurolab as nl


def multilayer(training_input, training_target):
	#Need to work out the min and max of each training datatype and use them to 
	#define each input to the neural net.
	minmax = nl.tool.minmax(training_input)

	#Setting up the neural network as a multi-layer perceptron with (days_of_data +1) inputs,
	#1 hidden layer with 10 neurons and 1 output layer
	net = nl.net.newff(minmax, [30,20,1], transf=[nl.trans.LogSig(),nl.trans.LogSig(),nl.trans.LogSig()])

	#Train the neural network using gradient descent with backprop trainer with set epochs and output display
	net.errorf = nl.error.CEE()
	training_error = net.train(training_input, training_target,epochs=500,show=10,goal=0.2)

	#Evaluate the networks performance on the training data
	simulated_target = np.round(net.sim(training_input))
	correct_predictions = np.sum( np.equal(simulated_target, training_target) ) / len(simulated_target) * 100
	print("The network is {0:.2f}% correct from the training data.".format(correct_predictions))

	#Ask if the network should be saved or discarded
	if input("Would you like to save the network (y/n):") == 'y':
		net.save(input("Enter the filename to save to (.net):"))
		print("File saved successfully")

	return net