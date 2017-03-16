import sys
import numpy as np
from yahoo_finance import Share

def get_share_data(share_name='ANZ.AX', start_date='2005-01-01', end_date='2016-01-01', days_of_data=30):
	#Srape historical share data
	#The yahoo-finance package is used to gather the share data

	try:
		historical_data = np.load("Data/ANZ Data.npy")
		print("Data successfully loaded from locally stored file")
	except:
		#Scrape the data for the given settings and exit if there is an error
		print("Attempting to scrape data for", share_name)
		try:
			historical_data = Share(share_name).get_historical(start_date, end_date)
			np.save("Data/ANZ Data.npy",historical_data)
		except:
			print("Error in scraping share data. Share name is probably incorrect or Yahoo Finance is down.")
			quit()
		print("Scrape succesful")


	#Process the returned data into 3 lists of: open_price, close_price, and volume
	try:
		open_price = [float(historical_data[i]['Open']) for i in range(0,len(historical_data)) ]
		close_price = [float(historical_data[i]['Close']) for i in range(0,len(historical_data)) ]
		volume = [float(historical_data[i]['Volume']) for i in range(0,len(historical_data)) ]
	except ValueError:
		print("Error in processing share data.")
		quit()

	# Take the historical data and form a training set for the neural net.
	# Each training example is built from a range of days and contains:
	# an input for the open share price on each day in the range and the volume on the last day
	# The output is binary:whether the opening share price increased on the day after the last day.

	training_input = np.array([])
	training_target = np.array([])

	for i in range(0, len(open_price)-days_of_data):
		training_input = np.append(training_input, open_price[i:i+days_of_data])
		training_target = np.append(training_target, open_price[i+days_of_data-1] > open_price[i+days_of_data] )


	#The above for loop makes 1-dim arrays with the values in them. Need to use reshape
	#on the input and training data to make it 2-dim. The -1 in reshape will be filled 
	#automatically and (days_of_data +1) is the number of columns for each input. Likewise
	#for the target array.

	training_input = np.reshape(training_input, (-1, days_of_data))
	training_target = np.reshape(training_target, (-1,1))
	
	return (training_input, training_target)