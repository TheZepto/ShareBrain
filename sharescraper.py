import sys
import numpy as np
from yahoo_finance import Share #The yahoo-finance package is used to gather the share data

def get_share_data_boolean_target(
	share_name='ANZ.AX',
	start_date='2005-01-01',
	end_date='2016-01-01',
	days_of_data=30,
	use_existing_data=True):
	# Srape historical share data
	# Returns:
	# - training_input array with the share's open price, close price and volume for
	#   the number of days specified in days_of_data between the start_date and end_date.
	# - training_target array consist of a boolean value indicating if the closing price 
	#   tomorrow is greater than the closing price today.

	if use_existing_data:
		try:
			historical_data = np.load("Data/ANZ Data.npy").tolist()
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
	else:
		print("Attempting to scrape data for", share_name)
		try:
			historical_data = Share(share_name).get_historical(start_date, end_date)
			np.save("Data/ANZ Data.npy",historical_data)
		except:
			print("Error in scraping share data. Share name is probably incorrect or Yahoo Finance is down.")
			quit()
		print("Scrape succesful")

	# Reverse the order of the historical data so the list starts at start_date
	historical_data.reverse()
	
	# Process the returned data into 3 lists of: open_price, close_price, and volume
	try:
		open_price = [float(historical_data[i]['Open']) for i in range(0,len(historical_data)) ]
		close_price = [float(historical_data[i]['Close']) for i in range(0,len(historical_data)) ]
		volume = [float(historical_data[i]['Volume']) for i in range(0,len(historical_data)) ]
		high_price = [float(historical_data[i]['High']) for i in range(0,len(historical_data)) ]
		low_price = [float(historical_data[i]['Low']) for i in range(0,len(historical_data)) ]

	except ValueError:
		print("Error in processing share data.")
		quit()

	# Take the historical data and form a training set for the neural net.
	# Each training example is built from a range of days and contains:
	# the open and close share price on each day in the range.
	# The output is boolean indicating if the close price tomorrow is greater than today.

	training_input = np.array([])
	training_target = np.array([])

	training_example_number = len(open_price) - days_of_data

	for i in range(0, training_example_number):
		training_input = np.append(training_input, volume[i:i+days_of_data])
		training_input = np.append(training_input, high_price[i:i+days_of_data])
		training_input = np.append(training_input, low_price[i:i+days_of_data])
		training_input = np.append(training_input, open_price[i:i+days_of_data])
		training_input = np.append(training_input, close_price[i:i+days_of_data])
		training_target = np.append(training_target, close_price[i+days_of_data] > close_price[i+days_of_data-1] )

	# The above for loop makes 1-dim arrays with the values in them. Need to use reshape
	# on training input to make it 2-dim. Number of columns is 3*days_of_data to account for 
	# open price, close price and volume. The -1 in reshape will be filled 
	# automatically and (days_of_data +1) is the number of columns for each input. Likewise
	# for the target array.

	training_input = np.reshape(training_input, (-1, 5*days_of_data))
	training_target = np.reshape(training_target, (-1,))
	
	return (training_input, training_target)