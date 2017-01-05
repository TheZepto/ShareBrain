import sys
import numpy as np
import neurolab as nl
from yahoo_finance import Share

#This program was designed by Dane Laban to analyse share data and predict
#whether the share price would continue to increase in the foreseeable future.
#A neural network performs the analysis and neurolab is being used to provide
#this functionality.
print("Welcome to ShareBrain")
print("Designed and implemented by Dane Laban")


#Srape historical share data
#The yahoo-finance package is used to gather the share data
#
#Configurable settings
share_name = 'ANZ.AX'
start_date = '2016-01-01'
end_date = '2016-02-01'

#Scrape the data for the given settings and exit if there is an error
print("Attempting to scrape data for", share_name)
try:
	historical_data = Share(share_name).get_historical(start_date, end_date)
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

#Take the historical data and form a training set for the neural net.
#Each training example has 2 linear inputs: the opening share price of the day
#and the previous day; and 1 binary output: whether the opening share price
#increased on the day after.
training_input = np.array([])
training_target = np.array([])

for i in range(1, len(open_price)-1):
	training_input = np.append(training_input, [open_price[i-1:i+1]])
	training_target = np.append(training_target, [open_price[i+1] > open_price[i]])

#The above for loop makes 1-dim arrays with the values in them. Need to use reshape
#on the input data to make it 2-dim. The -1 in reshape will be filled automatically
#and the 2 is there as there should be 2 columns for each input.
training_input = np.reshape(training_input, (-1,2))

