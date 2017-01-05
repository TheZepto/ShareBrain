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
print("Scrape succesful", historical_data)

#Process the returned data into 3 lists of: open_price, close_price, and volume
try:
	open_price = [float(historical_data[i]['Open']) for i in range(0,len(historical_data)) ]
	close_price = [float(historical_data[i]['Close']) for i in range(0,len(historical_data)) ]
	volume = [float(historical_data[i]['Volume']) for i in range(0,len(historical_data)) ]
except ValueError:
	print("Error in processing share data.")
	quit()