#This program was designed by Dane Laban to analyse share data and predict
#whether the share price would continue to increase in the foreseeable future.
#A neural network performs the analysis and neurolab is being used to provide
#this functionality.
print("#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@")
print("@         Welcome to ShareBrain          #")
print("# Designed and implemented by Dane Laban @")
print("@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#")

#common libraries
import sys
import numpy as np

#my libraries
import shares
import nlnet


(training_input, training_target) = shares.get_share_data()

net = nlnet.multilayer(training_input, training_target)