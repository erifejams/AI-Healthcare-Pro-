#CREATE TRAINING AND TEST LIST

import random
import numpy as np
import pandas as pd

#THIS IS TO READ THE TRAINING DATA
training = pd.read_csv (r'Data/trainingData.csv')

#SHUFFLE DATA - THIS IS TO MAKE SURE WE HAVE RANDOM SAMPLE 
random.shuffle(list(training))
training = np.array(training)

#create train and test lists. X - Question, Y - Anwser
#the O was the index id
train_x = list(training[:, 1])
train_y = list(training[:, 2])
#print(training)