##THIS CODE IS FOR EVALUATING SENTENCE WITH NEGATIVE OR POSTIVE,
#SO THAT IT CAN BE USED ON LATER SENTENCES TO TEST IF THEY ARE NEGATIVE OR POSTIVE
##I WOULD PROBS NEED TO TRAIN IT WITH MORE DATE SO THAT THE ACCURACY CAN BE HIGHER FOR CHECKING IF THE SENTENCE IS NEGATIVE OR POSTIVE
import re
import pandas as pd
from textblob import TextBlob

pos_correct = 0
pos_count = 0
neg_correct = 0
neg_count = 0

#getting data
data= pd.read_csv("Data/trainingData.csv")

# this converts into a list
question_list = data.Question.to_list()
answer_list = data.Answer.to_list()

#putting the question and answer into one big list
allSentences = question_list + answer_list #35948


#to make it easier for going into the database then it would be easier for text comparison
#at the end the sentence will come out as a string
allSentences2 = []
for i in allSentences:
  i =re.sub(r'\n', ' ', i)
  i =re.sub('\(', '', i) 
  i =re.sub(r'\)', '', i) 
  i =re.sub(r',', '', i) 
  i =re.sub(r'-', '', i)
  i =re.sub(r'/', '', i)  
  i =re.sub(r'/', '', i)
  i = re.sub(r',', '', i)
  i= re.sub(r"' '", ' ', i) 
  i= re.sub(r"\[", '', i) #removing the brackets around the sentence
  i= re.sub(r"\]", '', i)
  i= re.sub(r"'", '', i)

  allSentences2.append(i)


#to make all the words in answer lowercase
for i in range(len(allSentences2)):
    allSentences2[i] = allSentences2[i].lower()
#print(allSentences2[7])

#goes through the training data and see which is negative and which is positive
for i in allSentences2:
    analysis = TextBlob(i)
    if analysis.sentiment.subjectivity > 0.8:
        if analysis.sentiment.polarity > 0:
            pos_correct += 1
        pos_count += 1

        if analysis.sentiment.polarity <= 0:
            neg_correct += 1
        neg_count += 1

#sentiment respresentation of data
#print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count)) #58.46231920832277%
#print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count)) #41.53768079167724%