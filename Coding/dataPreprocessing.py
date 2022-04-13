#DATA PREPROCESSING STEPS WHICH ARE: 
    #Removing punctuations like . , ! $( ) * % @ AND Tokenization
    #Removing Stopwords - MAY DO THIS STEP LATER, BUT I THOUGHT I SHOULDN'T REMOVE OR IT WILL CHANGE THE MEANING OF THE SENTENCE
    #Stemming - LESS ACCURATE RESULTS, SO FOR NORMALIZATION OF THE TEXT WENT WITH LEMMITZATION
    #Lemmatization

#uncomment to download needed packages
#nltk.download("punkt")
#nltk.download('stopwords') 
#nltk.download('omw-1.4')

import re
import ReadingDataFiles as dataFiles
import nltk
from nltk.stem import WordNetLemmatizer 


#for example don't was translated to donât for some reasons, so have to change it to a common 
#removing elements that don't shouldn't be there
def cleaning(text):
    textCleaning =re.sub(r'\n', ' ', text)
    textCleaning =re.sub('\(', '', textCleaning) 
    textCleaning =re.sub(r'\)', '', textCleaning) 
    textCleaning =re.sub(r',', '', textCleaning) 
    textCleaning =re.sub(r'-', '', textCleaning)
    textCleaning =re.sub(r'/', '', textCleaning)  
    textCleaning =re.sub(r'/', '', textCleaning)
    textCleaning =re.sub(r'Â', '', textCleaning)
    textCleaning =re.sub(r'â', "'", textCleaning)
    textCleaning =re.sub(r'¢', '', textCleaning)
    return textCleaning


#replaces the shorten version to full format
def ExtendText(text):
    # specific
    phrase = re.sub(r"won't", 'will not', text)
    phrase = re.sub(r"can't", 'can not', phrase)
    phrase = re.sub(r"ain't", 'are not', phrase)
    phrase = re.sub(r"aren't", 'am not', phrase)
    phrase = re.sub(r"cause", 'because', phrase)
    phrase = re.sub(r"could've", 'could have', phrase)
    phrase = re.sub(r"couldn't", 'could not', phrase)
    phrase = re.sub(r"didn't", 'did not', phrase)
    phrase = re.sub(r"doesn't", 'does not', phrase)
    phrase = re.sub(r"don't", 'do not', phrase)
    phrase = re.sub(r"hadn't", 'had not', phrase)
    phrase = re.sub(r"haven't", 'have not', phrase)
    phrase = re.sub(r"he'd", 'he had', phrase)
    phrase = re.sub(r"he'll", 'he will', phrase)
    phrase = re.sub(r"he's", 'he is', phrase)
    phrase = re.sub(r"i'd", 'I had', phrase)
    phrase = re.sub(r"i'll", 'I will', phrase)
    phrase = re.sub(r"i've", 'I have', phrase)
    phrase = re.sub(r"isnvt", 'is not', phrase)
    phrase = re.sub(r"it'll", 'it will', phrase)
    phrase = re.sub(r"it's", 'it is', phrase)
    phrase = re.sub(r"she'll", 'she will', phrase)
    phrase = re.sub(r"she's", 'she is', phrase)
    phrase = re.sub(r"should've", 'should have', phrase)
    phrase = re.sub(r"shouldn't", 'should not', phrase)
    phrase = re.sub(r"that's", 'that is', phrase)
    phrase = re.sub(r"there's", 'there is', phrase)
    phrase = re.sub(r"they'll", 'they are', phrase)
    phrase = re.sub(r"they're", 'it will', phrase)
    phrase = re.sub(r"we'll", 'we will', phrase)
    phrase = re.sub(r"we're", 'we are', phrase)
    phrase = re.sub(r"wevve", 'we have', phrase)
    phrase = re.sub(r"what's", 'what is', phrase)
    phrase = re.sub(r"where's", 'where is', phrase)
    phrase = re.sub(r"you're", 'you are', phrase)
    phrase = re.sub(r"why's", 'why is', phrase)
    phrase = re.sub(r"y'all", 'you all', phrase)
    phrase = re.sub(r"wouldn't", 'we will', phrase)
    phrase = re.sub(r"I'm", 'I am', phrase)
    phrase = re.sub(r"There's", 'There is', phrase)
    return phrase

#PUNCTUATION AND TOKENIZATION STEP
#DEFINING THE FUNCTION TO REMOVE PUNCTUATION AND TOKENIZE 
def remove_punctuation_tokenize(text):
    #removes the punctuations
    punctuationfree = nltk.RegexpTokenizer(r"\w+")
    #tokenizes sentence - splits it into smaller parts
    individual_words = punctuationfree.tokenize(text)
    return individual_words

#NOT REMOVING STOPWORDS AS IT COULD CHANGE THE MEANING WHEN DOING SENTIMENT ANALYSIS

#LEMMATIZATION STEP
#lemmatization provides better results, but there might be trouble with speed, if there is go with stemming
wordnet_lemmatizer = WordNetLemmatizer()
#defining the function for lemmatization
def lemmatizer(text):
    lemmatized_text = wordnet_lemmatizer.lemmatize(text) 
    return lemmatized_text


#this is to clean the words
dataFiles.concate_data['CleanQuestion']= dataFiles.concate_data['Question'].apply(lambda x:cleaning(str(x))) 
dataFiles.concate_data['CleanAnswer']= dataFiles.concate_data['Answer'].apply(lambda x:cleaning(str(x)))

#to is to get the full extended version of the word
dataFiles.concate_data['ExtendedFormQuestion']= dataFiles.concate_data['CleanQuestion'].apply(lambda x:ExtendText(str(x))) 
dataFiles.concate_data['ExtendedFormAnswer']= dataFiles.concate_data['CleanAnswer'].apply(lambda x:ExtendText(str(x)))

#storing the puntuation and tokenized text 
dataFiles.concate_data['PuncTokenQuestion']= dataFiles.concate_data['ExtendedFormQuestion'].apply(lambda x:remove_punctuation_tokenize(str(x))) #i put str(x), cause it expects it back as a string unless it gives an error
dataFiles.concate_data['PuncTokenAnswer']= dataFiles.concate_data['ExtendedFormAnswer'].apply(lambda x:remove_punctuation_tokenize(str(x)))


#storing the Lemmatization
dataFiles.concate_data['QuestionLemmatized'] = dataFiles.concate_data['PuncTokenQuestion'].apply(lambda x:lemmatizer(str(x)))
dataFiles.concate_data['AnswerLemmatized'] = dataFiles.concate_data['PuncTokenAnswer'].apply(lambda x:lemmatizer(str(x)))


#REMOVING COLUMNS NOT NEEDED AGAIN
#inplace=True means data is modified and updated
dataFiles.concate_data.drop(['Question', 'Answer' ,'CleanQuestion', 'CleanAnswer', 'ExtendedFormQuestion', 'ExtendedFormAnswer', 'PuncTokenQuestion', 'PuncTokenAnswer'], axis = 1, inplace = True)

#CHANGE THE NAME OF THE COLUMNS 
dataFiles.concate_data.rename(columns={"QuestionLemmatized": "Question"}, inplace=True)
dataFiles.concate_data.rename(columns={"AnswerLemmatized": "Answer"}, inplace=True)


#change the orders of the columns
column_names = ["Question", "Answer"]
#makes the index start from the start again
dataFiles.concate_data = dataFiles.concate_data.reindex(columns=column_names)

#print(dataFiles.concate_data)
#datatype int64
#print(dataFiles.concate_data['Question'].count()) #17973
#print(dataFiles.concate_data['Answer'].count()) #17973

#CREATE THE TRAINING CSV DATA FILE
dataFiles.concate_data.to_csv('Data/trainingData.csv')
