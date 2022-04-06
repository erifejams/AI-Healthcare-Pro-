#THIS IS THE WHERE USER RESPONSE IS COLLECTED 
import os
import random

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

#from nltk
from keras.models import load_model

#for initializing the greeting and bye
Introduce_Ans = ["My name is InTa.","you can called me InTa.","Im InTa :) ","My name is InTa. and my nickname is InTa and i am happy excited to have someone to talk to :) "]
GREETING_INPUTS = ("hello", "hi","hiii","hii","hiiii","hiiii", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hii there", "hi there", "hello", "I am glad! You are talking to me"]


#starting here
#basic chatbot build
class Chatbot():
    def __init__(self, name):
        print("Hi, my name is ", name)
        self.name = name
        self.model = load_model('chatbot_model.h5')
        
# to do introductions to the user (basic greetings)
"""If user's input is a greeting, return a greeting response"""
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# to introduce self
def IntroduceInTa(sentence):
    return random.choice(Introduce_Ans)

#get user response
def getUserResponse():
    #need to tokanize user response
    return

#printing name of AI
if __name__ == "__main__":
    ai = Chatbot(name = "InTa")
        
