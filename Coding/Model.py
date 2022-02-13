import csv
import json
import numpy as np
#HAD TO INSTALL THE LIBRARY USING "pip install nltk"
#import nlkt

#starting here
#basic chatbot build
class Chatbot():
    def __init__(self, name):
        print(name)
        self.name = name

#printing name of AI
if __name__ == "__main__":
    ai = Chatbot(name = "Interactive Talker")


