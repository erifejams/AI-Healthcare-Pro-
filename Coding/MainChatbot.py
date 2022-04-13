#THIS IS THE MAIN FILE WHERE IT RUNS
import os

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

#from nltk
from DecodeSentences import talk_with_bot, databaseUser, cursor
        
#main
if __name__ == "__main__":
    #gets the chatbot to talk
    talk_with_bot()
    #adds the user inputs to the database, if not here, then user input would not be commited/added to database
    databaseUser.commit()
    #closes the cursor for the database
    cursor.close()
        
