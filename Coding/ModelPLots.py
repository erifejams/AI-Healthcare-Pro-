
import os

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
import Model as md

plt.plot(md.chatbotModel.history['accuracy'], label='training set accuracy')
plt.plot(md.chatbotModel.history['loss'], label = 'training set loss')
plt.show()
plt.legend()
