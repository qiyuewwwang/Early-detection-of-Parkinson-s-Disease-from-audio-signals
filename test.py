from pack2 import *

path = "trainedModel4.sav" #Model path
clf4 = loadModel(path) #Model loading

print(predict(clf4, "PD_test.wav"))#Predicition