#from textwrap import indent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2 as cv
from tensorflow.keras import datasets, layers, models

#-------------------------------------------------INFO--------------------------------------------------
#-------------------------------------------------INFO--------------------------------------------------
#This needs to be run in Python 3.8 through the python conda environment because of tensorflow and other 
# libraries
#-------------------------------------------------INFO--------------------------------------------------
#-------------------------------------------------INFO--------------------------------------------------


#np.set_printoptions(threshold=sys.maxsize)

#(training_images, training_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#print(training_labels[:30])

#for i in range(16):
#    plt.subplot(4, 4, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.imshow(training_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[training_labels[i][0]])

#plt.show()

print(tf.__version__)

def main(training_images, training_labels, test_images, test_labels):

    learnModel(training_images, training_labels, test_images, test_labels)

def predictImage(single_image_path):
    return predictionModel(single_image_path)


def learnModel(training_images, training_labels, test_images, test_labels):
    #define the model, train it and save it
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(280, 400, 1))) 
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu')) #Convulotional layers filters images by their features (example: a cat has small legs)
    model.add(layers.MaxPooling2D((2,2))) #Max Pooling layers filter the important information
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax')) #Last layer
    
    print("1")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("2")
    model.fit(training_images, training_labels, epochs=2, validation_data=(test_images, test_labels))
    
    print("3")
    loss, accuracy = model.evaluate(test_images, test_labels)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(accuracy))
    model.summary()
    
    #model.save('retina_disease_classifier.model')


def predictionModel(single_image):
    model = models.load_model("./retina_disease_classifier.model") #This loads an already trained model

    class_names = ['Healthy', 'Glaucomatous', 'Diabetic Retinopathy']

    #Here an image is inputed to get the prediction
    img = cv.imread(single_image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Changes an image color scheme from BGR to RGB

    #plt.imshow(img, cmap=plt.cm.binary)
    #plt.show()

    img = np.array([img]).reshape(-1, 280, 400, 1) / 255
    prediction = model.predict(img)
    index = np.argmax(prediction)
    print("Prediction is {}".format(class_names[index]))
    return class_names[index]