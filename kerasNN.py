from textwrap import indent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2 as cv
from tensorflow.keras import datasets, layers, models


#np.set_printoptions(threshold=sys.maxsize)

(training_images, training_labels), (test_images, test_labels) = datasets.cifar10.load_data()
training_images, test_images = training_images / 255, test_images / 255

training_images = training_images[:20000]
training_labels = training_labels[:20000]
test_images = test_images[:20000]
test_labels = test_labels[:20000]

#print(training_labels[:30])

#for i in range(16):
#    plt.subplot(4, 4, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.imshow(training_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[training_labels[i][0]])

#plt.show()

def learnModel(training_images, training_labels, test_images, test_labels):
    #define the model, train it and save it
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3))) 
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu')) #Convulotional layers filters images by their features (example: a cat has small legs)
    model.add(layers.MaxPooling2D((2,2))) #Max Pooling layers filter the important information
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) #Last layer
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(training_images, training_labels, epochs=20, validation_data=(test_images, test_labels))
    
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    model.summary()
    
    model.save('image_classifier.model')


def predictionModel():
    model = models.load_model('image_classifier.model') #This loads an already trained model

    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    #Here an image is inputed to get the prediction
    img = cv.imread('plane.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) #Changes an image color scheme from BGR to RGB

    #plt.imshow(img, cmap=plt.cm.binary)
    #plt.show()

    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    print(f"Prediction is {class_names[index]}")


#learnModel(training_images, training_labels, test_images, test_labels)
predictionModel()