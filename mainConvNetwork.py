from PIL import Image
from os import listdir
from os.path import isfile, join
from matplotlib import image
from matplotlib.pyplot import show
from numpy import asarray, single
import numpy as np
import sys
import logging
import gzip
import convNetwork


#-------------------------------------------------INFO--------------------------------------------------
#-------------------------------------------------INFO--------------------------------------------------
#This is the main file for the Convolutional Network
#-------------------------------------------------INFO--------------------------------------------------
#-------------------------------------------------INFO--------------------------------------------------

#np.set_printoptions(threshold=sys.maxsize)

imageDatasetPath = "./Dataset/"
singleImagePath = "./Dataset/22135_h.jpg"

imageLength = 400
imageHeigth = 280
imagePixelNum  = imageLength * imageHeigth


def main():

    training_images = createTrainingImages(imageDatasetPath)
    training_labels = createTrainingLabels(imageDatasetPath)
    test_images= createTrainingImages(imageDatasetPath)
    test_labels = createTrainingLabels(imageDatasetPath)

    convNetwork.main(training_images, training_labels, test_images, test_labels)
    #convNetwork.predictImage(singleImagePath)


def createTrainingImages(imageDatasetPath):
    files = [f for f in listdir(imageDatasetPath) if isfile(join(imageDatasetPath, f))]
    training_images = []

    for fileName in files:
        imagePath = imageDatasetPath + fileName
        training_images.append(convertImageArray(imagePath))
    
    training_images = np.array(training_images).reshape(-1, imageHeigth, imageLength, 1)

    return training_images


def createTrainingLabels(imageDatasetPath):
    files = [f for f in listdir(imageDatasetPath) if isfile(join(imageDatasetPath, f))]
    training_labels = []

    for fileName in files:
        imagePath = imageDatasetPath + fileName
        training_labels.append(getDiseaseType(fileName))

    training_labels = np.reshape(training_labels, (len(training_labels), 1))

    return training_labels


def convertImageArray(imagePath): 
    image = Image.open(imagePath)
    # convert image to numpy array
    data = asarray(image)
    #data = np.reshape(data, (imagePixelNum, 1))
    data = data / 255
    return data


def getDiseaseType(imageName):
    condition = imageName.split('.')[0]
    condition = condition.split('_')[1]

    if(condition == "h"):
        return 0
    elif(condition == "g"):
        return 1
    elif(condition == "dr"):
        return 2
    else:
        logging.error("Invalid Input")

main()