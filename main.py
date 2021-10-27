from PIL import Image
from os import listdir
from os.path import isfile, join
from numpy import asarray
import numpy as np
import sys
import logging
import gzip
import network


#-------------------------------------------------INFO--------------------------------------------------
#-------------------------------------------------INFO--------------------------------------------------
#This needs to be run in Python 3.9 because the logging library doesnt have the encoding function in
#previous versions
#-------------------------------------------------INFO--------------------------------------------------
#-------------------------------------------------INFO--------------------------------------------------

#np.set_printoptions(threshold=sys.maxsize)

imageDatasetPath = "./Dataset/"
single_image = "./Dataset/0114_g.jpg"

imageLength = 400
imageHeigth = 280
imagePixelNum  = imageLength * imageHeigth


def main():
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)

    training_data = createTrainingData(imageDatasetPath)
    test_data = createTestData(imageDatasetPath)
    test_image = createTestImage(single_image)
    
    net = network.Network([112000, 20, 3])
    net.SGD(training_data, 52, 10, 2, test_data=test_data, test_image=test_image) #call the training dataset here


def convertImageArray(imagePath): 
    image = Image.open(imagePath)
    # convert image to numpy array
    data = asarray(image)
    data = np.reshape(data, (imagePixelNum, 1))
    data = data / 255
    return data


def getDiseaseTypeArray(imageName):
    condition = imageName.split('.')[0]
    condition = condition.split('_')[1]

    if(condition == "h"):
        return [[1.0],[0], [0]]
    elif(condition == "g"):
        return [[0], [1.0], [0]]
    elif(condition == "dr"):
        return [[0], [0], [1.0]]
    else:
        logging.error("Invalid Input")
        

def createTrainingData(imageDatasetPath):
    files = [f for f in listdir(imageDatasetPath) if isfile(join(imageDatasetPath, f))]
    training_inputs = []
    training_results = []

    for fileName in files:
        imagePath = imageDatasetPath + fileName
        training_inputs.append(convertImageArray(imagePath))
        training_results.append(getDiseaseTypeArray(fileName))

    return zip(training_inputs, training_results)


def createTestData(imageDatasetPath):
    files = [f for f in listdir(imageDatasetPath) if isfile(join(imageDatasetPath, f))]
    test_inputs = []
    test_results = []

    for fileName in files:
        imagePath = imageDatasetPath + fileName
        test_inputs.append(convertImageArray(imagePath))
        test_results.append(getDiseaseType(fileName))

    return zip(test_inputs, test_results)


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


def createTestImage(singleImagePath):

    singleImage = convertImageArray(singleImagePath)

    return singleImage


main()