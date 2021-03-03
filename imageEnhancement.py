from PIL import Image
from os import listdir
from os.path import isfile, join
import logging
from numpy import asarray
import numpy as np
import sys

imageDatasetOriginalPath1 = "./DatasetOriginal/"
imageDatasetOriginalPath2 = "./TestDatasetOriginal/Healthy/"
imageDatasetOriginalPath3 = "./TestDatasetOriginal/DR2/"
imageDatasetOriginalPath = imageDatasetOriginalPath1
imageDatasetPath = "./Dataset/"
UPLOAD_FOLDER = "./uploads/"

imageLength = 400
j = 0

#np.set_printoptions(threshold=sys.maxsize)

def main():
    #logging.basicConfig(filename='imageEnhancement.log', encoding='utf-8', level=logging.INFO)

    convertAllImage()

def convertSingleImage(fileName):
    imagePath = UPLOAD_FOLDER + fileName
    convertImageToGray(imagePath)
    removeImageBorder(image)
    imageResizeAndCrop(image1)
    newImagePath = UPLOAD_FOLDER + fileName
    image2.save(newImagePath)

def convertAllImage():
    files = [f for f in listdir(imageDatasetOriginalPath) if isfile(join(imageDatasetOriginalPath, f))]
    for fileName in files:
        imagePath = imageDatasetOriginalPath + fileName
        convertImageToGray(imagePath)
        removeImageBorder(image)
        imageResizeAndCrop(image1)
        transposeAndSaveImage(image2, fileName)


def convertImageToGray(imagePath):   #This converts an image to gray
    global image

    image = Image.open(imagePath)
    image = image.convert('L')

    return image


def removeImageBorder(image):   #This removes the excess black borders
    loopBreak = False
    previousBlackLeftPosition = 0
    global image1

    data = asarray(image)
    data = data / 255
    matrixSize = data.shape
    startLookingForEdge = (matrixSize[0] / 2) - 50

    matrixSize = data.shape
    blackLeft = len(data) / 2
    blackRight = 0
    iteration = 0

    for i in data:

        if(iteration < startLookingForEdge):
            iteration = iteration + 1
            continue

        for j in range(len(i)):

            if( i[j] >= 0 and i[j] <= 0.2 ):
                previousBlackLeftPosition = j
            elif( i[j] > 0.2 ):

                if( previousBlackLeftPosition > blackLeft):
                    loopBreak = True
                else:
                    x = matrixSize[1]
                    x = x - 1

                    while (i[x] < 0.2):

                        blackRight = x

                        x = x - 1
                
                break

        if loopBreak: break
        blackLeft = previousBlackLeftPosition

    #crop image
    box = (blackLeft, 0, blackRight, matrixSize[0])
    image = image.crop(box)
    image1 = image

    return image1


def imageResizeAndCrop(image):  #This converts the image a smaller scale
    global image2

    imagePercentDivider = float(image.size[0])
    if(imagePercentDivider < 0.51):
        imagePercentDivider = 1
    wpercent = (imageLength/imagePercentDivider)
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((imageLength,hsize), Image.ANTIALIAS)

    data = asarray(image)
    matrixSize = data.shape

    remainder = matrixSize[0] - 280
    remainder = remainder / 2
    bottomRemainder = matrixSize[0] - remainder

    box = (0, remainder, 400, bottomRemainder)
    image = image.crop(box)

    image2 = image
    return image2


def transposeAndSaveImage(image, imageName): #This trasposes each image in four different ways
    global j
    i = 0

    newImagePath = imageDatasetPath + str(i) + str(j) + imageName
    image.save(newImagePath)
    #logging.info("Created new image: " + newImagePath)

    i = i + 1

    newImagePath = imageDatasetPath + str(i) + str(j) + imageName
    image3 = image.transpose(Image.FLIP_TOP_BOTTOM)
    image3.save(newImagePath)
    #logging.info("Created new image: " + newImagePath)

    i = i + 1

    newImagePath = imageDatasetPath + str(i) + str(j) + imageName
    image4 = image.transpose(Image.FLIP_LEFT_RIGHT)
    image4.save(newImagePath)
    #logging.info("Created new image: " + newImagePath)

    i = i + 1

    newImagePath = imageDatasetPath + str(i) + str(j) + imageName
    image5 = image.transpose(Image.FLIP_LEFT_RIGHT)
    image5 = image5.transpose(Image.FLIP_TOP_BOTTOM)
    image5.save(newImagePath)
    #logging.info("Created new image: " + newImagePath)

    j = j +1


#main()