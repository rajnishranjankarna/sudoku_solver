# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:03:14 2023

@author: nitis
"""

print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import intializePredectionModel, preProcess, reorder, biggestContour, splitBoxes, getPredection, displayNumbers, drawGrid, stackImages
import sudokoSolver
import cv2
import numpy as np


########################################################################
pathImage = "Resources/sudoku3.jpg"
heightImg = 450
widthImg = 450
model = intializePredectionModel()  # LOAD THE CNN MODEL
########################################################################


#### 1. PREPARE THE IMAGE
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)

# #### 2. FIND ALL COUNTOURS
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (255,0,255), 3) # DRAW ALL DETECTED CONTOURS

#### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
print(biggest)   #printing happening


if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)    #printing happening
    cv2.drawContours(imgBigContour, biggest, -1, (0,0,255), 25)  #DRAW THE BIGGEST CONTOURS in RED color
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GENERATE TRANSFORMSTION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))
   #cv2.imshow("Sample",boxes[65])
    numbers = getPredection(boxes, model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1) # for number greater than 0, put 0, otherwise put 1 in that box
    print(posArray)


    #### 5. FIND SOLUTION OF THE BOARD
    board = np.array_split(numbers,9)
    print(board)
    try:
        sudokoSolver.solve(board)
    except:
        pass
    print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

    # #### 6. OVERLAY SOLUTION
    pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])
    stackedImage = stackImages(imageArray, 0.5)
    cv2.imshow('stacked images', stackedImage)
    
else: 
    print("nothing is found, try again")
    

cv2.imshow("1st", img)
cv2.imshow("2nd", imgThreshold)
cv2.imshow("3rd", imgContours)
cv2.imshow("4th", imgBigContour)
cv2.imshow("5th", imgDetectedDigits)
cv2.imshow("6th", imgSolvedDigits)
cv2.imshow("7th", imgInvWarpColored)
cv2.imshow("8th", inv_perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()





cv2.imshow("Sample",boxes[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("imgWarpColored ",imgWarpColored )
cv2.waitKey(0)
cv2.destroyAllWindows()



