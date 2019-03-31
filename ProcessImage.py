
import cv2
import numpy as np
import math
import PossibleChar
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
###########################
#ann = cv2.ml.ANN_MLP_create()
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(300, 2), random_state=1)

MIN_PIXEL_WIDTH =10
MIN_PIXEL_HEIGHT =20

MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 0.8

MIN_PIXEL_AREA = 150

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.01
MAX_DIAG_SIZE_MULTIPLE_AWAY = 10.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30


MIN_CONTOUR_AREA = 70

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
kNearest = cv2.ml.KNearest_create()



###################################################################################################
class ProcessImage:
 
 
 imgPlate = None
 imgGrayscale = None
 imgThresh = None

 LocationOfPlateInScene = None

 strChars = ""
        
 def loadKNNDataAndTrainKNN(self):
                  
     Classifications = np.loadtxt("classifications_final.txt", np.float32)
     Classifications = Classifications.reshape((Classifications.size,1))
     X = np.loadtxt("flattened_final.txt", np.float32)
     #ann.setLayerSizes(np.float32([600, 300, 1]))
     #ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
     #ann.train(X,cv2.ml.ROW_SAMPLE,npaClassifications)

     #return True    

     #clf.fit(X, npaClassifications)                         
     #MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
      
     #beta_1=0.9, beta_2=0.999, early_stopping=False,
     #  epsilon=1e-08, hidden_layer_sizes=(100, 2), learning_rate='constant',
      # learning_rate_init=0.001, max_iter=2000, momentum=0.9,
       #nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       #solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       #warm_start=False)# -*- coding: utf-8 -*-

     kNearest.setDefaultK(1)                                                            
     kNearest.train(X, cv2.ml.ROW_SAMPLE, Classifications)    
    
     return True

    
    
   
###################################################################################################
 
 def checkIfPossibleChar(self,possibleChar):
     if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
     else:
        return False


###################################################################################################
 def distanceBetweenChars(self,firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

###################################################################################################
 def angleBetweenChars(self,firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           
        fltAngleInRad = math.atan(fltOpp / fltAdj)      
    else:
        fltAngleInRad = 1.5708                          

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)     

    return fltAngleInDeg

###################################################################################################
 

 def initPlate(self,obj):
    plate=obj.plate
    listOfMatchingChars=[]
    
    height, width, numChannels = plate.shape
    #possiblePlate = PossiblePlate.PossiblePlate()  
    
    plateGrayscale, plateThresh= self.preprocess(plate) 
    
    imgContours,contours, hierarchy = cv2.findContours(plateThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #imgContours = np.zeros((height, width, 3), np.uint8)
    #cv2.drawContours(imgContours, contours, -1, SCALAR_WHITE)
    #cv2.imshow("imgcontours",imgContours)
    for contour in contours:                        
        possibleChar = PossibleChar.PossibleChar(contour)

        if self.checkIfPossibleChar(possibleChar):              
            listOfMatchingChars.append(possibleChar)
            
    
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        

    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = self.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    self.LocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )


    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = plate.shape      

    imgRotated = cv2.warpAffine(plate, rotationMatrix, (width, height))       

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    self.imgPlate = imgCropped        
    cv2.imshow("plate",self.imgPlate)
    cv2.imwrite("imgPlate.png",self.imgPlate)
    self.recognizeChars(self.imgPlate)
    
    #return possiblePlate
######################################################
 def recognizeChars(self,plate):
       plateGrayscale, plateThresh= self.preprocess(plate)
       strChars = ""               
       #imgContours, contours, hierarchy = cv2.findContours(plateThreshScene, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
       cv2.imshow("Grayscale",plateGrayscale)     
       cv2.waitKey(0)
       cv2.imshow("Threshold of plate",plateThresh)
       cv2.waitKey(0)
       imgContours, contours, hierarchy = cv2.findContours(plateThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       height,width,numChannels=plate.shape
       imgContours = np.zeros((height, width, 3), np.uint8)
       cv2.drawContours(imgContours, contours, -1, SCALAR_WHITE)
       cv2.imshow("contours",imgContours)
       cv2.waitKey(0)
       listOfMatchingChars=[]
       for contour in contours:                        
        possibleChar = PossibleChar.PossibleChar(contour)

        if self.checkIfPossibleChar(possibleChar):              
            listOfMatchingChars.append(possibleChar)
       
       height, width = plateThresh.shape

       imgThreshColor = np.zeros((height, width, 3), np.uint8)

       listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        

       cv2.cvtColor(plateThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)
    
#############
       for currentChar in listOfMatchingChars:                                         
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2,SCALAR_GREEN, 2)          

        cv2.imshow("individual characters",imgThreshColor)
        cv2.waitKey(0)
        imgROI = plateThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           
        
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        

        npaROIResized = np.float32(npaROIResized)    
        
        
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              
        strCurrentChar = str(chr(int(npaResults[0])))            

        strChars = strChars + strCurrentChar 
       print "Number Plate: " + strChars
       return


##############################################################

 def preprocess(self,imgOriginal):
    
     imgGrayscale = self.extractValue(imgOriginal)
     imgMaxContrastGrayscale = self.maximizeContrast(imgGrayscale)

     height, width = imgGrayscale.shape

     imgBlurred = np.zeros((height, width, 1), np.uint8)

     imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

     imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
     #flag=flag+1
     kernel = np.ones((2,2),np.uint8)
     erosion = cv2.erode(imgThresh,kernel,iterations = 1)
     #cv2.imshow("erosion", erosion)
     return imgGrayscale, erosion
     
 def extractValue(self,imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
###################################################################################################

 def maximizeContrast(self,imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat
    
##############################################################










############################################################
'''  
 def detectPlatesInScene(self,imgOriginalScene):
    listOfPossiblePlates = []                   
    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)


    cv2.imshow("ROI", imgOriginalScene)
    cv2.waitKey(0)
    preprocessObj=Preprocess.Preprocess()
    imgGrayscaleScene, imgThreshScene = preprocessObj.preprocess(imgOriginalScene)         
    
     
    cv2.imshow("ROI-grayscale", imgGrayscaleScene)
    cv2.waitKey(0)
    cv2.imshow("ROI-threshold", imgThreshScene)
    cv2.waitKey(0)
    listOfPossibleCharsInScene = self.findPossibleCharsInScene(imgThreshScene,imgOriginalScene)
    print "findPossibleCharsInScene Returned"
    #if Main.showSteps == True: 
        #print "step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene))         

    imgContours = np.zeros((height, width, 3), np.uint8)

    contours = []

    for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        
    cv2.drawContours(imgContours, contours, -1, SCALAR_WHITE)
    cv2.imshow("ROI-contour", imgContours)
    cv2.waitKey(0)
    listOfListsOfMatchingCharsInScene = self.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   
        possiblePlate = self.extractPlate(imgOriginalScene, listOfMatchingChars)         

        if possiblePlate.imgPlate is not None:                          
            listOfPossiblePlates.append(possiblePlate)                  
    return listOfPossiblePlates
    
###################################################################################################
  
 def findPossibleCharsInScene(self,imgThresh,imgOriginalScene):
    listOfPossibleChars = []                

    intCountOfPossibleChars = 0
    
    imgThreshCopy = imgThresh.copy()
    
    imgContours, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
    
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
    

    for i in range(0, len(contours)):                       

        #if Main.showSteps == True:
        cv2.drawContours(imgContours, contours, i, SCALAR_WHITE)
        possibleChar = PossibleChar.PossibleChar(contours[i])
        if self.checkIfPossibleChar(possibleChar):                   
            intCountOfPossibleChars = intCountOfPossibleChars + 1           
            listOfPossibleChars.append(possibleChar)                        
    return listOfPossibleChars

 

###################################################################################################

 
 def extractPlate(self,imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()          

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        

    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = self.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )


    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped         
    
    return possiblePlate
    
 ######################################################

 
 def detectCharsInPlates(self,Plates):
    imgContours = None
    contours = []
    PreprocessObj=Preprocess.Preprocess()
    
    #if len(Plates) == 0:          
     #   return Plates             


    #for possiblePlate in listOfPossiblePlates:       
        
    Plates.imgGrayscale, Plates.imgThresh = PreprocessObj.preprocess(Plates.imgPlate)
    Plates.imgPlate=cv2.resize(Plates.imgPlate, (0, 0), fx = 1.6, fy = 1.6)
    #Plates.imgGrayscale=cv2.resize(Plates.imgGrayScale, (0, 0), fx = 1.6, fy = 1.6)
    Plates.imgThresh = cv2.resize(Plates.imgThresh, (0, 0), fx = 1.6, fy = 1.6)
    cv2.imshow("plateGrayscale",Plates.imgGrayscale)
    cv2.imshow("plateThresh",Plates.imgThresh)
    thresholdValue, Plates.imgThresh = cv2.threshold(Plates.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    listOfPossibleCharsInPlate = self.findPossibleCharsInPlate(Plates.imgGrayscale, Plates.imgThresh)
    height, width, numChannels = Plates.imgPlate.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
    del contours[:]                                         

    for possibleChar in listOfPossibleCharsInPlate:
        contours.append(possibleChar.contour)

    cv2.drawContours(imgContours, contours, -1, SCALAR_WHITE)
    cv2.imshow("plateContours",imgContours)
    listOfListsOfMatchingCharsInPlate = self.findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

    if (len(listOfListsOfMatchingCharsInPlate) == 0):			
        Plates.strChars = ""
        #continue						

    for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              
        listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        
        listOfListsOfMatchingCharsInPlate[i] = self.removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i]) 
                 
    intLenOfLongestListOfChars = 0
    intIndexOfLongestListOfChars = 0

    for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
        if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
            intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
            intIndexOfLongestListOfChars = i

    longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
    Plates.strChars = self.recognizeCharsInPlate(Plates.imgThresh, longestListOfMatchingCharsInPlate)
    #return listOfPossiblePlates
    
   
    return Plates

###################################################################################################
 def findPossibleCharsInPlate(self,imgGrayscale, imgThresh):
    listOfPossibleChars = []                        
    contours = []
    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        
        possibleChar = PossibleChar.PossibleChar(contour)

        if self.checkIfPossibleChar(possibleChar):              
            listOfPossibleChars.append(possibleChar)       

    return listOfPossibleChars
 '''
###################################################################################################
'''
 def findListOfListsOfMatchingChars(self,listOfPossibleChars):
    listOfListsOfMatchingChars = []                 

    for possibleChar in listOfPossibleChars:                       
        listOfMatchingChars = self.findListOfMatchingChars(possibleChar, listOfPossibleChars)        
        listOfMatchingChars.append(possibleChar)                
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue                            
        listOfListsOfMatchingChars.append(listOfMatchingChars)      
        listOfPossibleCharsWithCurrentMatchesRemoved = []
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        recursiveListOfListsOfMatchingChars = self.findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             
        break       
    return listOfListsOfMatchingChars


###################################################################################################
 def findListOfMatchingChars(self,possibleChar, listOfChars):
    listOfMatchingChars = []                
    for possibleMatchingChar in listOfChars:                
        if possibleMatchingChar == possibleChar:    
                                                    
            continue                                
        fltDistanceBetweenChars = self.distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = self.angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        
    return listOfMatchingChars                  

 '''

'''
 def removeInnerOverlappingChars(self,listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        
                                                                            
                if self.distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar) 
                            print "contour removed"
                    
                    else:                                                                       
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           
                    
    return listOfMatchingCharsWithInnerCharRemoved

###################################################################################################
 def recognizeCharsInPlate(self,imgThresh, listOfMatchingChars):

    strChars = ""               

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)
    
#############
    for currentChar in listOfMatchingChars:                                         
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2,SCALAR_GREEN, 2)          

        cv2.imshow("individual characters",imgThreshColor)
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           
        
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        

        npaROIResized = np.float32(npaROIResized)    
        
        
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              
        strCurrentChar = str(chr(int(npaResults[0])))            

        strChars = strChars + strCurrentChar 
    return strChars
        
        

        #print ann.predict(npaROIResized)
        
                           
        #sklearn mlp
        #print str(chr(int(clf.predict(npaROIResized))))
        


    #if Main.showSteps == True: # show steps 
     #   cv2.imshow("10", imgThreshColor)

     
'''
############################################









