import cv2
import numpy as np
import ProcessImage
import FindType

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
lower_yellow=np.array([17,100,100])
upper_yellow=np.array([35,255,255])

lower_white = np.array([0,0,135], dtype=np.uint8)
upper_white = np.array([255,120,255], dtype=np.uint8)
possiblePlates=[]
plates=[]
class Image:
  originalImg=None
  plate=None
    
  def main(self,filename):
     
    ProcessImageObj=ProcessImage.ProcessImage()
    LogoObj=IdentifyLogoTemplate.IdentifyLogo()
    Type="Private Vehicle"
    KNNTrainingSuccessful = ProcessImageObj.loadKNNDataAndTrainKNN()         
    if KNNTrainingSuccessful == False:                               
        print "unuccessful\n"               
        return                                                          
    self.originalImg  = cv2.imread(filename)    
    if self.originalImg is None:                            
        print "image not read"      
        return      
                                            
    self.ROI(self.originalImg)
    #if len(listOfPossibleROI)!=0:
     # for eachPlate in listOfPossibleROI:
    self.plate=cv2.imread("plate.jpg")
    
    #listOfPossiblePlates = DetectPlatesObj.detectPlatesInScene(plate)           
    #listOfPossiblePlates = DetectPlatesObj.detectCharsInPlates(listOfPossiblePlates) 
    self.plate=ProcessImageObj.initPlate(self)
    #listOfPossiblePlates = DetectPlatesObj.detectCharsInPlates(plate)  
    #Plates=DetectPlatesObj.detectCharsInPlates(plate)
    #ProcessImageObj.recognizeChars(self.plate.imgPlate)
    
    '''
    licPlate=Plates
    
    if Plates is None:                          
        print "\nno license plates were detected\n"             
    else:                                                       
        ##commented since starting
        
        #listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
    
        #licPlate = listOfPossiblePlates[0]
      
        #cv2.imshow("imgPlate", licPlate.imgPlate)  
        #cv2.imwrite("imgPlate.png",licPlate.imgPlate)
        
        #cv2.imshow("imgThresh", licPlate.imgThresh)
        #cv2.waitKey(0)
        #if len(licPlate.strChars) == 0:                     
         #   print "\nno characters were detected\n\n"       
          #  return                                          
   
        print "\nlicense plate read from image = "+licPlate.strChars        
        
        licPlate=Plates       
        #cv2.imshow("imgPlate", licPlate.imgPlate)  
        cv2.imwrite("imgPlate.png",licPlate.imgPlate)
    
        #cv2.imshow("imgThresh", licPlate.imgThresh)
        cv2.waitKey(0)
        
        cv2.imwrite("originalImg.png", originalImg)    
        print "\nlicense plate read from image = "+licPlate.strChars
     
    '''
    lower_red_limit=190
    lower_green_limit=130
    lower_blue_limit=10
    upper_red_limit=255
    upper_green_limit=230
    upper_blue_limit=75
    colors= FindType.colorz("imgPlate.png",5)
    for color in colors:
           red= color[1:3]
           green=color[3:5]
           blue=color[5:7]
           #print red,green,blue+"\n"
           intRed = int(red, 16)
           intGreen=int(green,16)
           intBlue=int(blue,16)
           #print intRed,intGreen,intBlue
           if(intRed>lower_red_limit and intRed<upper_red_limit and intGreen>lower_green_limit and intGreen<upper_green_limit and intBlue>lower_blue_limit and intBlue<upper_blue_limit):
               Type= "Commercial Vehicle"
            
    print "Type of Vehicle: " + Type  
    
    #FindZoneObj.findZone(licPlate.strChars)
    
    
    #imgROI=cv2.imread("plate.jpg")
   
    #mask=cv2.imread("imgPlate.jpg")
  
       
    cv2.waitKey(0)	
    return
###################################################################################################

  def ROI(self,img):
    
    #img =cv2.imread("pic10.jpg")
    #img = cv2.dilate(img, np.ones((3, 3)))
    width,height,channels=img.shape
    #print width
    if  width>1000:
        img = cv2.resize(img,(0,0),fx=0.3,fy=0.3)
    elif width>900:
        img = cv2.resize(img,(0,0),fx=0.45,fy=0.45)
    elif width>800:
        img = cv2.resize(img,(0,0),fx=0.4,fy=0.4)
    elif width>700:
        img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    #img = cv2.resize(img, (700, 700))
    
    cv2.imshow("pic",img)
    cv2.waitKey(0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    output_yellow = cv2.bitwise_and(img,img, mask= mask)
    cv2.imshow("inrange",output_yellow)
    cv2.waitKey(0)
    mask1 = cv2.inRange(hsv, lower_white, upper_white)
    output_white = cv2.bitwise_and(img,img, mask= mask1)
    cv2.imshow("inrangeWhite",output_white)
    cv2.waitKey(0)

    output_white=cv2.cvtColor(output_white, cv2.COLOR_BGR2GRAY)
    imgContours, contours, npaHierarchy = cv2.findContours(output_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width,channels = img.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(imgContours, contours, -1, SCALAR_WHITE)
    cv2.imshow("Contours", imgContours)
    cv2.waitKey(0)
	
    output_yellow=cv2.cvtColor(output_yellow, cv2.COLOR_BGR2GRAY)
    imgContours1, contours1, npaHierarchy = cv2.findContours(output_yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width,channels = img.shape
    imgContours1 = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(imgContours1, contours1, -1, SCALAR_WHITE)
    cv2.imshow("ContoursYellow", imgContours1)
    cv2.waitKey(0)
    imgContours2 = np.zeros((height, width, 3), np.uint8)

    for eachContour in contours:     
       rect = cv2.minAreaRect(eachContour)
       box = cv2.boxPoints(rect)
       box = np.int0(box)
       x1=box[[0],[0]]
       #x2=box[[1],[0]]
       x3=box[[2],[0]]
       #x4=box[[3],[0]]
       y1=box[[0],[1]]
       #y2=box[[1],[1]]
       y3=box[[2],[1]]
       #y4=box[[3],[1]]
       
       w=abs(x3-x1)
       h=abs(y3-y1)
       #cv2.drawContours(img,[box],0,Main.SCALAR_GREEN,2)
       #cv2.imshow("contours",img)
       areaBoundingRect=w*h
       area=cv2.contourArea(eachContour)
       if w>h  and areaBoundingRect>2000 and areaBoundingRect-area<2000:
          possiblePlates.append(eachContour)
    for eachContour in contours1:
        #x,y,w,h = cv2.boundingRect(eachContour)
       
       rect = cv2.minAreaRect(eachContour)
       box = cv2.boxPoints(rect)
       box = np.int0(box)
       x1=box[[0],[0]]
       x3=box[[2],[0]]
       y1=box[[0],[1]] 
       y3=box[[2],[1]]
       
       w=abs(x3-x1)
       h=abs(y3-y1)
       #cv2.drawContours(img,[box],0,Main.SCALAR_GREEN,2)
       #cv2.imshow("contours",img)
       
       areaBoundingRect1=w*h
       area1=cv2.contourArea(eachContour)
       
       if w>h  and areaBoundingRect1>2000  and areaBoundingRect1-area1<3000:
          possiblePlates.append(eachContour)
    cv2.drawContours(imgContours2,possiblePlates,-1,SCALAR_WHITE)
    cv2.imshow("possiblePlates",imgContours2)
    cv2.waitKey(0)
    
    for eachPossiblePlate in possiblePlates:
        x,y,w,h=cv2.boundingRect(eachPossiblePlate)
        if y-100>0:
           plates.append(img[y:y+h+5,x:x+w])
           logos.append(img[y-90:y,x:x+w])
        #if raw_input("Only number plate?")=='y':
         #  plates.append(img[y:y+h,x:x+w])
        #else
            #plates.append(img[y:y+h,x:x+w])
        
        '''
       rect = cv2.minAreaRect(eachPossiblePlate)
       box = cv2.boxPoints(rect)
       box = np.int0(box)
       x1=box[[0],[0]]
       x2=box[[1],[0]]
       x3=box[[2],[0]]
       y1=box[[0],[1]]
       y2=box[[1],[1]]
       y3=box[[2],[1]]
       X1=x1[0]
       X2=x2[0]
       X3=x3[0]
       Y1=y1[0]
       Y2=y2[0]
       Y3=y3[0]
       print X1,Y1
       print X3,Y3
       if Y3>Y1:
          plates.append(img[Y3:Y1,X1:X3])
       else:
           plates.append(img[Y1:Y3,X1:X3])
        '''
        
    for eachPlate in plates:
        if width>600:
            eachPlate=cv2.resize(eachPlate,(0,0),fx=2,fy=2)
        elif width>500:
            eachPlate = cv2.resize(eachPlate, (0,0),fx=2.5,fy=2.5)
            #print "image resized"
        elif width>400:
            eachPlate = cv2.resize(eachPlate, (0,0),fx=2.5,fy=2.5)
            

        #elif width>800:   
             #eachPlate = cv2.resize(eachPlate, (0,0),fx=2,fy=2)
        self.img=eachPlate
        cv2.imshow("plates",eachPlate)
        cv2.imwrite("plate.jpg",eachPlate)
        cv2.waitKey(0)
        
    for eachLogoROI in logos:
        if width>600:
            eachLogoROI=cv2.resize(eachLogoROI,(0,0),fx=2,fy=2)
        elif width>500:
            eachLogoROI= cv2.resize(eachLogoROI, (0,0),fx=2.5,fy=2.5)
            #print "image resized"
        elif width>400:
            eachLogoROI = cv2.resize(eachLogoROI, (0,0),fx=2.5,fy=2.5)
        cv2.imshow("logos",eachLogoROI)
        cv2.imwrite("logos.jpg",eachLogoROI)
        cv2.waitKey(0)

#if __name__ == "__main__":
#        main()
