import numpy as np
import cv2
import imutils
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
class IdentifyLogo:
  def identify(self):
   img=cv2.imread("logos.jpg")
# Convert to grayscale
   imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   imgThresh = cv2.adaptiveThreshold(imageGray, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

   '''
# Find template
   result = cv2.matchTemplate(imageGray,templateGray, cv2.TM_CCOEFF)
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
   print max_val
   top_left = max_loc
   h,w = templateGray.shape
   bottom_right = (top_left[0] + w, top_left[1] + h)
   cv2.rectangle(img,top_left, bottom_right,(0,0,255),4)
 
# Show result
#cv2.imshow("Template", template)
   cv2.imshow("Result", img)
   cv2.waitKey(0)
   '''
   found = None
   for i in range(0,4):
       template = cv2.imread(str(i)+'.jpg')
       templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
       templateThresh = cv2.adaptiveThreshold(templateGray, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

       #templateGray = cv2.Canny(templateGray, 50, 200)
       (tH, tW) = templateThresh.shape[:2]
       for scale in np.linspace(0.2, 1.0, 10)[::-1]:
            resized = imutils.resize(imgThresh, width = int(imgThresh.shape[1] * scale))
            r = imgThresh.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
			break
            #resized = cv2.Canny(resized, 50, 200)
            cv2.imshow("resized",resized)
            cv2.waitKey(0)
            result = cv2.matchTemplate(resized, templateThresh, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                 found = (maxVal, maxLoc, r)
                 logo=i
   (_, maxLoc, r) = found
   #print maxVal
   (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
   (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
   #cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
   cv2.imshow("Image", img)
   
   if logo==0:
       print "Vehicle: Volkswagen"
   elif logo==1 or logo==3:
       print "vehicle: Maruti Suzuki"
   elif logo==2:
       print "Vehicle: Honda"
   
   cv2.waitKey(0)
            
#if __name__ == "__main__":
 #       identify()
            
            
            
            
            
            
            
            