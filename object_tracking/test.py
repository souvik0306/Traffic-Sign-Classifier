import cv2 as cv
import imageio
cap = cv.VideoCapture(r'D:\Autonomous-Guided-Vehicle\Resources\highway.mp4')
#out = cv.VideoWriter('contour_output.wmv',cv.VideoWriter_fourcc(*'XVID'), 10, (700,500))

object_detector = cv.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
while True:
    ret, frame = cap.read()
    width,height = 700,500 #resizing 
    frame = cv.resize(frame,(width,height))

    # Extract region of interest 
    roi = frame[100:500,200:600]
    mask = object_detector.apply(roi) 
    _,mask = cv.threshold(mask,254,255,cv.THRESH_BINARY)
    countours, _ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 
    for cnt in countours:
        area = cv.contourArea(cnt)
        if area > 150:  #150 pixels
            #cv.drawContours(roi,[cnt],-1,(0,255,0),thickness=2)
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    #out.write(frame)
    cv.imshow("ROI",roi)
    cv.imshow("Mask",mask)
    key = cv.waitKey(30)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()

