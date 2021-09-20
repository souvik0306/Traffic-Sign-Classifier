import cv2 as cv
import imageio
cap = cv.VideoCapture(r'D:\Autonomous-Guided-Vehicle\Resources\highway.mp4')
out = cv.VideoWriter('contour_output.mp4',cv.VideoWriter_fourcc(*'XVID'), 10, (700,500))

object_detector = cv.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    width,height = 700,500 #resizing 
    frame = cv.resize(frame,(width,height))
    mask = object_detector.apply(frame) 

    countours, _ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 
    for cnt in countours:
        area = cv.contourArea(cnt)
        if area > 150:  #150 pixels
            cv.drawContours(frame,[cnt],-1,(0,255,0),thickness=2)
    out.write(frame)
    cv.imshow("Mask",frame)
    key = cv.waitKey(30)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()

