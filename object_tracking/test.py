import cv2 as cv
from tracker import *


tracker = EuclideanDistTracker()
cap = cv.VideoCapture(r'D:\Autonomous-Guided-Vehicle\Resources\highway.mp4')
out = cv.VideoWriter('numbered.mp4', cv.VideoWriter_fourcc(*'m', 'p', '4', 'v'), 20, (400, 400))

object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    width, height = 700, 500  # resizing
    frame = cv.resize(frame, (width, height))

    # Extract region of interest 
    roi = frame[100:500, 200:600]
    # Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    countours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in countours:
        area = cv.contourArea(cnt)
        if area > 150:  # 150 pixels
            # cv.drawContours(roi,[cnt],-1,(0,255,0),thickness=2)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            detections.append([x,y,w,h])

    # Object Tracking
    boxes = tracker.update(detections)
    print(boxes)
    for box_id in boxes:
        x,y,w,h,id = box_id
        cv.putText(roi,str(id),(x,y-15),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    out.write(roi)
    cv.imshow("Mask", mask)
    cv.imshow("ROI", roi)
    key = cv.waitKey(10)
    if key == 27:
        break

cap.release()
out.release()
cv.destroyAllWindows()
