import cv2 as cv
import numpy as np

body_classifier = cv.CascadeClassifier(r'D:\Autonomous-Guided-Vehicle\Pedestrian Tracking\haarcascade_fullbody.xml')

cap = cv.VideoCapture(r'D:\Autonomous-Guided-Vehicle\Resources\street2.wmv')
#out = cv.VideoWriter('Pedestrian.mp4', cv.VideoWriter_fourcc(*'m', 'p', '4', 'v'), 20, (700, 500))

while cap.isOpened():
    _,frame = cap.read()
    frame = cv.resize(frame,(700,500))
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    bodies = body_classifier.detectMultiScale(gray,1.6,3)

    for (x,y,w,h) in bodies:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.imshow("Frames",frame)
        #out.write(frame)

    if cv.waitKey(1) == 13:
        break
cv.destroyAllWindows()
cap.release()
#out.release()

