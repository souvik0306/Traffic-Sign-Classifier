import cv2 as cv

cap = cv.VideoCapture(r'D:\Autonomous-Guided-Vehicle\Resources\highway.mp4')
out = cv.VideoWriter('mask_output.mp4',cv.VideoWriter_fourcc(*'XVID'), 10, (700,500))

object_detector = cv.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    width,height = 700,500
    frame = cv.resize(frame,(width,height))

    mask = object_detector.apply(frame)
    new = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

    out.write(new)
    cv.imshow("Mask",mask)
    #cv.imshow("Frame",frame)
    key = cv.waitKey(30)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
out.release()

