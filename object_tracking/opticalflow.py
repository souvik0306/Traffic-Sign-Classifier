import numpy as np
import cv2 as cv

cap = cv.VideoCapture(r'D:\Autonomous-Guided-Vehicle\Resources\street3.wmv')

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

lucas_kanade_params = dict(winSize=(15, 15), maxLevel=2,
                           criteria=(cv.TERM_CRITERIA_EPS | cv.TermCriteria_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

_, pre_frame = cap.read()
#pre_frame = cv.resize(pre_frame, (500, 500))
gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)

prev = cv.goodFeaturesToTrack(gray, mask=None, **feature_params)

mask = np.zeros_like(pre_frame)

while True:
    _, frame = cap.read()
    #frame = cv.resize(frame, (500, 500))
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    new_corners, status, errors = cv.calcOpticalFlowPyrLK(gray, frame_gray, prev, None, **lucas_kanade_params)

    good_new = new_corners[status == 1]
    good_old = new_corners[status == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), 1)

    img = cv.add(frame, mask)

    cv.imshow('Optical Flow', img)
    if cv.waitKey == 27:
        break
    gray = frame_gray.copy()
    prev = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
cap.release()
