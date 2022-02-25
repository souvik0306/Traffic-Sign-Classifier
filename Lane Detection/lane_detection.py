import cv2
import matplotlib.pyplot as plt
import numpy as np

def lane_detect(img):
    img = cv2.resize(img,(550,275))

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(7,7),0)
    img_canny = cv2.Canny(img_blur,10,200)

    vertices = np.array([[(50,260),(160,98),(410,98),(550,260)]],dtype=np.int32)
    mask = np.zeros_like(img_gray)
    cv2.fillPoly(mask,vertices,255)

    # masked_image = cv2.bitwise_and(img_gray,mask)
    masked_image = cv2.bitwise_and(img_canny,mask)

    rho = 2
    theta = np.pi/100
    threshold = 20
    min_line_len = 100
    max_line_gap = 50
    lines = []
    lines = cv2.HoughLinesP(masked_image,rho,theta,threshold,np.array([]),min_line_len,max_line_gap)
    lines_image = np.zeros((masked_image.shape[0],masked_image.shape[1],3),dtype=np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(lines_image,(x1,y1),(x2,y2),[0,0,255],20)
    a = 1
    b = 1
    g = 0

    image_with_lines = cv2.addWeighted(img,a,lines_image,b,g)
    
    return cv2.imshow('Result',image_with_lines),cv2.waitKey(150)
    
cap = cv2.VideoCapture(r'lanes_clip.mp4')
while 1:
    _,frame = cap.read()
    result = lane_detect(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frame rate: ", int(fps), "FPS")
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()

