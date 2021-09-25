### Autonomous-Guided-Vehicle

Autonomous Braking in a vehicle is governed by certain key parameters - 
1. ***Obstacle Detection & Tracking*** - This revolves around identifying common objects in the path of a car.
2. ***Obstacle's Distance Estimation*** - Assessment of the distance of an obstacle from a particular point is fundamental for autonomous braking.
3. ***Obstacle's Speed Evaluation*** - Finding the relative speed between these two players is crucial for approximating stopping time and the required deacceleration amount.

Pedestrian Detection - 

<img src="https://github.com/souvik0306/Autonomous-Guided-Vehicle/blob/master/Resources/Pedestrian.gif" width="600" height="300">

> Obstacle Detection uses [`cv2.findContours`](https://docs.opencv.org/3.4.15/df/d0d/tutorial_find_contours.html)  to isolate contours in a masked image and sort out those which are above a certain threshold/value.

The function accepts three positional arguments `cv2.findContours(image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)` -
1. First argument takes in the source image/frame 
2. Second one is contour retrieval mode
3. Third argument is contour's approximation

Countour Map of the Region of Interest (ROI) - 

<img src="https://github.com/souvik0306/Autonomous-Guided-Vehicle/blob/master/Resources/roioutput.gif" width="600" height="300">

Numbered Map of the Region of Interest (ROI) -  

<img src="https://github.com/souvik0306/Autonomous-Guided-Vehicle/blob/master/Resources/numbered.gif" width="600" height="300">

Masked Video of a Highway - 

<img src="https://github.com/souvik0306/Autonomous-Guided-Vehicle/blob/master/Resources/mask_output.gif" width="600" height="300">

Countour Map of the Entire Video Frame - 

<img src="https://github.com/souvik0306/Autonomous-Guided-Vehicle/blob/master/Resources/contour_output.gif" width="600" height="300">

### *References* - 
Object Tracking PySource - [YouTube](https://www.youtube.com/watch?v=O3b8lVF93jU&list=LL&index=1&ab_channel=Pysource)

