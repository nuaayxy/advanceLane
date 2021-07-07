## Advanced Lane Finding


In this project, we create a pipeline to identify the lane boundaries in a video, 
the main function and code in detect.py
![code](./detect.py)
I saved examples of the output from each stage in the folder called `output_images`
There is also a video result
![output](.//final_output.avi)

There are issues when in the shadows or light changes suddenly, the lanes detected will tend to fail, then what i rely on is the previously successfully detected lines.
There are several safe proof way to avoid sudden jump

* check number of detected binary pixels and if below threshold we search around previous detected poly
* use a moving average of the lane filter away some jumps
* Use of different color spaces and combine them together such as LAB/HSV with gradients

Futher improvement would be using deep learning method for more robust detection especially these edge cases when lighting changes rapidly and we cannot hand engineer everything

The Project
---

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.(done)
* Apply a distortion correction to raw images.(done)
* Use color transforms, gradients, etc., to create a thresholded binary image.
![threshold find potential lane pixels](.//output_images/binary_lane.png)
* Apply a perspective transform to rectify binary image ("birds-eye view").
![warped pespective transform](.//output_images/warped.png)
* Detect lane pixels and fit to find the lane boundary. 
![lane fit curve](.//output_images/Figure_1.png)
* Determine the curvature of the lane and vehicle position with respect to center. 
* Warp the detected lane boundaries back onto the original image.
![lane boundry](.//output_images/lane.png)
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


![Lanes Image](./final_output_screenshot_07.07.2021.png)

