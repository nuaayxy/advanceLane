## Advanced Lane Finding


In this project, we create a pipeline to identify the lane boundaries in a video, 
the main function and code in detect.py
![code](./detect.py)
I saved examples of the output from each stage in the folder called `output_images`
There is also a video result
![output](.//output_images/output.avi)


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

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)
![Lanes Image](./final_output_screenshot_07.07.2021.png)

