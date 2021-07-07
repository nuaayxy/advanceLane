# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.lib.function_base import average
import line
#%matplotlib qt

## previous poly
# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
# allthese should be in object orientated progarmming but here just quick 
left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])
average_fit_left = []
average_fit_right = []
pre_left_fit =  np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
pre_right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

'''
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs =cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, None)
    return undist
    
    
'''

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # print( (leftx.size + rightx.size) / np.shape(nonzero)[1] )

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    # cv2.imshow('search', result)
    
    return result


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:-20, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int64(histogram.shape[0]//2)
    leftx_base_bias = 10
    leftx_base = np.argmax(histogram[ leftx_base_bias:(midpoint)])
    leftx_base += leftx_base_bias 
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 100
    # Set min5mum number of pixels found to recenter window
    minpix = 30

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int64(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(0,nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    #give them some initial values in case
    left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
    right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

    # safe check around previous correctly fit 
    nonzero = binary_warped.nonzero()
    ratio = ( (leftx.size + rightx.size) / np.shape(nonzero)[1] )
    if ratio < 0.8:
        search_around_poly(binary_warped, pre_left_fit, pre_right_fit)

    # Fit a second order polynomial to each using `np.polyfit`
    # print(np.shape(leftx)[0])
    if rightx.size != 0 and leftx.size!=0: 
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        
        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        
        plt.imshow(binary_warped)

    #do an moving average of 8 previous fits
    if len(average_fit_right) < 8:
        average_fit_right.append(left_fit)
        average_fit_left.append(right_fit)
    else:
        # average_fit_right.append(left_fit)
        # average_fit_left.append(right_fit)
        average_fit_left.pop(0)
        average_fit_right.pop(0)
    left_fit=  np.average(average_fit_left,axis=0)
    right_fit= np.average(average_fit_right,axis=0)

    pre_right_fit = right_fit
    pre_left_fit = left_fit

    return out_img, left_fit ,right_fit



def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    img_size = img.shape
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    


# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(40, 200)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]


    s_channel_hsv = hsv[:,:,1]
    v_channel_hsv = hsv[:,:,2]

    #cv2.imshow("cc",hsv)
    # Sobel x
    sobelx = cv2.Sobel(v_channel_hsv, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255


    v_binary_hsv = np.zeros_like(v_channel_hsv)
    v_binary_hsv[(v_channel_hsv >= 200)] = 255

    LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    B_channel = LAB[:,:,1]
    B_channel = LAB[:,:,2]
    B_binary = np.zeros_like(v_channel_hsv)
    # B_binary[(B_channel<125)] = 255
    B_binary[(B_channel<100)] = 255
    
    LUV = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    LUV_L_channel = LUV[:,:,0]
    LUV_binary = np.zeros_like(LUV_L_channel)
    LUV_binary[(LUV_L_channel>210)] = 255

    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

    # cv2.imshow("luv binary",  LUV_binary)
    # cv2.imshow("edge",  sxbinary)
    # cv2.imshow("lab b channel",  B_binary)


#     Along with color threshold to the B(range:145-200) in LAB for shading & brightness applying it to R in RGB in final pipeline can also help in detecting the yellow lanes.
# And thresholding L (range: 215-255) of LUV for whites.
    
    combined_binary = np.zeros_like(sxbinary)
    # combined_binary[(s_binary == 255) | (LUV_binary==255)] = 255
    combined_binary[((s_binary == 255) | (sxbinary == 255)) | (LUV_binary==255) | (B_binary==255)] = 255
    # | (B_binary==255) 
    # combined_binary[(s_binary == 255) ] = 255
#    combined_binary[(s_binary == 255) ] = 255
    
    cv2.imshow("binary_combined",combined_binary)
    return combined_binary



def calibrateCamera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    cv2.destroyAllWindows()
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs =cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
    print(ret, mtx, dist)
    return ret, mtx, dist
    
    
def measure_curvature_pixels(left_fit_cr, right_fit_cr ):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
        # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    y_eval = np.max(ploty)
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    
    left_point = left_fit_cr[0]*720**2 + left_fit_cr[1]*720 + left_fit_cr[2]
    right_point = right_fit_cr[0]*720**2 + right_fit_cr[1]*720 + right_fit_cr[2]
    mid_point = (left_point + right_point)//2 - 1280//2
    mid_point = mid_point * (3.7/700)
    
    print(left_curverad, 'm', right_curverad, 'm', mid_point)
    return left_curverad, right_curverad, mid_point


def drawlane(warped, left_fit, right_fit, M, undist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    return result


def main():
    ret, mtx, dist = calibrateCamera()
    #read video
    cap = cv2.VideoCapture('project_video.mp4')
    
#    example = cv2.imread("./examples/warped_straight_lines.jpg")
#    cv2.imshow("k",example)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
   
    size = (frame_width, frame_height) 
    result = cv2.VideoWriter('filename.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 30, size) 

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_binary = pipeline(frame)
        # cv2.imshow("b", frame)
     
        undist = cv2.undistort(frame_binary, mtx, dist, None, mtx)
             
        # Given src and dst points, calculate the perspective transform matrix
        src = np.float32([[568,470],[717,470], [260,680],[1043,680]] )
        dst = np.float32([[200,0],  [1000,0],[200,680], [1000,680]])
        M = cv2.getPerspectiveTransform(src, dst)

        img_size = (undist.shape[1], undist.shape[0])
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)
        out_img,left_fit ,right_fit = fit_polynomial(warped)   
        cv2.imshow("d", out_img)
        left_curve, right_curve, mid = measure_curvature_pixels(left_fit, right_fit ) 
        lane_result  = drawlane(warped, left_fit, right_fit, np.linalg.inv(M), frame)
        cv2.putText(lane_result, "left curvature: "+ str(int(left_curve)) + " m",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),2)
        cv2.putText(lane_result, "right curvature: "+ str(int(right_curve)) + " m",(100,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),2)
        cv2.putText(lane_result, "vehicle is "+ str(round(mid,3)) + "m from center",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),2)
        cv2.imshow("final_output", lane_result)
        result.write(lane_result) 


        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    
    
    
    #find pixels that related to lane line
    
    #sliding window and history prior to fit poly
    
    #draw 

if __name__ == "__main__":
    main()
