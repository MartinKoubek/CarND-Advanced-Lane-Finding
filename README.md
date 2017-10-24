# Advanced Lane Finding

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video.  

<img src="./output_images/7_test1_result.png_1.png" alt="hog" width="600">

The youtube link with a result is provided here:
https://youtu.be/s-wlc-a5e60

## The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!



## Rubric points

### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

Any camera lens will have distortion problem. In order to fix distortion, following steps were done:

* Calibration images with chess metrix were loaded
* Chessboard corners were found
* Camera calibration were found with cv2.calibrateCamera function
* Each image (video frame) was calibrated

Original image             |  Calibrated image
:-------------------------:|:-------------------------:
![Original image]( ./output_images/calibration1.jpg)   |  ![Calibrated image]( ./output_images/0_camera_calibration_1.png)

Original image             |  Calibrated image
:-------------------------:|:-------------------------:
![Original image]( ./output_images/calibration2.jpg)   |  ![Calibrated image]( ./output_images/0_camera_calibration_2.png)


### Apply a distortion correction to raw images.

Distortion was applied to each image (video frame).

Original image             |  Calibrated image
:-------------------------:|:-------------------------:
![Original image]( ./output_images/1_straight_lines2_original_image_1.png)   |  ![Calibrated image]( ./output_images/2_straight_lines2_calibrated_image_1.png)


### Use color transforms, gradients, etc., to create a thresholded binary image.

Color transform is key part of line detection. In the project I use a combination of S channel of HLS and L channel of LUV.

Original image             |  Gradient image
:-------------------------:|:-------------------------:
![Original image]( ./output_images/2_straight_lines2_calibrated_image_1.png)   |  ![Calibrated image]( ./output_images/3_straight_lines2_gradient_image_1.png)
![Original image]( ./output_images/2_test5_calibrated_image_1.png)   |  ![Calibrated image]( ./output_images/3_test5_gradient_image_1.png)

### Apply a perspective transform to rectify binary image ("birds-eye view").

The perspective transform was used by estimated parameters.

Original image             |  Birds-eye view image
:-------------------------:|:-------------------------:
![Original image]( ./output_images/4_straight_lines2_transform_orig_1_image.png)   |  ![Calibrated image]( ./output_images/4_straight_lines2_transform_image_1.png)
![Original image]( ./output_images/4_test5_transform_orig_1_image.png)   |  ![Calibrated image]( ./output_images/4_test1_transform_image_1.png)


### Detect lane pixels and fit to find the lane boundary.

Lane detection is shown on the following picture.

Detect rawly line             |  Draw line
:-------------------------:|:-------------------------:
![Original image]( ./output_images/5_straight_lines2_line_detection_1.png)   |  ![Calibrated image]( ./output_images/6_straight_lines2_line_determination_1.png)
![Original image]( ./output_images/5_test5_line_detection_1.png)   |  ![Calibrated image]( ./output_images/6_test5_line_determination_1.png)

### Determine the curvature of the lane and vehicle position with respect to center.

Curvature detection was done by these parameters:
* ym_per_pix = 30.48/100
*  xm_per_pix = 3.7/100

The radius of curvature is computed upon calling the DetectLines.detect() method. At the end of method curvature is 
calculated as polynomial f(y)=A y^2 +B y + C  and the radius of curvature is given by R = [(1+(2 Ay +B)^2 )^3/2]/|2A|.

The distance from the center of the lane is computed in DetectLines.detect() method, which measures the distance between center of an image and middle point of left and right line.

### Warp the detected lane boundaries back onto the original image.

Warped image is shown on the following picture.

![Un warped image]( ./output_images/7_straight_lines2_result.png_1.png)

![Un warped image]( ./output_images/7_test1_result.png_1.png)


### Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The youtube link with a result provided here:
https://youtu.be/s-wlc-a5e60

## Discussion


### What problems/issues did you face in your implementation of this project?
I had a trouble with coding in python, specificly numpy arrays as I did not know what numpy functions makes with arrays.

### Where will your pipeline likely fail? 
* Low lights and large lane shadows
* Bumbing road

### What could you do to make it more robust?
I see three crutial steps to improve in Lane detection
* Use color transforms, gradients - this step can be very much improved by using various filters in different lighting scenarios
* Detect lane pixels - detection of "right" pixels is very crutial as time to time "obstacles" are detected as part of line
* Lane smoothing and filtering  - averaging lines in a time, removing "wrong" lines can rapidly improve lane detection





