## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

The Project
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

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.

[//]: # (Image References)

[image1]: ./output_images/undistorted.jpg "Undistorted"
[image2]: ./output_images/0_0_transformed.jpg "Road Transformed"
[image3]: ./output_images/0_1_filtered.jpg "HLS+Sobel+ROI Masks"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/0_2_birdeye.jpg "bird's-eye view"
[image6]: ./output_images/0_3-hist_detected.jpg "Fit Visual"
[image7]: ./output_images/0_4_output.jpg "Output"
[video1]: ./output_images/project_video.mp4 "Video"

### Camera Calibration

The code is contained in `./utils/camera_calibration.py`).  

`CameraCalibration.calc_camera_calibration_params()` takes a folder on chessboard images (nx=9) & (ny=6) and detects all chessboard corners in a every image. The output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The camera calibration params are saved in `my_wide_dist_pickle.p` for later use.
Distortion correction is done to test images using the `cv2.undistort()` function: 

![alt text][image1]

### Pipeline

#### 0. Overview & Code Structure

The starting point script is called `run_me.py` within which a function `process_image` takes an input image and processes it to find the lane lines.

##### `process_image` performs the following tasks:
  - Undistort the image useing the previously calcuated camera parameters,
  - Calculate Sobel x-gradients, HSL image, and ROI and creates a binary image containing filtered image with the lane lines included,
  - Warps the perspective to create a bird's-eye's view of the road,
  - Uses `find_lane_from_histo` or `search_around_poly` functions to find the left and right lanes (How the function decides on which search function to use is described below),
  - Measures curvature and distance of the vechile to the center of the road and plots the detected lane lines and measured data to the image frame.

The main (but costly) algorithm for lane detection is `find_lane_from_histo` which uses sliding windows and histogram of all detected points in y-direction to find the lane lines.

`search_around_poly` is a simpler algorithm that only scans (with some margin) around the previously detected lanes.

##### Algorithm selection

`process_image` runs `find_lane_from_histo` to find the lanes. After that, `search_around_poly` is used to find the lanes in the comming frames. 

If `search_around_poly` fails to find the lane, it will use the previously detected lanes until a threshold of `CONSEC_FAILED_DET_THRS=10` is reached. This threshold says that after 10 consecutuve failure in `search_around_poly`, the other algorithm,  `find_lane_from_histo`, should be used to find the lanes, a "reset" in a sense. After the reset, `search_around_poly` will be used until the threshold is reached again.

Failure thresholding mentioned above, works for each lane separately; i.e a reset might happen for the left lane while the right lane is still being calculated with `search_around_poly` algorithm (`num_consec_failed_detctn_l` and `num_consec_failed_detctn_r` for number of consecutive failed detections for the left and right lane respectively).

Please note that each lane has its own data and confidence defined according to class `Lane` defined in `utils/lane.py`. Confidence is defined as the number of detected lane point divided by the area of scanned region (e.g `len(leftx) / left_poly_area`).

##### Debugging

At the bottom of `run_me.py`, there is a loop that open the input video and sends the frame to `process_image`. For debugging purposes, two variables are defined; `show_plots` and `save_plots`.

If `show_plots` is set to True, all the generated plots for each frame will be shown.

If `save_plots` is set to True, all the generated plots for each frame will be saved as `{frame_number}_*_filename.jpg` in the output folder. This can be used to find annomolies in the algorithms, or find and process a distinct frame separately, but slows down the execution significantly.

Other than `run_me.py`, all the other functions are collected in .py files inside `utils` folder.

#### 1. A distortion-corrected image

Distortion correction applied to one of the test images:
![alt text][image2]

#### 2. Color transforms and gradient filtering to create a thresholded binary image

`./utils/misc.py/calc_hsl_sobelx_mask()` uses a combination of color (HLS color space) and gradient thresholds to generate a binary image. It returns `s_binary` (the color thresholded mask) and `sx_binary` (the Sobel's x-gradient thresholded mask). Both masks are applied to the image in `run.py/process_image()` in section `# apply HLS thresholding & sobel x filtering on the image`.
An ROI masking (`top_left=(510, 440), top_right=(770, 440), bottom_left=(100, 719), bottom_right=(1180, 719)`) is also done in `run.py/process_image()` in section (`# apply ROI mask`).
The final mask (color+gradient+ROI) is saved in `lanes_mask`.

![alt text][image3]

#### 3. Perspective transform

`./utils/misc.py/calc_tranform_matrix()` takes in four points on the source image as references and calculates the transform matrix `W` which transforms the source points into a rectangle with left & right margins of `lr_margin=200`. The reference coordinates are measured on a sample image with straight lane lines. 
A test image and its warped counterpart is shown to verify that the lines appear parallel in the warped image.

![alt text][image4]

`run_me.py/process_image()` in section `# apply perspective transform` performs the view transform to find the bird's-eye view of the lanes. The output is saved in `lanes_bird's-eye` variable.

![alt text][image5]

#### 4. Identify lane-line pixels and fit their positions with a polynomial

`./utils/sliding_window.py/find_lane_from_histo()` takes in the bird's-eye view of the lanes and applies histogram thresholding to find the lane lines. It divides the image vertically into `nwindows=9` sliding windows for each lane to get a better estimate of the lane lines. A minimum of `minpix=20` pixels are needed to be found in each box.

`./utils/misc.py/fit_poly()` takes in the deteced points for each lane and fits a second order polynomial to it.


`find_lane_from_histo()` returns 
 - a color image with left and right detected lane lines,
 - coordinates iof those points in `left_fitx` and `right_fitx` and `ploty`,
 - the polynomial coeffients for the left and right lane `left_fit_coeff` and `right_fit_coeff`. 
- the confidences as defined before.

The data corresponding to each lane is save in its `Lane` object accordingly (`left_lane`, `right_lane` objects).

![alt text][image6]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

Lanes curvature calculations are done in `run.py/process_image()` in section `# lanes curvature measurement` by calling the function `utils/misc.py/measure_curvature_pixels()`. Two variables `mx` and `my` are passed to this function which define the meter-per-pixel scales in x and y direction. 

A scaled parabola `x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c` is used to calculate the curvature in meter scale instead of pixels. `a` is replaced with `mx / (my ** 2) * a` and `b` is replaced with `(mx/my) * b` is the curvature calculation equation.


#### 6. Example image of the result plotted back down onto the road

`utils/misc.py/plot_lanes_on_road()` takes in the undistorted image along with x & y coordinates of the lane lines points, transofroms those points from bird's-eye view back to original camera view, and fills the spaces between them (space between lanes) and plots the output image.

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4)

---

### Future works

- Tune the threshold values to lane-detection confidence values.
- Add logic to handle when lane does not exist for many consecutive frames and both algorithm fail in detection.
- Both algorithms fails in too curvy roads, a better solution should be implemented.
- Study the algorithms while changing lanes
