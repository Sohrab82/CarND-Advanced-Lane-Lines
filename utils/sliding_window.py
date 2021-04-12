import numpy as np
import cv2
from utils.misc import fit_poly


def find_lane_pixels(binary_warped, nwindows, margin, minpix):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    # nwindows = 9
    # Set the width of the windows +/- margin
    # margin = 100
    # Set minimum number of pixels found to recenter window

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
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

    # if the search window is partially out of the image
    # means that probably the right/left lane has exited
    # the right/left border of the image
    right_lane_in_roi = True
    left_lane_in_roi = True

    # total area scanned for lane points
    right_search_area = 0
    left_search_area = 0

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### Identify the nonzero pixels in x and y within the window ###
        if left_lane_in_roi:
            good_left_inds = ((win_xleft_low <= nonzerox) & (nonzerox <= win_xleft_high) & (
                win_y_low <= nonzeroy) & (nonzeroy <= win_y_high)).nonzero()[0]
        else:
            good_left_inds = np.array([], dtype=np.int)

        if right_lane_in_roi:
            good_right_inds = ((win_xright_low <= nonzerox) & (nonzerox <= win_xright_high) & (
                win_y_low <= nonzeroy) & (nonzeroy <= win_y_high)).nonzero()[0]
        else:
            good_right_inds = np.array([], dtype=np.int)

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # cv2.imshow('sdf', out_img)
        # cv2.waitKey(0)

        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            # binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high], axis=0)
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        elif win_xleft_low < 0:
            # search window is partially out of the image
            # meaning that probably the right lane has exited the right border of the image
            left_lane_in_roi = False

        if len(good_right_inds) > minpix:
            # binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high], axis=0)
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
        elif win_xright_high >= binary_warped.shape[1]:
            # search window is partially out of the image
            # meaning that probably the right lane has exited the right border of the image
            right_lane_in_roi = False

        # calculate the search area (window size) excluding the areas that are out of the image
        win_xleft_low = min(0, win_xleft_low)
        win_xleft_high = max(win_xleft_high, binary_warped.shape[1])
        left_search_area += (win_xleft_high - win_xleft_low) * window_height

        win_xright_low = min(0, win_xleft_low)
        win_xright_high = max(win_xright_high, binary_warped.shape[1])
        right_search_area += (win_xright_high - win_xright_low) * window_height

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    left_conf = len(left_lane_inds) / left_search_area
    right_conf = len(right_lane_inds) / right_search_area

    # Extract left and right line pixel positions
    if len(left_lane_inds) > 0:
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
    else:
        # left line not detected
        leftx = np.array([])
        lefty = np.array([])
    if len(right_lane_inds) > 0:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    else:
        # right line not detected
        rightx = np.array([])
        righty = np.array([])

    return leftx, lefty, left_conf, rightx, righty, right_conf, out_img


def find_lane_from_histo(binary_warped, nwindows, margin, minpix,
                         left_lane, right_lane,
                         update_left_lane, update_right_lane):

    # Find our lane pixels first
    leftx, lefty, left_conf, rightx, righty, right_conf, out_img = find_lane_pixels(
        binary_warped, nwindows, margin, minpix)

    ### Fit a second order polynomial to each using `np.polyfit` ###
    if len(leftx) != 0:
        if update_left_lane:
            left_fitx, left_fit_coeff, ploty = fit_poly(
                binary_warped.shape, leftx, lefty)
    else:
        # lane not detected even with histo search
        # so just keep the lane previous time step
        update_left_lane = False

    if len(rightx) != 0:
        if update_right_lane:
            right_fitx, right_fit_coeff, ploty = fit_poly(
                binary_warped.shape, rightx, righty)
    else:
        # lane not detected even with histo search
        # so just keep the lane previous time step
        update_right_lane = False

    ## Visualization ##
    # Colors in the left and right lane regions
    if len(leftx) != 0:
        out_img[lefty, leftx] = [255, 0, 0]

    if len(rightx) != 0:
        out_img[righty, rightx] = [0, 0, 255]

    print('search from histo')
    # print(left_conf)
    # print(right_conf)
    if update_left_lane:
        left_lane.update(left_fitx, left_fit_coeff, ploty, 1.0)
    if update_right_lane:
        right_lane.update(right_fitx, right_fit_coeff, ploty, 1.0)

    return out_img
