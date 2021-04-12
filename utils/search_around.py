import numpy as np
import cv2
from utils.misc import fit_poly


def search_around_poly(binary_warped, margin, left_lane, right_lane):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    # margin = 100

    left_fit_coeff = left_lane.fit_coeff
    right_fit_coeff = right_lane.fit_coeff

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit_coeff[0] * (nonzeroy**2) + left_fit_coeff[1] * nonzeroy +
                                   left_fit_coeff[2] - margin)) & (nonzerox < (left_fit_coeff[0] * (nonzeroy**2) +
                                                                               left_fit_coeff[1] * nonzeroy + left_fit_coeff[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_coeff[0] * (nonzeroy**2) + right_fit_coeff[1] * nonzeroy +
                                    right_fit_coeff[2] - margin)) & (nonzerox < (right_fit_coeff[0] * (nonzeroy**2) +
                                                                                 right_fit_coeff[1] * nonzeroy + right_fit_coeff[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    new_left_fitx, new_left_fit_coeff, ploty = fit_poly(
        binary_warped.shape, leftx, lefty)
    new_right_fitx, new_right_fit_coeff, ploty = fit_poly(
        binary_warped.shape, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([new_left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([new_left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([new_right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([new_right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # area of the lane with its two marginalied lines around it
    left_poly_area = cv2.contourArea(
        np.array(left_line_pts).astype(np.float32))
    right_poly_area = cv2.contourArea(
        np.array(right_line_pts).astype(np.float32))

    left_conf = len(leftx) / left_poly_area
    right_conf = len(rightx) / right_poly_area
    # print(left_conf)
    # print(right_conf)

    conf_thr = 0.05
    if left_conf > conf_thr:
        left_valid = True
        left_lane.update(new_left_fitx, new_left_fit_coeff, ploty, left_conf)
    else:
        left_valid = False
        print(f'Left conf {left_conf}')
    if right_conf > conf_thr:
        right_valid = True
        right_lane.update(
            new_right_fitx, new_right_fit_coeff, ploty, right_conf)
    else:
        right_valid = False
        print(f'Right conf {right_conf}')

    return result, left_valid, right_valid
