import numpy as np
import cv2


def calc_hsl_sobelx_mask(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    return s_binary, sxbinary


def calc_roi_mask(image_shape, top_left, top_right, bottom_left, bottom_right):
    roi = np.zeros(image_shape, np.uint8)
    roi = cv2.fillPoly(
        roi, np.array([[top_left, top_right, bottom_right, bottom_left]]), 1)
    return roi


def calc_tranform_matrix(image_shape,
                         src_top_left, src_top_right, src_bottom_left, src_bottom_right, lr_margin):
    # src_top_left, .. are (x,y) corrdinates of a box to be mapped in another box
    tb_margin = 100
    W = cv2.getPerspectiveTransform(np.float32([
        src_top_left,
        src_top_right,
        src_bottom_right,
        src_bottom_left]),
        np.float32([
            (lr_margin, tb_margin),
            (image_shape[1] - lr_margin, tb_margin),
            (image_shape[1] - lr_margin, image_shape[0]),
            (lr_margin, image_shape[0])])
    )
    return W


def measure_curvature_pixels(left_fit_coeff, right_fit_coeff, y_eval, mx, my):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # y_eval: y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    # Calculation of R_curve (radius of curvature)

    # # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 30/720 # meters per pixel in y dimension
    # xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Once the parabola coefficients are obtained, in pixels, convert them into meters. If the parabola is x= a*(y**2) +b*y+c; and mx and my are the scale for the x and y axis, respectively (in meters/pixel); then the scaled parabola is x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c

    denum = np.absolute(2 * left_fit_coeff[0] * mx / (my ** 2))
    if denum != 0:
        left_curverad = (
            (1 + (2 * mx / (my ** 2) * left_fit_coeff[0] * y_eval * my + (mx / my) * left_fit_coeff[1])**2)**1.5) / denum
    else:
        print('BUG: SHOULD NOT HAPPEN')
        left_curverad = 1e6

    denum = np.absolute(2 * right_fit_coeff[0] * mx / (my ** 2))
    if denum != 0:
        right_curverad = (
            (1 + (2 * mx / (my ** 2) * right_fit_coeff[0] * y_eval * my + (mx / my) * right_fit_coeff[1])**2)**1.5) / denum
    else:
        print('BUG: SHOULD NOT HAPPEN')
        right_curverad = 1e6

    return left_curverad, right_curverad


def plot_lanes_on_road(undist, warped_size, left_fitx, right_fitx, ploty, Winv):
    # Create an image to draw the lines on
    warp_zero = np.zeros(warped_size).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Winv)
    newwarp = cv2.warpPerspective(
        color_warp, Winv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    return result


def fit_poly(img_shape, x, y):
    ### Fit a second order polynomial to each with np.polyfit() ###
    fit_coeff = np.polyfit(y, x, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    try:
        fitx = fit_coeff[0] * ploty**2 + \
            fit_coeff[1] * ploty + fit_coeff[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        fitx = 1 * ploty**2 + 1 * ploty
    return fitx, fit_coeff, ploty
