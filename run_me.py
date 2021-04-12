import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.camera_calibration import CameraCalibration
from utils.plotting import plot_both, plot_four
from utils.misc import calc_hsl_sobelx_mask, calc_roi_mask, calc_tranform_matrix, measure_curvature_pixels, plot_lanes_on_road
from utils.sliding_window import find_lane_from_histo
from utils.search_around import search_around_poly
from utils.lane import Lane


# calibrate the camera
# image_folder = 'camera_cal'
# nx = 9
# ny = 6
# ret, mtx, dist, rvecs, tvecs, object_points, image_points = CameraCalibration.calc_camera_calibration_params(
#     image_folder, nx, ny, False)

# # save the calibration params to file
# CameraCalibration.save_to_pickle(
#     './output_images/my_wide_dist_pickle.p', mtx, dist, object_points, image_points)

# load calibration params from file
mtx, dist, objpoints, imgpoints = CameraCalibration.load_from_pickle(
    './output_images/my_wide_dist_pickle.p')

W = calc_tranform_matrix((720, 1280),
                         src_top_left=(535, 495),
                         src_top_right=(755, 495),
                         src_bottom_left=(250, 690),
                         src_bottom_right=(1060, 690), lr_margin=200)
Winv = np.linalg.inv(W)

# test the transform matrix
test_image = cv2.imread('test_images/straight_lines1.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
transform_image = cv2.warpPerspective(
    test_image, W, test_image.shape[1::-1], flags=cv2.INTER_LINEAR)
plot_both(test_image, transform_image, 'Original Image',
          'Warped image', True, True, './output_images/warped_straight_lines.jpg')

# use camera calibration to undistort a test image and plot both
cal_test_image = cv2.imread('test_images/calibration_test.jpg')
undist_cal_test_image = cv2.undistort(cal_test_image, mtx, dist, None, mtx)
plot_both(cal_test_image, undist_cal_test_image, 'Original Image',
          'Undistorted Image',
          True,
          True,
          './output_images/undistorted.jpg')

right_lane = Lane()
left_lane = Lane()

# number of allowed failures in detection in consecuitve frames
CONSEC_FAILED_DET_THRS = 10
num_consec_failed_detctn_l = CONSEC_FAILED_DET_THRS + 1
num_consec_failed_detctn_r = CONSEC_FAILED_DET_THRS + 1


def process_image(test_image, show_plots=False, save_plots=False, fname_prefix=''):
    undist_test_image = cv2.undistort(test_image, mtx, dist, None, mtx)
    # plot undistorted/warped image
    plot_both(test_image, undist_test_image, 'Original Image',
              'Undistorted Image',
              show_plot=show_plots,
              save_plot=save_plots,
              output_fname=f'./output_images/{fname_prefix}_0_transformed.jpg')

    # apply HLS thresholding & sobel x filtering on the image
    hls_mask, sobx_mask = calc_hsl_sobelx_mask(undist_test_image)
    hls_sobel_mask = hls_mask | sobx_mask

    # apply ROI mask
    roi_mask = calc_roi_mask(undist_test_image.shape[:-1],
                             top_left=(510, 440), top_right=(770, 440), bottom_left=(100, 719), bottom_right=(1180, 719))
    lanes_mask = hls_sobel_mask & roi_mask

    # plot sobel+hsl+roi filtered
    hls_sobel_masks_rgb = np.dstack(
        (np.zeros_like(hls_mask), hls_mask, sobx_mask)) * 255
    plot_four(test_image,
              hls_sobel_masks_rgb,
              roi_mask * 255,
              lanes_mask * 255,
              'Original Image',
              'HLS & Sobel Masks',
              'ROI mask',
              'Filtered image',
              show_plot=show_plots,
              save_plot=save_plots,
              output_fname=f'./output_images/{fname_prefix}_1_filtered.jpg')

    # apply perspective transform
    lanes_birdeye = cv2.warpPerspective(
        lanes_mask, W, lanes_mask.shape[::-1], flags=cv2.INTER_LINEAR)
    plot_both(test_image, lanes_birdeye,
              'Original Image', 'Birdeye view of ROI',
              show_plot=show_plots,
              save_plot=save_plots,
              output_fname=f'./output_images/{fname_prefix}_2_birdeye.jpg')

    global num_consec_failed_detctn_l
    global num_consec_failed_detctn_r

    if num_consec_failed_detctn_l > CONSEC_FAILED_DET_THRS:
        num_consec_failed_detctn_l = 0
        left_from_histo = True
    else:
        left_from_histo = False

    if num_consec_failed_detctn_r > CONSEC_FAILED_DET_THRS:
        num_consec_failed_detctn_r = 0
        right_from_histo = True
    else:
        right_from_histo = False

    if right_from_histo or left_from_histo:
        out_img = find_lane_from_histo(
            lanes_birdeye,
            nwindows=9,
            margin=50,
            minpix=20,
            left_lane=left_lane,
            right_lane=right_lane,
            update_left_lane=left_from_histo,
            update_right_lane=right_from_histo)
        fig = plot_both(test_image, out_img, 'Original Image', 'Lanes detected',
                        show_plot=show_plots,
                        save_plot=save_plots,
                        output_fname='')
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_lane.fitx, left_lane.ploty, color='yellow')
        plt.plot(right_lane.fitx, right_lane.ploty, color='yellow')
        if show_plots:
            plt.show()
        if save_plots:
            fig.savefig(
                f'./output_images/{fname_prefix}_3-hist_detected.jpg', dpi=fig.dpi)
    else:
        # use search around previously detected polylines
        out_img, left_valid, right_valid = search_around_poly(
            lanes_birdeye,
            margin=25,
            left_lane=left_lane,
            right_lane=right_lane)
        if not left_valid:
            num_consec_failed_detctn_l += 1
            print(f'Invalid left {num_consec_failed_detctn_l}')
        else:
            num_consec_failed_detctn_l = 0
        if not right_valid:
            num_consec_failed_detctn_r += 1
            print(f'Invalid right {num_consec_failed_detctn_r}')
        else:
            num_consec_failed_detctn_r = 0

        fig = plot_both(test_image, out_img, 'Original Image',
                        'Search around poly',
                        show_plot=show_plots,
                        save_plot=save_plots,
                        output_fname='')
        # Plot the polynomial lines onto the image
        plt.plot(left_lane.fitx, left_lane.ploty, color='yellow')
        plt.plot(right_lane.fitx, right_lane.ploty, color='yellow')
        if show_plots:
            plt.show()
        if save_plots:
            fig.savefig(
                f'./output_images/{fname_prefix}_3_search_around.jpg', dpi=fig.dpi)

    result = plot_lanes_on_road(
        undist_test_image,
        lanes_birdeye.shape,
        left_lane.fitx, right_lane.fitx,
        left_lane.ploty,
        Winv)

    # coordinates of the bottom of the lanes in pixels
    left_px0 = left_lane.fitx[-1]
    right_px0 = right_lane.fitx[-1]

    # mx and my are the scale for the x and y axis, respectively (in meters/pixel);
    mx = 3.7 / (right_px0 - left_px0)
    my = 3.0 / 150
    # middle of the lane in pixel
    x_center_p0 = (right_px0 + left_px0) / 2.
    # distance of center of the lane to the camira mounting point in pixels
    d_center_p0 = lanes_birdeye.shape[1] / 2. - x_center_p0
    # distance of center of the lane to the camira mounting point in meters
    d_center_m = d_center_p0 * mx

    # lanes curvature measurement
    left_curverad, right_curverad = measure_curvature_pixels(
        left_lane.fit_coeff,
        right_lane.fit_coeff,
        np.max(left_lane.ploty),
        mx,
        my)
    cv2.putText(result, f'Curvature left: {int(left_curverad)} (m), right:{int(right_curverad)} (m)', (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(result, f'Vehicle at {-int(d_center_m*100)} (cm) from center', (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if show_plots:
        plt.imshow(result)
    if save_plots:
        plt.imsave(f'./output_images/{fname_prefix}_4_output.jpg', result)
    plt.close('all')
    return result


# run process_image for test images
# test_image = cv2.imread('test_images/test4.jpg')
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# process_image(test_image)

video_output = './output_images/harder_challenge_video_output.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

# clip1 = VideoFileClip("project_video.mp4").subclip(30, 35)
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(video_output, audio=False)

cap_in = cv2.VideoCapture('project_video.mp4')
# Check if camera opened successfully
if (cap_in.isOpened() == False):
    print("Error opening video stream or file")

fps = cap_in.get(cv2.CAP_PROP_FPS)

cap_out = cv2.VideoWriter(video_output, -1, fps, (1280, 720))
cap_out.set(cv2.CAP_PROP_FPS, fps)

# Read until video is completed
frame_no = 0
# start from this frame
start_frame = 0
# terminate at this frame
end_frame = 1e6 * fps

while(cap_in.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap_in.read()
    if frame_no < start_frame:
        frame_no += 1
        continue
    if frame_no > end_frame:
        break
    if not ret:
        break
    print(f'Frame no: {frame_no}')
    out_frame = process_image(frame,
                              show_plots=False,
                              save_plots=True,
                              fname_prefix=str(frame_no))
    cap_out.write(out_frame)
    frame_no += 1
    break
cap_in.release()
cap_out.release()
