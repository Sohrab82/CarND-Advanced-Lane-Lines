import pickle
import cv2
import numpy as np
import glob


class CameraCalibration():
    @staticmethod
    def load_from_pickle(file_name):
        dist_pickle = pickle.load(open(file_name, "rb"))
        objpoints = dist_pickle["objpoints"]
        imgpoints = dist_pickle["imgpoints"]
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        return mtx, dist, objpoints, imgpoints

    @staticmethod
    def save_to_pickle(file_name, mtx, dist, object_points, image_points):
        dist_pickle = {}
        dist_pickle['objpoints'] = object_points
        dist_pickle['imgpoints'] = image_points
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        pickle.dump(dist_pickle, open(file_name, 'wb'))

    @staticmethod
    def calc_camera_calibration_params(image_folder, nx, ny, plot_it=False):
        # this MUST be float32 and not float
        ref_object_points = np.zeros((nx * ny, 3), np.float32)
        ref_object_points[:, :2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)

        image_points = []
        object_points = []
        for image_file in glob.glob(image_folder + '/*.jpg'):
            image = cv2.imread(image_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if not ret:
                continue
            if plot_it:
                cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
                cv2.imshow('img', image)
                cv2.waitKey(500)
            image_points.append(corners)
            object_points.append(ref_object_points)
        if plot_it:
            cv2.destroyAllWindows()
        print(object_points[0].shape)
        print(image_points[0].shape)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image.shape[1::-1], None, None)

        return ret, mtx, dist, rvecs, tvecs, object_points, image_points
