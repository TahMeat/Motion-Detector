import numpy as np
import argparse
import sys
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import dilation

from PySide2 import QtCore, QtWidgets, QtGui
from skvideo.io import vread

np.float = np.float64
np.int = np.int_

# helper
def get_closest_bbox(obj, target) -> dict:
    bbox = obj['bbox'].get(target, None)
    if bbox is None:
        # find the closest previous key
        closest_keys = [k for k in obj['bbox'].keys() if k < target]
        if closest_keys:  # If there are any keys
            closest_key = max(closest_keys)
            bbox = obj['bbox'].get(closest_key, None)
    return bbox


# QtCore UI stuff
class QtParking(QtWidgets.QWidget):
    def __init__(self, frames, motion):
        # default QtDemo implementation
        super().__init__()

        self.setWindowTitle("Object Tracker")

        self.motion_detector = motion
        self.frames = frames
        self.current_frame = 0
        self.last_processed_frame = 0

        self.button = QtWidgets.QPushButton("Next Frame")

        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        h, w, c = self.frames[0].shape
        if c == 1:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

        # Configure slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0]-1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.frame_slider)

        # Connect functions
        self.button.clicked.connect(self.on_click)
        self.frame_slider.sliderMoved.connect(self.on_move)

        # added implementations for extra buttons.
        self.jump_forward_button = QtWidgets.QPushButton("Jump Forward 60 Frames")
        self.jump_backward_button = QtWidgets.QPushButton("Jump Backward 60 Frames")
        self.layout.addWidget(self.jump_forward_button)
        self.layout.addWidget(self.jump_backward_button)

        self.jump_forward_button.clicked.connect(self.jump_forward)
        self.jump_backward_button.clicked.connect(self.jump_backward)

    # default QtDemo functions
    @QtCore.Slot()
    def on_click(self):
        if self.current_frame == self.frames.shape[0]-1:
            return
        self.update_frame(self.current_frame + 1)

    @QtCore.Slot()
    def on_move(self, pos):
        self.update_frame(pos)

    # extra functions implemented
    @QtCore.Slot()
    def jump_forward(self):
        frame = min(self.current_frame + 60, self.frames.shape[0] - 1)
        self.update_frame(frame)

    @QtCore.Slot()
    def jump_backward(self):
        frame = max(self.current_frame - 60, 0)
        self.update_frame(frame)

    # helper to reduce code, and to call motion detector
    def update_frame(self, frame_number: int):
        # calculate difference from last processed frame.
        frame_difference = abs(frame_number - self.last_processed_frame)

        # check to skip between detections
        if frame_difference >= self.motion_detector.s:
            self.current_frame = frame_number
            self.last_processed_frame = frame_number

            # push current frame into motion detect
            self.motion_detector.capture_motion(self.frames, self.current_frame)

            h, w, c = self.frames[self.current_frame].shape
            if c == 1:
                img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
            else:
                img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)

            # painter from doc.qt.io docs.
            painter = QtGui.QPainter(img)

            # draw box for each active object
            for obj in self.motion_detector.active_objects:
                if obj['is_active']:
                    # draw previous frames
                    for i in range(1, 6):
                        prev_bbox = get_closest_bbox(obj, frame_number - i)
                        if prev_bbox:
                            painter.setPen(QtGui.QPen(QtGui.Qt.red, 1, QtGui.Qt.SolidLine))
                            minr, minc, maxr, maxc = prev_bbox
                            painter.drawRect(minc, minr, maxc - minc, maxr - minr)

                    # draw current frame
                    painter.setPen(QtGui.QPen(QtGui.Qt.blue, 1, QtGui.Qt.SolidLine))
                    cur_bbox = get_closest_bbox(obj, frame_number)
                    if cur_bbox:
                        minr, minc, maxr, maxc = cur_bbox
                        painter.drawRect(minc, minr, maxc - minc, maxr - minr)

            # end painting
            painter.end()

            self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
            self.frame_slider.setValue(self.current_frame)


# detection classes
class KalmanFilter:
    def __init__(self, delta_time: float):
        self.dt = delta_time

        # mean state
        self.x = np.array([[0], [0], [0], [0]], dtype=np.float64)

        # state transition
        self.D_i = np.array([[1, 0, self.dt, 0],
                             [0, 1, 0, self.dt],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        # covariance
        self.sig_k = np.eye(self.D_i.shape[1], dtype=np.float64)

        # noise
        self.r = np.array([[0.01, 0], [0, 0.01]])

        # measurement map
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

    def predict(self):
        # predicted mean x_k, and variance sig_k
        self.x = self.D_i @ self.x
        self.sig_k = self.D_i @ self.sig_k @ self.D_i.T

    def update(self, centroid) -> np.ndarray:
        y_i = np.array(centroid).reshape(-1, 1) - (self.H @ self.x)

        S = self.H @ self.sig_k @ self.H.T + self.r
        K_i = self.sig_k @ self.H.T @ np.linalg.inv(S)

        self.x += K_i @ y_i
        I = np.eye(self.D_i.shape[1])
        self.sig_k = (I - (K_i @ self.H)) @ self.sig_k

        return self.x[:2]


class MotionDetector:
    def __init__(self, alpha: int = 3, tau: float = 0.2, delta: int = 20, s: int = 1, N: int = 40) -> None:
        self.alpha = alpha
        self.tau = tau
        self.delta = delta
        self.s = s
        self.N = N

        self.active_objects = []
        
    def merge_filters(self, kf1: KalmanFilter, kf2: KalmanFilter) -> KalmanFilter:
        # average the state estimates (centroid, velocity)
        kf1.x[:2] = (kf1.x[:2] + kf2.x[:2]) / 2

        # combine the covariance matrices (assume they are independent)
        kf1.sig_k = (kf1.sig_k + kf2.sig_k) / 2
        
        return kf1


    # create update frames
    def capture_motion(self, frames: np.ndarray, current_frame: int) -> None:
        # check if given frame is within range
        if current_frame < 2: return

        # rgb to gray
        ppframe = rgb2gray(frames[current_frame-2])
        pframe = rgb2gray(frames[current_frame-1])
        cframe = rgb2gray(frames[current_frame])

        # difference between frames
        diff1 = np.abs(cframe - pframe)
        diff2 = np.abs(pframe - ppframe)

        # threshold for motion
        motion_frame = np.maximum(diff1, diff2)
        thresh_frame = motion_frame > self.tau

        # dilate pixels, then label.
        dilated_frame = dilation(thresh_frame, np.ones((6, 6)))
        label_frame = label(dilated_frame)
        regions = regionprops(label_frame)

        # predict
        for obj in self.active_objects:
            obj['kf'].predict()
            obj['missed'] += 1

        # loop through regions to update kalman
        for region in regions:
            # check max number detected
            if len(self.active_objects) >= self.N:
                break

            # get centroid
            centroid = region.centroid
            bbox = region.bbox

            # check if object is already being tracked
            tracked_obj = self.get_filter(centroid)

            if tracked_obj:
                # track for alpha param
                y, x = tracked_obj['kf'].update(centroid).flatten()

                # set old predicted as bbox if exists
                cur_bbox = get_closest_bbox(tracked_obj, current_frame - 1)

                if cur_bbox is None:
                    cur_bbox = tracked_obj['bbox'][current_frame - 1] = bbox

                # get bbox based on center.
                w, h = cur_bbox[3] - cur_bbox[1], cur_bbox[2] - cur_bbox[0]

                minr, minc = y - (h/2), x - (w/2)
                maxr, maxc = y + (h/2), x + (w/2)

                tracked_obj['bbox'][current_frame] = (minr, minc, maxr, maxc)
                tracked_obj['missed'] = 0
            else:
                kf = KalmanFilter(delta_time=0.1)
                kf.x[:2, 0] = centroid
                self.active_objects.append({
                    'kf': kf,
                    'bbox': {current_frame: bbox},
                    'is_active': False,
                    'activity': 1,
                    'missed': 0
                })
                
        # merge objects that are close enough
        for i, obj1 in enumerate(self.active_objects):
            for j, obj2 in enumerate(self.active_objects):
                if i != j:
                    # check if the two objects are sufficiently close to each other
                    dist = np.linalg.norm(obj1['kf'].x[:2] - obj2['kf'].x[:2])
                    if dist < self.delta:  # set a threshold for merging distance
                        # merge the two Kalman filters
                        self.merge_filters(obj1['kf'], obj2['kf'])
                        # remove obj2 from the active_objects list as it has been merged
                        self.active_objects.remove(obj2)
                        break

        # remove inactive objects
        self.active_objects = [obj for obj in self.active_objects if obj['missed'] <= self.alpha + 2 and obj['activity'] > 0]

        # check for inactivity
        for idx, obj in enumerate(self.active_objects):
            # found in frame, increase activity counter
            if obj['missed'] == 0:  obj['activity'] = min(obj['activity'] + 1, self.alpha)

            # if active for # of times in a row, set active.
            if obj['activity'] >= self.alpha: obj['is_active'] = True


    def get_filter(self, centroid: np.ndarray) -> dict:
        closest_filter, min_dist = None, float('inf')
        for obj in self.active_objects:
            dist = np.linalg.norm(obj['kf'].x[:2].reshape(-1) - centroid)
            if dist < min_dist and dist < self.delta:
                closest_filter, min_dist = obj, dist
        return closest_filter



if __name__ == "__main__":
    # launch app with parameters.
    parser = argparse.ArgumentParser(description="Demo for loading video with Qt5.")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str)
    parser.add_argument("--num_frames", metavar='n', type=int, default=-1)
    parser.add_argument("--grey", metavar='True/False', type=str, default=False)
    args = parser.parse_args()

    num_frames = args.num_frames

    if num_frames > 0:
        user_frames = vread(args.video_path, num_frames=num_frames, as_grey=args.grey)
    else:
        user_frames = vread(args.video_path, as_grey=args.grey)

    app = QtWidgets.QApplication([])

    # start motion detection (change parameters here.)
    # alpha: int = 5, tau: float = 0.1, delta: int = 10, s: int = 1, N: int = 40
    md = MotionDetector()

    widget = QtParking(user_frames, md)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())