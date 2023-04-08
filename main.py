import json
import os
import platform
import time

import cv2
import threading
import infer_yolov8
from pathlib import Path
import tkinter as tk
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'ultralytics') not in sys.path:
    sys.path.append(str(ROOT / 'ultralytics'))  # add yolov8 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# yolov8
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.nn.autobackend import AutoBackend


class CameraThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.source = 0  # file/dir/URL/glob, 0 for webcam
        #self.source = 'test_videos/4.mp4'
        self.cap = cv2.VideoCapture(self.source)


        self.stop_thread = False

    def run(self):
        while not self.stop_thread:
            ret, frame = self.cap.read()
            if ret:
                DetectionThread.frame = frame.copy()

    def stop(self):
        self.cap.release
        self.stop_thread = True


class DetectionThread(threading.Thread):
    frame = None
    frame_num = 0

    def __init__(self):
        threading.Thread.__init__(self)

        self.stop_thread = False
        # Load a model
        self.model = 'weights/yolov8s.pt'
        self.tracker = 'weights/osnet_x1_0_imagenet.pth'
        self.tracking_method = 'bytetrack'
        self.tracking_config = ROOT / 'trackers' / self.tracking_method / 'configs' / (self.tracking_method + '.yaml')

        self.device = 0
        self.imgsz = (640, 640)
        # Load model
        device = select_device(self.device)
        self.is_seg = '-seg' in str(self.model)
        self.model = AutoBackend(self.model, device=device, dnn=False, fp16=True)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_imgsz(self.imgsz, stride=self.stride)  # check image size
        self.tracker_list = []

        # entrance count
        self.entrance, self.records, self.center_traj = None, None, None

        with open('region_setting.json') as file:
            data = json.load(file)
        region_type = data['region_type']
        region_line1 = int(data['line1'])
        region_line2 = int(data['line2'])

        # do_entrance_counting
        self.id_set = set()
        self.interval_id_set = set()
        self.in_id_list = list()
        self.out_id_list = list()
        self.prev_center = dict()
        self.records = list()

        h, w_img, c = DetectionThread.frame.shape
        if region_type == 'both':
            self.entrance = [0, region_line1, w_img, region_line1, 0, region_line2, w_img,
                             region_line2]
        elif region_type == 'right':
            self.entrance = [0, region_line2, w_img, region_line2]
        elif region_type == 'left':
            self.entrance = [0, region_line1, w_img, region_line1]
        elif region_type == 'close':
            self.entrance = [0, 0, 0, 0, 0, 0, 0, 0]
        else:
            raise ValueError("region_type:{} unsupported.".format(
                region_type))
        # time and FPS
        self.start_time = time.time()
        self.frames = 0

    def run(self):
        while not self.stop_thread:
            start_runtime = time.time()
            if DetectionThread.frame is not None:
                frame = infer_yolov8.run(source=DetectionThread.frame, yolo_weights=self.model,
                                         reid_weights=self.tracker,
                                         tracking_method=self.tracking_method,
                                         tracking_config=self.tracking_config,
                                         exp_name='yolov8_infer',
                                         imgsz=self.imgsz,
                                         is_seg=self.is_seg,
                                         model=self.model,
                                         stride=self.stride,
                                         names=self.names,
                                         pt=self.pt,
                                         tracker_list=self.tracker_list,
                                         entrance=self.entrance,
                                         id_set=self.id_set,
                                         interval_id_set=self.interval_id_set,
                                         in_id_list=self.in_id_list,
                                         out_id_list=self.out_id_list,
                                         prev_center=self.prev_center,
                                         records=self.records,
                                         seen=self.frames
                                         )

                if platform.system() == 'Linux':  # allow window resize (Linux)
                    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow('Detection', frame.shape[1], frame.shape[0])

                # add FPS
                self.frames += 1
                elapsed_time = time.time() - start_runtime
                fps = 1 / elapsed_time

                cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

                # display frame
                cv2.imshow('Detection', frame)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    stop_program()
                    break
                DetectionThread.frame = None
        return

    def stop(self):
        self.stop_thread = True
        cv2.destroyAllWindows()
        # save predicted video
        print(f'Running time: {time.time() - self.start_time}; Detected frames: {self.frames};')

        # exit the main process
        print('Exiting...')
        os._exit(os.EX_OK)


def start_program():
    global camera_thread
    global detection_thread
    # Code to start the program goes here
    camera_thread = CameraThread()
    camera_thread.start()

    detection_thread = DetectionThread()
    detection_thread.start()

    camera_thread.join()
    detection_thread.join()
    pass


def stop_program():
    # Code to stop the program goes here
    camera_thread.stop()
    detection_thread.stop()
    print("STOP programs")
    pass


if __name__ == '__main__':
    # Create a new window
    window = tk.Tk()

    # Add a label
    label = tk.Label(window, text="Welcome to my program!", font=("Arial Bold", 20))
    label.pack(pady=10)

    # Add the start button
    start_button = tk.Button(window, text="Start Program", command=start_program)
    start_button.pack(pady=5)

    # Add the stop button
    stop_button = tk.Button(window, text="Stop Program", command=stop_program)
    stop_button.pack(pady=5)

    # Start the GUI main loop
    window.mainloop()
