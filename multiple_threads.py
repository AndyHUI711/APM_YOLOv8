import cv2
import threading

class CameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            cv2.imshow('Camera',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

class DetectionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # detection program implementation
        pass

if __name__ == '__main__':
    camera_thread = CameraThread()
    detection_thread = DetectionThread()

    camera_thread.start()
    detection_thread.start()

    camera_thread.join()
    detection_thread.join()
