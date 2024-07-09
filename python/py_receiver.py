import zmq
import Yolo
import pyshine as ps
import cv2


class Receiver:
    def __init__(self):
        self.context = None
        self.recv_socket = None
        self.ip = "tcp://localhost:5555"
        self.yolo = Yolo.YoloModel()
        self.q = None
        self.detect_img = []
        self.width = 0
        self.height = 0

    def recv_init(self, main_model_path, sub_model_path):
        self.context = zmq.Context()
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.connect(self.ip)
        self.yolo.init_model(main_model_path, sub_model_path)

    def process(self):
        if self.q is None:
            print("Start Process")
            self.q = self.yolo.pyshine_image_queue(self.recv_socket)
            img0 = []
            img0 = self.q.get()
            print("Load Image Success")
            self.detect_img = self.yolo.multi_detection(img0)
            self.height = self.detect_img.shape[0]
            self.width = self.detect_img.shape[1]

