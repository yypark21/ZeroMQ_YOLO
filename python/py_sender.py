import zmq
import base64, time
import cv2
import py_receiver


class Sender:
    def __init__(self):
        self.context = None
        self.send_socket = None
        self.image = []
        self.recv = py_receiver.Receiver()

    def process(self):
        # self.image = cv2.resize(self.recv.detect_img, self.recv.height, self.recv.width, cv2.INTER_CUBIC)
        encoded, buffer = cv2.imencode('.jpg', self.image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = base64.b64encode(buffer)
        self.send_socket.send_pyobj(data)
        self.refresh()
    def refresh(self):
        self.image = []
