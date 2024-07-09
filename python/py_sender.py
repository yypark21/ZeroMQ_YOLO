import zmq
import base64, time
import cv2


class Sender:
    def __init__(self):
        self.context = None
        self.send_socket = None
        self.image = []
        self.ip = "tcp://localhost:5555"

    def send_init(self, image):
        self.context = zmq.Context()
        self.send_socket = self.context.socket(zmq.PUSH)
        self.send_socket.connect(self.ip)
        self.image = image

    def process(self):
        encoded, buffer = cv2.imencode('.jpg', self.image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = base64.b64encode(buffer)
        self.send_socket.send(data)
        self.refresh()
    def refresh(self):
        self.image = []
