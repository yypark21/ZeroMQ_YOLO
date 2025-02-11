import zmq
import base64
import cv2
from multiprocessing import Queue
from Utils.interfaces import SenderInterface
import numpy as np
from threading import Thread


class Sender(SenderInterface):
    def __init__(self, logger, sender_queue):
        self.logger = logger
        self.context = zmq.Context()
        self.send_socket = self.context.socket(zmq.PUSH)
        self.send_socket.bind('tcp://localhost:5555')
        self.sender_queue = sender_queue
        self.thread = None

    def send_image(self, image):
        #image_bytes = image.tobytes()
        #data = base64.b64encode(image_bytes).decode('utf-8')
        #self.send_socket.send_string(data)
        #self.logger.info("Image sent.")
        self.send_socket(image.data)

    def run(self):
        self.logger.info("Sender process started.")
        try:
            while True:
                image = self.sender_queue.get()
                if image is None:
                    continue
                self.send_image(image)
        except Exception as e:
            self.logger.exception("An error occurred in the Sender process.")
        finally:
            self.logger.info("Sender process stopped.")
            self.context.term()

    def start(self):
        self.thread = Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
