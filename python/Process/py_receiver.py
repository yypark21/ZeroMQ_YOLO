import zmq
import numpy as np
import base64
from multiprocessing import Queue
from Utils.interfaces import ReceiverInterface
from Process.process import Processor
from threading import Thread


class Receiver(ReceiverInterface):
    def __init__(self, logger, sender_queue, param):
        self.logger = logger
        self.context = zmq.Context()
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.connect("tcp://localhost:5555")
        self.sender_queue = sender_queue
        self.processor = None
        self.image_width = param.image_width
        self.image_height = param.image_height
        self.thread = None

    def recv_image(self):
        image_bytes = self.recv_socket.recv()
        image_array = np.frombuffer(image_bytes, dtype=np.uint16).reshape((self.image_height, self.image_width))

        self.logger.info(f"Image received with shape: {image_array.shape}")
        return image_array

    def process_image(self, image):
        detect_img = self.processor.run(image)
        return detect_img

    def recv_init(self, param):
        self.processor = Processor(param, self.logger)
        self.processor.selected_init()
        self.logger.info("Receiver initialized with parameters.")

    def run(self):
        self.logger.info("Receiver process started.")
        try:
            while True:
                image = self.recv_image()
                if image is not None:
                    processed_image = self.process_image(image)
                    self.sender_queue.put(processed_image)
        except Exception as e:
            self.logger.exception("An error occurred in the Receiver process.")
        finally:
            self.logger.info("Receiver process stopped.")
            self.context.term()

    def start(self):
        self.thread = Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()