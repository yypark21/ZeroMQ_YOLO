import zmq
import time
import queue, threading
import numpy as np
from python.Process import process


class Receiver:
    def __init__(self):
        self.context = None
        self.recv_socket = None
        self.ip = "tcp://localhost:5555"
        self.model_type = ''
        self.q = None
        self.detect_img = []
        self.width = 0
        self.height = 0
        self.select_process = None

    def recv_init(self, param):
        self.context = zmq.Context()
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.connect(self.ip)
        self.select_process = process.Processor(param)
        self.select_process.selected_init()

    def pyshine_image_queue(self, client_socket):
        q = queue.Queue(maxsize=10)

        def get_image():
            while True:
                try:
                    if self.recv_socket.poll(1000, zmq.POLLIN):
                        image_bytes = self.recv_socket.recv(zmq.NOBLOCK)
                        width = 3072
                        height = 2048
                        image = np.frombuffer(image_bytes, dtype=np.uint16).reshape((height, width))
                        q.put(image)
                    else:
                        raise 'error: message timeout'

                except Exception as e:
                    print(f"Error receiving image: {e}")

        thread = threading.Thread(target=get_image)
        thread.start()
        if thread.is_alive() is False:
            raise "Unknown Thread End Error"

        return q

    def process(self):
        if self.q is None:
            print("Start Process")
            self.q = self.pyshine_image_queue(self.recv_socket)
            img0 = []
            img0 = self.q.get()
            print("Load Image Success")
            start_time = time.time()
            self.detect_img = self.select_process.run(img0)
            print("Tact : {}".format(time.time() - start_time))
            self.height = self.detect_img.shape[0]
            self.width = self.detect_img.shape[1]
