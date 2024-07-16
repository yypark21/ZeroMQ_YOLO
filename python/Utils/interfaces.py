from abc import ABC, abstractmethod
from multiprocessing import Queue


class SenderInterface(ABC):
    @abstractmethod
    def send_image(self, image):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def start(self):
        pass


class ReceiverInterface(ABC):
    @abstractmethod
    def recv_init(self, param):
        pass

    @abstractmethod
    def run(self, sender_queue: Queue):
        pass

    @abstractmethod
    def start(self):
        pass
