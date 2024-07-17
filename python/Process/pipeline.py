from Process.py_sender import Sender
from Process.py_receiver import Receiver
from Utils.param import Param
from Utils.interfaces import SenderInterface, ReceiverInterface
import multiprocessing


class Pipeline:
    def __init__(self, logger):
        self.logger = logger
        self.parameter = Param()
        self.sender_queue = multiprocessing.Queue()
        self.receiver: ReceiverInterface = Receiver(self.logger, self.sender_queue, self.parameter)
        self.sender: SenderInterface = Sender(self.logger, self.sender_queue)

    def run(self):
        self.logger.info("Starting the pipeline.")
        self.receiver.recv_init(self.parameter)
        self.receiver.start()
        self.sender.start()

        self.receiver.thread.join()
        self.sender.thread.join()
        self.logger.info("Pipeline execution finished.")