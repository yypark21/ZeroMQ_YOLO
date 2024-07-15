from python.Process import py_sender, py_receiver
from python.Utils import param



def main():
    receiver = py_receiver.Receiver()
    sender = py_sender.Sender()
    parameter = param.Param()
    receiver.recv_init(parameter)
    receiver.process()
    if receiver.detect_img is not None:
        sender.send_init(receiver.detect_img)
        sender.process()
    else:
        receiver.process()


if __name__ == '__main__':
    main()
