import py_receiver
import py_sender

receiver = py_receiver.Receiver()
sender = py_sender.Sender()
model_path = "D:/inspection.onnx"
if __name__ == '__main__':
    receiver.recv_init(model_path)
    receiver.process()
    if receiver.detect_img != []:
        sender.process()
    else :
        receiver.process()

