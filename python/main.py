import py_receiver
import py_sender

receiver = py_receiver.Receiver()
sender = py_sender.Sender()
main_model_path = "D:/inspection.onnx"
sub_model_path = "D:/inspection_sub.onnx"

if __name__ == '__main__':
    receiver.recv_init(main_model_path, sub_model_path)
    receiver.process()
    if receiver.detect_img:
        sender.process()
    else:
        receiver.process()
