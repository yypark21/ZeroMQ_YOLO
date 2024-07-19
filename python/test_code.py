import zmq
import numpy as np

import cv2
import base64

import socket

if __name__ == '__main__':
    ip = "tcp://*:5555"
    context = zmq.Context()
    send_socket = context.socket(zmq.PUSH)
    send_socket.bind(ip)

    # sender = imagezmq.ImageSender(connect_to=ip)

    # sender_name = socket.gethostname() # send your hostname with each image

    # image = open("../image/F 1_1.jpg", 'rb')

    # while 1 :
    #    sender.send_image(sender_name, image)

    # while 1 :
    #     f = open("../image/F 1_1.jpg", 'rb')
    #     bytes = bytearray(f.read())
    #     strng = base64.b64encode(bytes)
    #     send_socket.send(strng)
    #     f.close()

    while 1:
        image = cv2.imread("../image/F 1_1.tif", -1)
        send_socket.send(image.data)