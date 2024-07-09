import cv2
import onnx
import onnxruntime as ort
import zmq
import base64
import numpy as np, time
import pyshine as ps
from multiprocessing import Process
import torch, random, threading, queue
import matplotlib.pyplot as plt
import os

import torch.nn as nn
import torch.nn.init as init
from torch import nn


class YoloModel:
    def __init__(self):
        self.model_path = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels = ["Foreign Material", "Exception"]
        self.session = None
        self.input_name = ''
        self.output_name = []

    def init_model(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, image):
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        input_shape = self.session.get_inputs()[0].shape[2:]
        image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
        image = image_resized.astype(np.float32)
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0) / 255.0
        return image

    def postprocess(self, outputs, original_shape, confidence_threshold=0.5):
        # Extract results
        results = outputs[0].squeeze()  # Assuming the model output is of shape (1, N, 6)

        orig_height, orig_width = original_shape[:2]
        img_height, img_width = self.session.get_inputs()[0].shape[2:]

        detections = []
        num_detections = results.shape[0] // 6

        for result in (results):
            i = 0
            right = result[i * 6 + 0]
            bottom = result[i * 6 + 1]
            left = result[i * 6 + 2]
            top = result[i * 6 + 3]
            confidence = result[i * 6 + 4]
            class_id = int(result[i * 6 + 5])

            if confidence >= confidence_threshold:
                x = int(left * orig_width / img_width)
                y = int(top * orig_height / img_height)
                width = int((right -left) * orig_width / img_width)
                height = int((bottom - top) * orig_height / img_height)

                detections.append((confidence, (x, y, width, height), class_id))
            i +=1
        return detections
    def draw_boxes(self, image, detections, threshold=0.9):
        global rt_img
        for detection in detections :
            box = (x1,y1,x2,y2) = (detection[1][0],detection[1][1], detection[1][0] + detection[1][2], detection[1][2] + detection[1][3])
            class_idx = detection[2]
            score = detection[0]
            box = np.array(box)
            rt_img = (image.copy() / 256).astype(np.uint8)
            rt_img = cv2.cvtColor(rt_img, cv2.COLOR_GRAY2RGB)

            if score > threshold:
                x1, y1, x2, y2 = box.astype(int)
                label = f"{self.labels[class_idx]}: {score:.2f}"
                cv2.rectangle(rt_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(rt_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image , rt_img

    def detection(self, img):
        input_data = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        detections = self.postprocess(outputs, img.shape)
        raw_image,result_image = self.draw_boxes(img.copy(), detections)
        cv2.imwrite("D:/yoloraw.tif", raw_image)
        cv2.imwrite("D:/yolotest.tif", result_image)

        return result_image
    def pyshine_image_queue(self, client_socket):
        frame = [0]
        q = queue.Queue(maxsize=10)
        def getAudio():
            while True:
                try:
                    image_bytes = client_socket.recv()
                    width = 3072
                    height = 2048
                    image = np.frombuffer(image_bytes, dtype=np.uint16).reshape((height, width))
                    # source = cv2.imdecode(image, 1)
                    q.put(image)
                except:
                    pass

        thread = threading.Thread(target=getAudio, args=())
        thread.start()
        return q
