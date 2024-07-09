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

    def postprocess(self, outputs, original_shape):
        print("Outputs:", [output.shape for output in outputs])

        if len(outputs) == 3:
            boxes, scores, class_indices = outputs[0], outputs[1], outputs[2]
        elif len(outputs) == 2:
            boxes, scores_and_classes = outputs[0], outputs[1]
            scores = scores_and_classes[:, :, 4]
            class_indices = scores_and_classes[:, :, 5].astype(int)
        else:
            raise ValueError("Unexpected output format from the model")

        height, width = original_shape[:2]
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        return boxes, scores, class_indices
        return boxes, scores, class_indices

    def draw_boxes(self, image, boxes, scores, class_indices, threshold=0.5):
        for box, score, class_idx in zip(boxes, scores, class_indices):
            if score > threshold:
                x1, y1, x2, y2 = box.astype(int)
                label = f"{self.labels[class_idx]}: {score:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image

    def detection(self, img):
        input_data = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        boxes, scores, class_indices = self.postprocess(outputs, img.shape)
        result_image = self.draw_boxes(img.copy(), boxes, scores, class_indices)
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
