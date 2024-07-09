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
        self.sub_output_names = None
        self.sub_input_name = None
        self.main_output_names = None
        self.main_input_name = None
        self.model_path = ''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels = ["Foreign Material", "Exception"]
        self.main_session = None
        self.sub_session = None
        self.input_name = ''
        self.output_name = []

    def init_model(self, main_model_path, sub_model_path):
        self.main_session = ort.InferenceSession(main_model_path, providers=['CUDAExecutionProvider'])
        self.sub_session = ort.InferenceSession(sub_model_path, providers=['CUDAExecutionProvider'])
        # Init Main Session(Model)
        self.main_input_name = self.main_session.get_inputs()[0].name
        self.main_output_names = [output.name for output in self.main_session.get_outputs()]
        # Init Main Session(Model)
        self.sub_input_name = self.sub_session.get_inputs()[0].name
        self.sub_output_names = [output.name for output in self.sub_session.get_outputs()]

    def preprocess(self, image, session):
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        input_shape = session.get_inputs()[0].shape[2:]
        image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
        image = image_resized.astype(np.float32)
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0) / 255.0
        return image

    def postprocess(self, session, outputs, original_shape, confidence_threshold=0.9):
        # Extract results
        results = outputs[0].squeeze()
        orig_height, orig_width = original_shape[:2]
        img_height, img_width = session.get_inputs()[0].shape[2:]

        detections = []
        num_detections = results.shape[0] // 6

        for result in results:
            i = 0
            cx = result[i * 6 + 0]
            cy = result[i * 6 + 1]
            w = result[i * 6 + 2]
            h = result[i * 6 + 3]
            confidence = result[i * 6 + 4]
            class_id = int(result[i * 6 + 5])

            if confidence >= confidence_threshold:
                width = int((w) * orig_width / img_width)
                height = int((h) * orig_height / img_height)
                x = int((cx - w / 2) * orig_width / img_width)
                y = int((cy - h / 2) * orig_height / img_height)
                detections.append((confidence, (x, y, width, height), class_id))
            i += 1
        return detections

    def draw_boxes(self, image, detections, threshold=0.9):
        result_image = image.copy()
        length = detections.shape[0]
        for i in range(length):
            for detection in detections[i]:
                box = (x1, y1, x2, y2) = (
                    detection[1][0], detection[1][1], detection[1][0] + detection[1][2],
                    detection[1][1] + detection[1][3])
                class_idx = detection[2]
                score = detection[0]
                box = np.array(box)
                if score > threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    label = f"{self.labels[class_idx]}: {score:.2f}"
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return result_image

    def inference(self, session, output_names, input_name, input_data):
        return session.run(output_names, {input_name: input_data})

    def detection(self, img, session, output_names, input_name):
        input_data = self.preprocess(img, session)
        outputs = self.inference(session, output_names, input_name, input_data)
        detections = self.postprocess(session, outputs, img.shape)
        return detections

    def multi_detection(self, img):
        # Main Inference
        main_detections = self.detection(img, self.main_session, self.main_output_names, self.main_input_name)

        # Sub Inference
        sub_detections = self.detection(img, self.sub_session, self.sub_output_names, self.sub_input_name)

        # Add Results
        main_detections = np.asanyarray(main_detections, dtype=object)
        sub_detections = np.asanyarray(sub_detections, dtype=object)
        detections = np.stack((main_detections, sub_detections), axis=0)
        result_image = self.draw_boxes(img.copy(), detections)
        cv2.imwrite("D:/yolotest.tif", result_image)

        return result_image

    def pyshine_image_queue(self, client_socket):
        q = queue.Queue(maxsize=10)

        def get_image():
            while True:
                try:
                    image_bytes = client_socket.recv()
                    width = 3072
                    height = 2048
                    image = np.frombuffer(image_bytes, dtype=np.uint16).reshape((height, width))
                    q.put(image)
                except Exception as e:
                    print(f"Error receiving image: {e}")

        thread = threading.Thread(target=get_image)
        thread.start()
        return q
