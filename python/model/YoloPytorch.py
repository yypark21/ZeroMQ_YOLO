import cv2
import numpy as np
import time
import torch


class YoloModel:
    def __init__(self, model_loader):
        self.model = model_loader
        self.labels = ["Foreign Material", "Exception"]
        self.detections = []

        # Load the model and move to the appropriate device
        self.main_model = self.model.main_model
        self.sub_model = self.model.sub_model

    @staticmethod
    def preprocess(image, input_shape):
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
        image = image_resized.astype(np.float32)
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0) / 255.0
        return torch.from_numpy(image).to('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def postprocess(outputs, original_shape, confidence_threshold=0.9):
        results = outputs.squeeze().cpu().numpy()
        orig_height, orig_width = original_shape[:2]

        detections = []
        num_detections = results.shape[0] // 6
        idx = np.where(results[:, 4] > confidence_threshold)
        results = results[idx]
        for result in results:
            cx, cy, w, h, confidence, class_id = result[:6]
            width = int(w * orig_width)
            height = int(h * orig_height)
            x = int((cx - w / 2) * orig_width)
            y = int((cy - h / 2) * orig_height)
            detections.append((confidence, (x, y, width, height), int(class_id)))
        return detections

    def draw_boxes(self, image, detections, threshold=0.9):
        result_image = (image.copy() / 256).astype(np.uint8)

        for detection in detections:
            box = (x1, y1, x2, y2) = (
                detection[1][0], detection[1][1], detection[1][0] + detection[1][2], detection[1][1] + detection[1][3])
            class_idx = detection[2]
            score = detection[0]
            box = np.array(box)
            if score > threshold:
                x1, y1, x2, y2 = box.astype(int)
                label = f"{self.labels[class_idx]}: {score:.2f}"
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return result_image

    def inference(self, model, input_data):
        with torch.no_grad():
            outputs = model(input_data)
        return outputs

    def detection(self, img, model):
        input_shape = model.get_inputs()[0].shape[2:]
        input_data = self.preprocess(img, input_shape)
        outputs = self.inference(model, input_data)
        detections = self.postprocess(outputs, img.shape)
        return detections

    def multi_detection(self, img):
        # Main Inference
        st = time.time()
        main_detections = self.detection(img, self.main_model)
        print("main detect Tact : {}".format(time.time() - st))

        # Sub Inference
        st = time.time()
        sub_detections = self.detection(img, self.sub_model)
        print("sub detect Tact : {}".format(time.time() - st))

        # Add Results
        main_detections = np.asarray(main_detections, dtype=object)
        sub_detections = np.asarray(sub_detections, dtype=object)
        if main_detections.shape[0] == 0:
            self.detections = sub_detections
        elif sub_detections.shape[0] == 0:
            self.detections = main_detections
        else:
            self.detections = np.stack((main_detections, sub_detections), axis=0)

        st = time.time()
        result_image = self.draw_boxes(img.copy(), self.detections)
        print("vis time : {}".format(time.time() - st))
        cv2.imwrite("D:/yolotest.tif", result_image)

        return result_image
