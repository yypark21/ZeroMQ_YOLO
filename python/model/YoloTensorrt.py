import cv2
import numpy as np
import time
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class YoloModel:
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.labels = ["Foreign Material", "Exception"]
        self.detections = []

        # Load the model and move to the appropriate device
        self.main_engine = self.model_loader.main_engine
        self.sub_engine = self.model_loader.sub_engine

        # Create context and allocate buffers for main and sub engines
        self.main_context, self.main_buffers = self.allocate_buffers(self.main_engine)
        self.sub_context, self.sub_buffers = self.allocate_buffers(self.sub_engine)

        # Warm-up the GPU
        self.warm_up(self.main_context, self.main_buffers)
        self.warm_up(self.sub_context, self.sub_buffers)

    @staticmethod
    def allocate_buffers(engine):
        context = engine.create_execution_context()
        bindings = [None] * engine.num_bindings
        inputs, outputs = [], []
        for i in range(engine.num_bindings):
            binding_shape = context.get_binding_shape(i)
            size = trt.volume(binding_shape) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[i] = int(device_mem)
            if engine.binding_is_input(i):
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))
        stream = cuda.Stream()
        return context, (inputs, outputs, bindings, stream)

    def preprocess(self, image, input_shape):
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
        image = image_resized.astype(np.float32)
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0) / 255.0
        return image

    def postprocess(self, outputs, original_shape, confidence_threshold=0.9):
        results = outputs.squeeze().reshape(-1, 6)
        orig_height, orig_width = original_shape[:2]
        detections = []
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

    def inference(self, context, bindings, inputs, outputs, stream):
        [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
        stream.synchronize()

    def warm_up(self, context, buffers, iterations=10):
        inputs, outputs, bindings, stream = buffers
        dummy_input = np.random.random(inputs[0][0].shape).astype(np.float32)
        np.copyto(inputs[0][0], dummy_input.ravel())
        for _ in range(iterations):
            self.inference(context, bindings, inputs, outputs, stream)

    def detection(self, img, context, buffers):
        input_shape = (1, 3, 224, 224)
        inputs, outputs, bindings, stream = buffers
        input_data = self.preprocess(img, input_shape)
        np.copyto(inputs[0][0], input_data.ravel())
        self.inference(context, bindings, inputs, outputs, stream)
        detections = self.postprocess(outputs[0][0], img.shape)
        return detections

    def multi_detection(self, img):
        # Main Inference
        st = time.time()
        main_detections = self.detection(img, self.main_context, self.main_buffers)
        print("main detect Tact : {}".format(time.time() - st))

        # Sub Inference
        st = time.time()
        sub_detections = self.detection(img, self.sub_context, self.sub_buffers)
        print("sub detect Tact : {}".format(time.time() - st))

        # Add Results
        main_detections = np.asarray(main_detections, dtype=object)
        sub_detections = np.asarray(sub_detections, dtype=object)
        if main_detections.shape[0] == 0:
            self.detections = sub_detections
        elif sub_detections.shape[0] == 0:
            self.detections = main_detections
        else:
            self.detections = np.concatenate((main_detections, sub_detections), axis=0)

        st = time.time()
        result_image = self.draw_boxes(img.copy(), self.detections)
        print("vis time : {}".format(time.time() - st))
        cv2.imwrite("D:/yolotest.tif", result_image)

        return result_image
