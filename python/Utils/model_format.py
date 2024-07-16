import torch
import onnxruntime as ort
import tensorrt as trt


class ModelLoader:
    def __init__(self, model_path, sub_path, model_type):
        self.main_model_path = model_path
        self.sub_model_path = sub_path
        self.model_type = model_type
        if self.model_type == 'pt':
            self.main_model = None
            self.sub_model = None
            self.device = None
        elif self.model_type == 'onnx':
            self.main_session = None
            self.sub_session = None
            self.main_input_name = ''
            self.main_output_names = []
            self.sub_input_name = ''
            self.sub_output_names = []
        elif self.model_type == 'engine':
            self.main_engine = None
            self.sub_engine = None
        else:
            raise ValueError("Unselected model type Choose from 'pt', 'onnx', or 'engine'.")

    def load_model(self):
        if self.model_type == 'pt':
            return self.load_pt_model()
        elif self.model_type == 'onnx':
            return self.load_onnx_model()
        elif self.model_type == 'engine':
            return self.load_engine_model()
        else:
            raise ValueError("Unsupported model type. Choose from 'pt', 'onnx', or 'engine'.")

    def load_pt_model(self):
        import sys
        sys.path.append('C:/yolov5-master')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.main_model = torch.load(self.main_model_path, map_location=self.device)
        if self.sub_model_path:
            self.sub_model = torch.load(self.sub_model_path, map_location=self.device)
        else:
            self.sub_model = None
        self.main_model.eval()
        if self.sub_model:
            self.sub_model.eval()
        return self.main_model, self.sub_model

    def load_onnx_model(self):
        self.main_session = ort.InferenceSession(self.main_model_path)
        if self.sub_model_path:
            self.sub_session = ort.InferenceSession(self.sub_model_path)
        else:
            self.sub_session = None
        # Init Main Session(Model)
        self.main_input_name = self.main_session.get_inputs()[0].name
        self.main_output_names = [output.name for output in self.main_session.get_outputs()]
        # Init Sub Session(Model)
        if self.sub_session:
            self.sub_input_name = self.sub_session.get_inputs()[0].name
            self.sub_output_names = [output.name for output in self.sub_session.get_outputs()]
        return self.main_session, self.sub_session

    def load_engine_model(self):
        main_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.main_model_path, 'rb') as f, trt.Runtime(main_logger) as runtime:
            self.main_engine = runtime.deserialize_cuda_engine(f.read())

        if self.sub_model_path:
            sub_logger = trt.Logger(trt.Logger.WARNING)
            with open(self.sub_model_path, 'rb') as f, trt.Runtime(sub_logger) as runtime:
                self.sub_engine = runtime.deserialize_cuda_engine(f.read())
        else:
            self.sub_engine = None

        return self.main_engine, self.sub_engine
