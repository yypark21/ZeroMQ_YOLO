import torch
import onnxruntime as ort
import tensorrt as trt


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2)
        self.fc1 = torch.nn.Linear(32 * 6 * 6, 100)
        self.fc2 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.main_model = torch.load(self.main_model_path,map_location=self.device)
        self.sub_model = torch.load(self.sub_model_path,map_location=self.device)
        self.main_model.eval()
        self.sub_model.eval()

        return self.main_model, self.sub_model

    def load_onnx_model(self):
        self.main_session = ort.InferenceSession(self.main_model_path)
        self.sub_session = ort.InferenceSession(self.sub_model_path)
        # Init Main Session(Model)
        self.main_input_name = self.main_session.get_inputs()[0].name
        self.main_output_names = [output.name for output in self.main_session.get_outputs()]
        # Init Main Session(Model)
        self.sub_input_name = self.sub_session.get_inputs()[0].name
        self.sub_output_names = [output.name for output in self.sub_session.get_outputs()]
        return self.main_session, self.sub_session

    def load_engine_model(self):
        main_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.main_model_path, 'rb') as f, trt.Runtime(main_logger) as runtime:
            self.main_engine = runtime.deserialize_cuda_engine(f.read())

        sub_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.sub_model_path, 'rb') as f, trt.Runtime(sub_logger) as runtime:
            self.sub_engine = runtime.deserialize_cuda_engine(f.read())

        return self.main_engine, self.sub_engine
