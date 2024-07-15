from python.Utils import model_format


class Processor:
    def __init__(self, param):
        self.param = param
        self.loader = model_format.ModelLoader(self.param.main_model_path, self.param.sub_model_path,
                                               self.param.model_type)
        self.image = []
        self.processing_model = None

    def selected_init(self):
        self.loader.load_model()

        def select_model(model, loader):
            match model:
                case 'pt':
                    from python.model import YoloPytorch
                    self.processing_model = YoloPytorch.YoloModel(loader)

                case 'onnx':
                    from python.model import YoloOnnx
                    self.processing_model = YoloOnnx.YoloModel(loader)
                    self.processing_model.init_model()
                case 'torch':
                    from python.model import YoloTensorrt
                case _:
                    raise "Unknown Model Input"

        select_model(self.param.model_type, self.loader)

    def run(self, image):
        self.image = image
        detect_image = self.processing_model.multi_detection(image)

        return detect_image
