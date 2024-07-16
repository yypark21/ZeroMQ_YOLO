from python.Utils.model_format import ModelLoader


class Processor:
    def __init__(self, param, logger):
        self.param = param
        self.logger = logger
        self.loader = ModelLoader(self.param.main_model_path, self.param.sub_model_path, self.param.model_type)
        self.image = []
        self.processing_model = None

    def selected_init(self):
        self.loader.load_model()

        def select_model(model, loader):
            if model == 'pt':
                from python.model.YoloPytorch import YoloModel
                self.processing_model = YoloModel(loader)
            elif model == 'onnx':
                from python.model.YoloOnnx import YoloModel
                self.processing_model = YoloModel(loader)
                self.processing_model.init_model()
            elif model == 'engine':
                from python.model.YoloTensorrt import YoloModel
                self.processing_model = YoloModel(loader)
            else:
                raise ValueError("Unknown Model Input")

        select_model(self.param.model_type, self.loader)
        self.logger.info(f"Model {self.param.model_type} selected and initialized.")

    def run(self, image):
        self.image = image
        detect_image = self.processing_model.multi_detection(image)
        self.logger.info("Image processed with the model.")
        return detect_image
