import yaml


class Param:
    def __init__(self):
        self.param_path: str = './Utils/param.yaml'
        self.model_type: str = ''
        self.main_model_path: str = ''
        self.sub_model_path: str = ''
        self.image_width: int = 0
        self.image_height: int = 0
        self.dict_param: dict = {}
        self.get_param(self.param_path)

    def get_param(self, param_path):
        self.dict_param = yaml.load(open(param_path, 'r'), Loader=yaml.Loader)
        self.model_type = self.dict_param.get(list(self.dict_param.keys())[2])
        self.main_model_path = self.dict_param.get(list(self.dict_param.keys())[3])
        self.sub_model_path = self.dict_param.get(list(self.dict_param.keys())[4])
        self.image_width = self.dict_param.get(list(self.dict_param.keys())[5])
        self.image_height = self.dict_param.get(list(self.dict_param.keys())[6])

    def set_param(self, param_path):
        pass

