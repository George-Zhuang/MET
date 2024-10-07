from .random_model import RandomModel
from .vila_model import VILAModel

def get_model(model_name, **kwargs):
    if model_name == "random":
        return RandomModel(**kwargs)
    elif model_name == 'vila':
        return VILAModel(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not recognized")