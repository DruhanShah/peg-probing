class Config:
    """
    The class that dict_to_obj uses as a custom config class
    """

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = Config(v)
        self.__dict__.update(d)

    def __getitem__(self, k):
        if k in self.__dict__.keys() or not isinstance(k, int):
            return self.__dict__[k]
        return list(self.__dict__)[k]


def dict_to_obj(dictionary):
    """
    Convert (nested) dictionary to object, usually to create custom configs
    """
    
    return Config(dictionary)


def obj_to_dict(obj):
    """
    Convert object to dictionary, usually to unpack configs for Python use
    """

    if isinstance(obj, dict):
        return {k: obj_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: obj_to_dict(v) for k, v in obj.__dict__.items()}
    elif hasattr(obj, "__iter__"):
        return [obj_to_dict(v) for v in obj]
    else:
        return obj
