class DictToObj:
    """
    Converting a nested dictionart to objects (to handle custom configs)
    """

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = DictToObj(v)
        self.__dict__.update(d)

    def __getitem__(self, k):
        if k in self.__dict__.keys() or not isinstance(k, int):
            return self.__dict__[k]
        return list(self.__dict__)[k]
