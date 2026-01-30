"""
Base class for transforms
"""


class Transform:
    def __init__(self):
        pass

    def transform(self, x):
        raise NotImplementedError()

    def inverse_transform(self, x, s=None):
        raise NotImplementedError()
    
    def adjoint_transform(self, x, s=None):
        raise NotImplementedError()
