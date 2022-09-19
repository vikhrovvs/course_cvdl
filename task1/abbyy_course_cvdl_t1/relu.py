import numpy as np
from .base import BaseLayer


class ReluLayer(BaseLayer):
    """
    Слой, выполняющий Relu активацию y = max(x, 0).
    Не имеет параметров.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.mask = (input>0).astype(float)
        output = input * self.mask
        return output
#         return np.maximum(input, 0)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        # y = x * Mask
        output = output_grad * self.mask
        self.mask = None
        return output
