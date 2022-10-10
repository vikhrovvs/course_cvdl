import numpy as np
from .base import BaseLayer


class LinearLayer(BaseLayer):
    """
    Слой, выполняющий линейное преобразование y = x @ W.T + b.
    Параметры:
        parameters[0]: W;
        parameters[1]: b;
    Линейное преобразование выполняется для последней оси тензоров, т.е.
     y[B, ..., out_features] = LinearLayer(in_features, out_feautres)(x[B, ..., in_features].)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.parameters.append(np.random.randn(out_features, in_features))
        self.parameters.append(np.random.randn(out_features))

        self.parameters_grads.append(np.zeros((out_features, in_features)))
        self.parameters_grads.append(np.zeros(out_features))

        self.input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return input @ self.parameters[0].T + self.parameters[1]

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.parameters_grads[0] += np.sum(
            (output_grad.swapaxes(-1, -2) @ self.input).reshape(-1, self.input.shape[-1] * output_grad.shape[-1]),
            axis=0) \
            .reshape(self.parameters_grads[0].shape)
        self.parameters_grads[1] += np.sum(output_grad.reshape(-1, output_grad.shape[-1]), axis=0)
        return output_grad @ self.parameters[0]
