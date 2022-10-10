import numpy as np
from .base import BaseLayer


class MaxPoolLayer(BaseLayer):
    """
    Слой, выполняющий 2D Max Pooling, т.е. выбор максимального значения в окне.
    y[B, c, h, w] = Max[i, j] (x[B, c, h+i, w+j])

    У слоя нет обучаемых параметров.
    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.

    В качестве значений padding используется -np.inf, т.е. добавленные pad-значения используются исключительно
     для корректности индексов в любом положении, и никогда не могут реально оказаться максимумом в
     своем пуле.
    Гарантируется, что значения padding, stride и kernel_size будут такие, что
     в input + padding поместится целое число kernel, т.е.:
     (D + 2*padding - kernel_size)  % stride  == 0, где D - размерность H или W.

    Пример корректных значений:
    - kernel_size = 3
    - padding = 1
    - stride = 2
    - D = 7
    Результат:
    (Pool[-1:2], Pool[1:4], Pool[3:6], Pool[5:(7+1)])
    """

    def __init__(self, kernel_size: int, stride: int, padding: int):
        assert (kernel_size % 2 == 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super().__init__()

    @staticmethod
    def _pad_neg_inf(tensor, one_size_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        pad = [(0, 0)] * tensor.ndim
        for ax in axis:
            pad[ax] = (one_size_pad, one_size_pad)
        return np.pad(tensor, pad, 'constant', constant_values=-np.inf)

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride == 0

        padded = self._pad_neg_inf(input, self.padding)
        output_h = (input.shape[-2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_w = (input.shape[-1] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = np.zeros((padded.shape[0], padded.shape[1], output_h, output_w))

        for i in range(output.shape[-2]):
            for j in range(output.shape[-1]):
                output[:, :, i, j] = padded[:, :,
                                     i * self.stride:i * self.stride + self.kernel_size,
                                     j * self.stride:j * self.stride + self.kernel_size
                                     ].max(axis=(-2, -1))
        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        pass
