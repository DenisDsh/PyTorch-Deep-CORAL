import numpy as np
import matplotlib.pyplot as plt
import torch


def imshow(image_tensor, mean, std, title=None):
    """
    Imshow for normalized Tensors.
    Useful to visualize data from data loader
    """

    image = image_tensor.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def accuracy(output, target):

    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    accuracy = correct/total

    return accuracy


class Tracker:

    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
