from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

class eda(metaclass=ABCMeta):
    """
    Abstract base class for exploratory data analysis
    """
    def __init__(self, name):
        self.name = name
        
        @abstractmethod
        def plot_distribution(self):
            return NotImplementedError