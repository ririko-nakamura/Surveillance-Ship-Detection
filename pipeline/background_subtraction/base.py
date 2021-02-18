from abc import ABC, abstractmethod

class BackgroundSubtractorBase(ABC):

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def apply(self, img):
        pass