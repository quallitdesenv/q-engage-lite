from abc import ABC, abstractmethod


class DetectionRepositoryInterface(ABC):
    @abstractmethod
    def insert(self, detection):
        pass

    @abstractmethod
    def getall(self):
        pass

    @abstractmethod
    def clear(self):
        pass
