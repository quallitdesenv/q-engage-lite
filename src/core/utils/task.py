from abc import ABC, abstractmethod


class Task(ABC):
    @property
    def name(self):
        """Return the task name, defaults to class name."""
        return self.__class__.__name__
    
    @abstractmethod
    def run(self, bag=None):
        pass