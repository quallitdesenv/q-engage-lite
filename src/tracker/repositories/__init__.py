from .detection_repository import DetectionRepository
from .detection_repository_interface import DetectionRepositoryInterface

_detection_repo = DetectionRepository()

container = {
    DetectionRepositoryInterface: _detection_repo
}

__all__ = ['DetectionRepository', 'DetectionRepositoryInterface', 'container']