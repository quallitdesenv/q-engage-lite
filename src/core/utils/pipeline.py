from .task import Task
import os

class Logger:
    # Log levels: 0=NONE, 1=ERROR, 2=INFO, 3=DEBUG
    LOG_LEVEL = int(os.environ.get('LOG_LEVEL', '1'))  # Default to ERROR only
    
    def error(self, message: str):
        if self.LOG_LEVEL >= 1:
            print(f"[ERROR] {message}")
            try:
                with open('logs/pipeline.error.log', 'a') as f:
                    f.write(f"[ERROR] {message}\n")
            except:
                pass  # Silently fail if log directory doesn't exist
    
    def info(self, message: str):
        if self.LOG_LEVEL >= 2:
            print(f"[INFO] {message}")
    
    def debug(self, message: str):
        if self.LOG_LEVEL >= 3:
            print(f"[DEBUG] {message}")

class Pipeline():
    def __init__(self, tasks: list[Task]):
        self.task_classes = tasks
        self.tasks = []
        self.logger = Logger()
        self._initialized = False
    
    def __call__(self, frame, model):
        # Initialize tasks only once (first call)
        if not self._initialized:
            self.tasks = [task_class() for task_class in self.task_classes]
            self._initialized = True
        
        # Update frame and model in tasks that need them
        for task in self.tasks:
            if hasattr(task, 'frame'):
                task.frame = frame
            if hasattr(task, 'model'):
                task.model = model
        
        self.run()

    def run(self, idx=0, bag=None):
        """Execute pipeline tasks sequentially. Optimized to use loop instead of recursion."""
        for idx, task in enumerate(self.tasks):
            try:
                bag = task.run(bag)
                self.logger.debug(f"Task {task.name}[{idx + 1}/{len(self.tasks)}] completed")
            except Exception as e:
                self.logger.error(f"Error in task {task.name}: {e}")
                break

        self.logger.debug("Pipeline completed")
