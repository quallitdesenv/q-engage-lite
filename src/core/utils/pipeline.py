from .task import Task

class Logger:
    def error(self, message: str):
        print(f"[{self.__class__.__name__}] {message}")
        with open('logs/pipeline.error.log', 'a') as f:
            f.write(f"[{self.__class__.__name__}] {message}\n")
    def info(self, message: str):
        print(f"[{self.__class__.__name__}] {message}")

class Pipeline():
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks
        self.logger = Logger()
    
    def run(self, idx=0, bag=None):
        if idx >= len(self.tasks):
            self.logger.info("Pipeline completed successfully")
            return
        try:
            task = self.tasks[idx]
            bag = task.run(bag)
            self.logger.info(f"Task {task.name}[{idx + 1}/{len(self.tasks)}] completed successfully")
            self.run(idx + 1, bag)
        except Exception as e:
            self.logger.error(f"Error in task {task.name}: {e}")
