class Event:
    def __init__(self, name: str, data: dict = None):
        self.name = name
        self.data = data or {}
    
    def trigger(self):
        # Placeholder for event triggering logic
        pass