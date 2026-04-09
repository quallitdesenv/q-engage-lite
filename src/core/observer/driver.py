class LocalDriver:
    def __init__(self, settings):
        self.settings = settings
    def run(self, pipeline):
        pipeline.run()