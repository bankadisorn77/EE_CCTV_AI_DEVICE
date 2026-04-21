class Data():
    def __init__(self):
        self.config = None 
        self.confidence_threshold = 0.5
        self.LIFE = 'inactive'
        self.activeDevice = 0

    # setters
    def set_config(self, config):
        self.config = config
    def set_confidence_threshold(self, confidence_threshold):
        self.confidence_threshold = confidence_threshold
    def set_LIFE(self, LIFE):
        self.LIFE = LIFE
    def set_active_device(self, device_id):
        self.activeDevice = device_id
   
   



    # getters
    def get_config(self):
        return self.config
    def get_confidence_threshold(self):
        return self.confidence_threshold
    def get_LIFE(self):
        return self.LIFE
    def get_active_device(self):
        return self.activeDevice
