import sys

class BoxInfo:
    def __init__(self, label, xmin, ymin, xmax, ymax, image_name):
        self.label = label
        self.box = xmin, ymin, xmax, ymax
        self.image_name = image_name
        

sys.modules['boxinfo'] = sys.modules[__name__]